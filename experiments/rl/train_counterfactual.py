"""One-GPU counterfactual policy-gradient smoke trainer for Qwen3.5.

This script deliberately optimizes the exact score-function quantity from the
DeltaTrace appendix.  At the first response decision, it samples one selected
action and independent reference actions from the same policy.  The return is
a bounded verifier score from a short continuation sampled from that same
policy after the action.  The credit is ``G(selected)-mean(G(reference))`` and
is detached before weighting the selected-action log-probability.  It is
therefore an RL estimator with an explicit counterfactual baseline, not a
token-wise attribution heuristic.

The MATH examples are existing released data.  Short truncation and a small
LoRA target are intentional engineering settings for validating a stable
single-card path; they are not a final capability experiment.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from counterfactual import averaged_counterfactual_credit, score_function_loss
from lora import inject_lora, trainable_parameter_count


def _load_cases(path: Path, count: int) -> list[dict]:
    cases = [json.loads(line) for line in path.read_text().splitlines()]
    if count < 1 or count > len(cases):
        raise ValueError(f"count must be in [1, {len(cases)}]")
    return cases[:count]


def _tokenize_cases(tokenizer, cases, max_prompt_tokens: int, max_target_tokens: int, device: str):
    prompts, targets = [], []
    for case in cases:
        prompt = tokenizer(case["prompt"], add_special_tokens=False, truncation=True,
                           max_length=max_prompt_tokens, return_tensors="pt").input_ids[0]
        target = tokenizer(case["target"], add_special_tokens=False, truncation=True,
                           max_length=max_target_tokens, return_tensors="pt").input_ids[0]
        if target.numel() < 2:
            raise ValueError("each target needs at least two tokens")
        prompts.append(prompt.to(device))
        targets.append(target.to(device))
    return prompts, targets


def _pad(rows: list[torch.Tensor], pad: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(int(row.numel()) for row in rows)
    ids = torch.full((len(rows), width), pad, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for i, row in enumerate(rows):
        ids[i, : row.numel()] = row
        mask[i, : row.numel()] = 1
    return ids, mask


def _continuation_returns(
    model, prompt: torch.Tensor, selected_and_refs: torch.Tensor, target: torch.Tensor, eos_token_id: int
) -> torch.Tensor:
    """Sample the same continuation policy and score a bounded verifier."""

    target_continuation = target[1:]
    returns = []
    with torch.no_grad():
        for action in selected_and_refs:
            prefix = torch.cat((prompt, action[None]))[None]
            generated = model.generate(
                input_ids=prefix,
                max_new_tokens=int(target_continuation.numel()),
                do_sample=True,
                temperature=1.0,
                top_p=1.0,
                use_cache=True,
                pad_token_id=int(eos_token_id),
            )[0, prefix.shape[1] :]
            matches = min(int(generated.numel()), int(target_continuation.numel()))
            score = generated.new_zeros((), dtype=torch.float32)
            if matches:
                score = (generated[:matches] == target_continuation[:matches]).float().sum()
            returns.append(score / max(int(target_continuation.numel()), 1))
    return torch.stack(returns)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--cases", type=int, default=2)
    parser.add_argument("--references", type=int, default=2)
    parser.add_argument("--max-prompt-tokens", type=int, default=128)
    parser.add_argument("--max-target-tokens", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.steps < 1 or args.references < 1:
        raise ValueError("steps and references must be positive")

    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    torch.manual_seed(73)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="flash_attention_2",
        device_map={"": args.device}, local_files_only=True,
    )
    model.train()
    params = inject_lora(model, ("q_proj", "v_proj"), rank=args.rank, alpha=2 * args.rank)
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    cases = _load_cases(args.dataset, args.cases)
    prompts, targets = _tokenize_cases(tokenizer, cases, args.max_prompt_tokens, args.max_target_tokens, args.device)
    history = []
    started = time.perf_counter()
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        selected_logps, selected_returns, reference_returns = [], [], []
        for prompt, target in zip(prompts, targets):
            with torch.no_grad():
                policy_logits = model(input_ids=prompt[None], use_cache=False, logits_to_keep=1).logits[0, -1]
                probs = F.softmax(policy_logits.float(), dim=-1)
                selected = torch.multinomial(probs, 1)
                refs = torch.multinomial(probs, args.references, replacement=True)
            candidates = torch.cat((selected, refs), dim=0)
            returns = _continuation_returns(model, prompt, candidates, target, tokenizer.eos_token_id)
            selected_returns.append(returns[0])
            reference_returns.append(returns[1:])
            # Recompute the selected action log-probability with gradients.
            logits = model(input_ids=prompt[None], use_cache=False, logits_to_keep=1).logits[0, -1]
            selected_logps.append(F.log_softmax(logits.float(), dim=-1)[selected[0]])
        selected_return_tensor = torch.stack(selected_returns)
        reference_return_tensor = torch.stack(reference_returns)
        credit = averaged_counterfactual_credit(selected_return_tensor.detach(), reference_return_tensor.detach())
        loss = score_function_loss(torch.stack(selected_logps), credit.credit)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(params, 1.0).detach())
        optimizer.step()
        row = {
            "step": step, "loss": float(loss.detach()),
            "selected_return_mean": float(selected_return_tensor.mean()),
            "reference_return_mean": float(reference_return_tensor.mean()),
            "credit_mean": float(credit.credit.mean()), "credit_std": float(credit.credit.std(unbiased=False)),
            "grad_norm_before_clip": grad_norm,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(torch.device(args.device)),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(torch.device(args.device)),
        }
        if not all(torch.isfinite(torch.tensor(v)) for v in row.values() if isinstance(v, float)):
            raise FloatingPointError(f"non-finite training row: {row}")
        history.append(row)
        print(json.dumps(row), flush=True)
    report = {
        "status": "stable_smoke_complete", "return_definition": "policy_sampled_prefix_match",
        "model": args.model, "dataset": str(args.dataset),
        "steps": args.steps, "cases": args.cases, "references": args.references,
        "trainable_parameters": trainable_parameter_count(model),
        "elapsed_seconds": time.perf_counter() - started, "history": history,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()

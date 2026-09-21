"""Run one real token-attribution trace through the owner Qwen3.5 runner.

The selected and reference endpoints differ at one first-response action
position. The owner signed vector is checked for conservation. This diagnostic
does not produce reward-event log ratios or validate the RL advantage method.
"""

from __future__ import annotations

import json
import os
import sys
import time
import argparse
from collections import defaultdict
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="math.jsonl")
    parser.add_argument("--target-tokens", type=int, default=8)
    parser.add_argument("--output", default="/data/liangchen/deltatrace_resume_20260917/rl_dt_action_smoke.json")
    parser.add_argument("--lora", action="store_true")
    args = parser.parse_args()
    root = Path("/data/liangchen/deltatrace_resume_20260917/repo")
    source = root / "deltatrace"
    official = Path("/data/liangchen/deltatrace_resume_20260917/flashtrace_paper")
    env = json.loads(Path("/data/liangchen/deltatrace_resume_20260917/environment.json").read_text())["qwen35"]
    sys.path[:0] = [str(root), str(official), str(source / "clean/qwen35")]
    sys.path.append(env["ft_extension_root"])
    os.environ.update(env.get("runtime_environment", {}), HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    os.environ.setdefault("TRITON_CACHE_DIR", env["triton_cache"])
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", env["inductor_cache"])

    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    from deltatrace.profiles.official import make_qwen35_runner
    from finite_fla_gpu import make_compiled_finite_pullback, verify_native_sources
    from qwen35_answer_finite import PackedAnswerTargets
    from vendor_fa_finite_bf16_d256 import VendorFAFiniteP1BF16D256
    from deltatrace_credit import trace_token_attribution

    verify_native_sources(env["native_stage_source_sha256"])
    tokenizer = AutoTokenizer.from_pretrained(env["checkpoint"], local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        env["checkpoint"], dtype=torch.bfloat16, attn_implementation="eager",
        device_map={"": "cuda:0"}, local_files_only=True,
    ).eval()
    if args.lora:
        from lora import inject_lora
        inject_lora(model, ("q_proj", "v_proj"), rank=4, alpha=8.0)
        # Nonzero B exercises the effective-weight compatibility path used
        # after an optimizer update; the trainer itself keeps B zero at init.
        with torch.no_grad():
            for module in model.modules():
                if hasattr(module, "lora_B"):
                    module.lora_B.normal_(mean=0.0, std=1e-3)
    model.set_attn_implementation("flash_attention_2")
    finite_fa = VendorFAFiniteP1BF16D256(env["finite_library"], env["finite_library_sha256"])
    runner = make_qwen35_runner(
        model, finite_fa,
        make_compiled_finite_pullback(reuse_scalar_products=False,
                                      dynamic_shapes=env.get("dt_dynamic_shapes", False),
                                      compiler_options=env.get("dt_compiler_options", {})),
        dynamic_shapes=env.get("dt_dynamic_shapes", False),
        compiler_options=env.get("dt_compiler_options", {}),
    )
    case_path = official / "exp/exp2/data" / args.dataset
    case = json.loads(case_path.read_text().splitlines()[0])
    prompt_ids = tokenizer(case["prompt"], add_special_tokens=False, return_tensors="pt").input_ids.to("cuda:0")
    target_ids = tokenizer(case["target"], add_special_tokens=False, return_tensors="pt").input_ids[0, :args.target_tokens].to("cuda:0")
    prompt_len = int(prompt_ids.shape[1])
    selected = torch.cat((prompt_ids, target_ids[None]), dim=1)
    reference = selected.clone()
    reference[0, prompt_len] = (reference[0, prompt_len] + 1) % model.lm_head.out_features
    target_case = {"target_ids": target_ids.detach().cpu(), "prompt_length": prompt_len}
    def summarize(signed, root_effect, detail, elapsed):
        grouped = defaultdict(float)
        for row in detail.get("calls", []):
            grouped[row.get("kind")] += float(row.get("seconds", 0.0))
        return {
            "root_effect": float(root_effect.item()), "signed_sum": float(signed.sum().item()),
            "relative_residual": detail.get("relative_residual"), "seconds": elapsed,
            "runner_seconds": detail.get("complete_attribution_seconds_with_diagnostics"),
            "peak_allocated": detail.get("peak_allocated"), "peak_reserved": detail.get("peak_reserved"),
            "policy_credit_signed_vector_used": detail["policy_credit_signed_vector_used"],
            "call_seconds_by_kind": dict(sorted(grouped.items())),
        }

    runs = []
    for _ in range(2):
        started = time.perf_counter()
        signed, root_effect, detail = trace_token_attribution(
            runner, reference, selected, target_case, list(range(int(target_ids.numel()))),
            packed_answer_targets=PackedAnswerTargets,
        )
        runs.append(summarize(signed, root_effect, detail, time.perf_counter() - started))
    result = {
        "scope": "official text attribution only; not RL reward-event credit",
        "status": "passed", "prompt_tokens": prompt_len, "target_tokens": int(target_ids.numel()),
        "action_position": prompt_len, "selected_action": int(selected[0, prompt_len]),
        "reference_action": int(reference[0, prompt_len]), "runs": runs,
    }
    out = Path(args.output)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

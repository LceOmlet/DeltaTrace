"""Minimal Qwen3.5 runtime probe for the RL development branch.

The probe intentionally reuses the remote DeltaTrace environment.  It does not
install packages, alter model code, or run an attribution path.  Its output is
the baseline needed to distinguish model/runtime cost from DeltaTrace cost.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--text", default="What is 2+2? Answer with one number.")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument(
        "--disable-causal-conv1d",
        action="store_true",
        help="Use Transformers' torch causal-conv fallback when the installed CUDA extension cannot load.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch
    if args.disable_causal_conv1d:
        # The remote host is Ubuntu/GLIBC 2.31 while the preinstalled wheel
        # advertises GLIBC 2.32.  Let the owning Transformers implementation
        # select its documented torch fallback instead of changing the venv.
        from transformers.utils import import_utils

        import_utils.is_causal_conv1d_available = lambda: False
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    torch.set_grad_enabled(False)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model_started = time.perf_counter()
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        device_map={"": args.device},
        local_files_only=True,
    )
    model.eval()
    load_seconds = time.perf_counter() - model_started
    batch = tokenizer(args.text, return_tensors="pt").to(args.device)

    if args.repeats < 1:
        raise ValueError("--repeats must be positive")
    timings = []
    outputs = []
    for _ in range(args.repeats):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(torch.device(args.device))
        started = time.perf_counter()
        output = model.generate(
            **batch,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        timings.append(elapsed)
        outputs.append(output)
    output = outputs[-1]
    elapsed = timings[-1]
    generated = int(output.shape[1] - batch["input_ids"].shape[1])
    device = torch.device(args.device)
    report = {
        "model": str(args.model),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "prompt_tokens": int(batch["input_ids"].shape[1]),
        "generated_tokens": generated,
        "load_seconds": load_seconds,
        "generate_seconds": elapsed,
        "generated_tokens_per_second": generated / max(elapsed, 1e-9),
        "repeat_generate_seconds": timings,
        "steady_state_tokens_per_second": generated / max(sum(timings[1:]) / max(len(timings) - 1, 1), 1e-9)
        if len(timings) > 1 else None,
        "allocated_bytes_after_load": torch.cuda.memory_allocated(device),
        "reserved_bytes_after_load": torch.cuda.memory_reserved(device),
        "peak_allocated_bytes_generate": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes_generate": torch.cuda.max_memory_reserved(device),
        "text": args.text,
        "decoded": tokenizer.decode(output[0], skip_special_tokens=True),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""Create the tiny prompt parquet consumed by upstream verl-agent2.

The environments own task sampling and scoring.  These rows only provide the
prompt schema required by ``verl.utils.dataset.RLHFDataset``; this file does
not implement an RL loop or a task environment.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import Dataset


def make_rows(size: int, split: str, prompt_fill_tokens: int = 0) -> list[dict]:
    if size < 1:
        raise ValueError("size must be positive")
    if prompt_fill_tokens < 0:
        raise ValueError("prompt_fill_tokens must be non-negative")
    # Qwen's ``context`` token is a stable single-token filler in the pinned
    # tokenizer.  The fixture is only for long-context resource tests; task
    # dynamics and rewards remain owned by the upstream environment.
    filler = " ".join(["context"] * prompt_fill_tokens)
    prompt = [{"role": "user", "content": ""}]
    if filler:
        # Keep the fixture as prior conversation context.  The upstream
        # collector appends each environment observation to the last user
        # turn; placing filler in an assistant turn prevents it from being
        # replaced while preserving the official environment observation.
        prompt.append({"role": "assistant", "content": filler})
    return [
        {
            "data_source": "agent",
            "prompt": prompt,
            "ability": "agent",
            "extra_info": {"split": split, "index": i},
        }
        for i in range(size)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=1)
    parser.add_argument("--split", default="train")
    parser.add_argument("--prompt-fill-tokens", type=int, default=0)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = make_rows(args.size, args.split, args.prompt_fill_tokens)
    Dataset.from_list(rows).to_parquet(str(args.output))
    print(f"wrote {len(rows)} rows to {args.output} (prompt_fill_tokens={args.prompt_fill_tokens})")


if __name__ == "__main__":
    main()

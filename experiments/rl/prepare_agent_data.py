"""Create the tiny prompt parquet consumed by upstream verl-agent2.

The environments own task sampling and scoring.  These rows only provide the
prompt schema required by ``verl.utils.dataset.RLHFDataset``; this file does
not implement an RL loop or a task environment.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import Dataset


def make_rows(size: int, split: str) -> list[dict]:
    if size < 1:
        raise ValueError("size must be positive")
    return [
        {
            "data_source": "agent",
            "prompt": [{"role": "user", "content": ""}],
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
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(make_rows(args.size, args.split)).to_parquet(str(args.output))
    print(f"wrote {len(make_rows(args.size, args.split))} rows to {args.output}")


if __name__ == "__main__":
    main()

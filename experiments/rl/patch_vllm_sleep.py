"""Backport the vLLM v0.17.0 weight-pool context fix to installed 0.15/0.16.

Owner source:
https://github.com/vllm-project/vllm/blob/v0.17.0/vllm/v1/worker/gpu_worker.py
The original ``with A and B`` enters only B, leaving weights outside the
native sleep allocator. No allocator, weight transfer, or model math is copied.
Apply explicitly to the recorded installed gpu_worker.py, not on every run.
"""
import argparse
from pathlib import Path


OLD = '''        with self._maybe_get_memory_pool_context(
            tag="weights"
        ) and set_current_vllm_config(self.vllm_config):'''
NEW = '''        with (
            self._maybe_get_memory_pool_context(tag="weights"),
            set_current_vllm_config(self.vllm_config),
        ):'''


def patch_source(text: str) -> str:
    if OLD in text:
        if text.count(OLD) != 1:
            raise RuntimeError("ambiguous vLLM weight-pool context")
        return text.replace(OLD, NEW, 1)
    if NEW in text:
        return text
    raise RuntimeError("installed vLLM load_model does not match the recorded owner")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("worker_source", type=Path)
    args = parser.parse_args()
    original = args.worker_source.read_text()
    patched = patch_source(original)
    if patched != original:
        args.worker_source.write_text(patched)
        print(f"backported vLLM v0.17.0 weight-pool context: {args.worker_source}")
    else:
        print(f"already patched: {args.worker_source}")

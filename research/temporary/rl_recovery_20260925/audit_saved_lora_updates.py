"""Read completed native checkpoints; compare only saved LoRA tensors on CPU.

Uses torch's mmap loader, not a second checkpoint loader or training path.
Does not materialize the frozen 9B base weights or initialize an actor/GPU.
"""
import argparse
import datetime
import gc
import json
from pathlib import Path
import time

import torch


def local_tensor(value):
    return value.to_local() if hasattr(value, "to_local") else value


def audit(root, task, run):
    checkpoint_root = root / "runs" / run / task / "checkpoints"
    assert (checkpoint_root / "latest_checkpointed_iteration.txt").read_text().strip() == "2"
    directories = [checkpoint_root / f"global_step_{step}" / "actor" for step in (1, 2)]
    # These files were written by our own pinned upstream training workers.
    states = [torch.load(p / "model_world_size_1_rank_0.pt", map_location="cpu",
                         mmap=True, weights_only=False) for p in directories]
    names = [name for name in states[0] if "lora_" in name]
    assert names and set(names) == {name for name in states[1] if "lora_" in name}
    changes = []
    for name in names:
        before, after = [local_tensor(state[name]).detach() for state in states]
        assert before.device.type == after.device.type == "cpu"
        delta = after.float() - before.float()
        changes.append(dict(name=name, shape=list(before.shape), dtype=str(before.dtype),
                            elements=before.numel(), changed=int(torch.count_nonzero(delta)),
                            maximum_absolute_change=float(delta.abs().max()),
                            squared_change=float(delta.double().square().sum()),
                            finite=bool(torch.isfinite(before).all() and torch.isfinite(after).all())))
    optim = [torch.load(p / "optim_world_size_1_rank_0.pt", map_location="cpu",
                       mmap=True, weights_only=False) for p in directories]
    groups = [[{key: group[key] for key in ("lr", "initial_lr", "weight_decay") if key in group}
               for group in state["param_groups"]] for state in optim]
    steps = []
    for state in optim:
        values = [float(local_tensor(value["step"])) for value in state["state"].values() if "step" in value]
        steps.append(dict(min=min(values), max=max(values), count=len(values)))
    return dict(task=task, checkpoint_root=str(checkpoint_root), compared_steps=[1, 2],
                tensor_count=len(changes), changed_tensors=sum(x["changed"] > 0 for x in changes),
                changed_elements=sum(x["changed"] for x in changes),
                total_elements=sum(x["elements"] for x in changes),
                maximum_absolute_change=max(x["maximum_absolute_change"] for x in changes),
                l2_change=sum(x["squared_change"] for x in changes) ** 0.5,
                finite=all(x["finite"] for x in changes), optimizer_groups=groups,
                optimizer_steps=steps, tensors=changes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    start = time.monotonic()
    torch.set_num_threads(2)
    result = dict(recorded_at=datetime.datetime.now().astimezone().isoformat(),
                  scope="Saved native LoRA parameter changes between completed real task iterations; not an accuracy or performance evaluation.",
                  tasks=[])
    for task, run in (("Sokoban", "head-f0429ee-continuous"),
                      ("Webshop", "head-task-continuous-20260925"),
                      ("AppWorld", "head-task-continuous-20260925")):
        item = audit(args.root, task, run)
        result["tasks"].append(item)
        gc.collect()
        print(json.dumps({k: v for k, v in item.items() if k != "tensors"}), flush=True)
    result["seconds"] = time.monotonic() - start
    result["cuda_initialized"] = torch.cuda.is_initialized()
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")

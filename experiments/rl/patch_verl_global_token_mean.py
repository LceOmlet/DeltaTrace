"""Minimal default-inert backport of VERL's global token-mean denominator.

The pinned legacy actor normalizes each microbatch independently. Its native
gradient accumulation and DP averaging then weight short microbatches more.
The enabled owner path uses the actual optimizer minibatch's valid tokens.
Credit, PPO clipping, entropy/KL coefficients and optimizer calls stay native.
"""
from pathlib import Path
import argparse
import hashlib
import json


ACTOR_BASE_SHA = '3a65e173300be82a7a9e056a96227c4f746eabc3be778ef41c8d50138d52ce6c'
ACTOR_PATCHED_SHA = 'ec0924e6abe561542ddd863bcf7040b995b19a34e500394a0604db6fc1dc5941'
WORKER_BASE_SHA = '9dfd6ce795384b98faa962269e1eac98ad05ec65e42af2aac434c41db41641d3'


def replace_once(source, old, new):
    if new in source:
        return source
    if source.count(old) != 1:
        raise ValueError('Pinned owner anchor is absent or ambiguous: '+old[:90])
    return source.replace(old, new, 1)


def patch_actor(source):
    source = replace_once(source,
        '        multi_turn = data.meta_info.get("multi_turn", False)\n',
        '        multi_turn = data.meta_info.get("multi_turn", False)\n'
        '        global_batch_info = data.meta_info.get("global_batch_info")\n')
    source = replace_once(source,
        '                mini_batch = data\n',
        '                mini_batch = data\n'
        '                global_token_count = None\n'
        '                if global_batch_info is not None and self.config.loss_agg_mode == "token-mean":\n'
        '                    width = mini_batch["responses"].size(1)\n'
        '                    mask = mini_batch["loss_mask"][:, -width:] if multi_turn else mini_batch["attention_mask"][:, -width:]\n'
        '                    global_token_count = mask.sum()\n'
        '                    dp_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1\n'
        '                    if dp_size > 1:\n'
        '                        torch.distributed.all_reduce(global_token_count)\n')
    return replace_once(source,
        '                    if self.config.use_dynamic_bsz:\n                        # relative to the dynamic bsz\n',
        '                    if global_token_count is not None:\n'
        '                        # Backport the official global token denominator.\n'
        '                        # The native scalar already divides by this B4 mask;\n'
        '                        # undo that divisor and compensate native DP averaging.\n'
        '                        loss = policy_loss * response_mask.sum() * dp_size / global_token_count\n'
        '                    elif self.config.use_dynamic_bsz:\n                        # relative to the dynamic bsz\n')


def patch_worker(source):
    return replace_once(source,
        '            # perform training\n'
        '            with Timer(name="update_policy", logger=None) as timer:\n',
        '            # Explicit version activation; numerical loss ownership stays in the actor.\n'
        '            if os.environ.get("VERL_GLOBAL_TOKEN_MEAN") == "1":\n'
        '                data.meta_info["global_batch_info"] = {}\n'
        '            # perform training\n'
        '            with Timer(name="update_policy", logger=None) as timer:\n')


def apply(root):
    root = Path(root).resolve()
    changes = [('verl/workers/actor/dp_actor.py', ACTOR_BASE_SHA, patch_actor),
               ('verl/workers/fsdp_workers.py', WORKER_BASE_SHA, patch_worker)]
    prepared = []
    for name, expected, patch in changes:
        path = root/name
        if not path.parent.resolve().is_relative_to(root):
            raise ValueError('Owner parent resolves outside the isolated candidate: '+str(path))
        original = path.read_text(encoding='utf-8')
        modified = patch(original)
        if original != modified and hashlib.sha256(original.encode()).hexdigest() != expected:
            raise ValueError('Unexpected pinned owner source: '+str(path))
        if name.endswith('dp_actor.py') and hashlib.sha256(modified.encode()).hexdigest() != ACTOR_PATCHED_SHA:
            raise ValueError('Actor differs from the native gradient-tested patch')
        prepared.append((path, modified))
    report = {}
    for path, source in prepared:
        if path.is_symlink():
            path.unlink()
        path.write_text(source, encoding='utf-8', newline='')
        report[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    print(json.dumps(apply(args.root), indent=2))

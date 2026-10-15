"""Check the LoRA-only immutable serialized model state, CPU only.

Uses original Torch checkpoint loading with mmap and native Tensor equality.
It does not construct or run a model, optimizer, loss or attribution algorithm.
Nonpersistent runtime buffers are outside this serialized-state comparison.
"""
import hashlib
import json
from pathlib import Path
import time

import torch

torch.set_num_threads(4)
OUT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/textcraft-gradient-rootcause-20261013-v1')
manifest = json.loads((OUT / 'native-full-policy-direction-exchange-v86/case-manifest.json').read_bytes())
cases = {c['checkpoint']: c for c in manifest['cases'] if c['name'].endswith('_native')}
control_manifest = json.loads((OUT / 'native-full-policy-saved-update-feedback-v90/case-manifest.json').read_bytes())
control = next(c for c in control_manifest['cases'] if c['name'] == 'GRPO160_same_full80_seed')

def state_digest(model):
    h = hashlib.sha256()
    for name, value in model.items():
        if '.lora_' not in name:
            continue
        local = value.to_local() if hasattr(value, 'to_local') else value
        h.update(name.encode())
        h.update(local.detach().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()

sources = []
comparisons = []
for rank in (0, 1):
    models = {}
    paths = {}
    stats = {}
    for name, spec in [('DT40', cases[40]), ('DT80', cases[80]), ('GRPO160', control)]:
        path = Path(spec['checkpoint_path']) / f'model_world_size_2_rank_{rank}.pt'
        stat = path.stat()
        models[name] = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        paths[name] = path
        stats[name] = (stat.st_ino, stat.st_size, stat.st_mtime_ns)
        actual = state_digest(models[name])
        assert actual == spec['expected_trainable_SHA256_by_rank'][str(rank)], (name, rank)
        sources.append(dict(label=name, rank=rank, path=str(path), inode=stat.st_ino,
            bytes=stat.st_size, mtime_ns=stat.st_mtime_ns, native_LoRA_SHA256=actual))
    assert all(tuple(models[name]) == tuple(models['DT80']) for name in models)
    frozen = [name for name in models['DT80'] if '.lora_' not in name]
    assert frozen
    for reference in ('DT40', 'GRPO160'):
        rows = []
        for name in frozen:
            a, b = models['DT80'][name], models[reference][name]
            a = a.to_local() if hasattr(a, 'to_local') else a
            b = b.to_local() if hasattr(b, 'to_local') else b
            same_schema = a.shape == b.shape and a.dtype == b.dtype
            equal = bool(torch.equal(a, b)) if same_schema else False
            row = dict(name=name, local_shape=list(a.shape), reference_local_shape=list(b.shape),
                dtype=str(a.dtype), reference_dtype=str(b.dtype), local_elements=a.numel(), equal=equal)
            if not equal and same_schema:
                row['bitwise_equal'] = bool(torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)))
                row['nonfinite_DT80'] = int((~torch.isfinite(a)).sum())
                row['nonfinite_reference'] = int((~torch.isfinite(b)).sum())
                row['different_elements'] = int((a != b).sum())
                row['max_absolute_difference'] = float((a.float() - b.float()).abs().max())
            rows.append(row)
        comparisons.append(dict(rank=rank, left='DT80', right=reference, tensors=rows,
            tensors_compared=len(rows), local_elements=sum(x['local_elements'] for x in rows),
            all_equal=all(x['equal'] for x in rows), differing_tensor_names=[x['name'] for x in rows if not x['equal']]))
        print(f'CPU frozen-state comparison rank={rank} DT80 vs {reference}: equal={comparisons[-1]["all_equal"]}', flush=True)
    for name, path in paths.items():
        stat = path.stat()
        assert (stat.st_ino, stat.st_size, stat.st_mtime_ns) == stats[name], 'Serialized checkpoint changed during the read'
    del models

report = dict(unix=time.time(), scope=__doc__, sources=sources, comparisons=comparisons,
    all_serialized_frozen_parameters_equal=all(x['all_equal'] for x in comparisons),
    model_calls=0, backward_calls=0, optimizer_calls=0, training_changes=0,
    limits=['The equality covers serialized non-LoRA tensors at the saved checkpoints, not every actual in-memory parameter or nonpersistent buffer of the completed probes.',
            'Matching saved frozen tensors and LoRA identities still cannot by themselves establish identical native forward numerical execution. A failed invariant must be diagnosed separately from a training-root-cause claim.'])
dest = OUT / 'saved-frozen-parameter-equality-v96.json'
assert not dest.exists()
dest.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(dict(receipt=str(dest), report=report)))

"""Compare completed native carriers before attributing the forward mismatch.

Original DataProto loading and native Tensor equality only; CPU, no model call.
The comparison checks values, not merely schemas or rho1 loss scalars.
"""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from verl import DataProto

torch.set_num_threads(4)
OUT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/textcraft-gradient-rootcause-20261013-v1')
old_manifest = json.loads((OUT / 'native-full-policy-direction-exchange-v86/case-manifest.json').read_bytes())
new_manifest = json.loads((OUT / 'native-full-policy-saved-update-feedback-v90/case-manifest.json').read_bytes())
original = next(c for c in old_manifest['cases'] if c['name'] == 'full80_native')
baseline = next(c for c in new_manifest['cases'] if c['name'] == 'old80_before')
assert original['path'] == baseline['path'] and original['SHA256'] == baseline['SHA256']
path = Path(original['path'])
assert hashlib.sha256(path.read_bytes()).hexdigest() == original['SHA256']
data = DataProto.load_from_disk(str(path))
assert len(data) == 64

def describe(value):
    t = value.detach().cpu()
    return dict(shape=list(t.shape), dtype=str(t.dtype), stride=list(t.stride()),
                SHA256=hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest())

rows = []
for name in ('full40_native', 'full80_native', 'full_weights80_seed40', 'full_weights40_seed80'):
    p = OUT / 'native-full-policy-direction-exchange-v86' / f'{name}-native-logp.pkl'
    recorded = DataProto.load_from_disk(str(p))
    assert len(recorded) == len(data)
    tensors = []
    for key in data.batch.keys():
        if key in ('old_log_probs', 'ref_log_prob'):
            continue
        assert key in recorded.batch.keys(), key
        a, b = data.batch[key], recorded.batch[key]
        tensors.append(dict(field=key, equal=bool(torch.equal(a, b)),
                            original=describe(a), native_saved=describe(b)))
    identities = []
    for key in ('uid', 'traj_uid'):
        if key not in data.non_tensor_batch:
            continue
        assert key in recorded.non_tensor_batch
        a, b = data.non_tensor_batch[key], recorded.non_tensor_batch[key]
        identities.append(dict(field=key, equal=bool(np.array_equal(a, b)),
                               original_shape=list(a.shape), native_saved_shape=list(b.shape)))
    metadata = {key: dict(original=data.meta_info.get(key), native_saved=recorded.meta_info.get(key),
                        equal=data.meta_info.get(key) == recorded.meta_info.get(key))
                for key in ('temperature', 'multi_turn', 'micro_batch_size', 'use_dynamic_bsz', 'max_token_len')}
    rows.append(dict(case=name, source=str(p), SHA256=hashlib.sha256(p.read_bytes()).hexdigest(),
                     tensors=tensors, trajectory_identity=identities, metadata=metadata))

report = dict(unix=time.time(), scope=__doc__, original_source=dict(path=str(path), SHA256=original['SHA256']),
    v90_original_source_exactly_same=True, cases=rows,
    all_original_non_logprob_tensors_equal=all(v['equal'] for row in rows for v in row['tensors']),
    all_trajectory_UIDs_equal=all(v['equal'] for row in rows for v in row['trajectory_identity']),
    model_calls=0, backward_calls=0, optimizer_calls=0,
    limits=['The current v90 before-backward GPU inputs were not saved separately. Its manifest binds the identical original global64 carrier, but this alone does not verify every native worker transformation or complete model/buffer state.',
            'Fresh old/ref log-probabilities are expected to change with the actual forward point and are intentionally not claimed equal.',
            'Schema/identity checks and rho1 scalar equality cannot establish identical forward logits.'])
dest = OUT / 'completed-native-carrier-values-audit-v95.json'
assert not dest.exists()
dest.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(dict(receipt=str(dest), report=report)))

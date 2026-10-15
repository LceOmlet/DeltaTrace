"""Compare saved actual native whole-PG head adjoints at original80, CPU only.

This reads the already completed v86 and first v99 baseline captures. It does
not run a model/backward or infer a parameter-gradient causal share from a
head-adjoint cosine or unmatched cross-run conditional VJP.
"""
import hashlib
import json
from pathlib import Path
import time
import torch

torch.set_num_threads(4)
ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT = ROOT / 'receipts/textcraft-gradient-rootcause-20261013-v1'
old = OUT / 'native-full-policy-direction-exchange-v86'
new = OUT / 'native-saved-update-natural-PG-v99'
rows = []
assert json.loads((new / 'result.json').read_bytes())['observations'][0]['iteration'] == 'old80_before'
for rank in (0, 1):
    old_chain = json.loads((old / f'iterationfull80_native-rank{rank}-chain.json').read_bytes())
    new_chain = json.loads((new / f'iterationold80_before-rank{rank}-chain.json').read_bytes())
    assert old_chain['trainable_SHA256_before'] == new_chain['trainable_SHA256_before']
    for micro in range(8):
        pa = old / f'seed-full80_native-rank{rank}-micro{micro}.pt'
        pb = new / f'seed-old80_before-rank{rank}-micro{micro}.pt'
        stats_before = [(p.stat().st_size, p.stat().st_mtime_ns) for p in (pa, pb)]
        a, b = [torch.load(p, map_location='cpu', weights_only=True) for p in (pa, pb)]
        assert a.shape == b.shape and a.dtype == b.dtype
        equal = torch.equal(a, b)
        aa = bb = ab = dd = 0.0
        changed_support = differing_values = 0
        maximum_difference = 0.0
        left, right = a.reshape(-1), b.reshape(-1)
        for start in range(0, left.numel(), 65536):
            u, v = left[start:start+65536].double(), right[start:start+65536].double()
            difference = v - u
            aa += float(u.square().sum())
            bb += float(v.square().sum())
            ab += float((u * v).sum())
            dd += float(difference.square().sum())
            changed_support += int(((u == 0) != (v == 0)).sum())
            differing_values += int((u != v).sum())
            maximum_difference = max(maximum_difference, float(difference.abs().max()))
        hashes = []
        for path in (pa, pb):
            with path.open('rb') as f:
                hashes.append(hashlib.file_digest(f, 'sha256').hexdigest())
        assert stats_before == [(p.stat().st_size, p.stat().st_mtime_ns) for p in (pa, pb)]
        rows.append(dict(rank=rank, micro=micro, original_path=str(pa), original_SHA256=hashes[0],
            current_path=str(pb), current_SHA256=hashes[1], shape=list(a.shape), dtype=str(a.dtype),
            values_equal=equal, original_sumsq=aa, current_sumsq=bb, dot=ab,
            difference_sumsq=dd, maximum_abs_difference=maximum_difference,
            differing_values=differing_values, changed_zero_support_values=changed_support))
def aggregate(items):
    aa = sum(r['original_sumsq'] for r in items)
    bb = sum(r['current_sumsq'] for r in items)
    ab = sum(r['dot'] for r in items)
    dd = sum(r['difference_sumsq'] for r in items)
    return dict(original_RSS=aa ** .5, current_RSS=bb ** .5,
        norm_ratio=(bb / aa) ** .5, cosine=ab / (aa * bb) ** .5,
        relative_difference_RSS=(dd / aa) ** .5,
        differing_values=sum(r['differing_values'] for r in items),
        changed_zero_support_values=sum(r['changed_zero_support_values'] for r in items),
        all_values_equal=all(r['values_equal'] for r in items))
report = dict(unix=time.time(), scope=__doc__, rows=rows, all_ranks=aggregate(rows),
    by_rank={str(rank): aggregate([r for r in rows if r['rank'] == rank]) for rank in (0, 1)},
    model_calls=0, backward_calls=0, optimizer_calls=0, cuda_initialized=torch.cuda.is_initialized(),
    limits=['Both manifests bind the same original64 carrier and LoRA values; this CPU comparison does not prove identical complete in-memory forward numerical execution across those completed runs.',
        'Native BF16 head-vector changes quantify the actual saved difference. They are not DT-estimation error or a proven cause of the historical parameter-gradient growth.',
        'No cross-run J-vector subtraction, exact amplification bound, universal tolerance or root-repair acceptance is inferred here.'])
assert report['cuda_initialized'] is False
path = OUT / 'saved-whole-head-adjoint-comparison-v102.json'
path.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(dict(receipt=str(path), report=report)))

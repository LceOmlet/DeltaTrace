"""Resolve the final source commit on prepared-only records before native launch."""
import hashlib
import json
import os
from pathlib import Path
import re

R=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
A=R/'receipts/dtpo-symmetric-probability-ratio-postgroup-variance-20261018-v8'
target=os.environ['RL_CAMPAIGN_COMMIT']
assert re.fullmatch('[0-9a-f]{40}',target)
assert not (A/'controller-job.json').exists()
plan=json.loads((A/'six-combination-manifest.json').read_bytes())
assert plan['status']=='prepared_not_started'
previous=plan['source_commit']
changes=[]
paths=[A/'six-combination-manifest.json']
for s in plan['segments']:
    p=Path(s['prepared']);paths.append(p)
    for row in json.loads(p.read_bytes())['jobs']:
        assert not Path(row['run']).exists()
        paths.append(p.parent/(row['method']+'-prepared.json'))

def resolve(v):
    if isinstance(v,dict):
        return {k:(target if k in ('local_patch_commit','source_commit') and x==previous else resolve(x)) for k,x in v.items()}
    if isinstance(v,list):return [resolve(x) for x in v]
    return v

for path in paths:
    before=path.read_bytes();after=(json.dumps(resolve(json.loads(before)),indent=2)+'\n').encode()
    path.write_bytes(after)
    changes.append(dict(path=str(path),before_SHA256=hashlib.sha256(before).hexdigest(),after_SHA256=hashlib.sha256(after).hexdigest()))
(A/'prelaunch-source-resolution.json').write_text(json.dumps(dict(
    status='full_commit_pinned_before_any_training_launch',method_verified_commit=previous,
    execution_commit=target,changes=changes,computational_source_changes=0),indent=2)+'\n')
print(json.dumps(dict(status='full_commit_pinned',method_commit=previous,execution_commit=target)))

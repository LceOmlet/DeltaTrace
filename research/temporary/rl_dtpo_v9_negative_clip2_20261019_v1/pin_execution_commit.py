"""Pin full execution and method refs on prepared-only native records."""
import hashlib
import json
from pathlib import Path

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT=ROOT/'receipts/dtpo-v9-negative-clip2-postgroup-variance-20261019-v4'
METHOD='f26b3b37cb3b88e64978e7a472ae4244670a39f3'
owner=ROOT/'receipts/dtpo-symmetric-probability-ratio-postgroup-variance-20261018-v8/source/pin_execution_commit.py'
assert hashlib.sha256(owner.read_bytes()).hexdigest()=='a9d2316b63c1930c259e6ce3fe6f640b2dba321e672cb06ef282ef958ad0c243'
code=owner.read_text().replace("'receipts/dtpo-symmetric-probability-ratio-postgroup-variance-20261018-v8'",
    "'receipts/dtpo-v9-negative-clip2-postgroup-variance-20261019-v4'")
code=code.replace('method_verified_commit=previous,',f'method_verified_commit={METHOD!r},preparation_record_commit=previous,')
code=code.replace('method_commit=previous,',f'method_commit={METHOD!r},preparation_record_commit=previous,')
exec(compile(code,str(owner),'exec'),globals())
plan=json.loads((AUDIT/'six-combination-manifest.json').read_bytes())
paths=[AUDIT/'six-combination-manifest.json']
for segment in plan['segments']:
    p=Path(segment['prepared']);paths.append(p)
    paths.extend(p.parent/(row['method']+'-prepared.json') for row in json.loads(p.read_bytes())['jobs'])
for path in paths:
    row=json.loads(path.read_bytes());row['method_source_commit']=METHOD
    for job in row.get('jobs',[]):job['method_source_commit']=METHOD
    path.write_text(json.dumps(row,indent=2)+'\n')

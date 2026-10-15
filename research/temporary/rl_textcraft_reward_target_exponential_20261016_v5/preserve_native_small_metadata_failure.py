"""Retain completed native math and its subsequent metadata-only failure."""
import hashlib
import json
from pathlib import Path

import psutil

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT=ROOT/'receipts/textcraft-reward-target-exponential-20261016-v5'
job=json.loads((AUDIT/'native-behavior-job.json').read_bytes())
state=json.loads((AUDIT/'native-behavior-status.json').read_bytes())
assert state['status']=='failed' and 'verl.utils.experimental.torch_functional' in state['error']
assert not psutil.pid_exists(job['pid']) or psutil.Process(job['pid']).create_time()!=job['birth']
references=[]
for rank in (0,1):
    path=AUDIT/f'native-small-rank{rank}.json';value=json.loads(path.read_bytes())
    assert value['error']=="KeyError('verl.utils.experimental.torch_functional')"
    assert value['before']==value['after']
    assert len(value['finite_calls'])==1 and len(value['credit_calls'])==4
    assert all(c['exact_expression_equality'] for c in value['credit_calls'])
    assert value['optimizer_updates']==value['generated_rollouts']==value['quality_evaluations']==0
    for item in value['bindings'].values():
        assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()==item['SHA256']
    references.append(dict(rank=rank,path=str(AUDIT/'failed-binding-metadata-r2'/path.name),
        SHA256=hashlib.sha256(path.read_bytes()).hexdigest(),
        completed_computation_preserved=True,metadata_failure=value['error']))
destination=AUDIT/'failed-binding-metadata-r2';destination.mkdir()
for name in ('native-behavior-job.json','native-behavior-status.json','native-behavior.stdout.log',
             'native-behavior.stderr.log','native-small-rank0.json','native-small-rank1.json'):
    (AUDIT/name).rename(destination/name)
(AUDIT/'native-small-reused-receipts.json').write_text(json.dumps(references,indent=2)+'\n')
print('Preserved successful native credit/masks/unchanged state and the later metadata import failure; no repeated DT call.')

"""Archive only completed interface and scaling checks, never a quality claim."""
import hashlib
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT=ROOT/'receipts/dtpo-symmetric-probability-ratio-postgroup-variance-20261018-v8'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
suite=next(ET.parse(AUDIT/'CPU-behavior-tests.xml').getroot().iter('testsuite'))
assert int(suite.attrib['tests'])>=43
assert all(suite.attrib[k]=='0' for k in ('errors','failures','skipped'))
sources={}
for task in ('textcraft','appworld'):
    for method in ('dt','ppo','grpo'):
        p=AUDIT/f'{task}-{method}-source-prepared.json';v=json.loads(p.read_bytes())
        for file,expected in v['source_bindings'].items():assert sha(file)==expected,file
        sources[f'{task}-{method}']=dict(receipt=str(p),SHA256=sha(p),changes=v['changes'],preserved_owners=v['preserved_owners'])
norms={}
for size in (2,8):
    p=AUDIT/f'official-norm-{size}rank.json';v=json.loads(p.read_bytes())
    assert v['status']=='passed' and v['world_size']==size and not v['GPU_initialized']
    norms[str(size)]=dict(receipt=str(p),SHA256=sha(p),value=v['global_reference'])
assert norms['2']['value']==norms['8']['value']
record=dict(status='passed',recorded=time.time(),tests=int(suite.attrib['tests']),
    source_bindings=sources,official_distributed_norm_checks=norms,
    checks=['post-group/event-sum masked_var exactly once','Q/V/A common scale without centering',
        'native multi-target signed/canceling transport','policy/self/future/observation/zero masks',
        'v6/default regression','32768 CPU tensor extent','2/8 rank native DTensor full_tensor norm matches same single-process gradient',
        'backward/optimizer/clip body unchanged, logging precision only'],
    log=dict(path=str(AUDIT/'CPU-behavior-tests.log'),SHA256=sha(AUDIT/'CPU-behavior-tests.log')),
    xml=dict(path=str(AUDIT/'CPU-behavior-tests.xml'),SHA256=sha(AUDIT/'CPU-behavior-tests.xml')),
    GPU_capacity_or_training_quality_claim=False,model_calls=0,optimizer_steps=0,extra_rollouts=0)
(AUDIT/'CPU-behavior-verification.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(dict(status='passed',tests=record['tests'],norm_checks=norms)))

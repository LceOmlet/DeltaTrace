"""Execution-only retry of the CPU-verified SQL multi-message candidate.

The earlier queue was stopped by its GPU occupancy guard before any task.
Keep that receipt and reuse the same native sources, methods and configuration.
"""
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OLD='sql-native-multiturn-speed-20261016-v1'
VERSION='sql-native-multiturn-speed-20261016-v2'
AUDIT=ROOT/'receipts'/VERSION
DEST=ROOT/'candidates'/VERSION
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not DEST.exists(),'Preserve an existing execution candidate'
prior=ROOT/'receipts'/OLD
blocked=read(prior/'trial-status.json')
assert blocked['status']=='reserved_GPU_busy_no_training_stopped' and blocked['tasks']==[]
base_path=ROOT/'candidates'/OLD/'prepared.json'
record=read(base_path)
assert all(sha(path)==expected for path,expected in record['source_bindings'].items())
assert (prior/'CPU-verification.json').exists()
authorities={str(ROOT/name):sha(ROOT/name) for name in ('formal-training.json','active-training.json','active-source.json')}
DEST.mkdir(parents=True);(AUDIT/'source').mkdir(parents=True,exist_ok=True)
old_run=record['run'];run=str(ROOT/'runs'/VERSION/'sql')
record['options']={key:value.replace(old_run,run) if isinstance(value,str) else value
                   for key,value in record['options'].items()}
record['options']['trainer.experiment_name']=VERSION
record['effective_environment']={key:value.replace(old_run,run) if isinstance(value,str) else value
    for key,value in record['effective_environment'].items()}
record['effective_environment']['DT_SQL_NATIVE_FORMAT_RECEIPTS']=str(AUDIT/'native-turns')
record['effective_environment']['CUDA_VISIBLE_DEVICES']='6,7'
# Reuse the official command builder, not a copied Hydra/VERL parser.
import subprocess,os
env=dict(os.environ);env.update(record['effective_environment']);env['CUDA_VISIBLE_DEVICES']=''
builder='import json,sys;sys.path.insert(0,sys.argv[1]);from owner_runtime_options import owner_command;print(json.dumps(owner_command(json.loads(sys.stdin.read()))))'
record['argv']=json.loads(subprocess.check_output([env['VENV_PYTHON'],'-c',builder,record['entry']],
    input=json.dumps(record['options']),text=True,env=env))
assert record['argv'][0]==env['VENV_PYTHON']
record.update(version=VERSION,run=run,status='prepared_only_not_launched',source_commit=sys.argv[1],
    previous_attempt=dict(prepared=str(base_path),SHA256=sha(base_path),queue_guard=blocked),
    execution_changes=['Independent run/audit/version; same physical6/7 after original occupancy blocker exited'],
    method_or_training_hyperparameter_changes=[],CPU_checks_reused=True,model_DT_optimizer_calls=0)
path=DEST/'prepared.json';path.write_text(json.dumps(record,indent=2)+'\n')
(AUDIT/'prepared-index.json').write_text(json.dumps(dict(version=VERSION,tasks=[dict(path=str(path),task=record['task'])]),indent=2)+'\n')
tests={}
for name in ('SQL-boundary-tests.xml','native-multiturn-parity-tests.xml'):
    (AUDIT/name).write_bytes((prior/name).read_bytes());tests[name]=dict(path=str(prior/name),SHA256=sha(prior/name))
(AUDIT/'CPU-verification.json').write_text(json.dumps(dict(status='reused_unchanged_native_sources_CPU12_passed',
    previous_receipt=dict(path=str(prior/'CPU-verification.json'),SHA256=sha(prior/'CPU-verification.json')),
    tests=tests,source_bindings_checked=len(record['source_bindings']),
    model_DT_optimizer_calls=0,method_or_hyperparameter_changes=[],formal_authorities_unchanged=authorities),indent=2)+'\n')
for name in ('run_trials.py','submit_trials.py'):
    owner=ROOT/'receipts/appworld-sql-native-speed-20261016-v1/source'/name
    code=owner.read_text().replace('appworld-sql-native-speed-20261016-v1',VERSION)
    for old,new in [('[2,3]','[6,7]'),("'2,3'","'6,7'"),('(?:2|3)','(?:6|7)'),('GPU2/3','GPU6/7')]:code=code.replace(old,new)
    (AUDIT/'source'/name).write_text(code)
assert authorities=={p:sha(p) for p in authorities}
print(json.dumps(dict(status='prepared_only',version=VERSION,prepared=str(path),CPU_checks=12,
    model_DT_optimizer_calls=0,method_or_hyperparameter_changes=[]),indent=2))

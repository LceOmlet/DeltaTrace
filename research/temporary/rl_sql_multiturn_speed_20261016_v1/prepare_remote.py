"""Check the existing SkyRL multi-message mode against observed format failures."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
VERSION='sql-native-multiturn-speed-20261016-v1'
AUDIT=ROOT/'receipts'/VERSION
DEST=ROOT/'candidates'/VERSION
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not DEST.exists(),'Never overwrite a candidate'
previous_path=ROOT/'candidates/sql-native-format-observation-20261016-v1/prepared.json'
base=read(previous_path)
assert all(sha(path)==expected for path,expected in base['source_bindings'].items())
observed=read(ROOT/'receipts/sql-native-format-observation-20261016-v1/native-format-analysis.json')
assert observed['summary']['native_return_lines']=={'23':18,'41':22}
authorities={str(ROOT/name):sha(ROOT/name) for name in ('formal-training.json','active-training.json','active-source.json')}
entry=DEST/'entry'
entry.mkdir(parents=True)
AUDIT.mkdir(exist_ok=True)
for path in Path(base['entry']).iterdir():
    if path.is_file() and str(path) in base['source_bindings']:
        shutil.copyfile(path,entry/path.name)
path=entry/'owner_environment_configs.json'
cfg=read(path)
assert cfg['SkyRL-SQL']['generator']['use_conversation_multi_turn'] is False
cfg['SkyRL-SQL']['generator']['use_conversation_multi_turn']=True
path.write_text(json.dumps(cfg,indent=2)+'\n')
path=entry/'sql_owner_rollout.py'
before=path.read_text()
start=before.index('            def _update_agent_loop_state_with_singleturn_chat_template')
end=before.index('\n        cfg = GeneratorConfig',start)
method=before[start:end]
assert method.count('def ')==1 and 'super()._update_agent_loop_state_with_singleturn_chat_template' in method
multi=method.replace('_update_agent_loop_state_with_singleturn_chat_template','_update_agent_loop_state_with_multiturn_chat_template')
after=before[:end]+multi+before[end:]
assert after.replace(multi,'',1)==before
compile(after,str(path),'exec')
path.write_text(after)
run=ROOT/'runs'/VERSION/'sql'
options={key:value.replace(base['entry'],str(entry)).replace(base['run'],str(run))
         if isinstance(value,str) else value for key,value in base['options'].items()}
options['trainer.total_training_steps']=2
options['trainer.experiment_name']=VERSION
env=dict(os.environ)
env.update({key:value.replace(base['entry'],str(entry)).replace(base['run'],str(run))
            if isinstance(value,str) else value for key,value in base['effective_environment'].items()})
env['DT_SQL_NATIVE_FORMAT_RECEIPTS']=str(AUDIT/'native-turns')
env['CUDA_VISIBLE_DEVICES']='6,7'
env.pop('MACA_VISIBLE_DEVICES',None)
cpu=dict(env);cpu['CUDA_VISIBLE_DEVICES']=''
cpu['DT_SQL_NATIVE_FORMAT_RECEIPTS']=str(AUDIT/'CPU-fixture-turns')
command='''import json,sys
sys.path.insert(0,sys.argv[1])
from owner_runtime_options import owner_command
print(json.dumps(owner_command(json.loads(sys.stdin.read()))))
'''
argv=json.loads(subprocess.check_output([env['VENV_PYTHON'],'-c',command,str(entry)],input=json.dumps(options),env=cpu,text=True))
assert argv[0]==env['VENV_PYTHON']
resolved=subprocess.run(argv+['--cfg','job','--resolve'],env=cpu,cwd=entry,capture_output=True)
(DEST/'effective-config.yaml').write_bytes(resolved.stdout)
(DEST/'CPU-config.stderr.log').write_bytes(resolved.stderr)
resolved.check_returncode()
bindings={key.replace(base['entry'],str(entry)):value for key,value in base['source_bindings'].items()}
for name in ('owner_environment_configs.json','sql_owner_rollout.py'):
    bindings[str(entry/name)]=sha(entry/name)
assert all(sha(path)==expected for path,expected in bindings.items())
record=dict(base,version=VERSION,run=str(run),entry=str(entry),argv=argv,options=options,
    effective_environment=env,source_bindings=bindings,budget=dict(base['budget'],iterations=2),
    status='prepared_only',previous_prepared=str(previous_path),
    changes=['Original SkyRL use_conversation_multi_turn False->True',
             'Same passive turn-boundary recorder delegates to the original multiturn state update'],
    formal_deployed=False)
prepared_path=DEST/'prepared.json'
prepared_path.write_text(json.dumps(record,indent=2)+'\n')
(AUDIT/'prepared-index.json').write_text(json.dumps(dict(version=VERSION,tasks=[dict(task='SQL',path=str(prepared_path))],
    formal_authorities_unchanged=authorities),indent=2)+'\n')
native_test=subprocess.run([env['VENV_PYTHON'],'-m','pytest','-q','--import-mode=importlib','-p','no:cacheprovider',
    str(entry/'test_sql_owner_rollout.py'),'--junitxml='+str(AUDIT/'native-multiturn-parity-tests.xml')],
    env=cpu,cwd=entry,capture_output=True)
(AUDIT/'native-multiturn-parity-tests.stdout.log').write_bytes(native_test.stdout)
(AUDIT/'native-multiturn-parity-tests.stderr.log').write_bytes(native_test.stderr)
print(native_test.stdout.decode(errors='replace')[-3000:])
native_test.check_returncode()
# Reuse the existing boundary-test adapter and original collector; no new parser/reward/loss implementation.
owner=ROOT/'receipts/appworld-sql-native-speed-20261016-v1/source/verify_sql_boundary.py'
assert sha(owner)=='f166b8346208914645a5cbe57d08a5b75435fc0544ce9caf32ab87d2438540d5'
code=owner.read_text().replace('appworld-sql-native-speed-20261016-v1',VERSION)
code=code.replace(VERSION+'/SQL/prepared.json',VERSION+'/prepared.json')
seam=AUDIT/'source/verify_sql_boundary.py'
seam.write_text(code)
boundary=subprocess.run([env['VENV_PYTHON'],str(seam)],env=cpu,cwd=entry,capture_output=True)
(AUDIT/'boundary-adapter.stdout.log').write_bytes(boundary.stdout)
(AUDIT/'boundary-adapter.stderr.log').write_bytes(boundary.stderr)
print(boundary.stdout.decode(errors='replace')[-3000:])
boundary.check_returncode()
assert authorities=={path:sha(path) for path in authorities}
(AUDIT/'CPU-verification.json').write_text(json.dumps(dict(status='native_multi_message_parity_and_SQL_boundary_verified',
    original_Hydra_config_SHA256=sha(DEST/'effective-config.yaml'),source_bindings=len(bindings),
    model_DT_optimizer_calls=0,original_state_update_and_reward_unchanged=True,
    numerical_tolerances_changed=False),indent=2)+'\n')
print(json.dumps(dict(prepared=str(prepared_path),physical_gpus=[6,7],iterations=2,
    changes=record['changes'],model_DT_optimizer_calls=0)))

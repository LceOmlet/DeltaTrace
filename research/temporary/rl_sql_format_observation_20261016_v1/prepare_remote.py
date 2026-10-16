"""Observe native SQL turn artifacts without altering the owner computation."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
VERSION = 'sql-native-format-observation-20261016-v1'
AUDIT = ROOT / 'receipts' / VERSION
DEST = ROOT / 'candidates' / VERSION
read = lambda p: json.loads(Path(p).read_bytes())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not DEST.exists(), 'Never overwrite a previous candidate'
previous_path = ROOT / 'candidates/appworld-sql-native-speed-20261016-v3/SQL/prepared.json'
base = read(previous_path)
assert all(sha(path) == expected for path, expected in base['source_bindings'].items())
assert read(ROOT / 'receipts/appworld-sql-native-speed-20261016-v3/trial-status.json')['status'] == 'all_native_drivers_finished'
authorities = {str(ROOT/name):sha(ROOT/name) for name in ('formal-training.json','active-training.json','active-source.json')}
entry = DEST / 'entry'
entry.mkdir(parents=True)
AUDIT.mkdir(exist_ok=True)
for path in Path(base['entry']).iterdir():
    if path.is_file() and str(path) in base['source_bindings']:
        shutil.copyfile(path, entry/path.name)
path = entry / 'sql_owner_rollout.py'
before = path.read_text()
old = "'reward': turn.reward, 'output_ids': turn.output_ids.copy(),"
new = old + "\n                    'output_text': turn.output, 'observations': turn.new_obs,"
assert before.count(old) == 1
after = before.replace(old,new)
old_gather = "            return sql_trajectory_batch(self, collector, **collected)"
new_gather = '''            import json, os, time
            from pathlib import Path
            directory = Path(os.environ['DT_SQL_NATIVE_FORMAT_RECEIPTS'])
            directory.mkdir(parents=True, exist_ok=True)
            record = dict(turns=self.rollouts.records,
                          prompt_ids=[event.trajectory.prompt_ids for event in self.current],
                          terminal_scores=[sum(event.trajectory.reward) for event in self.current],
                          stop_reasons=[event.trajectory.stop_reason for event in self.current])
            (directory / ('native-turns-' + str(time.time_ns()) + '.json')).write_text(json.dumps(record) + '\\n')
''' + old_gather
assert after.count(old_gather) == 1
after = after.replace(old_gather, new_gather)
assert after.replace(new_gather,old_gather).replace(new,old) == before
compile(after,str(path),'exec')
path.write_text(after)
run = ROOT / 'runs' / VERSION / 'sql'
options = {key:value.replace(base['entry'],str(entry)).replace(base['run'],str(run))
           if isinstance(value,str) else value for key,value in base['options'].items()}
options['trainer.total_training_steps'] = 1
options['trainer.experiment_name'] = VERSION
env = dict(os.environ)
env.update({key:value.replace(base['entry'],str(entry)).replace(base['run'],str(run))
            if isinstance(value,str) else value for key,value in base['effective_environment'].items()})
env['CUDA_VISIBLE_DEVICES'] = '6,7'
env.pop('MACA_VISIBLE_DEVICES',None)
env['DT_SQL_NATIVE_FORMAT_RECEIPTS'] = str(AUDIT / 'native-turns')
cpu = dict(env)
cpu['CUDA_VISIBLE_DEVICES'] = ''
command = '''import json,sys
sys.path.insert(0,sys.argv[1])
from owner_runtime_options import owner_command
print(json.dumps(owner_command(json.loads(sys.stdin.read()))))
'''
argv = json.loads(subprocess.check_output([env['VENV_PYTHON'],'-c',command,str(entry)],
                                          input=json.dumps(options),env=cpu,text=True))
assert argv[0] == env['VENV_PYTHON']
cfg = subprocess.run(argv+['--cfg','job','--resolve'],env=cpu,cwd=entry,capture_output=True)
(DEST / 'effective-config.yaml').write_bytes(cfg.stdout)
(DEST / 'CPU-config.stderr.log').write_bytes(cfg.stderr)
cfg.check_returncode()
bindings = {key.replace(base['entry'],str(entry)):value for key,value in base['source_bindings'].items()}
bindings[str(path)] = sha(path)
assert all(sha(path) == expected for path,expected in bindings.items())
budget = dict(base['budget'],iterations=1)
record = dict(base,version=VERSION,entry=str(entry),run=str(run),argv=argv,options=options,
              effective_environment=env,source_bindings=bindings,budget=budget,status='prepared_only',
              previous_prepared=str(previous_path),physical_gpus=[6,7],
              observation_scope='Read exact native TurnOutput fields and trajectories; original super/gather/step computation unchanged')
prepared_path = DEST / 'prepared.json'
prepared_path.write_text(json.dumps(record,indent=2)+'\n')
assert authorities == {p:sha(p) for p in authorities}
(AUDIT/'prepared-index.json').write_text(json.dumps(dict(version=VERSION,tasks=[dict(task='SQL',path=str(prepared_path))],
    formal_authorities_unchanged=authorities),indent=2)+'\n')
(AUDIT/'CPU-verification.json').write_text(json.dumps(dict(status='original_Hydra_configuration_verified',
    extra_native_artifact_fields_only=True, exact_reverse_source_replacement=True,
    original_SQL_source=base['source_bindings'][str(Path(base['entry'])/'sql_owner_rollout.py')],
    observed_SQL_source=sha(path),model_DT_optimizer_calls=0),indent=2)+'\n')
shutil.copyfile(ROOT/'receipts/appworld-sql-native-speed-20261016-v3/SQL-boundary-tests.xml',AUDIT/'SQL-boundary-tests.xml')
print(json.dumps(dict(prepared=str(prepared_path),physical_gpus=[6,7],iterations=1,observer_SHA256=sha(path))))

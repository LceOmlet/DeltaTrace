"""Give the existing protected stop owner the two user-replaced DT identities."""
import argparse
import ast
import hashlib
import importlib.util
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[3]
STAGE = REPO / 'research/temporary/rl_upstream_alignment_20260929'
sys.path.insert(0, str(STAGE))
spec = importlib.util.spec_from_file_location('existing_stage', STAGE / 'stage_environment_entry.py')
stage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage)
owner = STAGE / 'resume_at_native_checkpoint.py'
assert hashlib.sha256(owner.read_bytes()).hexdigest() == 'b5be347a4ba7deb2f02b7c892f375642b91d4992290d64f1d7b7c2659b211f0f'
parser = argparse.ArgumentParser()
parser.add_argument('task', choices=('AppWorld', 'TextCraft'))
task = parser.parse_args().task
pid, birth, old_version, key = {
    'AppWorld': (2975137, 1792192446.91, 'appworld-dtpo-reward-target-exponential-formal-20261017-v5', 'AppWorld_DTPO'),
    'TextCraft': (878971, 1792091830.03, 'textcraft-dt-reward-target-exponential-20261016-v5', 'TextCraft_DTPO'),
}[task]
receipt = stage.ROOT + '/receipts/dtpo-exponential-credit-variance-20261017-v6/stop-' + old_version
descriptor = receipt + '/current-source-stop-descriptor.json'
preflight = r'''
from pathlib import Path
import hashlib,json,psutil,subprocess,time
r=Path(@ROOT@);a=r/'receipts/dtpo-exponential-credit-variance-20261017-v6';base=Path(@RECEIPT@)
read=lambda p:json.loads(Path(p).read_bytes());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
j,=[j for j in read(r/'active-training.json')['jobs'] if j.get('task')==@TASK@ and j.get('method')=='dt']
assert (j.get('version'),j['pid'],j['birth'])==(@VERSION@,@PID@,@BIRTH@)
assert psutil.Process(j['pid']).create_time()==j['birth']
new=read(a/(@KEY@+'-prepared.json'));assert new['status']=='CPU_configuration_and_behavior_verified_not_started'
assert all(sha(p)==h for p,h in new['source_bindings'].items())
s=read(j['source_receipt']);assert all(sha(p)==h for p,h in s['source_bindings'].items())
entry=Path(j['entry']);verl=Path(j['verl_root']);dt=Path(j['dt_root'])
partition=lambda root:{str(Path(p).relative_to(root)):h for p,h in s['source_bindings'].items() if Path(p).is_relative_to(root)}
base.mkdir(parents=True,exist_ok=True);assert not(base/'completed-stop.json').exists()
for name in ('formal-training.json','active-training.json','active-source.json'):(base/('before-'+name)).write_bytes((r/name).read_bytes())
(base/'source-before-stop.json').write_bytes(Path(j['source_receipt']).read_bytes())
(base/'train-before-stop.log').write_bytes(Path(j['log']).read_bytes())
(base/'physical-GPU-before-stop.txt').write_text(subprocess.check_output(['mx-smi'],text=True))
d=dict(entry=str(entry),verl_root=str(verl),dt_root=str(dt),entry_sha256=partition(entry),owner_sha256=partition(verl),dt_source_sha256=partition(dt),
       selected_pid=j['pid'],selected_birth=j['birth'],selected_version=j['version'],source_receipt=j['source_receipt'],
       reason='User explicitly requested fresh v6 on both benchmarks; preserve old method history without curve or checkpoint splicing',
       new_prepared_version=new['version'],stop_only=True,checkpoint_restore_requested=False,collected=time.time())
(base/'current-source-stop-descriptor.json').write_text(json.dumps(d,indent=2)+'\n')
print(json.dumps(dict(task=@TASK@,old_pid=j['pid'],new_prepared_version=new['version'],descriptor=str(base/'current-source-stop-descriptor.json'))))
'''
values = {'@ROOT@': repr(stage.ROOT), '@RECEIPT@': repr(receipt), '@TASK@': repr(task),
          '@VERSION@': repr(old_version), '@PID@': str(pid), '@BIRTH@': repr(birth), '@KEY@': repr(key)}
for before, after in values.items():
    preflight = preflight.replace(before, after)
stage.remote('source ' + stage.ENTRY + '/metax-entry.env.sh\nCUDA_VISIBLE_DEVICES="" "$VENV_PYTHON" - <<\'PY\'\n' + preflight + '\nPY\n')
tree = ast.parse(owner.read_text())
template = next(n for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)
                and 'processes=[parent]+parent.children(recursive=True)' in n.value)
original = template.value
signal_block = original[original.index('for p in reversed(processes):'):original.index('record.update(finished_unix=')]
assert original.count("if j['task']==task") == 3
adapted = original.replace("if j['task']==task", "if j['task']==task and j.get('method')=='dt' and j['pid']==" + str(pid))
anchor = "    source=read(source_path)\n"
alias = '''    effective=source['effective_environment']
    partition=lambda root:{str(Path(p).relative_to(root)):h for p,h in source['source_bindings'].items() if Path(p).is_relative_to(root)}
    source.update(dt_root=effective['DT_ROOT'],verl_root=effective['VERL_ROOT'],
                  entry_sha256=partition(Path(old['entry'])),verl_sha256=partition(Path(old['verl_root'])),owner_head_sha256={})
'''
assert adapted.count(anchor) == 1
adapted = adapted.replace(anchor, anchor + alias)
anchor = "other={j['task']:(j['pid'],j['observed_process_created_unix']) for j in active_at_boundary['jobs'] if j['task']!=task}"
assert adapted.count(anchor) == 1
adapted = adapted.replace(anchor, "other={str(j['task'])+':'+str(j.get('method')):(j['pid'],j.get('birth',j.get('observed_process_created_unix'))) for j in active_at_boundary['jobs'] if j['pid']!=old['pid'] and psutil.pid_exists(j['pid']) and psutil.Process(j['pid']).status()!=psutil.STATUS_ZOMBIE}")
assert signal_block in adapted
template.value = adapted
sys.argv = [str(__file__), '--task', task, '--minimum-step', '0', '--prepared', descriptor,
            '--receipt', receipt, '--stop-only', '--stop-now']
exec(compile(ast.fix_missing_locations(tree), str(owner), 'exec'), globals())
reconcile = r'''
from pathlib import Path
import json,time,os,uuid
r=Path(@ROOT@);base=Path(@RECEIPT@);stop=json.loads((base/'completed-stop.json').read_bytes())
assert stop['prior_driver_pid']==@PID@ and stop['prior_created_unix']==@BIRTH@ and not stop['remaining_non_zombie']
for name in ('formal-training.json','active-training.json','active-source.json','receipts/appworld-formal-training-20261017-v1/formal-training.json'):
 p=r/name;d=json.loads(p.read_bytes())
 for j in d.get('jobs',[]):
  if j.get('task')==@TASK@ and j.get('method')=='dt':
   assert j['pid']==@PID@ and j['birth']==@BIRTH@
   j.update(status='user_replaced_by_fresh_v6',active=False,alive=False,stopped_unix=stop['finished_unix'],stop_receipt=str(base/'completed-stop.json'))
 tmp=p.with_name(p.name+'.tmp-'+uuid.uuid4().hex);tmp.write_text(json.dumps(d,indent=2)+'\n');os.replace(tmp,p)
print(json.dumps(dict(task=@TASK@,stopped_driver=@PID@,stop_receipt=str(base/'completed-stop.json'),new_training_submitted=False)))
'''
for before, after in values.items():
    reconcile = reconcile.replace(before, after)
stage.remote('source ' + stage.ENTRY + '/metax-entry.env.sh\nCUDA_VISIBLE_DEVICES="" "$VENV_PYTHON" - <<\'PY\'\n' + reconcile + '\nPY\n')

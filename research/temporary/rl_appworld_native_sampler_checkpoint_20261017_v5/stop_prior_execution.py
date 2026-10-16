"""Pass one exact formal identity to the existing protected stop-only owner."""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

REPO=Path(__file__).resolve().parents[3]
STAGE_DIR=REPO/'research/temporary/rl_upstream_alignment_20260929'
sys.path.insert(0,str(STAGE_DIR))
spec=importlib.util.spec_from_file_location('existing_stage',STAGE_DIR/'stage_environment_entry.py')
stage=importlib.util.module_from_spec(spec);spec.loader.exec_module(stage)
helper=STAGE_DIR/'resume_at_native_checkpoint.py'
assert hashlib.sha256(helper.read_bytes()).hexdigest()=='b5be347a4ba7deb2f02b7c892f375642b91d4992290d64f1d7b7c2659b211f0f'
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--method',choices=('dt','ppo'),required=True)
args=parser.parse_args()
method=args.method
old_version,pid,birth,new_version={
    'dt':('appworld-dtpo-reward-target-exponential-formal-20261017-v4',2477154,1792187585.22,
          'appworld-dtpo-reward-target-exponential-formal-20261017-v5'),
    'ppo':('appworld-native-PPO-formal-20261017-v2',2086772,1792183876.72,
           'appworld-native-PPO-formal-20261017-v3'),
}[method]
receipt=stage.ROOT+'/receipts/appworld-formal-training-20261017-v1/stop-'+old_version+'-sampler-state'
descriptor=receipt+'/current-source-stop-descriptor.json'

preflight=r'''
from pathlib import Path
import hashlib,json,psutil,subprocess,time
r=Path(@ROOT@);read=lambda p:json.loads(Path(p).read_bytes());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
j,=[j for j in read(r/'active-training.json')['jobs'] if j.get('task')=='AppWorld' and j.get('method')==@METHOD@]
assert (j['version'],j['pid'],j['birth'])==(@VERSION@,@PID@,@BIRTH@)
assert psutil.Process(j['pid']).create_time()==j['birth']
new=read(r/'candidates'/@NEW_VERSION@/'prepared.json')
assert new['status']=='CPU_configuration_verified_not_started' and new['sampler_checkpoint_artifact']=='loop_sampler.pt'
assert all(sha(p)==h for p,h in new['source_bindings'].items())
s=read(j['source_receipt']);assert all(sha(p)==h for p,h in s['source_bindings'].items())
entry=Path(j['entry']);verl=Path(j['verl_root']);dt=Path(j['dt_root'])
partition=lambda root:{str(Path(p).relative_to(root)):h for p,h in s['source_bindings'].items() if Path(p).is_relative_to(root)}
base=Path(@RECEIPT@);base.mkdir(parents=True,exist_ok=True)
assert not(base/'completed-stop.json').exists()
for name in ('formal-training.json','active-training.json','active-source.json'):(base/('before-'+name)).write_bytes((r/name).read_bytes())
(base/'source-before-stop.json').write_bytes(Path(j['source_receipt']).read_bytes())
(base/'train-before-stop.log').write_bytes(Path(j['log']).read_bytes())
physical=subprocess.check_output(['mx-smi'],text=True);(base/'physical-GPU-before-stop.txt').write_text(physical)
d=dict(entry=str(entry),verl_root=str(verl),dt_root=str(dt),entry_sha256=partition(entry),owner_sha256=partition(verl),dt_source_sha256=partition(dt),
       source_receipt=j['source_receipt'],source_bindings=s['source_bindings'],selected_method=@METHOD@,selected_version=j['version'],selected_pid=j['pid'],selected_birth=j['birth'],
       reason='Install verified actual LOOP sampler RNG/cursor checkpoint support; existing process has no export command and cannot be losslessly hot-switched',
       stop_only=True,checkpoint_restore_requested=False,new_prepared_version=@NEW_VERSION@,
       new_configuration_receipt=str(r/'receipts'/@NEW_VERSION@/'CPU-verification.json'),original_protected_stop_owner_SHA256=@HELPER_SHA@,
       observed=time.time())
(base/'current-source-stop-descriptor.json').write_text(json.dumps(d,indent=2)+'\n')
print(json.dumps(dict(method=@METHOD@,pid=j['pid'],birth=j['birth'],descriptor=str(base/'current-source-stop-descriptor.json'),new_prepared_version=@NEW_VERSION@)))
'''
values={'@ROOT@':repr(stage.ROOT),'@METHOD@':repr(method),'@VERSION@':repr(old_version),
        '@PID@':str(pid),'@BIRTH@':repr(birth),'@NEW_VERSION@':repr(new_version),
        '@RECEIPT@':repr(receipt),'@HELPER_SHA@':repr(hashlib.sha256(helper.read_bytes()).hexdigest())}
for key,value in values.items():preflight=preflight.replace(key,value)
stage.remote("source "+stage.ENTRY+"/metax-entry.env.sh\nCUDA_VISIBLE_DEVICES=\"\" \"$VENV_PYTHON\" - <<'PY'\n"+preflight+"\nPY\n")

tree=ast.parse(helper.read_text())
template=next(n for n in ast.walk(tree) if isinstance(n,ast.Constant) and isinstance(n.value,str) and 'processes=[parent]+parent.children(recursive=True)' in n.value)
original=template.value
signal_block=original[original.index('for p in reversed(processes):'):original.index("record.update(finished_unix=")]
selected=f"if j['task']==task and j.get('method')=={method!r} and j['pid']=={pid}"
assert original.count("if j['task']==task")==3
adapted=original.replace("if j['task']==task",selected)
alias='''    effective=source['effective_environment']
    partition=lambda root:{str(Path(p).relative_to(root)):h for p,h in source['source_bindings'].items() if Path(p).is_relative_to(root)}
    source.update(dt_root=effective['DT_ROOT'],verl_root=effective['VERL_ROOT'],
                  entry_sha256=partition(Path(old['entry'])),verl_sha256=partition(Path(old['verl_root'])),owner_head_sha256={})
'''
anchor='    source=read(source_path)\n'
assert adapted.count(anchor)==1
adapted=adapted.replace(anchor,anchor+alias)
anchor="other={j['task']:(j['pid'],j['observed_process_created_unix']) for j in active_at_boundary['jobs'] if j['task']!=task}"
assert adapted.count(anchor)==1
adapted=adapted.replace(anchor,"other={str(j['task'])+':'+str(j.get('method'))+':'+str(j['pid']):(j['pid'],j['observed_process_created_unix']) for j in active_at_boundary['jobs'] if j['pid']!=old['pid']}")
assert signal_block in adapted,'Original signal/wait/kill lifecycle must remain byte-identical'
template.value=adapted
sys.argv=[str(__file__),'--task','AppWorld','--minimum-step','0','--prepared',descriptor,'--receipt',receipt,'--stop-only','--stop-now']
exec(compile(ast.fix_missing_locations(tree),str(helper),'exec'),globals())

reconcile=r'''
from pathlib import Path
import json,time,os,uuid
r=Path(@ROOT@);base=Path(@RECEIPT@);stop=json.loads((base/'completed-stop.json').read_bytes());assert stop['prior_driver_pid']==@PID@ and stop['prior_created_unix']==@BIRTH@ and not stop['remaining_non_zombie']
for name in ('formal-training.json','active-training.json','active-source.json','receipts/appworld-formal-training-20261017-v1/formal-training.json'):
 p=r/name;d=json.loads(p.read_bytes())
 for j in d.get('jobs',[]):
  if j.get('task')=='AppWorld' and j.get('method')==@METHOD@:
   assert j['pid']==@PID@ and j['birth']==@BIRTH@
   j.update(status='stopped_to_install_required_native_sampler_checkpoint_state',active=False,alive=False,
            stopped_unix=stop['finished_unix'],stop_receipt=str(base/'completed-stop.json'),complete_AppWorld_sampler_resume_supported=False)
 tmp=p.with_name(p.name+'.tmp-'+uuid.uuid4().hex);tmp.write_text(json.dumps(d,indent=2)+'\n');os.replace(tmp,p)
print(json.dumps(dict(method=@METHOD@,stopped_driver=@PID@,stop_receipt=str(base/'completed-stop.json'),new_training_submitted=False)))
'''
for key,value in values.items():reconcile=reconcile.replace(key,value)
stage.remote("source "+stage.ENTRY+"/metax-entry.env.sh\nCUDA_VISIBLE_DEVICES=\"\" \"$VENV_PYTHON\" - <<'PY'\n"+reconcile+"\nPY\n")

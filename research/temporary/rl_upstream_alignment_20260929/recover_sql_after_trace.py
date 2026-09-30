"""Restart the terminal SQL job with its already-deployed actor source.

This operational receipt does not implement recovery or training algorithms.
No completed SQL checkpoint exists; the unchanged native formal command starts
from the original model. Other jobs and all training settings are preserved.
"""
import hashlib
import subprocess
from pathlib import Path

from stage_environment_entry import remote, ROOT, ENTRY, REPO


def main():
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    remote(r'''set -e
source @ENTRY@/metax-entry.env.sh
"$VENV_PYTHON" - <<'PY'
from pathlib import Path
import hashlib,json,os,psutil,re,shutil,subprocess,time
root=Path('@ROOT@');read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
active=read(root/'active-training.json');old=next(j for j in active['jobs'] if j['task']=='SkyRL-SQL')
assert old['pid']==1876409 and old['devices']==[0,1]
assert not any(psutil.pid_exists(pid) for pid in [1876409,1880429,1884758,1886413])
assert not list(Path(old['checkpoints']).rglob('latest_checkpointed_iteration.txt'))
state=subprocess.check_output(['mx-smi'],text=True)
for device in old['devices']:
    assert not re.search(r'\|\s*'+str(device)+r'\s+\d+\s+\S',state),state
overlay_path=root/'receipts/owner-b8-dispatch-20260930/actor-response-padding/sql-live/complete.json'
overlay=read(overlay_path)
workers=overlay if isinstance(overlay,list) else overlay['workers']
assert sorted(w['pid'] for w in workers)==[1884758,1886413]
actor_files={w['effective_forward_source'] for w in workers};assert len(actor_files)==1
actor=Path(actor_files.pop())
assert sha(actor)=='1f862e8bbdaad6fa116d0670772ad41269529a3a1e4a5b1eb383352d0372e9bd'
old_entry=Path(old['entry']);old_verl=Path(old['verl_root'])
prior_source=read(Path(old['source_receipt']))
for name,expected in prior_source['entry_sha256'].items():assert sha(old_entry/name)==expected,name
lock=read(old_entry/'verified_runtime.json')
for name,expected in lock['dt_source_sha256'].items():assert sha(Path(old['dt_root'])/name)==expected,name
base=root/'runs/sql-padding-restart-20261001';entry=base/'sql-entry'
verl=root/'candidates/official-verl-20bd331-sql-padding-20261001'
assert not base.exists() and not verl.exists(), 'Inspect an existing submission; never overwrite or duplicate it'
base.mkdir()
for name in ['active-training.json','active-source.json']:(base/('prior-'+name)).write_bytes((root/name).read_bytes())
shutil.copytree(old_entry,entry,ignore=shutil.ignore_patterns('__pycache__'))
shutil.copytree(old_verl,verl,ignore=shutil.ignore_patterns('__pycache__','.pytest_cache'))
shutil.copy2(actor,verl/'verl/workers/actor/dp_actor.py')
entry_hash={name:sha(entry/name) for name in prior_source['entry_sha256']}
assert entry_hash==prior_source['entry_sha256']
owner_hash={str(p.relative_to(verl)):sha(p) for p in (verl/'verl').rglob('*.py')}
changed=[n for n,h in owner_hash.items() if h!=sha(old_verl/n)]
assert changed==['verl/workers/actor/dp_actor.py'],changed
output=base/'sql-dt';output.mkdir()
source=dict(prior_source,unix=time.time(),entry_sha256=entry_hash,verl_root=str(verl),verl_sha256=owner_hash,
    numerical_override={name:sha(verl/name) for name in prior_source.get('numerical_override',{})},
    submission_repository_commit=@REVISION@,submission_script_sha256=@SCRIPT_SHA@,
    prior_driver_pid=old['pid'],prior_source_receipt=old['source_receipt'],prior_source_sha256=sha(Path(old['source_receipt'])),
    completed_overlay_receipt=str(overlay_path),completed_overlay_sha256=sha(overlay_path),
    actor_padding_sha256=sha(actor),resumed_checkpoint=None,
    reason='SQL rank1 exited immediately after native mcTracer detach; original driver/other rank then exited. No completed formal checkpoint exists. Freeze the previously effective source and restart the original formal command; no profile attach on production workers again.')
(output/'source.json').write_text(json.dumps(source,indent=2)+'\n')
env=os.environ.copy();env.pop('MACA_VISIBLE_DEVICES',None)
env.update(VERL_ROOT=str(verl),DT_ENTRY_ROOT=str(entry),DT_ROOT=old['dt_root'],CUDA_VISIBLE_DEVICES='0,1')
env['PYTHONPATH']=':'.join([str(entry),str(verl),env['PYTHONPATH']])
argv=[str(entry/'launch_sql_native.py') if a==str(old_entry/'launch_sql_native.py') else str(output) if a==old['output'] else a for a in old['argv']]
assert argv[0]==env['VENV_PYTHON'] and argv[1]==str(entry/'launch_sql_native.py')
with (output/'train.log').open('wb') as stream:
    proc=subprocess.Popen(argv,env=env,cwd=str(output),stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
job=dict(old,pid=proc.pid,observed_process_created_unix=psutil.Process(proc.pid).create_time(),started_unix=time.time(),
    argv=argv,entry=str(entry),verl_root=str(verl),output=str(output),log=str(output/'train.log'),
    checkpoints=str(output/'checkpoints'),source_receipt=str(output/'source.json'),
    status='formal_restart_submitted_after_profiler_related_exit',prior_driver_pid=old['pid'])
(output/'job.json').write_text(json.dumps(job,indent=2)+'\n')
manifest=dict(active,manifest=str(base/'formal-training.json'),jobs=[job if j['task']=='SkyRL-SQL' else j for j in active['jobs']])
manifest.setdefault('retired_jobs',[]).append(dict(old,status='terminal_after_native_profiler_detach',replacement_pid=proc.pid))
Path(manifest['manifest']).write_text(json.dumps(manifest,indent=2)+'\n')
(root/'active-training.json').write_text(json.dumps(manifest,indent=2)+'\n')
sources=read(root/'active-source.json');sources.update(unix=time.time(),manifest=manifest['manifest'])
sources['jobs']=[dict(task=job['task'],pid=job['pid'],entry=job['entry'],verl_root=job['verl_root'],source_receipt=job['source_receipt'],runtime_override=None,actor_microbatch=4,lora_rank=8,lora_alpha=16) if x['task']=='SkyRL-SQL' else x for x in sources['jobs']]
(root/'active-source.json').write_text(json.dumps(sources,indent=2)+'\n')
print(json.dumps({k:job[k] for k in ['task','pid','entry','verl_root','log','status']},indent=2))
PY
'''.replace('@ROOT@',ROOT).replace('@ENTRY@',ENTRY)
       .replace('@REVISION@',repr(revision)).replace('@SCRIPT_SHA@',repr(script_sha)))


if __name__ == '__main__':
    main()

"""Read-only deployment snapshot; never imports or launches training code."""
import json,subprocess
from pathlib import Path
from stage_environment_entry import remote, ROOT, ENTRY, AUDIT, REPO, SSH, SCP

remote(r'''set -e
source @ENTRY@/metax-entry.env.sh
"$VENV_PYTHON" - <<'PY'
from pathlib import Path
import hashlib,json,time,psutil,datetime,re
root=Path('@ROOT@');read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
active=read(root/'active-training.json');lock=read(Path('@ENTRY@')/'verified_runtime.json')
out=root/'receipts/owner-b8-dispatch-20260930';now=time.time()
def artifact(path):
    p=Path(path)
    return dict(path=str(p),exists=p.is_file(),sha256=sha(p) if p.is_file() else None)
result=dict(observed_unix=now,observed_utc=datetime.datetime.fromtimestamp(now,datetime.timezone.utc).isoformat(),
    role='Read-only snapshot, not a launcher or a replacement for remote active-training.json; never infer current execution from old receipts alone',
    authoritative_remote={n:artifact(root/n) for n in ['active-training.json','active-source.json']},
    manifest=artifact(active['manifest']),baseline=artifact(Path('@ENTRY@')/'verified_runtime.json'),jobs=[])
for j in active['jobs']:
    entry=Path(j['entry']);vr=Path(j['verl_root']);src=Path(j.get('source_receipt',str(Path(j['output'])/'source.json')))
    source=read(src)
    rec=dict(task=j['task'],method=j['method'],pid=j['pid'],devices=j['devices'],started_unix=j['started_unix'],
        entry=str(entry),verl_root=str(vr),dt_root=j['dt_root'],source=artifact(src),output=j['output'],log=j['log'],
        checkpoints=j['checkpoints'],budget=j['budget'],launch=artifact(Path(j['output'])/'launch.json'),
        declared_resource_config={k:j.get(k) for k in ['lora_rank','lora_alpha','actor_microbatch','log_prob_micro_batch_size_per_gpu']})
    if psutil.pid_exists(j['pid']):
        proc=psutil.Process(j['pid']);rec['process']=dict(alive=proc.is_running(),created_unix=proc.create_time(),
            pid_identity_matches=abs(proc.create_time()-j['observed_process_created_unix'])<.02,status=proc.status(),
            workers=[p.pid for p in proc.children(recursive=True) if 'WorkerDict' in p.name()])
    else:rec['process']=dict(alive=False)
    rec['entry_files']={}
    for n,h in source.get('entry_sha256',{}).items():
        p=entry/n;actual=sha(p) if p.is_file() else None
        rec['entry_files'][n]=dict(recorded_sha256=h,actual_sha256=actual,matches=actual==h)
    rec['owner_files']={n:artifact(vr/n) for n in [
        'verl/workers/actor/dp_actor.py','verl/workers/fsdp_workers.py','verl/trainer/ppo/core_algos.py',
        'verl/utils/experimental/torch_functional.py','verl/models/transformers/monkey_patch.py',
        'verl/models/transformers/qwen3_vl.py','verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py']}
    rec['runtime_overrides']=[]
    # These are completed original worker RPC receipts, not new live imports.
    for path in [root/'receipts/owner-entropy-20260930'/f"live-{j['task']}-complete.json",
                 out/f"live-{j['task']}-complete.json"]:
        if not path.is_file():continue
        value=read(path);workers=value if isinstance(value,list) else value.get('workers',[])
        belongs=workers and all(w['pid'] in rec['process'].get('workers',[]) for w in workers)
        if not belongs:continue
        effective_files={}
        for w in workers:
            if w.get('effective_forward_source'):
                p=Path(w['effective_forward_source']);expected=w['effective_source_sha256']
                effective_files[str(p)]=dict(**artifact(p),expected_sha256=expected,matches=sha(p)==expected)
            for n,expected in w.get('source_sha256',{}).items():
                p=Path(w['source'])/n
                effective_files[str(p)]=dict(**artifact(p),expected_sha256=expected,matches=sha(p)==expected)
        rec['runtime_overrides'].append(dict(receipt=artifact(path),workers=workers,effective_files=effective_files,
            applicability='Completed RPC receipt for these currently alive worker PIDs; does not claim a fresh method introspection'))
    rec['environment_roots']={k:v for k,v in source.items() if k.endswith('_root') and k not in ['dt_root','verl_root']}
    with Path(j['log']).open('rb') as f:
        f.seek(0,2);f.seek(max(0,f.tell()-200000));tail=f.read().decode(errors='replace')
    rec['recent_progress']=[re.sub(r'\x1b\[[0-9;]*[mA]','',s) for s in tail.splitlines()
        if re.search(r'Rounds \d|n_rollouts_collected=|actor/grad_norm|step:|Traceback|OutOfMemory',s)][-3:]
    rec['completed_checkpoints']=[dict(path=str(p),value=p.read_text()) for p in Path(j['checkpoints']).rglob('latest_checkpointed_iteration.txt')]
    result['jobs'].append(rec)
result['unchanged_numerical_files']={}
for n,h in lock['dt_source_sha256'].items():
    p=Path(lock['paths']['dt'])/n
    result['unchanged_numerical_files']['DT/'+n]=dict(path=str(p),sha256=sha(p),expected_sha256=h,matches=sha(p)==h)
for n,item in lock['native_fla_files'].items():
    p=Path(item['path']);h=item['expected']
    result['unchanged_numerical_files']['FLA/'+n]=dict(path=str(p),sha256=sha(p),expected_sha256=h,matches=sha(p)==h)
for item in lock['installed_restored_files']:
    p=Path(item['path']);h=item['expected_verified_sha256']
    result['unchanged_numerical_files'][str(p)]=dict(path=str(p),sha256=sha(p),expected_sha256=h,matches=sha(p)==h)
(out/'current-runtime-snapshot.json').write_text(json.dumps(result,indent=2))
print(json.dumps(dict(observed_utc=result['observed_utc'],jobs=[dict(task=j['task'],pid=j['pid'],
    alive=j['process']['alive'],entry_hash_mismatches=[n for n,v in j['entry_files'].items() if not v['matches']],
    runtime_receipts=len(j['runtime_overrides']),progress=j['recent_progress']) for j in result['jobs']],
    numerical_hash_mismatches=[n for n,v in result['unchanged_numerical_files'].items() if not v['matches']]),indent=2))
PY
'''.replace('@ROOT@',ROOT).replace('@ENTRY@',ENTRY))
target=REPO/'experiments/rl/current_runtime.json'
subprocess.run(SCP+[f'{SSH[-1]}:{ROOT}/receipts/owner-b8-dispatch-20260930/current-runtime-snapshot.json',str(target)],check=True)
result=json.loads(target.read_text())
result['code_repository_commit_at_collection']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
result['recording_command']='C:/Users/Administrator/miniconda3/python.exe -X utf8 research/temporary/rl_upstream_alignment_20260929/record_current_runtime.py'
revision=lambda short:subprocess.check_output(['git','rev-parse',short],cwd=REPO,text=True).strip()
baseline=json.loads((REPO/'experiments/rl/verified_runtime.json').read_text())
result['version_mapping']=dict(
    dt_runtime_release=revision(baseline['runtime_release']),
    dt_numerical_reference=revision(baseline['dt_reference_revision']),
    verl_upstream_commit=baseline['verl_owner_commit'],
    latest_actor_head_resource_fix_commit=revision('dc4e4d7'),
    version_record_introduced_commit=revision('071b751'),
    meaning='Deployment directory IDs, upstream commits, DT numerical reference, actor patch commit and documentation commit are distinct identifiers; none substitutes for another',
    packages=baseline['packages'])
for job in result['jobs']:
    # Only label the head with this fix when the actual recorded file digest
    # agrees. A later runtime change must remain explicit, never be relabeled.
    expected_files={
        'verl/utils/experimental/torch_functional.py':'e285c3353bddbffae38df346b44014ee5038b606531874ddbeb10ee1241f77de',
        'verl/models/transformers/monkey_patch.py':'3c78654e0ebada3727a852d8e96720eb59053a391a361ee44056dc96c449d693',
        'verl/models/transformers/qwen3_vl.py':'ebc52fb35812a9b0a5d9251f4e42e6fdbc15a076308390f2da81e47eea56a9d7'}
    effective={n:job['owner_files'][n]['sha256'] for n in expected_files}
    for overlay in job['runtime_overrides']:
        for name,record in overlay['effective_files'].items():
            for n in expected_files:
                if name.endswith('/'+n):effective[n]=record['sha256']
    matches=effective==expected_files
    job['version_mapping']=dict(startup_deployment_directory=Path(job['entry']).parent.name,
        actor_head_fix_commit=revision('dc4e4d7') if matches else None,
        actor_fix_identification='matched actual head, dispatch and original wrapper SHA' if matches else 'unidentified: inspect source, do not assume this patch is active',
        actor_head_effective_sha256=effective,
        runtime_overrides_present=bool(job['runtime_overrides']))
    for name,record in job['entry_files'].items():
        p=REPO/'experiments/rl'/name
        if p.is_file():
            import hashlib
            record['workspace_sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
            record['matches_workspace_bytes']=record['workspace_sha256']==record['actual_sha256']
target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print('Saved',target)

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
    rec['startup_provenance']={k:source[k] for k in [
        'submission_repository_commit','submission_script_sha256','prepared_receipt',
        'prepared_receipt_sha256','prior_driver_pid','resume_from','completed_checkpoint_marker',
        'actor_fix_commit','dt_dispatch_commit','resume_entry_commit','actor_padding_sha256',
        'padding_comparison_receipt','padding_comparison_receipt_sha256'] if k in source}
    rec['native_training_workload']=[]
    if psutil.pid_exists(j['pid']):
        proc=psutil.Process(j['pid']);rec['process']=dict(alive=proc.is_running(),created_unix=proc.create_time(),
            pid_identity_matches=abs(proc.create_time()-j['observed_process_created_unix'])<.02,status=proc.status(),
            workers=[p.pid for p in proc.children(recursive=True) if 'WorkerDict' in p.name()])
        for child in proc.children(recursive=True):
            if 'TaskRunner' not in child.name():continue
            paths={f.path for f in child.open_files() if '/worker-' in f.path and f.path.endswith('.out')}
            for path in paths:
                with Path(path).open(errors='replace') as stream:
                    lines=[line.strip() for line in stream if line.startswith(
                        ('Size of train dataloader:', 'Total training steps:'))]
                if lines:rec['native_training_workload'].append(dict(pid=child.pid,path=path,lines=lines,
                    meaning='Original trainer log; epochs, rollout iterations and optimizer updates are distinct units'))
    else:rec['process']=dict(alive=False)
    rec['entry_files']={}
    for n,h in source.get('entry_sha256',{}).items():
        p=entry/n;actual=sha(p) if p.is_file() else None
        rec['entry_files'][n]=dict(recorded_sha256=h,actual_sha256=actual,matches=actual==h)
    rec['owner_files']={n:artifact(vr/n) for n in [
        'verl/workers/actor/dp_actor.py','verl/workers/fsdp_workers.py','verl/trainer/ppo/core_algos.py',
        'verl/utils/experimental/torch_functional.py','verl/models/transformers/monkey_patch.py',
        'verl/models/transformers/qwen3_vl.py','verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py']}
    for name,record in rec['owner_files'].items():
        expected=source.get('verl_sha256',{}).get(name)
        record['startup_recorded_sha256']=expected
        record['matches_startup_source']=record['sha256']==expected if expected is not None else None
    rec['runtime_overrides']=[]
    rec['pending_runtime_operations']=[]
    if j['task']=='SkyRL-SQL':
        submitted=out/'actor-response-padding/sql-live/submitted.json'
        completed=submitted.with_name('complete.json')
        if submitted.is_file() and not completed.is_file():
            operation=read(submitted)
            if (operation['driver_pid']==j['pid'] and
                    operation['driver_created_unix']==rec['process'].get('created_unix')):
                rec['pending_runtime_operations'].append(dict(receipt=artifact(submitted),
                    operation=operation,
                    meaning='Submission exists without completion receipt; not an effective code override or proof the submitting process is still alive'))
    # These are completed original worker RPC receipts, not new live imports.
    override_paths=[root/'receipts/owner-entropy-20260930'/f"live-{j['task']}-complete.json",
                    out/f"live-{j['task']}-complete.json"]
    if j['task']=='SkyRL-SQL':
        override_paths.append(out/'actor-response-padding/sql-live/complete.json')
    for path in override_paths:
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
    lines=[re.sub(r'^\([^\n)]* pid=\d+\)\s*','',re.sub(r'\x1b\[[0-9;]*[mA]','',s))
           for s in tail.splitlines()]
    rec['recent_progress']=[s for s in lines if re.match(
        r'Rounds \d|n_rollouts_collected=|step:\d+\b|\[loop_transport\]|\[owner_trajectory\]|Traceback \(most recent call last\)|(?:torch\.)?(?:cuda\.)?OutOfMemoryError:',s)][-3:]
    rec['completed_checkpoints']=[dict(path=str(p),value=p.read_text()) for p in Path(j['checkpoints']).rglob('latest_checkpointed_iteration.txt')]
    result['jobs'].append(rec)
result['prepared_versions']=[]
for label,receipt_dir,supersedes in [
    ('appworld-balanced-resume-20261001','appworld-balanced-resume',None),
    ('appworld-balanced-padding-resume-20261001','appworld-balanced-padding-resume',
     'appworld-balanced-resume-20261001'),
    ('appworld-request-dispatch-20261001','appworld-request-dispatch',
     'appworld-balanced-padding-resume-20261001')]:
    path=out/receipt_dir/'prepared.json'
    if not path.is_file():continue
    value=read(path)
    prepared=dict(id=label,receipt=artifact(path),prepared_unix=value['prepared_unix'],
        entry=value['entry'],verl_root=value['verl_root'],dt_root=value['dt_root'],
        supersedes_for_future_appworld_resume=supersedes,
        active_jobs_with_this_entry=[j['task'] for j in result['jobs'] if j['entry']==value['entry']],
        entry_files={},owner_files={},tests=artifact(path.parent/'cpu-tests.xml'))
    for name,expected in value['entry_sha256'].items():
        p=Path(value['entry'])/name
        prepared['entry_files'][name]=dict(**artifact(p),expected_sha256=expected,
                                         matches=p.is_file() and sha(p)==expected)
    for name,expected in value['owner_head_sha256'].items():
        p=Path(value['verl_root'])/name
        prepared['owner_files'][name]=dict(**artifact(p),expected_sha256=expected,
                                         matches=p.is_file() and sha(p)==expected)
    if value.get('padding_comparison_receipt'):
        p=Path(value['padding_comparison_receipt'])
        prepared['padding_comparison']=dict(**artifact(p),
            expected_sha256=value['padding_comparison_receipt_sha256'],
            matches=p.is_file() and sha(p)==value['padding_comparison_receipt_sha256'])
    if value.get('request_dispatch_receipt'):
        p=Path(value['request_dispatch_receipt'])
        prepared['request_dispatch']=dict(**artifact(p),
            code_commit=value['request_dispatch_commit'],
            expected_sha256=value['request_dispatch_receipt_sha256'],
            matches=p.is_file() and sha(p)==value['request_dispatch_receipt_sha256'])
    result['prepared_versions'].append(prepared)
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
import hashlib
recorder=Path(__file__).resolve()
result['recorder_source']=dict(path=str(recorder.relative_to(REPO)).replace('\\','/'),
    sha256=hashlib.sha256(recorder.read_bytes()).hexdigest(),
    last_committed_change=subprocess.check_output(['git','log','-1','--format=%H','--',str(recorder)],cwd=REPO,text=True).strip(),
    differs_from_head=bool(subprocess.check_output(['git','diff','HEAD','--',str(recorder)],cwd=REPO,text=True)))
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
for prepared in result['prepared_versions']:
    prepared['source_commits']=dict(dt_dispatch=revision('4c0cbdd'),
        native_resume_entry=revision('2036246'),actor_head=revision('dc4e4d7'))
    if prepared['id'] in ('appworld-balanced-padding-resume-20261001','appworld-request-dispatch-20261001'):
        prepared['source_commits'].update(padding_source_archive=revision('0c80b41'),
                                         padding_owner_comparison=revision('44e1149'))
        prepared['local_helpers']={}
        prepare_script=('prepare_appworld_request_resume.py' if prepared['id']=='appworld-request-dispatch-20261001'
                        else 'prepare_appworld_padding_resume.py')
        for name in [prepare_script,'submit_prepared_appworld_resume.py']:
            p=recorder.parent/name;relative=p.relative_to(REPO).as_posix()
            prepared['local_helpers'][name]=dict(path=relative,
                sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                last_committed_change=subprocess.check_output(['git','log','-1','--format=%H','--',relative],cwd=REPO,text=True).strip(),
                matches_head_bytes=p.read_bytes()==subprocess.check_output(['git','show','HEAD:'+relative],cwd=REPO))
        if prepared['id']=='appworld-request-dispatch-20261001':
            prepared['source_commits']['request_dispatch']=prepared['request_dispatch']['code_commit']
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
    actor_name='verl/workers/actor/dp_actor.py'
    sources={None:job['owner_files'][actor_name]}
    for overlay in job['runtime_overrides']:
        for worker in overlay['workers']:
            path=worker.get('effective_forward_source')
            if not path and worker.get('source_sha256',{}).get(actor_name):
                path=worker['source']+'/'+actor_name
            if path and path in overlay['effective_files']:
                sources[worker['rank']]=overlay['effective_files'][path]
    if len(sources)>1:sources.pop(None)
    job['version_mapping']['actor_forward_sources']=[dict(rank=rank,**source) for rank,source in sources.items()]
    job['version_mapping']['actor_forward_evidence']='Frozen startup source plus completed PID-bound overrides; no live method introspection'
    for name,record in job['entry_files'].items():
        p=REPO/'experiments/rl'/name
        if p.is_file():
            import hashlib
            record['workspace_sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
            record['matches_workspace_bytes']=record['workspace_sha256']==record['actual_sha256']
target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print('Saved',target)

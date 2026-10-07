source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
"$VENV_PYTHON" - <<'PY'

from pathlib import Path
import hashlib,json,psutil,subprocess,time
root=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
base=root/'receipts/direct-target-semantics-20261007'
expected={'TextCraft':(3327460,1791334993.84),'AppWorld':(2360541,1791325655.01)}
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
records={}
for task,(pid,birth) in expected.items():
    path=base/task.lower()/'direct-target-stop-20261007/completed-stop.json'
    stopped=read(path)
    assert stopped['prior_driver_pid']==pid and stopped['prior_created_unix']==birth
    assert stopped['remaining_non_zombie']==[]
    remaining=[]
    for old in stopped['processes']:
        try:
            p=psutil.Process(old['pid'])
            if p.create_time()==old['created_unix'] and p.status()!=psutil.STATUS_ZOMBIE:remaining.append(p.pid)
        except psutil.NoSuchProcess:pass
    assert not remaining,remaining
    records[task]=dict(path=str(path),sha256=sha(path),pid=pid,pid_birth=birth,
        stopped_unix=stopped['finished_unix'],remaining_non_zombie=[],
        last_marker_observed=stopped['marker_step'],checkpoint=stopped['checkpoint'],
        checkpoint_observation_scope=stopped['checkpoint_observation_scope'])
active=read(root/'active-training.json')
paths={root/'active-training.json',root/'formal-training.json',Path(active['manifest'])}
for path in paths:
    manifest=read(path)
    for job in manifest['jobs']:
        if job['task'] not in expected:continue
        pid,birth=expected[job['task']]
        assert job['pid']==pid and job['observed_process_created_unix']==birth
        item=records[job['task']]
        job.update(status='stopped_auxiliary_label_target_semantics',stopped_unix=item['stopped_unix'],
            stop_receipt=item['path'],stop_receipt_sha256=item['sha256'],
            stop_reason='Auxiliary forecast/category target differs from the actual generated verifier/execution input requested by the user; no replacement method deployed.')
    manifest['last_status_observed_unix']=time.time()
    path.write_text(json.dumps(manifest,indent=2)+'\n')
sources=read(root/'active-source.json')
for job in sources['jobs']:
    if job['task'] not in expected:continue
    assert job['pid']==expected[job['task']][0]
    item=records[job['task']]
    job.update(status='stopped_auxiliary_label_target_semantics',stop_receipt=item['path'],
        stop_receipt_sha256=item['sha256'],stopped_unix=item['stopped_unix'])
sources['unix']=time.time();(root/'active-source.json').write_text(json.dumps(sources,indent=2)+'\n')
backup=dict(pid=3591046,pid_birth=1791337520.95)
try:
    p=psutil.Process(backup['pid']);backup.update(same_identity=p.create_time()==backup['pid_birth'],status=p.status())
except psutil.NoSuchProcess:backup['status']='not_present'
physical=subprocess.check_output(['mx-smi'],text=True)
receipt=dict(observed_unix=time.time(),scope='Original stop receipts, exact recorded process identities, manifest status reconciliation and physical GPU only. No model, environment RPC, checkpoint creation/restore or replacement training.',
    stopped=records,metadata_paths={str(p):sha(p) for p in paths},active_source_sha256=sha(root/'active-source.json'),
    physical_gpu=physical,backup_identity_observed=backup,training_submitted=False,checkpoint_restore_requested=False)
(base/'terminal-state.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))

PY

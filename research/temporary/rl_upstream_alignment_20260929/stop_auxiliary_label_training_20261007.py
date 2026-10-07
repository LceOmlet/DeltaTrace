"""Compose the existing stop-only lifecycle for the two identity-bound jobs.

This does not implement training, a verifier, checkpoint creation or recovery.
It stops the rejected auxiliary-label experiment through the same lifecycle
owner's stop-only path, without waiting for or creating another checkpoint.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import stage_environment_entry as stage

LOCAL = stage.AUDIT / 'direct-target-semantics-20261007'
REMOTE = stage.ROOT + '/receipts/direct-target-semantics-20261007'
EXPECTED = {
    'TextCraft': (3327460, 1791334993.84,
                  '5faa6e2d9e7be2b2f55e8014d3b648b944d725bf912f76410633574b6b80fbb8'),
    'AppWorld': (2360541, 1791325655.01,
                 'c83b96debba90d85476e58b56b2c8c7f3900901f26ce41094e8f60ac489c7789'),
}
CODE = r'''
from pathlib import Path
import hashlib,json,psutil,time
root=Path(@ROOT@);base=Path(@REMOTE@);expected=@EXPECTED@
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
active=json.loads((root/'active-training.json').read_bytes());out=[]
base.mkdir(parents=True,exist_ok=True)
for task,(pid,birth,source_sha) in expected.items():
    job=next(j for j in active['jobs'] if j['task']==task)
    assert job['pid']==pid and job['observed_process_created_unix']==birth
    parent=psutil.Process(pid);assert parent.create_time()==birth
    assert parent.status()!=psutil.STATUS_ZOMBIE
    assert sha(job['source_receipt'])==source_sha
    source=json.loads(Path(job['source_receipt']).read_bytes())
    assert sha(Path(job['entry'])/'reward_readout.py')=='94a7afbc09da72b62572d31fd32a6534f6e8f3daf656fce1011cdfa68b3c3e2b'
    marker=Path(job['checkpoints'])/'latest_checkpointed_iteration.txt'
    try:step=int(marker.read_text().strip())
    except FileNotFoundError:step=0
    folder=base/task.lower();folder.mkdir(exist_ok=True)
    prepared=folder/'current-source-stop-descriptor-v2.json'
    assert not prepared.exists(), 'Inspect the prior descriptor rather than repeat the transition'
    owner=dict(source['verl_sha256'])
    for name,digest in source.get('owner_head_sha256',{}).items():
        assert name not in owner or owner[name]==digest
        owner[name]=digest
    descriptor=dict(entry=job['entry'],verl_root=job['verl_root'],dt_root=job['dt_root'],
        entry_sha256=source['entry_sha256'],owner_sha256=owner,
        dt_source_sha256=source['dt_source_sha256'],source_receipt=job['source_receipt'],
        source_receipt_sha256=source_sha,reason='Current target is an appended forecast/category label, not the actual sampled official verifier/execution input requested by the user.',
        stop_only=True,checkpoint_restore_requested=False,observed_unix=time.time())
    prepared.write_text(json.dumps(descriptor,indent=2)+'\n')
    out.append(dict(task=task,pid=pid,birth=birth,source_sha256=source_sha,
        minimum_step=step,prepared=str(prepared),receipt=str(folder/'direct-target-stop-20261007'),
        checkpoint_marker=str(marker),marker_sha256=sha(marker) if marker.is_file() else None))
for name in ('active-training.json','formal-training.json','active-source.json'):
    (base/('before-v2-'+name)).write_bytes((root/name).read_bytes())
print(json.dumps(dict(observed_unix=time.time(),jobs=out)))
'''

if __name__ == '__main__':
    LOCAL.mkdir(parents=True,exist_ok=True)
    output=LOCAL/'prepared-stop-boundary-v2.json'
    assert not output.exists(), 'Inspect the existing boundary and stop receipts; do not repeat'
    code=CODE.replace('@ROOT@',repr(stage.ROOT)).replace('@REMOTE@',repr(REMOTE)).replace('@EXPECTED@',repr(EXPECTED))
    command='source '+stage.ENTRY+'/metax-entry.env.sh\n"$VENV_PYTHON" - <<\'PY\'\n'+code+'\nPY\n'
    (LOCAL/'prepare-stop-boundary.sh').write_text(command,encoding='utf-8')
    result=subprocess.run(stage.SSH+['bash','-s'],input=command.encode(),capture_output=True,timeout=60)
    (LOCAL/'prepare-stop-boundary.stderr.txt').write_bytes(result.stderr)
    if result.returncode:
        print(result.stderr.decode(errors='replace'))
        result.check_returncode()
    output.write_bytes(result.stdout)
    observed=json.loads(result.stdout)
    for job in observed['jobs']:
        subprocess.run([sys.executable,str(stage.AUDIT/'resume_at_native_checkpoint.py'),
            '--task',job['task'],'--minimum-step',str(job['minimum_step']),
            '--prepared',job['prepared'],'--receipt',job['receipt'],'--stop-only','--stop-now'],check=True)
    print(json.dumps(dict(stopped_tasks=[j['task'] for j in observed['jobs']],
        existing_helper='resume_at_native_checkpoint.py --stop-only --stop-now',
        helper_sha256=hashlib.sha256((stage.AUDIT/'resume_at_native_checkpoint.py').read_bytes()).hexdigest(),
        checkpoint_restore_requested=False,training_submitted=False)))

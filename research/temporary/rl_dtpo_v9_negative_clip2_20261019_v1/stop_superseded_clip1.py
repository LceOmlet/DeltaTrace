"""Use the original process owner to retire the user-replaced clip1 queue."""
import hashlib
import json
import os
from pathlib import Path
import psutil
import runpy
import sys
import time


ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OLD=ROOT/'receipts/dtpo-v9-clip-u-postgroup-variance-20261019-v2'
NEW=ROOT/'receipts/dtpo-v9-negative-clip2-postgroup-variance-20261019-v3'
SEG=ROOT/'receipts/textcraft-eight-gpu-v9-clip-u-20261019-v2'


def main():
    result=NEW/'superseded-clip1-stopped.json'
    assert not result.exists(), 'Inspect the completed transition instead of repeating it'
    assert json.loads((NEW/'CPU-behavior-verification.json').read_bytes())['status']=='passed'
    previous=json.loads((SEG/'campaign-state.json').read_bytes())
    job,=previous['jobs'];controller=json.loads((OLD/'controller-job.json').read_bytes())
    assert job['version']=='textcraft-dtpo-v9-clip-u-8gpu-200-20261019-v2'
    finalizing='--finalize-observed-exit' in sys.argv
    if not finalizing:
        for row in [job,controller]:
            p=psutil.Process(row['pid'])
            assert p.create_time()==row['birth'] and p.status()!=psutil.STATUS_ZOMBIE
    verified=OLD/'textcraft-dtpo-v9-clip-u-8gpu-200-20261019-v2-checkpoint-5-verified.json'
    proof=json.loads(verified.read_bytes())
    assert proof['status']=='passed_native_shared_checkpoint_state_metadata' and proof['step']==5
    assert len(proof['roles'])==8 and (Path(proof['checkpoint'])/'data.pt').is_file()
    original=ROOT/'receipts/two-task-eight-gpu-sequential-20261017-v2/source'
    path=original/'run_campaign.py'
    assert hashlib.sha256(path.read_bytes()).hexdigest()=='f71ff4d85087c18d37bdb9fd2e40dbf7f36763e64b528e772612f24004e85441'
    sys.path.insert(0,str(original))
    native=runpy.run_path(str(path),run_name='original_user_replacement_process_owner')
    reader=native['original_log_reader']()
    if finalizing:
        record=json.loads((NEW/'superseded-clip1-stop-started.json').read_bytes())
        assert record['original_job']['pid']==job['pid'] and record['original_job']['birth']==job['birth']
        members=[]
        for p in psutil.process_iter():
            try:
                if os.getsid(p.pid) in [job['pid'],controller['pid']] and p.status()!=psutil.STATUS_ZOMBIE:
                    members.append(dict(pid=p.pid,birth=p.create_time()))
            except (ProcessLookupError,psutil.NoSuchProcess):pass
        assert not members, 'The original sessions still have live members; do not resubmit'
        record['training_stop']=dict(remaining_non_zombie=[],observed_unix=time.time(),
            observation='Original stop call completed; its short survivor observation elapsed before driver cleanup finished. Both original sessions are now absent.',
            additional_signals=0)
    else:
        snapshot=reader['collect'](str(ROOT),[job['output']]);current,=snapshot['jobs']
        record=dict(status='stopping_superseded_clip1_after_explicit_human_negative2_revision',started=time.time(),
            reason='Human changed per-event u negative clip1 to2; retain the old run and start the revised method from base',
            original_controller=controller,original_job=job,completed=reader['completed'](current),
            original_checkpoint_verification=str(verified),checkpoint=proof['checkpoint'],
            pre_stop_source_snapshot=snapshot,new_method_u_clip=[-2,1],new_training_submitted=False)
        native['write'](NEW/'superseded-clip1-stop-started.json',record)
        release=native['release_finished_session'];release.__globals__['AUDIT']=SEG
        # Retire the controller first; the driver's independent native session stays
        # identifiable until its own original process-owner call completes.
        record['controller_stop']=release(controller['pid'],controller['birth'])
        native['write'](NEW/'superseded-clip1-stop-started.json',record)
        record['training_stop']=release(job['pid'],job['birth'])
        native['write'](NEW/'superseded-clip1-stop-started.json',record)
    assert not record['controller_stop']['remaining_non_zombie']
    assert not record['training_stop']['remaining_non_zombie']
    record.update(status='stopped_by_user_negative2_method_replacement',finished=time.time(),
        retained_native_checkpoint_step=5,no_old_weights_reused=True)
    native['write'](result,record)
    job.update(status=record['status'],active=False,alive=False,completed_iterations=record['completed'],
        stopped_unix=record['finished'],user_method_replacement_receipt=str(result))
    previous.update(status=record['status'],user_method_replacement_receipt=str(result),finished_unix=record['finished'])
    native['write'](Path(job['run'])/'source.json',job)
    native['write'](SEG/'campaign-state.json',previous)
    for name in ['formal-training.json','active-training.json','active-source.json']:
        p=ROOT/name;state=json.loads(p.read_bytes())
        state['jobs']=[job if x.get('pid')==job['pid'] and x.get('birth')==job['birth'] else x for x in state['jobs']]
        native['write'](p,state)
    old_plan=json.loads((OLD/'six-combination-manifest.json').read_bytes())
    old_plan.update(status=record['status'],replacement_receipt=str(result))
    native['write'](OLD/'six-combination-manifest.json',old_plan)
    (NEW/'physical-GPU-after-user-method-replacement.txt').write_text(native['verify_GPU_empty']())
    print(json.dumps(dict(status=record['status'],completed=record['completed'],checkpoint=record['checkpoint'],
        receipt=str(result),remaining_non_zombie=[],physical_GPUs_empty=True)))


if __name__=='__main__':main()

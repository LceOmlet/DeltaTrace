"""Withdraw failed RPC-wide retention; retain negative2 and the native B4 boundary."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
PARENT=ROOT/'receipts/dtpo-v9-negative-clip2-postgroup-variance-20261019-v3'
AUDIT=ROOT/'receipts/dtpo-v9-negative-clip2-postgroup-variance-20261019-v4'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    state=json.loads((ROOT/'receipts/textcraft-eight-gpu-v9-negative-clip2-20261019-v3/campaign-state.json').read_bytes())
    assert state['status']=='needs_attention_no_failed_combination_skipped'
    old=state['jobs'][0];assert old['completed_iterations']==0
    assert (Path(old['run'])/'exit-code.txt').read_text().strip()=='1'
    assert not (Path(old['checkpoints'])/'latest_checkpointed_iteration.txt').exists()
    import psutil
    for pid,birth in [(state['controller_pid'],state['controller_birth']),(old['pid'],old['birth'])]:
        assert not psutil.pid_exists(pid) or psutil.Process(pid).create_time()!=birth or psutil.Process(pid).status()==psutil.STATUS_ZOMBIE
    AUDIT.mkdir(exist_ok=False);(AUDIT/'source').mkdir()
    for name in ['CPU-behavior-verification.json','native-verification-summary.json','allocation-owner-verification.json','superseded-clip1-stopped.json']:
        shutil.copyfile(PARENT/name,AUDIT/name)
    proof=json.loads((AUDIT/'native-verification-summary.json').read_bytes())
    proof.update(reused_from=str(PARENT/'native-verification-summary.json'),
        tested_cache_boundary='RPC-wide cache, two ranks; not accepted for eight-rank accumulation',
        selected_cache_boundary='Original B4 release, previously completed six native iterations with identical DT kernels',
        reuse_scope='Negative2 numerics, native transport, unchanged DT kernel and per-rank B4/32768; eight-rank whole-stage memory is verified separately by formal execution')
    (AUDIT/'native-verification-summary.json').write_text(json.dumps(proof,indent=2)+'\n')
    failure=dict(status='RPC_cache_retention_withdrawn_after_native_host_allocation_failure',collected=time.time(),
        failed_run=old['version'],completed_iterations=0,optimizer_updates=0,
        failure_site='mcHostAlloc / native_attention_capture.copy_capture_tensor CPU pinned empty_strided',
        previous_state=str(ROOT/'receipts/textcraft-eight-gpu-v9-negative-clip2-20261019-v3/campaign-state.json'),
        original_stderr=old['stderr'],original_stderr_SHA256=sha(old['stderr']),
        first_native_generation_measurement=str(PARENT/'bounded-native-generation-observation.json'),
        method_changed=False,event_u_clip=[-2,1],DT_NATIVE_HOST_CACHE_BATCH='1',
        DT_NATIVE_HOST_CACHE_REPLAY_LAYER='0',VERL_RELEASE_UNUSED_HOST_CACHE='1',
        queue_combinations_skipped=0,source_histories_preserved=True)
    (AUDIT/'failed-RPC-retention-execution.json').write_text(json.dumps(failure,indent=2)+'\n')
    for task in ('textcraft','appworld'):
        for method in ('dt','ppo','grpo'):
            p=PARENT/f'{task}-{method}-source-prepared.json';row=json.loads(p.read_bytes())
            assert all(sha(path)==value for path,value in row['source_bindings'].items())
            row['execution_parent_source_receipt']=str(p)
            if method=='dt':
                assert row['effective_environment']['DT_EXPONENTIAL_CREDIT_NEGATIVE_CLIP']=='2'
                row['effective_environment']['DT_NATIVE_HOST_CACHE_BATCH']='1'
                row['resource_version']='restored_native_B4_release_boundary_negative2_20261019_v4'
                row['native_resource_receipt']=str(AUDIT/'failed-RPC-retention-execution.json')
            (AUDIT/p.name).write_text(json.dumps(row,indent=2)+'\n')
    owner=PARENT/'source/prepare_campaign.py'
    assert sha(owner)=='9c4e4cde1a2e67580bdc304388b8d61406919f41d523008893855de04e385c34'
    code=owner.read_text().replace('20261019-v3','20261019-v4')
    exec(compile(code,str(owner),'exec'),dict(__name__='__main__',__file__=str(owner)))
    path=AUDIT/'six-combination-manifest.json';plan=json.loads(path.read_bytes())
    plan.update(method_source_commit='f26b3b37cb3b88e64978e7a472ae4244670a39f3',
        resource_rollback=str(AUDIT/'failed-RPC-retention-execution.json'))
    path.write_text(json.dumps(plan,indent=2)+'\n')
    for name in ['run_campaign.py','launch_campaign.py','pin_execution_commit.py','inspect_native_workers.py','report_native_utilization.py']:
        text=(PARENT/'source'/name).read_text().replace('20261019-v3','20261019-v4')
        (AUDIT/'source'/name).write_text(text)
    print(json.dumps(dict(status='negative2_fresh_queue_native_B4_cache_boundary_prepared',audit=str(AUDIT),all_six_fresh=True,method_changes=0)))

if __name__=='__main__':main()

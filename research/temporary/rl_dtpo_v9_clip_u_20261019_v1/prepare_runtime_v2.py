"""Configure the existing Torch host allocator after the observed MCR rejection."""
import hashlib
import json
import os
from pathlib import Path
import shutil

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OLD=ROOT/'receipts/dtpo-v9-clip-u-postgroup-variance-20261019-v1'
AUDIT=ROOT/'receipts/dtpo-v9-clip-u-postgroup-variance-20261019-v2'
SOURCE=Path(__file__).parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    AUDIT.mkdir(exist_ok=False);(AUDIT/'source').mkdir()
    for name in ('CPU-behavior-verification.json','native-verification-summary.json','prior-v5-observation.json'):
        shutil.copyfile(OLD/name,AUDIT/name)
    copies=[]
    for path in sorted((OLD/'native-capture-copy-owner-comparison').glob('rank*.json')):
        row=json.loads(path.read_bytes())
        assert row['status']=='passed_original_owner_B4_32768_BF16_copy'
        assert all(x['exact_values'] and x['original_strides'] for x in row['rows'])
        copies.append(dict(path=str(path),SHA256=sha(path),record=row))
    assert len(copies)==8
    failures=[]
    for path in sorted((OLD/'native-host-allocation-eight-process-v1').glob('True-rank*.json')):
        row=json.loads(path.read_bytes());assert row['status']=='native_allocation_failed'
        failures.append(dict(path=str(path),SHA256=sha(path),record=row))
    assert len(failures)==8
    allocation=dict(status='passed_original_capture_copy_configured_native_hostalloc',
        observed_failure=str(OLD/'native-first-iteration-failure.json'),failed_concurrent_allocations=failures,
        original_owner_copy_checks=copies,method_changes=0,training_workload_changes=0,
        DT_only_environment=dict(PYTORCH_CUDA_ALLOC_CONF='expandable_segments:False,pinned_use_cuda_host_register:False,pinned_num_register_threads:8',
            DT_NATIVE_HOST_CACHE_REPLAY_LAYER='0',DT_NATIVE_HOST_CACHE_BATCH='1'),
        scope='Original allocator/caching configuration; formal eight-rank training still requires natural verification',
        aborted_false_72GiB_check='Interrupted by exact PID/birth; not passed and not a capacity claim',
        reason='MCR mcHostRegisterImpl rejects size >= 0.8*sysinfo.freeram; reclaimable cache is ignored. Torch native cudaHostAlloc avoids that precheck.',
        guard_binary='/opt/maca/lib/libmcruntime.so',guard_binary_SHA256=sha('/opt/maca/lib/libmcruntime.so'))
    (AUDIT/'allocation-owner-verification.json').write_text(json.dumps(allocation,indent=2)+'\n')
    for task in ('textcraft','appworld'):
        for method in ('dt','ppo','grpo'):
            name=f'{task}-{method}-source-prepared.json';row=json.loads((OLD/name).read_bytes())
            row['resource_parent_source_receipt']=str(OLD/name)
            if method=='dt':
                row['effective_environment'].update(allocation['DT_only_environment'])
                row['allocation_owner_verification']=str(AUDIT/'allocation-owner-verification.json')
            (AUDIT/name).write_text(json.dumps(row,indent=2)+'\n')
    owner=SOURCE/'prepare_campaign.py'
    code=owner.read_text().replace('dtpo-v9-clip-u-postgroup-variance-20261019-v1','dtpo-v9-clip-u-postgroup-variance-20261019-v2')
    code=code.replace('-20261019-v1','-20261019-v2')
    anchor="                row['options']['+ray_init.runtime_env.env_vars.DT_CREDIT_MODE'] = candidate['credit_mode']"
    assert code.count(anchor)==1
    code=code.replace(anchor,anchor+"\n                for key in ('PYTORCH_CUDA_ALLOC_CONF','DT_NATIVE_HOST_CACHE_REPLAY_LAYER'):\n                    row['options']['+ray_init.runtime_env.env_vars.'+key] = candidate['effective_environment'][key]")
    exec(compile(code,str(owner),'exec'),dict(__name__='__main__',__file__=str(owner)))
    manifest=AUDIT/'six-combination-manifest.json';plan=json.loads(manifest.read_bytes())
    plan.update(execution_version='20261019-v9-clip-u-runtime-v2',method_source_commit='a88d479c199f9b3171a7d29e8a5840dc4759a6ea',
        allocation_owner_verification=str(AUDIT/'allocation-owner-verification.json'),failed_prior_execution=str(OLD/'native-first-iteration-failure.json'))
    for segment in plan['segments']:
        path=Path(segment['prepared']);prepared=json.loads(path.read_bytes())
        for row in prepared['jobs']:
            row['method_source_commit']='a88d479c199f9b3171a7d29e8a5840dc4759a6ea'
            if row['method']=='dt':
                row['resource_version']='native-hostalloc-replay-cache-20261019-v2'
                row['native_resource_receipt']=str(AUDIT/'allocation-owner-verification.json')
            individual=path.with_name(row['method']+'-prepared.json')
            individual.write_text(json.dumps(row,indent=2)+'\n')
        path.write_text(json.dumps(prepared,indent=2)+'\n')
    manifest.write_text(json.dumps(plan,indent=2)+'\n')
    for filename in ('run_campaign.py','launch_campaign.py','inspect_native_workers.py'):
        code=(SOURCE/filename).read_text().replace('postgroup-variance-20261019-v1','postgroup-variance-20261019-v2')
        code=code.replace('eight-gpu-v9-clip-u-20261019-v1','eight-gpu-v9-clip-u-20261019-v2')
        if filename=='run_campaign.py':
            anchor="        assert json.loads((AUDIT/'CPU-behavior-verification.json').read_bytes())['status']=='passed'"
            assert code.count(anchor)==1
            code=code.replace(anchor,"        assert json.loads((AUDIT/'allocation-owner-verification.json').read_bytes())['status']=='passed_original_capture_copy_configured_native_hostalloc'\n"+anchor)
        (AUDIT/'source'/filename).write_text(code)
    owner=ROOT/'receipts/dtpo-symmetric-probability-ratio-postgroup-variance-20261018-v8/source/pin_execution_commit.py'
    assert sha(owner)=='a9d2316b63c1930c259e6ce3fe6f640b2dba321e672cb06ef282ef958ad0c243'
    code=owner.read_text().replace("'receipts/dtpo-symmetric-probability-ratio-postgroup-variance-20261018-v8'",repr(str(AUDIT.relative_to(ROOT))))
    (AUDIT/'source/pin_execution_commit.py').write_text(code)
    print(json.dumps(dict(status='prepared_runtime_v2_not_started',manifest=str(manifest),method_source=plan['method_source_commit'],resource_only=True)))

if __name__=='__main__':main()

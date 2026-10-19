"""Select the existing RPC cache boundary after measured B4 release stalls.

No DT kernel, allocator, model, credit expression, or official trainer changes.
The consumed-root candidate is retained as measured but is not deployed.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT=ROOT/'receipts/dtpo-v9-negative-clip2-postgroup-variance-20261019-v3'
RESOURCE=ROOT/'receipts/dt-native-consumed-root-lifetime-20261019-v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    assert not (AUDIT/'controller-job.json').exists()
    assert not (AUDIT/'native-behavior-job.json').exists()
    comparison=json.loads((RESOURCE/'native-comparison-job.json').read_bytes())
    assert comparison['status']=='completed' and comparison['native_DT_vectors_equal']==[True]
    before=AUDIT/'before-native-RPC-cache-boundary';before.mkdir(exist_ok=False)
    changes=[]
    for task in ('textcraft','appworld'):
        source_path=AUDIT/f'{task}-dt-source-prepared.json'
        segment=ROOT/'receipts'/f'{task}-eight-gpu-v9-negative-clip2-20261019-v3'
        prepared_path=segment/'campaign-prepared.json'
        source=json.loads(source_path.read_bytes());prepared=json.loads(prepared_path.read_bytes())
        row=next(x for x in prepared['jobs'] if x['method']=='dt')
        assert all(sha(p)==value for p,value in row['source_bindings'].items())
        assert row['effective_environment']['DT_NATIVE_HOST_CACHE_BATCH']=='1'
        assert row['effective_environment']['VERL_RELEASE_UNUSED_HOST_CACHE']=='1'
        for p in (source_path,prepared_path,segment/'dt-prepared.json',segment/'dt-effective-config.yaml'):
            (before/(task+'-'+p.name)).write_bytes(p.read_bytes())
        flag='DT_NATIVE_HOST_CACHE_BATCH'
        for item in (row,source):
            item['effective_environment'][flag]='0'
            item['native_resource_receipt']=str(AUDIT/'native-RPC-cache-boundary-prepared.json')
        key='+ray_init.runtime_env.env_vars.'+flag
        assert str(row['options'][key])=='1'
        previous_arg,=[arg for arg in row['argv'] if arg.partition('=')[0]==key]
        row['options'][key]='0'
        row['argv'][row['argv'].index(previous_arg)]=key+'='+json.dumps('0')
        row['resource_version']='native-cache-reuse-until-DT-RPC-20261019-v3'
        env=dict(os.environ,**row['effective_environment']);env['CUDA_VISIBLE_DEVICES']=''
        result=subprocess.run(row['argv']+['--cfg','job','--resolve'],cwd=row['verl_root'],env=env,
            stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=90)
        (segment/'dt-native-RPC-cache-config.stderr').write_bytes(result.stderr)
        assert result.returncode==0,result.stderr.decode(errors='replace')[-4000:]
        (segment/'dt-effective-config.yaml').write_bytes(result.stdout)
        source_path.write_text(json.dumps(source,indent=2)+'\n')
        (segment/'dt-prepared.json').write_text(json.dumps(row,indent=2)+'\n')
        prepared_path.write_text(json.dumps(prepared,indent=2)+'\n')
        changes.append(dict(task=task,run=row['version'],DT_ROOT=row['dt_root'],
            DT_NATIVE_HOST_CACHE_BATCH='0',VERL_RELEASE_UNUSED_HOST_CACHE='1',
            resolved_configuration=str(segment/'dt-effective-config.yaml'),
            resolved_configuration_SHA256=sha(segment/'dt-effective-config.yaml')))
    (AUDIT/'prepared-launch.json').write_bytes((ROOT/'receipts/textcraft-eight-gpu-v9-negative-clip2-20261019-v3/dt-prepared.json').read_bytes())
    record=dict(status='prepared_existing_native_RPC_cache_boundary_pending_native_validation',
        collected=time.time(),changes=changes,DT_kernel_changes=0,credit_changes=0,
        consumed_root_candidate_selected=False,
        consumed_root_comparison=str(RESOURCE/'native-comparison-job.json'),
        comparison_scope='Same B4/32768 vectors exact; lifetime candidate did not improve time, remains outside deployment',
        reason='Measured native host-emptyCache occupied a zero-GPU interval; reuse the existing cache across B4 calls and keep the original RPC final release',
        full_eight_rank_efficiency_verified=False,full_eight_rank_capacity_verified=False)
    (AUDIT/'native-RPC-cache-boundary-prepared.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record))

if __name__=='__main__':main()

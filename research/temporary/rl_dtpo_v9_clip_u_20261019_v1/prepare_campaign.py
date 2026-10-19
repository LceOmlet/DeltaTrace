"""Compose the accepted six-run native configuration owner for v9 identities."""
import hashlib
import json
import os
from pathlib import Path

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT=ROOT/'receipts/dtpo-v9-clip-u-postgroup-variance-20261019-v1'
RESOURCE=ROOT/'receipts/six-run-fresh-validated-efficiency-20261018-v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def version_for(task,method):
    label='dtpo-v9-clip-u' if method=='dt' else method
    return f'{task.lower()}-{label}-8gpu-200-20261019-v1'

def main():
    assert json.loads((AUDIT/'CPU-behavior-verification.json').read_bytes())['status']=='passed'
    for name in ('native-resource-owner-prepared.json','native-resource-owner-deployed.json','selected-native-resources.json'):
        (AUDIT/name).write_bytes((RESOURCE/name).read_bytes())
    source=ROOT/'receipts/six-run-fresh-after-efficiency-20261017-v1/source/prepare_fresh_queue.py'
    assert sha(source)=='3cf1500344a22215986cb69ad2a951be86daf77b22e9fb2ef432116d78b33355'
    code=source.read_text()
    code=code.replace("'receipts/six-run-fresh-after-efficiency-20261017-v1'",repr(str(AUDIT.relative_to(ROOT))))
    code=code.replace("'-eight-gpu-sequential-20261017-v1'","'-eight-gpu-symmetric-probability-ratio-postgroup-variance-20261018-v8'")
    code=code.replace("'-eight-gpu-sequential-20261017-v2'","'-eight-gpu-v9-clip-u-20261019-v1'")
    code=code.replace("old_row['version'].removesuffix('-v1') + '-v2'","version_for(task,method)")
    code=code.replace('expandable_segments:False,pinned_use_cuda_host_register:True',
        'expandable_segments:False,pinned_use_cuda_host_register:True,pinned_num_register_threads:8')
    anchor="            assert row['options']['trainer.resume_mode'] == 'disable'\n"
    assert code.count(anchor)==1
    code=code.replace(anchor,"""            candidate = json.loads((AUDIT / (task.lower() + '-' + method + '-source-prepared.json')).read_bytes())
            for field in ('entry','verl_root','dt_root','source_bindings'):
                row[field] = candidate[field]
            row['effective_environment'] = json.loads(json.dumps(candidate['effective_environment']).replace(old_row['version'],version))
            row['actor_gradient_norm'] = candidate['actor_gradient_norm']
            row['local_shard_key'] = candidate['local_shard_key']
            if method == 'dt':
                row['options']['+ray_init.runtime_env.env_vars.DT_CREDIT_MODE'] = candidate['credit_mode']
                row['method_version'] = 'DTPO-v5-exponential-u-clipped-minus1-plus1-v9'
                row['credit_variance_normalization'] = candidate['credit_variance_normalization']
            assert row['options']['trainer.resume_mode'] == 'disable'
""")
    exec(compile(code,str(source),'exec'),dict(__name__='__main__',__file__=str(source),version_for=version_for))
    path=AUDIT/'six-combination-manifest.json';plan=json.loads(path.read_bytes())
    plan.update(selected_native_resources=str(AUDIT/'selected-native-resources.json'),
        execution_version='20261019-v9-clip-u',DT_method_version='v5-exponential-u-clipped-minus1-plus1-v9',
        credit_variance_normalization='post_group_sum_no_mean_subtraction',extra_model_samples=0,
        actor_gradient_norm='preclip_global_native_DTensor_full_tensor',source_commit=os.environ['RL_CAMPAIGN_COMMIT'])
    path.write_text(json.dumps(plan,indent=2)+'\n')

if __name__=='__main__':main()

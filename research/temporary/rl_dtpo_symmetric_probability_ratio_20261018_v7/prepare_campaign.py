"""Compose the pinned six-run preparation with isolated v7 DT source identities."""
import hashlib
import json
import os
from pathlib import Path

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT=ROOT/'receipts/dtpo-symmetric-probability-ratio-20261018-v7'
PRIOR=ROOT/'receipts/six-run-fresh-validated-efficiency-20261018-v1'

def main():
    for name in ('native-resource-owner-prepared.json','native-resource-owner-deployed.json','selected-native-resources.json'):
        (AUDIT/name).write_bytes((PRIOR/name).read_bytes())
    source=ROOT/'receipts/six-run-fresh-after-efficiency-20261017-v1/source/prepare_fresh_queue.py'
    assert hashlib.sha256(source.read_bytes()).hexdigest()=='3cf1500344a22215986cb69ad2a951be86daf77b22e9fb2ef432116d78b33355'
    code=source.read_text()
    code=code.replace("'receipts/six-run-fresh-after-efficiency-20261017-v1'","'receipts/dtpo-symmetric-probability-ratio-20261018-v7'")
    code=code.replace("'-eight-gpu-sequential-20261017-v1'","'-eight-gpu-sequential-20261018-v5'")
    code=code.replace("'-eight-gpu-sequential-20261017-v2'","'-eight-gpu-symmetric-probability-ratio-20261018-v7'")
    code=code.replace("old_row['version'].removesuffix('-v1') + '-v2'",
        "old_row['version'].replace('dtpo-v6-', 'dtpo-symmetric-probability-ratio-v7-').removesuffix('-20261018-v5') + '-20261018-v7'")
    code=code.replace('expandable_segments:False,pinned_use_cuda_host_register:True',
        'expandable_segments:False,pinned_use_cuda_host_register:True,pinned_num_register_threads:8')
    anchor="            assert row['options']['trainer.resume_mode'] == 'disable'\n"
    assert code.count(anchor)==1
    code=code.replace(anchor,"""            if method == 'dt':
                candidate = json.loads((AUDIT / (task.lower() + '-source-prepared.json')).read_bytes())
                for field in ('entry','verl_root','dt_root','source_bindings'):
                    row[field] = candidate[field]
                row['effective_environment'] = json.loads(json.dumps(candidate['effective_environment']).replace(old_row['version'],version))
                row['options']['+ray_init.runtime_env.env_vars.DT_CREDIT_MODE'] = candidate['credit_mode']
                row['method_version'] = 'DTPO-symmetric-probability-ratio-v7'
                row['credit_variance_normalization'] = False
            assert row['options']['trainer.resume_mode'] == 'disable'
""")
    exec(compile(code,str(source),'exec'),dict(__name__='__main__',__file__=str(source)))
    path=AUDIT/'six-combination-manifest.json';plan=json.loads(path.read_bytes())
    plan.update(selected_native_resources=str(AUDIT/'selected-native-resources.json'),
        execution_version='20261018-symmetric-probability-ratio-v7',DT_method_version='symmetric-probability-ratio-v7',
        supersedes=str(PRIOR/'six-combination-manifest.json'),
        extra_model_samples=0,credit_variance_normalization=False,
        source_commit=os.environ['RL_CAMPAIGN_COMMIT'])
    path.write_text(json.dumps(plan,indent=2)+'\n')

if __name__=='__main__':main()

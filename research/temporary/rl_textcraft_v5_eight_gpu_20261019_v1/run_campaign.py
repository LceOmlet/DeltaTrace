"""Compose the pinned native process owner for the one requested v5 run."""
import hashlib
import json
from pathlib import Path
from prepare_campaign import ROOT,AUDIT,VERSION,sha

def ready():
    row=json.loads((AUDIT/'dt-prepared.json').read_bytes())
    assert json.loads((AUDIT/'CPU-behavior-verification.json').read_bytes())['status']=='passed'
    for path,digest in row['source_bindings'].items():
        assert sha(path)==digest,path

if __name__=='__main__':
    owner=ROOT/'receipts/two-task-eight-gpu-sequential-20261017-v2/source/run_campaign.py'
    assert sha(owner)=='f71ff4d85087c18d37bdb9fd2e40dbf7f36763e64b528e772612f24004e85441'
    code=owner.read_text().replace("'three_native_CPU_configs_resolved_and_common_workload_equal'",
        "'native_v5_eight_gpu_configuration_resolved'").replace(
        "'all_three_native_200_iteration_runs_completed'","'requested_v5_native_200_iteration_run_completed'")
    namespace={'__name__':'pinned_native_v5_process_owner','__file__':str(owner)}
    exec(compile(code,str(owner),'exec'),namespace)
    namespace['main'](audit=AUDIT,version=VERSION,readiness=ready)

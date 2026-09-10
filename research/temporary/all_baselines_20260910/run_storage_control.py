"""Verify lossless CPU saved-tensor storage, then continue the full controller."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);a=p.parse_args();b=a.base
    command=[sys.executable,'-u',str(HERE/'evaluate_baselines.py'),'--environment',str(b/'environment.json'),
        '--preflight',str(b/'all_baselines_preflight.json'),'--output',str(b/'all_baselines_cpu_control_v2'),
        '--datasets','vt_h2_c3','--indices','0','--methods','AttnLRP']
    with (b/'all_baselines_storage_control_v2.log').open('x') as log:
        result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:raise SystemExit(result.returncode)
    old=b/'all_baselines_v2/AttnLRP/vt_h2_c3/000';new=b/'all_baselines_cpu_control_v2/AttnLRP/vt_h2_c3/000'
    with np.load(old/'vectors.npz') as left,np.load(new/'vectors.npz') as right:
        assert set(left.files)==set(right.files)
        for name in left.files:assert np.array_equal(left[name],right[name],equal_nan=True),name
        keys=left.files
    receipt=dict(status='bitwise_equal_all_vectors',keys=keys,time=time.time(),
        old_results_sha256=sha(old/'results.json'),new_results_sha256=sha(new/'results.json'),
        old_vectors_sha256=sha(old/'vectors.npz'),new_vectors_sha256=sha(new/'vectors.npz'))
    (b/'all_baselines_storage_control.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)
    command=[sys.executable,'-u',str(HERE/'run_controller.py'),'--environment',str(b/'environment.json'),
        '--preflight',str(b/'all_baselines_preflight.json'),'--output',str(b/'all_baselines_v2'),
        '--logs',str(b/'all_baselines_logs_v3'),'--methods','Perturbation','CLP','IFR','AttnLRP','REAGENT',
        '--mlm',str(b/'longformer-base-4096')]
    raise SystemExit(subprocess.call(command))

if __name__=='__main__':main()

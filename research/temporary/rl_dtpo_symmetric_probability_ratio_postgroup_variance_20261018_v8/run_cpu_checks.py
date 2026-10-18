"""Use the sourced runtime merged with the actual candidate environment."""
import json
import os
from pathlib import Path
import subprocess
import sys

R=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
A=R/'receipts/dtpo-symmetric-probability-ratio-postgroup-variance-20261018-v8'
row=json.loads((A/'textcraft-dt-source-prepared.json').read_bytes())
env=dict(os.environ,**row['effective_environment'])
env['CUDA_VISIBLE_DEVICES']=''
env['DT_V8_NORM_AUDIT']=str(A)
python=env['VENV_PYTHON']
tests=[str(A/'source/test_symmetric_credit.py'),str(A/'source/test_counterfactual.py'),
    str(R/'receipts/dtpo-exponential-credit-variance-20261017-v6/source/test_v6_behavior.py')]
with (A/'CPU-behavior-tests.log').open('wb') as log:
    result=subprocess.run([python,'-m','pytest','-q',*tests,'--junitxml='+str(A/'CPU-behavior-tests.xml')],
        env=env,cwd=row['entry'],stdout=log,stderr=subprocess.STDOUT)
if result.returncode:
    print((A/'CPU-behavior-tests.log').read_text()[-16000:]);result.check_returncode()
print((A/'CPU-behavior-tests.log').read_text()[-1800:],flush=True)
for world in (2,8):
    with (A/f'official-norm-{world}rank.log').open('wb') as log:
        result=subprocess.run([python,'-m','torch.distributed.run','--standalone','--nnodes=1',
            '--nproc-per-node='+str(world),str(A/'source/verify_native_norm_logging.py')],
            env=env,cwd=row['entry'],stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:
        print((A/f'official-norm-{world}rank.log').read_text()[-10000:]);result.check_returncode()
    print((A/f'official-norm-{world}rank.json').read_text(),flush=True)
subprocess.run([python,str(A/'source/record_cpu_verification.py')],env=env,check=True)

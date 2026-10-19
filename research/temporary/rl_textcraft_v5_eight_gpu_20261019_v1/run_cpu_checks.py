"""Run existing v5 interface tests on the selected current provisioned sources."""
import json
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
from prepare_campaign import ROOT,AUDIT,sha

row=json.loads((AUDIT/'dt-prepared.json').read_bytes())
env=dict(os.environ,**row['effective_environment'])
env['CUDA_VISIBLE_DEVICES']=''
env['PYTHONPATH']=str(AUDIT/'source')+':'+env['PYTHONPATH']
tests=[str(AUDIT/'source/test_behavior_v5.py'),str(AUDIT/'source/test_multi_target_v5.py'),
    str(ROOT/'receipts/textcraft-weighted-reward-targets-20261015-v4/source/test_direct_target_transport.py')]
argv=[env['VENV_PYTHON'],'-m','pytest','-q',*tests,'-k','not official_GRPO_PPO_and_fast_owner_bindings',
    '--junitxml='+str(AUDIT/'CPU-behavior-tests.xml'),'-p','no:cacheprovider']
with (AUDIT/'CPU-behavior-tests.log').open('xb') as log:
    result=subprocess.run(argv,cwd=row['entry'],env=env,stdout=log,stderr=subprocess.STDOUT)
print((AUDIT/'CPU-behavior-tests.log').read_text()[-2500:])
result.check_returncode()
cases=ET.parse(AUDIT/'CPU-behavior-tests.xml').findall('.//testcase')
for path,digest in row['source_bindings'].items():assert sha(path)==digest,path
record=dict(status='passed',tests=len(cases),returncode=result.returncode,mode=env['DT_CREDIT_MODE'],
    algorithm_code_changes=0,model_calls=0,optimizer_updates=0,CUDA_VISIBLE_DEVICES='',
    inherited_actual_sources=row['source_bindings'],
    source_parent_commit=row['execution_parent_commit'],
    inherited_eight_rank_norm_verification=str(ROOT/'receipts/dtpo-symmetric-probability-ratio-postgroup-variance-20261018-v8/CPU-behavior-verification.json'),
    legacy_binding_test_excluded='The historical test hardcodes pre-global-logging actor and pre-host-lifetime runner SHAs. Current accepted binding hashes are checked directly, without changing numerical tests.',
    junit=str(AUDIT/'CPU-behavior-tests.xml'),log=str(AUDIT/'CPU-behavior-tests.log'))
(AUDIT/'CPU-behavior-verification.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({k:record[k] for k in ['status','tests','mode','model_calls','optimizer_updates']}))

"""Check original v5 regressions and the clipped v9 seam on actual candidates."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET
from prepare_sources import ROOT,AUDIT,MODE,sha

def main():
    row=json.loads((AUDIT/'textcraft-dt-source-prepared.json').read_bytes())
    legacy=AUDIT/'legacy-v5-tests';legacy.mkdir(exist_ok=True)
    source=ROOT/'receipts/textcraft-reward-target-exponential-20261016-v5/source'
    for name in ('test_behavior_v5.py','test_multi_target_v5.py'):
        text=(source/name).read_text().replace(
            "AUDIT=ROOT/'receipts/textcraft-reward-target-exponential-20261016-v5'",
            "AUDIT=ROOT/'receipts/dtpo-v9-clip-u-postgroup-variance-20261019-v1'")
        (legacy/name).write_text(text)
    (AUDIT/'baseline').mkdir(exist_ok=True)
    shutil.copyfile(source.parent/'baseline/counterfactual.py',AUDIT/'baseline/counterfactual.py')
    env=dict(os.environ,**row['effective_environment']);env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    env['PYTHONPATH']=row['entry']+':'+str(legacy)+':'+env['PYTHONPATH']
    records=[]
    for label,mode,tests,extra in [
        ('v5-regression','reward_target_exponential_v5',[legacy/'test_behavior_v5.py',legacy/'test_multi_target_v5.py',
            ROOT/'receipts/textcraft-weighted-reward-targets-20261015-v4/source/test_direct_target_transport.py'],
            ['-k','not official_GRPO_PPO_and_fast_owner_bindings']),
        ('v9-clip',MODE,[AUDIT/'source/test_credit_v9.py'],[])]:
        env['DT_CREDIT_MODE']=mode
        junit=AUDIT/(label+'-CPU-tests.xml');log=AUDIT/(label+'-CPU-tests.log')
        argv=[env['VENV_PYTHON'],'-m','pytest','-q','--import-mode=importlib',*[str(p) for p in tests],*extra,
            '--junitxml='+str(junit),'-p','no:cacheprovider']
        with log.open('wb') as stream:result=subprocess.run(argv,cwd=row['entry'],env=env,stdout=stream,stderr=subprocess.STDOUT)
        print(log.read_text()[-5000:],flush=True);result.check_returncode()
        cases=ET.parse(junit).findall('.//testcase')
        records.append(dict(label=label,mode=mode,tests=len(cases),junit=str(junit),log=str(log),returncode=result.returncode))
    for task in ('textcraft','appworld'):
        for method in ('dt','ppo','grpo'):
            prepared=json.loads((AUDIT/f'{task}-{method}-source-prepared.json').read_bytes())
            for path,expected in prepared['source_bindings'].items():assert sha(path)==expected,path
    record=dict(status='passed',groups=records,tests=sum(r['tests'] for r in records),mode=MODE,
        clip_range=[-1,1],clip_before_event_coefficients=True,post_group_native_masked_var=True,
        preserved_official_actor_core='All six source bindings verified; actor/core/optimizer computations unchanged',
        inherited_GPU_capacity='Existing native B4/rank exact32768 receipt; new CPU extent does not claim a new GPU/model acceptance',
        legacy_binding_test_excluded='Historical test hardcodes pre-global-logging actor/runner SHA; all actual preserved source bindings checked above',
        model_calls=0,DT_calls=0,optimizer_updates=0,CUDA_VISIBLE_DEVICES='')
    (AUDIT/'CPU-behavior-verification.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)

if __name__=='__main__':main()

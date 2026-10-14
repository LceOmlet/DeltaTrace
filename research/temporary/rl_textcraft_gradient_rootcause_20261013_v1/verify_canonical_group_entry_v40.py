"""Check canonical entry regeneration against frozen deployed bytes, then run owner-interface CPU tests."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
spec=importlib.util.spec_from_file_location('strict_entry',HERE.parent/'rl_upstream_alignment_20260929/stage_environment_entry.py')
entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
remote=entry.ROOT+'/receipts/textcraft-gradient-rootcause-20261013-v1'
original=HERE/'v40-original-ray_trainer.py'
subprocess.run(entry.SCP+[entry.SSH[-1]+':'+remote+'/v40-original-ray_trainer.py',str(original)],check=True)
assert hashlib.sha256(original.read_bytes()).hexdigest()=='7366557b482e604d66f47bdc4841ea147ffaac7c80538fa000544bb4e92eb619'
spec=importlib.util.spec_from_file_location('canonical_patch',REPO/'experiments/rl/patch_verl_agent2.py')
patch=importlib.util.module_from_spec(spec);spec.loader.exec_module(patch)
text=patch.patch_dt_advantage_preprocessing(original.read_text(encoding='utf-8'))
_,text=patch.patch_dt_reward_boundary('',text)
digest=hashlib.sha256(text.encode()).hexdigest()
assert digest=='63b0d623380bab3173eaf2c7148bddf21b557827625bd5eff25c5b4faa05daa4',digest
assert hashlib.sha256((REPO/'experiments/rl/dt_training_batch.py').read_bytes()).hexdigest()=='a4e1f70b42082972324199b328975fb87ec6600ab9a572d5ca2b2040bebc54f7'
assert patch.patch_dt_advantage_preprocessing(text)==text
assert patch.patch_dt_reward_boundary('',text)[1]==text
testdir=remote+'/canonical-owner-interface-tests-v40'
subprocess.run(entry.SSH+['mkdir','-p',testdir],check=True)
files=['test_direct_target_transport.py','test_rollout_credit.py']
for name in files:
    subprocess.run(entry.SCP+[str(REPO/'experiments/rl'/name),entry.SSH[-1]+':'+testdir+'/'+name],check=True)
script=rf'''source {shlex.quote(entry.ENTRY+'/metax-entry.env.sh')}
"$VENV_PYTHON" - <<'PY'
import json,os,subprocess
from pathlib import Path
root=Path({entry.ROOT!r});testdir=Path({testdir!r})
launch=json.loads((root/'runs/textcraft-dt-grpo-multiplier-no-whiten-20261014-v1/launch.json').read_bytes())
env=dict(os.environ);env.update(launch['effective_environment']);env['CUDA_VISIBLE_DEVICES']=''
r=subprocess.run([env['VENV_PYTHON'],'-m','pytest','-q','test_direct_target_transport.py',
                  'test_rollout_credit.py::test_collector_to_trainer_keeps_q_v_and_upstream_actor_gradient'],
                 cwd=testdir,env=env,capture_output=True,text=True,timeout=180)
(testdir/'pytest.stdout.txt').write_text(r.stdout)
(testdir/'pytest.stderr.txt').write_text(r.stderr)
print(json.dumps(dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)))
PY
'''
r=subprocess.run(entry.SSH+['bash','-s'],input=script.encode(),capture_output=True,timeout=220)
assert r.returncode==0,r.stderr.decode(errors='replace')
tests=json.loads(r.stdout)
report=dict(canonical_trainer_exact_deployed_SHA256=digest,canonical_DT_batch_exact_deployed_SHA256='a4e1f70b42082972324199b328975fb87ec6600ab9a572d5ca2b2040bebc54f7',
            patch_regeneration_idempotent=True,CPU_owner_interface_tests=tests,
            test_sources={name:hashlib.sha256((REPO/'experiments/rl'/name).read_bytes()).hexdigest() for name in files},
            training_workers_mutated=False,new_model_calls=0)
dest=HERE/'canonical-entry-regression-v40.json'
dest.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report))
assert tests['returncode']==0

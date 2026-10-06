"""Prepare an isolated trainer-only owner whitening candidate; no model run."""
import hashlib
import json
from pathlib import Path
import subprocess

from stage_environment_entry import AUDIT, REPO, ROOT, SSH, SCP

OUT = ROOT + '/receipts/textcraft-official-whitening-20261006-v1'
VERL = ROOT + '/candidates/textcraft-official-whitening-20261006-v1/verl'
BASELINE = ROOT + '/candidates/textcraft-truncated-terminal-resume-20261005-v2/verl'
LOCAL = AUDIT / 'textcraft-degradation-20261005/official-whitening-20261006/v1'


if __name__ == '__main__':
    LOCAL.mkdir(parents=True, exist_ok=True)
    files = {
        'source/experiments/rl/patch_verl_agent2.py': REPO / 'experiments/rl/patch_verl_agent2.py',
        'source/tests/test_dt_official_whitening.py': REPO / 'tests/test_dt_official_whitening.py',
        'source/owner-pristine-trainer.py': AUDIT / 'recipe-sources/verl-agent-20bd331/verl/trainer/ppo/ray_trainer.py',
    }
    hashes = {name: hashlib.sha256(p.read_bytes()).hexdigest() for name, p in files.items()}
    subprocess.run(SSH + ['mkdir', '-p', OUT + '/source/experiments/rl', OUT + '/source/tests'], check=True)
    for name, path in files.items():
        subprocess.run(SCP + [str(path), f'{SSH[-1]}:{OUT}/{name}'], check=True)
    script = r'''/opt/conda/bin/python - <<'PY'
import ast, hashlib, importlib.util, json, os, pathlib, shutil, subprocess, sys, time
out=pathlib.Path('__OUT__'); baseline=pathlib.Path('__BASELINE__'); candidate=pathlib.Path('__VERL__')
hashes=__HASHES__
for name, expected in hashes.items():
 p=out/name; assert hashlib.sha256(p.read_bytes()).hexdigest()==expected; ast.parse(p.read_bytes())
trainer_rel='verl/trainer/ppo/ray_trainer.py'
assert hashlib.sha256((baseline/trainer_rel).read_bytes()).hexdigest()=='8816ea4e5f9a95a1dfe1067349eee9101da8bf4ec916bcc28a5b03288976f0df'
spec=importlib.util.spec_from_file_location('patch_verl_agent2',out/'source/experiments/rl/patch_verl_agent2.py')
patch=importlib.util.module_from_spec(spec); spec.loader.exec_module(patch)
if not candidate.exists():
 shutil.copytree(baseline,candidate,ignore=shutil.ignore_patterns('.git','__pycache__','*.pyc'))
 p=candidate/trainer_rel; text=patch.patch_dt_advantage_preprocessing(p.read_text())
 p.write_text(text); ast.parse(text)
assert patch.patch_dt_advantage_preprocessing((candidate/trainer_rel).read_text())==(candidate/trainer_rel).read_text()
checks={}; changed=[]
for p in baseline.rglob('*'):
 if not p.is_file() or '.git' in p.relative_to(baseline).parts or '__pycache__' in p.parts or p.suffix=='.pyc':continue
 rel=p.relative_to(baseline); q=candidate/rel
 a=hashlib.sha256(p.read_bytes()).hexdigest();b=hashlib.sha256(q.read_bytes()).hexdigest()
 checks[str(rel)]={'baseline':a,'candidate':b}
 if a!=b:changed.append(str(rel))
assert changed==[trainer_rel],changed
environment_path=out/'recovered-runtime-environment.json'; recovered=json.loads(environment_path.read_bytes())
# This exported constant is read from the already recorded, reused env source.
recovered['environment']['FLA_BOUNDED_NORM_TUNING']='1'
environment_path.write_text(json.dumps(recovered,indent=2)+'\n')
env=dict(os.environ);env.update(recovered['environment'])
env.update(CUDA_VISIBLE_DEVICES='',VERL_ROOT=str(candidate),DT_VERL_PRISTINE_TRAINER_SOURCE=str(out/'source/owner-pristine-trainer.py'),
 DT_VERL_TRAINER_SOURCE=str(baseline/trainer_rel),DT_VERL_TORCH_FUNCTIONAL_SOURCE=str(baseline/'verl/utils/torch_functional.py'))
with (out/'focused-owner-tests.stdout.txt').open('wb') as log:
 test=subprocess.run([env['VENV_PYTHON'],'-m','unittest','discover','-s',str(out/'source/tests'),'-p','test_dt_official_whitening.py','-v'],cwd=out/'source',env=env,stdout=log,stderr=subprocess.STDOUT)
assert test.returncode==0,(out/'focused-owner-tests.stdout.txt').read_text()
x=dict(prepared_unix=time.time(),status='prepared_only_not_formal_deployment',upstream_commit='20bd331',
 baseline=str(baseline),candidate=str(candidate),changed_files=changed,all_file_sha256=checks,
 trainer_sha256=checks[trainer_rel]['candidate'],source_sha256=hashes,
 recovered_environment={'path':str(environment_path),'sha256':hashlib.sha256(environment_path.read_bytes()).hexdigest()},
 focused_owner_tests={'path':str(out/'focused-owner-tests.stdout.txt'),'sha256':hashlib.sha256((out/'focused-owner-tests.stdout.txt').read_bytes()).hexdigest()},
 scope='Only DT trainer actor-advantages preprocessing and official helper import change. Raw Q/V/A, actor/core/worker/DT and configuration unchanged. No model/rollout/backward/update.')
(out/'candidate-source.json').write_text(json.dumps(x,indent=2)+'\n')
print(json.dumps({k:x[k] for k in ('status','candidate','changed_files','trainer_sha256','recovered_environment','focused_owner_tests')}))
PY
'''
    script = (script.replace('__OUT__', OUT).replace('__BASELINE__', BASELINE)
              .replace('__VERL__', VERL).replace('__HASHES__', repr(hashes)))
    (LOCAL / 'prepare-candidate.sh').write_text(script, encoding='utf-8')
    result = subprocess.run(SSH + ['bash', '-s'], input=script.encode(), capture_output=True)
    (LOCAL / 'prepare-candidate.stdout.txt').write_bytes(result.stdout + result.stderr)
    print(result.stdout.decode(errors='replace'))
    print(result.stderr.decode(errors='replace'))
    result.check_returncode()
    subprocess.run(SCP + [f'{SSH[-1]}:{OUT}/{name}' for name in
        ('candidate-source.json', 'focused-owner-tests.stdout.txt', 'recovered-runtime-environment.json')]
        + [str(LOCAL)], check=True)

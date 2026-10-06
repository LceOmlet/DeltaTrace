/opt/conda/bin/python - <<'PY'
import ast, hashlib, importlib.util, json, os, pathlib, shutil, subprocess, sys, time
out=pathlib.Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/textcraft-official-whitening-20261006-v1'); baseline=pathlib.Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/candidates/textcraft-truncated-terminal-resume-20261005-v2/verl'); candidate=pathlib.Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/candidates/textcraft-official-whitening-20261006-v1/verl')
hashes={'source/experiments/rl/patch_verl_agent2.py': '84155a172b4c41a4deea7f1a2dff6a6b2044f99035331841393c8ca210cf61b9', 'source/tests/test_dt_official_whitening.py': 'f53a09962229704ba227d7d122ab57387c4896d129e3f6cbb0ae3e92d75dbfea', 'source/owner-pristine-trainer.py': '69dab6ef8a0521704468c8158c6b19c782d5455da49a3a1582d4bcd6a2a51e30'}
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

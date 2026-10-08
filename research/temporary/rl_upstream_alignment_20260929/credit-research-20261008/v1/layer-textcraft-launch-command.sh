source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
CUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<'PY'
ROOT='/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922'
OUT='/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/credit-layer-development-textcraft-20261008-v1'
TASK='textcraft'
HASHES={'inspect_layer_collection.py': '8e9b1c8bdab5b73c86dfe908be73eced86cf060824eac01983e0e600fe18296d', 'layer-collection-inputs.json': 'e835680d4b527829fae2b2c116c58b198fb3eda7bf12da137e11871e411f87a5', 'inspect_action_curve.py': '7277fade4e9b1cb49f825e4cb2fc54453c9ff3ce4859b20350aa2bd67ffaf3a6', 'inspect_extreme_endpoint.py': '8a72887a32bd11533486ddee6dc0d36b4980bfb77ed94eadc7a9a13f031b36d5'}
SOURCE_SHA='2796233e2683f1939896c74b2b578c242dbd7a7f235b9ef61cbedd398f61be52'
COMMIT='2951f70e7a853898f97e5a726cba66f13404df1b'
import ast,hashlib,json,os,psutil,re,subprocess,time
from pathlib import Path
root=Path(ROOT);out=Path(OUT)
assert not (out/'launch.json').exists(),'Do not duplicate a collection launch'
for name,h in HASHES.items():
 assert hashlib.sha256((out/name).read_bytes()).hexdigest()==h
 if name.endswith('.py'):ast.parse((out/name).read_text())
source_path=root/'runs/direct-target-prefix-runtime-20261007-v1'/TASK/(TASK+'-dt')/'source.json'
assert hashlib.sha256(source_path.read_bytes()).hexdigest()==SOURCE_SHA
source=json.loads(source_path.read_bytes())
runner=source['actual_CPU_imports']['qwen35_dense_finite_runner']
assert hashlib.sha256(Path(runner['path']).read_bytes()).hexdigest()==runner['sha256']
assert psutil.Process(2833207).create_time()==1791370325.16
capture=root/'receipts/direct-target-prefix-runtime-20261007-v1/textcraft-first-dt'
assert not any((capture/f'rank{rank}-release-update').exists() for rank in (0,1))
physical=subprocess.run(['mx-smi'],capture_output=True,text=True,check=True).stdout
assert not re.search(r'^\|\s*[45]\s+\d+\s+\S',physical,re.M),'Diagnostic devices occupied'
(out/'before-physical.txt').write_text(physical)
env=dict(os.environ,**source['environment']);env.pop('MACA_VISIBLE_DEVICES',None);env.pop('RAY_ADDRESS',None)
env['CUDA_VISIBLE_DEVICES']='4,5';env['DT_TASK']=source['startup_options']['env.env_name']
env['DT_MAX_STEPS']=str(source['startup_options']['env.max_steps'])
env['PYTHONPATH']=':'.join([str(out),source['pythonpath'],str(Path(source['dt_root'])/'clean/qwen35')])
argv=[env['VENV_PYTHON'],str(out/'inspect_layer_collection.py'),'--source',str(source_path),'--output',str(out/'results'),'--case',TASK]
if TASK=='textcraft':
 evidence=root/'receipts/direct-target-textcraft-author-curve-20261008-v1/textcraft-taskrunner-resolved-training-steps.json'
 assert hashlib.sha256(evidence.read_bytes()).hexdigest()=='57874a6f4491da68e5001fdf4f71a9d787b2dd54ec91a62b7216f1caa58da24d'
 argv+=['--owner-total-training-steps','330','--owner-total-steps-evidence',str(evidence)]
with (out/'driver.log').open('xb') as stream:
 process=subprocess.Popen(argv,env=env,cwd=out,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(task=TASK,pid=process.pid,birth=psutil.Process(process.pid).create_time(),launched_unix=time.time(),
 code_commit=COMMIT,scripts=HASHES,argv=argv,devices=[4,5],source_path=str(source_path),source_sha256=SOURCE_SHA,
 actual_recorded_runner=runner,plan_sha256=HASHES['layer-collection-inputs.json'],
 operations=dict(optimizer=0,scheduler=0,rollout=0,checkpoint_restore=0),
 diagnostic_DT_B4_calls=12,wall_budget_per_worker_seconds=1800,formal_release=False,new_GDN_candidate=False,
 scope='Passively retain original joint coefficients, then contract native single-EOS states for all already frozen queries. No observer path, finite rule changes, training value replacements or new sample selection.')
(out/'launch.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))

PY

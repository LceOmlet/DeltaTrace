source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
CUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<'PY'
ROOT='/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922'
OUT='/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/credit-single-background-range-resume-appworld-20261009-v1'
TASK='appworld'
HASHES={'inspect_single_background_collection.py': 'b6c4c34114575703c8da0c2599440c013761244c06058aafbcc6d742af3977da', 'layer-collection-inputs.json': 'e835680d4b527829fae2b2c116c58b198fb3eda7bf12da137e11871e411f87a5', 'inspect_action_curve.py': '7277fade4e9b1cb49f825e4cb2fc54453c9ff3ce4859b20350aa2bd67ffaf3a6', 'inspect_extreme_endpoint.py': '8a72887a32bd11533486ddee6dc0d36b4980bfb77ed94eadc7a9a13f031b36d5', 'diagnostic_range_owner.py': '33b169b3fb660eb8ce57a6bda6ecb04c7f7235a029aa04038ff447b2faddf6fd'}
COMMIT='26a6ba76b2f395b8680fccafa952eb71f8ee536a'
NONFINITE_REPLAY=False
CAPTURE_FLA_SEED=False
import ast,hashlib,json,os,psutil,re,subprocess,time
from pathlib import Path
root=Path(ROOT);out=Path(OUT)
for name,h in HASHES.items():
 assert hashlib.sha256((out/name).read_bytes()).hexdigest()==h
 if name.endswith('.py'):ast.parse((out/name).read_text())
source_path=root/'runs/direct-target-prefix-runtime-20261007-v1'/TASK/(TASK+'-dt')/'source.json'
source=json.loads(source_path.read_bytes())
expected={'textcraft':'2796233e2683f1939896c74b2b578c242dbd7a7f235b9ef61cbedd398f61be52','appworld':'58209daa0fccfea4b70645465e96ea5d203f9d309187b7a64b405cbd9fd47da0'}[TASK]
assert hashlib.sha256(source_path.read_bytes()).hexdigest()==expected
runner=source['actual_CPU_imports']['qwen35_dense_finite_runner']
assert hashlib.sha256(Path(runner['path']).read_bytes()).hexdigest()==runner['sha256']
plan=json.loads((out/'layer-collection-inputs.json').read_bytes())['tasks'][TASK]
cpu=json.loads((root/'receipts/credit-single-background-appworld-20261009-v1/reference-adapter-cpu.json').read_bytes())
assert cpu['queries']==165 and cpu['exact_one_slot_per_query'] and not cpu['cuda_initialized']
physical=subprocess.run(['mx-smi'],capture_output=True,text=True,check=True).stdout
assert not re.search(r'^\|\s*[45]\s+\d+\s+\S',physical,re.M),'Diagnostic devices occupied'
env=dict(os.environ,**source['environment']);env.pop('MACA_VISIBLE_DEVICES',None);env.pop('RAY_ADDRESS',None)
env['DT_SINGLE_BACKGROUND_RESUME_PAIR']='6'
env['DT_SINGLE_BACKGROUND_RESUME_ROUND']='1'
env['DT_DIAGNOSTIC_FLA_RANGE_OWNER']=str(out/'diagnostic_range_owner.py')
env['CUDA_VISIBLE_DEVICES']='4,5';env['DT_TASK']=source['startup_options']['env.env_name']
if NONFINITE_REPLAY:env['DT_SINGLE_BACKGROUND_NONFINITE_REPLAY']='1'
if CAPTURE_FLA_SEED:env['DT_CAPTURE_FLA_PRECAST']='1'
env['DT_MAX_STEPS']=str(source['startup_options']['env.max_steps'])
env['PYTHONPATH']=':'.join([str(out),source['pythonpath'],str(Path(source['dt_root'])/'clean/qwen35')])
argv=[env['VENV_PYTHON'],str(out/'inspect_single_background_collection.py'),'--source',str(source_path),'--output',str(out/'results'),'--case',TASK]
if TASK=='textcraft':
 evidence=root/'receipts/direct-target-textcraft-author-curve-20261008-v1/textcraft-taskrunner-resolved-training-steps.json'
 assert hashlib.sha256(evidence.read_bytes()).hexdigest()=='57874a6f4491da68e5001fdf4f71a9d787b2dd54ec91a62b7216f1caa58da24d'
 argv+=['--owner-total-training-steps','330','--owner-total-steps-evidence',str(evidence)]
calls_per_rank=sum(max(plan['batches'][i]['native_paired_forwards'],plan['batches'][i+1]['native_paired_forwards']) for i in range(6,len(plan['batches']),2))-1
if NONFINITE_REPLAY:calls_per_rank=1
with (out/'driver.log').open('xb') as stream:
 process=subprocess.Popen(argv,env=env,cwd=out,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(task=TASK,pid=process.pid,birth=psutil.Process(process.pid).create_time(),launched_unix=time.time(),
 code_commit=COMMIT,scripts=HASHES,argv=argv,devices=[4,5],source_path=str(source_path),source_sha256=expected,
 isolated_range_owner=dict(path=env['DT_DIAGNOSTIC_FLA_RANGE_OWNER'],sha256=HASHES['diagnostic_range_owner.py']),actual_recorded_runner=runner,plan_sha256=HASHES['layer-collection-inputs.json'],cpu=cpu,
 operations_excluded=['optimizer','scheduler','rollout','checkpoint_restore','new_native_single_deletion'],
 diagnostic_DT_B4_calls_per_rank=calls_per_rank,wall_budget_per_worker_seconds=1800,formal_release=False,
 reuse_completed='Original 74 points and prior 8-point failed-call replay remain separate, never repeated',scope='Only remaining frozen source queries with original single-EOS reference, not a replacement training estimator or quality candidate.',physical_before=physical)
(out/'launch.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))

PY

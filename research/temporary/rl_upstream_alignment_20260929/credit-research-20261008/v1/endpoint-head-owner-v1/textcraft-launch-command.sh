source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
CUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<'PY'
ROOT='/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922'
OUT='/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/endpoint-head-collection-textcraft-20261009-v1'
TASK='textcraft'
HASHES={'inspect_conditional_collection.py': 'a51f29a1057c17159143ab8348da9f287724ad8cc81db2c9e041c7c40b3bc4c8', 'inspect_author_collection.py': '0ac43043bc8efa6645794f7d6a17cf0d2db82c631646eb47e6983eb77de12946', 'inspect_action_curve.py': '7277fade4e9b1cb49f825e4cb2fc54453c9ff3ce4859b20350aa2bd67ffaf3a6', 'inspect_extreme_endpoint.py': '8a72887a32bd11533486ddee6dc0d36b4980bfb77ed94eadc7a9a13f031b36d5', 'endpoint_head_seed.py': '593b7ebddfff594844c228b67b8dcf9e6b411bed485e248987dda7f41e872014', 'textcraft-v3-rank0.json': '9a0d7159108a07b92dd37627c3554ac99d44823fa56a515cfaab61ef7e6f48fd', 'textcraft-v3-rank1.json': '61a0ed713c7c86f036b3fade45e3eb20d59a1459de894b12200c0870dbb5c900'}
SOURCE_SHA='2796233e2683f1939896c74b2b578c242dbd7a7f235b9ef61cbedd398f61be52'
COMMIT='8861da161ad486955fff8c4d73647c3e85f87ef5'
import ast,hashlib,json,os,psutil,re,subprocess,time
from pathlib import Path
root=Path(ROOT);out=Path(OUT)
assert not (out/'launch.json').exists(),'Do not duplicate this comparison launch'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
for name,h in HASHES.items():
 p=out/('candidate' if name in ('qwen35_decoder_finite.py','qwen35_dense_finite_runner.py') else '')/name
 assert sha(p)==h
 if p.suffix=='.py':ast.parse(p.read_text())
plan=json.loads((out/'comparison-inputs.json').read_bytes())
for item in plan['tasks'][TASK]['candidate']['files']:assert sha(item['path'])==item['sha256']
source_path=root/'runs/direct-target-prefix-runtime-20261007-v1'/TASK/(TASK+'-dt')/'source.json'
assert sha(source_path)==SOURCE_SHA
source=json.loads(source_path.read_bytes())
runner=source['actual_CPU_imports']['qwen35_dense_finite_runner']
assert sha(runner['path'])==runner['sha256']
physical=subprocess.run(['mx-smi'],capture_output=True,text=True,check=True).stdout
assert not re.search(r'^\|\s*[45]\s+\d+\s+\S',physical,re.M),'Research devices occupied'
(out/'before-physical.txt').write_text(physical)
env=dict(os.environ,**source['environment']);env.pop('MACA_VISIBLE_DEVICES',None);env.pop('RAY_ADDRESS',None)
env['CUDA_VISIBLE_DEVICES']='4,5';env['DT_TASK']=source['startup_options']['env.env_name']
env['DT_MAX_STEPS']=str(source['startup_options']['env.max_steps'])
# The candidate directory is deliberately absent: original imports stay original.
dt=Path(source['dt_root']);qwen=json.loads(Path(env['DT_ENVIRONMENT_JSON']).read_bytes())['qwen35']
official=env.get('DT_OFFICIAL_ROOT') or qwen['official_root']
assert sha(Path(official)/'ft_ifr_improve.py')==plan['metric_owner']['sha256']
env['PYTHONPATH']=':'.join([str(out),str(dt),official,str(dt/'clean/qwen35'),source['pythonpath'],qwen['ft_extension_root']])
# Resolve the demonstrated missing metric import before loading the actor.
subprocess.run([env['VENV_PYTHON'],'-c','import ft_ifr_improve; import torch; assert not torch.cuda.is_initialized()'],env=dict(env,CUDA_VISIBLE_DEVICES='-1'),check=True,stdout=subprocess.DEVNULL,timeout=45)
argv=[env['VENV_PYTHON'],str(out/'inspect_conditional_collection.py'),'--source',str(source_path),'--output',str(out/'results'),'--case',TASK]
if TASK=='textcraft':
 evidence=root/'receipts/direct-target-textcraft-author-curve-20261008-v1/textcraft-taskrunner-resolved-training-steps.json'
 assert sha(evidence)=='57874a6f4491da68e5001fdf4f71a9d787b2dd54ec91a62b7216f1caa58da24d'
 argv+=['--owner-total-training-steps','330','--owner-total-steps-evidence',str(evidence)]
with (out/'driver.log').open('xb') as stream:
 p=subprocess.Popen(argv,env=env,cwd=out,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(task=TASK,pid=p.pid,birth=psutil.Process(p.pid).create_time(),launched_unix=time.time(),
 code_commit=COMMIT,scripts=HASHES,argv=argv,devices=[4,5],source_path=str(source_path),source_sha256=SOURCE_SHA,
 actual_recorded_runner=runner,comparison_inputs_sha256=sha(out/'comparison-inputs.json'),
 operations=dict(optimizer=0,scheduler=0,rollout=0,checkpoint_restore=0),
 diagnostic_DT_B4_calls_per_rank=plan['DT_calls_per_rank'],wall_budget_per_worker_seconds=plan['tasks'][TASK]['wall_budget_seconds'],
 native_calls_per_rank=plan['native_calls_per_rank'],candidate_deployed=False,scope=plan['controls'])
(out/'launch.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))

PY

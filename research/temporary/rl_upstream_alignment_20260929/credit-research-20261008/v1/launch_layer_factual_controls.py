"""Launch the already staged passive diagnostic only when GPUs4/5 are free."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('transport',HERE.parents[1]/'stage_environment_entry.py')
transport=importlib.util.module_from_spec(spec)
spec.loader.exec_module(transport)
task=sys.argv[1]
assert task in ('textcraft','appworld')
suboperations='--suboperations' in sys.argv[2:]
attention_branches='--attention-branches' in sys.argv[2:]
assert not attention_branches or suboperations
revision=next((v.split('=',1)[1] for v in sys.argv[2:] if v.startswith('--revision=')), 'v1')
assert revision in ('v1','v2')
tag='attention-gate' if attention_branches else ('layer-suboperations' if suboperations else 'layer-factual-controls')
folder=HERE/(f'{tag}-{task}'+('-'+revision if revision!='v1' else ''))
prepared=json.loads((folder/'preparation.json').read_bytes())
code=r'''
import ast,hashlib,json,os,psutil,re,subprocess,time
from pathlib import Path
out=Path(OUT);root=Path(ROOT)
assert not (out/'launch.json').exists(),'Do not duplicate this collection'
prep=json.loads((out/'preparation.json').read_bytes())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
for name,digest in prep['files'].items():
 assert sha(out/name)==digest
 if name.endswith('.py'):ast.parse((out/name).read_text())
source_path=root/'runs/direct-target-prefix-runtime-20261007-v1'/TASK/(TASK+'-dt')/'source.json'
expected=dict(textcraft='2796233e2683f1939896c74b2b578c242dbd7a7f235b9ef61cbedd398f61be52',appworld='58209daa0fccfea4b70645465e96ea5d203f9d309187b7a64b405cbd9fd47da0')
assert sha(source_path)==expected[TASK]
source=json.loads(source_path.read_bytes())
runner=source['actual_CPU_imports']['qwen35_dense_finite_runner']
assert sha(runner['path'])==runner['sha256']
physical=subprocess.run(['mx-smi'],capture_output=True,text=True,check=True).stdout
assert not re.search(r'^\|\s*[45]\s+\d+\s+\S',physical,re.M),'GPU4/5 occupied; do not launch or displace their work'
env=dict(os.environ,**source['environment']);env.pop('MACA_VISIBLE_DEVICES',None);env.pop('RAY_ADDRESS',None)
env['CUDA_VISIBLE_DEVICES']='4,5';env['DT_LAYER_FACTUAL_CONTROL']='1'
if SUBOPERATIONS:env['DT_LAYER_SUBOPERATIONS']='1'
if ATTENTION_BRANCHES:env['DT_ATTENTION_BRANCHES']='1'
env['DT_TASK']=source['startup_options']['env.env_name'];env['DT_MAX_STEPS']=str(source['startup_options']['env.max_steps'])
dt=Path(source['dt_root']);q=json.loads(Path(env['DT_ENVIRONMENT_JSON']).read_bytes())['qwen35']
env['PYTHONPATH']=':'.join([str(out),str(dt),env.get('DT_OFFICIAL_ROOT') or q['official_root'],str(dt/'clean/qwen35'),source['pythonpath'],q['ft_extension_root']])
argv=[env['VENV_PYTHON'],str(out/'inspect_layer_collection.py'),'--source',str(source_path),'--output',str(out/'results'),'--case',TASK]
if TASK=='textcraft':
 evidence=root/'receipts/direct-target-textcraft-author-curve-20261008-v1/textcraft-taskrunner-resolved-training-steps.json'
 assert sha(evidence)=='57874a6f4491da68e5001fdf4f71a9d787b2dd54ec91a62b7216f1caa58da24d'
 argv+=['--owner-total-training-steps','330','--owner-total-steps-evidence',str(evidence)]
with (out/'driver.log').open('xb') as stream:
 process=subprocess.Popen(['timeout','--kill-after=10s','2100s',*argv],cwd=out,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(task=TASK,pid=process.pid,birth=psutil.Process(process.pid).create_time(),launched_unix=time.time(),
 argv=['timeout','--kill-after=10s','2100s',*argv],devices=[4,5],source_path=str(source_path),source_sha256=sha(source_path),
 scripts=prep['files'],base_commit=COMMIT,files_committed_at_launch=bool(prep.get('source_git_blobs')),
 source_git_blobs=prep.get('source_git_blobs',{}),
 per_card_DT_batch=4,actual_paired_native_batch=8,diagnostic_DT_calls=12,
 diagnostic_native_calls_per_rank=30 if TASK=='textcraft' else 37,
 coefficient_change=False,production_modified=False,optimizer=0,rollout=0,checkpoint_restore=0,
 wall_budget_per_worker_seconds=1800,hard_process_limit_seconds=2100,
 host_available_bytes=psutil.virtual_memory().available,
 suboperations=SUBOPERATIONS,
 attention_branches=ATTENTION_BRANCHES,
 purpose=('Read unchanged native/DT suboperation contractions on all frozen queries, including residual-add rounding and actual versus recomputed head coefficient.' if SUBOPERATIONS else
          'Measure same-call factual-state contraction controls on all frozen queries; not a credit correction, new sample, or candidate rule.'))
(out/'launch.json').write_text(json.dumps(receipt,indent=2)+'\n');(out/'before-physical.txt').write_text(physical)
print(json.dumps(receipt))
'''
header='OUT='+repr(prepared['remote'])+'\nROOT='+repr(transport.ROOT)+'\nTASK='+repr(task)+'\nSUBOPERATIONS='+repr(suboperations)+'\nATTENTION_BRANCHES='+repr(attention_branches)+'\nCOMMIT='+repr(subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())+'\n'
script='source '+transport.ENTRY+'/metax-entry.env.sh\nCUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<\'PY\'\n'+header+code+'\nPY\n'
(folder/'launch-command.sh').write_text(script,encoding='utf-8')
result=subprocess.run(transport.SSH+['bash','-s'],input=script.encode(),capture_output=True,timeout=55)
(folder/'launch.stdout').write_bytes(result.stdout);(folder/'launch.stderr').write_bytes(result.stderr)
if result.returncode:print(result.stderr.decode(errors='replace'))
result.check_returncode()
value=json.loads(result.stdout)
(folder/'launch.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
print(json.dumps(value))

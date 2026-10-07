set -eu
source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
"$VENV_PYTHON" - <<'PY'

from pathlib import Path
import hashlib,json,os,psutil,re,subprocess,time
root=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922');out=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/direct-target-credit-sample-20261007-v1');task='textcraft'
population_path=out/'population.json';population=json.loads(population_path.read_bytes())['tasks'][task]
assert hashlib.sha256(population_path.read_bytes()).hexdigest()=='fef26ccbf632233b226158e36d456b9b847658b126f2821f8ad27d79b544c8f2'
p=Path(population['source']['path']);source=json.loads(p.read_bytes())
assert hashlib.sha256(p.read_bytes()).hexdigest()==population['source']['sha256']
assert not (out/(task+'-launch.json')).exists(),'Do not launch this diagnosis twice'
assert psutil.Process(2833207).create_time()==1791370325.16
for rank in (0,1):assert not (root/'receipts/direct-target-prefix-runtime-20261007-v1/textcraft-first-dt'/('rank'+str(rank)+'-release-update')).exists()
assert not psutil.pid_exists(2786671)
for name in ('appworld','textcraft'):
 f=out/(name+'-launch.json')
 if f.exists():
  record=json.loads(f.read_bytes());assert not psutil.pid_exists(record['pid']),'Previous bounded diagnosis still running'
physical=subprocess.run(['mx-smi'],capture_output=True,text=True,check=True).stdout
assert not re.search(r'^\|\s*[45]\s+\d+\s+\S',physical,re.M),'Selected devices are occupied'
(out/(task+'-before-physical.txt')).write_text(physical)
for name,h in {'inspect_sample.py': '5a02e2470e9a9e617a33fd2c7c006de0f45a4f58c2a39dde78b4f317b7560d8d', 'inspect_extreme_endpoint.py': '8a72887a32bd11533486ddee6dc0d36b4980bfb77ed94eadc7a9a13f031b36d5'}.items():assert hashlib.sha256((out/name).read_bytes()).hexdigest()==h
env=dict(os.environ,**source['environment']);env.pop('RAY_ADDRESS',None);env.pop('MACA_VISIBLE_DEVICES',None)
env['CUDA_VISIBLE_DEVICES']='4,5';env['DT_TASK']=source['startup_options']['env.env_name'];env['DT_MAX_STEPS']=str(source['startup_options']['env.max_steps'])
dt=Path(env['DT_ROOT']);qwen=json.loads(Path(env['DT_ENVIRONMENT_JSON']).read_bytes())['qwen35']
assert hashlib.sha256((dt/'clean/qwen35/qwen35_dense_finite_runner.py').read_bytes()).hexdigest()=='628006b637516f8d62e95583a9eb51fe9038ea7931798e2c1f42c28e154cf24f'
env['PYTHONPATH']=':'.join([str(out),str(dt),env.get('DT_OFFICIAL_ROOT') or qwen['official_root'],str(dt/'clean/qwen35'),source['pythonpath'],qwen['ft_extension_root']])
argv=[env['VENV_PYTHON'],str(out/'inspect_sample.py'),'--task',task,'--source',str(p),'--population',str(population_path),'--output',str(out/('results-'+task))]
if task=='textcraft':argv+=['--resolved-steps','330']
with (out/(task+'-driver.log')).open('xb') as stream:process=subprocess.Popen(argv,cwd=str(out),env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
record=dict(task=task,pid=process.pid,birth=psutil.Process(process.pid).create_time(),launched_unix=time.time(),devices=[4,5],argv=argv,scripts={'inspect_sample.py': '5a02e2470e9a9e617a33fd2c7c006de0f45a4f58c2a39dde78b4f317b7560d8d', 'inspect_extreme_endpoint.py': '8a72887a32bd11533486ddee6dc0d36b4980bfb77ed94eadc7a9a13f031b36d5'},source_path=str(p),source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),population_sha256=hashlib.sha256(population_path.read_bytes()).hexdigest(),planned_native_calls_per_rank=4,DT=0,optimizer_steps=0,formal_restart=False,update_released=False)
(out/(task+'-launch.json')).write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))

PY

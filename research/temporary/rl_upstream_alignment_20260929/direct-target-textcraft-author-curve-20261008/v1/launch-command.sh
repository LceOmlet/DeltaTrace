set -eu
source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
"$VENV_PYTHON" - <<'PY'

from pathlib import Path
import hashlib,json,os,psutil,re,subprocess,time
root=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922');out=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/direct-target-textcraft-author-curve-20261008-v1')
p=root/'runs/direct-target-prefix-runtime-20261007-v1/textcraft/textcraft-dt/source.json';source=json.loads(p.read_bytes())
assert hashlib.sha256(p.read_bytes()).hexdigest()=='2796233e2683f1939896c74b2b578c242dbd7a7f235b9ef61cbedd398f61be52'
assert not (out/'launch.json').exists(),'Do not launch this diagnosis twice'
assert psutil.Process(2833207).create_time()==1791370325.16
for rank in (0,1):assert not (root/'receipts/direct-target-prefix-runtime-20261007-v1/textcraft-first-dt'/('rank'+str(rank)+'-release-update')).exists()
for pid in (2786671,3714027,3902140):assert not psutil.pid_exists(pid)
physical=subprocess.run(['mx-smi'],capture_output=True,text=True,check=True).stdout
assert not re.search(r'^\|\s*[45]\s+\d+\s+\S',physical,re.M),'Selected devices are occupied'
(out/'before-physical.txt').write_text(physical)
for name,h in {'inspect_action_curve.py': '7277fade4e9b1cb49f825e4cb2fc54453c9ff3ce4859b20350aa2bd67ffaf3a6', 'inspect_extreme_endpoint.py': '8a72887a32bd11533486ddee6dc0d36b4980bfb77ed94eadc7a9a13f031b36d5', 'textcraft-taskrunner-resolved-training-steps.json': '57874a6f4491da68e5001fdf4f71a9d787b2dd54ec91a62b7216f1caa58da24d'}.items():assert hashlib.sha256((out/name).read_bytes()).hexdigest()==h
env=dict(os.environ,**source['environment']);env.pop('RAY_ADDRESS',None);env.pop('MACA_VISIBLE_DEVICES',None)
env['CUDA_VISIBLE_DEVICES']='4,5';env['DT_TASK']=source['startup_options']['env.env_name'];env['DT_MAX_STEPS']=str(source['startup_options']['env.max_steps'])
dt=Path(env['DT_ROOT']);qwen=json.loads(Path(env['DT_ENVIRONMENT_JSON']).read_bytes())['qwen35']
official=env.get('DT_OFFICIAL_ROOT') or qwen['official_root']
assert hashlib.sha256((Path(official)/'ft_ifr_improve.py').read_bytes()).hexdigest()=='583f4b7d0426407eb9a517f173365762860a1f4382f472dffb5c07de7d3e94a1'
env['PYTHONPATH']=':'.join([str(out),str(dt),official,str(dt/'clean/qwen35'),source['pythonpath'],qwen['ft_extension_root']])
argv=[env['VENV_PYTHON'],str(out/'inspect_action_curve.py'),'--source',str(p),'--output',str(out/'results'),'--case','textcraft','--owner-total-training-steps','330','--owner-total-steps-evidence',str(out/'textcraft-taskrunner-resolved-training-steps.json')]
with (out/'driver.log').open('xb') as stream:process=subprocess.Popen(argv,cwd=str(out),env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
record=dict(pid=process.pid,birth=psutil.Process(process.pid).create_time(),launched_unix=time.time(),devices=[4,5],argv=argv,scripts={'inspect_action_curve.py': '7277fade4e9b1cb49f825e4cb2fc54453c9ff3ce4859b20350aa2bd67ffaf3a6', 'inspect_extreme_endpoint.py': '8a72887a32bd11533486ddee6dc0d36b4980bfb77ed94eadc7a9a13f031b36d5', 'textcraft-taskrunner-resolved-training-steps.json': '57874a6f4491da68e5001fdf4f71a9d787b2dd54ec91a62b7216f1caa58da24d'},source_path=str(p),source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),author_k=20,code_commit='2b98d7f96fadd0ecf4af50109f7144251a02a580',case='textcraft',planned_native_forward_calls_per_rank=42,DT_calls=0,optimizer_steps=0,formal_restart=False,update_released=False)
(out/'launch.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))

PY

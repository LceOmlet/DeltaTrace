set -eu
source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_qwen35_20260912/env/bin/python - <<'PY'

import hashlib,importlib.util,json,os,pathlib,psutil,resource,subprocess,time
P=pathlib.Path;R=P('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT=R/'candidates/textcraft-fresh-row-prefix-20261007-v2';env=dict(os.environ,**json.loads((OUT/'run-env.json').read_bytes()))
env.update(CUDA_VISIBLE_DEVICES='',MACA_VISIBLE_DEVICES='-1')
T=R/'third_party/AgentGym-d014732d9fe39b975c368c03749bfd50950067f6'
extras=R/'environments/textcraft-extras-20260930'
assert (extras/'gymnasium/__init__.py').is_file(),str(extras)
env['PYTHONPATH']=str(extras)+':'+str(T/'agentenv')+':'+str(T/'agentenv-textcraft')+':'+env['PYTHONPATH']
sha=lambda p:hashlib.sha256(P(p).read_bytes()).hexdigest()
binding=lambda p:dict(path=str(p),sha256=sha(p),bytes=P(p).stat().st_size)
assert sha(T/'agentenv-textcraft/agentenv_textcraft/launch.py')=='c05434ce411b77b7618295666453aaa3d78c575b0caa60fa2eb503e3dda71abf'
assert sha(T/'agentenv-textcraft/agentenv_textcraft/server.py')=='b3ccfd7cc80a47e2e1add4b158950d7e27418eddaa0369c9edd5af2fdca29925'
listeners=[c for c in psutil.net_connections(kind='inet') if c.laddr and c.laddr.port==36005 and c.status=='LISTEN']
assert not listeners, 'Inspect existing listener rather than replace it'
argv=[env['VENV_PYTHON'],'-c','from agentenv_textcraft import launch; launch()','--host','0.0.0.0','--port','36005']
started=time.time()
for name in ['service-ready.log','official-textcraft-service.log']:
 p=OUT/name
 if p.exists():
  preserved=p.with_name(p.name+'.before-'+sha(p)[:12])
  if not preserved.exists():preserved.write_bytes(p.read_bytes())
with (OUT/'official-textcraft-service.log').open('ab') as log:
 proc=subprocess.Popen(argv,cwd=T/'agentenv-textcraft',env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
birth=psutil.Process(proc.pid).create_time()
ready_code=r"""
import hashlib,json,os,pathlib,sys,time
from types import SimpleNamespace
from textcraft_environment_entry import owner_module
P=pathlib.Path
path=P(os.environ['AGENTGYM_RL_ROOT'])/'AgentGym-RL/verl/utils/agentgym/client.py'
owner=owner_module(path,'original_textcraft_ready_client')
config=json.loads((P(os.environ['DT_ENTRY_ROOT'])/'owner_environment_configs.json').read_bytes())['TextCraft']['train']['actor_rollout_ref']['agentgym']
assert config['env_addr']=='http://127.0.0.1:36005'
client=owner.init_env_client(SimpleNamespace(**config))
try:
 reset=client.reset(0);observation=client.observe()
 result=dict(status='official_client_create_reset_observe_close',client_module=str(path),client_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
  concrete_client=client.__class__.__module__+'.'+client.__class__.__name__,environment_id=client.env_id,
  reset_keys=sorted(reset),observation_chars=len(observation),config=config)
finally: client.close()
P(sys.argv[1]).write_text(json.dumps(result,indent=2)+'\n')
"""
with (OUT/'service-ready.log').open('wb') as log:
 ready=subprocess.run([env['VENV_PYTHON'],'-c',ready_code,str(OUT/'service-ready-client.json')],env=env,cwd=OUT/'entry',stdout=log,stderr=subprocess.STDOUT,timeout=75)
assert ready.returncode==0,(ready.returncode,str(OUT/'service-ready.log'))
p=psutil.Process(proc.pid);assert p.create_time()==birth and p.status()!=psutil.STATUS_ZOMBIE
listeners=[dict(pid=c.pid,address=list(c.laddr),status=c.status) for c in psutil.net_connections(kind='inet') if c.laddr and c.laddr.port==36005 and c.status=='LISTEN']
assert any(c['pid']==proc.pid for c in listeners)
record=dict(status='official_TextCraft_service_live_real_client_ready',observed_unix=time.time(),pid=proc.pid,pid_birth=birth,
 live=True,process_status=p.status(),argv=argv,working_directory=str(T/'agentenv-textcraft'),port=36005,listeners=listeners,
 owner=dict(commit='d014732d9fe39b975c368c03749bfd50950067f6',launch=binding(T/'agentenv-textcraft/agentenv_textcraft/launch.py'),
  server=binding(T/'agentenv-textcraft/agentenv_textcraft/server.py'),client=binding(T/'agentenv/agentenv/envs/textcraft.py')),
 ready=binding(OUT/'service-ready-client.json'),log=str(OUT/'official-textcraft-service.log'),
 CUDA_VISIBLE_DEVICES=env['CUDA_VISIBLE_DEVICES'],MACA_VISIBLE_DEVICES=env['MACA_VISIBLE_DEVICES'],
 dependency_overlay=dict(path=str(extras),gymnasium=binding(extras/'gymnasium/__init__.py'),scope='Existing recorded TextCraft extras; no installation or modification'),
 service_RSS_bytes=p.memory_info().rss,phase_wall_seconds=time.time()-started,
 model=False,DT=False,GPU=False,checkpoint=False,training_submitted=False,other_services_changed=False,
 scope='Original author launch() function and existing provisioned assets; original author init_env_client then native create/reset/observe/close only; no alternative HTTP implementation')
(OUT/'service-current.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(dict(service=str(OUT/'service-current.json'),sha256=sha(OUT/'service-current.json'),pid=proc.pid,pid_birth=birth,ready=True,phase_wall_seconds=record['phase_wall_seconds'])),flush=True)
# Reuse one unchanged existing CPU validator test, without changing any options.
test=OUT/'entry/test_owner_entry_launch.py'
assert test.is_file(),str(test)
with (OUT/'official-launcher-validator.log').open('wb') as log:
 result=subprocess.run([env['VENV_PYTHON'],'-m','pytest','-q',str(test)+'::test_textcraft_formal_owner_workload_and_native_validator',
  '--junitxml='+str(OUT/'official-launcher-validator.xml'),'-p','no:cacheprovider','--basetemp='+str(OUT/'validator-tmp')],
  cwd=OUT/'entry',env=env,stdout=log,stderr=subprocess.STDOUT,timeout=90)
rec=dict(returncode=result.returncode,test=binding(test),log=binding(OUT/'official-launcher-validator.log'),
 junit=binding(OUT/'official-launcher-validator.xml'),CUDA_VISIBLE_DEVICES='',MACA_VISIBLE_DEVICES='-1',
 original_test_unchanged=True,model=False,DT=False,GPU=False,checkpoint=False,training_submitted=False,
 child_maxrss_bytes=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss*1024,
 scope='Single existing original TextCraft formal workload/Hydra/native-validator CPU test; no new experiment or numeric test')
(OUT/'official-launcher-validator.json').write_text(json.dumps(rec,indent=2)+'\n')
print(json.dumps(dict(validator=str(OUT/'official-launcher-validator.json'),sha256=sha(OUT/'official-launcher-validator.json'),returncode=result.returncode)),flush=True)
assert result.returncode==0

PY

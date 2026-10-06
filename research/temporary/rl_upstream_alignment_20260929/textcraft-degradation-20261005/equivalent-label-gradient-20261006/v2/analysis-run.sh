/opt/conda/bin/python - <<'PY'
import ast,hashlib,json,pathlib,psutil,subprocess,time
out=pathlib.Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/textcraft-label-gradient-20261006-v2'); base=pathlib.Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/textcraft-native-minibatch-20261006-v4')
source=out/'analyze_textcraft_label_gradients.py'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='fb407dff0dbfbdfbc75244303aa5eb29b38cc891d679f9f24249e8a0b91075ac'
ast.parse(source.read_bytes())
assert (out/'completed.json').is_file(), 'Read the existing observation phase first'
job=json.loads((out/'job.json').read_bytes())
parent=psutil.Process(job['reused_environment_pid'])
assert abs(parent.create_time()-job['reused_environment_pid_birth'])<.05
env=parent.environ(); env.pop('RAY_ADDRESS',None)
for key,value in job['runtime_environment'].items():
 if value is None: env.pop(key,None)
 else: env[key]=value
env['CUDA_VISIBLE_DEVICES']=''
env['OMP_NUM_THREADS']='1'
argv=[env['VENV_PYTHON'],'-u',str(source),'--root',str(out)]
started=time.time()
with (out/'analysis.stdout.txt').open('wb') as log:
 result=subprocess.run(argv,cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT)
receipt=dict(source=dict(path=str(source),sha256='fb407dff0dbfbdfbc75244303aa5eb29b38cc891d679f9f24249e8a0b91075ac'),
 argv=argv,returncode=result.returncode,started_unix=started,completed_unix=time.time(),
 original_environment_pid=parent.pid,original_environment_pid_birth=parent.create_time(),
 cuda_visible_devices='',model_initializations=0,DT_calls=0,optimizer_steps=0)
if result.returncode==0:
 p=out/'label-gradient-analysis.json'; receipt['output']=dict(path=str(p),
  sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)
(out/'analysis-cpu-run.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
raise SystemExit(result.returncode)
PY

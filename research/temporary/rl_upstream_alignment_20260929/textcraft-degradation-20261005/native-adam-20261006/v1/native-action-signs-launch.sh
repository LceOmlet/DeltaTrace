/opt/conda/bin/python - <<'PY'
import ast,hashlib,json,pathlib,psutil,subprocess,time
out=pathlib.Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/textcraft-native-adam-20261006-v1'); source=out/'analyze_textcraft_native_action_signs.py'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='469b1d8969452929990f09303e5d6f1e9f3d2fddf51b08de17188898ca981317'
ast.parse(source.read_bytes())
assert (out/'completed.json').is_file() and not (out/'native-action-signs.json').exists()
job=json.loads((out/'job.json').read_bytes())
assert job['reused_environment_pid']==1856052 and job['reused_environment_pid_birth']==1791197944.7
parent=psutil.Process(1856052); assert abs(parent.create_time()-1791197944.7)<.05
env=parent.environ(); env.pop('RAY_ADDRESS',None)
env.update(CUDA_VISIBLE_DEVICES='',MACA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONPATH=str(out)+':'+job['runtime_environment']['PYTHONPATH'])
input_path=pathlib.Path(job['input']['path'])
assert hashlib.sha256(input_path.read_bytes()).hexdigest()==job['input']['sha256']=='45ae51e3e18f206e238a1f1e934eaf9a4074c9361dabdab83bb3e00a2b4b084d'
argv=[env['VENV_PYTHON'],str(source),'--input-dir',str(out),'--minibatch-path',str(input_path),'--output',str(out/'native-action-signs.json')]
began=time.time()
with (out/'native-action-signs.stdout.txt').open('wb') as log:
 result=subprocess.run(argv,cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT)
assert result.returncode==0,'Inspect saved CPU output; this path never launches a model'
data=json.loads((out/'native-action-signs.json').read_bytes())
assert all(value==0 for value in data['operations'].values())
assert not data['runtime']['resources_after']['cuda_initialized'] and not data['runtime']['resources_after']['distributed_initialized']
receipt=dict(source=dict(path=str(source),sha256='469b1d8969452929990f09303e5d6f1e9f3d2fddf51b08de17188898ca981317'),observed_unix=time.time(),seconds=time.time()-began,
 reused_environment_pid=parent.pid,reused_environment_pid_birth=parent.create_time(),argv=argv,
 input=job['input'],output=dict(path=str(out/'native-action-signs.json'),sha256=hashlib.sha256((out/'native-action-signs.json').read_bytes()).hexdigest()),
 operations=data['operations'],runtime=data['runtime'])
(out/'native-action-signs-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
PY

/opt/conda/bin/python - <<'PY'
import ast,hashlib,json,pathlib,psutil,subprocess,time
out=pathlib.Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/textcraft-native-adam-20261006-v1'); source=out/'analyze_textcraft_native_task_direction.py'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='9bcb58619c7c1caa2f4da1221abc941d125dd989028bda79fc3d52423c9ffbe0'
ast.parse(source.read_bytes())
assert (out/'completed.json').is_file() and not (out/'native-task-direction.json').exists()
job=json.loads((out/'job.json').read_bytes())
parent=psutil.Process(job['reused_environment_pid'])
assert abs(parent.create_time()-job['reused_environment_pid_birth'])<.05
env=parent.environ(); env.pop('RAY_ADDRESS',None)
env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
input_path=pathlib.Path(job['input']['path'])
assert hashlib.sha256(input_path.read_bytes()).hexdigest()==job['input']['sha256']
argv=[env['VENV_PYTHON'],str(source),'--input-dir',str(out),'--output',str(out/'native-task-direction.json')]
began=time.time()
with (out/'native-task-direction.stdout.txt').open('wb') as log:
 result=subprocess.run(argv,cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT)
assert result.returncode==0,'Inspect the saved CPU analysis failure; no model job is launched by this path'
data=json.loads((out/'native-task-direction.json').read_bytes())
assert all(value==0 for value in data['operations'].values())
assert not data['runtime']['resources_after']['cuda_initialized']
assert not data['runtime']['resources_after']['distributed_initialized']
receipt=dict(source=dict(path=str(source),sha256='9bcb58619c7c1caa2f4da1221abc941d125dd989028bda79fc3d52423c9ffbe0'),observed_unix=time.time(),seconds=time.time()-began,
 reused_environment_pid=parent.pid,reused_environment_pid_birth=parent.create_time(),argv=argv,
 output=dict(path=str(out/'native-task-direction.json'),sha256=hashlib.sha256((out/'native-task-direction.json').read_bytes()).hexdigest()),
 operations=data['operations'])
(out/'native-task-direction-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
PY

/opt/conda/bin/python - <<'PY'
import hashlib,json,pathlib,psutil,subprocess,time
out=pathlib.Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/reward-query-clock-repair-20261006-v1')
assert not (out/'query-clock-repair.json').exists()
job=json.loads(pathlib.Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/textcraft-native-adam-20261006-v1/job.json').read_bytes())
parent=psutil.Process(job['reused_environment_pid'])
assert abs(parent.create_time()-job['reused_environment_pid_birth']) < .05
for name,expected in {'reward_readout.py': '94a7afbc09da72b62572d31fd32a6534f6e8f3daf656fce1011cdfa68b3c3e2b', 'test_reward_readout.py': 'ae16a359967851d87019af59de7e8e208e87a64fa7dad375149bb05ff47a4b8c', 'verify_reward_query_clock_cpu.py': 'de42b11a4e7c48bd017881fb2caf59fa585d40a16d23e208b3adf0112008866b'}.items():
 assert hashlib.sha256((out/name).read_bytes()).hexdigest()==expected
env=parent.environ(); env.pop('RAY_ADDRESS',None)
env.update(job['runtime_environment'])
env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
 TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',DT_QUERY_REPAIR_ROOT=str(out))
env['PYTHONPATH']=str(out)+':'+env['PYTHONPATH']
start=time.time()
with (out/'cpu.stdout.txt').open('wb') as log:
 result=subprocess.run([env['VENV_PYTHON'],str(out/'verify_reward_query_clock_cpu.py')],
  cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=120)
print((out/'cpu.stdout.txt').read_text())
receipt=dict(observed_unix=time.time(),seconds=time.time()-start,exit_code=result.returncode,
 reused_environment_pid=parent.pid,reused_environment_pid_birth=parent.create_time(),sources={'reward_readout.py': '94a7afbc09da72b62572d31fd32a6534f6e8f3daf656fce1011cdfa68b3c3e2b', 'test_reward_readout.py': 'ae16a359967851d87019af59de7e8e208e87a64fa7dad375149bb05ff47a4b8c', 'verify_reward_query_clock_cpu.py': 'de42b11a4e7c48bd017881fb2caf59fa585d40a16d23e208b3adf0112008866b'})
(out/'launch-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt)); raise SystemExit(result.returncode)
PY

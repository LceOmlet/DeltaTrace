/opt/conda/bin/python - <<'PY'
import hashlib,json,pathlib,psutil,subprocess,time
native=pathlib.Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/textcraft-native-minibatch-20261006-v4'); out=pathlib.Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/textcraft-query-clock-gradient-20261006-v2'); expected='a48837c17a291e500400a058e42f7a5d4a1c965946e7b15c92b133d500a9974a'
job=json.loads((native/'job.json').read_bytes())
parent=psutil.Process(job['reused_environment_pid'])
assert abs(parent.create_time()-job['reused_environment_pid_birth'])<.05
env=parent.environ()
env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1')
env.pop('RAY_ADDRESS',None)
prepared=json.loads((native/'prepared-diagnostic.json').read_bytes())
source_dirs={}
for name,label in [('textcraft_owner_rollout.py','entry'),('dp_actor.py','verl'),('qwen35_dense_finite_runner.py','dt')]:
    paths=[pathlib.Path(p) for p in prepared['sources'] if pathlib.Path(p).name==name]
    assert len(paths)==1
    path=paths[0]
    assert hashlib.sha256(path.read_bytes()).hexdigest()==prepared['sources'][str(path)]
    source_dirs[label]=str(path.parent if label=='entry' else path.parents[3] if label=='verl' else path.parents[2])
env['PYTHONPATH']=':'.join([str(out),str(native),str(native.parent/'textcraft-native-readout-20261006-v2'),source_dirs['entry'],source_dirs['verl'],source_dirs['dt']])
script=out/'analyze_saved_query_clock_credit.py'
assert hashlib.sha256(script.read_bytes()).hexdigest()==expected
assert (native/'native-minibatch-completed.json').is_file()
started=time.time(); before=psutil.virtual_memory().available
assert (out/'completed.json').is_file()
assert hashlib.sha256((out/'clock-recomputed-native-minibatch.pkl').read_bytes()).hexdigest()=='ef83aa38b1fffdc1971d32c4ddf3da6eab45f7a791ee65c72e874237b7b32a8c'
helper=native.parent/'textcraft-native-readout-20261006-v2/analyze_saved_native_b4_denominators.py'
assert hashlib.sha256(helper.read_bytes()).hexdigest()=='ce1860c0de8c364a6d5599523ed86c14d39d8400238c45656b04e783f71d6f3e'
argv=[env['VENV_PYTHON'],str(script),'--input-dir',str(out),'--minibatch-path',str(native/'native-optimizer-minibatch.pkl'),'--output',str(out/'native-query-clock-credit-summary.json')]
with (out/'native-query-clock-credit-cpu.stdout.txt').open('wb') as stream:
    result=subprocess.run(argv,cwd=out,env=env,stdout=stream,stderr=subprocess.STDOUT)
receipt={
    'scope':'CPU saved original DataProto analysis only; no model, forward, DT, backward or state update',
    'argv':argv,'analyzer_sha256':expected,'wrapper_pid':psutil.Process().pid,
    'wrapper_pid_birth':psutil.Process().create_time(),
    'reused_environment_pid':parent.pid,'reused_environment_pid_birth':parent.create_time(),
    'cuda_visible_devices':'','pythonpath':env['PYTHONPATH'],
    'started_unix':started,'completed_unix':time.time(),'exit_code':result.returncode,
    'host_available_before_bytes':before,'host_available_after_bytes':psutil.virtual_memory().available,
    'stdout':str(out/'native-query-clock-credit-cpu.stdout.txt'),'result':str(out/'native-query-clock-credit-summary.json'),
    'source_prepared_sha256':hashlib.sha256((native/'prepared-diagnostic.json').read_bytes()).hexdigest(),
    'source_job_sha256':hashlib.sha256((native/'job.json').read_bytes()).hexdigest()}
(out/'native-query-clock-credit-cpu-run.json').write_text(json.dumps(receipt,indent=2)+'\n')
print((out/'native-query-clock-credit-cpu.stdout.txt').read_text()[-8000:])
print(json.dumps(receipt))
raise SystemExit(result.returncode)
PY

"""Stop only the recorded diagnostic that missed its observation boundary."""
import subprocess
from stage_environment_entry import ROOT,SSH

script=r'''/opt/conda/bin/python - <<'PY'
import json,pathlib,psutil,time
out=pathlib.Path('__ROOT__/receipts/textcraft-native-minibatch-20261006-v2')
j=json.loads((out/'job.json').read_bytes());p=psutil.Process(j['pid'])
assert p.pid==4050498 and abs(p.create_time()-j['pid_birth'])<.05
assert 'verify_textcraft_native_minibatch.py' in p.cmdline()[2]
processes=[p]+p.children(recursive=True)
record=dict(unix=time.time(),pid=p.pid,pid_birth=p.create_time(),
 reason='DiagnosticTrainer alias did not bind to original Ray-serialized TaskRunner globals; original 256 rollout began instead of requested64. Stop before update and correct the diagnostic seam only.',
 processes=[dict(pid=q.pid,birth=q.create_time()) for q in processes],
 optimizer_boundary_snapshot_exists=(out/'native-optimizer-minibatch.pkl').exists(),
 completion_exists=(out/'native-minibatch-completed.json').exists())
(out/'stopped-wrong-diagnostic-scope.json').write_text(json.dumps(record,indent=2)+'\n')
for q in reversed(processes):
 try:q.terminate()
 except psutil.NoSuchProcess:pass
_,alive=psutil.wait_procs(processes,timeout=5)
for q in alive:
 try:q.kill()
 except psutil.NoSuchProcess:pass
print(json.dumps(record))
PY
'''.replace('__ROOT__',ROOT)
if __name__=='__main__':
    r=subprocess.run(SSH+['bash','-s'],input=script.encode(),capture_output=True)
    print(r.stdout.decode(errors='replace'));print(r.stderr.decode(errors='replace'));r.check_returncode()

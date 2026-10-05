"""Run a new CPU-only analysis beside existing readout v2 receipts; no job launch."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess


LOCAL = Path(__file__).resolve().parent
AUDIT = LOCAL.parents[2]
spec = importlib.util.spec_from_file_location("original_stage_environment_entry", AUDIT / "stage_environment_entry.py")
stage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage)
NATIVE = stage.ROOT + "/receipts/textcraft-native-minibatch-20261006-v4"
OUT = stage.ROOT + "/receipts/textcraft-native-readout-20261006-v2"
ANALYZER = "analyze_saved_native_b4_denominators.py"
RESULT = "native-b4-denominator-analysis.json"
RECEIPT = "native-b4-denominator-cpu-run.json"
LOG = "native-b4-denominator-cpu.stdout.txt"


SCRIPT = r'''/opt/conda/bin/python - <<'PY'
import hashlib,json,pathlib,psutil,subprocess,time
native=pathlib.Path('__NATIVE__'); out=pathlib.Path('__OUT__'); expected='__HASH__'
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
env['PYTHONPATH']=':'.join([str(out),str(native),source_dirs['entry'],source_dirs['verl'],source_dirs['dt']])
script=out/'__ANALYZER__'
assert hashlib.sha256(script.read_bytes()).hexdigest()==expected
assert (native/'native-minibatch-completed.json').is_file()
started=time.time(); before=psutil.virtual_memory().available
argv=[env['VENV_PYTHON'],str(script),str(native),str(out/'__RESULT__')]
with (out/'__LOG__').open('wb') as stream:
    result=subprocess.run(argv,cwd=out,env=env,stdout=stream,stderr=subprocess.STDOUT)
receipt={
    'scope':'CPU saved original DataProto analysis only; no model, forward, DT, backward or state update',
    'argv':argv,'analyzer_sha256':expected,'wrapper_pid':psutil.Process().pid,
    'wrapper_pid_birth':psutil.Process().create_time(),
    'reused_environment_pid':parent.pid,'reused_environment_pid_birth':parent.create_time(),
    'cuda_visible_devices':'','pythonpath':env['PYTHONPATH'],
    'started_unix':started,'completed_unix':time.time(),'exit_code':result.returncode,
    'host_available_before_bytes':before,'host_available_after_bytes':psutil.virtual_memory().available,
    'stdout':str(out/'__LOG__'),'result':str(out/'__RESULT__'),
    'source_prepared_sha256':hashlib.sha256((native/'prepared-diagnostic.json').read_bytes()).hexdigest(),
    'source_job_sha256':hashlib.sha256((native/'job.json').read_bytes()).hexdigest()}
(out/'__RECEIPT__').write_text(json.dumps(receipt,indent=2)+'\n')
print((out/'__LOG__').read_text()[-8000:])
print(json.dumps(receipt))
raise SystemExit(result.returncode)
PY
'''


def main():
    path = LOCAL / ANALYZER
    ast.parse(path.read_text(encoding="utf-8"))
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    subprocess.run(stage.SCP + [str(path), f"{stage.SSH[-1]}:{OUT}/{ANALYZER}"], check=True)
    script = SCRIPT
    for key, value in {"NATIVE": NATIVE, "OUT": OUT, "HASH": expected, "ANALYZER": ANALYZER,
                       "RESULT": RESULT, "RECEIPT": RECEIPT, "LOG": LOG}.items():
        script = script.replace("__" + key + "__", value)
    result = subprocess.run(stage.SSH + ["bash", "-s"], input=script.encode(), capture_output=True)
    (LOCAL / "native-b4-denominator-ssh.stdout.txt").write_bytes(result.stdout + result.stderr)
    print(result.stdout.decode(errors="replace"))
    print(result.stderr.decode(errors="replace"))
    for name in (RESULT, RECEIPT, LOG):
        subprocess.run(stage.SCP + [f"{stage.SSH[-1]}:{OUT}/{name}", str(LOCAL / name)], check=True)
    receipt = json.loads((LOCAL / RECEIPT).read_bytes())
    receipt["local_launcher"] = {"path": str(Path(__file__).resolve()),
                                 "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    # Separate local provenance; preserve the exact downloaded run receipt.
    (LOCAL / "native-b4-denominator-local-fetch.json").write_text(json.dumps({
        "scope": "Exact downloaded files; local runner binding is separate from the original remote receipt",
        "launcher": receipt["local_launcher"], "files": [{"path": str(LOCAL / name),
            "sha256": hashlib.sha256((LOCAL / name).read_bytes()).hexdigest(),
            "bytes": (LOCAL / name).stat().st_size} for name in (RESULT, RECEIPT, LOG)]}, indent=2) + "\n")
    result.check_returncode()


if __name__ == "__main__":
    main()

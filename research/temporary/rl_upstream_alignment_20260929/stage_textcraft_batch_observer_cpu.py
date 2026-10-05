"""Verify an isolated observation seam with the provisioned original CPU owners.

No model, environment episode, DT attribution, CUDA work or optimizer is run.
The copied input is the saved actual checkpoint25/step26 token-ID receipt.
"""
import hashlib
import json
import subprocess
import tarfile

from stage_environment_entry import AUDIT, ROOT, SSH, SCP

OUT = ROOT + '/receipts/textcraft-native-batch-observer-20261005-v1'
FILES = ['observe_textcraft_native_batches.py', 'test_observe_textcraft_native_batches.py']

SCRIPT = r'''/opt/conda/bin/python - <<'PY'
import ast,hashlib,json,os,pathlib,psutil,subprocess,time
root=pathlib.Path('__ROOT__');out=pathlib.Path('__OUT__')
assert not (out/'cpu-test-result.json').exists(), 'Read the completed test rather than repeating it'
subprocess.run(['tar','-xf',str(out/'source.tar'),'-C',str(out)],check=True)
sources={}
for name,sha in __HASHES__.items():
 p=out/name;actual=hashlib.sha256(p.read_bytes()).hexdigest()
 assert actual==sha,(name,actual,sha)
 if p.suffix=='.py':ast.parse(p.read_bytes())
 sources[str(p)]=actual
old_entry=root/'candidates/textcraft-rollout-scope-20261001/entry'
manifest=json.loads((out/'textcraft-degradation-20261005/source.json').read_bytes())
old_verl=pathlib.Path(manifest['verl_root']);old_dt=pathlib.Path(manifest['dt_root'])
protocol=old_verl/'verl/protocol.py'
assert hashlib.sha256(protocol.read_bytes()).hexdigest()=='2ae51f003f72d6ad0f288d5e2d8e94ad172a69422239a8612ba6d1d9dffeb4aa'
formal=psutil.Process(1856052)
assert abs(formal.create_time()-1791197944.7)<.05
env=formal.environ();env.pop('RAY_ADDRESS',None)
tail=env['PYTHONPATH']
tail=tail.replace(str(root/'candidates/appworld-eval-client-routing-resume-20261005-v1/entry'),str(old_entry))
tail=tail.replace(str(root/'candidates/appworld-native-prefix-resume-20261005-v2/verl'),str(old_verl))
tail=tail.replace(str(root/'candidates/appworld-native-prefix-resume-20261005-v2/deltatrace'),str(old_dt))
env.update(PYTHONPATH=str(out)+':'+tail,CUDA_VISIBLE_DEVICES='',VERL_ROOT=str(old_verl),DT_ROOT=str(old_dt),DT_TASK='TextCraft')
started=time.time()
cmd=[env['VENV_PYTHON'],'-m','pytest',str(out/'test_observe_textcraft_native_batches.py'),'-q','--junitxml='+str(out/'cpu-tests.xml')]
with (out/'cpu-test.log').open('wb') as log:
 result=subprocess.run(cmd,cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT)
receipt=dict(role='Real original DataProto CPU save/roundtrip observation tests; no model, DT, environment episode or optimizer',
 sources=sources,original_protocol=dict(path=str(protocol),sha256=hashlib.sha256(protocol.read_bytes()).hexdigest()),
 python=env['VENV_PYTHON'],command=cmd,started_unix=started,finished_unix=time.time(),exit_code=result.returncode,
 reused_environment_pid=formal.pid,reused_environment_pid_birth=formal.create_time(),CUDA_VISIBLE_DEVICES='',
 log=str(out/'cpu-test.log'),junit=str(out/'cpu-tests.xml'))
(out/'cpu-test-result.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt));print((out/'cpu-test.log').read_text()[-3500:])
raise SystemExit(result.returncode)
PY'''

if __name__ == '__main__':
    hashes = {name: hashlib.sha256((AUDIT / name).read_bytes()).hexdigest() for name in FILES}
    for name in ['checkpoint-matched-records.json', 'source.json']:
        relative = 'textcraft-degradation-20261005/' + name
        hashes[relative] = hashlib.sha256((AUDIT / relative).read_bytes()).hexdigest()
    archive = AUDIT / 'textcraft-degradation-20261005/native-batch-observer-source-v1.tar'
    with tarfile.open(archive, 'w') as tar:
        for relative in hashes:
            tar.add(AUDIT / relative, arcname=relative)
    subprocess.run(SSH + ['mkdir', '-p', OUT], check=True)
    subprocess.run(SCP + [str(archive), f'{SSH[-1]}:{OUT}/source.tar'], check=True)
    script = SCRIPT.replace('__ROOT__', ROOT).replace('__OUT__', OUT).replace('__HASHES__', repr(hashes))
    result = subprocess.run(SSH + ['bash', '-s'], input=script.encode(), capture_output=True)
    (AUDIT / 'textcraft-degradation-20261005/native-batch-observer-cpu.stdout.txt').write_bytes(result.stdout)
    print(result.stdout.decode(errors='replace'))
    print(result.stderr.decode(errors='replace'))
    result.check_returncode()

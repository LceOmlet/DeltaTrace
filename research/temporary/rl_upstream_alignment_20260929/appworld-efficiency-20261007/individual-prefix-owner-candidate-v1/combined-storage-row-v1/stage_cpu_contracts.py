"""Stage only isolated sources and execute original CPU owner interfaces."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile

HERE = Path(__file__).resolve().parent
AUDIT = HERE.parents[2]
REPO = next(p for p in HERE.parents if (p/'experiments/rl/PLAN.md').exists())
sys.path.insert(0, str(AUDIT))
from stage_environment_entry import ROOT, SCP, SSH

OUT = ROOT+'/candidates/appworld-row-cuts-finite-20261007-v1/combined-storage-row-v1'
ROW = ROOT+'/candidates/appworld-row-cuts-finite-20261007-v1/native-representation-candidate'
ENV = ROOT+'/candidates/appworld-row-cuts-finite-20261007-v1/real-b8-v3/run-env.json'

REMOTE = r"""
import hashlib,json,os,pathlib,subprocess,tarfile,time
out=pathlib.Path(@OUT@)
def sha(path):return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
with tarfile.open(out/'cpu-sources.tar') as archive:
 for item in archive.getmembers():
  assert item.isfile() and not pathlib.PurePosixPath(item.name).is_absolute() and '..' not in pathlib.PurePosixPath(item.name).parts
  path=out/item.name;path.parent.mkdir(exist_ok=True)
  assert not path.exists(),path
  path.write_bytes(archive.extractfile(item).read())
for name,expected in @SHAS@.items():assert sha(out/name)==expected,name
assert sha(pathlib.Path(@ROW@)/'check_contracts.py')=='3bf4fac86eb4657f427e1fdcb210926501fe7073607489e26d95b7eb129cfc93'
env_path=pathlib.Path(@ENV@);env=json.loads(env_path.read_bytes())
env.update(CUDA_VISIBLE_DEVICES='',MACA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
env.pop('RAY_ADDRESS',None)
python=env['VENV_PYTHON']
assert python=='/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_qwen35_20260912/env/bin/python'
command=[python,str(out/'check_contracts.py'),'--torch-interface',
 '--row-contracts',str(pathlib.Path(@ROW@)/'check_contracts.py'),
 '--storage-tests',str(out/'test_native_prefix_boundary_rows.py'),
 '--geometry',str(out/'results_actual_prefix_request_geometry_20261004.json'),
 '--output',str(out/'cpu-contracts.json')]
wrapper=r'''
import hashlib,json,os,pathlib,psutil,runpy,sys,time,traceback
out=pathlib.Path(sys.argv[1]);argv=json.loads(sys.argv[2]);process=psutil.Process()
started=time.time();before=process.memory_full_info();error=None
sys.argv=argv[1:]
try:runpy.run_path(argv[1],run_name='__main__')
except BaseException:
 error=traceback.format_exc();raise
finally:
 import resource
 torch=sys.modules.get('torch');after=process.memory_full_info()
 data=dict(scope='CPU owner interface/actual literal-ID dispatch only; no model, capture, attention, DT, optimizer, deployment or GPU operations.',
  pid=os.getpid(),pid_birth=process.create_time(),started_unix=started,completed_unix=time.time(),
  elapsed_seconds=time.time()-started,pss_before_bytes=before.pss,pss_after_bytes=after.pss,
  rss_after_bytes=after.rss,max_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
  CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),MACA_VISIBLE_DEVICES=os.environ.get('MACA_VISIBLE_DEVICES'),
  CUDA_initialized=None if torch is None else torch.cuda.is_initialized(),
  torch_version=None if torch is None else torch.__version__,error=error)
 with (out/'cpu-execution.json').open('x') as stream:json.dump(data,stream,indent=2);stream.write('\n')
'''
with (out/'cpu.stdout.txt').open('xb') as stdout,(out/'cpu.stderr.txt').open('xb') as stderr:
 result=subprocess.run([python,'-c',wrapper,str(out),json.dumps(command)],env=env,stdout=stdout,stderr=stderr)
record=dict(scope='Isolated CPU-only stage/execution; all original sources/real-b8-v3/formal paths unchanged.',
 directory=str(out),returncode=result.returncode,source_sha256=@SHAS@,
 existing_row_contracts=dict(path=str(pathlib.Path(@ROW@)/'check_contracts.py'),sha256=sha(pathlib.Path(@ROW@)/'check_contracts.py')),
 reused_environment=dict(path=str(env_path),sha256=sha(env_path),values_printed=False),
 command=command,observed_unix=time.time(),
 outputs={name:dict(path=str(out/name),sha256=sha(out/name),bytes=(out/name).stat().st_size)
          for name in ('cpu-contracts.json','cpu-execution.json','cpu.stdout.txt','cpu.stderr.txt') if (out/name).exists()})
(out/'cpu-stage-execution.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
raise SystemExit(result.returncode)
"""


def main():
    files = {name: HERE/name for name in ('check_contracts.py','source-manifest.json',
        'candidate/qwen35_native_prefix_artifacts.py','candidate/native_prefix_leases.py',
        'baseline/qwen35_native_prefix_artifacts.py','baseline/native_prefix_leases.py')}
    files.update({'test_native_prefix_boundary_rows.py': REPO/'experiments/rl/test_native_prefix_boundary_rows.py',
        'results_actual_prefix_request_geometry_20261004.json': REPO/'experiments/rl/results_actual_prefix_request_geometry_20261004.json'})
    hashes = {name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in files.items()}
    assert hashes['candidate/qwen35_native_prefix_artifacts.py']=='37a86074f430efd837c878b5409ce22ff3aac5ebdc984936ea5be5647d7b97d4'
    assert hashes['candidate/native_prefix_leases.py']=='b94756147cc6f8e59fb39baa1c2737aa655b0d9c6b87091969a65c9071ad0852'
    archive=HERE/'cpu-sources.tar'
    with tarfile.open(archive,'w') as tar:
        for name,path in files.items():tar.add(path,arcname=name)
    subprocess.run(SSH+['bash','-s'],input=('set -eu\ntest ! -e '+shlex.quote(OUT)+'\nmkdir '+shlex.quote(OUT)+'\n').encode(),check=True)
    subprocess.run(SCP+[str(archive),SSH[-1]+':'+OUT+'/cpu-sources.tar'],check=True)
    script=REMOTE.replace('@OUT@',repr(OUT)).replace('@ROW@',repr(ROW)).replace('@ENV@',repr(ENV)).replace('@SHAS@',repr(hashes))
    compile(script,'remote CPU staging','exec')
    result=subprocess.run(SSH+['/usr/bin/python3','-'],input=script.encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    (HERE/'cpu-stage.stdout.json').write_bytes(result.stdout)
    (HERE/'cpu-stage.stderr.txt').write_bytes(result.stderr)
    for name in ('cpu-contracts.json','cpu-execution.json','cpu-stage-execution.json','cpu.stdout.txt','cpu.stderr.txt'):
        subprocess.run(SCP+[SSH[-1]+':'+OUT+'/'+name,str(HERE/name)],check=False)
    print(result.stdout.decode('utf-8',errors='replace'))
    if result.returncode:print(result.stderr.decode('utf-8',errors='replace'),file=sys.stderr)
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()

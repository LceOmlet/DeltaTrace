"""Use the existing saved-operand FA verifier, original reference and assertions.

This one primitive check is not a development-set attribution-quality result.
"""
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
AUDIT=HERE.parents[2]
REPO=AUDIT.parents[2]
sys.path.insert(0,str(AUDIT))
from stage_environment_entry import ROOT,ENTRY,SSH


def main():
    compiled=json.loads((HERE/'compiled-owner.json').read_bytes())
    assert compiled['returncode']==0
    files={name:base64.b64encode((REPO/'experiments/rl'/name).read_bytes()).decode()
           for name in ('verify_saved_fa_dtypes.py','verify_official_kernel_tolerances.py')}
    version=hashlib.sha256((REPO/'experiments/rl/verify_saved_fa_dtypes.py').read_bytes()+compiled['library_sha256'].encode()).hexdigest()[:12]
    dest=ROOT+'/receipts/research-conditional-fa-original-check-20261008-v1-'+version
    script=f'''set -eu
source {ENTRY}/metax-entry.env.sh
CUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<'PY'
FILES={files!r}
COMPILED={compiled!r}
DEST={dest!r}
ROOT={ROOT!r}
import ast,base64,hashlib,json,os,re,subprocess,time
from pathlib import Path
import psutil
root=Path(ROOT);dest=Path(DEST);dest.mkdir(parents=True,exist_ok=True)
assert not (dest/'launch.json').exists(),'Do not duplicate this primitive check'
candidate=Path(COMPILED['remote_directory'])
library=candidate/'libfinite_conditional_research.so'
assert hashlib.sha256(library.read_bytes()).hexdigest()==COMPILED['library_sha256']
for name,encoded in FILES.items():
 raw=base64.b64decode(encoded);ast.parse(raw.decode())
 (dest/name).write_bytes(raw)
source_path=root/'runs/direct-target-prefix-runtime-20261007-v1/textcraft/textcraft-dt/source.json'
assert hashlib.sha256(source_path.read_bytes()).hexdigest()=='2796233e2683f1939896c74b2b578c242dbd7a7f235b9ef61cbedd398f61be52'
source=json.loads(source_path.read_bytes())
physical=subprocess.run(['mx-smi'],capture_output=True,text=True,check=True).stdout
assert not re.search(r'^\\|\\s*4\\s+\\d+\\s+\\S',physical,re.M),'GPU4 is occupied'
env=dict(os.environ,**source['environment']);env.pop('MACA_VISIBLE_DEVICES',None)
env['CUDA_VISIBLE_DEVICES']='4'
env['PYTHONPATH']=':'.join([str(candidate),str(dest),source['pythonpath'],str(Path(source['dt_root'])/'clean/qwen35')])
argv=[env['VENV_PYTHON'],'-u',str(dest/'verify_saved_fa_dtypes.py'),
 '--operands',str(root/'receipts/upstream-alignment-20260929/actual-dt-layer3-boundaries.pt'),
 '--sources',str(root/'receipts/training-setup/official-kernel-tests'),
 '--output',str(dest/'original-fa-check.json'),'--finite-rule','conditional',
 '--finite-library',str(library),'--finite-library-sha256',COMPILED['library_sha256']]
started=time.time();peak=0;timed_out=False
with (dest/'driver.log').open('xb') as log:
 process=subprocess.Popen(argv,env=env,cwd=dest,stdout=log,stderr=subprocess.STDOUT)
 birth=psutil.Process(process.pid).create_time()
 launch=dict(pid=process.pid,birth=birth,started_unix=started,argv=argv,device=4,
  source_sha256={{name:hashlib.sha256((dest/name).read_bytes()).hexdigest() for name in FILES}},
  candidate=COMPILED['source_sha256'],library_sha256=COMPILED['library_sha256'],
  physical_before=physical,model_calls=0,production_modified=False)
 (dest/'launch.json').write_text(json.dumps(launch,indent=2)+'\\n')
 print(json.dumps(dict(phase='launched_original_primitive_check',pid=process.pid,birth=birth)),flush=True)
 while process.poll() is None:
  try:
   parent=psutil.Process(process.pid)
   total=0
   for child in [parent,*parent.children(recursive=True)]:
    try:total+=child.memory_full_info().pss
    except (psutil.NoSuchProcess,psutil.AccessDenied):pass
   peak=max(peak,total)
  except psutil.NoSuchProcess:pass
  if time.time()-started>180:
   timed_out=True;process.terminate();break
  time.sleep(.25)
 code=process.wait()
result=dict(launch=launch,returncode=code,timed_out=timed_out,
 elapsed_seconds=time.time()-started,sampled_peak_tree_PSS_bytes=peak,
 physical_after=subprocess.run(['mx-smi'],capture_output=True,text=True,check=True).stdout,
 driver_log=(dest/'driver.log').read_text(errors='replace'),
 numerical=json.loads((dest/'original-fa-check.json').read_bytes()) if (dest/'original-fa-check.json').exists() else None,
 remote_directory=str(dest))
(dest/'result.json').write_text(json.dumps(result,indent=2)+'\\n')
print(json.dumps(result))
PY
'''
    (HERE/'original-fa-command.sh').write_bytes(script.encode())
    run=subprocess.run(SSH+['bash','-s'],input=script.encode(),capture_output=True)
    (HERE/'original-fa.stderr.txt').write_bytes(run.stderr)
    (HERE/'original-fa.stdout.txt').write_bytes(run.stdout)
    run.check_returncode()
    result=json.loads(run.stdout.splitlines()[-1])
    (HERE/'original-fa-check.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('launch','physical_after','driver_log')},ensure_ascii=False))
    if result['returncode'] or result['numerical'] is None:print(result['driver_log'])


if __name__=='__main__':main()

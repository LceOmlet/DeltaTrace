"""Compile an isolated owner extension using the recorded MetaX compiler.

No model import, numerical query, environment setup, cache reset or production
file change. Compilation resources and the exact source are recorded together.
"""
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
AUDIT=HERE.parents[2]
sys.path.insert(0,str(AUDIT))
from stage_environment_entry import ROOT,ENTRY,SSH


def main():
    files={name:base64.b64encode(path.read_bytes()).decode() for name,path in {
        'vendor_fa_finite_p1_bf16_d256.cu':HERE/'prepared/vendor_fa_finite_p1_bf16_d256.cu',
        'vendor_fa_finite_bf16_d256.py':HERE/'prepared/vendor_fa_finite_bf16_d256.py',
        'conditional_attention_rows.cuh':HERE/'conditional_attention_rows.cuh',
        'owner.patch':HERE/'owner.patch','prepared-owner.json':HERE/'prepared-owner.json',
    }.items()}
    compile_owner=json.loads((AUDIT/'appworld-efficiency-20261007/individual-prefix-owner-candidate-v1/compiled-owner.json').read_bytes())
    version=hashlib.sha256((HERE/'prepared-owner.json').read_bytes()).hexdigest()[:12]
    candidate=ROOT+'/candidates/research-conditional-attention-20261008-v1-'+version
    script=f'''set -eu
source {ENTRY}/metax-entry.env.sh
CUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<'PY'
FILES={files!r}
COMPILE_OWNER={compile_owner!r}
DEST={candidate!r}
ROOT={ROOT!r}
import ast,base64,hashlib,json,os,subprocess,time
from pathlib import Path
import psutil
dest=Path(DEST);dest.mkdir(parents=True,exist_ok=True)
started=time.time()
for name,encoded in FILES.items():
 raw=base64.b64decode(encoded)
 path=dest/name
 if path.exists():
  assert path.read_bytes()==raw,(str(path),'different prepared version; preserve it')
 else:path.write_bytes(raw)
original=Path(ROOT+'/candidates/appworld-row-cuts-finite-20261007-v1/vendor_fa_finite_p1_bf16_d256.cu')
assert hashlib.sha256(original.read_bytes()).hexdigest()=='9ebcef18cec94f45a875694acac0fed4b46064f13d3d0af1a4581c931df54c8b'
ast.parse((dest/'vendor_fa_finite_bf16_d256.py').read_text())
library=dest/'libfinite_conditional_research.so'
assert not library.exists(),'Do not silently repeat a completed isolated compilation'
command=COMPILE_OWNER['command'].copy()
command[-3:]=[str(dest/'vendor_fa_finite_p1_bf16_d256.cu'),'-o',str(library)]
log=dest/'compile.log'
before=psutil.virtual_memory().available
peak=0;timed_out=False
with log.open('wb') as stream:
 process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,cwd=dest)
 while process.poll() is None:
  total=0
  try:
   parent=psutil.Process(process.pid)
   for child in [parent,*parent.children(recursive=True)]:
    try:total+=child.memory_full_info().pss
    except (psutil.NoSuchProcess,psutil.AccessDenied):pass
  except psutil.NoSuchProcess:pass
  peak=max(peak,total)
  if time.time()-started>180:
   timed_out=True
   try:
    parent=psutil.Process(process.pid)
    for child in parent.children(recursive=True):child.terminate()
   except psutil.NoSuchProcess:pass
   process.terminate()
   break
  time.sleep(.2)
 code=process.wait()
hold=Path(ROOT+'/receipts/direct-target-prefix-runtime-20261007-v1/textcraft-first-dt')
birth=psutil.Process(2833207).create_time()
assert birth==1791370325.16
release=[(hold/('rank'+str(r)+'-release-update')).exists() for r in (0,1)]
assert not any(release)
result=dict(status='Isolated compilation only' if code==0 else 'Unaccepted compilation failed',
 started_unix=started,finished_unix=time.time(),returncode=code,timed_out=timed_out,
 command=command,sampled_peak_tree_PSS_bytes=peak,
 available_host_before=before,available_host_after=psutil.virtual_memory().available,
 source_sha256={{name:hashlib.sha256((dest/name).read_bytes()).hexdigest() for name in FILES}},
 library_sha256=hashlib.sha256(library.read_bytes()).hexdigest() if library.exists() else None,
 log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),
 log_tail=log.read_text(errors='replace')[-16000:],
 formal_textcraft=dict(pid=2833207,birth=birth,release_exists=release),
 model_calls=0,GPU_calls=0,production_modified=False)
(dest/'compiled-owner.json').write_text(json.dumps(result,indent=2)+'\\n')
print(json.dumps(result))
PY
'''
    (HERE/'compile-command.sh').write_bytes(script.encode())
    completed=subprocess.run(SSH+['bash','-s'],input=script.encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    (HERE/'compile.stderr.txt').write_bytes(completed.stderr)
    (HERE/'compile.stdout.txt').write_bytes(completed.stdout)
    completed.check_returncode()
    result=json.loads(completed.stdout)
    result['local_launcher_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result['remote_directory']=candidate
    (HERE/'compiled-owner.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('command','log_tail','source_sha256')},ensure_ascii=False))
    if result['returncode']!=0:print(result['log_tail'])


if __name__=='__main__':main()

"""Bounded research diagnostic using the existing remote runtime and owner.

No environment configuration, model load, deployment or original receipt edit.
"""
import base64
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
AUDIT=HERE.parents[2]
REPO=HERE.parents[5]
sys.path.insert(0,str(AUDIT))
from stage_environment_entry import ROOT,ENTRY,SSH


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--composition',action='store_true')
    parser.add_argument('--official-derivative',action='store_true')
    args=parser.parse_args()
    compiled=json.loads((HERE/'compiled-owner.json').read_bytes())
    assert compiled['returncode']==0
    worker=REPO/'experiments/rl/verify_saved_fa_dtypes.py' if args.official_derivative else HERE/'verify_conditional_finite_local.py'
    files={path.name:base64.b64encode(path.read_bytes()).decode() for path in
        [worker,REPO/'experiments/rl/verify_official_kernel_tolerances.py']}
    label='composition-derivative' if args.official_derivative else ('composition-local' if args.composition else 'nonzero-local')
    if args.composition or args.official_derivative:
        prepared=json.loads((HERE/'composition-prepared.json').read_bytes())
        helper=HERE/'conditional_attention_endpoints.py'
        assert hashlib.sha256(helper.read_bytes()).hexdigest()==prepared['helper_sha256']
        decoder=HERE/'composition-prepared/textcraft/qwen35_decoder_finite.py'
        assert hashlib.sha256(decoder.read_bytes()).hexdigest()==prepared['tasks']['textcraft'][0]['generated_sha256']
        for path in [helper,decoder]:files[path.name]=base64.b64encode(path.read_bytes()).decode()
    version=hashlib.sha256(json.dumps(files,sort_keys=True).encode()+compiled['library_sha256'].encode()+label.encode()).hexdigest()[:12]
    dest=ROOT+'/receipts/research-conditional-'+label+'-20261008-v1-'+version
    extra_args=(['--finite-rule','conditional','--conditional-composition'] if args.official_derivative
                else (['--composition'] if args.composition else []))
    script=f'''set -eu
source {ENTRY}/metax-entry.env.sh
CUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<'PY'
FILES={files!r}
COMPILED={compiled!r}
DEST={dest!r}
ROOT={ROOT!r}
EXTRA={extra_args!r}
WORKER={worker.name!r}
import ast,base64,hashlib,json,os,re,subprocess,time
from pathlib import Path
import psutil
root=Path(ROOT);dest=Path(DEST);dest.mkdir(parents=True,exist_ok=True)
assert not (dest/'launch.json').exists(),'Do not duplicate this bounded diagnostic'
candidate=Path(COMPILED['remote_directory']);library=candidate/'libfinite_conditional_research.so'
assert hashlib.sha256(library.read_bytes()).hexdigest()==COMPILED['library_sha256']
for name,encoded in FILES.items():
 raw=base64.b64decode(encoded);ast.parse(raw.decode());(dest/name).write_bytes(raw)
source_path=root/'runs/direct-target-prefix-runtime-20261007-v1/textcraft/textcraft-dt/source.json'
assert hashlib.sha256(source_path.read_bytes()).hexdigest()=='2796233e2683f1939896c74b2b578c242dbd7a7f235b9ef61cbedd398f61be52'
source=json.loads(source_path.read_bytes())
physical=subprocess.run(['mx-smi'],capture_output=True,text=True,check=True).stdout
assert not re.search(r'^\\|\\s*4\\s+\\d+\\s+\\S',physical,re.M),'GPU4 is occupied'
env=dict(os.environ,**source['environment']);env.pop('MACA_VISIBLE_DEVICES',None)
env['CUDA_VISIBLE_DEVICES']='4'
env['PYTHONPATH']=':'.join([str(dest),str(candidate),source['pythonpath'],str(Path(source['dt_root'])/'clean/qwen35')])
argv=[env['VENV_PYTHON'],'-u',str(dest/WORKER),
 '--operands',str(root/'receipts/upstream-alignment-20260929/actual-dt-layer3-boundaries.pt'),
 '--sources',str(root/'receipts/training-setup/official-kernel-tests'),
 '--output',str(dest/'nonzero-local.json'),'--finite-library',str(library),
 '--finite-library-sha256',COMPILED['library_sha256']]+EXTRA
started=time.time();peak=0;timed_out=False
with (dest/'driver.log').open('xb') as log:
 process=subprocess.Popen(argv,env=env,cwd=dest,stdout=log,stderr=subprocess.STDOUT)
 birth=psutil.Process(process.pid).create_time()
 launch=dict(pid=process.pid,birth=birth,started_unix=started,argv=argv,device=4,
  source_sha256={{name:hashlib.sha256((dest/name).read_bytes()).hexdigest() for name in FILES}},
  library_sha256=COMPILED['library_sha256'],physical_before=physical,model_calls=0,production_modified=False)
 (dest/'launch.json').write_text(json.dumps(launch,indent=2)+'\\n')
 while process.poll() is None:
  try:
   parent=psutil.Process(process.pid);total=0
   for child in [parent,*parent.children(recursive=True)]:
    try:total+=child.memory_full_info().pss
    except (psutil.NoSuchProcess,psutil.AccessDenied):pass
   peak=max(peak,total)
  except psutil.NoSuchProcess:pass
  if time.time()-started>180:
   timed_out=True;process.terminate();break
  time.sleep(.25)
 code=process.wait()
hold=root/'receipts/direct-target-prefix-runtime-20261007-v1/textcraft-first-dt'
formal=dict(pid=2833207,birth=psutil.Process(2833207).create_time(),
 release_exists=[(hold/('rank'+str(i)+'-release-update')).exists() for i in (0,1)])
assert formal['birth']==1791370325.16 and formal['release_exists']==[False,False]
result=dict(launch=launch,returncode=code,timed_out=timed_out,elapsed_seconds=time.time()-started,
 sampled_peak_tree_PSS_bytes=peak,formal_textcraft=formal,
 physical_after=subprocess.run(['mx-smi'],capture_output=True,text=True,check=True).stdout,
 driver_log=(dest/'driver.log').read_text(errors='replace'),
 numerical=json.loads((dest/'nonzero-local.json').read_bytes()) if (dest/'nonzero-local.json').exists() else None,
 remote_directory=str(dest))
(dest/'result.json').write_text(json.dumps(result,indent=2)+'\\n')
print(json.dumps(result))
PY
'''
    (HERE/(label+'-command.sh')).write_bytes(script.encode())
    run=subprocess.run(SSH+['bash','-s'],input=script.encode(),capture_output=True)
    (HERE/(label+'.stdout.txt')).write_bytes(run.stdout)
    (HERE/(label+'.stderr.txt')).write_bytes(run.stderr)
    run.check_returncode()
    result=json.loads(run.stdout.splitlines()[-1])
    (HERE/(label+'.json')).write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    compact={k:v for k,v in result.items() if k not in ('launch','physical_after','driver_log','numerical')}
    compact['numerical']={k:v for k,v in (result['numerical'] or {}).items() if k not in ('predicted','reference','error')}
    print(json.dumps(compact,ensure_ascii=False))
    if result['returncode']:print(result['driver_log'])


if __name__=='__main__':main()

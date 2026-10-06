"""Run already prepared FA checks independently of the failed GDN reference check."""
import json
from pathlib import Path
import subprocess
import sys

AUDIT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(AUDIT))
from stage_environment_entry import SSH, ROOT

REMOTE = r'''
import hashlib,json,os,re,subprocess,sys,time
from pathlib import Path
import psutil
p=Path(ROOT)/'candidates/appworld-row-cuts-finite-20261007-v1/real-b8-v3-official-checks-v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
prepared=json.loads((p/'prepared.json').read_bytes());s=prepared['sources']
physical=subprocess.check_output(['mx-smi'],text=True)
if re.search(r'^\|\s+2\s+\d+\s+',physical.split('| Process:')[-1],re.M):raise RuntimeError('GPU2 occupied')
for x in s.values():assert sha(x['path'])==x['sha256'],x
run=p/('independent-fa-'+str(time.time_ns()));run.mkdir()
receipt=dict(scope='Existing FA commands only; GDN failure remains separate and unresolved',pid=os.getpid(),birth=psutil.Process().create_time(),started=time.time(),checks=[],status='running',prepared_sha256=sha(p/'prepared.json'),physical_gpu=physical,checkpoint=False,model=False,optimizer=False)
def save(): (run/'execution.json').write_text(json.dumps(receipt,indent=2)+'\n')
save();print(json.dumps(dict(execution=str(run/'execution.json'),pid=receipt['pid'])),flush=True)
try:
 for rank in (0,1):
  item=prepared['rank_records'][str(rank)];assert sha(item['path'])==item['sha256']
  r=json.loads(Path(item['path']).read_bytes());target=run/f'rank{rank}';target.mkdir()
  for key in ('native_fa3','actual_finite'):assert sha(r[key]['path'])==r[key]['sha256']
  commands=[('varlen',[sys.executable,'-u',s['varlen_observer']['path'],'--operands',r['native_fa3']['path'],'--official-source',s['FA_tests']['path'],'--existing-checker',s['dense_observer']['path'],'--output',str(target/'original-varlen-output.json'),'--device','cuda']),
   ('finite',[sys.executable,'-u',s['actual_row_helper']['path'],'--native-operands',r['native_fa3']['path'],'--finite-operands',r['actual_finite']['path'],'--saved-verifier',s['saved_verifier']['path'],'--sources',str(Path(s['FA_tests']['path']).parent),'--row-wrapper',s['row_wrapper']['path'],'--row-library',s['row_library']['path'],'--scalar-wrapper',s['scalar_wrapper']['path'],'--scalar-library',s['scalar_library']['path'],'--environment-json',s['numerical_environment']['path'],'--output-dir',str(target/'original-finite-rows')])]
  for name,cmd in commands:
   check=dict(rank=rank,name=name,command=cmd,started=time.time());receipt['checks'].append(check);save()
   out=target/(name+'.stdout.txt');err=target/(name+'.stderr.txt')
   with out.open('xb') as o,err.open('xb') as e:
    child=subprocess.Popen(cmd,stdout=o,stderr=e,cwd=p);check.update(pid=child.pid,birth=psutil.Process(child.pid).create_time());save();code=child.wait()
   check.update(returncode=code,finished=time.time(),stdout_sha256=sha(out),stderr_sha256=sha(err),stdout=str(out),stderr=str(err));save()
   print(json.dumps(check),flush=True)
   if code:raise RuntimeError('Original FA command failed; preserved stderr '+str(err))
 receipt['status']='passed'
except BaseException:
 receipt['status']='failed';raise
finally:receipt['finished']=time.time();save()
'''

if __name__ == '__main__':
    prefix = f'source {ROOT}/receipts/environment-only-20260930/entry/metax-entry.env.sh\nunset MACA_VISIBLE_DEVICES\nCUDA_VISIBLE_DEVICES=2 "$VENV_PYTHON" -u - <<\'PYREMOTE\'\n'
    script = prefix + 'ROOT=' + repr(ROOT) + '\n' + REMOTE + '\nPYREMOTE\n'
    result = subprocess.run(SSH + ['bash', '-s'], input=script.encode(), capture_output=True)
    directory = Path(__file__).parent / 'offline-checks-v1'
    (directory / 'independent-fa-execution.stdout.txt').write_bytes(result.stdout)
    (directory / 'independent-fa-execution.stderr.txt').write_bytes(result.stderr)
    print(result.stdout.decode(errors='replace')[-20000:])
    print(result.stderr.decode(errors='replace')[-6000:])
    raise SystemExit(result.returncode)

"""Run one bounded CPU-only inspection of already saved FA operands."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from stage_environment_entry import ROOT, SSH, SCP


def main():
    repo=HERE.parents[4]
    source=HERE/'inspect_saved_FA_source_support.py'
    relative=source.relative_to(repo).as_posix()
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    blob=subprocess.check_output(['git','show',f'{commit}:{relative}'],cwd=repo)
    assert blob==source.read_bytes()
    digest=hashlib.sha256(blob).hexdigest()
    target=ROOT+'/receipts/credit-FA-source-support-20261009-v2'
    original=ROOT+'/receipts/credit-attention-pv-textcraft-20261009-v2'
    python='/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_qwen35_20260912/env/bin/python'
    subprocess.run(SSH+['mkdir','-p',target],check=True)
    subprocess.run(SCP+[str(source),f'{SSH[-1]}:{target}/{source.name}'],check=True)
    command=['timeout','600','env','CUDA_VISIBLE_DEVICES=-1','OMP_NUM_THREADS=2','MKL_NUM_THREADS=2',
             python,'-u',target+'/'+source.name,'--directory',original,'--output',target+'/support.json']
    remote=f'''set -e
test "$(sha256sum {shlex.quote(target+'/'+source.name)} | cut -d' ' -f1)" = {shlex.quote(digest)}
test ! -e {shlex.quote(target+'/launch.json')}
CUDA_VISIBLE_DEVICES=-1 {shlex.quote(python)} - <<'PY'
import json,os,psutil,subprocess,time
from pathlib import Path
target=Path({target!r})
assert psutil.virtual_memory().available>8*(1<<30)
command={command!r}
log=(target/'driver.log').open('xb')
p=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record=dict(unix=time.time(),pid=p.pid,birth=psutil.Process(p.pid).create_time(),
    command=command,source_commit={commit!r},source_sha256={digest!r},
    original_directory={original!r},host_available_bytes=psutil.virtual_memory().available,
    operations=dict(model=0,FA=0,DT=0,GPU=0,optimizer=0,rollout=0),
    source_is_candidate=False,production_modified=False)
(target/'launch.json').write_text(json.dumps(record,indent=2)+'\\n')
print(json.dumps(record))
PY
'''
    (HERE/'FA-source-support-launch-command.sh').write_text(remote,encoding='utf-8',newline='\n')
    run=subprocess.run(SSH+['bash','-s'],input=remote.encode(),capture_output=True,check=True)
    (HERE/'FA-source-support-launch.stderr').write_bytes(run.stderr)
    (HERE/'FA-source-support-launch.json').write_bytes(run.stdout)
    print(run.stdout.decode())


if __name__=='__main__':
    main()

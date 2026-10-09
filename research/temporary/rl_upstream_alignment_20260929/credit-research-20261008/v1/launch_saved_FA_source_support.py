"""Run one bounded CPU-only inspection of already saved FA operands."""
import hashlib
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from stage_environment_entry import ROOT, SSH, SCP


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--key-regions',action='store_true',help='Original FA value-key partition readout on GPU4')
    args=parser.parse_args()
    repo=HERE.parents[4]
    source=HERE/('inspect_saved_FA_PV_key_regions.py' if args.key_regions else 'inspect_saved_FA_source_support.py')
    relative=source.relative_to(repo).as_posix()
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    blob=subprocess.check_output(['git','show',f'{commit}:{relative}'],cwd=repo)
    assert blob==source.read_bytes()
    digest=hashlib.sha256(blob).hexdigest()
    target=ROOT+('/receipts/credit-FA-PV-key-regions-20261009-v1' if args.key_regions else '/receipts/credit-FA-source-support-20261009-v2')
    original=ROOT+'/receipts/credit-attention-pv-textcraft-20261009-v2'
    python='/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_qwen35_20260912/env/bin/python'
    subprocess.run(SSH+['mkdir','-p',target],check=True)
    subprocess.run(SCP+[str(source),f'{SSH[-1]}:{target}/{source.name}'],check=True)
    command=['timeout','600','env','CUDA_VISIBLE_DEVICES='+('4' if args.key_regions else '-1'),'OMP_NUM_THREADS=2','MKL_NUM_THREADS=2',
             python,'-u',target+'/'+source.name,'--directory',original,'--output',target+'/support.json']
    remote=f'''set -e
test "$(sha256sum {shlex.quote(target+'/'+source.name)} | cut -d' ' -f1)" = {shlex.quote(digest)}
test ! -e {shlex.quote(target+'/launch.json')}
CUDA_VISIBLE_DEVICES=-1 {shlex.quote(python)} - <<'PY'
import json,os,psutil,subprocess,time,re
from pathlib import Path
target=Path({target!r})
assert psutil.virtual_memory().available>8*(1<<30)
physical=subprocess.run(['mx-smi'],capture_output=True,text=True,check=True).stdout if {args.key_regions!r} else None
if physical is not None:assert not re.search(r'^\|\s*4\s+\d+\s+\S',physical,re.M),'Research GPU4 occupied'
command={command!r}
log=(target/'driver.log').open('xb')
p=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record=dict(unix=time.time(),pid=p.pid,birth=psutil.Process(p.pid).create_time(),
    command=command,source_commit={commit!r},source_sha256={digest!r},
    original_directory={original!r},host_available_bytes=psutil.virtual_memory().available,
    operations=dict(model=0,expected_FA={480 if args.key_regions else 0},DT=0,GPU={1 if args.key_regions else 0},optimizer=0,rollout=0),
    physical_before=physical,
    source_is_candidate=False,production_modified=False)
(target/'launch.json').write_text(json.dumps(record,indent=2)+'\\n')
print(json.dumps(record))
PY
'''
    prefix='FA-key-regions' if args.key_regions else 'FA-source-support'
    (HERE/(prefix+'-launch-command.sh')).write_text(remote,encoding='utf-8',newline='\n')
    run=subprocess.run(SSH+['bash','-s'],input=remote.encode(),capture_output=True,check=True)
    (HERE/(prefix+'-launch.stderr')).write_bytes(run.stderr)
    (HERE/(prefix+'-launch.json')).write_bytes(run.stdout)
    record=json.loads(run.stdout)
    print(json.dumps({k:v for k,v in record.items() if k!='physical_before'}))


if __name__=='__main__':
    main()

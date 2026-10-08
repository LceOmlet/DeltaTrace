"""Read the isolated frozen comparison's real phases and resources."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from stage_environment_entry import ROOT,ENTRY,SSH


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',choices=('textcraft','appworld'),required=True)
    parser.add_argument('--version',default='v1')
    parser.add_argument('--candidate-kind',choices=('conditional_gdn','conditional_mixers'),default='conditional_gdn')
    args=parser.parse_args()
    namespace=args.candidate_kind.replace('_','-')+'-collection-'
    remote=ROOT+'/receipts/'+namespace+args.task+'-20261009-'+args.version
    folder=HERE/(namespace+args.version)
    body='''import json,psutil,subprocess,time
from pathlib import Path
out=Path(OUT);launch=json.loads((out/'launch.json').read_bytes())
same=False;processes=[]
try:
 p=psutil.Process(launch['pid']);same=p.create_time()==launch['birth']
 if same:
  for child in [p,*p.children(recursive=True)]:
   try:processes.append(dict(pid=child.pid,birth=child.create_time(),status=child.status(),cpu=child.cpu_times()._asdict(),memory=child.memory_full_info()._asdict(),cmd=child.cmdline()))
   except psutil.Error:pass
except psutil.NoSuchProcess:pass
records={};logs={}
for rank in range(2):
 p=out/'results'/('rank'+str(rank)+'.json')
 if p.exists():records[str(rank)]=json.loads(p.read_bytes())
 p=out/'results'/('rank'+str(rank)+'-phases.jsonl')
 if p.exists():logs[str(rank)]=p.read_text()[-7000:]
for p in (out/'results').glob('gdn-context-pid*.jsonl'):
 logs[p.name]=p.read_text()[-9000:]
r=dict(unix=time.time(),launch=launch,driver_alive=same,processes=processes,records=records,phase_logs=logs,
 physical=subprocess.run(['mx-smi'],capture_output=True,text=True).stdout,
 host=psutil.virtual_memory()._asdict(),log=(out/'driver.log').read_text()[-10000:])
print(json.dumps(r))
'''
    shell='source '+ENTRY+'/metax-entry.env.sh\nCUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<\'PY\'\nOUT='+repr(remote)+'\n'+body+'\nPY\n'
    folder.mkdir(exist_ok=True)
    name=args.task+'-observe-'+str(int(time.time()))
    (folder/(name+'-command.sh')).write_bytes(shell.encode())
    result=subprocess.run(SSH+['bash','-s'],input=shell.encode(),capture_output=True,timeout=40)
    (folder/(name+'.stderr')).write_bytes(result.stderr)
    result.check_returncode()
    value=json.loads(result.stdout)
    (folder/(name+'.json')).write_text(json.dumps(value,indent=2)+'\n')
    for rank,record in value['records'].items():
        (folder/(args.task+'-rank'+rank+'.json')).write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(alive=value['driver_alive'],records={rank:{k:record.get(k) for k in
        ('phase','elapsed_seconds','batch','variant','layer','allocated','reserved','traceback')}
        for rank,record in value['records'].items()},
        gdn_phases={k:v[-1500:] for k,v in value['phase_logs'].items() if k.startswith('gdn')},
        log=value['log'][-2800:],physical=value['physical'],host_available=value['host']['available'])))


if __name__=='__main__':main()

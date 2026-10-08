"""Read the bounded collection's actual PID, phase and resources; no job mutation."""
import json
import argparse
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from stage_environment_entry import ENTRY,ROOT,SSH

BODY=r'''
import hashlib,json,os,psutil,subprocess,time
from pathlib import Path
root=Path(TARGET)
data={'unix':time.time(),'files':{},'processes':[]}
for name in ('launch.json','batching-cpu-replay.json','results/actor-initialization.json','results/completed.json',
             'results/rank0.json','results/rank1.json'):
 p=root/name
 if p.exists():
  raw=p.read_bytes()
  try:value=json.loads(raw)
  except json.JSONDecodeError:value={'read_during_write':True,'bytes':len(raw)}
  if globals().get('SUMMARY_ONLY',False) and name.startswith('results/rank') and 'batches' in value:
   value={**{k:value.get(k) for k in ('phase','unix','elapsed_seconds','rank','pid','birth','operations','traceback')},
          'completed_points':sum(len(b['points']) for b in value['batches']),
          'batch_indices':[b['index'] for b in value['batches']], 'summary_only':True}
  data['files'][name]={'path':str(p),'sha256':hashlib.sha256(raw).hexdigest(),'value':value}
launch=data['files'].get('launch.json',{}).get('value',{})
if launch:
 try:
  driver=psutil.Process(launch['pid'])
  assert driver.create_time()==launch['birth'],'PID was reused'
  for p in ([] if globals().get('PHASE_ONLY',False) else [driver]+driver.children(recursive=True)):
   try:
    info=p.as_dict(attrs=['pid','name','status','create_time','cpu_times'])
    info['cpu_times']=list(info['cpu_times']);m=p.memory_full_info()
    info.update(rss=m.rss,pss=getattr(m,'pss',None),uss=getattr(m,'uss',None))
    data['processes'].append(info)
   except psutil.NoSuchProcess:pass
  data['driver_alive']=True
 except psutil.NoSuchProcess:data['driver_alive']=False
for rank in (0,1):
 p=root/f'results/rank{rank}-phases.jsonl'
 if p.exists():
  lines=p.read_bytes().splitlines();events=[]
  for line in lines[-8:]:
   try:events.append(json.loads(line))
   except json.JSONDecodeError:pass
  data[f'rank{rank}_last_phases']=events
  if INCLUDE_PHASES:
   data[f'rank{rank}_all_phases']=[json.loads(line) for line in lines if line]
p=root/'driver.log'
if p.exists():data['driver_log_tail']='\n'.join(p.read_text(errors='replace').splitlines()[-80:])
data['physical']=('Not sampled: phase-only read' if globals().get('PHASE_ONLY',False)
                  else subprocess.run(['mx-smi'],capture_output=True,text=True,check=True).stdout)
data['host_memory']=psutil.virtual_memory()._asdict()
hold=Path(ROOT)/'receipts/direct-target-prefix-runtime-20261007-v1/textcraft-first-dt'
data['textcraft_hold']={'driver_pid':2833207,'expected_birth':1791370325.16,
 'release_exists':{str(r):(hold/f'rank{r}-release-update').exists() for r in (0,1)}}
try:data['textcraft_hold']['actual_birth']=psutil.Process(2833207).create_time()
except psutil.NoSuchProcess:data['textcraft_hold']['actual_birth']=None
data['cgroup']={}
for name in ('memory.usage_in_bytes','memory.stat'):
 p=Path('/sys/fs/cgroup/memory')/name
 if p.exists():data['cgroup'][name]=p.read_text()
print(json.dumps(data))
'''

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--layer-task',choices=('textcraft','appworld'))
    parser.add_argument('--factual-controls', action='store_true',
                        help='Observe the separately staged passive factual-control run')
    parser.add_argument('--suboperations', action='store_true',
                        help='Observe the separately staged passive suboperation readout')
    parser.add_argument('--revision',choices=('v1','v2'),default='v1')
    parser.add_argument('--summary-only', action='store_true',
                        help='Read phases/resources without retransferring growing point ledgers')
    parser.add_argument('--phase-only', action='store_true',
                        help='Omit process PSS and driver query while checking a live phase')
    args=parser.parse_args()
    target=ROOT+'/receipts/credit-author-development-collection-20261008-v1'
    prefix='author-collection'
    if args.layer_task:
        target=ROOT+'/receipts/credit-layer-development-'+args.layer_task+'-20261008-v1'
        prefix='layer-'+args.layer_task
        if args.factual_controls:
            target=ROOT+'/receipts/credit-layer-factual-controls-'+args.layer_task+'-20261008-v1'
            prefix='layer-factual-controls-'+args.layer_task
        if args.suboperations:
            target=ROOT+'/receipts/credit-layer-suboperations-'+args.layer_task+'-20261008-'+args.revision
            prefix='layer-suboperations-'+args.layer_task+('-'+args.revision if args.revision!='v1' else '')
    elif args.factual_controls:
        parser.error('--factual-controls requires --layer-task')
    elif args.suboperations:
        parser.error('--suboperations requires --layer-task')
    body=('ROOT='+repr(ROOT)+'\nTARGET='+repr(target)+'\nINCLUDE_PHASES='+repr(bool(args.layer_task))
          +'\nSUMMARY_ONLY='+repr(args.summary_only)+'\n'+BODY)
    body='PHASE_ONLY='+repr(args.phase_only)+'\n'+body
    shell='source '+ENTRY+'/metax-entry.env.sh\nCUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<\'PY\'\n'+body+'\nPY\n'
    run=subprocess.run(SSH+['bash','-s'],input=shell.encode(),capture_output=True,timeout=120)
    (HERE/(prefix+'-observation.stderr.txt')).write_bytes(run.stderr);run.check_returncode()
    data=json.loads(run.stdout)
    out=HERE/(prefix+'-observations');out.mkdir(exist_ok=True)
    path=out/(str(int(data['unix']))+'.json');path.write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
    for name,binding in data['files'].items():
        if name.startswith('results/rank') and not binding['value'].get('read_during_write') and not args.summary_only:
            (out/Path(name).name).write_text(json.dumps(binding['value'],indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'receipt':str(path),'unix':data['unix'],'driver_alive':data.get('driver_alive'),
        'cpu_replay_receipt':data['files'].get('batching-cpu-replay.json',{}).get('sha256'),
        'ranks':{str(r):data.get(f'rank{r}_last_phases',[])[-1:] for r in (0,1)},
        'completed':data['files'].get('results/completed.json',{}).get('value'),
        'host_available':data['host_memory']['available'],'textcraft_hold':data['textcraft_hold']},ensure_ascii=False))

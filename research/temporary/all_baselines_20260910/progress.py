"""Compact operational progress only; never reads gold or computes quality."""
import argparse
from collections import Counter,defaultdict
import json
import os
from pathlib import Path
import statistics

def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);a=p.parse_args();b=a.base
    by_task=Counter();by_method=Counter();times=defaultdict(list);latest=[]
    for path in (b/'all_baselines_v2').glob('*/*/*/results.json'):
        r=json.loads(path.read_bytes());assert r['status']=='complete'
        by_task[r['dataset']]+=1;by_method[r['method']]+=1;times[r['dataset'],r['method']].append(r['seconds'])
        latest.append((path.stat().st_mtime,{k:r[k] for k in ('dataset','index','method','seconds')}))
    state=json.loads((b/'all_baselines_logs_v3/controller.json').read_bytes())
    phase=state['phases'][-1];pid=phase.get('pid');alive=False
    if pid:
        try:os.kill(pid,0);alive=True
        except ProcessLookupError:pass
    remaining=0
    for (task,method),values in times.items():
        remaining+=max(0,(48 if task=='hotpotqa_long' else 100)-len(values))*statistics.median(values)
    print(json.dumps(dict(status=state.get('status',phase['status']),phase=phase['name'],pid=pid,alive=alive,
        completed=sum(by_method.values()),total=2240,by_task=dict(by_task),by_method=dict(by_method),
        estimated_remaining_operation_hours=round(remaining/3600,2),
        historical_failed_attempts=len(list((b/'all_baselines_v2').glob('*/*/*/failed_attempt_*.json'))),
        latest=[r for _,r in sorted(latest,key=lambda x:x[0])[-3:]])),flush=True)

if __name__=='__main__':main()

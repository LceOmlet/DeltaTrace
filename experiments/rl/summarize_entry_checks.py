"""Read original job/log/checkpoint artifacts; do not infer health from liveness."""
import argparse
import json
from pathlib import Path
import re
import time
import psutil


def summarize(directory):
    job=json.loads((directory/'job.json').read_text())
    try:
        p=psutil.Process(job['pid'])
        live=p.status()!=psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        live=False
    sessions=list(Path('/tmp/ray').glob(f"session_*_{job['pid']}"))
    metric_rows=[]
    phases=[]
    dt_batches=[]
    for session in sessions:
        for file in (session/'logs').glob('worker*.out'):
            for line in file.read_text(errors='replace').splitlines():
                if line.startswith('step:'):
                    values={k:float(v) for k,v in re.findall(r'([\w/]+):(-?[\d.]+)',line)}
                    metric_rows.append({k:v for k,v in values.items() if k in
                        ('step','actor/grad_norm','actor/pg_loss','episode/reward/mean','episode/success_rate',
                         'episode/length/mean','response_length/mean','response_length/max','prompt_length/max',
                         'timing_s/gen','timing_s/adv','timing_s/old_log_prob','timing_s/update_actor',
                         'timing_s/save_checkpoint','timing_s/step','perf/total_num_tokens')})
                if line.startswith('[DT rollout]'):phases.append(line)
                if line.startswith('[DT EOS minibatch]'):dt_batches.append(line)
    generation=[]
    for file in directory.glob('raw-vllm-*.jsonl'):
        for line in file.read_text().splitlines():
            generation.extend(json.loads(line)['rows'])
    marker=directory/'checkpoints/latest_checkpointed_iteration.txt'
    return dict(job=directory.name,pid=job['pid'],live=live,age_seconds=time.time()-job['started_unix'],
        checkpoint_step=int(marker.read_text()) if marker.exists() else None,
        metrics=sorted(metric_rows,key=lambda m:m['step']),last_phases=phases[-2:],
        last_dt_batches=dt_batches[-2:],completed_responses=len(generation),
        generated_tokens=sum(r['tokens'] for r in generation),nonfinite_logprobs=sum(r['nonfinite'] for r in generation))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--job',type=Path,action='append',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    result=dict(unix=time.time(),scope='bounded checks, not formal training',jobs=[summarize(d) for d in a.job])
    a.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

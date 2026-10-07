"""Read at most 2 MiB of each exact live v3 worker log; no runtime calls."""
from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import subprocess
import time


REMOTE = r'''
import base64, hashlib, json, os, pathlib, psutil, sys, time
spec = json.loads(base64.b64decode(sys.argv[1]))
def guard(job):
    p = psutil.Process(job['pid'])
    assert abs(p.create_time()-job['birth']) < .02 and p.status() != 'zombie', 'driver identity mismatch'
    raw = pathlib.Path(job['source']).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == job['source_sha256'], 'source identity mismatch'
    return {'pid':p.pid,'birth':p.create_time(),'status':p.status(),'source_sha256':hashlib.sha256(raw).hexdigest()}
out = {'observed_unix':time.time(),'scope':'read-only exact-process guarded original worker tails; no model/RPC/GPU calls','jobs':[]}
for job in spec:
    item={'task':job['task'],'guard_before':guard(job),'logs':[]}
    for worker in job['workers']:
        p=psutil.Process(worker['pid'])
        assert abs(p.create_time()-worker['birth']) < .02 and p.status()!='zombie', 'worker identity mismatch'
        path=pathlib.Path(worker['path'])
        with path.open('rb') as f:
            size=os.fstat(f.fileno()).st_size
            start=max(0,size-2*1024*1024)
            f.seek(start)
            tail=f.read(2*1024*1024)
        item['logs'].append({'worker_pid':p.pid,'worker_birth':p.create_time(),'path':str(path),'file_bytes_at_read':size,'tail_start':start,'tail_bytes':len(tail),'tail_sha256':hashlib.sha256(tail).hexdigest(),'tail_b64':base64.b64encode(tail).decode('ascii')})
    item['guard_after']=guard(job)
    out['jobs'].append(item)
out['completed_unix']=time.time()
print(json.dumps(out,ensure_ascii=False))
'''


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def range_stats(values):
    values=[float(x) for x in values if x is not None]
    finite=[x for x in values if math.isfinite(x)]
    return {'count':len(values),'nonfinite':len(values)-len(finite),
            'min':min(finite) if finite else None,'max':max(finite) if finite else None,
            'max_abs':max(map(abs,finite)) if finite else None,
            'mean':sum(finite)/len(finite) if finite else None}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--phase',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    phase=json.loads(args.phase.read_text(encoding='utf-8'))
    spec=[]
    for item in phase['jobs']:
        j=item['job']
        workers=[]
        for p in item['processes']:
            if 'WorkerDict.actor_rollout' not in p.get('name',''):continue
            paths=[x['path'] for x in item['logs'] if x['path'].endswith('-'+str(p['pid'])+'.out')]
            assert len(paths)==1
            workers.append({'pid':p['pid'],'birth':p['birth'],'path':paths[0]})
        assert len(workers)==2
        spec.append({'task':j['task'],'pid':j['pid'],'birth':j['observed_process_created_unix'],
                     'source':j['source_receipt'],'source_sha256':item['source_sha256'],'workers':workers})
    stage_path=Path(__file__).parents[1]/'stage_environment_entry.py'
    module_spec=importlib.util.spec_from_file_location('recorded_stage',stage_path)
    stage=importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(stage)
    encoded=base64.b64encode(json.dumps(spec).encode()).decode()
    # Python source and identities travel over stdin; the strict existing SSH owner is unchanged.
    script="/opt/conda/bin/python - '"+encoded+"' <<'PY'\n"+REMOTE+"\nPY\n"
    cp=subprocess.run(stage.SSH+['bash','-s'],input=script.encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=60)
    if cp.returncode:raise RuntimeError(cp.stderr.decode(errors='replace'))
    data=json.loads(cp.stdout)
    args.output.mkdir(parents=True,exist_ok=True)
    receipt={'observed_unix':data['observed_unix'],'completed_unix':data['completed_unix'],
             'scope':data['scope'],'collector':{'path':str(Path(__file__).resolve()),'sha256':sha(Path(__file__))},
             'input_phase':{'path':str(args.phase.resolve()),'sha256':sha(args.phase)},
             'ssh_owner':{'path':str(stage_path.resolve()),'sha256':sha(stage_path)},'jobs':[]}
    decoder=json.JSONDecoder()
    for job in data['jobs']:
        result={'task':job['task'],'guard_before':job['guard_before'],'guard_after':job['guard_after'],'logs':[]}
        for log in job['logs']:
            raw=base64.b64decode(log.pop('tail_b64'))
            name=f"{job['task'].lower()}-worker-{log['worker_pid']}"
            tail_path=args.output/(name+'.original-tail.out')
            tail_path.write_bytes(raw)
            log['local_tail']={'path':str(tail_path.resolve()),'sha256':sha(tail_path)}
            text=raw.decode('utf-8',errors='replace')
            reports=[]
            for n,match in enumerate(re.finditer(r'\[DeltaTrace readout\] ',text)):
                try:
                    report,end=decoder.raw_decode(text[match.end():])
                except json.JSONDecodeError:continue
                exact=text[match.end():match.end()+end].encode('utf-8')
                path=args.output/f'{name}-report-{n}.json'
                path.write_bytes(exact+b'\n')
                traces=report.get('traces',[])
                summary={k:report.get(k) for k in ('task','target_semantics','target_normalization','reward_events','trajectories','finite_trace_calls','joint_target_requests','minibatch_size','target_self_tokens','prior_source_tokens','policy_tokens','empty_joint_targets','zero_reward_trajectories','seconds','raw_advantage_groups')}
                summary['actual_context_lengths']=range_stats(report.get('actual_context_lengths',[]))
                summary['trace_fields']={k:range_stats([x.get(k) for x in traces]) for k in ('target_tokens','actual_context_tokens','compute_tokens','target_self_tokens','prior_source_tokens','root_effect','signed_sum','conservation_residual','factual_target_logp','reference_target_logp','reward')}
                summary['trace_uids']=[x.get('traj_uid') for x in traces]
                reports.append({'path':str(path.resolve()),'sha256':sha(path),'original_json_sha256':hashlib.sha256(exact).hexdigest(),'summary':summary})
            timings=[]
            for m in re.finditer(r'\[DT direct joint minibatch\] trajectories=(\d+) length=(\d+) batch=(\d+)/(\d+) seconds=([\d.eE+-]+)',text):
                timings.append({'trajectories':int(m[1]),'length':int(m[2]),'batch':int(m[3]),'total_batches':int(m[4]),'seconds':float(m[5])})
            log['reports']=reports
            log['minibatch_timings']=timings
            log['minibatch_seconds_summary']=range_stats([x['seconds'] for x in timings])
            result['logs'].append(log)
        receipt['jobs'].append(result)
    receipt['interpretation']={'dp_padding':'TextCraft trace rows are 88 per rank / 176 total: 170 unique requested trace trajectories plus 6 DP pad duplicates. Raw groups are pre-whitening and include these duplicates.','whole_batch_whiten':'Not inferred from per-worker raw groups; original trainer owns whole-batch whitening.','trace_residual':'Diagnostic only; no new numerical acceptance threshold.'}
    out=args.output/'extraction-receipt.json'
    out.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'receipt':str(out.resolve()),'sha256':sha(out),'jobs':[{'task':j['task'],'workers':[{'pid':l['worker_pid'],'reports':len(l['reports']),'timings':len(l['minibatch_timings']),'summary':[r['summary'] for r in l['reports']]} for l in j['logs']]} for j in receipt['jobs']]},ensure_ascii=False))


if __name__=='__main__':main()

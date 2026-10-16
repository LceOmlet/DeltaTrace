"""Summarize the original reader and exact native structured DT reports."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import time

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT=ROOT/'receipts/sql-native-multiturn-speed-20261016-v2'
queue=json.loads((AUDIT/'trial-status.json').read_bytes())
assert queue['status']=='all_native_drivers_finished' and len(queue['tasks'])==1
job=queue['tasks'][0]
assert job['exit_code']==0
sessions=list(Path('/tmp/ray').glob('session_*_'+str(job['pid'])))
assert len(sessions)==1
session=sessions[0]
logs=list((session/'logs').glob('worker-*.out'))
assert any(job['run'] in p.read_text(errors='replace') for p in logs),'Original TaskRunner must identify this exact isolated run'
# Recover the immutable native Ray artifact after normal worker shutdown.
# This records provenance; it does not parse/reconstruct training metrics.
links={str(job['pid']):dict(birth=job['birth'],ray_session=str(session),
    source='Exact original Ray directory driver PID and original TaskRunner literal run path')}
(AUDIT/'native-Ray-session-links.json').write_text(json.dumps(links,indent=2)+'\n')
p=ROOT/'receipts/sql-native-multiturn-speed-20261016-v1/source/collect_status.py'
code=p.read_text().replace('sql-native-multiturn-speed-20261016-v1','sql-native-multiturn-speed-20261016-v2')
with contextlib.redirect_stdout(io.StringIO()):exec(compile(code,str(p),'exec'),globals())
snapshot=json.loads((AUDIT/'latest-trial-snapshot.json').read_bytes())
current=snapshot['jobs'][0]
assert current['completed_iterations']==2 and current['exit_code']==0
reports=[];workloads=[]
for path in logs:
    index=0
    for number,line in enumerate(path.read_text(errors='replace').splitlines(),1):
        marker='[DeltaTrace readout] '
        if marker in line:
            index+=1
            reports.append(dict(iteration=index,path=str(path),line=number,report=json.loads(line.split(marker,1)[1])))
        if '[DT direct workload]' in line:workloads.append(dict(path=str(path),line=number,text=line))
stages=['gen','old_log_prob','adv','update_actor','step']
means={key:sum(row['values']['timing_s/'+key] for row in current['metrics'])/2 for key in stages}
DT=[]
for iteration in (1,2):
    pair=[item for item in reports if item['iteration']==iteration]
    assert len(pair)==2
    assert all(item['report']['prefix_lease_used'] and item['report']['per_token_probability_queries']==0 for item in pair)
    traces=[trace for item in pair for trace in item['report']['traces']]
    DT.append(dict(iteration=iteration,rank_seconds=[item['report']['seconds'] for item in pair],
        parallel_wall_seconds=max(item['report']['seconds'] for item in pair),
        dispatched_requests=len(traces),compute_tokens=sum(t['compute_tokens'] for t in traces),
        actual_context_tokens=sum(t['actual_context_tokens'] for t in traces),
        maximum_actual_context=max(t['actual_context_tokens'] for t in traces),
        B4_calls_per_rank=[item['report']['finite_trace_calls'] for item in pair]))
record=dict(version='sql-native-multiturn-speed-20261016-v2',collected_unix=time.time(),source_commit=queue['source_commit'],
    job=job,native_Ray_provenance=links,completed_iterations=2,metrics=current['metrics'],mean_stage_seconds=means,
    native_DT=DT,native_workloads=workloads,original_reports=reports,
    source_provenance='Exact prepared binding SHA checked before launch; native structured DT executed. No live worker import RPC was captured before normal shutdown, so do not claim that stronger verification.',
    limitations=['Two bounded original iterations, not formal full-batch throughput or task-quality evaluation',
        'Native SQL generic tool_call_count=0 is not a valid count of SQLite executions',
        'Two rank DT times are parallel, not additive','Natural contexts below32768; PPO capacity is verified separately'],
    formal_mutations=0)
(AUDIT/'complete-measurements.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(dict(completed_iterations=2,mean_stage_seconds=means,native_DT=DT),indent=2))

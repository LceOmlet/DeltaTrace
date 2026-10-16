"""Summarize original complete-step metrics and native structured DT reports."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import runpy
import time

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT = ROOT / 'receipts/appworld-sql-native-speed-20261016-v3'
with contextlib.redirect_stdout(io.StringIO()):
    runpy.run_path(str(AUDIT/'source/collect_status.py'), run_name='original_complete_trial_reader')
snapshot = json.loads((AUDIT/'latest-trial-snapshot.json').read_bytes())
assert snapshot['queue']['status'] == 'all_native_drivers_finished'
assert all(job['exit_code'] == 0 and job['completed_iterations'] == 2 for job in snapshot['jobs'])
reports = []
events = []
warnings = []
tasks = {}
for job in snapshot['jobs']:
    tasks[job['task']] = dict(completed_iterations=job['completed_iterations'],exit_code=job['exit_code'],
                             pid=job['pid'],birth=job['birth'],metrics=job['metrics'])
    for path in (Path(job['ray_session'])/'logs').glob('worker-*.*'):
        if path.suffix not in ('.out','.err'):
            continue
        lines = path.read_text(errors='replace').splitlines()
        rank_report_index = 0
        for number,line in enumerate(lines,1):
            marker = '[DeltaTrace readout] '
            if marker in line:
                report = json.loads(line.split(marker,1)[1])
                rank_report_index += 1
                reports.append(dict(task=job['task'],iteration=rank_report_index,
                                    path=str(path),line=number,report=report))
            if any(marker in line for marker in ('[loop_transport]','[loop_trajectory]','[DT direct workload]')):
                events.append(dict(task=job['task'],path=str(path),line=number,text=line))
            if 'fx graph cache unable to load compiled graph' in line:
                warnings.append(dict(path=str(path),line=number,text='\n'.join(lines[number-1:number+8])))
all_imports = {task:json.loads((AUDIT/('actual-worker-imports-'+task.lower()+'.json')).read_bytes()) for task in tasks}
for record in all_imports.values():
    assert all(module.get('matches_prepared_source') is not False for worker in record['results'] for module in worker['modules'].values())
DT = []
for iteration in (1,2):
    pair = [record for record in reports if record['task']=='AppWorld' and record['iteration']==iteration]
    assert len(pair) == 2
    traces = [trace for record in pair for trace in record['report']['traces']]
    unique = {}
    duplicate_workload = []
    for trace in traces:
        uid = trace['traj_uid']
        if uid in unique:
            # Batch padding and approximate finite outputs can differ across
            # repeated dispatches. Only the original workload fields define
            # the duplicate here; this is not a DT numerical-accuracy test.
            for key in ('reward','target_tokens','actual_context_tokens','target_self_tokens','prior_source_tokens'):
                assert unique[uid][key] == trace[key]
            duplicate_workload.append(dict(traj_uid=uid,first_compute_tokens=unique[uid]['compute_tokens'],
                                           repeated_compute_tokens=trace['compute_tokens']))
        else:
            unique[uid] = {key:value for key,value in trace.items() if key != 'trajectory_index'}
    context = sum(trace['actual_context_tokens'] for trace in traces)
    compute = sum(trace['compute_tokens'] for trace in traces)
    unique_context = sum(trace['actual_context_tokens'] for trace in unique.values())
    DT.append(dict(iteration=iteration,original_rank_reports=pair,
        native_DT_wall_seconds=max(record['report']['seconds'] for record in pair),
        rank_seconds=[record['report']['seconds'] for record in pair],
        unique_requests=len(unique),dispatched_requests=len(traces),padded_duplicate_requests=len(traces)-len(unique),
        duplicate_workload=duplicate_workload,
        unique_context_tokens=unique_context,dispatched_context_tokens=context,
        dispatched_compute_tokens=compute,length_padding_token_overhead=(compute-context)/context,
        duplicated_context_token_overhead=(context-unique_context)/unique_context,
        maximum_actual_context=max(trace['actual_context_tokens'] for trace in traces),
        no_per_token_probability_queries=all(record['report']['per_token_probability_queries']==0 for record in pair),
        native_prefix_lease_used=all(record['report']['prefix_lease_used'] for record in pair)))
result = dict(collected_unix=time.time(),version='appworld-sql-native-speed-20261016-v3',
    source_commit=snapshot['queue']['source_commit'],tasks=tasks,AppWorld_DT=DT,
    actual_worker_imports=all_imports,native_owner_progress=events,nonfatal_native_cache_warnings=warnings,
    snapshot_SHA256=hashlib.sha256((AUDIT/'latest-trial-snapshot.json').read_bytes()).hexdigest(),
    limits=['Two bounded native iterations; not full formal batch throughput or quality comparison',
            'SQL all original terminal rewards -1 and official group multipliers zero: DT skipped, not timed',
            'Original SQL generic tool_count metadata is absent/zero and not a valid count of SQL executions',
            'Rank DT seconds run in parallel; original adv wall includes max-rank DT plus orchestration',
            'Natural AppWorld context below32768; existing v5 capacity receipt is separate',
            'Cache temporary-file warnings retained; both native drivers exited0, no cache cleared'])
result['postprocessing_attempt1'] = 'Initial full-trace equality guard rejected duplicate rows with different batch-padding/approximate-output fields; corrected to compare workload fields only. No model or training mutation.'
(AUDIT/'native-DT-reports.json').write_text(json.dumps(dict(collected_unix=result['collected_unix'],reports=reports),indent=2)+'\n')
(AUDIT/'complete-measurements.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(tasks={key:{k:v for k,v in value.items() if k!='metrics'} for key,value in tasks.items()},
    AppWorld_DT=[{k:v for k,v in record.items() if k!='original_rank_reports'} for record in DT]),indent=2))

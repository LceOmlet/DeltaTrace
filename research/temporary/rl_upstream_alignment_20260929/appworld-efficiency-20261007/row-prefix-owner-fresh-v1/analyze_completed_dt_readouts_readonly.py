"""Read original completed DT report costs using the existing bound log reader."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('batch_reader', HERE / 'analyze_actual_dt_batches_readonly.py')
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)

OLD = """current['readout']=dict(line=line_no,raw_line_sha256=hashlib.sha256(line.encode()).hexdigest(),
                seconds=report.get('seconds'),finite_trace_calls=report.get('finite_trace_calls'),
                event_contrasts=report.get('event_contrasts'),nonzero_advantages=report.get('nonzero_advantages'),
                conservation_failures=report.get('conservation_failures'))"""
NEW = """current['readout']=dict(line=line_no,raw_line_sha256=hashlib.sha256(line.encode()).hexdigest(),
                seconds=report.get('seconds'),finite_trace_calls=report.get('finite_trace_calls'),
                event_contrasts=report.get('event_contrasts'),nonzero_advantages=report.get('nonzero_advantages'),
                conservation_failures=report.get('conservation_failures'),
                shared_native_prefix=report.get('shared_native_prefix'),
                trace_scalar_aggregates=dict(trace_count=len(report.get('traces',[])),
                    trace_keys=sorted({k for t in report.get('traces',[]) for k in t}),
                    context_tokens_sum=sum(t['context_tokens'] for t in report.get('traces',[])),
                    compute_tokens_sum=sum(t['compute_tokens'] for t in report.get('traces',[])),
                    query_tokens_sum=sum(t['query_tokens'] for t in report.get('traces',[])),
                    actual_row_lengths_count=len(report.get('actual_row_lengths',[])),
                    actual_row_lengths_sum=sum(report.get('actual_row_lengths',[])),
                    actual_row_lengths_min=min(report.get('actual_row_lengths',[]),default=None),
                    actual_row_lengths_max=max(report.get('actual_row_lengths',[]),default=None)),
                scalar_fields={k:v for k,v in report.items() if isinstance(v,(int,float,bool)) or v is None},
                existing_top_level_cost_fields={k:v for k,v in report.items()
                    if any(t in k.lower() for t in ('second','time','cost','memory','bytes'))
                    and isinstance(v,(int,float,bool,str))})"""


if __name__ == '__main__':
    assert reader.REMOTE.count(OLD) == 1
    remote = reader.REMOTE.replace(OLD, NEW).replace('@ROOT@', repr(reader.stage.ROOT))
    anchor = 'assert psutil.Process(2360541).create_time()==1791325655.01'
    workload_reader = '''workload_path=Path(job['log'])
workload_raw=workload_path.read_bytes()
workload_lines=[]
for line_no,line in enumerate(workload_raw.decode(errors='replace').splitlines(),1):
    if '[DT source workload] ' in line:
        values={k:int(v) for k,v in re.findall(r'([a-z_]+)=(\\d+)',line.split('[DT source workload] ',1)[1])}
        workload_lines.append(dict(line=line_no,raw=line,**values))
workload_source=dict(path=str(workload_path),observed_prefix_bytes=len(workload_raw),
    sha256_of_observed_prefix=hashlib.sha256(workload_raw).hexdigest(),lines=workload_lines)
'''
    assert remote.count(anchor) == 1
    remote = remote.replace(anchor, workload_reader + anchor)
    anchor = 'driver=dict(pid=2360541,birth=1791325655.01),source='
    assert remote.count(anchor) == 1
    remote = remote.replace(anchor, 'driver=dict(pid=2360541,birth=1791325655.01),source_workload=workload_source,source=')
    script = ("/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_qwen35_20260912/env/bin/python - <<'PY'\n"
              + remote + '\nPY\n')
    result = subprocess.run(reader.stage.SSH + ['bash', '-s'], input=script.encode(), capture_output=True, timeout=65)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors='replace'))
    data = json.loads(result.stdout)
    for log in data['logs']:
        for group in log['groups']:
            reader.summarize(group)
            readout = group['readout']
            if readout is None:
                continue
            batch_sum = group['summary']['sum_logged_batch_seconds']
            prefix = readout.get('shared_native_prefix') or {}
            preparation = prefix.get('capture_and_preparation_seconds')
            trace = readout['trace_scalar_aggregates']
            common_prefix = prefix.get('original_shared_prefix_token_slots')
            trace['compute_minus_context_slots'] = trace['compute_tokens_sum'] - trace['context_tokens_sum']
            trace['compute_minus_legacy_common_prefix_slots'] = (
                trace['compute_tokens_sum'] - common_prefix if common_prefix is not None else None)
            trace['query_target_tokens_from_row_lengths'] = trace['context_tokens_sum'] - trace['actual_row_lengths_sum']
            trace['query_target_scope_check'] = (
                trace['query_target_tokens_from_row_lengths'] == trace['query_tokens_sum'] + trace['trace_count'])
            group['descriptive_cost_accounting'] = dict(
                report_seconds=readout['seconds'], logged_batch_seconds=batch_sum,
                capture_and_preparation_seconds=preparation,
                report_minus_logged_batches_seconds=readout['seconds'] - batch_sum,
                report_minus_logged_batches_and_preparation_seconds=(
                    readout['seconds'] - batch_sum - preparation if preparation is not None else None))
        complete = [g for g in log['groups'] if g['readout'] is not None]
        partial = [g for g in log['groups'] if g['readout'] is None]
        log['completed_groups_totals'] = dict(
            completed_groups=len(complete),
            report_seconds=sum(g['readout']['seconds'] for g in complete),
            logged_batch_seconds=sum(g['summary']['sum_logged_batch_seconds'] for g in complete),
            capture_and_preparation_seconds=sum((g['readout'].get('shared_native_prefix') or {}).get(
                'capture_and_preparation_seconds', 0.) for g in complete),
            finite_trace_calls=sum(g['readout']['finite_trace_calls'] for g in complete),
            event_contrasts=sum(g['readout']['event_contrasts'] for g in complete),
            partial_groups=[dict(group=g['group'],completed_logged_batches=g['completed_logged_batches'],
                                 planned_batches=g['planned_batches']) for g in partial])
    data['scope'] = ('Original report scalar/capture costs and original per-batch log lines only. '
        'Large IDs, per-token vectors and minimum-batch payloads are omitted. '
        'Differences are descriptive outer-scope remainder, not identified copy/compile time. '
        'Ranks are concurrent and not added. Partial groups do not have a final report. '
        'compute_tokens is the original full batch maximum and original_shared_prefix_token_slots '
        'still uses the original common MIN boundary. Their difference preserves legacy metadata '
        'accounting only, and is not the actual individual-row suffix slots, FA traffic or FLOPs. '
        'No production logging, parameter, threshold, GPU, model, Ray RPC or checkpoint operation.')
    data['analysis_script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    data['reused_reader_sha256'] = hashlib.sha256(Path(reader.__file__).read_bytes()).hexdigest()
    data['local_owner_source_binding'] = []
    for path in (HERE.parents[1] / 'appworld-efficiency-20261007' / 'individual-prefix-owner-candidate-v1' /
                 'combined-storage-row-v1' / 'candidate' / 'native_prefix_leases.py',
                 HERE.parents[4] / 'experiments' / 'rl' / 'dt_training_batch.py'):
        if path.exists():
            data['local_owner_source_binding'].append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    output = HERE / ('completed-dt-readout-costs-' + str(int(data['observed_end_unix'])) + '-readonly.json')
    output.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(receipt=str(output),sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        script_sha256=data['analysis_script_sha256'],
        source_workload=data['source_workload'],
        logs=[dict(path=log['path'],source_sha256=log['sha256_of_observed_prefix'],totals=log['completed_groups_totals'],
            groups=[dict(group=g['group'],status=g['status'],batches=g['completed_logged_batches'],
                plan=g['plan'],costs=g.get('descriptive_cost_accounting'),readout=g['readout'])
                for g in log['groups']]) for log in data['logs']]),indent=2))

"""Read the fixed fresh owner's existing DT scalar log lines once; no model calls."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import subprocess

HERE = Path(__file__).resolve().parent
AUDIT = HERE.parents[1]
spec = importlib.util.spec_from_file_location('stage', AUDIT / 'stage_environment_entry.py')
stage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage)

REMOTE = r'''
from pathlib import Path
import hashlib,json,psutil,re,time
root=Path(@ROOT@)
expected_source='c83b96debba90d85476e58b56b2c8c7f3900901f26ce41094e8f60ac489c7789'
job=next(j for j in json.loads((root/'active-training.json').read_bytes())['jobs'] if j['task']=='AppWorld')
assert job['pid']==2360541 and job['observed_process_created_unix']==1791325655.01
driver=psutil.Process(2360541)
assert driver.create_time()==1791325655.01
source=Path(job['source_receipt'])
assert hashlib.sha256(source.read_bytes()).hexdigest()==expected_source
descendants={p.pid:p for p in driver.children(recursive=True)}
begin=time.time()
logs=[]
for rank,pid in enumerate((2367855,2369144)):
    assert pid in descendants, (pid,'not a current driver descendant')
    process=descendants[pid]
    matches=[]
    for session in Path('/tmp/ray').glob('*_2360541'):
        matches.extend((session/'logs').glob('worker-*-'+str(pid)+'.out'))
    assert len(matches)==1, (pid,[str(p) for p in matches])
    path=matches[0]
    raw=path.read_bytes()
    groups=[]
    current=None
    for line_no,line in enumerate(raw.decode(errors='replace').splitlines(),1):
        if '[DT EOS plan] ' in line:
            values={k:int(v) for k,v in re.findall(r'(events|contrasts|batches)=(\d+)',line)}
            current=dict(group=len(groups)+1,plan=dict(line=line_no,raw=line,**values),batches=[],readout=None)
            groups.append(current)
        elif '[DT EOS minibatch] ' in line:
            assert current is not None, (pid,line_no,'minibatch without plan')
            payload=line.split('[DT EOS minibatch] ',1)[1]
            values=dict(re.findall(r'([a-z_]+)=([^\s]+)',payload))
            index,total=map(int,values['batch'].split('/'))
            current['batches'].append(dict(line=line_no,raw=line,batch=index,planned_batches=total,
                contrasts=int(values['contrasts']),context_length=int(values['length']),
                seconds=float(values['seconds']),d_min=float(values['d_min']),d_max=float(values['d_max'])))
        elif '[DeltaTrace readout] ' in line and current is not None:
            report=json.loads(line.split('[DeltaTrace readout] ',1)[1])
            current['readout']=dict(line=line_no,raw_line_sha256=hashlib.sha256(line.encode()).hexdigest(),
                seconds=report.get('seconds'),finite_trace_calls=report.get('finite_trace_calls'),
                event_contrasts=report.get('event_contrasts'),nonzero_advantages=report.get('nonzero_advantages'),
                conservation_failures=report.get('conservation_failures'))
    logs.append(dict(rank=rank,pid=pid,process_birth=process.create_time(),process_name=process.name(),
        path=str(path),observed_prefix_bytes=len(raw),sha256_of_observed_prefix=hashlib.sha256(raw).hexdigest(),
        observed_mtime=path.stat().st_mtime,groups=groups))
assert psutil.Process(2360541).create_time()==1791325655.01
print(json.dumps(dict(observed_begin_unix=begin,observed_end_unix=time.time(),
    driver=dict(pid=2360541,birth=1791325655.01),source=dict(path=str(source),sha256=expected_source),logs=logs)))
'''


def summarize(group):
    rows = group['batches']
    seconds = [row['seconds'] for row in rows]
    ordered = sorted(seconds)
    group['status'] = 'complete_readout_returned' if group['readout'] is not None else 'partial'
    group['completed_logged_batches'] = len(rows)
    group['planned_batches'] = group['plan']['batches']
    group['all_planned_batch_lines_present'] = (
        [row['batch'] for row in rows] == list(range(1, group['plan']['batches'] + 1)))
    group['summary'] = dict(
        sum_logged_batch_seconds=sum(seconds), mean_seconds=statistics.mean(seconds) if seconds else None,
        median_seconds=statistics.median(seconds) if seconds else None,
        p95_seconds_nearest_rank=ordered[math.ceil(.95 * len(ordered)) - 1] if ordered else None,
        max_seconds=max(seconds) if seconds else None,
        logged_batch_max_context_min=min((row['context_length'] for row in rows), default=None),
        logged_batch_max_context_max=max((row['context_length'] for row in rows), default=None),
        d_min=min((row['d_min'] for row in rows), default=None),
        d_max=max((row['d_max'] for row in rows), default=None))


if __name__ == '__main__':
    script = ("/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_qwen35_20260912/env/bin/python - <<'PY'\n"
              + REMOTE.replace('@ROOT@', repr(stage.ROOT)) + '\nPY\n')
    result = subprocess.run(stage.SSH + ['bash', '-s'], input=script.encode(), capture_output=True, timeout=65)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors='replace'))
    data = json.loads(result.stdout)
    for log in data['logs']:
        for group in log['groups']:
            summarize(group)
    data['analysis_script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    data['scope'] = ('One read of exact fresh-driver descendant original .out files. Batch seconds are the original '
                     'complete_attribution_seconds_with_diagnostics field, not whole DT group/iteration time. '
                     'Context is each logged batch maximum request context including query/target, not every row length. '
                     'No added production logging, threshold, profiler, RPC, model, checkpoint, or runtime change. '
                     'Ranks run concurrently; sums must not be added as sequential wall time. '
                     'P95 uses the descriptive nearest-rank order statistic; no acceptance criterion.')
    output = HERE / ('actual-dt-batches-' + str(int(data['observed_end_unix'])) + '-readonly.json')
    output.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(receipt=str(output),sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        script_sha256=data['analysis_script_sha256'],
        logs=[dict(path=log['path'],pid=log['pid'],source_sha256=log['sha256_of_observed_prefix'],
                   groups=[{key:group[key] for key in ('group','status','completed_logged_batches','planned_batches','summary')}
                           for group in log['groups']]) for log in data['logs']]), indent=2))

"""Save an original-reader snapshot and show only its relevant progress fields."""
import contextlib
import io
import json
from pathlib import Path
import re
import runpy
import shutil

AUDIT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/appworld-sql-native-speed-20261016-v2')
with contextlib.redirect_stdout(io.StringIO()):
    runpy.run_path(str(AUDIT / 'source/collect_status.py'), run_name='original_trial_status_sample')
out = json.loads((AUDIT / 'latest-trial-snapshot.json').read_bytes())
saved = AUDIT / ('trial-snapshot-' + str(int(out['collected_unix'])) + '.json')
shutil.copyfile(AUDIT / 'latest-trial-snapshot.json', saved)
jobs = []
for job in out['jobs']:
    summary = {k: job[k] for k in ('task', 'pid', 'birth', 'alive', 'exit_code', 'workers', 'completed_iterations')}
    summary['tail'] = job['tail'][-5:]
    summary['original_parser_error_records_count'] = len(job['errors'])
    if job['metrics']:
        latest = job['metrics'][-1]
        summary['latest_original_metrics'] = {k: v for k, v in latest['values'].items()
            if k.startswith('timing_s/') or k in ('perf/total_num_tokens', 'episode/length/mean',
              'episode/tool_call_count/mean', 'response_length/mean', 'episode/reward/mean')}
        summary['latest_metric_sources'] = latest['sources']
    jobs.append(summary)
print(json.dumps(dict(snapshot=str(saved), collected_unix=out['collected_unix'], queue_status=out['queue'].get('status'),
    jobs=jobs, physical_GPU67_lines=[line for line in out['mx_smi'].splitlines() if re.match(r'\|\s*(?:6|7)\s', line)],
    cgroup_usage_GiB=out['memory_usage_bytes']/2**30, protected_current_processes=out['protected_current_processes']), indent=2))

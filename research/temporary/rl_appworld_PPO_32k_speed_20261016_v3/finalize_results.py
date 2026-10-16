"""Summarize completed original TaskRunner PPO stage metrics without a new log parser."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import runpy
import statistics
import time

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT = ROOT/'receipts/appworld-native-PPO-32k-speed-20261016-v3'
with contextlib.redirect_stdout(io.StringIO()):
    runpy.run_path(str(AUDIT/'source/collect_status.py'), run_name='native_PPO_complete_log_reader')
snapshot = json.loads((AUDIT/'latest-trial-snapshot.json').read_bytes())
assert snapshot['queue']['status'] == 'all_native_drivers_finished'
assert len(snapshot['jobs']) == 1
job = snapshot['jobs'][0]
assert job['exit_code'] == 0 and job['completed_iterations'] == 2
assert not job['errors'], job['errors']
binding_path = AUDIT/'actual-worker-imports-appworld.json'
bindings = json.loads(binding_path.read_bytes())
assert len(bindings['results']) == 2
for worker in bindings['results']:
    assert worker['FA_deterministic'] == '0'
    assert worker['critic_memory'] and all(v['activation_offload'] and v['native_max_seq_len'] == 8192
                                          for v in worker['critic_memory'])
    assert all(v.get('matches_prepared_source') is not False for v in worker['modules'].values())
stage_keys = ('gen', 'old_log_prob', 'ref', 'values', 'adv', 'update_critic', 'update_actor', 'step')
metrics = job['metrics']
means = {}
per_million_input_tokens = {}
for key in stage_keys:
    values = [v['values'].get('timing_s/'+key) for v in metrics]
    means[key] = statistics.mean(values) if all(v is not None for v in values) else None
    tokens = [v['values'].get('perf/total_num_tokens') for v in metrics]
    per_million_input_tokens[key] = (statistics.mean(t*1e6/n for t,n in zip(values,tokens))
                                      if all(t is not None and n is not None and n>0 for t,n in zip(values,tokens)) else None)
result = dict(version='appworld-native-PPO-32k-speed-20261016-v3', collected_unix=time.time(),
              source_commit=snapshot['queue']['source_commit'], job=job,
              mean_stage_seconds=means, mean_seconds_per_million_full_input_tokens=per_million_input_tokens,
              actual_worker_imports=bindings, snapshot_SHA256=hashlib.sha256((AUDIT/'latest-trial-snapshot.json').read_bytes()).hexdigest(),
              numerical_32768_receipt=str(ROOT/'receipts/PPO-32k-numeric-reference-20261016-v3/same-session-status.json'),
              cost_32768_receipt=str(ROOT/'receipts/PPO-32k-native-efficiency-20261016-v2/performance-exec2/status.json'),
              native_short_cost_receipt=str(ROOT/'receipts/PPO-32k-native-efficiency-20261016-v4/performance-exec3/status.json'),
              formal_training_mutations=0, method_or_training_hyperparameter_changes=[],
              limitations=['Two bounded native LOOP/PPO iterations, not formal240-trajectory whole-iteration throughput',
                           'Natural task context and synthetic exact32768 capacity/numerical tests remain separate',
                           'Native GAE adv time is not DT time; missing stages remain null',
                           'Full input tokens include environment observations; generation throughput requires generated tokens',
                           'Different natural rollouts cannot establish a controlled actor kernel speedup'])
(AUDIT/'complete-measurements.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(dict(completed_iterations=2, exit_code=0, mean_stage_seconds=means,
                      mean_seconds_per_million_full_input_tokens=per_million_input_tokens), indent=2))

"""Record the completed historical DT timing/import audit, without changing training."""
import hashlib
import json
from pathlib import Path
import time

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT=ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1'
RUN=ROOT/'runs/textcraft-dt-grpo-multiplier-no-whiten-20261014-v1'
paths={k:OUT/name for k,name in dict(
    historical='DT-historical-timing-versions-v48.json',
    actual='actual-DT-worker-bindings-v49.json',
    inputs='first-DT-request-comparison-v50.json',
    raw_rollout='native-first-rollout-metadata-v52.json').items()}
data={k:json.loads(p.read_bytes()) for k,p in paths.items()}
history=data['historical'];actual=data['actual'];inputs=data['inputs'];raw=data['raw_rollout']
expected={'qwen35_gdn_finite':'3f51f5f7569a4d6b410e7acef89635a50f2e6ecadfa43eb94d1fd4fdd655dfd1',
          'qwen35_dense_finite_runner':'198f78db4abf58cd8461c5bab86d1ce9c325a46e4295c8cc54d83bd9e20e59cd',
          'qwen35_decoder_finite':'44153c172bde57ccf15f26998c5b4c1a986621b11751ead71023a3a8e8404486',
          'accelerated.qwen35.qwen35_retained_capture':'5ad0bf9b8f9ec1868a20d964ad7a4a41c328f4da40c5d993f7ef92e8a5eaa810',
          'accelerated.qwen35.qwen35_code_local_capture':'993ae72a49dc90901bfcbb7817e5fd32ad6004fbe630a190b775c6f1e77fbe72'}
for w in actual['workers']:
    assert {n:w['modules'][n]['SHA256'] for n in expected}==expected
    assert all(w['flags'][k] for k in ('offload_replay_mixer','compile_gdn_scalar_rules','pin_replay_host',
        'fa_coefficient_suffix','gdn_coefficient_suffix','compact_gdn_captures','pin_root_host','reuse_native_prefix'))
    assert 'gate_destination' in w['functions']['retained_decoder']['names']
assert raw['runs']['old']['SHA256']==raw['runs']['new']['SHA256']
assert inputs['content_multiset_overlap']==135
old_hash={r['input_SHA256'] for r in inputs['runs']['old']['rows']}
positive={r['input_SHA256'] for r in inputs['runs']['new']['rows'] if r['multiplier']>0}
negative={r['input_SHA256'] for r in inputs['runs']['new']['rows'] if r['multiplier']<0}
assert positive.issubset(old_hash) and negative.isdisjoint(old_hash)
summary={}
for label,item in history['runs'].items():
    batches=[b for log in item['worker_logs'].values() for b in log['first_two_round_batches'] if b['round']==1]
    by_rank=[sum(b['seconds'] for b in log['first_two_round_batches'] if b['round']==1)
             for log in item['worker_logs'].values() if log['first_two_round_batches']]
    summary[label]=dict(requests=inputs['runs'][label]['summary']['count'],
        actual_causal_tokens=inputs['runs'][label]['summary']['length_sum'],
        padded_B4_tokens=sum(b['trajectories']*b['length'] for b in batches),
        B4_calls_both_ranks=len(batches),core_batch_seconds_by_rank=by_rank,
        max_causal_length=inputs['runs'][label]['summary']['length_max'],
        native_advantage_seconds=357.424 if label=='old' else 1095.178)
report=dict(unix=time.time(),status='completed_readonly_historical_version_and_workload_audit',
    sources={k:dict(path=str(p),SHA256=hashlib.sha256(p.read_bytes()).hexdigest()) for k,p in paths.items()},
    actual_workers=actual['workers'],same_first_rollout_bytes=True,unchanged_DT_imports_verified=expected,
    first_iteration=summary,added_negative_requests=inputs['new_negative'],retained_positive_requests=inputs['new_positive'],
    old_positive_requests_no_longer_traced=len(old_hash-positive),
    ratios={k:summary['new'][k]/summary['old'][k] for k in
        ('requests','actual_causal_tokens','padded_B4_tokens','native_advantage_seconds')},
    interpretation='Replacing raw-return with native group multipliers requires tracing nonzero negative coefficients on former zero-reward trajectories; these first-iteration trajectories are substantially longer. The original joint-target DT/capture/numerical version and accelerations remain active.',
    limits=['Token workload is an accounting quantity, not an exact wall-time model.',
        'No identical four-row B4 composition was available across these two first iterations; kernel-level performance equality is not proven.',
        'Different physical GPU pairs and native batch shapes prevent attributing every remaining timing difference.',
        'This timing/version audit does not establish or repair the old accelerating gradient-norm root cause.'],
    model_calls=0,DT_calls=0,optimizer_calls=0,training_parameter_changes=0,rootcause_repair_verified=False,
    rootcause_goal_status='active',hourly_automation='PAUSED')
dest=RUN/'runtime-overrides/DT-timing-version-audit-v51.json'
assert not dest.exists();dest.write_text(json.dumps(report,indent=2)+'\n')
# Add only a link to observed runtime evidence; keep launch/source snapshots immutable.
for name in ('formal-training.json','active-training.json','active-source.json'):
    p=ROOT/name;value=json.loads(p.read_bytes());matches=[j for j in value['jobs'] if j.get('pid')==1419849]
    assert len(matches)==1
    backup=RUN/'deployment-receipts'/(name+'.before-timing-v51.json')
    assert not backup.exists();backup.write_bytes(p.read_bytes())
    matches[0]['actual_DT_timing_version_audit']=str(dest)
    matches[0]['actual_DT_timing_version_audit_observed_unix']=report['unix']
    p.write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps(dict(receipt=str(dest),report=report)))

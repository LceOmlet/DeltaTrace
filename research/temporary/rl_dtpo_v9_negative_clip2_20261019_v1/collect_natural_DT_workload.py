"""Reuse the existing native readout-log reader for completed production rows.

Read original per-rank B4 workload reports. These are observed workload units,
not measured FLOPs or an alternative model/DT timing implementation.
"""
import ast
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
owner = HERE.parent / 'rl_dtpo_v9_clip_u_20261019_v1/audit_first_two_iterations.py'
assert hashlib.sha256(owner.read_bytes()).hexdigest() == 'ca25cfbe63539a7aac6dcb632cc2c856fbca9ce28d9f4254ce9dd066cc4a2b75'
text = owner.read_text()
template = next(n.args[0].value for n in ast.walk(ast.parse(text))
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                and n.func.id == 'run_python' and n.args
                and isinstance(n.args[0], ast.Constant))
reader = template[template.index('rank_reports=[]'):template.index('steps=[]')]
old = 'DT_calls=groups[:2],actor_RPC_cleanup=cleanup[:2]'
assert reader.count(old) == 1
reader = reader.replace(old, 'DT_calls=groups,actor_RPC_cleanup=cleanup')
sys.path.insert(0, str(HERE.parent/'rl_dtpo_v9_clip_u_20261019_v1'))
from transport import run_python

code = r'''
import json,hashlib,pathlib,runpy,time
from pathlib import Path
R=pathlib.Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
A=R/'receipts/dtpo-v9-negative-clip2-postgroup-variance-20261019-v4'
native_metric_source=R/'receipts/textcraft-gradient-rootcause-20261013-v1/original_plot_training_progress.py'
assert hashlib.sha256(native_metric_source.read_bytes()).hexdigest()=='c20559e38e651284859837735b8e7d15fe7ca59b090c62b7bc651228f6ba3607'
owner=runpy.run_path(str(native_metric_source),run_name='original_natural_DT_workload_reader')
job=json.loads((R/'receipts/textcraft-eight-gpu-v9-negative-clip2-20261019-v4/campaign-state.json').read_bytes())['jobs'][0]
snapshot=owner['collect'](str(R),[job['output']]);current,=snapshot['jobs'];assert current['pid_alive']
''' + reader + r'''
capacity_path=A/'root-pageable-native-capacity-results.json'
capacity=json.loads(capacity_path.read_bytes());assert capacity['status']=='completed'
excluded_capacity_readouts=[]
for row in rank_reports:
    matches=[c['value'] for c in capacity['rows'] if row['path'].endswith('-'+str(c['value']['pid'])+'.out')]
    expected,=matches
    signature={t['traj_uid'] for t in expected['native_report']['traces']}
    kept=[];excluded=[]
    for ordinal,group in enumerate(row['DT_calls'],1):
        uids={t['traj_uid'] for t in group['native']['traces']}
        if uids==signature:
            assert group['native']==expected['native_report']
            excluded.append(dict(source=group['source'],original_ordinal=ordinal,UIDs=sorted(uids),reason='Exact original capacity-receipt report, not a training iteration'))
        else:kept.append(group)
    assert len(excluded)==1
    excluded_capacity_readouts.extend(excluded);row['DT_calls']=kept
def original_B4_widths(report):
    traces=report['traces'];size=report['minibatch_size'];widths=[]
    for first in range(0,len(traces),size):
        values={t['compute_tokens'] for t in traces[first:first+size]}
        assert len(values)==1;widths.append(values.pop())
    assert len(widths)==report['finite_trace_calls']
    return widths
pair_path=A/'existing-DT-gpu-resident-pair-results.json'
pair=json.loads(pair_path.read_bytes());assert pair['successful_same_input_pair_count']==8
production_path=Path(job['run'])/'runtime-overrides/actual-worker-imports-small-GPU-resident-production-v1.json'
production=json.loads(production_path.read_bytes())
alignment_proofs=[]
for row in rank_reports:
    worker,=[r for r in production['results'] if row['path'].endswith('-'+str(r['pid'])+'.out')]
    paired,=[r for r in pair['same_input_pairs'] if r['rank']==worker['rank']]
    width21=original_B4_widths(row['DT_calls'][20]['native'])
    assert width21[paired['minibatch']-1]==paired['paired_shape'][1]
    width22=original_B4_widths(row['DT_calls'][21]['native'])
    counters=worker['small_GPU_resident_production']['counters']
    assert worker['small_GPU_resident_production']['actor_scheduler_epoch']==22
    small=[w for w in width22 if w<=2048]
    assert len(small)==counters['small_calls']
    assert len(width22)-len(small)==counters['offloaded_calls']
    assert max(small)==counters['largest_small_extent']
    alignment_proofs.append(dict(rank=worker['rank'],pid=worker['pid'],step21_B4_widths=width21,
        step21_validation_width=paired['paired_shape'][1],step22_B4_widths=width22,
        step22_counters=counters,step21_pair_source=str(pair_path),step22_counter_source=str(production_path)))
steps=[]
for metric in current['metrics']:
    step=metric['step']
    if step not in (14,15,16,17,18,20,22):continue
    groups=[row['DT_calls'][step-1] for row in rank_reports]
    raw=[row['native'] for row in groups]
    steps.append(dict(step=step,native_metric=metric,readouts=groups,
        finite_B4_calls_per_rank=[r['finite_trace_calls'] for r in raw],
        actual_DT_rows_including_original_padding=sum(r['trajectories'] for r in raw),
        unique_trajectory_uids=len({t['traj_uid'] for r in raw for t in r['traces']}),
        native_factual_padded_compute_token_slots=sum(t['compute_tokens'] for r in raw for t in r['traces']),
        readout_seconds_by_rank=[r['seconds'] for r in raw],
        passive_observation_present=step in (17,18),
        resource_mode='resident_small2048_plus_original_long' if step>=22 else 'original_root_pageable_mixer_offloaded'))
report=dict(status='completed_original_readout_workload_collection',collected=time.time(),run=job['run'],
    original_native_metric_reader=str(native_metric_source),original_native_metric_reader_SHA256=hashlib.sha256(native_metric_source.read_bytes()).hexdigest(),
    original_readout_reader_SHA256=READOUT_READER_SHA,original_readout_reader_selection='Existing first-two-iteration reader; remove the exact capacity-receipt UID/report on each worker, then select completed production step indices.',
    report_version='matched_capacity_report_v2',excluded_capacity_readouts=excluded_capacity_readouts,alignment_proofs=alignment_proofs,
    capacity_receipt=dict(path=str(capacity_path),SHA256=hashlib.sha256(capacity_path.read_bytes()).hexdigest()),
    latest_completed=owner['completed'](current),phase=current['phase_label'],steps=steps,
    source_snapshot=snapshot,
    scope='Original factual padded compute-token slots include original DP duplicates, and are not actual kernel FLOPs. Coefficient/query suffix lengths, target sizes, host allocation and communication also affect DT cost. Different realized workloads do not establish a controlled kernel speedup.',
    additional_model_calls=0,additional_DT_calls=0,training_mutations=0)
target=A/'natural-DT-production-workload.json'
if target.exists():
    prior=json.loads(target.read_bytes())
    if prior.get('report_version')!='matched_capacity_report_v2':
        previous_hash=hashlib.sha256(target.read_bytes()).hexdigest()
        prior.update(status='superseded_incorrect_step_linkage_not_accepted',original_draft_SHA256=previous_hash,
            correction='The draft treated the extra original capacity readout as a training iteration. Formal native metrics/plots were not changed. This corrected receipt removes exact capacity-report matches before linking workload rows to steps.')
        rejected=A/'natural-DT-workload-ordinal-draft-not-accepted.json'
        assert not rejected.exists()
        rejected.write_text(json.dumps(prior,indent=2)+'\n')
target.write_text(json.dumps(report,indent=2)+'\n');report['remote_receipt']=str(target);print(json.dumps(report))
'''
code = code.replace('original_readout_reader_SHA256=READOUT_READER_SHA',
                    'original_readout_reader_SHA256=' + repr(hashlib.sha256(owner.read_bytes()).hexdigest()))
report = json.loads(run_python(code, timeout=45))
target = Path('experiments/rl/results_dtpo_v9_negative2_natural_DT_workload_20261019_v4.json')
target.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(latest_completed=report['latest_completed'],steps=[dict(
    step=s['step'],DT_seconds=s['native_metric']['values']['timing_s/adv'],
    factual_padded_compute_slots=s['native_factual_padded_compute_token_slots'],
    B4_calls=s['finite_B4_calls_per_rank'],rows=s['actual_DT_rows_including_original_padding']) for s in report['steps']]),ensure_ascii=False))

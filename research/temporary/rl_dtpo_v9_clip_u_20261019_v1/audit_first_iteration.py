"""Audit a natural complete iteration using the fixed native log owner only.

This script reads saved logs/configuration. It never invokes a model, DT,
optimizer, environment, or profiler. Historical observations are kept separate.
"""
import datetime as dt
import json
from pathlib import Path

from transport import HERE, run_python


report = json.loads(run_python(r'''
import hashlib,json,runpy,time
from pathlib import Path
R=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
A=R/'receipts/dtpo-v9-clip-u-postgroup-variance-20261019-v2'
parser=R/'receipts/textcraft-gradient-rootcause-20261013-v1/original_plot_training_progress.py'
assert hashlib.sha256(parser.read_bytes()).hexdigest()=='c20559e38e651284859837735b8e7d15fe7ca59b090c62b7bc651228f6ba3607'
owner=runpy.run_path(str(parser),run_name='v9_complete_iteration_efficiency_read_only')
segment=R/'receipts/textcraft-eight-gpu-v9-clip-u-20261019-v2'
job=json.loads((segment/'campaign-state.json').read_bytes())['jobs'][0]
snapshot=owner['collect'](str(R),[job['output']])
current,=snapshot['jobs']
first,=[m for m in current['metrics'] if m['step']==1]
snapshot_path=A/'first-complete-iteration-source-snapshot.json'
if not snapshot_path.exists():
 snapshot_path.write_text(json.dumps(snapshot,indent=2)+'\n')
references=[]
for label,seg in [
 ('previous_v6_eight_GPU_first_iteration','textcraft-eight-gpu-sequential-20261018-v5'),
 ('previous_v8_eight_GPU_first_iteration','textcraft-eight-gpu-symmetric-probability-ratio-postgroup-variance-20261018-v8')]:
 path=R/'receipts'/seg/'dt-latest-source-snapshot.json'
 saved=json.loads(path.read_bytes())
 old,=saved['jobs']
 prepared=json.loads((path.parent/'dt-prepared.json').read_bytes())
 retained=dict(prepared,log=str(Path(prepared['output'])/'train.log'))
 native_records,_=owner['read_job_logs'](retained)
 native_metrics,_=owner['parse_logs'](native_records)
 metric,=[m for m in native_metrics if m['step']==1]
 saved_first,=[m for m in old['metrics'] if m['step']==1]
 assert metric['values']==saved_first['values']
 boundary=metric['sources'][0]['line']
 retained_path=Path(retained['log'])
 dt_records=[dict(path=str(retained_path),line=n,text=line)
  for n,line in enumerate(retained_path.read_text(errors='replace').splitlines(),1)
  if n<boundary and any(marker in line for marker in
   ('[DT direct target workload]','[DT direct joint minibatch]','[native_host_cache_batch]','[DeltaTrace readout]'))]
 references.append(dict(label=label,path=str(path),SHA256=hashlib.sha256(path.read_bytes()).hexdigest(),
  collected_at=saved['collected_at'],run=prepared['run'],metric=metric,options=prepared['options'],
  source_bindings=prepared['source_bindings'],original_DT_workload_log_records=dt_records))
stage_keys=['timing_s/gen','timing_s/reward','timing_s/old_log_prob','timing_s/ref',
 'timing_s/adv','timing_s/update_actor','timing_s/dump_rollout_generations',
 'timing_s/save_checkpoint','timing_s/step']
workload_keys=['data.train_batch_size','env.rollout.n','env.max_steps',
 'actor_rollout_ref.actor.ppo_mini_batch_size','actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu',
 'actor_rollout_ref.actor.ppo_epochs','actor_rollout_ref.rollout.max_model_len',
 'actor_rollout_ref.rollout.max_num_batched_tokens','actor_rollout_ref.rollout.max_num_seqs',
 'trainer.n_gpus_per_node','trainer.save_freq']
def summarize(metric):
 v=metric['values'];total=v['perf/total_num_tokens']
 stages={k:v.get(k) for k in stage_keys}
 additive=[v[k] for k in stage_keys if k not in ('timing_s/step',) and k in v]
 return dict(step=metric['step'],stages_seconds=stages,
  stage_sum_seconds=sum(additive),step_minus_stage_sum_seconds=v['timing_s/step']-sum(additive),
  total_native_input_tokens=total,response_length_including_tool_returns=v.get('response_length/mean'),
  tool_interaction_count=v.get('episode/tool_call_count/mean'),
  old_forward_ms_per_native_input_token=1000*v['timing_s/old_log_prob']/total,
  ref_forward_ms_per_native_input_token=1000*v['timing_s/ref']/total,
  actor_update_ms_per_native_input_token=1000*v['timing_s/update_actor']/total,
  native_metric_sources=metric['sources'])
current_summary=summarize(first)
for ref in references:
 ref['summary']=summarize(ref['metric'])
 ref['workload_comparison']={k:dict(current=job['options'].get(k),reference=ref['options'].get(k))
  for k in workload_keys}
 ref['actor_input_normalized_time_ratio_current_over_reference']=(
  current_summary['actor_update_ms_per_native_input_token']/ref['summary']['actor_update_ms_per_native_input_token'])
 ref['raw_stage_time_ratio_current_over_reference']={k:first['values'][k]/ref['metric']['values'][k]
  for k in ['timing_s/gen','timing_s/old_log_prob','timing_s/ref','timing_s/adv','timing_s/update_actor','timing_s/step']}
raw=[]
readouts=[]
log_root=Path(current['ray_session'])/'logs'
markers=('[DT direct target workload]','[DT direct joint minibatch]',
 '[native_host_cache_batch]','[DT direct target]','[DT batch')
for path in sorted(log_root.glob('worker-*.out')):
 for number,line in enumerate(path.read_text(errors='replace').splitlines(),1):
  if any(marker in line for marker in markers):
   raw.append(dict(path=str(path),line=number,text=line))
  if '[DeltaTrace readout] ' in line:
   native_json=line.split('[DeltaTrace readout] ',1)[1]
   decoded,_=json.JSONDecoder().raw_decode(native_json)
   readouts.append(dict(path=str(path),line=number,report=decoded))
assert len(readouts)==8, 'Audit exactly the first natural eight-rank DT call'
uids={trace['traj_uid'] for row in readouts for trace in row['report']['traces']}
dt_work=dict(rank_reports=len(readouts),B_per_rank=4,
 finite_B4_calls_per_rank=[r['report']['finite_trace_calls'] for r in readouts],
 computed_event_rows_including_original_DP_padding=sum(r['report']['trajectories'] for r in readouts),
 unique_computed_trajectory_uids=len(uids),
 native_factual_padded_compute_token_slots=sum(t['compute_tokens'] for r in readouts for t in r['report']['traces']),
 rank_RPC_seconds=[r['report']['seconds'] for r in readouts],
 original_readout_sources=[dict(path=r['path'],line=r['line']) for r in readouts])
releases={}
for row in raw:
 if '[native_host_cache_batch] ' in row['text']:
  native,_=json.JSONDecoder().raw_decode(row['text'].split('[native_host_cache_batch] ',1)[1])
  releases.setdefault(str(native['pid']),[]).append(dict(seconds=native['release_seconds'],
   path=row['path'],line=row['line']))
dt_work['native_B4_cache_release_by_worker']={pid:dict(total_seconds=sum(r['seconds'] for r in rows),records=rows)
 for pid,rows in releases.items()}
metadata=Path(job['run'])/'runtime-overrides/actual-worker-imports-after-first-update-v1.json'
binding=None
if metadata.exists():
 binding_record=json.loads(metadata.read_bytes())
 binding=dict(path=str(metadata),SHA256=hashlib.sha256(metadata.read_bytes()).hexdigest(),
  collected_at=binding_record['collected_unix'],
  workers=[{k:r[k] for k in ('rank','pid','birth','actor_optimizer_entries','actor_optimizer_step_values',
   'actor_scheduler_epoch','DT_CREDIT_MODE','DP_world_size')} for r in binding_record['results']],
  modules_match_prepared=all(m.get('matches_prepared_source') is not False
   for r in binding_record['results'] for m in r['modules'].values()))
 assert len(binding['workers'])==8
 assert all(r['actor_optimizer_step_values']==[4.0] and r['actor_scheduler_epoch']==1
  and r['actor_optimizer_entries']==496 and r['DT_CREDIT_MODE']=='reward_target_exponential_clipped_v9'
  and r['DP_world_size']==8 for r in binding['workers'])
report=dict(collected_at=time.time(),status='first_native_complete_iteration_observed',
 efficiency_equivalent_to_previous_eight_GPU_first_iteration=False,
 current_run=job['run'],driver=dict(pid=job['pid'],birth=job['birth']),
 method_source_commit=job.get('method_source_commit'),execution_source_commit=job['local_patch_commit'],
 upstream_commit=job['upstream_commit'],source_snapshot=str(snapshot_path),
 source_snapshot_SHA256=hashlib.sha256(snapshot_path.read_bytes()).hexdigest(),
 native_parser=str(parser),native_parser_SHA256=hashlib.sha256(parser.read_bytes()).hexdigest(),
 current=first,current_summary=current_summary,current_options=job['options'],references=references,
 original_DT_workload_log_records=raw,original_DT_readouts=readouts,DT_actual_work=dt_work,
 first_update_native_metadata=binding,
 additional_calls=dict(model=0,DT=0,optimizer=0,environment=0,profiling=0),
 limitations=[
  'Previous v6/v8 are separate methods/runs. Only unchanged workload and resource paths are compared; no quality conclusion.',
  'Native input-token normalization includes tool history and is not a pure kernel benchmark.',
  'DT adv wall time is not normalized by the full actor batch: zero-credit event filtering changes its actual work.',
  'First-iteration compilation is included where the native timer includes it; no unmeasured compile subtraction.',
  'Native stage timers are driver wall times; rank times and nested timers must not be added to them.',
  'Physical utilization time series and memory bandwidth are unavailable unless supplied by existing native observer logs.',
  'Native MFU estimator does not support Qwen3.5; its zero is not zero GPU computation.'
 ])
path=A/'first-complete-iteration-efficiency-audit.json'
path.write_text(json.dumps(report,indent=2)+'\n')
report['remote_receipt']=str(path)
print(json.dumps(report))
'''))

path = Path('experiments/rl/results_dtpo_v9_first_iteration_efficiency_20261019_v2.json')
report['collected_at_beijing'] = dt.datetime.fromtimestamp(
    report['collected_at'], dt.timezone(dt.timedelta(hours=8))).isoformat()
path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
print(json.dumps(dict(result=str(path), current=report['current_summary'],
    references=[dict(label=r['label'], summary=r['summary'],
        actor_input_normalized_time_ratio_current_over_reference=r['actor_input_normalized_time_ratio_current_over_reference'])
        for r in report['references']]), ensure_ascii=False))

reference = report['references'][1]
current = report['current_summary']['stages_seconds']
previous = reference['summary']['stages_seconds']
rows = []
for label, key in [('生成与环境', 'timing_s/gen'), ('old 前向', 'timing_s/old_log_prob'),
        ('ref 前向', 'timing_s/ref'), ('DT/优势构造', 'timing_s/adv'),
        ('actor 更新', 'timing_s/update_actor'), ('整轮', 'timing_s/step')]:
    rows.append(f'| {label} | {current[key]:.3f} | {previous[key]:.3f} | {(current[key]/previous[key]-1)*100:+.1f}% |')
release_totals = [x['total_seconds'] for x in report['DT_actual_work']['native_B4_cache_release_by_worker'].values()]
note = f'''# v9 执行 v2 首个完整迭代的效率核对

采集：{report['collected_at_beijing']}。当前 TextCraft DTPO 完成第1轮；该比较只核对效率，不比较训练质量。

| 阶段 | 当前 v9 秒 | 历史八卡 v8 首轮秒 | 当前变化 |
|---|---:|---:|---:|
{chr(10).join(rows)}

当前原生输入921230 token，历史922204 token，相差-0.106%；32组×8、B4/rank、global64、epochs1及每轮4次actor更新配置一致。阶段之和与整轮相差0.008秒，未发现计时之外的大段未解释等待。按原生输入token折算，actor更新仍慢17.1%，不能仅用轨迹长度解释。第1轮未触发save5，检查点时间缺失不填0。

DT实际处理149个唯一非零信用轨迹，原DP补齐为160行，八rank各5个B4，共40个B4。原readout报告的padding后计算长度槽位合计515792；它不是整批actor输入921230 token，不能用后者宣称DT单位工作量加速或减速。

自然首轮完成后的原只读metadata RPC已实读八rank：各496个Adam状态，全部step=4、scheduler=1、DP8及批准v9模式。已自然导入的readout、counterfactual、DT runner及actor/core源码均与准备绑定一致；dt_training_batch未在actor worker导入，不将其冒称为该worker已导入模块。此检查不调用模型或optimizer。

原生B4缓存释放计时各rank累计{min(release_totals):.3f}–{max(release_totals):.3f}秒，单次最长约27.4秒；历史保留日志同位置多为毫秒至0.1秒。该项在DT总时长内，不能再加到整轮上。当前资源配置使用原cudaHostAlloc及RPC内缓存，以避开已复现的MCR mcHostRegister限制；此首轮证明训练能通过该位置，同时暴露资源路径的实际耗时。尚不能将DT全部差额归因于这两个配置，也不能把未单独测到的编译时间扣除。

历史v6首轮868.221秒、DT150.903秒，亦作为独立参考保存在JSON。其方法及实际DT筛选负载与v9不同，未拼接历史曲线。原native MFU不支持Qwen3.5而输出0，不将其解释为零算力；没有现成连续物理利用率/显存带宽记录，不补估。

所有数值、配置、八rank原readout、缓存释放原行号及历史原日志路径见 [JSON记录]({path.resolve().as_posix()})。本次只读已有训练状态，新增模型、DT、optimizer、环境和profiling调用均为0。
'''
(HERE/'first-iteration-efficiency.md').write_text(note, encoding='utf-8')

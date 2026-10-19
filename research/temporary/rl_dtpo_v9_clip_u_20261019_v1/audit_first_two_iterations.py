"""Read the first two natural iterations; retain native timer/cleanup evidence."""
import datetime as dt
import json
from pathlib import Path

from transport import HERE, run_python


report = json.loads(run_python(r'''
import hashlib,json,psutil,runpy,time
from pathlib import Path
R=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
A=R/'receipts/dtpo-v9-clip-u-postgroup-variance-20261019-v2'
parser=R/'receipts/textcraft-gradient-rootcause-20261013-v1/original_plot_training_progress.py'
assert hashlib.sha256(parser.read_bytes()).hexdigest()=='c20559e38e651284859837735b8e7d15fe7ca59b090c62b7bc651228f6ba3607'
owner=runpy.run_path(str(parser),run_name='v9_first_two_native_iteration_audit')
job=json.loads((R/'receipts/textcraft-eight-gpu-v9-clip-u-20261019-v2/campaign-state.json').read_bytes())['jobs'][0]
snapshot=owner['collect'](str(R),[job['output']]);current,=snapshot['jobs']
metrics=[m for m in current['metrics'] if m['step'] in (1,2)]
assert [m['step'] for m in metrics]==[1,2], 'Wait for the real complete second iteration'
controller=json.loads((A/'controller-job.json').read_bytes())
alive=lambda pid,birth:psutil.pid_exists(pid) and psutil.Process(pid).create_time()==birth and psutil.Process(pid).is_running()
assert alive(controller['pid'],controller['birth']) and current['pid_alive']
source_snapshot=A/'first-two-complete-native-source-snapshot.json'
if not source_snapshot.exists():source_snapshot.write_text(json.dumps(snapshot,indent=2)+'\n')
timers=['timing_s/gen','timing_s/old_log_prob','timing_s/ref','timing_s/adv','timing_s/update_actor','timing_s/step']
reference=[]
for seg in ['textcraft-eight-gpu-sequential-20261018-v5','textcraft-eight-gpu-symmetric-probability-ratio-postgroup-variance-20261018-v8']:
 prep=R/'receipts'/seg/'dt-prepared.json';j=json.loads(prep.read_bytes())
 j['log']=str(Path(j['output'])/'train.log')
 records,_=owner['read_job_logs'](j);native,_=owner['parse_logs'](records)
 reference.append(dict(version=seg,run=j['run'],metrics=[m for m in native if m['step'] in (1,2)],
  prepared=str(prep),prepared_SHA256=hashlib.sha256(prep.read_bytes()).hexdigest()))
rank_reports=[]
for path in sorted((Path(current['ray_session'])/'logs').glob('worker-*.out')):
 groups=[];pending=[];cleanup=[]
 for number,line in enumerate(path.read_text(errors='replace').splitlines(),1):
  source=dict(path=str(path),line=number)
  if '[native_host_cache_batch] ' in line:
   payload,_=json.JSONDecoder().raw_decode(line.split('[native_host_cache_batch] ',1)[1])
   pending.append(dict(source=source,native=payload))
  if '[DeltaTrace readout] ' in line:
   payload,_=json.JSONDecoder().raw_decode(line.split('[DeltaTrace readout] ',1)[1])
   groups.append(dict(source=source,native=payload,B4_cache_releases=pending));pending=[]
  if '[native_host_cache] phase=update_actor ' in line:
   # Retain the existing owner's literal scalar and its full source line.
   cleanup.append(dict(source=source,text=line,
    release_seconds=float(line.rsplit(' release_s=',1)[1])))
 if groups:
  assert len(groups)>=2 and len(cleanup)>=2
  for group in groups[:2]:assert len(group['B4_cache_releases'])==group['native']['finite_trace_calls']
  rank_reports.append(dict(path=str(path),DT_calls=groups[:2],actor_RPC_cleanup=cleanup[:2]))
assert len(rank_reports)==8
steps=[]
for index,metric in enumerate(metrics):
 v=metric['values'];dt_rows=[r['DT_calls'][index] for r in rank_reports]
 uids={t['traj_uid'] for r in dt_rows for t in r['native']['traces']}
 cleanup=[r['actor_RPC_cleanup'][index]['release_seconds'] for r in rank_reports]
 b4_releases=[sum(x['native']['release_seconds'] for x in r['B4_cache_releases']) for r in dt_rows]
 refs=[]
 for ref in reference:
  old,=[m for m in ref['metrics'] if m['step']==metric['step']]
  ov=old['values'];refs.append(dict(version=ref['version'],native_metric=old,
   total_native_input_tokens_ratio=v['perf/total_num_tokens']/ov['perf/total_num_tokens'],
   stage_wall_time_ratios={k:v[k]/ov[k] for k in timers},
   actor_update_input_token_normalized_time_ratio=(v['timing_s/update_actor']/v['perf/total_num_tokens'])/(ov['timing_s/update_actor']/ov['perf/total_num_tokens'])))
 steps.append(dict(step=metric['step'],native_metric=metric,references=refs,
  unique_computed_trajectory_uids=len(uids),
  DT_computed_rows_including_padding=sum(r['native']['trajectories'] for r in dt_rows),
  DT_B4_calls_per_rank=[r['native']['finite_trace_calls'] for r in dt_rows],
  DT_reported_padded_compute_token_slots=sum(t['compute_tokens'] for r in dt_rows for t in r['native']['traces']),
  DT_B4_cache_release_seconds_by_rank=b4_releases,actor_RPC_cleanup_seconds_by_rank=cleanup))
candidate_path=R/'receipts/dt-native-consumed-root-lifetime-20261019-v1/prepared.json'
candidate=json.loads(candidate_path.read_bytes())
assert candidate['status']=='prepared_only_not_natively_validated_not_deployed'
assert all(str(p)!=candidate['candidate_root'] for p in [job['dt_root'],job['effective_environment']['DT_ROOT']])
report=dict(status='two_natural_native_iterations_completed_efficiency_still_under_review',
 collected=time.time(),run=job['run'],driver=dict(pid=job['pid'],birth=job['birth']),
 controller=dict(pid=controller['pid'],birth=controller['birth']),phase=current['phase_label'],
 completed=owner['completed'](current),checkpoint=current['checkpoint_step'],
 source_snapshot=str(source_snapshot),source_snapshot_SHA256=hashlib.sha256(source_snapshot.read_bytes()).hexdigest(),
 method_source_commit=job['method_source_commit'],execution_source_commit=job['local_patch_commit'],
 steps=steps,rank_native_evidence=rank_reports,reference=reference,
 prepared_resource_candidate=dict(path=str(candidate_path),SHA256=hashlib.sha256(candidate_path.read_bytes()).hexdigest(),record=candidate),
 training_DT_ROOT=job['dt_root'],candidate_deployed=False,
 additional_calls=dict(model=0,DT=0,optimizer=0,generation=0,profiling=0),
 limitations=[
  'Native stage timing includes the logged RPC cleanup; cleanup must not be added to the stage a second time.',
  'Per-rank cleanup durations overlap. They are not summed into wall time or subtracted to invent pure kernel time.',
  'Historical runs have different realized trajectory lengths and DT filtering, even when actor workload configuration is equal.',
  'The candidate changes no numerical expression but has no native performance/equality/capacity acceptance yet.',
  'Existing physical utilization series and memory bandwidth are missing; no zero or inferred utilization is substituted.'
 ])
out=A/'first-two-complete-iteration-efficiency-audit.json';out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
'''))

report['collected_at_beijing'] = dt.datetime.fromtimestamp(
    report['collected'], dt.timezone(dt.timedelta(hours=8))).isoformat()
path = Path('experiments/rl/results_dtpo_v9_two_iteration_efficiency_20261019_v2.json')
path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
rows = []
for step in report['steps']:
    v = step['native_metric']['values']
    rows.append(dict(step=step['step'],timers={k:value for k,value in v.items() if k.startswith('timing_s/')},
        total_tokens=v['perf/total_num_tokens'],DT_rows=step['DT_computed_rows_including_padding'],
        B4_calls=step['DT_B4_calls_per_rank'],
        actor_cleanup_range=[min(step['actor_RPC_cleanup_seconds_by_rank']),max(step['actor_RPC_cleanup_seconds_by_rank'])],
        reference_ratios=[dict(version=r['version'],wall_time=r['stage_wall_time_ratios'],
            actor_per_token=r['actor_update_input_token_normalized_time_ratio']) for r in step['references']]))
print(json.dumps(dict(result=str(path),collected=report['collected_at_beijing'],steps=rows,
    candidate_status=report['prepared_resource_candidate']['record']['status']),ensure_ascii=False))

second = report['steps'][1]
current = second['native_metric']['values']
reference = second['references'][1]['native_metric']['values']
table = []
for label, key in [('生成与环境','timing_s/gen'),('old 前向','timing_s/old_log_prob'),
        ('ref 前向','timing_s/ref'),('DT/优势构造','timing_s/adv'),
        ('actor 更新','timing_s/update_actor'),('整轮','timing_s/step')]:
    table.append(f'| {label} | {current[key]:.3f} | {reference[key]:.3f} | {(current[key]/reference[key]-1)*100:+.1f}% |')
text = f'''# v9 执行 v2：两次自然迭代的效率证据

采集：{report['collected_at_beijing']}。TextCraft DTPO已完成2/200，其他五组尚未开始。

第2轮的比较仍显示资源开销，不能把首轮差额全部归为首次编译：

| 阶段 | 当前v9第2轮秒 | 历史八卡v8第2轮秒 | 当前变化 |
|---|---:|---:|---:|
{chr(10).join(table)}

当前输入938251 token，历史961003 token，当前少2.37%；actor按输入token折算仍慢17.45%。当前DT实际118条唯一轨迹，补齐128行，八rank各4个B4，原readout报告的padding计算长度槽位合计438960。DT的筛选和padding负载与actor整批不同，表中DT比例仅是阶段时长比。

原生actor RPC末尾缓存释放，第1轮各rank15.274–17.386秒、第2轮14.923–16.386秒。第2轮DT的B4末尾缓存释放各rank累计32.031–42.552秒。这些值已包含在对应阶段内，各rank相互重叠，不能相加成额外wall time，也不能据此拼出一个未测的“纯内核耗时”。

代码核对发现，当前`DT_NATIVE_HOST_CACHE_REPLAY_LAYER=0`同时使已消费root/kwargs保留到B4结束。独立候选只在原runner中将消费后的张量释放与`torch._C._host_emptyCache`调用分开；保持原Torch缓存owner、原数值计算、flag=1和observer行为。候选SHA为`9f13c898c9faec72f57d77a81966315f25950c251e4381598b8f5442e287c5cf`，**仅准备、未原生验证、未部署**，不能宣称已加速或容量已验收。正式训练仍使用runner `6a112224`及执行提交`2875c40d`。

完整原生日志路径/行号、两轮八rank原readout、缓存释放记录、历史对应迭代和候选身份见 [JSON证据]({path.resolve().as_posix()})。本次读取和候选准备没有新增模型、DT、optimizer、生成或GPU profiling调用。
'''
(HERE/'two-iteration-efficiency.md').write_text(text,encoding='utf-8')

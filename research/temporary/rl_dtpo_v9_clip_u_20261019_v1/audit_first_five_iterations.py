"""Extend the existing two-iteration native evidence reader to the first save5 window."""
import datetime as dt
import json
from pathlib import Path

from transport import run_python

owner = Path(__file__).with_name('audit_first_two_iterations.py')
code = owner.read_text(encoding='utf-8').split("report['collected_at_beijing']", 1)[0]
for old, new in [
    ('in (1,2)', 'in range(1,6)'),
    ('==[1,2]', '==list(range(1,6))'),
    ('len(groups)>=2 and len(cleanup)>=2', 'len(groups)>=5 and len(cleanup)>=5'),
    ('groups[:2]', 'groups[:5]'),
    ('cleanup[:2]', 'cleanup[:5]'),
    ('first-two-complete-native-source-snapshot.json', 'first-five-complete-native-source-snapshot.json'),
    ('first-two-complete-iteration-efficiency-audit.json', 'first-five-complete-iteration-efficiency-audit.json'),
    ('two_natural_native_iterations_completed_efficiency_still_under_review',
     'five_natural_native_iterations_completed_first_shared_checkpoint_verified_efficiency_under_review'),
    ('Existing physical utilization series and memory bandwidth are missing; no zero or inferred utilization is substituted.',
     'Physical phase observations resumed within iteration5; a complete iteration activity average and memory bandwidth remain unavailable.'),
]:
    assert old in code, old
    code = code.replace(old, new)
namespace = dict(__file__=str(owner), __name__='existing_native_window_reader')
exec(compile(code, str(owner), 'exec'), namespace)
report = namespace['report']
report['reference_retained_cache_release_examples'] = json.loads(run_python(r'''
import json,re
from pathlib import Path
R=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
A=R/'receipts/dtpo-v9-clip-u-postgroup-variance-20261019-v2'
report=json.loads((A/'first-five-complete-iteration-efficiency-audit.json').read_bytes())
rows=[]
for reference in report['reference']:
 end=reference['metrics'][-1]['sources'][0]
 path=Path(end['path']);examples=[]
 for number,line in enumerate(path.read_text(errors='replace').splitlines()[:end['line']],1):
  if '[native_host_cache] phase=update_actor ' in line:
   match=re.search(r'\brelease_s=([0-9.]+)',line)
   if match:examples.append(dict(source=dict(path=str(path),line=number),text=line,release_seconds=float(match.group(1))))
 rows.append(dict(version=reference['version'],examples=examples,
  scope='Literal retained Ray echoes through the original fifth metric. Deduplicated echoes are not complete eight-rank measurements or reliably aligned per-iteration samples.'))
print(json.dumps(rows))
'''))
report['collected_at_beijing'] = dt.datetime.fromtimestamp(
    report['collected'], dt.timezone(dt.timedelta(hours=8))).isoformat()
phases = [('生成与环境', 'timing_s/gen'), ('old前向', 'timing_s/old_log_prob'),
          ('ref前向', 'timing_s/ref'), ('DT/优势构造', 'timing_s/adv'),
          ('actor更新', 'timing_s/update_actor'), ('共享检查点', 'timing_s/save_checkpoint'),
          ('rollout写出', 'timing_s/dump_rollout_generations'), ('整轮', 'timing_s/step')]
aggregates = []
for label, reference_index in [('当前v9', None), ('历史八卡v6', 0), ('历史八卡v8', 1)]:
    metrics = [s['native_metric'] if reference_index is None else
               s['references'][reference_index]['native_metric'] for s in report['steps']]
    aggregate = dict(label=label,steps=[m['step'] for m in metrics],
        input_tokens=sum(m['values']['perf/total_num_tokens'] for m in metrics),
        stage_seconds={key:sum(m['values'].get(key,0) for m in metrics) for _,key in phases})
    aggregate['rounding_or_uninstrumented_seconds'] = [
        dict(step=m['step'],seconds=m['values']['timing_s/step'] - sum(
            v for k,v in m['values'].items() if k.startswith('timing_s/') and k!='timing_s/step'))
        for m in metrics]
    aggregates.append(aggregate)
report['first_five_aggregate'] = aggregates
path = Path('experiments/rl/results_dtpo_v9_five_iteration_efficiency_20261019_v2.json')
path.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
# Retain exactly the same derived evidence beside its native sources on the shared mount.
run_python("from pathlib import Path\nPath("+repr('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/dtpo-v9-clip-u-postgroup-variance-20261019-v2/first-five-complete-iteration-efficiency-audit.json')+
           ").write_text("+repr(json.dumps(report,indent=2)+'\n')+")\n")
current, _, reference = aggregates
rows = [f"| {label} | {current['stage_seconds'][key]/5:.2f} | {reference['stage_seconds'][key]/5:.2f} | "
        + (f"{(current['stage_seconds'][key]/reference['stage_seconds'][key]-1)*100:+.1f}% |"
           if reference['stage_seconds'][key] else '— |') for label,key in phases]
actor_ratio=(current['stage_seconds']['timing_s/update_actor']/current['input_tokens'])/(
    reference['stage_seconds']['timing_s/update_actor']/reference['input_tokens'])
text=f'''# v9执行v2：首个完整save5窗口效率对账

采集：{report['collected_at_beijing']}。当前TextCraft DTPO v9完成5/200，共享原生checkpoint5已由八rank元数据读取器核验；其他五组尚未开始。正式训练继续。

以下均为各自原生第1–5轮的合计除以5，两组都包含第5轮保存成本。历史v8只作相同配置工作量的效率参照，不是本次新队列PPO/GRPO结果。

| 阶段 | 当前v9秒/轮 | 历史八卡v8秒/轮 | 耗时变化 |
|---|---:|---:|---:|
{chr(10).join(rows)}

当前5轮输入token合计{current['input_tokens']:.0f}，历史v8为{reference['input_tokens']:.0f}；按输入token折算actor更新耗时变化为{(actor_ratio-1)*100:+.1f}%。这是包含原生offload、同步和缓存释放的更新阶段，不能当作纯更新内核吞吐。DT的真实筛选、padding和计算长度另逐轮列在JSON中，不用actor整批token量替代DT负载。

原生日志中的所有阶段与整轮逐轮对账，差额为{[round(x['seconds'],6) for x in current['rounding_or_uninstrumented_seconds']]}秒。第5轮共享检查点耗时{report['steps'][4]['native_metric']['values']['timing_s/save_checkpoint']:.3f}秒，已经计入总计；缓存释放也已经计入对应RPC阶段。各rank耗时相互重叠，不跨rank相加或从阶段中扣除来虚构纯内核时间。

当前5轮actor RPC末尾的原生缓存释放，每rank每轮约14.92–17.58秒。历史v8保留的前5轮Ray回显，同名释放记录约0.31–0.39秒；原路径/行号和字面值已另存。这些历史回显受到Ray去重/异步发送影响，不是完整八rank逐轮计时，但能证实该原生资源回收调用本身的开销发生了明显变化。不能把缓存释放当作PG/backward/optimizer数学，也不能仅凭此将所有阶段差额归给分配器。

物理观察从第5轮中段恢复，原生phase归约记录见results_dtpo_v9_native_observation_20261019_v2.json；当前没有完整轮次的物理活动率平均值，显存带宽未记录。现有候选仍仅准备，未原生数值/性能验收或部署，当前正式DT runner保持6a112224。

原日志路径和行号、八rankDT原readout、B4/RPC清理时长及历史对应轮次见 [完整JSON]({path.resolve().as_posix()})。本次读取不增加模型、DT、optimizer、生成或GPU profiling调用。
'''
owner.with_name('five-iteration-efficiency.md').write_text(text,encoding='utf-8')
print(json.dumps(dict(result=str(path),collected=report['collected_at_beijing'],aggregates=aggregates,
    actor_per_input_token_ratio_vs_v8=actor_ratio,training_changes=0),ensure_ascii=False))

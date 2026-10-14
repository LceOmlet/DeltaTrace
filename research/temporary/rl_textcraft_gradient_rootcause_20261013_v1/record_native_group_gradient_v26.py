"""Record native DT group components without declaring a training repair."""
from datetime import datetime
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
status = json.loads((HERE/'native-DT-group-gradient-components-v26-status-final.json').read_bytes())['report']
assert status['job']['pid'] == 4118131 and not status['alive']
state = status['state']
assert state['phase'] == 'complete_native_gradient_chain_localization_not_repair'
assert all(state[k] == 0 for k in ('optimizer_updates','scheduler_updates','DT_calls','rollout_calls','validation_calls','GRPO_calls'))
records = [json.loads((HERE/'native-DT-group-gradient-components-v26'/f'iterationoriginal81_groups-rank{rank}-PG.json').read_bytes()) for rank in (0,1)]
labels = ['full_DT','same_reward_DT','other_groups_DT']
checks = []
for rank, record in enumerate(records):
    base = record['passes']['full_DT']['microbatch_losses']
    assert len(base) == 8
    for label in labels:
        rows = record['passes'][label]['microbatch_losses']
        assert len(rows) == 8
        entropy_difference = max(abs(a['weighted_entropy']['value']-b['weighted_entropy']['value']) for a,b in zip(base,rows))
        assert entropy_difference == 0 and all(x['ppo_kl'] == 0 and x['pg_clipfrac'] == 0 for x in rows)
        checks.append(dict(rank=rank,label=label,entropy_difference=entropy_difference,
            native_clip_return=record['passes'][label]['optimizer_boundary']['native_clip_return']))
assert records[0]['gradient_statistics'] == records[1]['gradient_statistics']
for chain in status['chain_reports']:
    c = chain['value']
    assert c['trainable_SHA256_before'] == c['trainable_SHA256_after']
    assert c['scheduler_before'] == c['scheduler_after'] == 80
stats = records[0]['gradient_statistics']
p = stats['inner_products']
a,b,c = labels
residual_squared = p[a+':'+a]+p[b+':'+b]+p[c+':'+c]-2*p[a+':'+b]-2*p[a+':'+c]+2*p[b+':'+c]
residual = math.sqrt(max(0,residual_squared))
now = datetime.now().astimezone().isoformat()
report = dict(observed_local=now,version='native-DT-group-gradient-components-v26',
    status='completed_native_group_partition_not_repair',goal_status='active',
    job=status['job'],actual_worker_bindings=state['bindings'],
    parameter_gradient_statistics=stats,forward_point_checks=checks,
    gradient_partition_residual_norm=residual,
    gradient_partition_residual_over_full=residual/stats['norms'][a],
    unchanged_state_receipts=status['chain_reports'],
    conclusions=[
        'In this unchanged late global64, the native DT gradient on sampled same-positive-reward groups has norm2.22119, versus2.22405 for full DT and0.298264 for the complementary groups.',
        'Original GRPO gives exactly zero advantages to these groups. No GRPO model or training was repeated.',
        'The three original forward points have identical recorded per-B4 entropy, zero PPO KL and zero clipping fractions. Native parameter/scheduler state is unchanged.',
        'The gradient sum is not numerically exact: the measured partition residual is0.0322125, or1.44837% of full norm. This is retained rather than assumed away or given a new pass tolerance.',
        'This identifies a material DTPO-specific signal at one late state. It does not establish why that signal grew across85 iterations or justify removing the groups or changing PLAN.'],
    optimizer_updates=0,scheduler_updates=0,DT_calls=0,rollout_calls=0,validation_calls=0,GRPO_calls=0,
    rootcause_resolved=False,accepted_repair=None,deployed_repair=None)
path = HERE/'native-DT-group-gradient-summary-v26.json'
path.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
entry = dict(observed_local=now,version=report['version'],status=report['status'],job=report['job'],
    actual_worker_bindings=state['bindings'],norms=stats['norms'],
    inner_products=stats['inner_products'],gradient_partition_residual=residual,
    record=str(path),rootcause_resolved=False,accepted_repair=None,deployed_repair=None)
for name in ('current_runtime.json','results_textcraft_gradient_rootcause_20261013_v1.json'):
    dest = ROOT/'experiments/rl'/name
    value = json.loads(dest.read_text(encoding='utf-8'))
    dest.write_text(json.dumps({'20261014_DTPO_group_gradient_v26_completed':entry,**value},indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
text = f'''## {now} 原DTPO组分项v26完成，根因未完成

诊断4118131/1791963025.77已完成退出，实际worker4121570/1791963048.51、4123090/1791963063.02；原actor3a65e173/corefc2f992b、checkpoint80/scheduler80及trainable前后保持。源9670d2c6/harness4ddc2fef。原iteration81第一global64不改输入，三次原PG反向所得全DT范数2.224052、同正奖励组2.221189、其余组0.298264；原GRPO对应同奖励组优势严格0，没有重跑GRPO。两rank各8个B4的三次entropy逐值相等、PPO KL与clipfrac全0。分项梯度相加残差范数0.0322125（全范数的1.44837%）保留，不称精确相加或引入新容差。

这是DTPO特有信号的晚期定位，不是85轮增长根因或修复验收；不据此屏蔽组或改Q/V/A/白化/原PPO。错token惩罚与奖励幅度的信用质量问题仍单独挂起。旧DTPO/分母候选停止、GRPO不重做、小时检查PAUSED；goal active，accepted/deployed repair均无。完整记录：{path}。

'''
dest = ROOT/'experiments/rl/RUNTIME_RECORD.md'
dest.write_text(text+dest.read_text(encoding='utf-8'),encoding='utf-8')
dest = ROOT/'experiments/rl/REMOTE_ENVIRONMENT.md'
dest.write_text(f'## {now} v26完成退出，无训练更新\n\n4118131/1791963025.77及诊断worker已退出；原PPO2/3与GRPO4/5不改，后续设备操作仍重查mx-smi。原owner/环境/缓存保持，小时检查PAUSED，goal active；实际绑定和范数范围见RUNTIME_RECORD顶部。\n\n'+dest.read_text(encoding='utf-8'),encoding='utf-8')
print(json.dumps(dict(record=str(path),norms=stats['norms'],partition_residual=residual,rootcause_resolved=False)))

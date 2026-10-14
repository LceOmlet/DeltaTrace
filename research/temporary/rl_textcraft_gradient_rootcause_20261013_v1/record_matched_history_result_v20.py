"""Record the completed native matched-history derivative comparison and limits."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
def read(name):
    value=json.loads((HERE/name).read_text(encoding='utf-8'))
    return value.get('report',value)

status=read('read_native_gradient_chain_v4-dt.json')
assert status['job']['pid']==3701818 and status['job']['birth']==1791959128.46
assert not status['alive']
assert status['state']['phase']=='complete_native_gradient_chain_localization_not_repair'
assert all(status['state'][k]==0 for k in ('optimizer_updates','scheduler_updates','DT_calls','rollout_calls','validation_calls','GRPO_calls'))
spec=read('prepare_action_observation_match_spec_v20.json')
coefficient=read('verify_same_target_derivatives_v20.json')
positions=read('audit_derivative_carrier_positions_v21.json')
assert coefficient['target_ID_sequences_exact_equal']
assert all(row['mismatched_active_tokens']==0 for row in positions['cases'])
cases=[];digests=[]
folder=HERE/'matched-action-observation-target-derivatives-v20'
for case in ('early','late'):
    records=[json.loads((folder/f'iteration{case}-rank{rank}-PG.json').read_bytes()) for rank in (0,1)]
    assert records[0]['gradient_statistics']==records[1]['gradient_statistics']
    chain=[r['value'] for r in status['chain_reports'] if r['value']['iteration']==case]
    assert len(chain)==2
    assert all(r['trainable_SHA256_before']==r['trainable_SHA256_after'] and r['scheduler_before']==r['scheduler_after']==5 for r in chain)
    digests.append([r['trainable_SHA256_before'] for r in chain])
    diagnostics=[]
    for rank,record in enumerate(records):
        losses=record['passes']['native_target']['microbatch_losses']
        assert len(losses)==8
        diagnostics.append(dict(rank=rank,native_microbatches=len(losses),
            ppo_kl=[r['ppo_kl'] for r in losses],
            clip_fraction=[r['pg_clipfrac'] for r in losses],
            native_clip_return=record['passes']['native_target']['optimizer_boundary']['native_clip_return']))
    cases.append(dict(case=case,checkpoint=5,actual_native_global_sharded_target_gradient_norm=records[0]['gradient_statistics']['norms']['native_target'],
        native_owners=records[0]['sources'],effective_configuration=records[0]['effective_config'],
        native_forward_diagnostics=diagnostics,unchanged_trainable_and_scheduler=chain))
assert digests[0]==digests[1]
ratio=cases[1]['actual_native_global_sharded_target_gradient_norm']/cases[0]['actual_native_global_sharded_target_gradient_norm']
now=datetime.now().astimezone().isoformat()
report=dict(observed_local=now,version='matched-action-observation-target-derivatives-v20',
    status='completed_native_diagnostic_not_training_repair',goal_status='active',job=status['job'],
    actual_worker_bindings=status['state']['bindings'],cases=cases,late_over_early_gradient_norm=ratio,
    same_target_coefficients=coefficient,position_owner_audit=positions,
    matched_facts=dict(pairs=15,initial_state_action_observation_ID_sequences_exact_equal=True,
        selection=spec['selection'],source_reasoning_counts=[dict(early=p['early'][0]['source_tokens'],late=p['late'][0]['source_tokens'])
            for p in spec['selected_tasks']]),
    conclusions=[
        'At the same original checkpoint5 and common per-target loss coefficients, differences between15 actual reasoning histories produce a28.27x target-gradient contrast while initial states, executed action IDs and tool-observation IDs/order match.',
        'Different target coefficients, target/action content, observations and evolved weights are not necessary for this finite contrast. This is stronger input localization than the earlier68x task-only comparison.',
        'The contrast measures all naturally differing reasoning-token values, lengths and placements together. It does not identify length alone, explain why training adopted those histories, establish a unique85-iteration root cause, or accept a repair.',
        'Valid positions agree with the original VERL position owner in all four v11/v20 carriers. The suspected position-reuse issue was not reproduced and does not justify a code change.',
    ],limits=spec['limits']+[
        'The global sharded diagnostic norm and native console mean-of-minibatch norms have distinct aggregation scopes.',
        'Common CPU target coefficients differ by at most5.82e-11 through native floating-point loss arithmetic, rather than exact bitwise equality.',
        'Neither this comparison nor late-state reweighting predicts training from scratch with a corrected denominator. Historical feedback remains open.',
        'The fixed-factual-suffix endpoint discrepancy remains excluded as a claimed causal-value error.'],
    optimizer_updates=0,scheduler_updates=0,DT_calls=0,rollout_calls=0,validation_calls=0,GRPO_calls=0,
    rootcause_resolved=False,accepted_repair=None,deployed_repair=None,
    diagnostic_source_commit=status['job']['diagnostic_commit'],
    recording_source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
dest=HERE/'matched-history-gradient-summary-v20.json'
dest.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
entry=dict(observed_local=now,version=report['version'],status=report['status'],goal_status='active',
    diagnostic_source_commit=report['diagnostic_source_commit'],job=report['job'],
    actual_worker_bindings=report['actual_worker_bindings'],
    early_target_norm=cases[0]['actual_native_global_sharded_target_gradient_norm'],
    late_target_norm=cases[1]['actual_native_global_sharded_target_gradient_norm'],
    late_over_early=ratio,coefficients=coefficient['coefficient_vector_L2_by_case'],
    position_mismatches=[r['mismatched_active_tokens'] for r in positions['cases']],
    record=str(dest.resolve()),rootcause_resolved=False,accepted_repair=None,deployed_repair=None)
for name in ('current_runtime.json','results_textcraft_gradient_rootcause_20261013_v1.json'):
    path=ROOT/'experiments/rl'/name
    old=json.loads(path.read_text(encoding='utf-8'))
    old={'20261014_matched_history_gradient_v20':entry,**old}
    path.write_text(json.dumps(old,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
texts={
    'RUNTIME_RECORD.md':f'''## {now} 相同动作/工具返回/系数的原生target梯度对比v20完成，goal active

诊断3701818/1791959128.46已完成并退出；physical0/1、source commit17a21a85，原harness838c5a1a。原两worker3705626/1791959156.87与3707293/1791959171.47实际导入actor3a65e173/corefc2f992b。两个case均加载原checkpoint5，496项trainable摘要在各case前后及case之间保持，scheduler5不变，无optimizer/scheduler/DT/rollout/validation/GRPO更新。

原保存历史中全部15对相同初始状态、相同完整action/target IDs与tool-observation IDs/顺序，各取首个chronological原样本；未按梯度/优势选择。两侧使用每对同一个late target损失系数，原global64/B4接口中另49行是相同原始事实输入、零诊断导数，不冒充64个独立匹配。原VERL CPU loss回读系数L2均约0.00776583，每对最大差5.82e-11；不是bitwise相等。原native preclip全分片target范数early0.01159884、late0.32794592，倍率{ratio:.6f}；原clip返回值分别保留。输入history差异在固定其余上述量时已足以造成这个有限对比，但不能把length、reasoning内容、相对位置分开归因，也不是完整85轮根因或修复验收。

代码检查曾提出载体position_ids沿用风险，v21只读用原compute_position_id_with_mask核对v11/v20四载体，所有有效token/target位置差异均0；未复现接线缺陷，不改代码，不撤回有效对比。端点差异仍不当真实因果价值误差。下一步集中实际优势方向如何推动history变化及其与原范数增长的联系，不启动长训或扩展不相关测试。原固定VERL/PPO、正式配置和方法保持；旧DTPO/分母候选停止、GRPO不重做、小时检查PAUSED。根因和修复仍未完成，accepted/deployed均无。完整回执：{dest.resolve()}。

''',
    'REMOTE_ENVIRONMENT.md':f'''## {now} v20有限原生反向已完成退出，PPO/GRPO保持

v20 3701818/1791959128.46完成退出，physical0/1后续使用仍重查mx-smi。实际worker3705626/3707293原actor/core/固定checkpoint5绑定及状态不变已回读，无训练更新；不是修复部署。原环境/缓存/VERL/PPO不改，旧DTPO/候选不恢复，正式GRPO4/5与已授权PPO2/3保持，小时检查PAUSED。v21原位置函数CPU审计未发现v11/v20有效位置差异，不能把未复现猜想当缺陷。详情见RUNTIME_RECORD顶部和matched-history-gradient-summary-v20.json；修复goal active。

'''}
for name,text in texts.items():
    path=ROOT/'experiments/rl'/name
    path.write_text(text+path.read_text(encoding='utf-8'),encoding='utf-8')
print(json.dumps(dict(record=str(dest.resolve()),ratio=ratio,rootcause_resolved=False)))

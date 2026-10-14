"""Persist completed conditional gain evidence and verified PPO restoration."""
from datetime import datetime
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
now=datetime.now().astimezone().isoformat()
native=json.loads((HERE/'native-common-update-feedback-v47-status-final.json').read_text())['report']
assert not native['alive'] and native['state']['native_optimizer_updates_executed']==8
account=json.loads((HERE/'account_common_update_feedback_v57-dt.json').read_text())['report']
chain=json.loads((HERE/'completed-common-update-chain-v58.json').read_text())
restore=json.loads((HERE/'record_native_PPO_restore_v59-dt.json').read_text())
dt=json.loads((HERE/'read_DT_group_status_v40-dt.json').read_text())['report']
latest=dt['native_metrics'][-1]
report=dict(observed_at=now,status='conditional_native_update_gain_comparison_complete_PPO_restored',
    diagnostic_job=native['job'],actual_native_arms=native['state']['observations'],
    feedback_accounting=account,chain_accounting=chain,PPO_native_restore=restore,
    latest_new_DT_native_metric=latest,latest_new_DT_metric_observed_unix=dt['unix'],
    rootcause_goal_status='active',rootcause_repair_verified=False,hourly_automation='PAUSED',
    conclusion='On original checkpoint80 and frozen original81 data/state, removing only the measured same-positive-group component reverses the subsequent fixed-adjoint VJP growth. This is direct conditional driver evidence, not a full85-iteration causal proof or acceptance of every aspect of the new method.',
    preserved_limits=account['limits'])
path=ROOT/'experiments/rl/results_textcraft_common_update_feedback_20261014_v1.json'
path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
runtime=ROOT/'experiments/rl/current_runtime.json';value=json.loads(runtime.read_text(encoding='utf-8-sig'))
key='20261014_completed_native_update_gain_v47_and_PPO_restore_v59'
summary=dict(observed_at=now,detailed_receipt=str(path),diagnostic_job=native['job'],diagnostic_status=native['state']['phase'],
    native_optimizer_updates=8,fixed_seed_growth=account['observed_local_feedback_signs'],
    before_VJP_relative_difference=account['before_VJP_relative_difference'],
    actual_native_net_step_cosine=account['native_four_step_net_displacement_cosine_between_arms'],
    latest_new_DT_native_metric=latest,PPO_restore=restore['report'],rootcause_goal_status='active',
    rootcause_repair_verified=False,hourly_automation='PAUSED')
runtime.write_text(json.dumps({key:summary,**{k:v for k,v in value.items() if k!=key}},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
text=f'''## {now} v47有限原生更新对照完成；PPO完整原生恢复已实际回读

诊断2209311/1791984676.94已退出，源49ed4686/main61e1f086；两臂各4笔原native optimizer和1次scheduler，前80/step320、后81/step324实际核对。原checkpoint80参数摘要两臂相同、原81完整256输入/optimizer history/RNG相同；不生成、不重新DT、不调用验证分数或GRPO模型，不恢复旧正式DTPO。full fixed-seed VJP1.10860283→1.18873406（+7.2281%），只去掉原128行同正回报组PG分量则1.11356949→1.04943170（−5.7597%）。两次before范数差0.4480%保留，它不是误差上界或容差。16份原BF16 seed的shape/dtype/SHA与hook应用norm均一致。

实际四笔净参数位移范数full0.0106721、ablation0.0099790、cos0.912093；不能把原生AdamW的非线性结果当梯度分项线性相加。full每B4梯度RSS增3.8876%、累积方向因子增3.2155%，ablation分别−6.3496%/+0.6299%。layer0参数VJP分别+24.642%/−13.760%，输入伴随量分别+4.345%/−14.402%。最后layer31的post-add/input增益不随总VJP同向：full略降、ablation略升，因此不把原先last-MLP历史sign变化直接当本次唯一原因。详细原数据和近似限制：{path.as_posix()}。这是原状态下共同分量影响更新方向及后续内部敏感性的直接条件证据；尚不宣称全部85迭代历史根因已充分证明、某条增长定律成立或新方法长期已修好。goal保持active。

PPO恢复PID{restore['report']['job']['pid']}/出生{restore['report']['job']['birth']}、physical2/3，同原run，原VERL resume_path checkpoint15；只改resume_mode/resume_from_path。两个实际worker的actor与critic均已回读optimizer step60/scheduler15，actor496与critic428条/rank；原actor3a65e173/corefc2f992b/worker9dfd6ce7/critic0de81db3/checkpointf65b6538保持。原DataLoader data.pt保留并由原trainer加载；训练RNG与actor rollout_gen_random_states保存在原extra_state，实际checkpoint manager的rollout RNG owner指向原rollout manager并有state。早期v55读了worker错误属性导致False，v56已在真正owner回读更正，这是诊断字段错误，不是训练RNG丢失。恢复核验：{restore['receipt']}。原日志15/330并进入下一轮生成；尚不把新完整迭代算作已完成。

新DT已观测{latest['step']}/330，最新reward={latest['values'].get('critic/score/mean')}、entropy={latest['values'].get('actor/entropy_loss')}、grad_norm={latest['values'].get('actor/grad_norm')}，原TaskRunner来源行号在上述JSON；原GRPO继续、其他项目6/7保持，小时自动检查仍PAUSED。DT耗时/版本审计v51完成：首轮原rollout字节完全一致，原加速/数值版本实际一致；151→236请求和新增长负乘子轨迹使实际token量3.12倍，不能仅凭请求数声称正常或声称底层算子等速。未引入额外训练改动。

'''
for name in ('RUNTIME_RECORD.md','REMOTE_ENVIRONMENT.md'):
    p=ROOT/'experiments/rl'/name;p.write_text(text+p.read_text(encoding='utf-8-sig'),encoding='utf-8')
print(str(path))

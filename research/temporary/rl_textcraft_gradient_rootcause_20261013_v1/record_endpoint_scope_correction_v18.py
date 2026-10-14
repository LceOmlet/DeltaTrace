"""Record the user's fixed-suffix counterfactual objection and narrow its claims."""
from datetime import datetime,timezone,timedelta
import hashlib
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
def read(name):
    value=json.loads((HERE/name).read_text(encoding='utf-8'))
    return value.get('report',value)
endpoint=read('read_native_source_values_v15.json')
direction=read('account_native_QV_action_direction_v12.json')
sensitivity=read('account_native_QV_action_direction_v12-dt.json')
counter=read('prepare_Action_source_counterevents_v17.json')
assert endpoint['state']['phase']=='complete_native_source_values_not_repair' and not endpoint['alive']
assert counter['status']=='prepared_only_not_submitted_CPU_prefix_and_probability_evidence_retained'
now=datetime.now(timezone(timedelta(hours=8))).isoformat()
result=dict(observed_local=now,version='endpoint-reference-scope-correction-v18',
    status='fixed_suffix_endpoint_discrepancy_is_not_causal_value_error_rootcause_unresolved',
    goal_status='active',recording_source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
    calculable_quantity='d_endpoint = sum over the same target positions of log p_theta(Y_j | factual prefix_j) minus the analogous sum with only source i replaced by EOS. All other saved IDs, future reasoning, tool observations and target positions are held fixed.',
    discrepancy='d_DT-d_endpoint is a discrepancy relative to this fixed-suffix model reference, subject to the native numerical differences below. It is not a measured causal world-return/Q/V error.',
    user_objection='Future reasoning and tool responses can retain information from the intervened token. Holding those descendants factual does not marginalize the continuation under the intervention.',
    teacher_forcing_distinction='Teacher forcing preceding Y tokens is normal when scoring that joint Y. The unresolved causal issue is conditioning on factual non-Y descendants/observations while claiming a marginal intervention or action-independent return baseline.',
    idealization_preserved='The agreed exact-model-estimate/world-variable premise is retained. This diagnostic did not establish that its fixed-future endpoint measures the corresponding world variable; no exact world simulator or zero approximation error requirement is introduced.',
    withdrawn_inferences=[
        'Do not call the21-point native reference values precise true counterfactual V values or their differences true DT value-estimation errors.',
        'Do not infer that a more accurate causal DT advantage would preserve the same preference or training gradient trend because a fixed-suffix readout retained nonzero V.',
        'Do not treat substitution of21 endpoint values at old weights/inputs with a fixed old whitening scale as a complete repaired objective or a prediction of fresh training.',
        'Do not prioritize additional endpoint-error experiments as a root-cause route without demonstrating their relevance to the observed accelerating parameter-gradient trend.'],
    retained_observations=dict(
        original_loss_finite_direction=dict(groups=len(direction['groups']),
            negative_V_favors_Action=sum(g['baseline_term_favors_Action'] for g in direction['groups']),
            Q_centering_direction_flipped=sum(g['baseline_term_changes_descent_from_negative_to_positive'] for g in direction['groups']),
            scope='Actual saved original DT coefficients through original fixed VERL loss at the old-policy point. Local logit directions, not a population-baseline proof or full parameter-update prediction.'),
        native_endpoint=dict(job=endpoint['job'],phase=endpoint['state']['phase'],alive_at_read=endpoint['alive'],
            summaries_by_rank=endpoint['summaries_by_rank'],
            interpretation='Fixed-suffix model endpoint observations only. Identity pair d reaches about0.43, so not an exact numerical oracle. Both ranks are repeats of the same points.'),
        reference_sensitivity=dict(
            scope=sensitivity['native_source_value_views']['scope'],
            by_rank={str(rank):dict(
                negative_V_favors_Action=sum(g['Action_logit_descent_direction'][f'native_reference_negative_V_rank{rank}']>0 for g in sensitivity['groups']),
                Q_centering_direction_flipped_B4=[g['B4'] for g in sensitivity['groups']
                    if g['Q_plus_centering_descent']<0<g['Q_plus_centering_descent']+g['Action_logit_descent_direction'][f'native_reference_negative_V_rank{rank}']])
                for rank in (0,1)},
            interpretation='Arithmetic sensitivity to the measured endpoints, not a causal-value-error correction.'),
        prepared_v17=dict(status=counter['status'],model_calls=0,
            query_points=counter['distinct_source_points'],submission='Not submitted; launcher does not accept v17.',
            all85_opposite_prefix_audit=counter['all85_observed_opposite_prefix_audit'])),
    research_priority='Return to the demonstrated original policy-gradient contributions and their increasing input-conditioned sensitivity; connect the actual DT advantage directions to the saved trajectory/context change and norm trend. Preserve uncertainty about causal attribution and historical denominator feedback. Do not substitute a value-error number for that argument.',
    accepted_DTPO_repair=None,deployed_DTPO_repair=None,original_VERL_PPO_changed=False,
    old_DTPO_resumed=False,GRPO_restarted=False,hourly_automation='PAUSED',
    new_GPU_or_training_calls_by_this_record=0,
    source_snapshots={name:dict(path=str(HERE/name),SHA256=hashlib.sha256((HERE/name).read_bytes()).hexdigest())
        for name in ['account_native_QV_action_direction_v12.py','account_first_Thought_baseline_roles_v13.py',
                     'observe_native_source_delete_values_v14.py','read_native_source_values_v15.py',
                     'prepare_Action_source_counterevents_v17.py','record_endpoint_scope_correction_v18.py']})
dest=HERE/'endpoint-reference-scope-correction-v18.json'
dest.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
runtime_path=ROOT/'experiments/rl/current_runtime.json'
runtime=json.loads(runtime_path.read_text(encoding='utf-8'))
runtime={'20261014_endpoint_reference_scope_correction_v18':result,**runtime}
runtime_path.write_text(json.dumps(runtime,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
result_path=ROOT/'experiments/rl/results_textcraft_gradient_rootcause_20261013_v1.json'
current=json.loads(result_path.read_text(encoding='utf-8'))
current.update(observed_local=now,status=result['status'],goal_active=True,endpoint_reference_scope_correction_v18=result)
result_path.write_text(json.dumps(current,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
text=f'''## {now} 用户指出固定事实后缀泄漏，端点差异解释纠正v18；goal active

本轮v15原生单EOS读出3376169/1791956103.54已完成并退出，两worker3381794/3383655实际原actor3a65e173/原checkpoint80、scheduler80及trainable摘要前后保持；无DT/反向/optimizer/rollout/验证调用。读出使用原DeltaTrace forward_root、PackedAnswerTargets与selected_target_log_probs，未复制模型或PPO。实际源commitfd1c27f6/SHA d337ad90；此前v14 d8d53abd在native_target_logit_rows导入阶段失败、未产生端点结果，现场保留。v12/v13/v15/v16是诊断版本，不是训练修复或正式部署。

用户质疑成立：所谓“误差”实际是固定事实未来reasoning/O/target位置、只替换一个source EOS的模型端点差异。后继可能保留被删token的信息，这不等于在干预后边缘化未来的真实回报反事实，更不自动是动作无关PPO baseline。对联合Y做teacher forcing本身与固定非Y后继/O作因果推断须区分。保留用户模型精确估计对应世界变量的理想化前提；本次测量没有证明所读端点就是那个变量，不能撤销前提或要求新world simulator。

21点native端点V中位数0.282/0.228，DT0.430，仅是上述模型参照；identity pair d最大0.435/0.408，不能称精确native真值。原DT系数经原VERL loss的14个有限同前缀组中12个negative-V项推Action、3个翻转Q+centering方向，此实际损失观察保留；用21端点数值替换该分项、固定旧白化的v16仍12个正向/各3个翻转，只是旧状态的算术敏感性，不能证明精确因果DT仍会如此或从头修复的训练结局。

v17只完成CPU反向样本前缀/概率账目与prepared-only资产，未提交GPU，launcher不接受v17。收住将端点误差当主要根因的路线；后续以原PG分项、真实输入/context变化及加速范数的联系为主，不新增价值误差测试或调参。原固定VERL保持，旧DTPO/分母候选停止、GRPO不重做、小时检查PAUSED；accepted/deployed repair均无，goal active。完整纠正：{dest}。

'''
path=ROOT/'experiments/rl/RUNTIME_RECORD.md'
path.write_text(text+path.read_text(encoding='utf-8'),encoding='utf-8')
path=ROOT/'experiments/rl/REMOTE_ENVIRONMENT.md'
path.write_text(f'''## {now} 端点诊断v15已完成，v17未提交，因果解释纠正

v15 3376169/1791956103.54已完成并退出，无训练更新；v17只有CPU prepared-only资产，未占GPU，launcher不接受。不是DTPO修复/正式部署；继续使用任何设备前重新核对mx-smi。原环境/缓存/VERL/PPO未改，旧DTPO/分母候选不恢复、GRPO不重做、小时检查PAUSED、goal active。固定事实后缀的native端点差异不当真实回报价值误差，详细范围见RUNTIME_RECORD顶部与endpoint-reference-scope-correction-v18.json。

'''+path.read_text(encoding='utf-8'),encoding='utf-8')
print(json.dumps(dict(path=str(dest),status=result['status'],goal_status='active'),ensure_ascii=False))

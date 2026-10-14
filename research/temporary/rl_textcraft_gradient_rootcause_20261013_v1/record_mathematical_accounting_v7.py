"""Synthesize completed observations without new model or training calls."""
from collections import defaultdict
from datetime import datetime,timezone,timedelta
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
def read(name):
    value=json.loads((HERE/name).read_text(encoding='utf-8'))
    return value.get('report',value)
chain=read('read_native_gradient_chain_v4.json')
exposure=read('account_target_exposure_temporal_v5.json')
state_groups=read('account_success_state_groups_v6.json')
force=read('account_native_target_force_v5.json')
matched=read('account_matched_task_changes_v7.json')
derivatives=read('summarize_historical_target_derivatives_v4.json')
assert chain['state']['phase']=='complete_native_gradient_chain_localization_not_repair'
assert not chain['alive']

adjoints={}
for step in (6,81):
    groups=defaultdict(lambda:{'input_sumsq':0.,'output_sumsq':0.})
    for block in chain['chain_reports']:
        value=block['value']
        if value['iteration']!=step:continue
        for row in value['records']:
            boundary=row['boundary'].split('.model.model.')[-1]
            groups[boundary]['input_sumsq']+=(row.get('input_adjoint_norm') or 0.)**2
            groups[boundary]['output_sumsq']+=(row.get('output_adjoint_norm') or 0.)**2
    adjoints[step]={k:{'input_norm':v['input_sumsq']**.5,'output_norm':v['output_sumsq']**.5} for k,v in groups.items()}
quantities=[]
for step in (6,81):
    observation=next(r for r in chain['state']['observations'] if r['iteration']==step)
    previous=next(r for r in derivatives['rows'] if r['iteration']==step and r['mini']==0)
    c=previous['output_score']['groups']['B4']['target_coefficient_L2']
    h=adjoints[step]['norm']['output_norm']
    g=observation['gradient_statistics']['norms']['native_target']
    quantities.append({'saved_iteration':step,'fixed_checkpoint':5,'target_coefficient_L2_C':c,
        'concatenated_native_head_adjoint_norm_H':h,'native_parameter_gradient_norm_G':g,
        'H_over_C':h/c,'G_over_H':g/h})
ratio={key:quantities[1][key]/quantities[0][key] for key in quantities[0] if key not in ('saved_iteration','fixed_checkpoint')}

tasks=defaultdict(list)
for entry in matched['iterations']:
    for row in entry['task_features']:tasks[row['task_key']].append(row)
paired=[]
for key,rows in tasks.items():
    a=[r for r in rows if 51<=r['iteration']<=70]
    b=[r for r in rows if 71<=r['iteration']<=85]
    if not a or not b:continue
    values={}
    for feature in ('success_count','positive_policy_count_median','positive_target_fraction_median',
                    'first_response_empty_Thought_fraction','first_response_chars_before_first_Action_median'):
        av=[r[feature] for r in a if r[feature] is not None]
        bv=[r[feature] for r in b if r[feature] is not None]
        if av and bv:values[feature]=(statistics.median(av),statistics.median(bv))
    paired.append(values)
late_pairs={}
for feature in paired[0]:
    values=[r[feature] for r in paired if feature in r]
    late_pairs[feature]={'n':len(values),'peak_window_median_across_tasks':statistics.median(a for a,b in values),
        'decline_window_median_across_tasks':statistics.median(b for a,b in values),
        'paired_difference_median':statistics.median(b-a for a,b in values),
        'increased':sum(b>a for a,b in values),'decreased':sum(b<a for a,b in values)}
force_shares=[]
for mini,values in force['native81_output_force_by_mini'].items():
    f=values['Action_after_empty_Thought']['L2_lower_squared']/values['all_target_tokens']['L2_lower_squared']
    force_shares.append({'mini':int(mini),'native_output_logit_squared_share_bounds':[f/(2-f),2*f/(1+f)]})

now=datetime.now(timezone(timedelta(hours=8))).isoformat()
source_names=['account_target_exposure_temporal_v5.py','account_native_target_force_v5.py',
    'account_success_state_groups_v6.py','account_matched_task_changes_v7.py',
    'audit_same_prefix_DT_values_v5.py','record_mathematical_accounting_v7.py']
result={'observed_local':now,'version':'mathematical-accounting-v7','status':'diagnostic_accounting_only_rootcause_unresolved',
    'goal_status':'active','goal_thread':'01a137fd-9599-7d60-9301-a0f129d5ec03',
    'user_constraint':'Outside DT advantages retain the original fixed verified VERL/PPO. Mathematical accounting must precede another repair or training experiment.',
    'original_fixed_actor_SHA256':'3a65e173300be82a7a9e056a96227c4f746eabc3be778ef41c8d50138d52ce6c',
    'original_fixed_core_SHA256':'fc2f992b16fb7fb23aebc683ad5f00136cf61fc9013cd426f5046983babe7299',
    'normalizer_candidate':'Unaccepted, undeployed and stopped. Historical reweighting is a diagnostic objective comparison; it is not an authorized replacement of fixed VERL or proof of the fresh-training outcome.',
    'counterfactual_scope_correction':'Large reaggregated target gradients at old late weights and inputs do not prove that starting with another denominator would have the same growth. Earlier weight and sampling-distribution feedback is not excluded.',
    'new_GPU_model_calls':0,'new_training_runs':0,'new_optimizer_or_scheduler_updates':0,'GRPO_reruns':0,
    'completed_prior_native_chain':{'job':chain['job'],'phase':chain['state']['phase'],'alive_at_read':chain['alive'],
        'actual_worker_bindings':chain['state']['bindings'],'source_commit':'cf665a3b66ab9321186b2cb5c4b4b4383d11c18f',
        'source_SHA256':'018189692d9a64db86dea650e257e9c78878fd7e5631c39f3e8df36f515427db',
        'quantities':quantities,'late_over_early_ratios':ratio,'boundary_adjoint_norms':adjoints},
    'native_85_exposure_windows':exposure['windows'],
    'matched_initial_tasks':{'native81_UID_calibration':matched['native81_UID_calibration'],
        'unique_decoded_initial_states':matched['unique_task_states'],
        'early1_30_to_late61_85':matched['matched_task_summary'],
        'peak51_70_to_decline71_85':late_pairs},
    'counterevidence_to_single_factor_claims':[
        'Native target actor coefficient falls, rather than rises: window1-10 median6.162 to81-85 median2.202. Raw-credit standard deviation rises. A collapsing whitening variance does not describe this evidence.',
        'The immediate Action-after-empty-Thought token contributes only the reported bounded share of old-policy output-logit squared force; it is not proven to dominate parameter gradients.',
        'Eight-success-state counts peak around61-70 and then decrease. They explain a DTPO/GRPO credit difference but cannot alone explain the later accelerating growth.',
        'The same-checkpoint6/81 derivative pair changes task composition as well as generated histories and coefficients. Matched-task length evidence does not retroactively turn it into a matched-task gradient pair.',
        'The earlier decoded-input grouping used a common system prompt. Its zero saturated-group counts are unavailable, not genuine zeros; corrected state grouping and native81 UID checks are preserved.',
    ],
    'immediate_Action_old_policy_output_force_shares':force_shares,
    'mathematical_accounting':{
        'native_gradient':'g = sum_i c_i u_i, u_i = grad_theta log pi(a_i|h_i), c_i = whitened_A_i/(16*N_B4(i)) for this native complete global64.',
        'squared_norm':'||g||^2 = sum_i c_i^2||u_i||^2 + 2 sum_{i<j} c_i*c_j*<u_i,u_j>. Coefficient energy alone cannot identify parameter norm or positive cross terms.',
        'factorization':'G = C * (H/C) * (G/H) is a descriptive identity on two saved derivative carriers at the same checkpoint5. The ratios are not independent causal percentages, a general growth law, or a repair test.',
        'target_source_gap':'For two r=1 sampled tokens, target coefficient minus source coefficient equals V_hat_source/sigma after the same whole-batch whitening. This is a sampled credit contrast; it does not itself prove a biased conditional expectation.',
        'GRPO_difference':'A native eight-row group with all rewards equal has zero outcome advantages. DTPO can retain nonzero target/source contrasts in that same group. This does not prove which part causes the temporal norm growth.',
        'unbiasedness_scope':'Exact deletion values require the PLAN coverage/action-independent expected baseline conditions to imply the original expected policy gradient. Target positivity or finite same-reward branch samples alone neither prove nor disprove these conditions.',
    },
    'remaining_question':'Identify why the original DT token advantages drive the broadly observed history/target-placement change, and which resulting score sensitivity or cross-token direction explains the norm acceleration. Do not substitute a VERL change, scalar coefficient ratio, finite same-reward observation, or one-update validation score for that causal argument.',
    'rootcause_resolved':False,'accepted_repair':None,'deployed_repair':None,
    'old_DTPO_resumed':False,'hourly_automation':'PAUSED',
    'sources':{name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in source_names},
    'evidence':{name:{'path':str(HERE/name),'SHA256':hashlib.sha256((HERE/name).read_bytes()).hexdigest()} for name in [
        'read_native_gradient_chain_v4.json','summarize_historical_target_derivatives_v4.json',
        'audit_same_prefix_DT_values_v5.json','account_target_exposure_temporal_v5.json',
        'account_native_target_force_v5.json','account_success_state_groups_v6.json','account_matched_task_changes_v7.json']}}
dest=HERE/'mathematical-accounting-v7.json';dest.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
runtime_path=ROOT/'experiments/rl/current_runtime.json'
runtime=json.loads(runtime_path.read_text(encoding='utf-8'))
runtime['20261014_native_gradient_chain_v4']['status']='completed_and_exited_native_gradient_chain_localization_not_repair'
runtime['20261014_native_gradient_chain_v4']['completion_receipt']=result['completed_prior_native_chain']
runtime={'20261014_mathematical_accounting_v7':result,**runtime}
runtime_path.write_text(json.dumps(runtime,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
result_path=ROOT/'experiments/rl/results_textcraft_gradient_rootcause_20261013_v1.json'
current=json.loads(result_path.read_text(encoding='utf-8'))
current['observed_local']=now;current['goal_active']=True
current['status']='mathematical_accounting_v7_original_VERL_fixed_rootcause_unresolved'
current['native_gradient_chain_v4']=runtime['20261014_native_gradient_chain_v4']
current['mathematical_accounting_v7']=result
result_path.write_text(json.dumps(current,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
for name,text in {
    'RUNTIME_RECORD.md':f'''## {now} 原固定VERL保持，既有85轮数学验算v7，goal active

用户再次明确除DT优势外使用原固定已验证VERL/PPO。分母候选仍未接受/部署、已停止；历史重归约只作旧状态诊断，不能据残余大梯度推断从头正确训练的结局。未启动新GPU/训练/GRPO或optimizer/scheduler更新。本轮只读已有85轮原数据。

已完成的v4原生边界定位1874426/1791942147.05已退出，实际两worker导入actor3a65e173…、scheduler5及trainable摘要保持。相同checkpoint5上旧iteration6/81 target载体的G为0.0132115/1.0539996；这是联合旧输入/系数变化，且任务组成不同，不是从头训练或同任务因果验收。没有证明某原生算子错误，不能据大adjoint改kernel。

85轮中成功轨迹首target前policy-token中位数从约140降至7，target数量大致保持，target占比从6.6%升至53.8%；原target白化系数6.162→2.202，raw std未塌缩。同初始任务匹配365个有正奖励轨迹的task状态，364个生成变短、365个target占比增加，排除了全由任务组成改变解释该长度变化。峰值51–70与71–85的320个匹配task中138个成功数下降、39个上升；273/311个正奖励task生成继续变短。均为观察统计，不把相关性写为完整根因。

相同前缀r=1时target/source训练系数差为V_hat_source/sigma；这解释有限样本的信用对比，但不自动证明条件期望有偏或其单独导致范数曲线。空Thought之后单个Action的旧policy输出logit力平方份额有界且不足以独占解释；全8成功组在奖励峰值后减少，也不能单独解释后期加速。梯度平方含跨token内积，系数平方和不是参数梯度比例。已纠正decoded输入仅为公共system prompt的无效任务分组，使用首policy之前的真实初始观测并在native81以32个UID各8行校准。

回执{dest}。分析版本v5/v6/v7均非训练版本；根因与修复未完成，accepted/deployed repair均无；旧DTPO/过早候选停止，GRPO不重做、小时检查PAUSED。本线程goal active，无实际需用户判断事项时继续自主推进。

''',
    'REMOTE_ENVIRONMENT.md':f'''## {now} 原生边界定位v4已完成退出，本轮只读旧数据

PID1874426/出生1791942147.05的v4已完成并退出，实际worker1879160/1791942185.51、1880564/1791942200.11的原actor绑定与状态回执已回读。两case无optimizer/scheduler/生成/DT/验证/GRPO更新，不是正式训练/根因验收。此后本轮仅CPU读取原85轮与已完成梯度，未新占GPU。后续选设备前重查实际物理占用。原固定VERL/PPO保持，分母候选未部署；旧DTPO/过早候选停止，正式GRPO未改，小时检查PAUSED，修复goal active。详细数学与版本回执见RUNTIME_RECORD顶部及mathematical-accounting-v7.json。

'''
}.items():
    path=ROOT/'experiments/rl'/name
    path.write_text(text+path.read_text(encoding='utf-8'),encoding='utf-8')
print(json.dumps({'path':str(dest),'quantities':quantities,'ratios':ratio,'rootcause_resolved':False,'goal_status':'active'},ensure_ascii=False))

"""Persist measured token meaning, native V evidence and rejected composition."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[4]

def binding(path):
    data=path.read_bytes()
    return dict(path=path.as_posix(),bytes=len(data),sha256=hashlib.sha256(data).hexdigest())

conditional_path=REPO/'experiments/rl/results_native_conditional_V_20261008.json'
candidate_path=REPO/'experiments/rl/results_factual_V_diagnostic_20261008.json'
conditional=json.loads(conditional_path.read_bytes())
candidate=json.loads(candidate_path.read_bytes())
contexts=json.loads((HERE/'selected-target-context.json').read_bytes())
observation_path=sorted(HERE.glob('factual-v-observation-*.json'))[-1]
observation=json.loads(observation_path.read_bytes())
assert observation['completed'] and not observation['driver']['same_birth']
assert observation['textcraft_same_birth'] and not any(observation['textcraft_release_present'])
assert all(row['phase']=='complete' for row in observation['ranks'])
assert candidate['baseline_original_symmetric_vector_equals_prior_replay']
assert all(candidate['exact_target_endpoint_arrays_equal'].values())
assert not candidate['actual_joint_residuals']['diagnostic_factual_V']['conservation_verified']
point=next(p for p in candidate['sampled_points'] if p['row']==3 and p['mode']=='most_negative')
assert point['diagnostic_factual_V']['opposite_sign_to_previous_single_delete']
assert point['diagnostic_factual_V']['A']<point['original_symmetric_memory']['A']
metadata=contexts['tasks']['textcraft']['original_post_execution_metadata']
assert metadata['hits'] and metadata['hits'][0]['entries'][0]['executed_payload']=='n format'
sample_path=HERE.parents[1]/'direct-target-credit-sample-20261007/v1/sample-analysis.json'
sample=json.loads(sample_path.read_bytes())
format_point=next(p for p in sample['tasks']['textcraft']['points'] if p['token']==' Format')
result=dict(
    status='Extreme credit branch diagnosis advanced; no accepted numerical repair',
    observable_need='Distinguish meaningful large negative credit from a false deletion estimate, and keep the independently verified AppWorld cache-lifetime repair separate.',
    selected_actual_tokens=dict(TextCraft_Format=format_point,AppWorld_newline=point),
    original_target_context=binding(HERE/'selected-target-context.json'),
    TextCraft_parser_evidence=metadata,
    target_interpretation='The original post-execution hook reports the normalized payload n format. Its source span lies in the model\'s quoted Action template. This is an actual original-parser submission, not a new target invented by the training adapter. It is not silently removed or reparsed.',
    original_counterfactual_reference=binding(sample_path),
    native_conditional_V_receipt=binding(conditional_path),
    conditional_V_summary=conditional['summary'],
    rejected_full_vector_receipt=binding(candidate_path),
    rejected_full_vector_summary=dict(
        selected_newline=point,
        actual_joint_residuals=candidate['actual_joint_residuals'],
        biased12_sign_disagreements=candidate['biased_sample_sign_disagreements'],
        resource=candidate['resources'],
        baseline_vector_exact_reproduced=True,target_endpoints_exact_equal=True),
    diagnosis='The original FLA conditional V operation is not the same object as the complete symmetric joint-deletion attribution. The actual one-token deletion and joint-reference coefficients use different contexts; the earliest measured sign reversal is inside GDN30. A local V replacement does not preserve the coupled joint finite identity and worsens the full outlier. This tested composition is rejected, not corrected by rescaling.',
    original_author_curve_receipt=binding(REPO/'experiments/rl/results_existing_memory_author_curves_20261008.json'),
    candidate_author_curves='Not run: this explicitly unaccepted composition already worsens the reproduced extreme token and loses the original joint identity. No overall attribution-quality claim is made; prior original-author curves are preserved.',
    memory_repair=dict(commit='799224868e0a9c0f8031b6012bb71505ab801a35',
        receipt=binding(REPO/'experiments/rl/results_memory_capacity_20261008.json'),
        status='Verified on the previously failed actual B4 and repeated exact32768 DT capacity with original vLLM lifecycle, not formally deployed. Consumed FA replay cache is released in its owner after use; original HF/PEFT and all finite mathematics remain unchanged.'),
    unchanged=dict(PPO=True,QVA_formula=True,whitening=True,target_selection=True,reward=True,
        LoRA_rank=8,LoRA_alpha=16,actual_microbatch_per_GPU=4,DT_batch_per_GPU=4,
        model_update_released=False,formal_restart=False,checkpoint_restore=False,
        clipping_or_multiplier_added=False,tolerance_changed=False),
    limitations=[
        'This is a deliberately selected high-impact sample, not a population error rate or a claim that all large negative estimates are invalid.',
        'Native deletion here validates the chosen model-text counterfactual. No new environment counterfactual simulator is required or claimed.',
        'Conditional V residual, original finite conservation diagnostics and author RISE/MAS are separate measurements. None is mislabeled as a new official FA/FLA whole-DT tolerance.',
        'The failed CPU metadata lookup presumed identical DT and actor ranks. The successful lookup uses original traj_uid after native balancing; this was a diagnostic lookup error, not evidence of a training token mismatch.',
    ],
)
target=REPO/'experiments/rl/results_credit_causal_diagnosis_20261008.json'
target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
current=dict(state='native_conditional_V_complete_factual_V_composition_rejected_credit_unrepaired',
    receipt=binding(target),conditional_receipt=binding(conditional_path),candidate_receipt=binding(candidate_path),
    native_launch=conditional['launch'],candidate_launch=json.loads((HERE/'factual-v-launch.json').read_bytes()),
    terminal_observation=binding(observation_path),
    imported_files=conditional['imported_files'],active_worker_owners=[r for r in candidate['imported_runners']],
    original_source_sha256=candidate['source_sha256'],
    current_native_conditional_V=conditional['summary'],selected_newline=point,
    actual_joint_residuals=candidate['actual_joint_residuals'],
    formal=dict(TextCraft_pid=2833207,TextCraft_birth=1791370325.16,
        update_release_present=[False,False],AppWorld_state='terminal_not_restarted',
        production_profile_changed=False,checkpoint_restore=False),
    unchanged_config=result['unchanged'],
    disposition='Do not install the factual-V diagnostic as a runner profile or fix. Original symmetric profile, native operator sources, targets, QVA, reward and upstream PPO are preserved. No new numerical tolerance, credit clipping or residual compensation.',
    credit_repaired=False,memory_repair=result['memory_repair'])
runtime=REPO/'experiments/rl/current_runtime.json';raw=runtime.read_bytes()
key='latest_conditional_V_and_physical_token_diagnosis'
assert key not in json.loads(raw) and raw.endswith(b'}\r\n')
tail='\r\n'.join(json.dumps({key:current},ensure_ascii=False,indent=2).splitlines()[1:-1]).encode('utf8')
runtime.write_bytes(raw[:-3].rstrip(b'\r\n')+b',\r\n'+tail+b'\r\n}\r\n');json.loads(runtime.read_bytes())
ledger=REPO/'experiments/rl/RUNTIME_RECORD.md'
heading='## 2026-10-08 条件V对照和完整向量结束：局部替换未修复信用，拒绝部署'
note='''
原生FLA条件V诊断9738653b/PID2700340/birth1791409158.24已结束。
保存的真实GDN30单删除V端点，其余输入固定事实值、incoming state相同。
原生条件效应+14.116234，原forward V系数乘真实差+14.112987，差-0.003247；
对称V项+5.508959，差-8.607275。输入FP16 q/k/v/beta、FP32 raw_g均已记录。
这是条件V，不是独立token奖励或完整advantage，不另设finite容差/通过判据。

54b398d8/PID2758971/birth1791409734.01的完整B4诊断也结束，4/5已回到859MiB。
仅保留原forward V系数、其他原对称分量不动；基线全向量与端点逐值复现。
换行d从-4.415554变-6.153260，A从-61.295715变-351.936005，仍与原生单删除
d=+23.089031/A=+0.75相反。联合残差从-1.350164变-182.973199，未保持原有限恒等式。
因此候选拒绝，未校正、未部署，也不继续花费新增作者曲线来宣称它有效。
此前作者累计删除/RISE/MAS回执保留；本次没有总体归因质量结论。
两rank各2次DT；105.26/105.32秒冷基线与66.88/66.92秒暖候选不称加速比。
物理峰值54196MiB，PSS峰值9.050/8.333GB，无OOM。

CPU原保存元数据核对确认TextCraft Format所影响的目标来自官方parser已经POST的
动作，归一化为n format，源自模型复述的Action格式示例。不是桥中新造target，
不自行删除或重写。该token原生删除使附近format目标概率0.1054升至0.9708，
完整d=-2.214573/A=-8.157497；正式DT为-3.170397/-22.816927，方向有据、幅度偏大。
按traj_uid查原actor rank0 row5；DT rank1不等于balance后的actor rank1。

79922486显存生命周期修复的真实失败B4/双次32768容量回执独立保留，尚未正式部署。
Text原birth继续hold、release不存在；App正式仍terminal，无optimizer/恢复/重启。
当前信用未修复；不能把条件分支局部接近原生当成全链修复，也不能用倍率掩盖残差。
实际源码路径/SHA、启动commit/PID创建时间、固定配置及回执绑定在current_runtime末字段。
'''
raw=ledger.read_bytes();assert heading.encode('utf8') not in raw
end=raw.index(b'\n')+1
ledger.write_bytes(raw[:end]+('\n'+heading+'\n\n'+note.lstrip()+'\n\n').encode('utf8')+raw[end:])
print(json.dumps(dict(receipt=binding(target),status=current['state'],TextCraft_parser_payload=metadata['hits'][0]['entries'][0]['executed_payload']),ensure_ascii=False))

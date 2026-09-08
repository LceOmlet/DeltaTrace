"""Prepare terminal evidence publication; no writes until root supplies review.

Four-case continuity: prior NI1/MH1 plus fixed-candidate NI0/MH0. NI1 cost
comparison uses only the four new measured ABBA calls, never the warmups.
"""
import ast,hashlib,json,re,time
from pathlib import Path
import numpy as np

A=Path(__file__).resolve().parent;R=A.parent/'DeltaTrace';entries=[]
sha=lambda b:hashlib.sha256(b).hexdigest()
PRIOR_JOB='dt_layer0_fla_endpoint_average_whole_pilot_20260908_v1'
NEW_JOB='dt_layer0_fla_endpoint_average_stability_cost_20260909_v1'
PRIOR_SUMMARY='dt_layer0_fla_endpoint_average_whole_pilot_summary_20260908.json'
REPORT='MAS有限FLA端点平均_四样本连续证据与NI1成本确认_20260909.md'

# Fill only after root reviews terminal results and the independent audit.
# Required: status,next,conclusion,headline,expected_result_status,
# results_sha256,summary_name,summary_sha256,summary_status.
# Results are verified; final interpretation remains pending root's same-source
# FT review. Use dict(VERIFIED_RESULT_FIELDS, status=..., next=...,
# conclusion=..., headline=...) when that review is ready. No completion claim.
VERIFIED_RESULT_FIELDS={'expected_result_status': 'layer0_FLA_endpoint_average_stability_cost_10DT245FLA84score_complete',
 'results_sha256': '2f06cbf84592ffd4a8ca6ef5d10726b3d07486c2324ad5344c4caa1b4ca9b126',
 'summary_name': 'dt_layer0_fla_endpoint_average_stability_cost_summary_20260909.json',
 'summary_sha256': '018790f35f83b15f401447a12a909592f009e81a5299a7c8aaa6c0d647bb9959',
 'summary_status': 'independent_whole_pilot_audit_passed'}
PUBLICATION_REVIEW=dict(VERIFIED_RESULT_FIELDS,
 status='partial_MAS_improvement_with_acceptable_NI1_cost_MH0_FT_comparison_pending',
 next='Retain the fixed explicit profile as a partial improvement. MH0 MAS only fell by 0.002429686 and its middle prediction error remains +38.418959. No same-case Qwen3.5/e81b3be FT MH0 score exists: the saved 0.267664-0.292056 range is MH1 and cannot be substituted. Next freeze one complete unmodified FT Both attribution, one current DT attribution and two original 21-point score curves on actual MH0; preselect FT0, retain other hop vectors without extra scoring. Preserve each method original target semantics and same fixed answer/evaluation inputs. No FT changes, new candidate, generation, layer scan or automatic expansion. Goal remains active.',
 conclusion='四个已使用开发样本的 MAS 均改善，且 NI1 预热后未观察到明显耗时退化；这确认了一个有效的局部改进，不等于 MAS 错配已经修好。MH0 的 0.405780→0.403351 只下降约 0.6%，中段仍过估 +38.42。现有同模型 FT 的 MoreHop 数值属于另一条 MH1；尚缺 MH0 的原版 FT 同例对照，不能据此声称胜出或完成 Goal。保留已验证修法与历史默认，实现提供显式入口，不升级为无条件正确的方法。',
 headline='四例 MAS 有改善、NI1 成本近似持平，但 MH0 的高 MAS 与大幅条件过估仍未解决；下一步补固定原版 FT0 同例对照。')
PROFILE_CONFIG='configs/qwen35_mas_repair.json'
PROFILE_FACTORY='research/runtime/qwen35_mas_runner.py'
PROFILE_SHA256='166bf57c8e3128e3e4f672f8f7010f4ef65df54b294dc5d5869178d322b7f0d3'
FACTORY_SHA256='06dc76c9f8eaf308645431374e5a8b099cbd6ee3c81ccd1b4fd3b50dd19d5297'


# Reuse exactly the publication helpers used for the preceding accepted export.
# Execute function definitions only, never either publisher's main or review.
for filename,names in [
    ('publish_dt_official_input_NI0_20260908.py',{'clean_text','clean','export','write_keep_newline'}),
    ('publish_dt_FLA_reference_and_average_20260908.py',{'verified_job','export_job'})]:
    tree=ast.parse((A/filename).read_bytes())
    defs=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names]
    assert {node.name for node in defs}==names
    exec(compile(ast.Module(body=defs,type_ignores=[]),'<unchanged publication helper definitions>','exec'))


def measured_cost(result):
    """Read the four actually measured NI1 calls; warmups stay in total budget."""
    rows=[row for row in result['runs'] if row['phase']=='measured']
    assert [(row['number'],row['case'],row['method']) for row in rows]==[
        (2,'niah_mq_q2_1','control'),(3,'niah_mq_q2_1','candidate'),
        (4,'niah_mq_q2_1','candidate'),(5,'niah_mq_q2_1','control')]
    assert all(row['status']=='complete' for row in rows)
    warm=[row for row in result['runs'] if row['phase']=='warm']
    assert [(row['number'],row['method']) for row in warm]==[(0,'control'),(1,'candidate')]
    output={'scope':'Only new NI1 measured rows2,3,4,5; C,S,S,C after C/S warmups. Two observations each, descriptive only.',
        'methods':{},'warmup_rows_excluded':[row['number'] for row in warm],
        'warmup_outer_seconds_in_total_budget':[row['outer_attribute_seconds'] for row in warm]}
    for method in ['control','candidate']:
        selected=[row for row in rows if row['method']==method]
        item={'numbers':[row['number'] for row in selected],
            'outer_seconds':[row['outer_attribute_seconds'] for row in selected],
            'runner_seconds':[row['details']['complete_attribution_seconds_with_diagnostics'] for row in selected],
            'peak_allocated_bytes':[row['details']['peak_allocated'] for row in selected],
            'peak_reserved_bytes':[row['details']['peak_reserved'] for row in selected],
            'resident_allocated_before_pair':[row['GPU_allocated_before_pair'] for row in selected],
            'resident_allocated_before_attribute':[row['GPU_allocated_before_attribute'] for row in selected],
            'allocated_after_cleanup':[row['GPU_allocated_after_cleanup'] for row in selected],
            'peak_minus_before_pair':[row['memory_cost']['peak_allocated_minus_before_pair'] for row in selected],
            'peak_minus_before_attribute':[row['memory_cost']['peak_allocated_minus_before_attribute'] for row in selected],
            'root_effect':[row['details']['root_effect'] for row in selected],
            'signed_sum':[row['details']['signed_sum'] for row in selected],
            'relative_residual':[row['details']['relative_residual'] for row in selected]}
        item['median_outer_seconds']=float(np.median(item['outer_seconds']))
        item['median_runner_seconds']=float(np.median(item['runner_seconds']))
        assert result['cost_summary'][method]['runs']==item['numbers']
        assert abs(result['cost_summary'][method]['median_outer_seconds']-item['median_outer_seconds'])<1e-12
        output['methods'][method]=item
    c=output['methods']['control'];s=output['methods']['candidate']
    output['candidate_minus_control_median_outer_seconds']=s['median_outer_seconds']-c['median_outer_seconds']
    output['candidate_minus_control_median_outer_percent']=100*(s['median_outer_seconds']/c['median_outer_seconds']-1)
    output['candidate_minus_control_median_peak_allocated_bytes']=int(np.median(s['peak_allocated_bytes']))-int(np.median(c['peak_allocated_bytes']))
    return output


def actual_budget(result):
    finite=result.get('finite_counts',{})
    return {'scope':'New stability/cost job only; includes warmups, quality calls, compilation, original scoring and any failure.',
        'wall_seconds':result['seconds'],'model_loads':result['model_loads'],
        'eager_initializations':result['native_eager_diagnostics'],
        'DT_entered':result['DT_entered'],'DT_returned':result['DT_returned'],
        'returned_DT_by_phase':{phase:sum(row['phase']==phase and 'details' in row for row in result['runs'])
            for phase in ['warm','measured','quality']},
        'scorer_entered':result['scorer_entered'],'scorer_returned':result['scorer_returned'],
        'native_decoder_replays_in_returned_ledgers':sum(row.get('counts',{}).get('native_decoder_replays',0) for row in result['runs']),
        'finite_counts':finite,'FT_calls':result['FT_calls'],'generation_calls':result['generation_calls'],
        'incomplete_native_calls':'unknown for nonreturned backend/attribute; never replace actual counts with planned budget',
        'publication_model_or_scorer_calls':0}


def four_case_records(prior,current):
    output={}
    for job,keys in [(prior,['niah_mq_q2_1','morehopqa_1']),(current,['niah_mq_q2_0','morehopqa_0'])]:
        result=job['result']
        for key in keys:
            case=result['cases'][key];methods={}
            for method in ['control','candidate']:
                curve=case['curves'][method];assert curve['status']=='complete'
                rise,mas,alignment=curve['return_metrics']
                methods[method]={'RISE':rise,'MAS':mas,'alignment_augmented_AUC':alignment,
                    'needle':curve['needle'],'gold_count':len(set(case['gold'] or [])&set(case['input']['keep'])),
                    'clean_native_score':curve['scores'][0],'allEOS_native_score':curve['scores'][20]}
            output[key]={'job':job['name'],'input':case['input'],'methods':methods,
                'candidate_minus_control':{name:methods['candidate'][name]-methods['control'][name] for name in ['RISE','MAS','alignment_augmented_AUC']},
                'candidate_minus_control_native_endpoints':{name:methods['candidate'][name]-methods['control'][name]
                    for name in ['clean_native_score','allEOS_native_score']},
                'fixed_control_masks':case['fixed_control_masks'],
                'source_results_sha256':job['summary']['results_sha256']}
    return output


def report_text(prior,current,review,cases,cost,budget):
    metrics=[];changes=[];fixed=[];endpoint=[]
    for key,case in cases.items():
        for method,item in case['methods'].items():
            needle='无 author gold'
            if item['needle'] is not None:
                count=item['needle']*item['gold_count'];assert abs(count-round(count))<1e-6
                needle=f'{round(count)}/{item["gold_count"]}'
            metrics.append(f'| {key} | {method} | {item["RISE"]:.9f} | {item["MAS"]:.9f} | {item["alignment_augmented_AUC"]:.9f} | {needle} |')
        delta=case['candidate_minus_control']
        direction=lambda x:'退步' if x>0 else ('改善' if x<0 else '相同')
        changes.append(f'- {key}：RISE {delta["RISE"]:+.9f}（{direction(delta["RISE"])}），MAS {delta["MAS"]:+.9f}（{direction(delta["MAS"])}）。')
        points=case['fixed_control_masks']['points']
        improved=sum(abs(row['candidate']['prediction_minus_actual'])<abs(row['control']['prediction_minus_actual']) for row in points if 0<row['step']<20)
        for row in points:
            if row['step'] in [1,10,20]:
                fixed.append(f'| {key} | {row["step"]} | {row["control"]["prediction_minus_actual"]:+.6f} | {row["candidate"]["prediction_minus_actual"]:+.6f} | {improved}/19 |')
        e=case['candidate_minus_control_native_endpoints']
        endpoint.append(f'| {key} | {e["clean_native_score"]:+.6f} | {e["allEOS_native_score"]:+.6f} |')
    measured=[]
    for method,item in cost['methods'].items():
        for i,number in enumerate(item['numbers']):
            measured.append(f'| {number} | {method} | {item["outer_seconds"][i]:.6f} | {item["runner_seconds"][i]:.6f} | {item["peak_allocated_bytes"][i]} | {item["peak_reserved_bytes"][i]} | {item["resident_allocated_before_pair"][i]} | {item["peak_minus_before_pair"][i]} |')
    c=cost['methods']['control'];s=cost['methods']['candidate']
    return f'''# 固定 FLA 端点平均：四样本连续证据与 NI1 成本

{review['conclusion']}

当前结论仅为 `partial_verified_improvement`。四个固定样本的 MAS 下降不表示 MH0 已修好：MH0 的原 MAS 仍为 0.403351，同 control 中段删除输入的预测减实际仍为 +38.418959；NI0 早段仍为 +33.824864。FT 同源对照和后续判定由独立核对决定，本发布器不把局部收益升级为目标完成，也不安排额外 GPU 实验。

保持同一可选候选 `finite_fla_by_layer={{0: average}}` 与当前第 0 层 symmetric norm-gate。新增 NI0/MH0 只改变固定作者样本，全部有限公式、native FA/FLA、原评分器及目标保持前次冻结身份；没有根据新曲线选择层、权重或规则。NI1/MH1 是前一任务的已验证结果，NI0/MH0 是本次开发稳定性样本，不能称作独立留出集或普遍修复证据。运行时默认空映射不变。

| 样本 | 方法 | 原 RISE | 原 MAS | 原 alignment AUC | needle |
|---|---|---:|---:|---:|---:|
{chr(10).join(metrics)}

原 RISE/MAS 都是越低越好。candidate 减 control 的逐样本方向如下，保留所有退步：

{chr(10).join(changes)}

每个质量样本/方法使用自己的完整 signed 向量、FP32 evaluated 向量、排序和 21 个实际删除输入，由原作者函数返回全部指标。两次任务的原生分数不互换；也不把旧 FT 表重新解释为新任务的匹配对照。NI0 使用原始 588-token 管线，不能与旧 605-token chat-template 结果混用。MH0 的原输入、目标、实际 gold/跨度由作者缓存及原映射函数确定；没有 gold 时不填造 needle。

| 同次 control 删除输入 | step | control 预测减实际 | candidate 预测减实际 | 中间点绝对误差下降 |
|---|---:|---:|---:|---:|
{chr(10).join(fixed)}

这张表只在对应任务真实 control mask 上收缩两个新向量，零额外评分，不是另外计算的 MAS 曲线。端点和正负抵消继续保留，较好的局部或中段误差不保证整条原指标曲线改善。

| 同任务 candidate 减 control | clean 原生分数差 | all-EOS 原生分数差 |
|---|---:|---:|
{chr(10).join(endpoint)}

全部 B2 root effects、signed sums、相对残差与重复调用向量保存在账本和二进制 NPZ，不要求逐位相同。表中的 B1 scorer 端点差与归因 B2 root 漂移是不同量，不能混称完全一致或用其中之一覆盖另一项。

NI1 成本只取新任务的第 2、3、4、5 次调用，即 C/S/S/C；第 0、1 次 C/S warmup 完全排除在下面的比较统计之外，仍计入实际总资源。一个完整模型常驻，同样的峰值重置和原 runner 同步/CPU checkpoint 诊断保留，候选端点复制、额外 layer0 FLA 和六次均值操作均在计时内。没有 `empty_cache` 或按方法卸载模型。

| measured 调用 | 方法 | outer 秒 | runner 秒 | allocated 峰值 bytes | reserved 峰值 bytes | B2 前常驻 allocated bytes | 峰值减 B2 前 bytes |
|---|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(sorted(measured))}

当前交付配置记录为 [qwen35_mas_repair.json](../../configs/qwen35_mas_repair.json)，对应 [make_qwen35_mas_runner](../../research/runtime/qwen35_mas_runner.py)。该 factory 只组装已实测的配置；没有更改历史 runner 或其他 runtime 默认，也没有向生产归因提供删除状态 A。

每方法仅两次观察：control outer 中位数 {c['median_outer_seconds']:.6f} 秒，candidate {s['median_outer_seconds']:.6f} 秒，相差 {cost['candidate_minus_control_median_outer_seconds']:+.6f} 秒（{cost['candidate_minus_control_median_outer_percent']:+.3f}%）。runner 中位数分别为 {c['median_runner_seconds']:.6f}/{s['median_runner_seconds']:.6f} 秒。两者接近，不称作加速；candidate measured allocated 峰值增加 {cost['candidate_minus_control_median_peak_allocated_bytes']:,} bytes。这是本机、固定 NI1、同进程 ABBA 的描述性结果，不是总体速度、普遍内存上界或跨硬件保证；不混入前次冷编译时间来扩大差异。

新任务总墙时 {budget['wall_seconds']:.6f} 秒，实际 DT 进入/返回 {budget['DT_entered']}/{budget['DT_returned']}，scorer {budget['scorer_entered']}/{budget['scorer_returned']}。预算包含两个 warmup、四个 measured、四个质量 DT、模型加载、eager 初始化、编译、84 个原 scorer 和任何失败；详细真实后端/原生调用数见[决策账本](../../evidence/dt_FLA_average_stability_cost_decision_20260909.json)。既有 NI1/MH1 和机制诊断成本继续保留在前一任务 receipt，不重复计算成新资源。

下一步：{review['next']}

结果中的 `input_ids`/`target_ids` 来自已登记来源的公开作者基准，保留数字以便核验原输入、目标与删除哈希；这些 token ID 不是私有权重。签名数值 NPZ 直接按原二进制字节入 Git，manifest 标记 `byte_identical=true`。不备份私有 `.pt` 激活或模型权重，不展开大型 NPZ 为 JSON。

证据：[新任务原始结果](../../evidence/{NEW_JOB}/results.json)、[新任务独立审计](../../evidence/{current['summary_name']})、[前次四曲线结果](../../evidence/{PRIOR_JOB}/results.json)、[前次审计](../../evidence/{PRIOR_SUMMARY})、[机制与前次收益](MAS当前NI有限FLA参考内容错配与完整端点平均实测_20260908.md)。Goal 仍 active，发布脚本不会自动改为 complete。
'''


def main():
    review=PUBLICATION_REVIEW
    assert isinstance(review,dict),'Prepared only: root final review is required before any publication write.'
    for key in ['status','next','conclusion','headline','expected_result_status','results_sha256','summary_name','summary_sha256','summary_status']:
        assert isinstance(review.get(key),str) and review[key],key
    prior=verified_job(PRIOR_JOB,'layer0_FLA_endpoint_average_4DT98FLA84score_pilot_complete',PRIOR_SUMMARY)
    current=verified_job(NEW_JOB,review['expected_result_status'],review['summary_name'])
    assert sha((A/PRIOR_SUMMARY).read_bytes())=='d42462d955b2ea19568def9957f6e0ee1547ca7758e02ec76faa907fcd2304a4'
    assert current['summary']['status']==review['summary_status']
    assert sha((A/review['summary_name']).read_bytes())==review['summary_sha256']
    assert sha((current['directory']/'results.json').read_bytes())==review['results_sha256']
    for filename,want in prior['protocol']['files_sha256'].items():
        if filename!='study.py':assert current['protocol']['files_sha256'][filename]==want,filename
    r=current['result'];cost=measured_cost(r);budget=actual_budget(r);cases=four_case_records(prior,current)
    state_path=A/'research_state_20260906.json';state_raw=state_path.read_bytes();state=json.loads(state_raw)
    running=state.get('current_running_experiment')
    assert running is None or (running.get('name')==NEW_JOB and running.get('pid')==current['receipt']['pid']), 'Do not clear another running experiment.'
    authored={'research/reproduction_templates/dt_layer0_fla_endpoint_average_stability_cost_20260909.py':current['protocol']['files_sha256']['study.py'],
        'research/runtime/qwen35_dense_finite_runner.py':current['protocol']['files_sha256']['qwen35_dense_finite_runner.py'],
        'research/runtime/layer0_fla_endpoint_average_20260908.py':current['protocol']['files_sha256']['layer0_fla_endpoint_average_20260908.py'],
        'research/runtime/official_span_mapping.py':current['protocol']['files_sha256']['official_span_mapping.py']}
    authored[PROFILE_CONFIG]=PROFILE_SHA256;authored[PROFILE_FACTORY]=FACTORY_SHA256
    for rel,want in authored.items():assert sha((R/rel).read_bytes())==want,rel
    profile=json.loads((R/PROFILE_CONFIG).read_bytes())
    assert profile['norm_gate_rules']=={'0':'symmetric'} and profile['fla_endpoint_average_layers']==[0]
    assert profile['entrypoint']==PROFILE_FACTORY+':make_qwen35_mas_runner'
    assert profile['historical_runner_constructor_defaults_changed'] is False
    ast.parse((R/PROFILE_FACTORY).read_bytes())
    policy={'status':review['status'],'next':review['next'],'selected_profile':profile['profile'],
        'selected_profile_config':PROFILE_CONFIG,'selected_profile_entrypoint':profile['entrypoint'],
        'local_MAS_repair_status':'partial_verified_improvement','local_MAS_repair_complete':False,
        'overall_project_complete':False,'app_goal_status_at_publication':'active'}
    helpers=['build_dt_layer0_fla_endpoint_average_stability_cost_20260909.py',
        'analyze_dt_layer0_fla_endpoint_average_stability_cost_20260909.py',Path(__file__).name]
    for filename in helpers:ast.parse((A/filename).read_bytes(),filename=filename)
    decision={'status':review['status'],'next':review['next'],'conclusion':review['conclusion'],'goal_status':'active',
        'default_changed':False,'FT_changed':False,'native_FA_FLA_changed':False,'candidate_rule_changed':False,
        'local_MAS_repair_status':'partial_verified_improvement','local_MAS_repair_complete':False,
        'overall_project_complete':False,'app_goal_status_at_publication':'active','app_goal_tool_called':False,
        'selected_delivery_profile':{'config':PROFILE_CONFIG,'config_sha256':PROFILE_SHA256,'entrypoint':profile['entrypoint'],'factory_sha256':FACTORY_SHA256},
        'actual_budget':budget,'four_case_quality':cases,'NI1_measured_ABBA_cost':cost,
        'new_results_sha256':review['results_sha256'],'prior_results_sha256':prior['summary']['results_sha256'],
        'summary':review['summary_name'],'vectors_preserved_binary_byte_identical':True,
        'private_activations_or_weights_in_Git':False,'author_benchmark_token_ids_preserved':True}
    doc=report_text(prior,current,review,cases,cost,budget)
    # All terminal/source/receipt/review checks above precede any write.
    export_job(current)
    for filename in helpers:export(filename,'research/reproduction_templates/'+filename,True)
    export('dt_layer0_fla_endpoint_average_stability_cost_protocol_20260909.json',
        'research/reproduction_templates/dt_layer0_fla_endpoint_average_stability_cost_protocol_20260909.json',True)
    for rel in authored:
        raw=(R/rel).read_bytes();entries.append({'source_name':'repository_authored/'+rel,'path':rel,
            'private_original_sha256':sha(raw),'public_sha256':sha(raw),'byte_identical':True,'bytes':len(raw)})
    filename='dt_FLA_average_stability_cost_decision_20260909.json'
    (A/filename).write_text(json.dumps(decision,ensure_ascii=False,indent=2),encoding='utf-8');export(filename,'evidence/'+filename,True)
    write_keep_newline(R/'docs/history'/REPORT,doc)
    for rel,marker,link in [('README.md','**最新根因：','docs/history/'),
        ('docs/current_goal.md','**当前执行：','history/'),('docs/dt_optimization_priorities_20260908.md','**最新执行：','history/')]:
        path=R/rel;text=path.read_text(encoding='utf-8');start=text.index(marker);end=text.index('\n\n',start)
        write_keep_newline(path,text[:start]+marker+review['headline']+'** [四样本连续证据与 NI1 实测成本]('+link+REPORT+')。Goal 未完成。'+text[end:])
    for filename in ['qwen35_support_summary_20260908.json','official_data_resource_policy_20260907.json']:
        private=A/filename;obj=json.loads(private.read_bytes());obj['latest_FLA_average_stability_cost']=decision
        obj['current_DT_optimization_policy'].update(policy)
        private.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
        rel=('evidence/' if filename.startswith('qwen35') else 'configs/')+filename
        public=R/rel;obj=json.loads(public.read_bytes());obj['latest_FLA_average_stability_cost']=clean(decision)
        obj['current_DT_optimization_policy'].update(policy)
        write_keep_newline(public,json.dumps(obj,ensure_ascii=False,indent=2));raw=public.read_bytes()
        entries.append({'source_name':filename,'path':rel,'private_original_sha256':sha(private.read_bytes()),'public_sha256':sha(raw),'byte_identical':False,'bytes':len(raw)})
    path=R/'configs/pv_content_P1_development.json';obj=json.loads(path.read_bytes());obj['latest_FLA_average_stability_cost']=clean(decision)
    obj['current_DT_optimization_policy'].update(policy);write_keep_newline(path,json.dumps(obj,ensure_ascii=False,indent=2))
    (A/'state_recovery_20260908'/('before_FLA_average_stability_cost_'+str(time.time_ns())+'.json')).write_bytes(state_raw)
    state.update(current_running_experiment=None,latest_execution_status=review['status'],latest_FLA_average_stability_cost=decision,
        latest_report=REPORT,next_stage=review['next'],next_stage_status=review['status'],latest_priority_note=review['next'],
        next_execution_contract=review['next'],next_research_question=review['next'],latest_turn_verified_execution_budget=budget,
        last_goal_turn_classification={'classification':'progress','evidence':review['conclusion']})
    state['current_goal_revision_20260908']['sha256']=sha((R/'docs/current_goal.md').read_bytes());state['deltatrace_repository']['worktree_clean']=False
    state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
    path=R/'evidence/export_manifest.json';manifest=json.loads(path.read_bytes());unique={item['path']:item for item in entries}
    manifest['artifacts']=[item for item in manifest['artifacts'] if item['path'] not in unique]+list(unique.values())
    for item in manifest['artifacts']:assert sha((R/item['path']).read_bytes())==item['public_sha256'],item['path']
    write_keep_newline(path,json.dumps(manifest,ensure_ascii=False,indent=2))
    print(json.dumps({'status':review['status'],'verified_artifacts':len(manifest['artifacts']),
        'exports':len(unique),'goal_complete':False,'new_model_or_scorer_calls':0}))


if __name__=='__main__':main()

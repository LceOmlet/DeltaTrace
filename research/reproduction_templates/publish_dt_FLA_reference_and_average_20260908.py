"""Publish both terminal/audited FLA jobs after root supplies its final review.

Prepared only: do not run while the whole-input pilot is active. This module is
safe to import; main() refuses all writes until PUBLICATION_REVIEW is supplied.
Reuse the same four publication helpers as publish_dt_NI_layer0_internal.
"""
import ast,hashlib,json,re,time
from pathlib import Path
import numpy as np

A=Path(__file__).resolve().parent;R=A.parent/'DeltaTrace';entries=[]
sha=lambda b:hashlib.sha256(b).hexdigest()
REFERENCE_JOB='dt_NI_fla_reference_mismatch_20260908_v1'
PILOT_JOB='dt_layer0_fla_endpoint_average_whole_pilot_20260908_v1'
REPORT='MAS当前NI有限FLA参考内容错配与完整端点平均实测_20260908.md'

# Root fills this only after inspecting the downloaded wholepilot and audit.
# Required fields: status, next, conclusion, headline, expected_pilot_status,
# pilot_results_sha256, pilot_summary_name, pilot_summary_sha256,
# pilot_summary_status. No quality or promotion decision is inferred here.
PUBLICATION_REVIEW={'status': 'layer0_FLA_reference_mismatch_localized_and_endpoint_average_two_case_MAS_improvement_verified',
 'next': 'Keep the candidate fixed as an explicitly selectable layer0 endpoint average, with the default '
         'unchanged. Next freeze a separate bounded confirmation using reusable original author NI0/MH0 '
         'inputs and actual gold/source references, plus NI1 two warmups and four measured C,S,S,C cost '
         'calls. Exact new budget remains subject to its own frozen protocol. No layer/weight search, new '
         'candidate, production access to deleted-state A, shadow model/FA/FLA, FT modification or '
         'extrapolated speed claim. Goal remains active.',
 'conclusion': '完整传播的收益已在固定 NI1/MH1 上实测并独立核验：原 MAS 分别从 0.196942 降至 0.163010、从 0.269516 降至 0.230350。NI 的原 '
               'RISE 从 0.037903 升至 0.039194，存在退步，needle 保持 37/38；MH 的原 RISE 从 0.152059 降至 0.134412。保留显式可选的第 '
               '0 层端点平均配置，默认实现不变。这是两个固定样本的完整指标证据，不是普遍修复、稳定成本或 Goal 完成的证明。',
 'headline': '第 0 层 FLA 参考内容错配已定位；完整端点平均在 NI1/MH1 降低原 MAS，NI RISE 仍退步，继续固定候选做小规模确认。',
 'expected_pilot_status': 'layer0_FLA_endpoint_average_4DT98FLA84score_pilot_complete',
 'pilot_results_sha256': 'dd2a40473f055a54ac72b6e282ef27f97d1f9f90a24af8eaa0e931d1ecff5252',
 'pilot_summary_name': 'dt_layer0_fla_endpoint_average_whole_pilot_summary_20260908.json',
 'pilot_summary_sha256': 'd42462d955b2ea19568def9957f6e0ee1547ca7758e02ec76faa907fcd2304a4',
 'pilot_summary_status': 'independent_whole_pilot_audit_passed'}

tree=ast.parse((A/'publish_dt_official_input_NI0_20260908.py').read_bytes())
defs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in
    ['clean_text','clean','export','write_keep_newline']]
assert len(defs)==4
exec(compile(ast.Module(body=defs,type_ignores=[]),'<same NI layer0 publication helpers>','exec'))


def verified_job(name,expected_status,summary_name):
    directory=A/'snapshot/tmp'/('codex_'+name)
    result=json.loads((directory/'results.json').read_bytes())
    protocol=json.loads((directory/'protocol.json').read_bytes())
    receipt=json.loads((directory/'terminal_receipt.json').read_bytes())
    summary=json.loads((A/summary_name).read_bytes())
    assert receipt['proc_exists'] is False and result['status']==expected_status
    assert result['protocol']==protocol
    for filename,item in receipt['files'].items():
        raw=(directory/filename).read_bytes()
        assert sha(raw)==item['sha256'] and len(raw)==item['bytes'],(name,filename)
    for filename,want in protocol['files_sha256'].items():
        assert sha((directory/filename).read_bytes())==want,(name,filename)
    assert sha((directory/'results.json').read_bytes())==summary['results_sha256']
    assert sha((directory/'protocol.json').read_bytes())==summary['protocol_sha256']
    vector_hash=sha((directory/'vectors.npz').read_bytes())
    assert vector_hash==result['vectors_sha256']==receipt['files']['vectors.npz']['sha256']
    if 'vectors_sha256' in summary:assert vector_hash==summary['vectors_sha256']
    # These are the reviewed signed numeric evidence arrays. Validate their
    # serialization without making millions of lines of JSON or loading .pt.
    with np.load(directory/'vectors.npz',allow_pickle=False) as arrays:
        assert arrays.files
        for key in arrays.files:
            value=arrays[key]
            assert value.dtype.kind in 'biufc' and np.isfinite(value).all(),(name,key)
    return {'name':name,'directory':directory,'result':result,'protocol':protocol,
        'receipt':receipt,'summary':summary,'summary_name':summary_name}


def export_job(job):
    name=job['name'];base='snapshot${ARTIFACT_ROOT}/codex_'+name+'/'
    for filename in ['results.json','terminal_receipt.json','driver.log']:
        export(base+filename,'evidence/'+name+'/'+filename,True)
    export(base+'vectors.npz','evidence/'+name+'/vectors.npz',False)
    assert entries[-1]['byte_identical'] and entries[-1]['public_sha256']==job['result']['vectors_sha256']
    for filename in list(job['protocol']['files_sha256'])+['protocol.json']:
        export(base+filename,'research/reproduction_templates/'+name+'/'+filename,True)
    export(job['summary_name'],'evidence/'+job['summary_name'],True)


def budget(reference,pilot):
    d=reference['result'];p=pilot['result'];s=reference['summary']
    f=p.get('finite_counts',{}).get('FLA_backend',{})
    return {'scope':'Two newly completed jobs only; previous layer0 capture cost remains in its historical receipt.',
        'reference':{'seconds':d['seconds'],'calls':s['counts'],'CPU_calls':s['CPU_counts'],
            'native_FLA_adjoint_stages':2*d['native_input_adjoints_returned'],
            'GPU_peak_allocated_bytes':s['GPU_peak_allocated_bytes'],
            'GPU_peak_reserved_bytes':s['GPU_peak_reserved_bytes']},
        'wholepilot':{'seconds':p['seconds'],'model_loads':p['model_loads'],
            'native_eager_initializations':p['native_eager_diagnostics'],
            'DT_entered':p['DT_entered'],'DT_returned':p['DT_returned'],
            'scorer_entered':p['scorer_entered'],'scorer_returned':p['scorer_returned'],
            'finite_counts':p.get('finite_counts',{}),
            'native_decoder_replays_in_returned_ledgers':sum(x.get('counts',{}).get('native_decoder_replays',0) for x in p['runs']),
            'partial_native_ledger':'unknown where attribute/backend did not return; do not substitute planned counts',
            'native_FLA_adjoint_stages_from_returned_backends':f.get('native_adjoint_stages_from_returned_calls'),
            'FT_calls':p['FT_calls'],'generation_calls':p['generation_calls']},
        'total_job_wall_seconds':d['seconds']+p['seconds'],
        'publication_model_or_scorer_calls':0}


def report_text(reference,pilot,review,actual_budget):
    s=reference['summary'];p=pilot['result'];early=s['points']['1'];mid=s['points']['10']
    terms=[('decay_reference_content','遗忘项的参考状态'),('key_write_reference_content','key 写入项的参考内容'),
        ('query_reference_content','query 读取的参考内容'),('key_read_reference_content','key 预读取的参考内容'),
        ('beta_reference_correction_content','beta 校正的参考内容'),('v_reference_difference','v 参考差')]
    table='\n'.join('| '+label+' | '+format(early['signed_terms'][name]['net'],'+.6f')+' | '+
        format(mid['signed_terms'][name]['net'],'+.6f')+' |' for name,label in terms)
    metrics=[]
    for case,entry in p['cases'].items():
        for method,curve in entry.get('curves',{}).items():
            if curve.get('status')!='complete':continue
            rise,mas,augmented=curve['return_metrics']
            needle='无 gold' if curve['needle'] is None else format(curve['needle'],'.6f')
            metrics.append(f'| {case} | {method} | {rise:.9f} | {mas:.9f} | {augmented:.9f} | {needle} |')
    metrics_table=('| 固定样本 | 方法 | 原 RISE | 原 MAS | 原 alignment AUC | needle |\n'
        '|---|---|---:|---:|---:|---:|\n'+'\n'.join(metrics)) if metrics else '该终止任务没有完整返回的原指标曲线。'
    fixed=[];improved=[]
    for case,entry in p['cases'].items():
        points=entry.get('fixed_control_masks',{}).get('points',[])
        if not points:continue
        count=sum(abs(x['candidate']['prediction_minus_actual'])<abs(x['control']['prediction_minus_actual']) for x in points if 0<x['step']<20)
        improved.append(f'{case}: {count}/19 个中间删除点的绝对误差下降')
        for point in points:
            if point['step'] not in [1,10,20]:continue
            c=point['control']['prediction_minus_actual'];a=point['candidate']['prediction_minus_actual']
            fixed.append(f"| {case} | {point['step']} | {c:+.6f} | {a:+.6f} |")
    fixed_table=('| 同一次 control 删除输入 | step | control 预测减实际 | candidate 预测减实际 |\n'
        '|---|---:|---:|---:|\n'+'\n'.join(fixed)) if fixed else '无完整 control mask 账本。'
    improved_text='；'.join(improved)
    costs=[]
    for run in p['runs']:
        details=run.get('details',{})
        if not details:continue
        costs.append(f"| {run['case']} | {run['method']} | {details['complete_attribution_seconds_with_diagnostics']:.6f} | {details['peak_allocated']} | {details['peak_reserved']} |")
    cost_table=('| 固定样本 | 方法 | 完整归因秒数 | peak allocated bytes | peak reserved bytes |\n'
        '|---|---|---:|---:|---:|\n'+'\n'.join(costs)) if costs else '没有返回完整成本账本。'
    return f'''# 当前 NI 有限 FLA：参考内容错配与完整端点平均

{review['conclusion']}

当前 control 保留第 0 层 symmetric norm-gate。诊断使用保存的真实 EOS/原输入/删除输入中间量，固定原输入端点的真实 native 伴随与 BF16 seed，将 EOS 参考和实际删除状态参考的有限公式作差。它没有重新评分，也没有用参考公式替换模型输出。

| 参考错配项，预测减实际 | 早段 | 中段 |
|---|---:|---:|
{table}

中段六项净和 {mid['reference_sum']['net']:+.9f}，保存的 FLA 总误差 {mid['saved_FLA_error']['net']:+.9f}。最大两项是 decay {mid['signed_terms']['decay_reference_content']['net']:+.6f} 与 key-write {mid['signed_terms']['key_write_reference_content']['net']:+.6f}。单独的指数割线条件曲率为 {mid['conditional_exp_secant']['net']:+.3e}；GPU 到 CPU 公式转移净值 {mid['replayed_to_CPU_transfer']['net']:+.6f}，重锚定配对余项 {mid['matched_pair_remainder']['net']:+.6f}，共同 B1/B2 转移 {mid['common_B1_B2_transfer']['net']:+.6f}，均分别保留。不能把全部 raw-g 误差都叫指数曲率，也不能据此把提高精度当作主修法。

早段 key-write 为 {early['signed_terms']['key_write_reference_content']['net']:+.6f}，FLA 总误差为 {early['saved_FLA_error']['net']:+.6f}，会抵消其他层内正项。中段总 token/head 正项 {mid['saved_FLA_error']['positive']:+.6f}、负项 {mid['saved_FLA_error']['negative']:+.6f}；净误差掩盖了大量抵消。上述项是固定伴随和明确参考选择下的代数分解，不是对单个输入词独立干预的因果识别。配对闭合、all-EOS 控制和较小转移项都不保证任意部分删除或整网 MAS。

唯一完整候选通过 `finite_fla_by_layer={{0: wrapper}}` 显式注入，只在第 0 层交换全部 11 个实际端点字段，并以同一个 native do/scale 调用原有限 FLA 两次，平均六个 FP32 系数，不加负号。其余 23 个 GDN、8 个 FA、MLP 和原 native/FT 不变。默认空映射的计算保持原样。本轮每个方法每个样本只调用一次完整 DT；批处理接口接受成对端点，但没有新增多样本实验。

{metrics_table}

原 RISE/MAS 越低越好。NI needle 的独立分母是 38，两个方法均恢复 37；MH 缺少本例 gold，不填造 needle。

每种方法使用自己的真实 FP32 evaluated vector、排序及 21 个删除输入，调用原作者函数返回全部三项指标。另保存两种向量在同一次 control 删除输入上的 signed sum 与实际分数差，零额外评分；它不是新的 MAS 曲线。历史 FT 值只作原始背景保留，不能替代本轮匹配的 FT 运行或被重新解读为跨运行胜出。

{fixed_table}

{improved_text}。NI 中段从 +25.499778 降至 +15.326082，但 all-EOS 端点误差从 +0.376884 升至 +0.516266，仍原样记录。更好的中间删除拟合不能替代完整原指标，端点守恒也不能单独作为质量判定。

{cost_table}

NI candidate 峰值 allocated 增加 39,418,368 bytes（37.59375 MiB），MH 相同。冷编译与不同首次调用位置混在本轮时间中；后续固定候选的独立热调用成本确认尚未在本次预算里执行，不从这四次调用推断稳定加速。

诊断实耗 {s['seconds']:.6f} 秒：两个原生 FLA 伴随阶段、一次原 mixed 图，外加如实登记的 CPU64 公式核算；没有模型加载、模型前向或 scorer。完整 pilot 实耗 {p['seconds']:.6f} 秒，实际进入/返回 DT 为 {p['DT_entered']}/{p['DT_returned']}，scorer 为 {p['scorer_entered']}/{p['scorer_returned']}。全部已知调用、失败、编译与初始化计入[联合决策账本](../../evidence/dt_FLA_reference_and_average_decision_20260908.json)。归因时间包含原 runner 的同步/诊断及候选真实额外工作，未拿配对诊断成本替代生产候选；每格一次、样本间反转方法顺序，只作描述，不提供稳定速度或普遍质量保证。

下一步：{review['next']}

Git 直接保存两个 `vectors.npz` 的原始二进制字节。诊断数值包为 {len((reference['directory']/'vectors.npz').read_bytes())} bytes，SHA256 `{reference['result']['vectors_sha256']}`，manifest 标记 `byte_identical=true`；没有展开成数百万行 JSON。原始私有激活和新增私有伴随 `.pt` 不入 Git，仅保留结果中的哈希/大小记录。

证据：[FLA 诊断原始结果](../../evidence/{REFERENCE_JOB}/results.json)、[诊断审计](../../evidence/{reference['summary_name']})、[wholepilot 原始结果](../../evidence/{PILOT_JOB}/results.json)、[wholepilot 审计](../../evidence/{pilot['summary_name']})、[上一步层内定位](MAS当前NI第0层内部_有限FLA为主要剩余项_20260908.md)。Goal 仍 active。
'''


def main():
    review=PUBLICATION_REVIEW
    assert isinstance(review,dict),'Prepared only: root must supply the audited wholepilot publication decision before any write.'
    for key in ['status','next','conclusion','headline','expected_pilot_status','pilot_results_sha256',
                'pilot_summary_name','pilot_summary_sha256','pilot_summary_status']:
        assert isinstance(review.get(key),str) and review[key],key
    reference=verified_job(REFERENCE_JOB,'NI_FLA_reference_content_native_capture_and_CPU_audit_complete',
        'dt_NI_fla_reference_mismatch_summary_20260908.json')
    pilot=verified_job(PILOT_JOB,review['expected_pilot_status'],review['pilot_summary_name'])
    assert reference['summary']['status']=='independent_NI_FLA_reference_ledger_audit_passed'
    assert pilot['summary']['status']==review['pilot_summary_status']
    assert sha((A/review['pilot_summary_name']).read_bytes())==review['pilot_summary_sha256']
    assert sha((pilot['directory']/'results.json').read_bytes())==review['pilot_results_sha256']
    state_path=A/'research_state_20260906.json';state_raw=state_path.read_bytes();state=json.loads(state_raw)
    running=state.get('current_running_experiment')
    assert running is None or (running.get('name')==PILOT_JOB and running.get('pid')==pilot['receipt']['pid']), 'Do not clear another running experiment.'
    # Check repository-authored files before copying/publication, including the
    # exact runner/API used by this completed pilot, never an old donor hash.
    authored={'research/runtime/qwen35_dense_finite_runner.py':pilot['protocol']['files_sha256']['qwen35_dense_finite_runner.py'],
        'research/runtime/layer0_fla_endpoint_average_20260908.py':pilot['protocol']['files_sha256']['layer0_fla_endpoint_average_20260908.py'],
        'research/reproduction_templates/dt_layer0_fla_endpoint_average_whole_pilot_20260908.py':pilot['protocol']['files_sha256']['study.py'],
        'research/reproduction_templates/dt_NI_fla_reference_mismatch_20260908.py':reference['protocol']['files_sha256']['study.py'],
        'research/reproduction_templates/NI_fla_reference_mismatch_cpu_20260908.py':reference['protocol']['files_sha256']['NI_fla_reference_mismatch_cpu_20260908.py']}
    for rel,want in authored.items():assert sha((R/rel).read_bytes())==want,rel
    helpers=['build_dt_NI_fla_reference_mismatch_20260908.py','analyze_dt_NI_fla_reference_mismatch_20260908.py',
        'build_dt_layer0_fla_endpoint_average_whole_pilot_20260908.py','analyze_dt_layer0_fla_endpoint_average_whole_pilot_20260908.py',
        Path(__file__).name]
    for name in helpers:ast.parse((A/name).read_bytes(),filename=name)
    actual=budget(reference,pilot)
    decision={'status':review['status'],'next':review['next'],'conclusion':review['conclusion'],'goal_status':'active',
        'default_changed':False,'FT_changed':False,'native_FA_FLA_changed':False,'new_quality_metrics':False,
        'reference_results_sha256':reference['summary']['results_sha256'],'wholepilot_results_sha256':review['pilot_results_sha256'],
        'reference_summary':reference['summary_name'],'wholepilot_summary':pilot['summary_name'],
        'actual_budget':actual,'private_activations_in_Git':False,'vectors_preserved_binary_byte_identical':True}
    doc=report_text(reference,pilot,review,actual)
    # All completion, source, receipt and root-decision checks precede writes.
    for job in [reference,pilot]:export_job(job)
    for name in helpers:export(name,'research/reproduction_templates/'+name,True)
    for name in ['dt_NI_fla_reference_mismatch_protocol_20260908.json','dt_layer0_fla_endpoint_average_whole_pilot_protocol_20260908.json']:
        export(name,'research/reproduction_templates/'+name,True)
    for rel in authored:
        raw=(R/rel).read_bytes();entries.append({'source_name':'repository_authored/'+rel,'path':rel,
            'private_original_sha256':sha(raw),'public_sha256':sha(raw),'byte_identical':True,'bytes':len(raw)})
    name='dt_FLA_reference_and_average_decision_20260908.json'
    (A/name).write_text(json.dumps(decision,ensure_ascii=False,indent=2),encoding='utf-8');export(name,'evidence/'+name,True)
    write_keep_newline(R/'docs/history'/REPORT,doc)
    for rel,marker,prefix,link in [
        ('README.md','**最新根因：','**最新根因：','docs/history/'),
        ('docs/current_goal.md','**当前执行：','**当前执行：','history/'),
        ('docs/dt_optimization_priorities_20260908.md','**最新执行：','**最新执行：','history/')]:
        path=R/rel;text=path.read_text(encoding='utf-8');start=text.index(marker);end=text.index('\n\n',start)
        text=text[:start]+prefix+review['headline']+'** [真实诊断、完整指标与预算]('+link+REPORT+')。Goal 未完成。'+text[end:]
        write_keep_newline(path,text)
    for filename in ['qwen35_support_summary_20260908.json','official_data_resource_policy_20260907.json']:
        private=A/filename;obj=json.loads(private.read_bytes());obj['latest_FLA_reference_and_average']=decision
        obj['current_DT_optimization_policy'].update(status=review['status'],next=review['next'])
        private.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
        rel=('evidence/' if filename.startswith('qwen35') else 'configs/')+filename
        public=R/rel;obj=json.loads(public.read_bytes());obj['latest_FLA_reference_and_average']=clean(decision)
        obj['current_DT_optimization_policy'].update(status=review['status'],next=review['next'])
        write_keep_newline(public,json.dumps(obj,ensure_ascii=False,indent=2));raw=public.read_bytes()
        entries.append({'source_name':filename,'path':rel,'private_original_sha256':sha(private.read_bytes()),
            'public_sha256':sha(raw),'byte_identical':False,'bytes':len(raw)})
    path=R/'configs/pv_content_P1_development.json';obj=json.loads(path.read_bytes());obj['latest_FLA_reference_and_average']=clean(decision)
    obj['current_DT_optimization_policy'].update(status=review['status'],next=review['next']);write_keep_newline(path,json.dumps(obj,ensure_ascii=False,indent=2))
    (A/'state_recovery_20260908'/('before_FLA_reference_and_average_'+str(time.time_ns())+'.json')).write_bytes(state_raw)
    state.update(current_running_experiment=None,latest_execution_status=review['status'],latest_FLA_reference_and_average=decision,
        latest_report=REPORT,next_stage=review['next'],next_stage_status=review['status'],latest_priority_note=review['next'],
        next_execution_contract=review['next'],next_research_question=review['next'],latest_turn_verified_execution_budget=actual,
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

"""Validate DT-paper primary metrics and repeated cost; export the completed comparison."""
import argparse,csv,hashlib,json,math,shutil,sys
from pathlib import Path
from collections import defaultdict
import numpy as np
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
ids_sha=lambda x:hashlib.sha256(np.asarray(x,dtype=np.int64).tobytes()).hexdigest()
def csv_read(p):
    with Path(p).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def csv_write(p,rows):
    with Path(p).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def dump(p,r):p.write_text(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def independent_recall(row,values,fraction):
    values=np.asarray(values)
    assert values.shape==(len(row['user_positions']),) and np.isfinite(values).all()
    if row['dataset'].startswith('vt_'):
        keep=sorted(row['keep']);scores=np.maximum(values.astype(np.float32),0)
        budget=max(1,min(len(keep),math.ceil(len(keep)*fraction)))
        ranked=sorted(keep,key=lambda j:(-float(scores[j]),j))
        selected=ranked[:budget];gold=set(row['gold'])&set(keep)
        return len(set(selected)&gold)/len(gold),budget,selected,None
    groups=row['all_groups'];units=row['units']
    order=sorted([j for j,u in enumerate(units) if u['kind']=='sentence' and groups[j]],
        key=lambda j:(-float(np.sum(values[groups[j]],dtype=np.float64)),units[j]['start']))
    budget=math.ceil(len(row['all_body_tokens'])*fraction)
    selected=[];tokens=[]
    for j in order:
        if len(tokens)+len(groups[j])>budget:break
        selected.append(j);tokens.extend(groups[j])
    assert len(tokens)==len(set(tokens))
    facts={(units[j]['title'],units[j]['sentence_index']) for j in selected}
    gold=set(map(tuple,row['official_keys']))
    return len(facts&gold)/len(gold),budget,tokens,selected

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','runtime','recovery','cost','export','output','audit'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    source=a.source/'repo_dynamic'
    full=a.runtime/'full_dynamic_with_ifr'
    main_protocol=read(source/'experiments/qwen35_comparison/protocol.json')
    recovery_protocol=read(source/'experiments/qwen35_comparison/paper_recovery/protocol.json')
    curve_audit=read(a.audit/'saved_curves_audit_full.json')
    assert curve_audit['complete_full_benchmark'] and curve_audit['cases']==1243
    prepared_path=a.runtime/'receipts/paper_recovery_inputs.json'
    prepared=read(prepared_path)
    recovery=read(a.recovery/'results.json')
    assert recovery['status']=='complete' and recovery['generation_calls']==0
    assert recovery['protocol_sha256']==sha(source/'experiments/qwen35_comparison/paper_recovery/protocol.json')
    assert recovery['prepared_sha256']==sha(prepared_path)
    assert recovery['environment_sha256']==read(full/'identity.json')['environment_sha256']==prepared['environment_sha256']
    assert recovery['vectors_sha256']==sha(a.recovery/'vectors.npz')
    prepared_by={(r['dataset'],r['index']):r for r in prepared['cases']}
    recovered={}
    with np.load(a.recovery/'vectors.npz',allow_pickle=False) as vectors:
        for r in recovery['cases']:
            pair=r['dataset'],r['index'];row=prepared_by[pair];key=f'{pair[0]}_{pair[1]}'
            assert pair not in recovered and r['status']=='complete'
            assert r['input_sha256']==row['input_sha256']==ids_sha(row['input_ids'])
            assert r['reference_sha256']==row['reference_sha256']==ids_sha(row['reference_ids'])
            assert r['target_offsets']==row['target_offsets'] and r['FT_target_weights']==row['target_weights']
            signed=vectors[key+'_DT_signed_full']
            assert signed.shape==(len(row['input_ids']),) and np.isfinite(signed).all()
            recovered[pair]={}
            for method in recovery_protocol['methods']:
                values=vectors[key+'_'+method+'_prompt']
                if method=='DT':
                    assert np.array_equal(values,signed[row['user_positions']])
                    actual=[ids_sha([row['reference_ids'],row['input_ids']])]
                else:actual=[row['input_sha256']]
                assert r['actual_inputs'][method]==actual
                got=independent_recall(row,values,recovery_protocol['tasks'][pair[0]]['fraction'])
                metric=r['metrics'][method]
                assert abs(got[0]-metric['recall'])<1e-15 and got[1]==metric['budget'] and got[2]==metric['selected_tokens']
                if got[3] is not None:assert got[3]==metric['selected_units']
                recovered[pair][method]=got[0]
    assert len(recovered)==448 and recovered.keys()==prepared_by.keys()
    export_verification=read(a.export/'verification.json')
    assert export_verification['status']=='complete' and export_verification['cases']==1243
    assert export_verification['legacy_vt_hotpot_recall_used'] is False
    assert export_verification['paper_matched_recovery']['results_sha256']==sha(a.recovery/'results.json')
    rows=[];means=[]
    for task,spec in main_protocol['tasks'].items():
        r=read(full/task/'results.json')
        assert r['status']=='complete' and len(r['cases'])==spec['count']
        data=[]
        for case in r['cases']:
            pair=task,case['index']
            methods=['DT','FT_K1','ifr-tokenwise']+(['FT_K3'] if case['gold'] else [])
            for method in methods:
                metric=case['metrics'].get(method,{})
                recall=(recovered[pair][method] if pair in recovered else
                    case['FT_K3_needle'] if method=='FT_K3' else metric['needle'])
                row=dict(dataset=task,index=case['index'],method=method,recall=recall,
                         rise=metric.get('rise'),mas=metric.get('mas'))
                data.append(row);rows.append(row)
        for method in methods:
            sample=[x for x in data if x['method']==method]
            mean=dict(dataset=task,count=len(sample),method=method)
            for name in ('recall','rise','mas'):
                vals=[x[name] for x in sample]
                assert all(x is None for x in vals) or all(x is not None and np.isfinite(x) for x in vals)
                mean[name]=None if vals[0] is None else float(np.mean(vals))
            means.append(mean)
    expected_means={(r['dataset'],r['method']):r for r in csv_read(a.export/'task_means.csv')}
    for row in means:
        other=expected_means[row['dataset'],row['method']]
        assert int(other['count'])==row['count']
        for name in ('recall','rise','mas'):
            assert (not other[name] and row[name] is None) or abs(float(other[name])-row[name])<1e-12
    cost=read(a.cost/'results.json')
    cost_protocol=read(a.cost/'protocol.json')['efficiency']
    assert cost['status']=='complete' and cost['generation_calls']==0
    assert cost['vectors_sha256']==sha(a.cost/'vectors.npz')
    assert cost['protocol_sha256']==sha(a.cost/'protocol.json')
    assert cost['driver_sha256']==sha(a.cost/'benchmark_complete_calls.py')
    assert cost['environment_sha256']==recovery['environment_sha256']
    assert len(cost['examples'])==16
    ownership=cost['gpu_ownership'];owner=cost['runtime']['pid']
    assert len(ownership)==33 and all(x['expected_pid']==owner and x['pids']==[owner] for x in ownership)
    assert ownership[0]['label']=='after_model_load'
    cost_quality={};cost_quality_vectors={}
    for task in dict(cost_protocol['selection']):
        quality=read(full/task/'results.json')
        cost_quality.update({(task,x['index']):x for x in quality['cases']})
        with np.load(full/task/'vectors.npz',allow_pickle=False) as qv:
            for selected_task,index in cost_protocol['selection']:
                if selected_task!=task:continue
                for method,suffix in [('DT','DT_signed_full'),('FT_K1','FT_K1_prompt'),('FT_K3','FT_K3_prompt')]:
                    key=f'{task}_{index}_{suffix}'
                    if key in qv:cost_quality_vectors[task,index,method]=qv[key].copy()
    cost_rows=[];cost_summary=[];comparisons=[]
    with np.load(a.cost/'vectors.npz',allow_pickle=False) as v:
        keys=set()
        for ordinal,case in enumerate(cost['examples']):
            assert [case['dataset'],case['index']]==cost_protocol['selection'][ordinal]
            assert case['status']=='complete' and len(case['calls'])==12
            label=f"{case['dataset']}_{case['index']}"
            assert [x['label'] for x in ownership[1+2*ordinal:3+2*ordinal]]==[label+'_before',label+'_after']
            quality=cost_quality[case['dataset'],case['index']]
            original=np.asarray(quality['input_ids'],dtype=np.int64)
            assert case['input_sha256']==quality['input_sha256']==ids_sha(original)
            # The independently tokenized recovery inputs identify the same
            # tokenizer EOS used by the already-audited full quality endpoint.
            first_prepared=prepared['cases'][0]
            reference_tokens={first_prepared['reference_ids'][first_prepared['user_positions'][j]] for j in first_prepared['keep']}
            assert len(reference_tokens)==1
            eos=next(iter(reference_tokens))
            endpoint=original.copy();endpoint[np.asarray(quality['user_positions'])[quality['keep']]]=eos
            original_shape=[1,len(original)]
            for repeat in range(4):
                offset=(ordinal+repeat)%3
                expected=cost_protocol['methods'][offset:]+cost_protocol['methods'][:offset]
                calls=[r for r in case['calls'] if r['repeat']==repeat]
                assert [r['method'] for r in calls]==expected
                for call in calls:
                    assert call['warmup']==(repeat==0) and call['seconds']>0 and call['peak_allocated_bytes']>0
                    assert len(call['actual_model_inputs'])==1
                    assert call['actual_model_inputs'][0]['shape'][0]==(2 if call['method']=='DT' else 1)
                    actual=v[call['vector_key']]
                    expected_shape=(len(original),) if call['method']=='DT' else (len(quality['user_positions']),)
                    assert actual.shape==expected_shape and np.isfinite(actual).all()
                    if call['method']!='DT':
                        assert call['actual_model_inputs']==[dict(shape=original_shape,sha256=ids_sha(original))]
                    else:
                        assert call['actual_model_inputs']==[dict(shape=[2,len(original)],sha256=ids_sha(np.stack((endpoint,original))))]
                    keys.add(call['vector_key'])
                    reference=cost_quality_vectors.get((case['dataset'],case['index'],call['method']))
                    reported=call['full_quality_vector_comparison']
                    if reference is None:
                        assert reported is None and case['dataset']=='morehopqa' and call['method']=='FT_K3'
                    else:
                        difference=actual.astype(np.float64)-reference.astype(np.float64)
                        independently=dict(bitwise_equal=bool(np.array_equal(actual,reference)),
                            relative_l2=float(np.linalg.norm(difference)/max(np.linalg.norm(reference),1e-30)),
                            max_absolute_error=float(np.max(np.abs(difference))))
                        assert independently==reported
                        comparisons.append(independently)
            for method in cost_protocol['methods']:
                measured=[r for r in case['calls'] if r['method']==method and not r['warmup']]
                warm=[r for r in case['calls'] if r['method']==method and r['warmup']]
                assert len(measured)==3 and len(warm)==1
                cost_rows.append(dict(dataset=case['dataset'],index=case['index'],method=method,
                    median_seconds=float(np.median([r['seconds'] for r in measured])),
                    peak_allocated_GB=max(r['peak_allocated_bytes'] for r in measured)/1e9,
                    warmup_seconds=warm[0]['seconds'],measured_new_graphs=sum(r['new_torch_graphs'] for r in measured)))
        assert set(v.files)==keys and len(keys)==192
    for method in cost_protocol['methods']:
        sample=[r for r in cost_rows if r['method']==method]
        cost_summary.append(dict(method=method,count=len(sample),
            sum_per_case_median_seconds=sum(r['median_seconds'] for r in sample),
            mean_per_case_median_seconds=float(np.mean([r['median_seconds'] for r in sample])),
            peak_allocated_GB=max(r['peak_allocated_GB'] for r in sample),
            sum_warmup_seconds=sum(r['warmup_seconds'] for r in sample),
            measured_new_graphs=sum(r['measured_new_graphs'] for r in sample)))
    by={(r['dataset'],r['method']):r for r in means}
    macro={}
    for group,tasks in [('NIAH',[t for t in main_protocol['tasks'] if t.startswith('niah_')]),
                        ('VT',[t for t in main_protocol['tasks'] if t.startswith('vt_')])]:
        macro[group]={m:float(np.mean([by[t,m]['recall'] for t in tasks])) for m in ('DT','FT_K1','FT_K3')}
    macro['faithfulness']={m:{metric:float(np.mean([by[t,m][metric] for t in main_protocol['tasks']]))
        for metric in ('rise','mas')} for m in ('DT','FT_K1')}
    wins={metric:sum(by[t,'DT'][metric]<by[t,'FT_K1'][metric] for t in main_protocol['tasks']) for metric in ('rise','mas')}
    rng=np.random.default_rng(73);paired=[]
    exported_pairs={(r['dataset'],r['comparison'],r['metric']):r for r in csv_read(a.export/'paired_differences.csv')}
    for task in main_protocol['tasks']:
        task_rows=[r for r in rows if r['dataset']==task]
        dt_rows={r['index']:r for r in task_rows if r['method']=='DT'}
        for other in ('FT_K1','ifr-tokenwise','FT_K3'):
            other_rows={r['index']:r for r in task_rows if r['method']==other}
            if not other_rows:continue
            assert dt_rows.keys()==other_rows.keys()
            for metric in ('recall','rise','mas'):
                if dt_rows[0][metric] is None or other_rows[0][metric] is None:continue
                delta=np.array([dt_rows[i][metric]-other_rows[i][metric] for i in sorted(dt_rows)])
                boot=delta[rng.integers(0,len(delta),size=(10000,len(delta)))].mean(axis=1)
                low,high=np.quantile(boot,[.025,.975]);comparison='DT - '+other
                assert abs(float(delta.mean())-float(exported_pairs[task,comparison,metric]['paired_mean_difference']))<1e-12
                paired.append(dict(dataset=task,count=len(delta),comparison=comparison,metric=metric,
                    paired_mean_difference=float(delta.mean()),bootstrap_95_low=float(low),bootstrap_95_high=float(high),
                    DT_better_direction='positive' if metric=='recall' else 'negative'))
    a.output.mkdir(parents=True,exist_ok=False)
    csv_write(a.output/'task_means.csv',means);csv_write(a.output/'per_case.csv',rows)
    csv_write(a.output/'cost_per_case.csv',cost_rows);csv_write(a.output/'cost_summary.csv',cost_summary)
    csv_write(a.output/'paired_differences.csv',paired)
    result=dict(status='complete',authority='DeltaTrace manuscript, not a substitution of the FlashTrace evaluation policy',
        quality_cases=1243,quality_tasks=13,recovery_cases=448,cost_examples=16,cost_calls=192,
        macro=macro,DT_lower_task_counts=wins,cost_summary=cost_summary,
        full_quality_vector_comparisons=dict(compared=len(comparisons),
            bitwise_equal=sum(x['bitwise_equal'] for x in comparisons),
            max_relative_l2=max(x['relative_l2'] for x in comparisons)),
        saved_curves_audit_sha256=sha(a.audit/'saved_curves_audit_full.json'),
        recovery_results_sha256=sha(a.recovery/'results.json'),recovery_vectors_sha256=sha(a.recovery/'vectors.npz'),
        cost_results_sha256=sha(a.cost/'results.json'),cost_vectors_sha256=sha(a.cost/'vectors.npz'))
    result['cost_gpu_ownership_checks']=len(ownership)
    result['measured_new_graphs']=sum(r['measured_new_graphs'] for r in cost_summary)
    result['quality_bootstrap']=dict(resamples=10000,seed=73,unit='paired example within each task',
        interval='percentile 95%, unadjusted across tasks',rows=len(paired),
        random_stream='Independent quality-only resampling; original mixed quality/cost export retained in raw artifacts')
    dump(a.output/'verification.json',result)
    names={'niah_mq_q2':'NIAH MQ-Q2','niah_mq_q4':'NIAH MQ-Q4','niah_mq_q8':'NIAH MQ-Q8',
        'niah_mv_v2':'NIAH MV-V2','niah_mv_v4':'NIAH MV-V4','niah_mv_v8':'NIAH MV-V8',
        'vt_h2_c3':'VT H2 (10%)','vt_h4_c1':'VT H4 (10%)','vt_h6_c1':'VT H6 (20%)','vt_h10_c1':'VT H10 (30%)',
        'hotpotqa_long':'HotpotQA (10%)','math':'MATH','morehopqa':'MoreHopQA'}
    fmt=lambda x,pct=False:'—' if x is None else f'{100*x:.2f}%' if pct else f'{x:.4f}'
    lines=['# Qwen3.5-9B：按 DeltaTrace 论文完成的 DT／FT 对比','',
        '全部 13 个任务、1,243 例质量评估已完成；另完成 VT／HotpotQA 的 448 例论文恢复率评估和 16 例开发集的 192 次完整归因计时调用。','',
        '| 任务 | n | DT Recall ↑ | FT K3 Recall ↑ | DT RISE ↓ | FT K1 RISE ↓ | DT MAS ↓ | FT K1 MAS ↓ |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for task,spec in main_protocol['tasks'].items():
        d,f=by[task,'DT'],by[task,'FT_K1'];k=by.get((task,'FT_K3'),{})
        lines.append('| '+' | '.join([names[task],str(spec['count']),fmt(d['recall'],True),fmt(k.get('recall'),True),
            fmt(d['rise']),fmt(f['rise']),fmt(d['mas']),fmt(f['mas'])])+' |')
    lines+=['','NIAH 以发布的 eligible token 计 10% 预算；VT 只解释重构答案，在正文 token 上使用表中固定预算；HotpotQA 保留全响应，以原生句子有符号分数和排序，在包含标点与空白的 10% 全正文 token 成本内取最长排名前缀，按官方支持事实评分。MATH 和 MoreHopQA 无此口径的证据 gold，因此不填 Recall。','',
        'RISE／MAS 都评价完整固定响应加 EOS、采用 20 步删除。DT 的 RISE 使用有符号排序，MAS 使用正值排序；两方法共享相同完整输入和完全删除端点。FT K1 用于忠实度，FT K3 用于恢复率。','',
        f"NIAH 六任务宏平均 Recall：DT {fmt(macro['NIAH']['DT'],True)}，FT K3 {fmt(macro['NIAH']['FT_K3'],True)}。",
        f"VT 四任务按表中预算的宏平均 Recall：DT {fmt(macro['VT']['DT'],True)}，FT K3 {fmt(macro['VT']['FT_K3'],True)}。该值不是统一的 Recall@10%。",
        f"DT 在 {wins['rise']}/13 个任务的 RISE、{wins['mas']}/13 个任务的 MAS 上低于 FT K1。以上比较为观测均值；逐例数据和配对 bootstrap 区间见 CSV。",'',
        '置信区间在每个任务内按配对样本重采样 10,000 次，使用固定种子 73 和百分位 95% 区间，未作跨任务多重比较校正。此处单独重算质量指标区间；原始导出的质量及单次运行计时区间保留在原始材料中。','',
        '| 方法 | 16 例的逐例中位耗时之和（秒） | 平均逐例中位耗时（秒） | 最大已分配显存（GB） |',
        '|---|---:|---:|---:|']
    for r in cost_summary:
        lines.append(f"| {r['method']} | {r['sum_per_case_median_seconds']:.3f} | {r['mean_per_case_median_seconds']:.4f} | {r['peak_allocated_GB']:.3f} |")
    lines+=['',f"计时期间通过 {len(ownership)} 次 GPU 进程占用检查；测量调用中新建编译图 {result['measured_new_graphs']} 个。"]
    if result['measured_new_graphs']:
        lines+=['测量调用仍发生编译，表中耗时包含这些编译开销，不能作为完全预热后的稳定耗时。']
    lines+=['','效率对比将 DT 论文的配对方法计时规范用于 Qwen3.5：首 8 例 MQ-Q2 与首 8 例 MoreHopQA，每种方法每例预热一次，轮换调用顺序并测量三次；计时覆盖文本准备、捕获、传播、层重放、投影和 CPU 分数返回。模型加载、首次 eager 初始化、预热和磁盘写入分开记录。GB 为 10^9 字节，显存包含驻留模型权重。这是 DT／FT 的单例配对实验；论文原先的 Qwen3.5 B1/B2 实现对比另有独立记录。','',
        '方法采用相同 Qwen3.5-9B、BF16、固定输入／响应和已核验的动态编译实现；DT 使用既定有限传播规则，FT 使用现有官方 Qwen3.5 支持。没有为本次结果调整任务预算、目标范围或层规则。计时调用的完整向量均保存，与全量质量运行的向量比较保存在 verification.json；不把不同浮点执行称为逐位相同。','',
        '响应沿用发布数据中的 Qwen3-235B-A22B-2507 固定缓存；本实验的 Qwen3.5-9B 负责归因和删除扰动评分，未重新生成回答。','',
        '附加 IFR 和 FT K1 恢复率保留在 task_means.csv；全量作业内含编译的单次计时不替代上述重复计时。']
    lines+=['',
        '计时硬件为单张 MetaX C550（64 GiB）；PyTorch '+cost['runtime']['torch']+'，Transformers '+cost['runtime']['transformers']+'，BF16。',
        f"计时输出中有 {len(comparisons)} 条向量具备全量质量运行参照，其中 {sum(x['bitwise_equal'] for x in comparisons)} 条逐位一致；最大相对 L2 差为 {max(x['relative_l2'] for x in comparisons):.8g}。MoreHopQA 的 FT K3 在主质量运行中没有对应向量，相关 32 次计时输出仅检查完整性与有限值。"]
    (a.output/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()

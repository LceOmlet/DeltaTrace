"""Apply the user's token-positive HotpotQA Recall correction to saved vectors."""
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
from datetime import datetime, timezone

import numpy as np

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
SOURCE = BASE / 'cross_machine_backup/mirror'
OUT = BASE / 'hotpot_positive_sum_v1'
sys.path.insert(0, str(ROOT / 'experiments/official'))
from hotpot_retrieval_v3 import rank_sentences, select_prefix
from hotpot_evidence import supporting_fact_metrics

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read(path):
    return json.loads(path.read_bytes())

def ids_sha(ids):
    return hashlib.sha256(np.asarray(ids, dtype=np.int64).tobytes()).hexdigest()

def write(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')

def score(row, values, fraction, pooling):
    order, pooled = rank_sentences(values, row['units'], row['all_groups'], row['eligible_groups'], pooling=pooling)
    budget = math.ceil(len(row['all_body_tokens']) * fraction)
    selected, tokens = select_prefix(order, row['all_groups'], token_budget=budget)
    facts = supporting_fact_metrics(selected, row['units'], row['official_keys'])
    return dict(recall=facts['recall'], budget=budget, selected_tokens=tokens, selected_units=selected,
                ranked_units=order, ranked_scores=[pooled[i] for i in order], fraction=fraction,
                budget_unit='all_body_tokens', metric='supporting_fact_recall')

def main():
    OUT.mkdir(exist_ok=True)
    # This distinguishes token clipping from clipping a sentence's signed total.
    units=[dict(kind='sentence',start=0),dict(kind='sentence',start=2)]
    groups=[[0,1],[2,3]]
    values=np.array([5.,-4.,2.,0.])
    assert rank_sentences(values,units,groups,groups,pooling='signed_sum')[0]==[1,0]
    assert rank_sentences(values,units,groups,groups,pooling='positive_sum')==([0,1],{0:5.,1:2.})
    source_hashes={}
    cases=[]
    replacement={}
    all_ft_unchanged=True
    for model in ('qwen3','qwen35'):
        pp=SOURCE/(model+'_prepared_recovery.json')
        sp=SOURCE/(model+'_recovery_v2/status.json')
        prepared=read(pp);status=read(sp)
        source_hashes[pp.relative_to(SOURCE).as_posix()]=sha(pp)
        source_hashes[sp.relative_to(SOURCE).as_posix()]=sha(sp)
        assert status['status']=='complete' and status['prepared_sha256']==sha(pp)
        source_meta={(r['dataset'],r['index']):r for r in status['cases']}
        rows=[r for r in prepared['cases'] if r['dataset']=='hotpotqa_long']
        assert len(rows)==48 and {r['index'] for r in rows}==set(range(48))
        for row in rows:
            meta=source_meta[row['dataset'],row['index']]
            folder=SOURCE/(model+'_recovery_v2')/meta['path']
            for name,key in [('results.json','results_sha256'),('vectors.npz','vectors_sha256')]:
                p=folder/name;h=sha(p);assert h==meta[key]
                source_hashes[p.relative_to(SOURCE).as_posix()]=h
            old=read(folder/'results.json')
            assert old['input_sha256']==row['input_sha256']==ids_sha(row['input_ids'])
            assert old['reference_sha256']==row['reference_sha256']==ids_sha(row['reference_ids'])
            assert old['actual_inputs']['DT']==[ids_sha([row['reference_ids'],row['input_ids']])]
            for method in ('FT_K3','FT_K1'):
                assert old['actual_inputs'][method]==[row['input_sha256']]
            assert old['target_weights']==old['FT_target_weights']==row['target_weights']
            assert old['target_offsets']==row['target_offsets']
            result=dict(model=model,dataset=row['dataset'],index=row['index'],input_sha256=row['input_sha256'],
                        source_results_sha256=meta['results_sha256'],source_vectors_sha256=meta['vectors_sha256'],
                        target_weights=row['target_weights'],metrics={},budgets={},old_signed_recall={})
            with np.load(folder/'vectors.npz',allow_pickle=False) as vectors:
                assert np.array_equal(vectors['DT_prompt'],vectors['DT_signed_full'][row['user_positions']])
                for method in ('DT','FT_K3','FT_K1'):
                    values=vectors[method+'_prompt']
                    assert values.shape==(len(row['user_positions']),) and np.isfinite(values).all()
                    assert old['metrics'][method]==score(row,values,.1,'signed_sum')
                    result['old_signed_recall'][method]=old['metrics'][method]['recall']
                    result['metrics'][method]=score(row,values,.1,'positive_sum')
                    result['budgets'][method]=[]
                    for prior in old['budgets'][method]:
                        assert prior==score(row,values,prior['fraction'],'signed_sum')
                        new=score(row,values,prior['fraction'],'positive_sum')
                        # Independent equivalence check: clip tokens, then use the old sum.
                        assert new==score(row,np.maximum(values,0),prior['fraction'],'signed_sum')
                        result['budgets'][method].append(new)
                        if method.startswith('FT_'):
                            assert np.all(values>=0) and new==prior
            for method in ('FT_K3','FT_K1'):
                pairs=[('paper_recovery',result['metrics']['DT'],result['metrics'][method])]
                pairs += [('paper_recovery_budget_curve',a,b) for a,b in zip(result['budgets']['DT'],result['budgets'][method])]
                for scope,dt,ft in pairs:
                    replacement[model,row['index'],scope,str(dt['fraction']),method]=(dt['recall'],ft['recall'])
            cases.append(result)
    original_rows=list(csv.DictReader((BASE/'final_results/per_case.csv').open(encoding='utf-8')))
    updated_rows=[]
    changed=0
    for old in original_rows:
        r=old.copy()
        if r['dataset']=='hotpotqa_long' and r['metric']=='recall':
            dt,ft=replacement[r['model'],int(r['index']),r['scope'],r['budget_fraction'],r['FT_method']]
            r.update(DT=dt,FT=ft,difference_DT_minus_FT=dt-ft,
                     scope=r['scope'].replace('paper_recovery','paper_recovery_positive_sum'))
            changed+=1
        updated_rows.append(r)
    assert changed==96*2*7
    original_tables=list(csv.DictReader((BASE/'final_results/task_metrics.csv').open(encoding='utf-8')))
    tables=[];rng=np.random.default_rng(20260914)
    for old in original_tables:
        r=old.copy()
        if r['dataset']=='hotpotqa_long' and r['metric']=='recall':
            scope=r['scope'].replace('paper_recovery','paper_recovery_positive_sum')
            group=[x for x in updated_rows if x['model']==r['model'] and x['dataset']==r['dataset'] and
                   x['scope']==scope and x['budget_fraction']==r['budget_fraction'] and x['FT_method']==r['FT_method']]
            assert len(group)==48
            dt=np.array([float(x['DT']) for x in group]);ft=np.array([float(x['FT']) for x in group]);delta=dt-ft
            ci=np.quantile(delta[rng.integers(0,48,(5000,48))].mean(1),[.025,.975])
            r.update(DT=float(dt.mean()),FT=float(ft.mean()),difference_DT_minus_FT=float(delta.mean()),
                     difference_ci95_low=float(ci[0]),difference_ci95_high=float(ci[1]),scope=scope)
        tables.append(r)
    for name,rows in [('per_case.csv',updated_rows),('task_metrics.csv',tables)]:
        with (OUT/name).open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    protocol=dict(version='hotpot-token-positive-sum-v1',requested_change='User explicitly requested positive contributions for HotpotQA retrieval.',
                  models=['qwen3','qwen35'],cases_each=48,methods=['DT','FT_K3','FT_K1'],
                  pooling='sum(max(token_attribution, 0)) over every native sentence token; preserve source dtype before float64 accumulation',
                  budget='ceil(fraction * all native sentence body token count)',selection='unchanged ranked prefix, same tie-break and full sentence token charge',
                  primary_fraction=.1,fractions=[.05,.1,.2,.3,.4,.5],target='unchanged fixed full response and matched non-stop target weights',
                  supersedes='Only HotpotQA Recall signed-sum entries in final_results; other task metrics remain unchanged.',
                  metric='supporting_fact_recall',source_archive_sha256=read(SOURCE/'finalization_status.json')['archive_sha256'],
                  bootstrap='Within-task descriptive paired bootstrap, 5000 draws, seed 20260914',model_calls=0)
    write('protocol.json',protocol);write('rescored_cases.json',cases)
    shutil.copy2(__file__,OUT/'rescore_hotpot_positive.py')
    for name in ('hotpot_retrieval_v3.py','hotpot_evidence.py'):
        shutil.copy2(ROOT/'experiments/official'/name,OUT/name)
    names={'qwen3':'Qwen3-8B','qwen35':'Qwen3.5-9B'}
    lines=['# HotpotQA 正贡献 Recall 修正', '',
           '按用户指令，两个模型的全部 48 例均改为先将每个 token 的负贡献置零，再按原生句子求和、排序。DT 与 FT K1/K3 使用相同规则，固定目标、正文预算、句子成本和选取规则沿用原协议。使用已保存向量离线重算，没有新的模型调用。原 signed-sum 结果保留为历史，HotpotQA Recall 以后以此版本为准。', '',
           '| 模型 | 原 signed DT % | 正贡献 DT % | FT K3 % | FT K1 % | DT−FT K3 百分点 |',
           '|---|---:|---:|---:|---:|---:|']
    primary={}
    for model in names:
        rs=[c for c in cases if c['model']==model]
        means={m:float(np.mean([c['metrics'][m]['recall'] for c in rs])) for m in ('DT','FT_K3','FT_K1')}
        old=float(np.mean([c['old_signed_recall']['DT'] for c in rs]));primary[model]=dict(old_DT=old,**means)
        lines.append(f"| {names[model]} | {old*100:.2f} | {means['DT']*100:.2f} | {means['FT_K3']*100:.2f} | {means['FT_K1']*100:.2f} | {(means['DT']-means['FT_K3'])*100:+.2f} |")
    lines += ['', '## 全预算 Recall', '', '| 预算 | Qwen3 DT / FT K3 % | Qwen3.5 DT / FT K3 % |', '|---|---:|---:|']
    for fraction in protocol['fractions']:
        cells=[]
        for model in names:
            rs=[c for c in cases if c['model']==model]
            means=[np.mean([next(b['recall'] for b in c['budgets'][m] if b['fraction']==fraction) for c in rs]) for m in ('DT','FT_K3')]
            cells.append(f'{means[0]*100:.2f} / {means[1]*100:.2f}')
        lines.append(f"| {fraction*100:.0f}% | "+' | '.join(cells)+' |')
    plan=read(BASE/'protocol.json')
    lines += ['', '## 更新后的全指标表', '', '每格为 DT / FT。RISE、MAS、RISE+AP 使用 FT K1，越低越好；Recall 为百分比，使用 FT K3，越高越好。HotpotQA Recall 为本次正贡献修正，其余数值直接保留既有结果。MATH/MoreHopQA 无此协议所需 gold，Recall 留空。', '']
    for model in names:
        lines += [f'### {names[model]}', '', '| 任务 | n | RISE ↓ | MAS ↓ | RISE+AP ↓ | Recall % ↑ |', '|---|---:|---:|---:|---:|---:|']
        for task,n in plan[model]['tasks'].items():
            cells=[]
            for metric in ('rise','mas','rise_plus_ap','recall'):
                rs=[r for r in tables if r['model']==model and r['dataset']==task and r['metric']==metric and
                    (r['scope']=='released_full_response' or r['scope'] in ('released_NIAH','paper_recovery','paper_recovery_positive_sum') and r['FT_method']=='FT_K3')]
                assert len(rs)<=1
                if not rs:cells.append('—');continue
                r=rs[0];scale,digits=(100,2) if metric=='recall' else (1,4)
                cells.append(' / '.join(f'{float(r[k])*scale:.{digits}f}' for k in ('DT','FT')))
            lines.append(f'| {task} | {n} | '+' | '.join(cells)+' |')
        lines.append('')
    lines += ['`task_metrics.csv` 和 `per_case.csv` 为更新后的完整汇总。`rescored_cases.json` 保留每例每预算的选句、分数与原始向量哈希；源码和协议一并保存。原向量、耗时、旧口径结果在此前完整归档及异机备份中保留。此修正来自看到旧结果后的用户指令，不表述为结果揭晓前已冻结的协议。']
    (OUT/'RESULTS_zh.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    # No original result, vector, scorer snapshot or archive is modified.
    for rel,digest in source_hashes.items():assert sha(SOURCE/rel)==digest
    receipt=dict(status='complete',created_at_utc=datetime.now(timezone.utc).isoformat(),cases=96,cases_each=48,
                 model_calls=0,FT_vectors_nonnegative_and_all_results_unchanged=all_ft_unchanged,
                 source_files_unchanged=True,source_files_sha256=source_hashes,token_clipping_check='passed',
                 source_dtype_preserved=True,primary=primary,updated_per_case_rows=changed,
                 original_full_table_sha256=sha(BASE/'final_results/task_metrics.csv'),
                 original_per_case_sha256=sha(BASE/'final_results/per_case.csv'),
                 files={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in OUT.iterdir() if p.is_file() and p.name!='receipt.json'})
    write('receipt.json',receipt)
    (BASE/'latest_results.json').write_text(json.dumps(dict(version=protocol['version'],report='hotpot_positive_sum_v1/RESULTS_zh.md',
        tables='hotpot_positive_sum_v1/task_metrics.csv',per_case='hotpot_positive_sum_v1/per_case.csv',receipt='hotpot_positive_sum_v1/receipt.json',
        note='HotpotQA Recall supersedes signed-sum values. Original artifacts retained.'),indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='complete',cases=96,model_calls=0,primary=primary),ensure_ascii=False),flush=True)

if __name__=='__main__':
    main()

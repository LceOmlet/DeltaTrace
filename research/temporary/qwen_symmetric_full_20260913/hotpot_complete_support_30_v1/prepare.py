"""Prepare local CSV revision from saved Hotpot vectors and native sentence selections."""
from pathlib import Path
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
import numpy as np

OUT=Path(__file__).resolve().parent
BASE=OUT.parent
ROOT=BASE.parents[2]
MIRROR=BASE/'cross_machine_backup/mirror'
BL=ROOT/'research/temporary/all_baselines_20260910'
OLD35=ROOT/'research/temporary/qwen35_paper_full_20260913/raw_completion/paper_recovery_dynamic'
sys.path.insert(0,str(ROOT/'experiments/official'))
from hotpot_retrieval_v3 import rank_sentences,select_prefix
from hotpot_evidence import supporting_fact_metrics

hashes={}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def tracked(p):
    h=sha(p); hashes[p.relative_to(ROOT).as_posix()]=h
    return h
def read(p):tracked(p);return json.loads(p.read_bytes())
def read_csv(p):
    tracked(p)
    with p.open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def main():
    prepared={}
    for model in ('qwen3','qwen35'):
        prepared[model]={r['index']:r for r in read(MIRROR/(model+'_prepared_recovery.json'))['cases'] if r['dataset']=='hotpotqa_long'}
        assert set(prepared[model])==set(range(48))
    saved=read(BASE/'hotpot_positive_sum_v1/rescored_cases.json')
    positive_receipt=read(BASE/'hotpot_positive_sum_v1/receipt.json')
    assert sha(BASE/'hotpot_positive_sum_v1/rescored_cases.json')==positive_receipt['files']['rescored_cases.json']['sha256']
    selections=[]; records=[]; current={}

    def add(model,method,profile,row,selected,tokens,order,source_group,source_hash):
        budget=math.ceil(len(row['all_body_tokens'])*.3)
        assert tokens==[t for u in selected for t in row['all_groups'][u]]
        assert len(tokens)==len(set(tokens)) and len(tokens)<=budget
        facts=supporting_fact_metrics(selected,row['units'],row['official_keys'])
        keys={(row['units'][u]['title'],row['units'][u]['sentence_index']) for u in selected}
        independent=int(set(map(tuple,row['official_keys']))<=keys)
        assert facts['complete_support']==independent
        r=dict(model=model,dataset='hotpotqa_long',index=row['index'],method=method,profile=profile,
            complete_support=independent,budget_fraction=.3,budget_unit='all_body_tokens',budget_tokens=budget,
            spent_tokens=len(tokens),selected_sentences=len(selected),supporting_facts_hit=facts['true_positive'],
            gold_supporting_facts=facts['gold'],pooling='positive_sum',target_mode='full',
            input_sha256=row['input_sha256'],source_group=source_group,source_sha256=source_hash)
        records.append(r)
        selections.append(dict(model=model,index=row['index'],method=method,selected_units=selected,
            selected_tokens=tokens,ranked_units=order,selected_facts=[list(k) for k in sorted(keys)],
            official_keys=row['official_keys'],complete_support=independent))

    def score(model,method,profile,row,values,source_group,source_hash):
        values=np.asarray(values)
        assert values.shape==(len(row['user_positions']),) and np.isfinite(values).all()
        order,_=rank_sentences(values,row['units'],row['all_groups'],row['eligible_groups'],pooling='positive_sum')
        selected,tokens=select_prefix(order,row['all_groups'],token_budget=math.ceil(len(row['all_body_tokens'])*.3))
        add(model,method,profile,row,selected,tokens,order,source_group,source_hash)

    # Current retest methods: preserve saved positive rankings exactly.
    for case in saved:
        model=case['model'];row=prepared[model][case['index']]
        assert row['input_sha256']==case['input_sha256'] and row['target_weights']==case['target_weights']
        for method in ('DT','FT_K3','FT_K1'):
            profile=('pv-layer-symmetric-v1' if model=='qwen3' else 'gdn-symmetric-v1') if method=='DT' else method
            for b in case['budgets'][method]:
                f=supporting_fact_metrics(b['selected_units'],row['units'],row['official_keys'])
                assert f['recall']==b['recall']
                current[model,row['index'],method,b['fraction']]=f['complete_support']
            b=next(b for b in case['budgets'][method] if b['fraction']==.3)
            add(model,method,profile,row,b['selected_units'],b['selected_tokens'],b['ranked_units'],
                'symmetric_full_retest',case['source_vectors_sha256'])

    # Qwen3: original PV allocation, from the already identified historical full-target run.
    sys.path.insert(0,str(ROOT/'research/temporary/hotpot_fairness_20260910'))
    from common import full_cases
    original,original_receipts=full_cases()
    for i,(old,vectors) in original.items():
        row=prepared['qwen3'][i]
        for key in ('input_ids','input_sha256','target','target_weights','user_positions'):
            assert old[key]==row[key],('qwen3_original',i,key)
        score('qwen3','DT_original','content_P1',row,vectors['DT_target'][row['user_positions']],
              'source_v2_full_recall',hashlib.sha256(np.asarray(vectors['DT_target']).tobytes()).hexdigest())

    # The five tested Qwen3 baseline methods use their primary signed target aggregate;
    # apply token-positive sentence pooling uniformly, retaining native-normalized views as history.
    bl_plan=read(BL/'protocol.json');bl_analysis=read(BL/'analysis.json')
    items={r['index']:r for r in read(BL/'inputs.json')['cases'] if r['dataset']=='hotpotqa_long'}
    candidates={r['index']:r for r in read(BL/'candidates.json')['cases'] if r['dataset']=='hotpotqa_long'}
    labels={r['index']:r for r in read(BL/'labels.json')['cases'] if r['dataset']=='hotpotqa_long'}
    for name in ('inputs','candidates','labels'):
        assert sha(BL/(name+'.json'))==bl_plan[name+'_sha256']
    receipts={(r['method'],r['index']):r for r in bl_analysis['new_receipts'] if r['dataset']=='hotpotqa_long'}
    for method in ('Perturbation','REAGENT','CLP','IFR','AttnLRP'):
        for i in range(48):
            row=prepared['qwen3'][i];item=items[i]
            for key in ('input_ids','input_sha256','target','target_weights','user_positions'):
                assert item[key]==row[key],(method,i,key)
            assert candidates[i]['units']==row['units'] and candidates[i]['all_groups']==row['all_groups']
            assert labels[i]['official_keys']==row['official_keys']
            folder=BL/'raw'/method/'hotpotqa_long'/f'{i:03d}'
            receipt=read(folder/'results.json');vpath=folder/'vectors.npz';h=tracked(vpath)
            assert sha(folder/'results.json')==receipts[method,i]['results_sha256']
            assert h==receipt['vectors_sha256']==receipts[method,i]['vectors_sha256']
            assert receipt['status']=='complete' and receipt['input_sha256']==row['input_sha256']
            assert receipt['target_weights']==row['target_weights']
            with np.load(vpath,allow_pickle=False) as v:
                assert np.array_equal(v['applied_target_weights'],np.asarray(row['target_weights']))
                score('qwen3',method,'primary_target_sum',row,v['prompt_signed'],
                    'all_baselines_20260910',h)

    # Qwen3.5: tested tokenwise IFR and the separately named earlier DT implementation.
    verification=read(OLD35/'verification.json');old35=read(OLD35/'results.json')
    old35_hash=tracked(OLD35/'vectors.npz')
    assert old35_hash==old35['vectors_sha256']==verification['vectors_sha256']
    assert sha(OLD35/'results.json')==verification['results_sha256'] and old35['status']=='complete'
    with np.load(OLD35/'vectors.npz',allow_pickle=False) as v:
        for old in [r for r in old35['cases'] if r['dataset']=='hotpotqa_long']:
            row=prepared['qwen35'][old['index']]
            assert old['input_sha256']==row['input_sha256'] and old['reference_sha256']==row['reference_sha256']
            assert old['FT_target_weights']==row['target_weights'] and old['target_offsets']==row['target_offsets']
            for method,key,profile in [('IFR_tokenwise','ifr-tokenwise','ifr-tokenwise'),
                                       ('DT_legacy','DT','paper_recovery_dynamic_20260913')]:
                score('qwen35',method,profile,row,v[f"hotpotqa_long_{row['index']}_{key}_prompt"],
                    'qwen35_paper_recovery_dynamic',old35_hash)

    groups=defaultdict(list)
    for r in records:groups[r['model'],r['method']].append(r)
    method_order={'DT':0,'DT_original':1,'DT_legacy':1,'FT_K3':2,'FT_K1':3,'Perturbation':4,
                  'REAGENT':5,'CLP':6,'IFR':7,'IFR_tokenwise':7,'AttnLRP':8}
    summary=[]
    for (model,method),rs in sorted(groups.items(),key=lambda kv:(kv[0][0],method_order[kv[0][1]])):
        assert len(rs)==48 and {r['index'] for r in rs}==set(range(48))
        count=sum(r['complete_support'] for r in rs)
        summary.append(dict(model=model,dataset='hotpotqa_long',method=method,profile=rs[0]['profile'],
            cases=48,complete_cases=count,complete_support=count/48,budget_fraction=.3,budget_unit='all_body_tokens',
            pooling='positive_sum',target_mode='full',source_group=rs[0]['source_group']))
    assert len(groups)==14 and len(records)==672
    assert next(r['complete_cases'] for r in summary if r['model']=='qwen3' and r['method']=='DT')==41
    assert next(r['complete_cases'] for r in summary if r['model']=='qwen35' and r['method']=='DT')==46
    records.sort(key=lambda r:(r['model'],method_order[r['method']],r['index']))

    # Replace the current Hotpot recovery metric in the full local pairwise CSVs;
    # retain each existing budget-curve point and switch only the primary point to 30%.
    old_per_case=read_csv(BASE/'hotpot_positive_sum_v1/per_case.csv')
    old_table=read_csv(BASE/'hotpot_positive_sum_v1/task_metrics.csv')
    def revise(r):
        return r['dataset']=='hotpotqa_long' and r['metric']=='recall'
    changed=[]
    for old in old_per_case:
        r=old.copy()
        if revise(old):
            fraction=float(old['budget_fraction']) if old['scope'].endswith('_budget_curve') else .3
            dt=current[r['model'],int(r['index']),'DT',fraction]
            ft=current[r['model'],int(r['index']),r['FT_method'],fraction]
            r.update(metric='complete_support',DT=dt,FT=ft,difference_DT_minus_FT=dt-ft,
                scope=old['scope'].replace('paper_recovery_positive_sum','paper_complete_support_positive_sum'),
                budget_fraction=fraction)
        changed.append(r)
    tables=[];rng=np.random.default_rng(20260914);draws=rng.integers(0,48,(10000,48))
    for old in old_table:
        r=old.copy()
        if revise(old):
            fraction=float(old['budget_fraction']) if old['scope'].endswith('_budget_curve') else .3
            dt=np.array([current[r['model'],i,'DT',fraction] for i in range(48)])
            ft=np.array([current[r['model'],i,r['FT_method'],fraction] for i in range(48)])
            delta=dt-ft;ci=np.quantile(delta[draws].mean(1),[.025,.975])
            r.update(metric='complete_support',DT=float(dt.mean()),FT=float(ft.mean()),difference_DT_minus_FT=float(delta.mean()),
                difference_ci95_low=float(ci[0]),difference_ci95_high=float(ci[1]),budget_fraction=fraction,
                scope=old['scope'].replace('paper_recovery_positive_sum','paper_complete_support_positive_sum'))
        tables.append(r)
    assert sum(revise(r) for r in old_per_case)==1344 and sum(revise(r) for r in old_table)==28
    assert all(a==b for a,b in zip(old_per_case,changed) if not revise(a))
    assert all(a==b for a,b in zip(old_table,tables) if not revise(a))
    datasets={name:dict(columns=list(rs[0]),rows=[[r[k] for k in rs[0]] for r in rs]) for name,rs in
        [('hotpotqa_all_methods.csv',summary),('hotpotqa_per_case.csv',records),('task_metrics.csv',tables),('per_case.csv',changed)]}
    dump(OUT/'prepared_tables.json',datasets)
    dump(OUT/'selections.json',selections)
    previous=(BASE/'latest_results.json').read_bytes()
    (OUT/'previous_latest_results.json').write_bytes(previous)
    for p,h in hashes.items():assert sha(ROOT/p)==h,p
    protocol=dict(version='hotpot-complete-support-30pct-v1',authority='User requested local CSV records only.',
        metric='complete_support',metric_display_name='Full-support Recall',primary_display_name='Full-support Recall@30%',
        definition='1 iff every official supporting-fact (title,sentence_index) is selected; extra sentences allowed.',
        primary_budget_fraction=.3,budget_unit='all_body_tokens',pooling='sum(max(token_attribution,0)) within each native sentence',
        selection='same descending-score prefix; stop at first nonfitting sentence; charge all native sentence tokens',
        target='unchanged fixed full response with matched non-stop target weights',
        selection_history='User selected complete support and 30% after reviewing prior recall/completeness curves; retrospective.',
        cases_each=48,methods={m:[r['method'] for r in summary if r['model']==m] for m in ('qwen3','qwen35')},
        source_files_sha256=hashes,original_DT_sources=original_receipts,
        scope='Local CSV revision only. Paper files and prior results remain unchanged. Native-positive-row-normalized diagnostics are aggregation views, not additional algorithms.',
        model_calls=0,original_files_unchanged=True,changed_full_per_case_rows=1344,changed_full_summary_rows=28,
        bootstrap='10000 paired descriptive draws; seed 20260914; no multiplicity correction',
        previous_latest_sha256=hashlib.sha256(previous).hexdigest())
    dump(OUT/'protocol.json',protocol)
    print(json.dumps(dict(status='prepared',method_rows=summary,csv_rows={k:len(v['rows']) for k,v in datasets.items()},model_calls=0),ensure_ascii=False))

if __name__=='__main__': main()

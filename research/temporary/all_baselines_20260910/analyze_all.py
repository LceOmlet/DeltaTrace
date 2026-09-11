"""Verify new native vectors and recompute all seven fixed-policy comparisons."""
import argparse
from collections import Counter,defaultdict
import csv
import gzip
import json
from pathlib import Path
import time
import numpy as np
from common import HERE,ROOT,OLD,HOTPOT,TASKS,BASELINES,METHODS,sha,byte_sha,previous_cases
from score_cases import score_case,HP_METRICS
from execution_identity import verify_identity

def write_json(path,value,compact=False):
    path.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,
        indent=None if compact else 2,separators=(',',':') if compact else None)+'\n',encoding='utf-8')

def write_csv(path,rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)

def read_csv(path):
    with path.open(newline='',encoding='utf-8') as stream:return list(csv.DictReader(stream))

def assert_close(left,right):
    assert np.isclose(float(left),float(right),atol=2e-12,rtol=0),(left,right)

def verify_new(folder,item,method,preflight_hash):
    receipt=json.loads((folder/'results.json').read_bytes());vpath=folder/'vectors.npz'
    assert receipt['status']=='complete' and receipt['method']==method
    assert receipt['dataset']==item['dataset'] and receipt['index']==item['index']
    assert receipt['input_sha256']==item['input_sha256'] and receipt['target_weights']==item['target_weights']
    assert receipt['target_mode']==item['target_mode'] and receipt['complete_input_verified'] and receipt['model_restored']
    assert receipt['generation_calls']==0 and receipt['calls']['model_forwards']>0
    receipt['execution_version']=verify_identity(receipt['identity'],method,preflight_hash,sha(folder/'results.json'))
    assert receipt['vectors_sha256']==sha(vpath)
    count=len(item['user_positions']);weights=np.asarray(item['target_weights'],dtype=np.float64)
    with np.load(vpath) as vectors:
        assert np.array_equal(vectors['applied_target_weights'],weights)
        saved=vectors['prompt_signed'];native=vectors['prompt_native_positive_row_normalized']
        assert saved.shape==native.shape==(count,) and np.isfinite(saved).all() and np.isfinite(native).all()
        if method=='AttnLRP':
            raw=vectors['raw_aggregate'];assert raw.shape==(count+item['target_length'],) and np.isfinite(raw).all()
            assert np.array_equal(raw[:count],saved)
            positive=np.maximum(raw,0);expected_native=positive/(positive.sum(dtype=np.float32)+np.float32(1e-12))
            assert np.array_equal(expected_native[:count],native)
            assert int(vectors['lrp_guard_calls'])==36 and int(vectors['lrp_repaired_zero_ratios'])>=0
            receipt['lrp_repaired_zero_ratios']=int(vectors['lrp_repaired_zero_ratios'])
        else:
            raw=vectors['raw_matrix'];assert raw.shape==(item['target_length'],count+item['target_length'])
            assert np.isfinite(raw[weights>0,:count]).all() and not np.isinf(raw).any()
            finite=np.where(np.isnan(raw),0,raw)
            # Independent reduction: select rows instead of multiplying by weights.
            expected_signed=finite[weights>0].sum(axis=0,dtype=np.float64).astype(np.float32)
            assert np.array_equal(expected_signed[:count],saved)
            positive=np.maximum(finite,0);norm=positive/(positive.sum(axis=1,keepdims=True,dtype=np.float32)+np.float32(1e-8))
            expected_native=norm[weights>0].sum(axis=0,dtype=np.float64).astype(np.float32)
            assert np.array_equal(expected_native[:count],native)
        if method in ('Perturbation','CLP','REAGENT'):
            calls=receipt['causal_prefix_audit'];assert len(calls)==receipt['calls']['model_forwards']
            expected_groups=json.loads((HERE/'inputs_preflight.json').read_bytes())['cases']
            groups=next(r['native_sink_groups'] for r in expected_groups if (r['dataset'],r['index'])==(item['dataset'],item['index']))
            expected_pairs={(item['prompt_length']+g[0],len(g)) for g in groups if g}
            assert {(r['prefix_tokens'],r['response_tokens']) for r in calls}==expected_pairs
            for call in calls:
                n=call['prefix_tokens'];m=call['response_tokens']
                assert item['prompt_length']<=n<n+m<=len(item['input_ids'])
                allowed=set(item['user_positions'])|set(range(item['prompt_length'],n))
                assert set(call['changed_positions'])<=allowed
        result=(saved.copy(),native.copy())
    receipt['results_sha256']=sha(folder/'results.json')
    return result,receipt

def verify_reuse(rows,selections):
    hp={(int(r['index']),r['method'],r['budget_unit'],float(r['budget_value'])):r
        for r in read_csv(HOTPOT/'native_per_case.csv') if r['target_mode']=='full' and r['pooling']=='signed_sum'}
    vt={(r['dataset'],int(r['index']),r['method'],r['view'],float(r['fraction'])):r
        for r in read_csv(OLD/'full_recall/per_case.csv') if r['dataset'].startswith('vt_')}
    old_select={(r['index'],r['method'],r['budget_unit'],r['budget_value']):r
        for r in json.loads((HOTPOT/'selections.json').read_bytes()) if r['target_mode']=='full' and r['pooling']=='signed_sum'}
    checked=0
    for row,selected in zip(rows,selections):
        assert row['method'] in ('DT','FT_K3','FT_K1')
        method='DT_target' if row['method']=='DT' else row['method']
        if row['dataset']=='hotpotqa_long':
            key=(row['index'],method,row['budget_unit'],row['fraction']);old=hp[key]
            for metric in HP_METRICS:
                prior={'spent_tokens':'spent_all_tokens','unused_tokens':'unused_all_tokens'}.get(metric,metric)
                assert_close(row[metric],old[prior])
            assert selected['selected_units']==old_select[key]['selected_units'] and selected['selected_tokens']==old_select[key]['selected_tokens']
            assert selected['ranked_units']==old_select[key]['ranked_units'] and selected['ranked_scores']==old_select[key]['ranked_scores']
        else:
            old=vt[row['dataset'],row['index'],method,row['view'],row['fraction']]
            for metric in ('recall','budget','gold','ceiling'):assert_close(row[metric],old[metric])
        checked+=1
    assert checked==448*3*6
    return checked

def statistics(rows,plan):
    groups=defaultdict(list)
    for row in rows:
        key=(row['dataset'],row['method'],row['aggregation'],row['view'],row['budget_unit'],row['fraction'])
        groups[key].append(row)
    summaries=[]
    for key,data in sorted(groups.items()):
        item=dict(zip(('dataset','method','aggregation','view','budget_unit','fraction'),key))
        n=plan['tasks'][item['dataset']]['count'];assert len(data)==n and {r['index'] for r in data}==set(range(n))
        metrics=HP_METRICS if item['dataset']=='hotpotqa_long' else ('recall','precision','ceiling','budget','recall_tie_low','recall_tie_high')
        item.update(n=n,**{m:float(np.mean([r[m] for r in data])) for m in metrics});summaries.append(item)
    primary_rows=[r for r in rows if r['aggregation']=='signed_sum' and r['fraction']==.1 and
        ((r['dataset']=='hotpotqa_long' and r['budget_unit']=='all_body_tokens') or r['view']=='raw')]
    data={(r['dataset'],r['method'],r['index']):r['recall'] for r in primary_rows}
    primary=[];comparison=[];rng=np.random.default_rng(plan['bootstrap']['seed'])
    draws={t:rng.integers(0,plan['tasks'][t]['count'],size=(plan['bootstrap']['draws'],plan['tasks'][t]['count'])) for t in TASKS}
    level=plan['bootstrap']['confidence'];alpha=(1-level)/2
    for method in METHODS+('FT_K1',):
        row=dict(method=method)
        for task in TASKS:row[task]=float(np.mean([data[task,method,i] for i in range(plan['tasks'][task]['count'])]))
        row['vt_macro']=float(np.mean([row[t] for t in TASKS if t.startswith('vt_')]));primary.append(row)
    for method in METHODS[1:]:
        deltas={t:np.array([data[t,'DT',i]-data[t,method,i] for i in range(plan['tasks'][t]['count'])]) for t in TASKS}
        for taskset,name in [(['hotpotqa_long'],'hotpotqa_long'),([t for t in TASKS if t.startswith('vt_')],'vt_macro')]:
            effect=float(np.mean([deltas[t].mean() for t in taskset]))
            sampled=np.mean([deltas[t][draws[t]].mean(axis=1) for t in taskset],axis=0)
            comparison.append(dict(scope=name,comparator=method,dt_minus_comparator=effect,
                interval=np.quantile(sampled,[alpha,1-alpha]).tolist(),confidence=level,
                family_size=12,draws=plan['bootstrap']['draws'],descriptive=True))
    assert len(comparison)==12
    return summaries,primary,comparison

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--verify-available',action='store_true');p.add_argument('--reuse-only',action='store_true');a=p.parse_args()
    plan=json.loads((HERE/'protocol.json').read_bytes());assert plan['spec_sha256']==sha(HERE/'PROTOCOL.md')
    for name in ('inputs','candidates','labels'):assert sha(HERE/(name+'.json'))==plan[name+'_sha256']
    assert sha(OLD/'full_recall/analysis.json')==plan['previous_full_analysis_sha256']
    assert sha(HOTPOT/'analysis.json')==plan['hotpot_analysis_sha256']
    inputs={(r['dataset'],r['index']):r for r in json.loads((HERE/'inputs.json').read_bytes())['cases']}
    candidates={(r['dataset'],r['index']):r for r in json.loads((HERE/'candidates.json').read_bytes())['cases']}
    labels={(r['dataset'],r['index']):r for r in json.loads((HERE/'labels.json').read_bytes())['cases']}
    old,origins=previous_cases();rows=[];selections=[];costs=[];new_receipts=[];coverage=Counter()
    for key,item in inputs.items():
        prior,values=old[key]
        for method in ('DT','FT_K3','FT_K1'):
            scores=values[method][item['user_positions']] if method=='DT' else values[method]
            result,pred=score_case(item,candidates[key],labels[key],method,scores)
            rows.extend(result);selections.extend(pred)
    reused=verify_reuse(rows,selections)
    if a.reuse_only:
        print(json.dumps(dict(status='reused_DT_FT_exactly_reproduced',score_rows=reused)));return
    preflight_hash=sha(HERE/'inputs_preflight.json')
    mlm=json.loads((HERE/'mlm_verification.json').read_bytes())
    assert mlm['status']=='all_five_uploaded_assets_verified' and mlm['protocol_sha256']==sha(HERE/'protocol.json')
    assert mlm['revision']==plan['mlm']['revision'] and len(mlm['files'])==5
    assert [{k:r[k] for k in ('name','bytes','sha256')} for r in mlm['files']]==[{k:r[k] for k in ('name','bytes','sha256')} for r in plan['mlm']['files']]
    for key,item in inputs.items():
        for method in BASELINES:
            folder=a.run/method/item['dataset']/f'{item["index"]:03d}'
            if not (folder/'results.json').exists():
                assert a.verify_available,('Missing method-case result',method,key)
                continue
            (scores,native),receipt=verify_new(folder,item,method,preflight_hash)
            new_receipts.append({k:receipt[k] for k in ('dataset','index','method','input_sha256','results_sha256','vectors_sha256')})
            costs.append({k:receipt[k] for k in ('dataset','index','method','seconds','peak_allocated_bytes','calls')})
            if method=='AttnLRP':costs[-1]['lrp_repaired_zero_ratios']=receipt['lrp_repaired_zero_ratios']
            coverage[method]+=1
            if not a.verify_available:
                for aggregation,vector in [('signed_sum',scores),('native_positive_normalized',native)]:
                    result,pred=score_case(item,candidates[key],labels[key],method,vector,aggregation=aggregation)
                    rows.extend(result);selections.extend(pred)
    if a.verify_available:
        print(json.dumps(dict(status='available_vectors_verified_no_quality_scored',coverage=dict(coverage),reused_score_rows=reused)));return
    assert dict(coverage)=={m:448 for m in BASELINES} and len(new_receipts)==2240
    assert len(rows)==len(selections)==448*(3+5*2)*6
    summaries,primary,comparisons=statistics(rows,plan)
    out=dict(status='verified_complete_all_baselines',cases=448,new_method_cases=2240,reused_score_rows=reused,
        protocol_sha256=sha(HERE/'protocol.json'),preflight_sha256=preflight_hash,numeric_amendment_sha256=sha(HERE/'NUMERIC_FIX.md'),
        mlm_verification_sha256=sha(HERE/'mlm_verification.json'),
        retrospective_scope_choice=True,new_holdout=False,methods=list(METHODS),supplementary=['FT_K1'],
        coverage=dict(coverage),origins=origins,new_receipts=new_receipts,
        gpu_operation_seconds=sum(r['seconds'] for r in costs),costs=costs,
        summaries=summaries,primary=primary,comparisons=comparisons,
        code_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__),HERE/'score_cases.py',HERE/'common.py',
            ROOT/'experiments/official/hotpot_retrieval_v3.py',ROOT/'experiments/official/hotpot_evidence.py',
            ROOT/'experiments/official/retrieval_views.py',ROOT/'experiments/official/recovery_diagnostics.py']})
    write_csv(HERE/'per_case.csv',rows);write_csv(HERE/'summary.csv',summaries);write_csv(HERE/'primary_recall10.csv',primary)
    write_json(HERE/'selections.json',selections,compact=True);write_json(HERE/'analysis.json',out)
    print(json.dumps(dict(status=out['status'],new_method_cases=2240,primary=primary),indent=2))

if __name__=='__main__':main()

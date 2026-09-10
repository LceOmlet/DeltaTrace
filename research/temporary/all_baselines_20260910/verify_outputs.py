"""Independent ranks, full costs, set metrics and paired bootstrap from raw data."""
from collections import Counter,defaultdict
import csv
import itertools
import json
import math
import re
import statistics
from pathlib import Path
import numpy as np
from common import HERE,ROOT,OLD,HOTPOT,TASKS,BASELINES,METHODS,sha,previous_cases

def data(name):return json.loads((HERE/name).read_bytes())
def csv_rows(name):
    with (HERE/name).open(newline='',encoding='utf-8') as stream:return list(csv.DictReader(stream))
def close(a,b):assert math.isclose(float(a),float(b),rel_tol=1e-11,abs_tol=1e-12),(a,b)
def key(row):
    return row['dataset'],int(row['index']),row['method'],row['aggregation'],row['view'],row['budget_unit'],float(row['fraction'])

def raw_values(item,method,aggregation,old):
    if method in ('DT','FT_K3','FT_K1'):
        assert aggregation=='signed_sum'
        values=old[item['dataset'],item['index']][1][method]
        return values[item['user_positions']] if method=='DT' else values
    path=HERE/'raw'/method/item['dataset']/f'{item["index"]:03d}'/'vectors.npz'
    with np.load(path) as vectors:
        if method=='AttnLRP':
            values=vectors['raw_aggregate'].copy()
            if aggregation!='signed_sum':
                values=np.maximum(values,0);values/=values.sum(dtype=np.float32)+np.float32(1e-12)
        else:
            matrix=vectors['raw_matrix'];weights=np.asarray(item['target_weights'])
            assert np.isfinite(matrix[weights>0,:len(item['user_positions'])]).all()
            clean=np.nan_to_num(matrix,nan=0.)
            if aggregation!='signed_sum':
                clean=np.maximum(clean,0);clean/=clean.sum(axis=1,keepdims=True,dtype=np.float32)+np.float32(1e-8)
            # Independent matrix contraction, separate from the analysis reduction.
            values=(weights.astype(np.float64)@clean.astype(np.float64)).astype(np.float32)
        return values[:len(item['user_positions'])]

def main():
    analysis=data('analysis.json');assert analysis['status']=='verified_complete_all_baselines'
    assert analysis['cases']==448 and analysis['new_method_cases']==2240
    for name,digest in analysis['code_sha256'].items():assert sha(ROOT/name)==digest
    plan=data('protocol.json');assert sha(HERE/'protocol.json')==analysis['protocol_sha256']
    for name in ('inputs','candidates','labels'):assert sha(HERE/(name+'.json'))==plan[name+'_sha256']
    inputs={(r['dataset'],r['index']):r for r in data('inputs.json')['cases']}
    candidates={(r['dataset'],r['index']):r for r in data('candidates.json')['cases']}
    labels={(r['dataset'],r['index']):r for r in data('labels.json')['cases']}
    old,_=previous_cases();rows=csv_rows('per_case.csv');predictions=data('selections.json')
    actual={key(r):r for r in rows};selected={key(r):r for r in predictions}
    assert len(actual)==len(rows)==len(predictions)==len(selected)==34944 and set(actual)==set(selected)
    value_cache={};rank_cache={};nested=0;new_rows=0;budget_checks=0
    for identity,row in actual.items():
        task,index,method,aggregation,view,unit,fraction=identity
        item=inputs[task,index];c=candidates[task,index];label=labels[task,index];prediction=selected[identity]
        vkey=task,index,method,aggregation
        if vkey not in value_cache:value_cache[vkey]=raw_values(item,method,aggregation,old)
        values=value_cache[vkey];rkey=(*vkey,view)
        if task=='hotpotqa_long':
            groups=c['all_groups'];units=c['units'];gold=set(map(tuple,label['official_keys']))
            if rkey not in rank_cache:
                scores={i:math.fsum(float(values[t]) for t in groups[i]) for i,u in enumerate(units) if u['kind']=='sentence' and groups[i]}
                rank_cache[rkey]=sorted(scores,key=lambda i:(-scores[i],units[i]['start'])),scores
            order,scores=rank_cache[rkey];assert prediction['ranked_units']==order
            for i,value in zip(order,prediction['ranked_scores']):close(scores[i],value)
            budget=math.ceil(fraction*len(c['all_body_tokens'])) if unit=='all_body_tokens' else int(fraction)
            cumulative=list(itertools.accumulate(len(groups[i]) for i in order))
            wanted=order[:sum(v<=budget for v in cumulative)] if unit=='all_body_tokens' else order[:budget]
            tokens=[t for i in wanted for t in groups[i]]
            assert prediction['selected_units']==wanted and prediction['selected_tokens']==tokens
            assert len(tokens)==len(set(tokens))
            found={(units[i]['title'],units[i]['sentence_index']) for i in wanted};tp=len(found&gold)
            precision=tp/len(found) if found else 0.;recall=tp/len(gold)
            gold_indices=[i for i,u in enumerate(units) if u['kind']=='sentence' and (u['title'],u['sentence_index']) in gold]
            goldtokens={t for i in gold_indices for t in groups[i]};assert goldtokens==set(label['gold_all_body_tokens'])
            costs=[len(groups[i]) for i in gold_indices]
            affordable=[size for size in range(len(costs)+1) for subset in itertools.combinations(costs,size)
                if (sum(subset)<=budget if unit=='all_body_tokens' else size<=budget)]
            oracle=dict(precision=precision,recall=recall,f1=2*tp/(len(found)+len(gold)),exact_match=float(found==gold),
                complete_support=float(gold<=found),token_recall=len(set(tokens)&goldtokens)/len(goldtokens),
                spent_tokens=len(tokens),unused_tokens=budget-len(tokens) if unit=='all_body_tokens' else 0,
                selected_sentences=len(wanted),empty_selection=int(not wanted),fact_recall_ceiling=max(affordable)/len(gold))
            if unit=='all_body_tokens':assert len(tokens)<=budget
        else:
            keep=sorted(c['keep']);gold=set(label['gold'])&set(keep);positive=np.maximum(values.astype(np.float32),0)
            if rkey not in rank_cache:
                if view=='raw':order=sorted(keep,key=lambda i:(-float(positive[i]),i))
                else:
                    boundaries=[m.end() for m in re.finditer(r'(?<=[.!?])[ \t]+|\n+',' '+item['prompt'])]
                    group={i:sum(end<=c['offsets'][i][0] for end in boundaries) for i in keep}
                    members=defaultdict(list)
                    for i in keep:members[group[i]].append(i)
                    density={g:math.fsum(float(positive[i]) for i in ids)/len(ids) for g,ids in members.items()}
                    order=sorted(keep,key=lambda i:(-density[group[i]],-float(positive[i]),i))
                rank_cache[rkey]=order
            order=rank_cache[rkey];budget=math.ceil(fraction*len(keep));tokens=order[:budget]
            assert prediction['selected_tokens']==tokens
            hit=len(set(tokens)&gold)
            oracle=dict(recall=hit/len(gold),precision=hit/budget,ceiling=min(1.,budget/len(gold)),gold=len(gold),eligible=len(keep))
        assert budget==int(row['budget'])==prediction['budget']
        for metric,value in oracle.items():close(value,row[metric])
        new_rows+=int(method in BASELINES);budget_checks+=1
    for rkey in rank_cache:
        task,index,method,aggregation,view=rkey
        if task!='hotpotqa_long':continue
        for lo,hi in ((.05,.1),(.1,.2)):
            first=(task,index,method,aggregation,view,'all_body_tokens',lo)
            second=(task,index,method,aggregation,view,'all_body_tokens',hi)
            assert set(selected[first]['selected_units'])<=set(selected[second]['selected_units']);nested+=1
    assert new_rows==2240*2*6 and nested==48*13*2
    finish(analysis,plan,actual,rows,budget_checks,nested)

def finish(analysis,plan,actual,rows,budget_checks,nested):
    summary_groups=defaultdict(list)
    for row in rows:
        identity=tuple(row[k] for k in ('dataset','method','aggregation','view','budget_unit','fraction'))
        summary_groups[identity].append(row)
    summary=csv_rows('summary.csv');assert len(summary)==len(summary_groups)==390
    identifiers=('dataset','method','aggregation','view','budget_unit','fraction')
    for entry in summary:
        matching=summary_groups[tuple(entry[k] for k in identifiers)]
        assert len(matching)==int(entry['n'])==plan['tasks'][entry['dataset']]['count']
        for metric,value in entry.items():
            if metric in identifiers or metric=='n' or value=='':continue
            close(value,statistics.fmean(float(r[metric]) for r in matching))
    primary={r['method']:r for r in csv_rows('primary_recall10.csv')}
    assert set(primary)==set(METHODS)|{'FT_K1'}
    source={}
    for method,row in primary.items():
        for task in TASKS:
            view,unit=('native_sentence','all_body_tokens') if task=='hotpotqa_long' else ('raw','eligible_body_tokens')
            source[task,method]=np.array([float(actual[task,i,method,'signed_sum',view,unit,.1]['recall'])
                for i in range(plan['tasks'][task]['count'])])
            close(row[task],statistics.fmean(source[task,method]))
        close(row['vt_macro'],statistics.fmean(float(row[t]) for t in TASKS if t.startswith('vt_')))
    rng=np.random.default_rng(73)
    indices={t:rng.integers(0,plan['tasks'][t]['count'],(10000,plan['tasks'][t]['count'])) for t in TASKS}
    assert len(analysis['comparisons'])==12
    for row in analysis['comparisons']:
        tasks=['hotpotqa_long'] if row['scope']=='hotpotqa_long' else list(TASKS[:-1])
        deltas=[source[t,'DT']-source[t,row['comparator']] for t in tasks]
        close(row['dt_minus_comparator'],statistics.fmean(float(x.mean()) for x in deltas))
        samples=np.zeros(10000,dtype=np.float64)
        for task,delta in zip(tasks,deltas):samples+=delta[indices[task]].sum(axis=1)/len(delta)/len(tasks)
        alpha=.05/12/2
        np.testing.assert_allclose(row['interval'],np.quantile(samples,[alpha,1-alpha]),atol=1e-12,rtol=0)
        close(row['confidence'],1-.05/12)
    receipts={(r['dataset'],r['index'],r['method']):r for r in analysis['new_receipts']}
    assert len(receipts)==2240 and Counter(r['method'] for r in receipts.values())=={m:448 for m in BASELINES}
    cost_by={(r['dataset'],r['index'],r['method']):r for r in analysis['costs']};assert set(cost_by)==set(receipts)
    for identity,expected in receipts.items():
        task,index,method=identity;folder=HERE/'raw'/method/task/f'{index:03d}'
        assert sha(folder/'results.json')==expected['results_sha256'] and sha(folder/'vectors.npz')==expected['vectors_sha256']
        record=json.loads((folder/'results.json').read_bytes())
        assert record['status']=='complete' and record['generation_calls']==0
        for metric in ('seconds','peak_allocated_bytes'):close(record[metric],cost_by[identity][metric])
        assert record['calls']==cost_by[identity]['calls']
    close(analysis['gpu_operation_seconds'],math.fsum(r['seconds'] for r in analysis['costs']))
    controller=data('audit/all_baselines_logs_v3/controller.json')
    assert controller['status']=='complete' and all(r['status']=='complete' and r['returncode']==0 for r in controller['phases'])
    compatibility=data('execution_compatibility.json');assert len(compatibility['initial_three_pilot_bitwise_controls'])==3
    for record in compatibility['initial_three_pilot_bitwise_controls']:
        suffix=Path(record['method'])/record['dataset']/f'{record["index"]:03d}'
        left=HERE/'initial_pilot'/suffix;right=HERE/'raw'/suffix
        assert sha(left/'results.json')==record['first_results_sha256'] and sha(right/'results.json')==record['second_results_sha256']
        with np.load(left/'vectors.npz') as x,np.load(right/'vectors.npz') as y:
            assert set(x.files)==set(y.files)
            for name in x.files:assert np.array_equal(x[name],y[name],equal_nan=True)
    control=data('storage_control/verification.json');assert control['status']=='bitwise_equal_all_vectors'
    left=HERE/'raw/AttnLRP/vt_h2_c3/000';right=HERE/'storage_control/all_baselines_cpu_control_v2/AttnLRP/vt_h2_c3/000'
    assert sha(left/'results.json')==control['old_results_sha256'] and sha(right/'results.json')==control['new_results_sha256']
    assert sha(left/'vectors.npz')==sha(right/'vectors.npz')==control['old_vectors_sha256']==control['new_vectors_sha256']
    frozen=json.loads((ROOT/'deltatrace/clean/sources.json').read_bytes());method_files=0
    for family in frozen['models'].values():
        for name,record in family['files'].items():assert sha(ROOT/name)==record['sha256'];method_files+=1
    assert method_files==27
    for name,record in data('baseline_source_identity.json')['files'].items():
        payload=(HERE/'source_snapshot'/name).read_bytes().replace(b'\r\n',b'\n')
        from hashlib import sha256
        assert sha256(payload).hexdigest()==record['normalized_sha256']
    previous=json.loads((HOTPOT/'verification.json').read_bytes());unchanged=0
    for name,digest in previous['output_sha256'].items():assert sha(HOTPOT/name)==digest;unchanged+=1
    assert sha(OLD/'full_recall/analysis.json')==plan['previous_full_analysis_sha256']
    report_links=0
    for name in ('README.md','RESULTS.md'):
        text=(HERE/name).read_text(encoding='utf-8')
        for target in re.findall(r'\]\(([^)]+)\)',text):
            if target.startswith(('http:','https:','#')):continue
            if target=='verification.json':continue  # Written at the end of this verification.
            assert (HERE/target.split('#')[0]).exists(),target;report_links+=1
    outputs=['analysis.json','per_case.csv','summary.csv','primary_recall10.csv','selections.json','RESULTS.md','README.md',
        'execution_compatibility.json','inputs_preflight.json','storage_control/verification.json']
    result=dict(status='passed',case_count=448,new_method_cases=2240,score_rows=budget_checks,independent_new_score_rows=2240*12,
        nested_hotpot_transitions=nested,over_budget_selections=0,paired_family_comparisons=12,
        exact_initial_controls=3,bitwise_storage_controls=1,frozen_method_files=method_files,
        unchanged_hotpot_artifacts=unchanged,report_local_links=report_links,
        output_sha256={name:sha(HERE/name) for name in outputs},verifier_sha256=sha(Path(__file__)))
    (HERE/'verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()

"""Independently rebuild ranks, prefixes, full costs and set scores from raw data."""
import csv
import itertools
import json
import math
from pathlib import Path
import re
import statistics
import unittest
import numpy as np
from artifacts_v3 import HERE,ROOT,V2,OLD,sha,full_cases,read_run,read_bytes
from test_protocol import ProtocolTests

METHODS=('DT_target','FT_K1','FT_K3')
POOLS=('signed_sum','positive_mean_eligible')
METRICS=('precision','recall','f1','exact_match','complete_support','token_recall','fact_recall_ceiling',
         'spent_all_tokens','unused_all_tokens','selected_sentences','empty_selection')
def close(x,y):assert math.isclose(float(x),float(y),rel_tol=1e-12,abs_tol=1e-12),(x,y)
def data(name):return json.loads((HERE/name).read_bytes())
def csv_rows(name):
    with (HERE/name).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def key(r):return int(r['index']),r['target_mode'],r['method'],r['pooling'],r['budget_unit'],float(r['budget_value'])

def main():
    result=unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromTestCase(ProtocolTests))
    assert result.wasSuccessful() and result.testsRun==8
    a=data('analysis.json');candidates={r['index']:r for r in data('candidates.json')['cases']};labels={r['index']:r for r in data('labels.json')['cases']}
    for path,digest in a['code_sha256'].items():assert sha(ROOT/path)==digest
    for k,name in [('protocol_sha256','protocol.json'),('candidate_sha256','candidates.json'),('labels_sha256','labels.json'),('source_audit_sha256','source_audit.json')]:assert a[k]==sha(HERE/name)
    execution=data('execution_receipt.json')
    for path,digest in execution['files'].items():assert sha(HERE/path)==digest
    assert execution['manifest_sha256']==sha(HERE/'raw/conditioned_v3/manifest.json') and execution['commit']=='258a745'
    full,_=full_cases();records={'full':full,'answer_conditioned':{}}
    progress=json.loads(read_bytes(HERE/'raw/conditioned_v3','progress.json'))
    for shard in progress['completed']:
        r,v,h=read_run(HERE/'raw/conditioned_v3'/shard['chunk'])
        assert all(shard[k]==value for k,value in h.items())
        for row in r['cases']:
            i=row['index'];old=full[i][0];prefix=f'hotpotqa_long_{i}_answer_conditioned_'
            for k in ('input_ids','input_sha256','target','references'):assert row[k]==old[k]
            assert np.array_equal(row['DT_target_details']['endpoint_target_logprobs32'],old['DT_target_details']['endpoint_target_logprobs32'])
            records['answer_conditioned'][i]=(row,{m:v[prefix+m+('_signed_full' if m=='DT_target' else '_prompt')].copy() for m in METHODS})
        v.close()
    assert len(records['answer_conditioned'])==48
    ranks={};rank_scores={};signed_values={}
    for target,cases in records.items():
        for i,(row,vectors) in cases.items():
            c=candidates[i];units=c['units'];groups=c['all_groups'];eligible=c['eligible_groups']
            assert len(c['all_body_tokens'])==len(set(c['all_body_tokens']))
            assert sorted(t for j,u in enumerate(units) if u['kind']=='sentence' for t in groups[j])==c['all_body_tokens']
            ids=[j for j,u in enumerate(units) if u['kind']=='sentence' and groups[j]]
            for m in METHODS:
                values=vectors[m][row['user_positions']] if m=='DT_target' else vectors[m];signed_values[i,target,m]=values
                for pool in POOLS:
                    if pool=='signed_sum':scores={j:math.fsum(float(values[t]) for t in groups[j]) for j in ids}
                    else:scores={j:math.fsum(max(0.,float(np.float32(values[t]))) for t in eligible[j])/len(eligible[j]) for j in ids}
                    identity=i,target,m,pool
                    ranks[identity]=sorted(ids,key=lambda j:(-scores[j],units[j]['start']));rank_scores[identity]=scores
    predictions=data('selections.json');selected={key(r):r for r in predictions};rows=csv_rows('native_per_case.csv')
    assert len(predictions)==len(selected)==len(rows)==3456
    bykey={key(r):r for r in rows};assert set(bykey)==set(selected)
    for identity,r in bykey.items():
        i,target,m,pool,unit,fraction=identity;c=candidates[i];p=selected[identity];units=c['units'];groups=c['all_groups']
        order=ranks[i,target,m,pool];assert p['ranked_units']==order
        for j,value in zip(order,p['ranked_scores']):close(value,rank_scores[i,target,m,pool][j])
        budget=math.ceil(fraction*len(c['all_body_tokens'])) if unit=='all_body_tokens' else int(fraction)
        assert budget==int(r['budget'])==p['budget']
        if unit=='sentences':wanted=order[:budget]
        else:
            cumulative=list(itertools.accumulate(len(groups[j]) for j in order))
            wanted=order[:sum(cost<=budget for cost in cumulative)]
        assert p['selected_units']==wanted and len(wanted)==len(set(wanted))
        tokens=[t for j in wanted for t in groups[j]];assert p['selected_tokens']==tokens and len(tokens)==len(set(tokens))
        predicted={(units[j]['title'],units[j]['sentence_index']) for j in wanted}
        assert set(map(tuple,p['selected_keys']))==predicted
        gold=set(map(tuple,labels[i]['official_keys']));tp=len(predicted&gold);fp=len(predicted-gold);fn=len(gold-predicted)
        gold_ids=[j for j,u in enumerate(units) if u['kind']=='sentence' and (u['title'],u['sentence_index']) in gold]
        assert len(gold_ids)==len(gold)
        gold_tokens={t for j in gold_ids for t in groups[j]};assert gold_tokens==set(labels[i]['gold_all_body_tokens'])
        costs=[len(groups[j]) for j in gold_ids]
        affordable=[size for size in range(len(costs)+1) for subset in itertools.combinations(costs,size)
                    if (sum(subset)<=budget if unit=='all_body_tokens' else size<=budget)]
        oracle=dict(precision=tp/(tp+fp) if tp+fp else 0,recall=tp/len(gold),f1=2*tp/(2*tp+fp+fn),
            exact_match=int(fp+fn==0),complete_support=int(fn==0),true_positive=tp,predicted=len(predicted),gold=len(gold),
            token_recall=len(set(tokens)&gold_tokens)/len(gold_tokens),spent_all_tokens=len(tokens),selected_sentences=len(wanted),
            unused_all_tokens=budget-len(tokens) if unit=='all_body_tokens' else 0,empty_selection=int(not wanted),
            fact_recall_ceiling=max(affordable)/len(gold))
        for metric,value in oracle.items():close(r[metric],value)
        if unit=='all_body_tokens':assert len(tokens)<=budget
    nested=0
    for i,target,m,pool in ranks:
        for small,big in ((.05,.1),(.1,.2)):
            lo=i,target,m,pool,'all_body_tokens',small;hi=i,target,m,pool,'all_body_tokens',big
            assert set(selected[lo]['selected_units'])<=set(selected[hi]['selected_units'])
            assert float(bykey[lo]['recall'])<=float(bykey[hi]['recall']);nested+=1
    assert nested==1152
    for r in csv_rows('packing_per_case.csv'):
        i=int(r['index']);target=r['target_mode'];m=r['method'];policy=r['policy'];c=candidates[i]
        groups=c['eligible_groups'] if policy=='eligible_skip' else c['all_groups'];order=ranks[i,target,m,'positive_mean_eligible']
        denominator=sum(len(groups[j]) for j in order);budget=math.ceil(float(r['fraction'])*denominator)
        picked=[];spent=0
        for j in order:
            if spent+len(groups[j])<=budget:picked.append(j);spent+=len(groups[j])
            elif policy=='all_tokens_prefix':break
        predicted={(c['units'][j]['title'],c['units'][j]['sentence_index']) for j in picked};gold=set(map(tuple,labels[i]['official_keys']))
        tp=len(predicted&gold);close(r['recall'],tp/len(gold));close(r['f1'],2*tp/(len(predicted)+len(gold)))
        actual=sum(len(c['all_groups'][j]) for j in picked);close(r['spent_all_tokens'],actual)
        close(r['excess_over_all_token_budget'],max(0,actual-math.ceil(float(r['fraction'])*len(c['all_body_tokens']))))
    for filename,source,grouping,metrics in [
        ('native_summary.csv',rows,('target_mode','pooling','method','budget_unit','budget_value'),METRICS),
        ('packing_summary.csv',csv_rows('packing_per_case.csv'),('target_mode','policy','method','fraction'),('recall','f1','spent_all_tokens','excess_over_all_token_budget')),
        ('legacy_summary.csv',csv_rows('legacy_per_case.csv'),('target_mode','view','method','fraction'),('recall',))]:
        for s in csv_rows(filename):
            matching=[r for r in source if all(str(r[k])==str(s[k]) for k in grouping)]
            assert len(matching)==int(s['n'])==48
            for metric in metrics:close(s[metric],statistics.fmean(float(r[metric]) for r in matching))
    draws=np.random.default_rng(73).integers(0,48,size=(10000,48));primary=csv_rows('primary_recall10.csv');assert len(primary)==4
    for r in primary:
        ds=np.array([float(bykey[i,r['target_mode'],'DT_target',r['pooling'],'all_body_tokens',.1]['recall']) for i in range(48)])
        fs=np.array([float(bykey[i,r['target_mode'],'FT_K3',r['pooling'],'all_body_tokens',.1]['recall']) for i in range(48)])
        close(r['dt_recall'],ds.mean());close(r['ft_k3_recall'],fs.mean());close(r['dt_minus_ft'],(ds-fs).mean())
        low,high=np.quantile((ds-fs)[draws].mean(axis=1),[.00625,.99375])
        close(r['ci_low'],low);close(r['ci_high'],high);close(r['confidence'],.9875)
    frozen=json.loads((ROOT/'deltatrace/clean/sources.json').read_bytes());count=0
    for model in frozen['models'].values():
        for path,record in model['files'].items():assert sha(ROOT/path)==record['sha256'];count+=1
    assert count==27
    old=json.loads((V2/'protocol.json').read_bytes());assert sha(OLD/'full_recall/analysis.json')==old['prior_full_analysis_sha256']
    previous=json.loads((V2/'verification.json').read_bytes())
    for path,digest in previous['output_sha256'].items():assert sha(V2/path)==digest
    links=0
    for name in ('RESULTS.md','README.md'):
        for dest in re.findall(r'\]\(([^)]+)\)',(HERE/name).read_text(encoding='utf-8')):
            if '://' not in dest:
                if dest!='verification.json':assert (HERE/dest.split('#')[0]).exists(),dest
                links+=1
    artifacts=['analysis.json','candidates.json','labels.json','source_audit.json','selections.json','native_per_case.csv',
        'packing_per_case.csv','legacy_per_case.csv','score_diagnostics.csv','native_summary.csv','primary_recall10.csv',
        'packing_summary.csv','legacy_summary.csv','RESULTS.md','README.md','execution_receipt.json','raw/conditioned_v3/manifest.json']
    receipt=dict(status='passed',regression_tests=8,independent_raw_vector_rankings=len(ranks),native_score_rows=len(rows),
        exhaustive_gold_subset_budget_checks=len(rows),nested_budget_transitions=nested,over_budget_selections=0,
        independently_recomputed_packing_rows=2592,primary_bootstrap_comparisons=4,unchanged_complete_inputs=48,
        frozen_method_files=count,prior_v2_outputs_unchanged=len(previous['output_sha256']),report_local_links=links,
        verifier_sha256=sha(Path(__file__)),builder_sha256=sha(HERE/'build_report.py'),output_sha256={p:sha(HERE/p) for p in artifacts})
    (HERE/'verification.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in receipt.items() if k!='output_sha256'},indent=2))

if __name__=='__main__':main()

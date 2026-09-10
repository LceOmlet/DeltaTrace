"""Verify preserved full inputs and score fixed full-cost native retrieval."""
import argparse
import csv
import gzip
import json
import math
from pathlib import Path
import numpy as np
from artifacts_v3 import HERE,ROOT,V2,OLD,full_cases,read_run,read_bytes,sha,byte_sha
from hotpot_retrieval_v3 import rank_sentences,select_prefix
from hotpot_evidence import supporting_fact_metrics,fact_recall_ceiling,select_sentences,regex_order
from verify_full_recall_records import verify_records

METHODS=('DT_target','FT_K1','FT_K3')
METRICS=('precision','recall','f1','exact_match','complete_support','token_recall','fact_recall_ceiling',
         'spent_all_tokens','unused_all_tokens','selected_sentences','empty_selection')

def write_json(name,value,compact=False):
    (HERE/name).write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,
        indent=None if compact else 2,separators=(',',':') if compact else None)+'\n',encoding='utf-8')

def write_csv(name,rows):
    with (HERE/name).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def verify_run(folder,cache,tok,publication,full):
    plan=json.loads((HERE/'protocol.json').read_bytes());state=json.loads(read_bytes(folder,'progress.json'))
    manifest=json.loads((folder/'manifest.json').read_bytes())
    for file in manifest['files']:
        assert sha(folder/(file['path']+'.gz'))==file['compressed_sha256']
        data=read_bytes(folder,file['path']);assert byte_sha(data)==file['sha256'] and len(data)==file['bytes']
    assert state['status']=='complete' and state['case_count']==48 and state['controls']==8
    identity=state['identity']
    assert identity['plan_sha256']==sha(HERE/'protocol.json') and plan['spec_sha256']==sha(HERE/'PROTOCOL.md')
    assert identity['driver_sha256']==sha(HERE/'evaluate_conditioned.py') and identity['controller_sha256']==sha(HERE/'run_conditioned.py')
    assert plan['attribution_module_sha256']==sha(ROOT/'experiments/official/hotpot_retrieval_v3.py')
    assert plan['input_identity_sha256']==sha(HERE/'full_input_identity.json')
    expected_inputs={r['index']:r for r in json.loads((HERE/'full_input_identity.json').read_bytes())['cases']}
    original=json.loads(gzip.decompress((publication/'raw/hotpotqa_long.results.json.gz').read_bytes()))
    originals={'hotpotqa_long':{r['index']:r for r in original['cases']}}
    control,cv,ch=read_run(OLD/'raw/answer_dev_corrected_v1')
    assert ch['results_sha256']==plan['answer_control_results_sha256'] and ch['vectors_sha256']==plan['answer_control_vectors_sha256']
    controls={r['index']:r for r in control['cases'] if r['dataset']=='hotpotqa_long' and r['target_mode']=='answer_conditioned'}
    assert len(controls)==8
    cases={};receipts=[];overlaps=[];residuals=[];costs=[];endpoint_matches=[]
    metadata={'protocol_sha256':ROOT/'experiments/official/protocol.json','clean_sources_sha256':ROOT/'deltatrace/clean/sources.json',
        'evaluation_protocol_sha256':ROOT/'experiments/official/source_protocol.json','evidence_protocol_sha256':ROOT/'experiments/official/evidence_protocol.py',
        'recovery_diagnostics_sha256':ROOT/'experiments/official/recovery_diagnostics.py'}
    assert {r['chunk'] for r in state['completed']}==set(plan['chunks'])
    for item in state['completed']:
        chunk=item['chunk'];r,v,h=read_run(folder/chunk)
        assert all(h[k]==item[k] for k in h)
        assert r['driver_sha256']==identity['driver_sha256'] and r['fair_plan_sha256']==identity['plan_sha256']
        assert r['experiment']=='hotpot-context-cost-v3' and r['stage']=='context_v3'
        assert r['indices']==plan['chunks'][chunk] and r['chunk']==chunk
        assert r['choice']['targets']=={'hotpotqa_long':'answer_conditioned'}
        assert r['weighted_sources']==plan['weighted_sources'] and r['weight_identity']==control['weight_identity']
        assert r['FT_initial_target_adapter']['sha256']==plan['FT_initial_target_adapter_sha256']
        for k,path in metadata.items():assert r[k]==sha(path)
        assert len(r['cases'])==16 and {x['index'] for x in r['cases']}==set(plan['chunks'][chunk])
        _,_,res,_=verify_records(r['cases'],v,{'hotpotqa_long':cache},originals,tok)
        residuals.extend(res);costs.extend(r['costs'])
        for row in r['cases']:
            i=row['index'];old=full[i][0];expected=expected_inputs[i];prefix=f'hotpotqa_long_{i}_answer_conditioned_'
            assert row['target_mode']=='answer_conditioned' and row['complete_input_and_reference_preserved'] and i not in cases
            assert row['input_ids']==old['input_ids'] and row['target']==cache[i]['target']==old['target']
            assert row['input_sha256']==old['input_sha256']==expected['input_sha256']
            assert row['references']==old['references'] and row['references']['full']==expected['reference_sha256']
            lo,hi=cache[i]['indices_to_explain']
            wanted=[w*float(lo<=j<=hi) for j,w in enumerate(old['target_weights'])]
            assert row['target_weights']==wanted and wanted!=old['target_weights']
            assert not any(wanted[:lo]) and not any(wanted[hi+1:]) and any(wanted)
            # Same model executions; only downstream output attribution weights changed.
            assert np.array_equal(row['DT_target_details']['endpoint_target_logprobs32'],old['DT_target_details']['endpoint_target_logprobs32'])
            endpoint_matches.append(i)
            values={m:v[prefix+m+('_signed_full' if m=='DT_target' else '_prompt')].copy() for m in METHODS}
            if i in controls:
                assert row['development_overlap_bitwise_equal']
                for key in ('input_ids','input_sha256','target','target_mode','target_weights','keep','gold','references'):
                    assert row[key]==controls[i][key]
                for m in METHODS:assert np.array_equal(values[m],cv[prefix+m+('_signed_full' if m=='DT_target' else '_prompt')])
                overlaps.append(i)
            cases[i]=(row,values)
        receipts.append(dict(chunk=chunk,**h));v.close()
        print(json.dumps(dict(verified_chunk=chunk,full_inputs_preserved=len(cases))),flush=True)
    cv.close();assert set(cases)==set(range(48)) and set(overlaps)==set(controls)
    return cases,dict(status='verified_all_48_full_inputs_references_seeds_and_vectors',identity=identity,shards=receipts,
        bitwise_development_overlap_indices=sorted(overlaps),endpoint_logprobs_bitwise_equal_indices=sorted(endpoint_matches),
        weight_identity=control['weight_identity'],residuals=residuals,
        gpu_operation_seconds=sum(r['seconds'] for r in costs if r['status']=='returned'))

def summarize(rows,group_keys,metrics,*,principal=False):
    from collections import defaultdict
    groups=defaultdict(dict)
    for r in rows:
        key=tuple(r[k] for k in group_keys);pair=(r['index'],r['method'])
        assert pair not in groups[key];groups[key][pair]=r
    draws=np.random.default_rng(73).integers(0,48,size=(10000,48));summaries=[];primary=[]
    for key,data in sorted(groups.items()):
        assert set(data)=={(i,m) for i in range(48) for m in METHODS}
        item=dict(zip(group_keys,key));item.update(n=48,methods={},paired={})
        values={m:{metric:np.array([data[i,m][metric] for i in range(48)],dtype=np.float64) for metric in metrics} for m in METHODS}
        for m in METHODS:item['methods'][m]={metric:float(v.mean()) for metric,v in values[m].items()}
        for comparator in ('FT_K1','FT_K3'):
            item['paired'][comparator]={}
            for metric in metrics:
                delta=values['DT_target'][metric]-values[comparator][metric]
                main=principal and item['budget_unit']=='all_body_tokens' and item['budget_value']==.1 and comparator=='FT_K3' and metric=='recall'
                level=.9875 if main else .95;alpha=(1-level)/2
                effect=dict(dt_minus_ft=float(delta.mean()),interval=np.quantile(delta[draws].mean(axis=1),[alpha,1-alpha]).tolist(),confidence=level)
                item['paired'][comparator][metric]=effect
                if main:primary.append(dict(target_mode=item['target_mode'],pooling=item['pooling'],n=48,
                    dt=item['methods']['DT_target']['recall'],ft=item['methods']['FT_K3']['recall'],**effect))
        summaries.append(item)
    return summaries,primary

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('cache','tokenizer','publication'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--run',type=Path,default=HERE/'raw/conditioned_v3')
    a=p.parse_args()
    from tokenizers import Tokenizer
    tok=Tokenizer.from_file(str(a.tokenizer));cache=[json.loads(l) for l in a.cache.read_bytes().splitlines()]
    plan=json.loads((HERE/'protocol.json').read_bytes());candidate=json.loads((HERE/'candidates.json').read_bytes());labeldata=json.loads((HERE/'labels.json').read_bytes())
    assert candidate['protocol_sha256']==sha(HERE/'protocol.json') and candidate['cache_sha256']==sha(a.cache)
    assert candidate['tokenizer_sha256']==sha(a.tokenizer) and candidate['builder_sha256']==sha(HERE/'prepare_candidates.py')
    assert candidate['retrieval_module_sha256']==sha(ROOT/'experiments/official/hotpot_retrieval_v3.py')
    assert candidate['mapping_module_sha256']==sha(ROOT/'experiments/official/hotpot_evidence.py')
    assert labeldata['candidate_sha256']==sha(HERE/'candidates.json') and labeldata['official_source_sha256']==plan['source_sha256']
    cby={r['index']:r for r in candidate['cases']};lby={r['index']:r for r in labeldata['cases']}
    full,full_receipts=full_cases();conditioned,verification=verify_run(a.run,cache,tok,a.publication,full)
    rows=[];selections=[];packing=[];legacy=[];score_diagnostics=[];nested_checks=0
    old_selection=json.loads((V2/'selections.json').read_bytes())
    old_full_pred={(r['index'],r['method'],r['budget_value']):r['selected_units'] for r in old_selection if r['target_mode']=='full' and r['budget_unit']=='body_tokens'}
    for target,records in [('full',full),('answer_conditioned',conditioned)]:
        for i in range(48):
            row,vectors=records[i];c=cby[i];label=lby[i];units=c['units'];groups=c['all_groups'];eligible=c['eligible_groups']
            assert row['keep']==c['old_keep'] and row['input_ids']==full[i][0]['input_ids']
            assert c['prompt_sha256']==byte_sha(cache[i]['prompt'].encode())
            allbody=set(c['all_body_tokens']);oldbody={t for u,g in zip(units,eligible) if u['kind']=='sentence' for t in g}
            gold=set(label['gold_all_body_tokens']);goldkeys=label['official_keys']
            offsets=tok.encode(' '+cache[i]['prompt'],add_special_tokens=False).offsets
            for method in METHODS:
                values=vectors[method][row['user_positions']] if method=='DT_target' else vectors[method]
                assert np.isfinite(values).all()
                omitted=allbody-oldbody;assert not np.count_nonzero(values[sorted(omitted)])
                score_diagnostics.append(dict(index=i,target_mode=target,method=method,
                    all_body_tokens=len(allbody),eligible_body_tokens=len(oldbody),omitted_tokens=len(omitted),
                    signed_body_sum=float(np.sum(values[sorted(allbody)],dtype=np.float64)),
                    positive_body_sum=float(np.maximum(values[sorted(allbody)],0).sum(dtype=np.float64)),
                    negative_body_sum=float(np.minimum(values[sorted(allbody)],0).sum(dtype=np.float64))))
                for pooling in plan['pooling']:
                    order,pooled=rank_sentences(values,units,groups,eligible,pooling=pooling)
                    previous=set();previous_recall=0.
                    for budget_unit,budget_values in [('all_body_tokens',plan['body_token_budgets']),('sentences',plan['sentence_budgets'])]:
                        for value in budget_values:
                            budget=math.ceil(value*len(allbody)) if budget_unit=='all_body_tokens' else value
                            kwargs={'token_budget':budget} if budget_unit=='all_body_tokens' else {'sentence_budget':budget}
                            ids,tokens=select_prefix(order,groups,**kwargs)
                            metrics=supporting_fact_metrics(ids,units,goldkeys)
                            if budget_unit=='all_body_tokens':
                                assert previous<=set(ids) and metrics['recall']>=previous_recall
                                if value!=plan['body_token_budgets'][0]:nested_checks+=1
                                previous=set(ids);previous_recall=metrics['recall']
                            rows.append(dict(index=i,target_mode=target,method=method,pooling=pooling,budget_unit=budget_unit,budget_value=value,
                                budget=budget,all_body_tokens=len(allbody),**metrics,spent_all_tokens=len(tokens),
                                unused_all_tokens=budget-len(tokens) if budget_unit=='all_body_tokens' else 0,
                                selected_sentences=len(ids),empty_selection=int(not ids),token_recall=len(set(tokens)&gold)/len(gold),
                                fact_recall_ceiling=fact_recall_ceiling(units,groups,goldkeys,**kwargs)))
                            selections.append(dict(index=i,target_mode=target,method=method,pooling=pooling,budget_unit=budget_unit,budget_value=value,
                                budget=budget,ranked_units=order,ranked_scores=[pooled[j] for j in order],selected_units=ids,
                                selected_keys=[[units[j]['title'],units[j]['sentence_index']] for j in ids],selected_tokens=tokens))
                    if pooling=='positive_mean_eligible':
                        for fraction in plan['body_token_budgets']:
                            for policy in ('eligible_skip','all_tokens_skip','all_tokens_prefix'):
                                use_groups=eligible if policy=='eligible_skip' else groups
                                denominator=len(oldbody) if policy=='eligible_skip' else len(allbody)
                                budget=math.ceil(fraction*denominator)
                                ids,_=(select_prefix if policy=='all_tokens_prefix' else select_sentences)(order,use_groups,token_budget=budget)
                                if policy=='eligible_skip' and target=='full':assert ids==old_full_pred[i,method,fraction]
                                tokens={t for j in ids for t in groups[j]}
                                metrics=supporting_fact_metrics(ids,units,goldkeys)
                                packing.append(dict(index=i,target_mode=target,method=method,policy=policy,fraction=fraction,
                                    recall=metrics['recall'],f1=metrics['f1'],spent_all_tokens=len(tokens),
                                    excess_over_all_token_budget=max(0,len(tokens)-math.ceil(fraction*len(allbody)))))
                positive=np.maximum(values.astype(np.float32),0)
                orders={'raw_restored':sorted(c['old_keep'],key=lambda t:(-float(positive[t]),t)),
                    'regex_anchor_restored':regex_order(' '+cache[i]['prompt'],offsets,positive,c['old_keep'],content_anchor=True)}
                gold_old=set(label['restored_overlap_gold_old_eligible'])
                for view,order in orders.items():
                    for fraction in plan['body_token_budgets']:
                        budget=math.ceil(fraction*len(c['old_keep']))
                        legacy.append(dict(index=i,target_mode=target,method=method,view=view,fraction=fraction,
                            recall=len(set(order[:budget])&gold_old)/len(gold_old),budget=budget))
    counts=dict(rows=len(rows),selections=len(selections),packing=len(packing),legacy=len(legacy),nested_checks=nested_checks)
    assert counts==dict(rows=3456,selections=3456,packing=2592,legacy=1728,nested_checks=1152),counts
    summary,primary=summarize(rows,['target_mode','pooling','budget_unit','budget_value'],METRICS,principal=True)
    packing_summary,_=summarize(packing,['target_mode','policy','fraction'],('recall','f1','spent_all_tokens','excess_over_all_token_budget'))
    legacy_summary,_=summarize(legacy,['target_mode','view','fraction'],('recall',))
    assert len(primary)==4
    verification.update(full_vector_origins=full_receipts,unchanged_full_model_inputs=48,nested_budget_transitions=nested_checks,
        zero_attribution_on_excluded_body_tokens_checks=288,old_full_positive_mean_skip_selections_reproduced=432)
    dependencies=[Path(__file__),HERE/'artifacts_v3.py',HERE/'prepare_candidates.py',HERE/'test_protocol.py',
        ROOT/'experiments/official/hotpot_retrieval_v3.py',ROOT/'experiments/official/hotpot_evidence.py',
        OLD/'verify_full_recall_records.py',V2/'common.py']
    out=dict(status='verified_complete_context_preserving_full_cost_evaluation',case_count=48,official_gold_only=True,
        neutral_protocol_proven=False,new_holdout=False,protocol_sha256=sha(HERE/'protocol.json'),
        candidate_sha256=sha(HERE/'candidates.json'),labels_sha256=sha(HERE/'labels.json'),source_audit_sha256=sha(HERE/'source_audit.json'),
        verification=verification,primary_comparisons=primary,native_summary=summary,packing_summary=packing_summary,legacy_summary=legacy_summary,
        code_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in dependencies},
        bootstrap=dict(draws=10000,seed=73,family=4,confidence=.9875,paired_by_case=True,descriptive=True))
    write_csv('native_per_case.csv',rows);write_csv('packing_per_case.csv',packing);write_csv('legacy_per_case.csv',legacy)
    write_csv('score_diagnostics.csv',score_diagnostics);write_json('selections.json',selections,compact=True);write_json('analysis.json',out)
    print(json.dumps(dict(status=out['status'],primary_comparisons=primary),indent=2))

if __name__=='__main__':main()

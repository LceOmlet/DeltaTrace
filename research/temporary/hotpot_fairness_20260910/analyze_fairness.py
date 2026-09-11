"""Verify all raw artifacts and score the explicitly frozen HotpotQA correction."""
import argparse
import csv
import json
import math
from pathlib import Path
import numpy as np
from common import HERE,ROOT,OLD,TASK,sha,byte_sha,read_bytes,read_run,full_cases
from hotpot_evidence import (regex_order,native_token_order,sentence_order,select_sentences,
                            supporting_fact_metrics,fact_recall_ceiling)
from retrieval_views import sentence_density_order
from verify_full_recall_records import verify_records

METHODS=('DT_target','FT_K1','FT_K3')
METRICS=('precision','recall','f1','exact_match','complete_support','token_recall','fact_recall_ceiling',
         'spent_tokens','unused_tokens','selected_sentences')

def verify_archive(folder):
    manifest=json.loads((folder/'manifest.json').read_bytes())
    for row in manifest['files']:
        path=folder/(row['path']+'.gz')
        assert sha(path)==row['compressed_sha256']
        data=read_bytes(folder,row['path'])
        assert len(data)==row['bytes'] and byte_sha(data)==row['sha256']

def verify_answer(cache,tok,publication):
    plan=json.loads((HERE/'protocol.json').read_bytes());folder=HERE/'raw/answer_v2';verify_archive(folder)
    state=json.loads(read_bytes(folder,'progress.json'));identity=state['identity']
    assert state['status']=='complete' and state['case_count']==48 and state['controls']==8
    assert identity['plan_sha256']==sha(HERE/'protocol.json') and plan['spec_sha256']==sha(HERE/'PROTOCOL.md')
    assert identity['driver_sha256']==sha(HERE/'evaluate_answer.py') and identity['controller_sha256']==sha(HERE/'run_answer.py')
    control,cv,ch=read_run(OLD/'raw/answer_dev_corrected_v1')
    assert ch['results_sha256']==plan['answer_control_results_sha256'] and ch['vectors_sha256']==plan['answer_control_vectors_sha256']
    controls={r['index']:r for r in control['cases'] if r['dataset']==TASK and r['target_mode']=='answer_only'}
    assert len(controls)==8
    original=json.loads(__import__('gzip').decompress((publication/'raw'/(TASK+'.results.json.gz')).read_bytes()))
    originals={TASK:{r['index']:r for r in original['cases']}}
    cases={};overlaps=[];receipts=[];residuals=[];costs=[]
    metadata={'protocol_sha256':ROOT/'experiments/official/protocol.json',
              'clean_sources_sha256':ROOT/'deltatrace/clean/sources.json',
              'evaluation_protocol_sha256':ROOT/'experiments/official/source_protocol.json',
              'evidence_protocol_sha256':ROOT/'experiments/official/evidence_protocol.py',
              'recovery_diagnostics_sha256':ROOT/'experiments/official/recovery_diagnostics.py'}
    for item in state['completed']:
        chunk=item['chunk'];r,v,h=read_run(folder/chunk)
        assert all(h[k]==item[k] for k in h)
        assert r['driver_sha256']==identity['driver_sha256'] and r['fair_plan_sha256']==identity['plan_sha256']
        assert r['experiment']=='hotpot-answer-fairness-v2' and r['stage']=='fairness'
        assert r['chunk']==chunk and r['indices']==plan['chunks'][chunk]
        assert r['choice']['targets']=={TASK:'answer_only'} and r['weighted_sources']==plan['weighted_sources']
        assert r['weight_identity']==control['weight_identity']
        for name,path in metadata.items():assert r[name]==sha(path)
        assert r['FT_initial_target_adapter']['sha256']==plan['FT_initial_target_adapter_sha256']
        assert {x['index'] for x in r['cases']}==set(plan['chunks'][chunk]) and len(r['cases'])==16
        _,_,res,_=verify_records(r['cases'],v,{TASK:cache},originals,tok)
        residuals.extend(res);costs.extend(r['costs'])
        for row in r['cases']:
            i=row['index'];prefix=f'{TASK}_{i}_answer_only_'
            assert row['dataset']==TASK and row['target_mode']=='answer_only' and i not in cases
            values={m:v[prefix+m+('_signed_full' if m=='DT_target' else '_prompt')].copy() for m in METHODS}
            if i in controls:
                assert row['development_overlap_bitwise_equal']
                for key in ('input_ids','input_sha256','target','target_mode','target_weights','keep','gold','references'):
                    assert row[key]==controls[i][key]
                for m in METHODS:assert np.array_equal(values[m],cv[prefix+m+('_signed_full' if m=='DT_target' else '_prompt')])
                overlaps.append(i)
            cases[i]=(row,values)
        receipts.append(dict(chunk=chunk,**h));v.close()
        print(json.dumps(dict(verified_chunk=chunk,cases=len(cases))),flush=True)
    cv.close();assert set(cases)==set(range(48)) and set(overlaps)==set(controls)
    return cases,dict(status='all_48_answer_cases_independently_verified',shards=receipts,
        development_bitwise_overlap_indices=sorted(overlaps),weight_identity=control['weight_identity'],
        identity=identity,residuals=residuals,gpu_operation_seconds=sum(c['seconds'] for c in costs if c['status']=='returned'))

def percentile(values,indices,level):
    means=np.asarray(values,dtype=np.float64)[indices].mean(axis=1)
    alpha=(1-level)/2
    return np.quantile(means,[alpha,1-alpha]).tolist()

def summaries(rows,group_keys,metrics,subsets):
    from collections import defaultdict
    grouped=defaultdict(dict)
    for row in rows:
        key=tuple(row[k] for k in group_keys);index=row['index'];method=row['method']
        assert (index,method) not in grouped[key]
        grouped[key][index,method]=row
    output=[];primary=[];draw_cache={}
    for subset,indices in subsets.items():
        n=len(indices)
        if n not in draw_cache:draw_cache[n]=np.random.default_rng(73).integers(0,n,size=(10000,n))
        draws=draw_cache[n]
        for key,data in sorted(grouped.items()):
            item=dict(zip(group_keys,key));item.update(subset=subset,n=n,methods={},paired={})
            arrays={m:{metric:np.array([data[i,m][metric] for i in indices],dtype=np.float64) for metric in metrics} for m in METHODS}
            for m in METHODS:item['methods'][m]={metric:float(v.mean()) for metric,v in arrays[m].items()}
            for comparator in ('FT_K1','FT_K3'):
                item['paired'][comparator]={}
                for metric in metrics:
                    delta=arrays['DT_target'][metric]-arrays[comparator][metric]
                    main=(subset=='all48' and comparator=='FT_K3' and item.get('budget_unit')=='body_tokens'
                          and item.get('budget_value')==.1 and metric=='recall')
                    level=.9875 if main else .95
                    effect=dict(dt_minus_ft=float(delta.mean()),interval=percentile(delta,draws,level),confidence=level)
                    item['paired'][comparator][metric]=effect
                    if main:primary.append(dict(target_mode=item['target_mode'],labels=item['labels'],n=n,
                        dt=float(arrays['DT_target'][metric].mean()),ft=float(arrays[comparator][metric].mean()),**effect))
            output.append(item)
    return output,primary

def write_csv(name,rows):
    with (HERE/name).open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('cache','tokenizer','publication'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args()
    from tokenizers import Tokenizer
    tok=Tokenizer.from_file(str(a.tokenizer));cache=[json.loads(s) for s in a.cache.read_bytes().splitlines()]
    candidates=json.loads((HERE/'candidates.json').read_bytes());labeldata=json.loads((HERE/'labels.json').read_bytes())
    plan=json.loads((HERE/'protocol.json').read_bytes())
    assert candidates['cache_sha256']==sha(a.cache)==plan['tasks'][TASK]['cache_sha256']
    assert candidates['tokenizer_sha256']==sha(a.tokenizer)
    assert candidates['unit_code_sha256']==sha(ROOT/'experiments/official/hotpot_evidence.py')
    assert candidates['builder_sha256']==sha(HERE/'prepare_inputs.py')
    assert labeldata['candidate_sha256']==sha(HERE/'candidates.json')
    assert labeldata['semantic_review_sha256']==sha(HERE/'semantic_review.json')
    cby={r['index']:r for r in candidates['cases']};lby={r['index']:r for r in labeldata['cases']}
    full,full_receipts=full_cases();answer,verification=verify_answer(cache,tok,a.publication)
    facts=[];diagnostics=[];selected_rows=[];target_rows=[];regex_checks=0
    for mode,records in [('full',full),('answer_only',answer)]:
        for i in range(48):
            row,vectors=records[i];c=cby[i];labels=lby[i];text=' '+cache[i]['prompt'];enc=tok.encode(text,add_special_tokens=False)
            assert row['keep']==c['keep']==full[i][0]['keep']
            assert row['user_positions']==full[i][0]['user_positions'] and row['prompt_length']==full[i][0]['prompt_length']
            assert row['input_ids'][:row['prompt_length']]==full[i][0]['input_ids'][:full[i][0]['prompt_length']]
            assert c['prompt_sha256']==byte_sha(cache[i]['prompt'].encode())
            if mode=='answer_only':
                assert row['original_answer_token_span']==full[i][0]['original_answer_token_span']
                target_rows.append(dict(index=i,full_target_tokens=full[i][0]['target_length']-1,answer_target_tokens=row['target_length']-1,
                    full_seed_tokens=sum(full[i][0]['target_weights']),answer_seed_tokens=sum(row['target_weights'])))
            units,groups,keep=c['units'],c['groups'],c['keep']
            for method in METHODS:
                values=vectors[method]
                if method=='DT_target':values=values[row['user_positions']]
                values=np.maximum(values.astype(np.float32),0)
                legacy=regex_order(text,enc.offsets,values,keep,content_anchor=False)
                assert legacy==sentence_density_order(text,enc.offsets,values,keep);regex_checks+=1
                raw=sorted(keep,key=lambda t:(-float(values[t]),t))
                orders={'raw_cached':raw,'raw_restored':raw,'regex_cached':legacy,'regex_restored':legacy,
                    'regex_anchor_restored':regex_order(text,enc.offsets,values,keep,content_anchor=True),
                    'native_token_restored':native_token_order(values,groups)}
                for view,order in orders.items():
                    gold=set(labels['cached_gold_tokens'] if view.endswith('_cached') else labels['restored_overlap_gold_tokens'])
                    for fraction in (.05,.1,.2):
                        k=math.ceil(fraction*len(keep));selected=set(order[:k]);recall=len(selected&gold)/len(gold)
                        if view in ('raw_cached','regex_cached'):
                            stored=row['metrics'][method]['raw' if view=='raw_cached' else 'density']['points']
                            point=next(p for p in stored if p['fraction']==fraction)
                            assert recall==point['recall'] and k==point['budget']
                        diagnostics.append(dict(index=i,target_mode=mode,method=method,view=view,fraction=fraction,
                            recall=recall,budget=k,gold_tokens=len(gold),ceiling=min(1,k/len(gold))))
                order=sentence_order(values,units,groups)
                for budget_unit,budget_values in [('body_tokens',(.05,.1,.2)),('sentences',(2,4,8))]:
                    for budget_value in budget_values:
                        budget=math.ceil(budget_value*len(c['body_tokens'])) if budget_unit=='body_tokens' else budget_value
                        kwargs={'token_budget':budget} if budget_unit=='body_tokens' else {'sentence_budget':budget}
                        selected,tokens=select_sentences(order,groups,**kwargs)
                        selected_rows.append(dict(index=i,target_mode=mode,method=method,budget_unit=budget_unit,budget_value=budget_value,
                            budget=budget,selected_units=selected,selected_keys=[[units[j]['title'],units[j]['sentence_index']] for j in selected],
                            selected_tokens=tokens))
                        for label_view in plan['labels']:
                            gold_keys=labels[label_view];gold_set=set(map(tuple,gold_keys))
                            gold_tokens={t for j,u in enumerate(units) if (u['title'],u['sentence_index']) in gold_set for t in groups[j]}
                            metrics=supporting_fact_metrics(selected,units,gold_keys)
                            facts.append(dict(index=i,target_mode=mode,method=method,labels=label_view,budget_unit=budget_unit,budget_value=budget_value,
                                budget=budget,eligible_body_tokens=len(c['body_tokens']),**metrics,spent_tokens=len(tokens),
                                unused_tokens=budget-len(tokens) if budget_unit=='body_tokens' else 0,
                                selected_sentences=len(selected),gold_tokens=len(gold_tokens),token_recall=len(set(tokens)&gold_tokens)/len(gold_tokens),
                                fact_recall_ceiling=fact_recall_ceiling(units,groups,gold_keys,**kwargs)))
    assert len(facts)==3456 and len(diagnostics)==5184 and regex_checks==288
    subsets={'all48':list(range(48)),'unambiguous45':[i for i in range(48) if i not in plan['ambiguity_indices']]}
    native_summary,primary=summaries(facts,['target_mode','labels','budget_unit','budget_value'],METRICS,subsets)
    diagnostic_summary,_=summaries(diagnostics,['target_mode','view','fraction'],('recall','ceiling'),{'all48':list(range(48))})
    assert len(primary)==4
    verification.update(full_origins=full_receipts,legacy_regex_bitwise_order_checks=regex_checks,
        all_native_candidates_and_prompts_shared=True,all_legacy_scores_reproduced=True)
    dependencies=[Path(__file__),HERE/'common.py',HERE/'prepare_inputs.py',HERE/'test_evidence.py',ROOT/'experiments/official/hotpot_evidence.py',
        OLD/'verify_full_recall_records.py',OLD/'analyze_recall_pilot.py',ROOT/'experiments/official/retrieval_views.py',
        ROOT/'experiments/official/evidence_protocol.py']
    out=dict(status='verified_complete_hotpot_fairness_correction',case_count=48,targets=['answer_only','full'],
        retrospective=True,independent_holdout=False,protocol_sha256=sha(HERE/'protocol.json'),
        candidate_sha256=sha(HERE/'candidates.json'),labels_sha256=sha(HERE/'labels.json'),
        boundary_note_sha256=sha(HERE/'TOKEN_BOUNDARY_NOTE.md'),source_audit_sha256=sha(HERE/'source_audit.json'),
        verification=verification,code_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in dependencies},
        primary_comparator='FT_K3',primary_comparisons=primary,native_summary=native_summary,diagnostic_summary=diagnostic_summary,
        bootstrap=dict(draws=10000,seed=73,paired_by_case=True,primary_family_size=4,primary_confidence=.9875,
            interpretation='Descriptive intervals on a retrospective complete benchmark, not a new confirmatory holdout.'),
        target_token_summary={k:float(np.mean([r[k] for r in target_rows])) for k in target_rows[0] if k!='index'})
    write_csv('native_per_case.csv',facts);write_csv('diagnostic_per_case.csv',diagnostics);write_csv('target_tokens.csv',target_rows)
    (HERE/'selections.json').write_text(json.dumps(selected_rows,ensure_ascii=False,separators=(',',':'))+'\n',encoding='utf-8')
    (HERE/'analysis.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=out['status'],primary_comparisons=primary),indent=2))

if __name__=='__main__':main()

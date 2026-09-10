"""Independently verify the fixed answer-only layerwise PV comparison."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'experiments/official'))
from evidence_protocol import source_span,select_source_tokens,reference_token_ids,recovery_curve
from retrieval_views import sentence_density_order
from analyze import paired_bootstrap,interval
from analyze_recall_pilot import same_tree

TASKS=['vt_h2_c3','vt_h4_c1','vt_h6_c1','vt_h10_c1','hotpotqa_long']
FRACTIONS=[.05,.1,.2]
REFS=['content_P1','layer_symmetric']


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run','publication','data','tokenizer','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--parent',type=Path)
    p.add_argument('--freeze-choice',action='store_true')
    a=p.parse_args()
    from tokenizers import Tokenizer
    tok=Tokenizer.from_file(str(a.tokenizer))
    r=json.loads((a.run/'results.json').read_bytes())
    split=json.loads((HERE/'symmetric_split.json').read_bytes())
    assert r['status']=='complete' and r['experiment']=='symmetric-pilot-v1'
    assert r['driver_sha256']==digest(HERE/'evaluate_symmetric.py')
    assert r['split_sha256']==digest(HERE/'symmetric_split.json')
    assert r['plan_sha256']==split['plan_sha256']==digest(HERE/'SYMMETRIC_PILOT.md')
    assert r['vectors_sha256']==digest(a.run/'vectors.npz')
    for name,value in r['weighted_sources'].items():assert value==digest(HERE/name)
    stage=r['stage'];tasks=['vt_h2_c3','hotpotqa_long'] if stage=='development' else TASKS
    assert not a.freeze_choice or stage=='development'
    expected={(t,i) for t in tasks for i in split['tasks'][t][stage]}
    assert len(r['cases'])==len(expected) and {(x['dataset'],x['index']) for x in r['cases']}==expected
    assert r['selected_counts']=={t:len(split['tasks'][t][stage]) for t in tasks}
    vectors=np.load(a.run/'vectors.npz')
    parent_rows=parent_vectors=None
    if stage=='development':
        assert a.parent is not None
        parent=json.loads((a.parent/'results.json').read_bytes())
        assert parent['status']=='complete' and r['parent_control']['results_sha256']==digest(a.parent/'results.json')
        assert r['parent_control']['vectors_sha256']==parent['vectors_sha256']==digest(a.parent/'vectors.npz')
        parent_rows={(x['dataset'],x['index']):x for x in parent['cases'] if x['target_mode']=='answer_only'}
        parent_vectors=np.load(a.parent/'vectors.npz')
    caches={};originals={}
    for t in tasks:
        path=a.data/(t+'.jsonl');assert digest(path)==split['tasks'][t]['cache_sha256']
        caches[t]=[json.loads(x) for x in path.read_text(encoding='utf-8').splitlines()]
        old=json.loads(gzip.decompress((a.publication/'raw'/(t+'.results.json.gz')).read_bytes()))
        originals[t]={x['index']:x for x in old['cases']}
    rows=[];residuals=[];repeat_checks=0
    for row in r['cases']:
        t,i=row['dataset'],row['index'];cache,old=caches[t][i],originals[t][i]
        assert row['status']=='complete' and row['target_mode']=='answer_only'
        original_target=tok.encode(cache['target'],add_special_tokens=False)
        start,end=cache['indices_to_explain'];cs,ce=original_target.offsets[start][0],original_target.offsets[end][1]
        assert row['original_answer_token_span']==[start,end] and row['original_answer_char_span']==[cs,ce]
        assert row['target']==cache['target'][cs:ce]
        assert row['original_target_sha256']==hashlib.sha256(cache['target'].encode()).hexdigest()
        ids=np.asarray(row['input_ids'],dtype=np.int64);positions=row['user_positions'];eos=old['input_ids'][-1]
        assert hashlib.sha256(ids.tobytes()).hexdigest()==row['input_sha256']
        for name in ('user_positions','gold','prompt_length'):assert row[name]==old[name]
        assert row['author_keep']==old['keep']
        assert row['input_ids'][:row['prompt_length']]==old['input_ids'][:old['prompt_length']]
        target_ids=tok.encode(row['target'],add_special_tokens=False).ids+[eos]
        assert row['input_ids'][row['prompt_length']:]==target_ids and row['target_length']==len(target_ids)
        text=' '+cache['prompt'];enc=tok.encode(text,add_special_tokens=False)
        assert len(enc.ids)==len(positions)
        assert np.flatnonzero(np.asarray(enc.ids)!=ids[positions]).tolist()==row['standalone_token_boundary_differences']
        span=source_span(t,cache['prompt']);assert span==row['source_span']
        keep=select_source_tokens(span,enc.offsets,row['author_keep'],row['gold']);assert keep==row['keep']
        for name,token in [('eos',eos)]:
            ref=np.asarray(reference_token_ids(ids,[positions[j] for j in row['author_keep']],token),dtype=np.int64)
            assert hashlib.sha256(ref.tobytes()).hexdigest()==row['references'][name]
        weights=np.asarray(row['target_weights'],dtype=np.float64)
        assert weights.shape==(len(target_ids),) and set(weights)<={0.,1.} and weights.sum()>0
        for hops in (1,3):
            calls=row[f'FT_K{hops}_actual_target_aggregation'];assert calls
            assert all((x['start'],x['end'])==(0,len(target_ids)-2) for x in calls)
            actual=np.zeros_like(weights);actual[:-1]=calls[0]['weights'] if calls[0]['weights'] is not None else 1.
            assert np.array_equal(actual,weights)
        prefix=f'{t}_{i}_answer_only_'
        if parent_rows is not None:
            assert row['input_sha256']==parent_rows[t,i]['input_sha256']
            for method,old_method in [('DT_content_P1_signed_full','DT_target_signed_full'),('FT_K1_prompt','FT_K1_prompt'),('FT_K3_prompt','FT_K3_prompt')]:
                assert np.array_equal(vectors[prefix+method],parent_vectors[prefix+old_method])
                repeat_checks+=1
        deltas={}
        for method,views in row['metrics'].items():
            if method.startswith('DT_'):
                signed=vectors[prefix+method+'_signed_full'];assert signed.shape==ids.shape and np.isfinite(signed).all()
                scores=signed[positions].astype(np.float32)
                ref=method[3:]
                assert ref in REFS
                detail=row[method+'_details'];l0,l1=np.asarray(detail['endpoint_target_logprobs32'],dtype=np.float64)
                delta=float(((l1-l0)*weights).sum());deltas[ref]=delta
                assert np.isclose(delta,detail['target_delta_score32_sum64'],atol=1e-8)
                assert np.isclose(signed.sum(),detail['signed_sum'],atol=1e-10)
                assert detail['pv_rule']==ref
                activity=detail['finite_attention_activity']
                assert len(activity)==(72 if ref=='layer_symmetric' else 36)
                assert all(x['calls_attempted']==x['calls_enqueued']==1 for x in activity)
                assert sorted(x['layer'] for x in activity)==sorted(list(range(36))*(2 if ref=='layer_symmetric' else 1))
                assert np.array_equal(row['DT_content_P1_details']['endpoint_target_logprobs32'],row['DT_layer_symmetric_details']['endpoint_target_logprobs32'])
                residuals.append({'dataset':t,'index':i,'rule':ref,'delta':delta,'residual':detail['unassigned_total']})
            else:
                assert method in ('FT_K1','FT_K3');scores=vectors[prefix+method+'_prompt'].astype(np.float32)
            positive=np.maximum(scores,0);order=sentence_density_order(text,enc.offsets,positive,keep)
            assert len(order)==len(keep) and set(order)==set(keep)
            rank=np.zeros_like(positive);rank[order]=np.arange(len(keep),0,-1)
            for view,value in [('raw',positive),('density',rank)]:
                curve=recovery_curve(value,keep,row['gold'],FRACTIONS);same_tree(views[view],curve)
                for q in curve['points']:
                    rows.append({'dataset':t,'index':i,'method':method,'view':view,'fraction':q['fraction'],
                        'recall':q['recall'],'budget':q['budget'],'gold':q['gold'],'ceiling':q['ceiling']})
    table={(x['dataset'],x['index'],x['method'],x['view'],x['fraction']):x['recall'] for x in rows}
    means={};contrasts={};boots={}
    for ref in REFS:
        means[ref]={};contrasts[ref]={};boots[ref]={}
        for t in tasks:
            indices=split['tasks'][t][stage]
            dt=np.array([table[t,i,'DT_'+ref,'density',.1] for i in indices]);ft=np.array([table[t,i,'FT_K3','density',.1] for i in indices])
            means[ref][t]={'DT':float(dt.mean()),'FT_K3':float(ft.mean()),'difference':float((dt-ft).mean())}
            diff=np.array([[table[t,i,'DT_'+ref,'density',f]-table[t,i,'FT_K3','density',f] for f in FRACTIONS] for i in indices])
            rng=np.random.default_rng(np.random.SeedSequence(73,spawn_key=(TASKS.index(t),)))
            boot=paired_bootstrap(diff,rng);boots[ref][t]=boot
            contrasts[ref][t]={str(f):interval(diff[:,j],boot[:,j]) for j,f in enumerate(FRACTIONS)}
    out={'status':'verified_'+stage,'case_count':len(r['cases']),'driver_sha256':r['driver_sha256'],
        'analyzer_sha256':digest(Path(__file__)),'results_sha256':digest(a.run/'results.json'),'vectors_sha256':r['vectors_sha256'],
        'split_sha256':r['split_sha256'],'target_inputs_gold_budgets_and_references_verified':True,
        'repeat_control_vectors_bitwise_equal':repeat_checks,'means_at10':means,'paired_contrasts':contrasts,
        'residuals':residuals,'choice':r['choice'],'completed_gpu_operation_seconds':sum(c['seconds'] for c in r['costs'] if c['status']=='returned')}
    if stage=='development':
        ranked=sorted(REFS,key=lambda ref:(-min(v['difference'] for v in means[ref].values()),-np.mean([v['difference'] for v in means[ref].values()]),ref))
        chosen='layer_symmetric'  # The only preregistered candidate; P1 is a control.
        exact={t:all(x['recall']==x['ceiling'] for x in rows if x['dataset']==t and x['method'] in ('DT_'+chosen,'FT_K3') and x['view']=='density' and x['fraction']==.1) for t in tasks}
        eligible=any(v['difference']>0 for v in means[chosen].values()) and all(v['difference']>0 or (v['difference']==0 and exact[t]) for t,v in means[chosen].items())
        out.update(best_candidate=chosen,ranked_candidates=ranked,eligible_for_validation=eligible,joint_exact_ceiling=exact)
        if a.freeze_choice:
            assert eligible,'No reference qualifies; preserve the failure'
            choice={'status':'frozen_for_validation','rule':chosen, 'symmetric_sources':r['weighted_sources'],'split_sha256':r['split_sha256'],
                'development_results_sha256':out['results_sha256'],'development_vectors_sha256':r['vectors_sha256'],
                'development_means':means[chosen]}
            a.output.mkdir(parents=True,exist_ok=True);path=a.output/'choice.json'
            if path.exists():assert json.loads(path.read_bytes())==choice
            else:path.write_text(json.dumps(choice,indent=2)+'\n',encoding='utf-8')
    else:
        chosen=r['choice']['rule'];assert chosen=='layer_symmetric'
        assert r['choice']['symmetric_sources']==r['weighted_sources'];groups={}
        for name,members in [('VT',TASKS[:4]),('HotpotQA',TASKS[4:]),('all_five_tasks',TASKS)]:
            boot=np.mean([boots[chosen][t] for t in members],axis=0)
            at_ceiling=all(x['recall']==x['ceiling'] for x in rows if x['dataset'] in members and x['method'] in ('DT_'+chosen,'FT_K3') and x['view']=='density' and x['fraction']==.1)
            groups[name]={'DT':float(np.mean([means[chosen][t]['DT'] for t in members])),
                'FT_K3':float(np.mean([means[chosen][t]['FT_K3'] for t in members])),'joint_exact_ceiling':at_ceiling,
                'differences':{str(f):{'mean_difference':float(np.mean([contrasts[chosen][t][str(f)]['mean_difference'] for t in members])),
                    'ci95':np.quantile(boot[:,j],[.025,.975]).tolist()} for j,f in enumerate(FRACTIONS)}}
        out['selected_groups']=groups
        primary=[groups[k]['differences']['0.1']['mean_difference'] for k in ('VT','HotpotQA')]
        out['predeclared_aggregate_advantage_criterion_met']=(min(primary)>=0 and max(primary)>0 and
            all(groups[k]['differences']['0.1']['mean_difference']>0 or groups[k]['joint_exact_ceiling'] for k in ('VT','HotpotQA')) and
            groups['all_five_tasks']['differences']['0.1']['ci95'][0]>0)
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'analysis.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    with (a.output/'cases.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps({k:v for k,v in out.items() if k in ('status','case_count','means_at10','best_candidate','eligible_for_validation','selected_groups','predeclared_aggregate_advantage_criterion_met')},indent=2))


if __name__=='__main__':main()

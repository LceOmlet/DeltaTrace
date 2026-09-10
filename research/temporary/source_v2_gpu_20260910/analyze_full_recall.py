"""Verify complete 448-case coverage and fill raw/sentence Recall data tables."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from verify_full_recall_records import verify_records,TASKS,FRACTIONS
from full_recall_statistics import summarize_full_recall

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def saved_metric_rows(records):
    """Read exact previously verified curves whose parent file hash is frozen."""
    rows=[]
    for r in records:
        for m,views in r['metrics'].items():
            for view,curve in views.items():
                for q in curve['points']:
                    rows.append(dict(dataset=r['dataset'],index=r['index'],target_mode=r['target_mode'],method=m,
                        view=view,fraction=q['fraction'],recall=q['recall'],budget=q['budget'],gold=q['gold'],ceiling=q['ceiling'],origin='reserved_validation'))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run','parent','publication','data','tokenizer','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--verification-cache',type=Path)
    p.add_argument('--verify-available',action='store_true',help='Verify completed shards without making a partial quality table.')
    a=p.parse_args();started=time.perf_counter()
    from tokenizers import Tokenizer
    tok=Tokenizer.from_file(str(a.tokenizer))
    plan=json.loads((HERE/'full_recall_plan.json').read_bytes());state=json.loads((a.run/'progress.json').read_bytes())
    identity=json.loads((a.run/'execution_identity.json').read_bytes());choice=json.loads((HERE/'scope_choice.json').read_bytes())
    if a.verify_available:assert state['status'] in ('running','complete')
    else:assert state['status']=='complete' and state['new_completed']==368 and state['control_completed']==5
    assert state['identity']==identity
    assert identity['plan_sha256']==sha(HERE/'full_recall_plan.json') and identity['spec_sha256']==sha(HERE/'FULL_RECALL.md')
    assert identity['driver_sha256']==sha(HERE/'evaluate_full_recall.py') and identity['controller_sha256']==sha(HERE/'run_full_recall.py')
    assert plan['choice_sha256']==sha(HERE/'scope_choice.json')
    for n,h in identity['sources'].items():assert sha(HERE/n)==h
    receipts={x['chunk']:x for x in state['completed_chunks']}
    assert len(receipts)==len(state['completed_chunks']) and set(receipts)<=set(plan['chunks'])
    if not a.verify_available:assert set(receipts)==set(plan['chunks'])
    parent=json.loads((a.parent/'results.json').read_bytes());parent_analysis=json.loads((HERE/'target_scope/analysis.json').read_bytes())
    assert sha(a.parent/'results.json')==plan['parent_results_sha256']==parent_analysis['run_results_sha256']
    assert sha(a.parent/'vectors.npz')==plan['parent_vectors_sha256']==parent_analysis['run_vectors_sha256']
    assert sha(HERE/'target_scope/analysis.json')==plan['parent_analysis_sha256']
    assert parent_analysis['status']=='verified_validation' and parent['status']=='complete' and parent['choice']==choice
    assert parent_analysis['analyzer_sha256']==sha(HERE/'analyze_target_scope.py')
    parent_rows={(x['dataset'],x['index']):x for x in parent['cases']};assert len(parent_rows)==80
    parent_vectors=np.load(a.parent/'vectors.npz')
    caches={};originals={}
    for t in TASKS:
        path=a.data/(t+'.jsonl');assert sha(path)==plan['tasks'][t]['cache_sha256']
        caches[t]=[json.loads(x) for x in path.read_text(encoding='utf-8').splitlines()]
        assert len(caches[t])==plan['tasks'][t]['count']
        original=json.loads(gzip.decompress((a.publication/'raw'/(t+'.results.json.gz')).read_bytes()))
        originals[t]={x['index']:x for x in original['cases']}
    rows=saved_metric_rows(parent['cases']);diagnostics=[];residuals=[];overlaps=[];costs=[]
    coverage={t:set(plan['tasks'][t]['reused_indices']) for t in TASKS}
    origins=[dict(dataset=t,index=i,origin='reserved_validation',input_sha256=parent_rows[t,i]['input_sha256']) for t in TASKS for i in sorted(coverage[t])]
    metadata_files={'protocol_sha256':ROOT/'experiments/official/protocol.json','clean_sources_sha256':ROOT/'deltatrace/clean/sources.json',
        'evaluation_protocol_sha256':ROOT/'experiments/official/source_protocol.json','evidence_protocol_sha256':ROOT/'experiments/official/evidence_protocol.py',
        'recovery_diagnostics_sha256':ROOT/'experiments/official/recovery_diagnostics.py'}
    dependencies=[Path(__file__),HERE/'verify_full_recall_records.py',HERE/'analyze_recall_pilot.py',HERE/'analyze.py',
        ROOT/'experiments/official/evidence_protocol.py',ROOT/'experiments/official/retrieval_views.py']
    verification_identity=dict(code={str(x.relative_to(ROOT)).replace('\\','/'):sha(x) for x in dependencies},
        tokenizer_sha256=sha(a.tokenizer),cache_sha256={t:sha(a.data/(t+'.jsonl')) for t in TASKS},
        publication_sha256={t:sha(a.publication/'raw'/(t+'.results.json.gz')) for t in TASKS})
    verified_shards=[]
    if a.verification_cache:a.verification_cache.mkdir(parents=True,exist_ok=True)
    for name,chunk in plan['chunks'].items():
        if name not in receipts:continue
        folder=a.run/name;r=json.loads((folder/'results.json').read_bytes());receipt=receipts[name]
        assert r['status']=='complete' and r['stage']=='full' and r['experiment']=='target-full-v1'
        assert sha(folder/'results.json')==receipt['results_sha256']
        assert sha(folder/'vectors.npz')==receipt['vectors_sha256']==r['vectors_sha256']
        assert r['driver_sha256']==identity['driver_sha256'] and r['full_plan_sha256']==identity['plan_sha256']
        assert r['chunk']==name and r['chunk_spec']==chunk and r['choice']==choice
        assert r['full_spec_sha256']==plan['spec_sha256'] and r['parent_results_sha256']==plan['parent_results_sha256']
        assert r['parent_vectors_sha256']==plan['parent_vectors_sha256']
        assert r['weighted_sources']==choice['weighted_sources']
        assert r['weight_identity']==parent['weight_identity']
        for n,path in metadata_files.items():assert r[n]==sha(path)
        assert r['FT_initial_target_adapter']['sha256']==choice['FT_initial_target_adapter_sha256']
        assert r['selected_counts']=={chunk['dataset']:len(chunk['indices'])}
        assert len(r['cases'])==len(chunk['indices']) and {x['index'] for x in r['cases']}==set(chunk['indices'])
        assert all(x['dataset']==chunk['dataset'] and x['target_mode']==choice['targets'][x['dataset']] for x in r['cases'])
        vectors=np.load(folder/'vectors.npz')
        key=dict(identity=verification_identity,results_sha256=receipt['results_sha256'],vectors_sha256=receipt['vectors_sha256'])
        cached=a.verification_cache/(name+'.json') if a.verification_cache else None
        if cached and cached.exists():
            saved=json.loads(cached.read_bytes())
            assert saved['key']==key and saved['status']=='independently_verified_vectors'
            payload=saved['payload']
            assert hashlib.sha256(json.dumps(payload,sort_keys=True,allow_nan=False).encode()).hexdigest()==saved['payload_sha256']
            checked,diag,res,_=payload
            cache_hit=True
        else:
            checked,diag,res,count=verify_records(r['cases'],vectors,caches,originals,tok)
            cache_hit=False
            if cached:
                payload=[checked,diag,res,count]
                saved=dict(status='independently_verified_vectors',key=key,payload=payload,
                    payload_sha256=hashlib.sha256(json.dumps(payload,sort_keys=True,allow_nan=False).encode()).hexdigest())
                cached.write_text(json.dumps(saved,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        verified_shards.append(dict(chunk=name,results_sha256=key['results_sha256'],vectors_sha256=key['vectors_sha256'],
            verification_receipt_sha256=sha(cached) if cached else None))
        for row in r['cases']:
            t,i=row['dataset'],row['index']
            if i in chunk['control_indices']:
                old=parent_rows[t,i];assert row['overlap_control_bitwise_equal']
                for key in ('input_ids','input_sha256','target','target_mode','target_weights','keep','gold','references'):assert row[key]==old[key]
                prefix=f"{t}_{i}_{row['target_mode']}_"
                for suffix in ('DT_target_signed_full','FT_K1_prompt','FT_K3_prompt'):
                    assert np.array_equal(vectors[prefix+suffix],parent_vectors[prefix+suffix])
                overlaps.append(dict(dataset=t,index=i,bitwise_equal_vectors=3))
            else:
                assert i in chunk['new_indices'] and i not in coverage[t];coverage[t].add(i)
                origins.append(dict(dataset=t,index=i,origin=name,input_sha256=row['input_sha256']))
        for x in checked:
            if x['index'] in chunk['new_indices']:rows.append(dict(x,origin=name))
        diagnostics.extend(x for x in diag if x['index'] in chunk['new_indices'])
        residuals.extend(x for x in res if x['index'] in chunk['new_indices'])
        costs.extend(dict(x,chunk=name) for x in r['costs'])
        print(json.dumps(dict(verified_chunk=name,unique_cases=sum(map(len,coverage.values())),cached=cache_hit,seconds=time.perf_counter()-started)),flush=True)
        vectors.close()
    if a.verify_available:
        print(json.dumps(dict(status='available_shards_verified_no_quality_aggregation',chunks=len(receipts),unique_cases=len(origins))))
        return
    assert len(overlaps)==5 and all(coverage[t]==set(range(plan['tasks'][t]['count'])) for t in TASKS)
    assert len(origins)==448 and len({(x['dataset'],x['index']) for x in origins})==448
    statistics=summarize_full_recall(rows,{t:sorted(coverage[t]) for t in TASKS})
    out=dict(status='verified_complete_benchmark',case_count=448,new_cases=368,reused_cases=80,overlap_controls=overlaps,
        full_plan_sha256=identity['plan_sha256'],choice_sha256=sha(HERE/'scope_choice.json'),analyzer_sha256=sha(Path(__file__)),
        case_verifier_sha256=sha(HERE/'verify_full_recall_records.py'),statistics_sha256=sha(HERE/'full_recall_statistics.py'),
        experiment='fixed_target_full_recall_v1',new_independent_holdout=False,
        interpretation='Complete benchmark including development and prior validation; intervals are descriptive.',
        parent_results_sha256=plan['parent_results_sha256'],parent_analysis_sha256=plan['parent_analysis_sha256'],
        all_native_input_target_gold_reference_and_budget_checks_passed=True,
        verification_identity=verification_identity,verified_shards=verified_shards,
        additional_gpu_operation_seconds=sum(x['seconds'] for x in costs if x['status']=='returned'),
        reused_gpu_operation_seconds=parent_analysis['completed_gpu_operation_seconds'],
        costs=costs,case_origins=origins,residuals=residuals,assignment_diagnostics=diagnostics,**statistics)
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'analysis.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    for filename,data in [('per_case.csv',rows),('case_origins.csv',origins)]:
        with (a.output/filename).open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    print(json.dumps(dict(status=out['status'],case_count=448,primary_comparisons=out['primary_comparisons']),indent=2))


if __name__=='__main__':main()

"""Use the original collection analyzer, with frozen robust-tail identities."""
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'conditional-attention-owner-v1'))
from analyze_collection import analyze,ratio_bin


def main():
    folder=HERE/'conditional-gdn-collection-v1'
    inputs=folder/'comparison-inputs-textcraft.json'
    paths=[folder/('textcraft-rank'+str(i)+'.json') for i in range(2)]
    records=[json.loads(p.read_bytes()) for p in paths]
    assert all(r['phase']=='complete' for r in records)
    result=analyze(json.loads(inputs.read_bytes()),records)
    result['evidence']=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                        for p in [inputs,*paths]]
    result['source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    previous=HERE/'single-background-textcraft-analysis.json'
    stable=json.loads(previous.read_bytes())
    known={(p['traj_uid'],p['packed_slot']):p for p in stable['points']}
    measured={}
    for record in records:
        for batch in record['batches']:
            for row in batch['trajectories']:
                for q in row['single_deletions']:
                    key=(row['traj_uid'],q['packed_slot'])
                    assert key not in measured or measured[key]==q
                    measured[key]=q
    assert set(known)==set(measured)
    spurious=[];missed=[]
    for key,old in known.items():
        point=measured[key]
        d=point['fresh_conditional_gdn_d']
        values=dict(traj_uid=key[0],packed_slot=key[1],token_id=old['token_id'],
            initial_state_sha256=old['initial_state_sha256'],
            previously_examined=old['previously_examined'],native_interval=old['native_interval'],
            original_d=point['fresh_original_d'],candidate_d=d,
            candidate_A_over_r=-math.expm1(-d),
            remains_negative_tail=d < -math.log(2),remains_negative_sign=d < 0)
        if old['original_spurious_tail']:spurious.append(values)
        if old['original_missed_tail'] and 'uniform' in old['cohorts']:missed.append(values)
    result['robust_diagnostic_points']=dict(
        definition='Reuse the previously frozen native precision/layout intervals and original-spurious flags, not reclassifying the data from candidate values.',
        source=dict(path=str(previous),sha256=hashlib.sha256(previous.read_bytes()).hexdigest()),
        original_spurious_tail_points=len(spurious),
        original_spurious_tail_states=len({q['initial_state_sha256'] for q in spurious}),
        still_spurious_tail=sum(q['remains_negative_tail'] for q in spurious),
        still_opposite_sign=sum(q['remains_negative_sign'] for q in spurious),
        original_uniform_missed_tail_points=len(missed),
        missed_tail_still_outside_tail=sum(not q['remains_negative_tail'] for q in missed),
        spurious_points=spurious,missed_points=missed)
    times=[]
    for record in records:
        baseline=json.loads(Path(record['original_baseline']['path']).read_bytes()) if Path(record['original_baseline']['path']).is_file() else None
        # Remote baseline paths are mapped to the exact local SHA-verified
        # files already recorded in the comparison input, without rereading GPU data.
        if baseline is None:
            p=HERE/'conditional-attention-owner-v1'/Path(record['original_baseline']['path']).name
            assert hashlib.sha256(p.read_bytes()).hexdigest()==record['original_baseline']['sha256']
            baseline=json.loads(p.read_bytes())
        by_index={b['index']:b for b in baseline['batches']}
        for batch in record['batches']:
            before=by_index[batch['index']]['variants']['original']['detail']['complete_attribution_seconds_with_diagnostics']
            after=batch['variants']['conditional_gdn']['detail']['complete_attribution_seconds_with_diagnostics']
            times.append(dict(batch=batch['index'],width=batch['width'],original_seconds=before,
                candidate_seconds=after,ratio=after/before))
    result['DT_cost']=dict(scope='Same saved B4 identities and original diagnostic timing boundaries; separate runs, first-use costs retained. Not native-only kernel or official training throughput.',
        total_ratio=sum(q['candidate_seconds'] for q in times)/sum(q['original_seconds'] for q in times),
        median_batch_ratio=statistics.median(q['ratio'] for q in times),batches=times)
    factual=[row['reused_baseline_factual_difference'] for record in records for batch in record['batches']
             if batch['primary'] for row in batch['trajectories']]
    result['paired_factual_scores']=dict(points=len(factual),nonzero=sum(v!=0 for v in factual),
        max_abs_difference=max(abs(v) for v in factual),
        interpretation='Same original model/precision/Y facts; no official whole-model tolerance is invented.')
    bounds=[]
    for key,q in measured.items():
        lf=q['fresh_factual_target_logp']
        bounds.append(dict(traj_uid=key[0],packed_slot=key[1],token_id=known[key]['token_id'],
            initial_state_sha256=known[key]['initial_state_sha256'],cohorts=q['cohorts'],
            previously_examined=known[key]['previously_examined'],
            original_ratio_bin=ratio_bin(q['fresh_original_d']),
            native_ratio_bin=ratio_bin(q['fresh_native_single_d']),
            factual_logp=lf,
            original_implied_deleted_logp=lf-q['fresh_original_d'],
            candidate_implied_deleted_logp=lf-q['fresh_conditional_gdn_d'],
            native_deleted_logp=lf-q['fresh_native_single_d']))
    result['probability_bound_diagnostic']=dict(
        meaning='If d is interpreted as a deletion log-probability difference, implied log p_deleted=log p_factual-d must be <=0. A necessary probability property, not an invented numeric tolerance or a clipping repair.',
        unique_points=len(bounds),
        sampling_scope='Union counts describe saved identities, not a population rate. Uniform and tail census retain separate denominators and prediction/native cells below.',
        counts={name:sum(q[name]>0 for q in bounds) for name in
            ('original_implied_deleted_logp','candidate_implied_deleted_logp','native_deleted_logp')},
        largest_values={name:max(q[name] for q in bounds) for name in
            ('original_implied_deleted_logp','candidate_implied_deleted_logp','native_deleted_logp')},
        points=bounds,no_credit_modified=True)
    bound_cells={}
    for q in bounds:
        for cohort in q['cohorts']:
            key=(cohort,q['original_ratio_bin'],q['native_ratio_bin'],q['previously_examined'])
            bound_cells.setdefault(key,[]).append(q)
    result['probability_bound_diagnostic']['cells']=[dict(cohort=key[0],original_ratio_bin=key[1],
        native_ratio_bin=key[2],previously_examined=key[3],points=len(values),
        initial_states=len({q['initial_state_sha256'] for q in values}),
        counts={name:sum(q[name]>0 for q in values) for name in
            ('original_implied_deleted_logp','candidate_implied_deleted_logp','native_deleted_logp')})
        for key,values in bound_cells.items()]
    previous_FA={}
    FA_sources=[]
    for rank in range(2):
        p=HERE/'conditional-attention-owner-v1'/('textcraft-v3-rank'+str(rank)+'.json')
        FA_sources.append(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
        for batch in json.loads(p.read_bytes())['batches']:
            for row in batch['trajectories']:
                for q in row['single_deletions']:
                    previous_FA[row['traj_uid'],q['packed_slot']]=q
    overlap=[]
    for q in spurious:
        old=previous_FA[q['traj_uid'],q['packed_slot']]
        assert old['fresh_original_d']==q['original_d']
        overlap.append(dict(traj_uid=q['traj_uid'],packed_slot=q['packed_slot'],
            initial_state_sha256=q['initial_state_sha256'],GDN_d=q['candidate_d'],FA_d=old['fresh_conditional_d']))
    result['existing_candidate_overlap']=dict(sources=FA_sources,original_robust_spurious_points=len(overlap),
        FA_resolved_tail=sum(q['FA_d']>=-math.log(2) for q in overlap),
        GDN_resolved_tail=sum(q['GDN_d']>=-math.log(2) for q in overlap),
        both_resolved_tail=sum(q['FA_d']>=-math.log(2) and q['GDN_d']>=-math.log(2) for q in overlap),
        either_resolved_tail=sum(q['FA_d']>=-math.log(2) or q['GDN_d']>=-math.log(2) for q in overlap),
        points=overlap,
        limitation='Descriptive paired overlap only; not a mixed estimator, per-token selection, prediction of a combined result or authorization to deploy either rejected candidate.')
    result['conclusion']='Not accepted as an extreme-negative-credit repair: 12/18 robust spurious tails remain, both uniform misses remain outside the predicted tail, signed RISE point estimate worsens, and DT cost is about twice the baseline. MAS improvement alone does not establish the required repair. No AppWorld or held-out candidate run is launched from this result.'
    result['candidate_accepted']=False
    result['production_modified']=False
    (folder/'textcraft-analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(complete=result['complete'],metrics={k:{n:v for n,v in m.items() if n in
        ('paired_finite','available_equal_state_means','state_delta_standard_error')} for k,m in result['primary_metrics'].items()},
        tails={k:v for k,v in result['robust_diagnostic_points'].items() if k not in
            ('spurious_points','missed_points','source','definition')},
        DT_cost={k:v for k,v in result['DT_cost'].items() if k not in ('batches',)})))


if __name__=='__main__':main()

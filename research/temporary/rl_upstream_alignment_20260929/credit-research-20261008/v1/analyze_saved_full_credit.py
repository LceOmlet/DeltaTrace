"""Read saved complete vectors on CPU; no model/DT/native query or correction.

Probability-bound diagnostics are necessary conditions, separate from the
original author metrics and frozen native single-deletion comparisons.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records',nargs='+',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--include-endpoint-ledger',action='store_true')
    args=parser.parse_args()
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(2)
    sources=[];rows={}
    for name in args.records:
        path=Path(name);record=json.loads(path.read_bytes())
        assert record['phase']=='complete'
        sources.append(dict(path=str(path),sha256=sha(path),source_sha256=record['source_sha256']))
        for batch in record['batches']:
            for label,item in batch['variants'].items():
                artifact=Path(item['artifact'])
                assert sha(artifact)==item['sha256']
                saved=torch.load(artifact,map_location='cpu',weights_only=False)
                for i in range(batch['actual_rows']):
                    uid=saved['uids'][i]
                    assert uid==batch['uids'][i]
                    position=torch.tensor(saved['positions'][i],dtype=torch.long)
                    d=saved['signed'][i,position].double()
                    assert d.numel() and torch.isfinite(d).all()
                    selected=saved['selected'][i]
                    factual=saved['detail']['per_sample'][i]['factual_target_logp']
                    implied=factual-d
                    index=int(d.argmin())
                    q=next(q for q in batch['trajectories'] if q['traj_uid']==uid)
                    values=dict(traj_uid=uid,state=q['initial_state_sha256'],label=label,
                        primary=batch['primary'],previously_examined=q['previously_examined'],
                        source_tokens=d.numel(),factual_logp=factual,
                        probability_bound_violations=int((implied>0).sum()),
                        probability_bound_violation_fraction=float((implied>0).double().mean()),
                        log_ratio_bins={name:int(mask.sum()) for name,mask in (
                            ('ratio_le_1',d>=0),('ratio_1_to_2',(d<0)&(d>=-math.log(2))),
                            ('ratio_2_to_10',(d<-math.log(2))&(d>=-math.log(10))),
                            ('ratio_10_to_100',(d<-math.log(10))&(d>=-math.log(100))),
                            ('ratio_above_100',d<-math.log(100)))},
                        minimum=dict(d=float(d[index]),packed_slot=int(position[index]),
                            token_id=int(selected[position[index]]),
                            implied_deleted_logp=float(implied[index])),
                        original_vector=dict(path=str(artifact),sha256=item['sha256']))
                    if args.include_endpoint_ledger:
                        detail=saved['detail']['per_sample'][i]
                        reference=detail['reference_target_logp']
                        delta=factual-reference
                        source_sum=float(d.sum())
                        values['endpoint_ledger']=dict(
                            factual_target_logp=factual,reference_target_logp=reference,
                            joint_target_difference=delta,
                            owner_root_effect=detail['root_effect'],
                            recomputed_source_sum_FP64=source_sum,
                            full_signed_row_sum_FP64=float(saved['signed'][i].double().sum()),
                            owner_policy_credit_signed_sum=detail['policy_credit_signed_sum'],
                            source_sum_minus_joint_difference=source_sum-delta,
                            positive_source_count=int((d>0).sum()),
                            negative_source_count=int((d<0).sum()),
                            positive_source_sum=float(d[d>0].sum()),
                            negative_source_sum=float(d[d<0].sum()))
                    key=(label,uid)
                    assert key not in rows,'Do not silently double-weight duplicate trajectories'
                    rows[key]=values
    groups={}
    for label in sorted({k[0] for k in rows}):
        for primary in (True,False):
            chosen=[r for (variant,_),r in rows.items() if variant==label and r['primary']==primary]
            states={}
            for row in chosen:states.setdefault(row['state'],[]).append(row)
            groups[label+('_primary' if primary else '_extra_tail')]=dict(
                trajectories=len(chosen),states=len(states),
                source_tokens=sum(q['source_tokens'] for q in chosen),
                count_violations=sum(q['probability_bound_violations'] for q in chosen),
                state_equal_violation_fraction=math.fsum(
                    math.fsum(q['probability_bound_violation_fraction'] for q in state)/len(state)
                    for state in states.values())/len(states) if states else None,
                largest_negative=sorted(chosen,key=lambda q:q['minimum']['d'])[:5])
    result=dict(scope=__doc__,sources=sources,script_sha256=sha(__file__),
        necessary_bound='log p_deleted = factual_logp - d <= 0; descriptive only, no clipping or invented numeric tolerance.',
        aggregation='Primary frozen trajectories and extra tail trajectories remain separate. Report counts and state-equal bounded violation frequencies; no pooled raw advantage/exp moments.',
        groups=groups,rows=list(rows.values()),operations=dict(model_forward=0,DT=0,optimizer=0),
        all_CPU=True,production_modified=False)
    if args.include_endpoint_ledger:
        result['endpoint_ledger_interpretation']=(
            'The joint endpoint difference and the sum of individual deletion effects are different mathematical quantities. '
            'This ledger describes how the existing shared finite endpoint identity changes under the already-rejected '
            'conditional candidates; it is not an added conservation acceptance test, an official tolerance, a cause share '
            'or a correction. No coefficient is rescaled. Trajectory sums are finite recorded quantities, not population '
            'moments of an unbounded tail. Probability-bound, native deletion and author quality diagnostics remain separate.')
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:{n:v for n,v in g.items() if n!='largest_negative'} for k,g in groups.items()}))


if __name__=='__main__':main()

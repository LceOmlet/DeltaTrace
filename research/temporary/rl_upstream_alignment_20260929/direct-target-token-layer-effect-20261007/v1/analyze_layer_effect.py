"""Describe recorded owner boundaries; no new numerical tolerance or credit."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch

HERE=Path(__file__).resolve().parent


def artifact(path):
    path=Path(path).resolve()
    return dict(path=path.as_posix(),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,default=HERE/'actual-results');parser.add_argument('--output',type=Path,default=HERE/'analysis.json');parser.add_argument('--csv',type=Path,default=HERE/'boundary-effects.csv');prior=parser.add_mutually_exclusive_group();prior.add_argument('--skip-previous',action='store_true');prior.add_argument('--previous-joint',type=Path);args=parser.parse_args()
    root=args.input;transport=json.loads((root/'transport.json').read_bytes())
    for item in transport:assert artifact(item['local_path'])['sha256']==item['sha256']
    ranks=[json.loads((root/f'results/rank{r}.json').read_bytes()) for r in (0,1)]
    out=dict(scope='Original two DT calls on one real B4; original joint coefficients contracted with saved original single-deletion hidden endpoints. Diagnostic only, not a new estimator or numerical acceptance criterion.',ranks=[],sources=[artifact(root/'transport.json')])
    csv_rows=[]
    for rank in ranks:
        assert rank['phase']=='complete' and len(rank['cross_boundary_contractions'])==33
        boundaries=[]
        for layer in range(32,-1,-1):
            row=dict(rank=rank['rank'],**rank['cross_boundary_contractions'][str(layer)])
            if boundaries:
                row['change_from_output_boundary']=row['joint_coefficient_times_single_deletion_delta']-boundaries[-1]['joint_coefficient_times_single_deletion_delta']
            else:row['change_from_output_boundary']=None
            boundaries.append(row)
            csv_rows.append({k:v for k,v in row.items() if k not in ('shape','native_joint_pair_differences')})
        candidate=rank['geometry']['candidate'];candidate_row=candidate['row'];drift=[]
        for mode in ('single_EOS','original_joint_EOS'):
            for original_row in (i for i in range(4) if i!=candidate_row):
                records=[item for item in rank['native_root'] if item['mode']==mode]
                first=next((dict(layer=item['layer'],kind=item['kind'],**item['comparison']['pairs'][original_row]) for item in records if not item['comparison']['pairs'][original_row]['equal']),None)
                initial=next((item for item in records if item['layer']==0 and item['kind']=='input'),None)
                cache_comparison=None if initial is None else {name:value['pairs'][original_row] for name,value in initial['cache'].items()}
                drift.append(dict(mode=mode,original_row=original_row,first_nonidentical_native_boundary=first,cache_before_layer0=cache_comparison,
                    capture_available=bool(records),capture_limitation=None if records else 'The original root calls model.__call__, not the observed forward_root prefix method. No native-root/cache hook data were obtained; no zero-difference claim or rerun.'))
        single=rank['phases']['single_EOS'];joint=rank['phases']['original_joint_EOS']
        negative=next((b for b in boundaries if b['joint_coefficient_times_single_deletion_delta']<0),None)
        signed=torch.load(root/f'results/rank{rank["rank"]}-original_joint_EOS-signed.pt',map_location='cpu',weights_only=False)
        d=float(signed[candidate['row'],candidate['packed_slot']])
        out['ranks'].append(dict(rank=rank['rank'],pid=rank['pid'],birth=rank['birth'],finite_owner=rank['finite_owner'],
            candidate=candidate,single_source_signed=single['candidate_signed'],single_source_focused_root=single['detail']['per_sample'][candidate_row]['root_effect'],
            single_source_identity_root=[p['root_effect'] for i,p in enumerate(single['detail']['per_sample']) if i!=candidate_row],
            original_joint_candidate_signed=joint['candidate_signed'],original_saved_joint_candidate=candidate['saved_native_signed'],
            input_boundary_contraction=boundaries[-1]['joint_coefficient_times_single_deletion_delta'],
            input_contraction_minus_joint_candidate=boundaries[-1]['joint_coefficient_times_single_deletion_delta']-d,
            first_negative_during_reverse_traversal=negative,
            largest_boundary_changes=sorted(boundaries[1:],key=lambda x:abs(x['change_from_output_boundary']),reverse=True)[:5],
            factual_endpoint_all_equal=all(b['factual_endpoints_equal'] for b in boundaries),
            factual_endpoint_maxabs=max(b['factual_endpoint_maxabs'] for b in boundaries),
            native_identity_differences=drift,boundaries=boundaries,
            seconds={mode:rank['phases'][mode]['seconds'] for mode in rank['phases']},
            all_single_endpoint_snapshots_released=joint['remaining_snapshot_bytes']==0,
            sources=[artifact(root/f'results/rank{rank["rank"]}.json'),artifact(root/f'results/rank{rank["rank"]}-original_joint_EOS-signed.pt')]))
    out['controls']=dict(ranks_same_cross_contractions=[b['joint_coefficient_times_single_deletion_delta'] for b in out['ranks'][0]['boundaries']]==[b['joint_coefficient_times_single_deletion_delta'] for b in out['ranks'][1]['boundaries']],
        ranks_same_joint_signed=torch.equal(torch.load(root/'results/rank0-original_joint_EOS-signed.pt',map_location='cpu',weights_only=False),torch.load(root/'results/rank1-original_joint_EOS-signed.pt',map_location='cpu',weights_only=False)),
        source_sha256=ranks[0]['geometry']['source_sha256'],native_sha256=ranks[0]['geometry']['native_sha256'],
        no_observer_mode=True,no_model_parameter_update=True,no_credit_mutation=True)
    previous=HERE.parents[1]/'direct-target-native-mlp-memory-20261007/v2/actual-results-v6/results'
    unchanged=[]
    for rank in (() if args.skip_previous or args.previous_joint is not None else (0,1)):
        for mode,old_name in [('single_EOS','single-source-eos'),('original_joint_EOS','original_gpu_captures')]:
            old_path=previous/f'rank{rank}-{old_name}.pt'
            old=torch.load(old_path,map_location='cpu',weights_only=False)['signed']
            new_path=root/f'results/rank{rank}-{mode}-signed.pt'
            new=torch.load(new_path,map_location='cpu',weights_only=False)
            unchanged.append(dict(rank=rank,mode=mode,whole_signed_equal=torch.equal(old,new),maxabs=float((new-old).abs().max()),previous=artifact(old_path),current=artifact(new_path)))
    if args.previous_joint is not None:
        prior=torch.load(args.previous_joint,map_location='cpu',weights_only=False)
        assert prior['source_sha256']==out['controls']['source_sha256'] and prior['native_sha256']==out['controls']['native_sha256']
        for rank in (0,1):
            new_path=root/f'results/rank{rank}-original_joint_EOS-signed.pt'
            new=torch.load(new_path,map_location='cpu',weights_only=False)
            unchanged.append(dict(rank=rank,mode='original_joint_EOS',whole_signed_equal=torch.equal(prior['signed'],new),maxabs=float((new-prior['signed']).abs().max()),previous=artifact(args.previous_joint),current=artifact(new_path)))
    out['unchanged_original_outputs']=unchanged
    if args.skip_previous:out['previous_output_comparison_scope']='Not performed: the historical baseline is a different B4. No claim that current independent instrumented/uninstrumented replays are bitwise identical.'
    out['resource_scope']=[]
    for rank in (0,1):
        records=[json.loads(line) for line in (root/f'results/rank{rank}-phases.jsonl').read_text().splitlines()]
        out['resource_scope'].append(dict(rank=rank,max_recorded_pss_bytes=max(v['pss_bytes'] for v in records),
            max_single_endpoint_snapshot_bytes=max(v.get('remaining_single_endpoint_bytes',0) for v in records),
            final_snapshot_bytes=records[-1]['remaining_single_endpoint_bytes'],
            note='Sampled worker PSS and explicitly retained CPU endpoints, not container memory or physical VRAM peak.'))
    out['interpretation_scope']='A changed diagnostic contraction localizes a boundary where the joint-reference coefficients act differently on a real single-deletion hidden difference. It does not prove the finite operator is numerically wrong, prove any one layer causes all error, or authorize a correction. Factual endpoint differences and native cached identity-row differences are retained separately, not absorbed into a tolerance or multiplier.'
    args.output.write_text(json.dumps(out,indent=2)+'\n')
    with args.csv.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
    assert not torch.cuda.is_initialized()
    print(json.dumps(dict(controls=out['controls'],ranks=[{k:v[k] for k in ('rank','single_source_signed','single_source_focused_root','original_joint_candidate_signed','input_boundary_contraction','input_contraction_minus_joint_candidate','first_negative_during_reverse_traversal','factual_endpoint_all_equal','factual_endpoint_maxabs','seconds')} for v in out['ranks']])) )


if __name__=='__main__':main()

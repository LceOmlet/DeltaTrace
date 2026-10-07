"""Describe original author outputs; no copied sorting or RISE/MAS formula."""
import csv
import hashlib
import json
from pathlib import Path
import re
import argparse

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]


def receipt(path):
    return dict(path=path.as_posix(), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--memory',action='store_true');args=parser.parse_args()
    folder = HERE / ('memory-curve-results' if args.memory else 'curve-results')
    transport = json.loads((folder / 'transport.json').read_bytes())
    for item in transport:
        assert receipt(Path(item['local_path']))['sha256'] == item['sha256']
    ranks = [json.loads((folder / f'results/rank{rank}.json').read_bytes()) for rank in (0, 1)]
    inputs = json.loads((folder / 'curve-inputs.json').read_bytes())
    results = {}
    csv_rows = []
    all_points = []
    for rank, report in enumerate(ranks):
        assert report['phase'] == 'complete' and report['native_forward_calls'] == 42
        binding = inputs['ranks'][str(rank)]
        assert binding == report['profile_comparison_binding']
        assert report['author']['sha256'] == '583f4b7d0426407eb9a517f173365762860a1f4382f472dffb5c07de7d3e94a1'
        assert report['author']['k_default'] == 20
        mode = binding['profile']
        path = (HERE/'memory-results' if args.memory else HERE) / 'results' / f'rank0-{mode}.pt'
        assert receipt(path)['sha256'] == binding['provenance']['attribution_sha256']
        trace = torch.load(path, map_location='cpu', weights_only=False)
        candidate = binding['case']
        signed = trace['signed'][candidate['row']].double()
        views = {}
        for name, view in report['views'].items():
            points = view['score_points']
            assert len(points) == 21
            all_points.extend(points)
            previous = set()
            groups = []
            for index, point in enumerate(points):
                changed = set(point['changed_input_positions'])
                assert previous <= changed
                added = sorted(changed - previous)
                values = signed[added]
                group = dict(step=index, changed_positions=len(changed), newly_changed_positions=len(added),
                    original_signed_sum_of_newly_changed=float(values.sum()),
                    original_negative_count=int((values < 0).sum()),
                    original_positive_count=int((values > 0).sum()),
                    original_zero_count=int((values == 0).sum()),
                    native_logp=point['logp'],
                    native_kept_minus_deleted_for_step=None if index == 0 else points[index - 1]['logp'] - point['logp'],
                    candidate_changed=candidate['packed_slot'] in changed, seconds=point['seconds'])
                groups.append(group)
                csv_rows.append(dict(profile=mode, view=name, **group))
                previous = changed
            views[name] = dict(author_return=dict(zip(view['author_fields'], view['author_return'])),
                author_arrays=view['author_arrays'], groups=groups,
                candidate_first_changed_step=next(g['step'] for g in groups if g['candidate_changed']),
                native_forward_seconds=sum(p['seconds'] for p in points),
                raw_score_increases_on_deletion=[dict(from_step=i-1,to_step=i,logp_increase=points[i]['logp']-points[i-1]['logp']) for i in range(1,len(points)) if points[i]['logp'] > points[i-1]['logp']],
                note='Observed original deletion groups, not reconstructed sorting. Effects depend on their cumulative context; they are not individual-token ground truth.')
        results[mode] = dict(rank=rank, candidate=candidate, source_count=report['source_count'], views=views,
            author=report['author'], original_diagnostic=report['original_curve_diagnostic'],
            owners=report['owners'], peak_observed_pss_bytes=max(p['pss_bytes'] for v in report['views'].values() for p in v['score_points']))
    control = dict(
        same_original_source=all(r['source_sha256']==ranks[0]['source_sha256'] for r in ranks),
        same_source_positions=ranks[0]['source_positions']==ranks[1]['source_positions'],
        endpoint_scores_by_profile={mode:{name:[v['groups'][0]['native_logp'],v['groups'][-1]['native_logp']] for name,v in value['views'].items()} for mode,value in results.items()},
        maximum_twin_score_difference=max(abs(x) for p in all_points for x in p['twin_row_score_differences']),
        other_three_rows_scores_constant=len({tuple(p['other_three_joint_logp']) for p in all_points}) == 1,
        local_LoRA_B_shards_zero=all(not x['nonzero'] for r in ranks for x in r['lora_B_local_shards']))
    control['all_views_share_factual_score'] = len({v['groups'][0]['native_logp'] for r in results.values() for v in r['views'].values()}) == 1
    control['all_views_share_all_EOS_score'] = len({v['groups'][-1]['native_logp'] for r in results.values() for v in r['views'].values()}) == 1
    physical = []
    for line in (folder / 'curve-physical-mx-smi.jsonl').read_bytes().splitlines():
        event = json.loads(line)
        gpu = None
        for text in event['stdout'].splitlines():
            board = re.match(r'^\|\s*(\d+)\s+MetaX\s',text)
            if board: gpu = int(board[1])
            memory = re.search(r'(\d+)/(\d+) MiB',text)
            if memory and gpu in (4,5): physical.append(dict(gpu=gpu,used_mib=int(memory[1])))
    launch = json.loads((folder / 'launch.json').read_bytes())
    out = dict(status='Original author curve comparison measured; not a production rule change or credit repair',
        code_commit=launch['code_commit'],source_sha256=ranks[0]['source_sha256'],
        original_native_sha256=launch['native_sha256'],
        scope='Same real AppWorld trajectory and native B4 scorer; one previously measured existing attention-PV rule per DP rank. Original author cumulative deletion/RISE/MAS with k20 unchanged.',
        profiles=results,controls=control,
        measurements=dict(native_forwards_per_rank=42,DT=0,backward=0,optimizer=0,rollout=0,
            elapsed_driver_launch_to_last_worker_complete_seconds=max(r['unix'] for r in ranks)-launch['launched_unix'],
            physical_sampled_peak_mib={str(gpu):max(x['used_mib'] for x in physical if x['gpu']==gpu) for gpu in (4,5)}),
        interpretation='The existing content0 rule fixes the selected newline sign but does not improve this trajectory\'s original cumulative deletion/RISE/MAS results. Both rules assign a negative sum to their final signed-ranked deletion group, while preserving that group increases the native joint target score in that cumulative context. This documents an interaction-allocation mismatch, not merely a large coefficient or proof that all DT rankings are poor. No production rule switch is justified by these results.',
        sources=[receipt(folder/'transport.json'),receipt(folder/'results/effective-config.yaml'),receipt(folder/'curve-inputs.json'),receipt(REPO/('experiments/rl/results_existing_memory_rule_20261008.json' if args.memory else 'experiments/rl/results_existing_PV_rule_20261008.json'))],
        limitations=[
            'One selected high-impact trajectory, not a paper-scale/global quality result or a population error rate.',
            'The original metric uses its normalization, running minimum and evaluation penalties. Raw nonmonotonic native scores are preserved; none of these evaluation transforms change training credit.',
            'The joint/scattered action target is supplied by the existing literal-ID/scalar-score adapter, not the pristine single-suffix evaluation input. Its original scorer and target selection are called unchanged.',
            'Positive-only MAS input is evaluation only, never credit clipping or a production rule change.',
            'Control scores, conservation or these attribution metrics are not FA/FLA numerical tolerance criteria. No new pass/fail threshold is invented.',
            'Native single-token deletion and original cumulative curves answer different questions; neither replaces the other.',
        ],
        formal_restart=False, production_profile_changed=False,credit_repaired=False,
        TextCraft_first_update_released=False,checkpoint_restore=False,
        local_cuda_initialized=torch.cuda.is_initialized())
    if args.memory:
        out['scope']='Same real AppWorld trajectory and native B4 scorer; one previously measured existing averaged/forward memory rule per DP rank. Original author cumulative deletion/RISE/MAS with k20 unchanged.'
        out['interpretation']='Original author metrics and native cumulative deletion groups for the existing memory rules are reported without fitting a threshold. This selected trajectory is supplemental to the complete B4 vector and separately measured individual-token endpoints; it does not establish population quality or repaired training.'
    target = REPO/('experiments/rl/results_existing_memory_author_curves_20261008.json' if args.memory else 'experiments/rl/results_existing_PV_author_curves_20261008.json')
    target.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    (HERE/('memory-curve-analysis.json' if args.memory else 'curve-analysis.json')).write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    with (HERE/('memory-rule-curves.csv' if args.memory else 'rule-curves.csv')).open('w',newline='',encoding='utf8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
    print(json.dumps(dict(receipt=receipt(target),controls=control,measurements=out['measurements'],
        metrics={mode:{name:dict(author_return=v['author_return'],candidate_first_changed_step=v['candidate_first_changed_step']) for name,v in p['views'].items()} for mode,p in results.items()}),indent=2))


if __name__ == '__main__':
    main()

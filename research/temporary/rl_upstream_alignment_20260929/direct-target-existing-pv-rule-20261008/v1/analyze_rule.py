"""Describe the existing-rule comparison without fitting an acceptance rule."""
from pathlib import Path
import hashlib
import json
import math
import re
import argparse

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
MODES = ('original_content1', 'existing_content0')


def binding(p):
    return dict(path=p.resolve().as_posix(), bytes=p.stat().st_size,
                sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def main():
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group()
    group.add_argument('--memory',action='store_true');group.add_argument('--clean-gdn',action='store_true');args=parser.parse_args()
    memory_rules=args.memory or args.clean_gdn
    folder=HERE/('clean-gdn-results' if args.clean_gdn else 'memory-results') if memory_rules else HERE
    modes=('original_symmetric_memory','existing_clean_gdn' if args.clean_gdn else 'existing_forward_memory') if memory_rules else MODES
    transport_path=folder/('transport.json' if memory_rules else 'transport-manifest.json')
    manifest = json.loads(transport_path.read_bytes())
    for item in (manifest if memory_rules else manifest['files']):
        actual = binding(Path(item['local_path']) if memory_rules else HERE / item['name'])
        assert actual['bytes'] == item['bytes'] and actual['sha256'] == item['sha256']
    paths = [[folder / 'results' / f'rank{rank}-{mode}.pt' for mode in modes] for rank in (0, 1)]
    values = [[torch.load(p, map_location='cpu', weights_only=False) for p in pair] for pair in paths]
    cross_rank = {}
    for j, mode in enumerate(modes):
        cross_rank[mode] = dict(signed=torch.equal(values[0][j]['signed'], values[1][j]['signed']),
                               QVA={k: all(torch.equal(a[k], b[k]) for a, b in zip(values[0][j]['values'], values[1][j]['values']))
                                    for k in values[0][j]['values'][0]})
        assert cross_rank[mode]['signed'] and all(cross_rank[mode]['QVA'].values())
    endpoints = {k: values[0][0]['detail'][k] == values[0][1]['detail'][k]
                 for k in ('target_logp0', 'target_logp1')}
    assert all(endpoints.values())
    reference = HERE.parents[1] / 'direct-target-credit-sample-20261007/v1/sample-analysis.json'
    points = json.loads(reference.read_bytes())['tasks']['appworld']['points']
    compared = []
    for p in points:
        item = {k: p[k] for k in ('mode', 'row', 'traj_uid', 'response_slot', 'packed_slot', 'token_id', 'token', 'reward', 'native_single_delete_d', 'native_endpoint_expected_A_FP32')}
        for j, mode in enumerate(modes):
            value = values[0][j]
            d = float(value['signed'][p['row'], p['packed_slot']])
            a = float(value['values'][p['row']]['dt_token_advantages'][p['response_slot']])
            item[mode] = dict(d=d, A=a, opposite_sign_to_previous_single_delete=d * p['native_single_delete_d'] < 0)
        compared.append(item)
    rank_reports = [json.loads((folder / 'results' / f'rank{rank}.json').read_bytes()) for rank in (0, 1)]
    physical = []
    for line in (folder / 'results/physical-mx-smi.jsonl').read_bytes().splitlines():
        e = json.loads(line)
        gpu = None
        for text in e['stdout'].splitlines():
            board = re.match(r'^\|\s*(\d+)\s+MetaX\s', text)
            if board:
                gpu = int(board[1])
            memory = re.search(r'(\d+)/(\d+) MiB', text)
            if memory and gpu in (4, 5):
                physical.append(dict(unix=e['unix'], gpu=gpu, used_mib=int(memory[1])))
    resource = []
    for rank in (0, 1):
        phases = [json.loads(line) for line in (folder / 'results' / f'rank{rank}-phases.jsonl').read_bytes().splitlines()]
        resource.append(dict(rank=rank, physical_sampled_peak_mib=max(e['used_mib'] for e in physical if e['gpu'] == rank + 4),
                             process_phase_max_pss_bytes=max(e['pss_bytes'] for e in phases),
                             phases=binding(folder / 'results' / f'rank{rank}-phases.jsonl')))
        assert rank_reports[rank]['phase'] == 'complete' and rank_reports[rank]['profile_restored']
    result = dict(
        status='Existing interaction-order sensitivity measured; not a credit repair or deployment',
        diagnostic_code_commit=json.loads((folder/'launch.json').read_bytes())['code_commit'],
        launch=binding(folder / 'launch.json'), source_sha256=values[0][0]['source_sha256'],
        original_native_sha256=values[0][0]['native_sha256'],
        scope='Same actual high-impact AppWorld B4, same initialized model and native target endpoint scores; only original attention_pv_rules option differs. Original GDN symmetric, head, reward, Q/V/A and PPO remain unchanged.',
        existing_owner=binding(HERE.parents[1] / 'direct-target-finite-owner-audit-20261008/v1/finite-reference-owner-source.json'),
        previous_independent_single_delete_reference=binding(reference),
        transport=binding(transport_path),
        artifacts=[binding(p) for pair in paths for p in pair],
        exact_target_endpoint_arrays_equal=endpoints,
        cross_rank_exact_equal=cross_rank,
        sampled_points=compared,
        biased_sample_sign_disagreements={mode: sum(p[mode]['opposite_sign_to_previous_single_delete'] for p in compared) for mode in modes},
        sampled_points_count=len(compared),
        elapsed_seconds_by_rank=[{mode: r['modes'][mode]['seconds'] for mode in modes} for r in rank_reports],
        resources=resource,
        rule_definitions=dict(content1='Original deltaP*V0 + P1*deltaV finite interaction routing',
                              content0='Existing owner reverses finite endpoints and uses V1, selecting deltaP*V1 + P0*deltaV'),
        imported_runners=[r['runner'] for r in rank_reports],
        original_GDN_rules=rank_reports[0]['original_GDN_rules'],
        unchanged=dict(QVA=True, PPO=True, whitening=True, original_reward=True,
                       target_IDs_and_offsets=True, LoRA_rank=8, LoRA_alpha=16, actual_DT_B_per_rank=4,
                       optimizer_steps=0, backward_calls=0, rollout_calls=0, checkpoint_restore=0,
                       formal_restart=False, production_profile_changed=False),
        limitations=[
            'The twelve points are the previously selected biased high-impact B4 sample, not a population error rate. Their single-delete references come from a separate original-model diagnostic, not newly scored in this rule-comparison job.',
            'The current content1 replay d at the newline is -4.41555, while the older formal saved value is -4.29009. These independent replay results are not claimed to be bitwise identical; the current two-rule target endpoint arrays are exactly identical.',
            'The rule swap improves the largest negative outlier but increases sign disagreements in these twelve points from four to five. That does not justify changing the production rule.',
            'The second call is warm. Its shorter time is not evidence that content0 is a faster algorithm.',
            'Cross-rank equality, finite values, equal target endpoints and attribution conservation are not FA/FLA precision criteria or an individual-counterfactual accuracy proof. No tolerance or correction multiplier was introduced.',
            'The original diagnostic main reconstructs configuration from the frozen source startup options; it did not write an effective-config.yaml. Source and executed main SHA are bound, and worker rank8/alpha16/B4 checks ran. No missing file is represented as a runtime configuration capture.',
            'Cumulative deletion/RISE/MAS comparison is a separate subsequent evaluation, not completed by these single-token comparisons.',
        ],
        credit_repaired=False, formal_update_released=False,
        cuda_initialized_in_local_analysis=torch.cuda.is_initialized(),
    )
    if memory_rules:
        result['scope']='Same actual high-impact AppWorld B4, unchanged native target endpoint scores; only the existing finite_fla_by_layer map selects the original averaged or forward callback. Norm/gate, FA, head, reward, Q/V/A formula and PPO remain unchanged.'
        result['rule_definitions']=dict(original_symmetric_memory='Original average_memory_endpoint_orders of the compiled finite_fla callback',existing_forward_memory='Original compiled finite_fla callback, selected by its existing empty finite_fla_by_layer map')
        result['limitations']=[v for v in result['limitations'] if not v.startswith(('The current content1','The rule swap'))]
        result['limitations'].append('This evaluates existing memory rules. The earlier local forward-order improvement does not carry to the complete vector: the selected newline becomes more negative. Twelve selected points are not population accuracy. Original author curves are reported separately; no profile switch is deployed.')
        original=torch.load(HERE/'results/rank0-original_content1.pt',map_location='cpu',weights_only=False)
        result['baseline_original_symmetric_vector_equals_prior_replay']=torch.equal(values[0][0]['signed'],original['signed'])
        result['baseline_original_symmetric_vector_maxabs_vs_prior_replay']=float((values[0][0]['signed']-original['signed']).abs().max())
        result['unchanged']['QVA_formula']=result['unchanged'].pop('QVA')
        result['selected_B4_raw_advantage_moments']={}
        for j,mode in enumerate(modes):
            groups=values[0][j]['report']['raw_advantage_groups']
            outlier=float(values[0][j]['values'][3]['dt_token_advantages'][125])
            result['selected_B4_raw_advantage_moments'][mode]=dict(groups=groups,
                newline_advantage=outlier,
                newline_A_squared_share_of_prior_sources=outlier**2/groups['prior_source']['sumsq'],
                newline_A_squared_share_of_all_policy=outlier**2/sum(g['sumsq'] for g in groups.values()),
                scope='This selected B4 only; A-squared is not a parameter-gradient share or the original global-batch whitening variance.')
        previous_native=reference.parent/'results-appworld/appworld-rank0-most_negative-targets.pt'
        native=torch.load(previous_native,map_location='cpu',weights_only=False)
        selected=native['samples'].eq(3)
        future=selected & native['predictor_positions'].ge(2883)
        earlier=selected & ~future
        differences=native['factual_target_logp'].double()-native['reference_target_logp'].double()
        indices=future.nonzero().flatten()
        largest=indices[differences[indices].abs().argmax()]
        result['newline_previous_native_target_decomposition']=dict(
            source=binding(previous_native),target_tokens=int(selected.sum()),
            future_target_tokens=int(future.sum()),earlier_target_tokens=int(earlier.sum()),
            earlier_maxabs=float(differences[earlier].abs().max()) if earlier.any() else None,
            joint_d=float(differences[selected].sum()),
            positive_future_sum=float(differences[future].clamp_min(0).sum()),
            negative_future_sum=float(differences[future].clamp_max(0).sum()),
            dominant=dict(predictor_position=int(native['predictor_positions'][largest]),
                target_token_id=int(native['labels'][largest]),
                d=float(differences[largest]),
                factual_probability=float(native['factual_target_logp'][largest].double().exp()),
                deleted_probability=float(native['reference_target_logp'][largest].double().exp())),
            scope='Previously measured independent uncached native single deletion; not a new environment rollout or the cached DT endpoint. Empty earlier-target set is recorded as null, not zero evidence.')
        orders_path=HERE/'native-memory-orders-results/results/result.json'
        single_path=HERE/'native-single-output-results/results/result.json'
        orders=json.loads(orders_path.read_bytes())
        single=json.loads(single_path.read_bytes())
        keys=('q','k','v','g','beta')
        single_terms={key:sum(group['original_single_finite_terms'][key]
                             for group in single['groups']) for key in keys}
        order_terms={name:{key:sum(group['memory_orders'][i]['terms'][key]
                                   for group in orders['groups']) for key in keys}
                     for i,name in enumerate(('forward','reversed'))}
        errors={name:{key:terms[key]-single_terms[key] for key in keys}
                for name,terms in order_terms.items()}
        result['native_fla_V_branch_order_diagnosis']=dict(
            sources=[binding(orders_path),binding(single_path)],
            original_finite_on_actual_single_endpoints=single_terms,
            joint_coefficients_times_actual_single_delta=order_terms,
            component_difference_from_actual_single_finite=errors,
            reverse_total_component_difference=sum(errors['reversed'].values()),
            reverse_V_component_difference=errors['reversed']['v'],
            reverse_other_components_combined_difference=sum(errors['reversed'][key]
                                                           for key in keys if key!='v'),
            actual_native_single_output_effect=single['summary']['native_output_effect'],
            actual_single_finite_output_effect=single['summary']['original_single_finite_contraction'],
            scope='CPU reduction of previously saved original FLA measurements, no new model/operator run. The +12.016 V term belongs to the original finite decomposition on actual single-deletion endpoints, not a separately measured intervention on V, a reward, or a whole-token advantage. The full native single-output effect is +15.8275. This locates joint-context interaction allocation; it neither proves population error nor accepts a new rule or numerical tolerance.')
    if args.clean_gdn:
        result['scope']='Same original high-impact AppWorld B4; owner-preserved clean-v1 empty norm/gate and memory maps versus current symmetric profile. Current verified offload/chunking/kernels, FA, head, native endpoints, targets, Q/V/A formula and PPO unchanged. This is a numerical-rule diagnostic, not a historical-runtime restore or formal deployment.'
        result['rule_definitions']=dict(original_symmetric_memory='Current owner symmetric norm/gate and averaged memory rules',existing_clean_gdn='Preserved clean-v1 defaults: content1 norm/gate and original forward memory callback')
        result['preserved_clean_owner']=rank_reports[0]['preserved_clean_owner']
        result['limitations']=[v for v in result['limitations'] if not v.startswith('This evaluates existing memory rules.')]
        result['limitations'].append('The complete preserved GDN profile is compared, rather than only its memory rule. Selected-token improvements alone do not establish overall attribution quality; original author curves remain separate. No production switch or precision pass is asserted.')
    target = REPO / ('experiments/rl/results_existing_clean_GDN_20261008.json' if args.clean_gdn else ('experiments/rl/results_existing_memory_rule_20261008.json' if args.memory else 'experiments/rl/results_existing_PV_rule_20261008.json'))
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    (HERE / ('clean-gdn-analysis.json' if args.clean_gdn else ('memory-analysis.json' if args.memory else 'analysis.json'))).write_text(json.dumps(dict(points=compared, sign_disagreements=result['biased_sample_sign_disagreements']), ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(json.dumps(dict(receipt=binding(target), signs=result['biased_sample_sign_disagreements'],
                         worst_point=next(p for p in compared if p['row'] == 3 and p['mode'] == 'most_negative'),
                         resources=resource), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

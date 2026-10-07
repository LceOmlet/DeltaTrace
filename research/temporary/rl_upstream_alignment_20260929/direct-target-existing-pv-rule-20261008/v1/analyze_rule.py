"""Describe the existing-rule comparison without fitting an acceptance rule."""
from pathlib import Path
import hashlib
import json
import math
import re

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
MODES = ('original_content1', 'existing_content0')


def binding(p):
    return dict(path=p.resolve().as_posix(), bytes=p.stat().st_size,
                sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def main():
    manifest = json.loads((HERE / 'transport-manifest.json').read_bytes())
    for item in manifest['files']:
        actual = binding(HERE / item['name'])
        assert actual['bytes'] == item['bytes'] and actual['sha256'] == item['sha256']
    paths = [[HERE / 'results' / f'rank{rank}-{mode}.pt' for mode in MODES] for rank in (0, 1)]
    values = [[torch.load(p, map_location='cpu', weights_only=False) for p in pair] for pair in paths]
    cross_rank = {}
    for j, mode in enumerate(MODES):
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
        for j, mode in enumerate(MODES):
            value = values[0][j]
            d = float(value['signed'][p['row'], p['packed_slot']])
            a = float(value['values'][p['row']]['dt_token_advantages'][p['response_slot']])
            item[mode] = dict(d=d, A=a, opposite_sign_to_previous_single_delete=d * p['native_single_delete_d'] < 0)
        compared.append(item)
    rank_reports = [json.loads((HERE / 'results' / f'rank{rank}.json').read_bytes()) for rank in (0, 1)]
    physical = []
    for line in (HERE / 'results/physical-mx-smi.jsonl').read_bytes().splitlines():
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
        phases = [json.loads(line) for line in (HERE / 'results' / f'rank{rank}-phases.jsonl').read_bytes().splitlines()]
        resource.append(dict(rank=rank, physical_sampled_peak_mib=max(e['used_mib'] for e in physical if e['gpu'] == rank + 4),
                             process_phase_max_pss_bytes=max(e['pss_bytes'] for e in phases),
                             phases=binding(HERE / 'results' / f'rank{rank}-phases.jsonl')))
        assert rank_reports[rank]['phase'] == 'complete' and rank_reports[rank]['profile_restored']
    result = dict(
        status='Existing interaction-order sensitivity measured; not a credit repair or deployment',
        diagnostic_code_commit='9567ba35ab470f2429dc55c33f4b33f28f8d2e19',
        launch=binding(HERE / 'launch.json'), source_sha256=values[0][0]['source_sha256'],
        original_native_sha256=values[0][0]['native_sha256'],
        scope='Same actual high-impact AppWorld B4, same initialized model and native target endpoint scores; only original attention_pv_rules option differs. Original GDN symmetric, head, reward, Q/V/A and PPO remain unchanged.',
        existing_owner=binding(HERE.parents[1] / 'direct-target-finite-owner-audit-20261008/v1/finite-reference-owner-source.json'),
        previous_independent_single_delete_reference=binding(reference),
        transport=binding(HERE / 'transport-manifest.json'),
        artifacts=[binding(p) for pair in paths for p in pair],
        exact_target_endpoint_arrays_equal=endpoints,
        cross_rank_exact_equal=cross_rank,
        sampled_points=compared,
        biased_sample_sign_disagreements={mode: sum(p[mode]['opposite_sign_to_previous_single_delete'] for p in compared) for mode in MODES},
        sampled_points_count=len(compared),
        elapsed_seconds_by_rank=[{mode: r['modes'][mode]['seconds'] for mode in MODES} for r in rank_reports],
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
    target = REPO / 'experiments/rl/results_existing_PV_rule_20261008.json'
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    (HERE / 'analysis.json').write_text(json.dumps(dict(points=compared, sign_disagreements=result['biased_sample_sign_disagreements']), ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(json.dumps(dict(receipt=binding(target), signs=result['biased_sample_sign_disagreements'],
                         worst_point=next(p for p in compared if p['row'] == 3 and p['mode'] == 'most_negative'),
                         resources=resource), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

"""Describe saved native B4 accumulation and whole-direction propagation.

Only postprocess already recorded native backward results. No Fisher/NTK,
training, model, backward, optimizer, or credit implementation is introduced.
"""
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

HERE = Path(__file__).resolve().parent


def read(name):
    path = HERE / name
    value = json.loads(path.read_text(encoding='utf-8'))
    return value, dict(path=str(path.resolve()), SHA256=hashlib.sha256(path.read_bytes()).hexdigest())


matched, matched_source = read('native-gradient-ratio-chain-summary-v23.json')
whole, whole_source = read('native-full-policy-direction-exchange-v86-status-final.json')
contract, contract_source = read('audit_native_accumulation_contract_v107-dt.json')
contract = contract['report']
assert not any(contract[k] for k in ['update_has_no_sync', 'update_has_requires_gradient_sync',
                                    'update_has_set_is_last_backward', 'cuda_initialized'])
cases = defaultdict(dict)
for item in whole['report']['chain_reports']:
    value = item['value']
    assert value['trainable_SHA256_before'] == value['trainable_SHA256_after']
    assert value['scheduler_before'] == value['scheduler_after']
    assert value['native_accumulation_readout']['original_backward_calls'] == 8
    cases[value['iteration']][value['rank']] = value
assert all(set(ranks) == {0, 1} for ranks in cases.values())


def gram_description(name, matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    # The matched15 carrier has four explicit zero-loss placeholder B4s;
    # exclude those from direction-angle counts, retaining their original IDs.
    active = np.flatnonzero(np.diag(matrix) > 0)
    gram = matrix[np.ix_(active, active)]
    gram = (gram + gram.T) / 2
    trace = float(np.trace(gram))
    ones = np.ones(len(active))
    sum_energy = float(ones @ gram @ ones)
    eig, vectors = np.linalg.eigh(gram)
    order = np.argsort(eig)[::-1]
    eig, vectors = eig[order], vectors[:, order]
    diagonal = np.diag(gram)
    cosine = gram / np.sqrt(np.outer(diagonal, diagonal))
    cosine_eig = np.linalg.eigvalsh(cosine)[::-1]
    n = len(active)
    return dict(case=name, active_B4_indices=active.tolist(),
        global_gradient_norm=math.sqrt(sum_energy),
        B4_increment_RSS=math.sqrt(trace),
        directional_sum_factor=math.sqrt(sum_energy/trace),
        eigenvalue_fractions=(eig/trace).tolist(),
        leading_mode_ones_alignment_squared=float((vectors[:, 0]@ones)**2/n),
        leading_mode_fraction_of_sum_energy=float(eig[0]*(vectors[:, 0]@ones)**2/sum_energy),
        normalized_direction_eigenvalue_fractions=(cosine_eig/n).tolist(),
        mean_pairwise_B4_cosine=float((cosine.sum()-n)/(n*(n-1))) if n > 1 else None,
        B4_increment_energy_fractions=(diagonal/trace).tolist(),
        largest_B4_increment_energy_fraction=float(diagonal.max()/trace),
        off_diagonal_sum_energy=float(sum_energy-trace))


geometry = []
for case in matched['cases']:
    geometry.append(gram_description('matched15_theta5_'+case['case'], case['B4_increment_Gram']))
for name in ['full40_native', 'full80_native']:
    matrix = np.sum([np.asarray(c['native_accumulation_readout']['local_increment_Gram'])
                     for c in cases[name].values()], axis=0)
    geometry.append(gram_description(name, matrix))

forward_audits = []
fields = ['input_forward_RMS_median_all', 'input_forward_RMS_range_all',
          'output_forward_RMS_median_all', 'output_forward_RMS_range_all']
for a, b in [('full40_native', 'full_weights40_seed80'),
             ('full80_native', 'full_weights80_seed40')]:
    for rank in [0, 1]:
        left = {(r['micro'], r['boundary']): r for r in cases[a][rank]['records']}
        right = {(r['micro'], r['boundary']): r for r in cases[b][rank]['records']}
        assert set(left) == set(right)
        differences = [dict(micro=key[0], boundary=key[1], field=k)
                       for key in left for k in fields if left[key].get(k) != right[key].get(k)]
        forward_audits.append(dict(cases=[a, b], rank=rank, boundary_micro_pairs=len(left),
                                   compared_fields=fields, differences=differences))

boundary_norms = defaultdict(lambda: defaultdict(float))
for name, ranks in cases.items():
    for rank in ranks.values():
        for entry in rank['records']:
            match = re.search(r'\.layers\.(\d+)$', entry['boundary'])
            if match:
                # Native parameter reducer averages two ranks. Use the
                # corresponding logical loss scaling for local activations.
                boundary_norms[name][int(match.group(1))] += entry['output_adjoint_norm']**2/4
boundary_norms = {name: {layer: math.sqrt(v) for layer, v in rows.items()}
                  for name, rows in boundary_norms.items()}
propagation = []
for seed, a, b in [('whole40', 'full40_native', 'full_weights80_seed40'),
                   ('whole80', 'full_weights40_seed80', 'full80_native')]:
    before, after = boundary_norms[a], boundary_norms[b]
    assert set(before) == set(after) == set(range(32))
    ratios = {layer: after[layer]/before[layer] for layer in before}
    propagation.append(dict(fixed_native_seed=seed, weight40_case=a, weight80_case=b,
        layer_output_adjoint_RSS_weight40=before, layer_output_adjoint_RSS_weight80=after,
        weight80_over_weight40_RSS=ratios,
        adjacent_block_conditional_gain_ratio={layer: ratios[layer-1]/ratios[layer]
                                                for layer in range(1, 32)},
        layer0_ratio=ratios[0], layer24_ratio=ratios[24], layer31_ratio=ratios[31]))

report = dict(scope=__doc__, sources=[matched_source, whole_source, contract_source],
    native_accumulation_contract=dict(actor=contract['actor']['path'],
        actor_SHA256=contract['actor']['SHA256'],
        actor_no_sync_call=False, worker_sync_override_excerpts=contract['worker']['excerpts'],
        native_FSDP_default_reduce_grads=True,
        qualification='The native source default and unchanged observer path support per-backward synchronization; flags were not separately sampled at every saved backward.'),
    B4_geometry=geometry, within_weight_forward_summary_audits=forward_audits,
    fixed_whole_direction_propagation=propagation,
    findings=[
        'The fixed old81 whole PG view does not become more coherent from weight40 to80: mean B4 cosine and directional sum factor fall.',
        'A leading Gram eigenvalue is not evidence of a shared update direction: unequal B4 amplitudes strongly affect this weighted Gram.',
        'In both actual whole-direction exchanges, layer0 adjoint RSS increases14.16x or23.02x at weight80, with the same layer31 seed.',
        'The extra conditional gain is distributed through depth; several layer changes multiply. No single layer bug or independent causal share follows.',
    ], limits=[
        'These are saved finite weighted B4 gradient differences, not per-token covariance, Fisher, NTK, operator norms, or temporal optimizer alignment.',
        'Within-weight forward summary equality is recorded; it is not complete forward tensor equality or a new exact-replay requirement.',
        'The old81 input/advantage carrier is fixed. This does not reconstruct the whole historical40-to80 sampling/training path.',
        'The existing matched15 target contrast and the whole PG carrier have different scopes and are not combined into a causal percentage.',
        'No mechanism driving the temporal propagation change, root repair, or training-quality claim is accepted here.',
    ], new_model_calls=0, new_backward_calls=0, optimizer_updates=0,
    rootcause_resolved=False, accepted_repair=None, deployed_repair=None)
dest = HERE/'saved-native-gradient-geometry-v108.json'
dest.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
print(json.dumps(dict(path=str(dest.resolve()),
    B4=[{k:row[k] for k in ['case','global_gradient_norm','directional_sum_factor',
                            'mean_pairwise_B4_cosine','largest_B4_increment_energy_fraction']}
        for row in geometry],
    propagation=[{k:row[k] for k in ['fixed_native_seed','layer0_ratio','layer24_ratio','layer31_ratio']}
                 for row in propagation],
    within_weight_forward_summary_differences=sum(len(r['differences']) for r in forward_audits))))

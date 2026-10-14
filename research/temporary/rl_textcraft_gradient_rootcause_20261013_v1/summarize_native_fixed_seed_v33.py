"""Account the actual fixed-input/fixed-seed native VJP contrast.

Pure local arithmetic on completed native records. No model or gradient code.
"""
from collections import defaultdict
from datetime import datetime
import json
import math
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
status = json.loads((HERE / 'native-fixed-seed-propagation-v33-status-final.json').read_bytes())['report']
assert not status['alive']
assert status['state']['phase'] == 'complete_native_gradient_chain_localization_not_repair'
cases = {}
for case in ('late80_native', 'late5_native', 'late5_seed80'):
    chains = [r['value'] for r in status['chain_reports'] if r['value']['iteration'] == case]
    assert len(chains) == 2
    sums = defaultdict(lambda: defaultdict(float))
    branch = defaultdict(lambda: defaultdict(float))
    layers = defaultdict(float)
    sides = defaultdict(float)
    gram = [[0.] * 8 for _ in range(8)]
    seeds = []
    for r in chains:
        assert r['trainable_SHA256_before'] == r['trainable_SHA256_after']
        assert r['scheduler_before'] == r['scheduler_after'] == r['checkpoint']
        for b in r['records']:
            n = b['boundary'].split('base_model.model.')[-1]
            for side in ('input', 'output'):
                if side + '_adjoint_norm' in b:
                    assert b.get(side + '_adjoint_nonfinite_positions', 0) == 0
                    sums[n][side] += b[side + '_adjoint_norm'] ** 2
        for b in r['native_decoder_branch_backward']:
            n = b['name'].split('base_model.model.')[-1]
            for side in ('input', 'output'):
                for value in b['native_module_grad_' + side]:
                    assert value['nonfinite_positions'] == 0
                    branch[n][side] += value['norm'] ** 2
        a = r['native_accumulation_readout']
        assert a['original_backward_calls'] == 8
        for i in range(8):
            for j in range(8):
                gram[i][j] += a['local_increment_Gram'][i][j]
        for p in a['parameters']:
            match = re.search(r'layers\.(\d+)\.', p['name'])
            layers[match.group(1) if match else 'outside_decoder'] += p['local_preclip_sumsq']
            sides['A' if 'lora_A' in p['name'] else 'B'] += p['local_preclip_sumsq']
        seeds += [dict(rank=r['rank'], **s) for s in r['diagnostic_boundary_seeds']]
    boundaries = {n: {s: math.sqrt(v) / 2 for s, v in d.items()} for n, d in sums.items()}
    modules = {n: {s: math.sqrt(v) / 2 for s, v in d.items()} for n, d in branch.items()}
    residuals = []
    for layer in (29, 30, 31):
        n = f'model.layers.{layer}'
        g = boundaries[n]['output']
        m = modules[n + '.post_attention_layernorm']['input']
        mix = 'self_attn' if layer == 31 else 'linear_attn'
        mid = modules[n + '.' + mix]['output']
        a = modules[n + '.input_layernorm']['input']
        total = boundaries[n]['input']
        residuals.append(dict(layer=layer, output_norm=g, MLP_branch_norm=m, middle_total_norm=mid,
                              attention_branch_norm=a, input_total_norm=total,
                              MLP_cosine_from_norm_identity=(mid * mid - g * g - m * m) / (2 * g * m),
                              mixer_cosine_from_norm_identity=(total * total - mid * mid - a * a) / (2 * mid * a)))
    obs = next(o for o in status['state']['observations'] if o['iteration'] == case)
    G = obs['gradient_statistics']['norms']['native_target']
    energy = sum(gram[i][i] for i in range(8))
    reconstructed = math.sqrt(max(0, sum(sum(row) for row in gram)))
    cases[case] = dict(checkpoint=chains[0]['checkpoint'],
                       scope='Diagnostic VJP with the actual saved checkpoint80 block31-output seed' if case == 'late5_seed80' else 'Original native target policy gradient',
                       parameter_VJP_norm=G, boundaries=boundaries, branch_modules=modules,
                       residuals=residuals, parameter_sumsq_by_layer=dict(layers),
                       parameter_sumsq_by_adapter_side=dict(sides), seeds=seeds,
                       B4_increment_Gram=gram, B4_increment_energy_sqrt=math.sqrt(energy),
                       B4_directional_accumulation_factor=G / math.sqrt(energy),
                       Gram_reconstructed_parameter_norm=reconstructed,
                       Gram_reconstruction_difference=reconstructed - G,
                       parameter_shard_reconstruction_difference=math.sqrt(sum(layers.values())) - G)

a = cases['late80_native']
b = cases['late5_native']
c = cases['late5_seed80']
assert len(a['seeds']) == len(c['seeds']) == 16 and not b['seeds']
seed_checks = []
for source in a['seeds']:
    target = next(s for s in c['seeds'] if (s['rank'], s['micro']) == (source['rank'], source['micro']))
    assert source['shape'] == target['shape'] and source['dtype'] == target['dtype']
    seed_checks.append(dict(rank=source['rank'], micro=source['micro'],
                            applied_minus_saved_norm=target['applied_norm'] - source['applied_norm']))

g80 = a['boundaries']['model.layers.31']['output']
g5 = b['boundaries']['model.layers.31']['output']
g_fixed = c['boundaries']['model.layers.31']['output']
comparisons = []
for layer in (31, 30, 29, 28, 24, 15, 0):
    name = f'model.layers.{layer}'
    v80, v5, fixed = (x['boundaries'][name]['input'] for x in (a, b, c))
    comparisons.append(dict(layer=layer, native_weight_input_adjoint_ratio=v80 / v5,
                            native_relative_gain=(v80 / g80) / (v5 / g5),
                            fixed_incoming_seed_weight_and_internal_state_ratio=v80 / fixed,
                            incoming_direction_gain_at_fixed_checkpoint5=(fixed / g_fixed) / (v5 / g5)))

G80, G5, Gfixed = (x['parameter_VJP_norm'] for x in (a, b, c))
report = dict(observed_local=datetime.now().astimezone().isoformat(), version='native-fixed-seed-propagation-v33',
              status='completed_native_VJP_not_repair', job=status['job'], cases=cases,
              actual_worker_bindings=status['state']['bindings'], seed_checks=seed_checks,
              block31_output_adjoint_norms=dict(native80=g80, native5=g5, fixed_seed5=g_fixed),
              layer_input_comparisons=comparisons,
              parameter_comparison=dict(native_G80_over_G5=G80 / G5,
                                        actual_incoming_magnitude_ratio=g80 / g5,
                                        fixed_seed_weight_and_state_VJP_ratio=G80 / Gfixed,
                                        incoming_direction_gain_at_checkpoint5=(Gfixed / g_fixed) / (G5 / g5)),
              limits=[
                  'Fixed incoming seed is an exact stored native tensor, not a norm-matched synthetic direction. The intervention is not a complete PPO gradient above block31 and is not used for training.',
                  'Fixed-seed differences include both weight and induced forward-state differences; they are not an isolated layer-parameter intervention or an operator norm.',
                  'Direction gain is a finite directional quotient at checkpoint5, not a universal amplification factor or independent causal share.',
                  'Residual cosines are derived from separate norms, not directly measured inner products; numerical rounding remains.',
                  'This finite comparison cannot establish the full85-iteration mechanism or accept a repair.'],
              optimizer_updates=0, scheduler_updates=0, DT_calls=0, rollout_calls=0, validation_calls=0,
              GRPO_calls=0, rootcause_resolved=False, accepted_repair=None, deployed_repair=None)
path = HERE / 'native-fixed-seed-propagation-summary-v33.json'
path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps(dict(record=str(path), parameter_comparison=report['parameter_comparison'],
                      layer_input_comparisons=comparisons, seed_checks=seed_checks)))

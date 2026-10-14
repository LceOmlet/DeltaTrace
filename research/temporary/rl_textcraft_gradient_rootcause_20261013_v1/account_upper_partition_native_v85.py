"""Account the completed upper-state causal probe, including actual full PG.

Reuse original saved-record accounting. No model, loss, backward, optimizer,
scheduler, DT, generation or evaluation is called by this script.
"""
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
source = HERE / 'summarize_native_fixed_seed_v33.py'
source_sha = 'c09b9ecc7577388a50e58eaad105b346b22356d41548eba814a7aca2a6238e44'
assert hashlib.sha256(source.read_bytes()).hexdigest() == source_sha
prefix = source.read_text(encoding='utf-8').split("\na = cases['late80_native']", 1)[0]


def account(filename, names):
    code = prefix.replace('native-fixed-seed-propagation-v33-status-final.json', filename)
    old = "('late80_native', 'late5_native', 'late5_seed80')"
    assert old in code
    code = code.replace(old, repr(tuple(names)))
    old = "    G = obs['gradient_statistics']['norms']['native_target']"
    new = """    norm_values = obs['gradient_statistics']['norms']
    assert len(norm_values) == 1 and next(iter(norm_values)) in ('native_target', 'full_DT')
    G = next(iter(norm_values.values()))"""
    assert old in code
    code = code.replace(old, new)
    env = {'__file__': __file__}
    exec(compile(code, str(source), 'exec'), env)
    return env['status'], env['cases']


manifest = json.loads((HERE / 'prepare_upper_partition_full_PG_v83-dt.json').read_bytes())['report']
specs = {c['name']: c for c in manifest['cases']}
status, cases = account('native-upper-adapter-partition-v81-status-final.json', specs)
assert status['job']['pid'] == 1566496 and status['job']['birth'] == 1792018535.92
old40_status, old40 = account('native-midpoint-fixed-seed-v35-status-final.json', ['late40_seed80'])
old80_status, old80 = account('native-fixed-seed-propagation-v33-status-final.json', ['late80_native'])
fixed_baselines = {40: old40['late40_seed80'], 80: old80['late80_native']}
full_source = HERE / 'native-PG-driver-summary-v38.json'
full = json.loads(full_source.read_bytes())
full_baselines = {c['checkpoint']: c['gradient_statistics']['norms']['full_DT'] for c in full['cases']}
binding_checks = []
for binding in status['state']['diagnostic_parameter_partition_bindings']:
    case = specs[binding['case']]
    wanted = case['diagnostic_parameter_partition']['rank_artifacts'][str(binding['rank'])]
    assert binding['artifact_SHA256'] == wanted['SHA256']
    assert binding['base_trainable_SHA256'] == wanted['expected_base_trainable_SHA256']
    assert binding['result_trainable_SHA256'] == wanted['expected_result_trainable_SHA256']
    assert binding['scheduler_before'] == binding['scheduler_after'] == case['checkpoint']
    assert binding['optimizer_steps_before'] == binding['optimizer_steps_after'] == [4.0 * case['checkpoint']]
    assert not binding['unexpected_keys']
    binding_checks.append(binding)
assert len(binding_checks) == 8


def band_gain(item, first):
    b = item['boundaries']
    return b[f'model.layers.{first}']['input'] / b[f'model.layers.{first + 7}']['output']


def lower_parameter_norm(item):
    return math.sqrt(math.fsum(v for k, v in item['parameter_sumsq_by_layer'].items()
                              if k.isdigit() and int(k) < 24))


contrasts = []
seed_checks = []
for name, case in cases.items():
    spec = specs[name]
    base = spec['checkpoint']
    seeded = bool(spec.get('boundary_seed_source_case'))
    observation = next(o for o in status['state']['observations'] if o['iteration'] == name)
    assert observation['diagnostic_seed_intervention'] == seeded
    case['scope'] = ('Fixed original BF16 decoder31 VJP' if seeded else
                     'Complete original81 first-global64 policy gradient at the native rho1 point')
    case['parameter_intervention'] = spec['diagnostic_parameter_partition']
    case['native_gradient_key'] = next(iter(observation['gradient_statistics']['norms']))
    assert case['native_gradient_key'] == ('native_target' if seeded else 'full_DT')
    expected_seeds = fixed_baselines[80]['seeds']
    if seeded:
        assert len(case['seeds']) == len(expected_seeds) == 16
        for seed in case['seeds']:
            old = next(s for s in expected_seeds
                       if (s['rank'], s['micro']) == (seed['rank'], seed['micro']))
            assert seed['shape'] == old['shape'] and seed['dtype'] == old['dtype']
            assert seed['applied_norm'] == old['applied_norm']
            key = f"{seed['rank']}:{seed['micro']}"
            assert seed['source_SHA256'] == spec['boundary_seed_SHA256_by_rank_micro'][key]
            seed_checks.append(dict(case=name, rank=seed['rank'], micro=seed['micro'],
                                    source_SHA256=seed['source_SHA256'], applied_minus_saved_norm=0.0))
    else:
        assert not case['seeds']
    baseline_G = fixed_baselines[base]['parameter_VJP_norm'] if seeded else full_baselines[base]
    row = dict(case=name, body_checkpoint=base,
        upper_checkpoint=spec['diagnostic_parameter_partition']['donor_checkpoint'],
        readout='fixed_seed_VJP' if seeded else 'actual_full_original81_PG',
        baseline_global_parameter_norm=baseline_G,
        intervened_global_parameter_norm=case['parameter_VJP_norm'],
        intervened_over_baseline_parameter_norm=case['parameter_VJP_norm'] / baseline_G,
        lower0_23_parameter_norm=lower_parameter_norm(case),
        band_gains={f'{first}-{first+7}': band_gain(case, first) for first in (0, 8, 16, 24)},
        B4_directional_accumulation_factor=case['B4_directional_accumulation_factor'],
        Gram_reconstruction_difference=case['Gram_reconstruction_difference'])
    if seeded:
        row['baseline_lower0_23_parameter_norm'] = lower_parameter_norm(fixed_baselines[base])
        row['lower0_23_intervened_over_baseline'] = row['lower0_23_parameter_norm'] / row['baseline_lower0_23_parameter_norm']
        row['band_gain_ratios_to_same_body_baseline'] = {
            f'{first}-{first+7}': band_gain(case, first) / band_gain(fixed_baselines[base], first)
            for first in (0, 8, 16, 24)}
    contrasts.append(row)

report = dict(recorded_at=datetime.now().astimezone().isoformat(),
    status='completed_native_parameter_intervention_readouts_not_rootcause_repair',
    scope=__doc__, job=status['job'], actual_parameter_bindings=binding_checks,
    seed_checks=seed_checks, cases=cases, contrasts=contrasts,
    baseline_fixed_cases=fixed_baselines, baseline_full_original81_PG=full_baselines,
    original_accounting_source=dict(path=str(source), SHA256=source_sha),
    full_PG_baseline_source=dict(path=str(full_source), SHA256=hashlib.sha256(full_source.read_bytes()).hexdigest()),
    limits=[
        'Only the upper24-31 actual LoRA state is exchanged in each direction. Actual complete trainable hashes verify the intended composite state and no later parameter/scheduler change.',
        'Same-body fixed-seed contrasts isolate the effect of this parameter partition on the finite old input/seed. They are not an operator norm or a historical optimizer-component attribution.',
        'Full-PG and fixed-seed readouts use different original carriers. Their norm ratios are separate observables, not multiplicative causal shares.',
        'Full-PG readouts use the actual original81 complete first global64 and saved full256-whitened advantages at rho1, not all four later clipped optimizer steps or a whole85-iteration trend.',
        'Completed baselines were measured in separate native runs. Low-precision variation and all Gram/decomposition residuals remain reported; no new tolerances or all-layer decline gate.',
        'A lower probe norm alone cannot accept a repair. The connection to actual full policy-gradient growth and its historical update driver remains necessary.'
    ], new_model_calls=0, new_backward_calls=0, optimizer_updates=0, scheduler_updates=0,
    rootcause_repair_verified=False, accepted_repair=None, deployed_repair=None)
dest = HERE / 'upper-partition-native-accounting-v85.json'
dest.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps(dict(path=str(dest.resolve()), contrasts=contrasts), ensure_ascii=False))

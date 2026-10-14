"""Compare saved native control derivatives with completed DTPO derivatives.

This only accounts saved scalar/Gram records. It makes no model, gradient,
optimizer, scheduler, rollout, DT or validation call.
"""
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
accountant = HERE / 'summarize_native_fixed_seed_v33.py'
accountant_sha = 'c09b9ecc7577388a50e58eaad105b346b22356d41548eba814a7aca2a6238e44'
assert hashlib.sha256(accountant.read_bytes()).hexdigest() == accountant_sha
code = accountant.read_text(encoding='utf-8').split("\na = cases['late80_native']", 1)[0]
changes = {
    'native-fixed-seed-propagation-v33-status-final.json':
        'native-trained-control-propagation-v78-status-final.json',
    "('late80_native', 'late5_native', 'late5_seed80')":
        "('control160_native_target', 'control160_fixed_seed80')",
    "case == 'late5_seed80'": "case == 'control160_fixed_seed80'",
}
for old, new in changes.items():
    assert old in code, old
    code = code.replace(old, new)
exec(compile(code, str(accountant), 'exec'), globals())

assert status['job']['pid'] == 1248589
assert status['job']['birth'] == 1792015542.95
assert status['state']['phase'] == 'complete_native_gradient_chain_localization_not_repair'
for key in ('optimizer_updates', 'scheduler_updates', 'DT_calls', 'rollout_calls',
            'validation_calls', 'GRPO_calls'):
    assert status['state'][key] == 0, key

manifest_record = json.loads((HERE / 'saved-control-propagation-v78-current-manifest.json').read_bytes())
manifest = manifest_record['manifest']
expected_weights = manifest['cases'][0]['expected_trainable_SHA256_by_rank']
assert expected_weights == manifest['cases'][1]['expected_trainable_SHA256_by_rank']
binding_checks = []
for item in status['chain_reports']:
    rank = item['value']
    expected = expected_weights[str(rank['rank'])]
    assert rank['trainable_SHA256_before'] == rank['trainable_SHA256_after'] == expected
    assert rank['optimizer_steps'] == rank['scheduler_steps'] == 0
    assert rank['scheduler_before'] == rank['scheduler_after'] == 160
    binding_checks.append(dict(case=rank['iteration'], rank=rank['rank'],
                               native_trainable_SHA256=expected,
                               chain_path=item['path'], chain_SHA256=item['SHA256']))

old33_path = HERE / 'native-fixed-seed-propagation-summary-v33.json'
old33 = json.loads(old33_path.read_bytes())
old35_path = HERE / 'native-midpoint-fixed-seed-summary-v35.json'
old35 = json.loads(old35_path.read_bytes())
old80 = old33['cases']['late80_native']
old5 = old33['cases']['late5_seed80']
control = cases['control160_fixed_seed80']
natural = cases['control160_native_target']

seed_checks = []
seed_spec = manifest['cases'][1]
assert len(control['seeds']) == len(old80['seeds']) == 16
for source in old80['seeds']:
    target = next(s for s in control['seeds']
                  if (s['rank'], s['micro']) == (source['rank'], source['micro']))
    assert source['shape'] == target['shape']
    assert source['dtype'] == target['dtype'] == 'torch.bfloat16'
    assert source['applied_norm'] == target['applied_norm']
    assert target['source'].startswith(seed_spec['boundary_seed_source_directory'] + '/')
    key = f"{target['rank']}:{target['micro']}"
    # The manifest keys are the native rank/micro identity, not a new seed.
    expected_sha = seed_spec['boundary_seed_SHA256_by_rank_micro'].get(key)
    if expected_sha is None:
        expected_sha = seed_spec['boundary_seed_SHA256_by_rank_micro'][f"rank{target['rank']}-micro{target['micro']}"]
    assert target['source_SHA256'] == expected_sha
    seed_checks.append(dict(rank=target['rank'], micro=target['micro'],
                            source=target['source'], source_SHA256=expected_sha,
                            shape=target['shape'], dtype=target['dtype'],
                            applied_minus_saved_norm=target['applied_norm'] - source['applied_norm']))

boundary_contrasts = {}
for name, original in old80['boundaries'].items():
    if name not in control['boundaries']:
        continue
    comparison = control['boundaries'][name]
    boundary_contrasts[name] = {
        side: dict(DTPO80=value, GRPO160=comparison[side],
                   DTPO80_over_GRPO160=value / comparison[side] if comparison[side] else None)
        for side, value in original.items() if side in comparison
    }

bands = {}
for first in (0, 8, 16, 24):
    def gain(item):
        b = item['boundaries']
        return b[f'model.layers.{first}']['input'] / b[f'model.layers.{first+7}']['output']
    bands[f'{first}-{first+7}'] = dict(DTPO80=gain(old80), GRPO160=gain(control),
                                      DTPO80_over_GRPO160=gain(old80) / gain(control))

layer_contrasts = []
for layer in range(31, -1, -1):
    name = f'model.layers.{layer}'
    oldb, newb = old80['boundaries'][name], control['boundaries'][name]
    oldgain = oldb['input'] / oldb['output']
    newgain = newb['input'] / newb['output']
    layer_contrasts.append(dict(layer=layer, DTPO80_gain=oldgain, GRPO160_gain=newgain,
                                relative_directional_gain=oldgain/newgain,
                                log_relative_directional_gain=math.log(oldgain/newgain)))

report = dict(
    observed_local=datetime.now().astimezone().isoformat(), version='saved-control-propagation-accounting-v80',
    status='completed_native_control_derivative_comparison_not_rootcause_repair',
    original_native_job=status['job'], source_result_path=str(HERE / 'native-trained-control-propagation-v78-status-final.json'),
    source_result_SHA256=hashlib.sha256((HERE / 'native-trained-control-propagation-v78-status-final.json').read_bytes()).hexdigest(),
    accounting_owner=dict(path=str(accountant), SHA256=accountant_sha),
    source_manifest=dict(path=manifest_record['path'], SHA256=manifest_record['SHA256']),
    actual_bindings=status['state']['bindings'], binding_checks=binding_checks,
    cases=cases, fixed_seed_checks=seed_checks,
    fixed_seed_parameter_VJP=dict(DTPO5=old5['parameter_VJP_norm'], DTPO40=old35['G'],
                                  DTPO80=old80['parameter_VJP_norm'], GRPO160=control['parameter_VJP_norm'],
                                  DTPO80_over_GRPO160=old80['parameter_VJP_norm']/control['parameter_VJP_norm'],
                                  GRPO160_over_DTPO5=control['parameter_VJP_norm']/old5['parameter_VJP_norm']),
    same_coefficients_natural_target_PG=dict(DTPO80=old80['parameter_VJP_norm'],
                                           GRPO160=natural['parameter_VJP_norm'],
                                           DTPO80_over_GRPO160=old80['parameter_VJP_norm']/natural['parameter_VJP_norm']),
    fixed_seed_boundary_contrasts=boundary_contrasts,
    fixed_seed_band_gains=bands, fixed_seed_layer_gains=layer_contrasts,
    fixed_seed_accumulation=dict(
        DTPO80_energy_sqrt=old80['B4_increment_energy_sqrt'],
        GRPO160_energy_sqrt=control['B4_increment_energy_sqrt'],
        energy_sqrt_ratio=old80['B4_increment_energy_sqrt']/control['B4_increment_energy_sqrt'],
        DTPO80_direction=old80['B4_directional_accumulation_factor'],
        GRPO160_direction=control['B4_directional_accumulation_factor'],
        direction_ratio=old80['B4_directional_accumulation_factor']/control['B4_directional_accumulation_factor']),
    old_results=[dict(path=str(p), SHA256=hashlib.sha256(p.read_bytes()).hexdigest())
                 for p in (old33_path, old35_path)],
    limits=[
        'The saved control160 actor weights are measured on the unchanged15 old matched factual targets plus49 unchanged zero-derivative interface rows. This is not its natural training batch or a GRPO advantage/quality test.',
        'Control160 and DTPO80 have different training histories and step counts. Their contrast is conditional derivative geometry, not a randomized algorithm causal fraction.',
        'Fixed seed is the unchanged stored BF16 decoder31-output tensor, verified by original path/SHA and native applied norm. It is not the control policy gradient or an operator norm.',
        'Natural target PG uses the same old diagnostic coefficients with each actual model log-probabilities. Differences include the output loss direction; they cannot substitute the fixed-seed comparison.',
        'Layer/band gain products and norm-identity residual cosines are dependent finite accounting quantities, not independent causal shares or evidence to change official HF/PEFT/VERL code.',
        'B4 Gram increments use FP32 accumulated-gradient subtraction. Reconstruction residuals remain reported.',
        'These measurements isolate an important internal-sensitivity difference but do not establish which historical update component created it or accept a repair.'
    ],
    optimizer_updates=0, scheduler_updates=0, DT_calls=0, generation_calls=0,
    validation_score_calls=0, GRPO_algorithm_calls=0, rootcause_repair_verified=False,
    accepted_repair=None, deployed_repair=None)
path = HERE / 'saved-control-propagation-summary-v80.json'
path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps(dict(path=str(path.resolve()), fixed_seed=report['fixed_seed_parameter_VJP'],
                      natural_target=report['same_coefficients_natural_target_PG'],
                      band_gains=bands, accumulation=report['fixed_seed_accumulation']), ensure_ascii=False))

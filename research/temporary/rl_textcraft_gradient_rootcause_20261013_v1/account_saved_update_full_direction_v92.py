"""Account completed model-only native replay of saved update outcomes.

All calculations read saved native records; no model, backward or update call.
The common incoming adjoint is the actual complete original policy-loss one.
"""
from datetime import datetime
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
source = HERE / 'account_upper_partition_native_v85.py'
expected = '306e89e45abd6f056aae53ff3cc70b57d2982555a222fd411b0915085d60cfa7'
assert hashlib.sha256(source.read_bytes()).hexdigest() == expected
env = {'__file__': __file__}
exec(compile(source.read_text().split('\nmanifest =', 1)[0], str(source), 'exec'), env)
manifest = json.loads((HERE / 'prepare_saved_update_full_direction_v90-dt.json').read_bytes())['report']
specs = {c['name']: c for c in manifest['cases']}
status, cases = env['account']('native-full-policy-saved-update-feedback-v90-status-final.json', specs)
assert status['job']['pid'] == 2087993 and status['job']['birth'] == 1792023421.07
original = json.loads((HERE / 'completed-full-policy-seed-bindings-v89.json').read_bytes())['80']['value']
bindings = []
for item in status['state']['diagnostic_parameter_partition_bindings']:
    spec = specs[item['case']]
    wanted = spec['diagnostic_parameter_partition']['rank_artifacts'][str(item['rank'])]
    assert item['artifact_SHA256'] == wanted['SHA256']
    assert item['base_trainable_SHA256'] == wanted['expected_base_trainable_SHA256']
    assert item['result_trainable_SHA256'] == wanted['expected_result_trainable_SHA256']
    assert item['selected_parameter_count'] == 496
    assert item['scheduler_before'] == item['scheduler_after'] == spec['checkpoint']
    assert item['optimizer_steps_before'] == item['optimizer_steps_after'] == [float(4 * spec['checkpoint'])]
    assert not item['unexpected_keys']
    bindings.append(item)
assert len(bindings) == 12
seed_checks = []
for name, value in cases.items():
    assert len(value['seeds']) == 16
    for seed in value['seeds']:
        old = next(v for v in original['files'] if (v['rank'], v['micro']) == (seed['rank'], seed['micro']))
        assert seed['shape'] == old['shape'] and seed['dtype'] == old['dtype']
        assert seed['source_SHA256'] == old['SHA256']
        assert seed['applied_norm'] == old['applied_norm']
        seed_checks.append(dict(case=name, rank=seed['rank'], micro=seed['micro'], SHA256=seed['source_SHA256'], applied_minus_saved_norm=0.0))
    value['scope'] = 'Native parameter VJP for the same unchanged actual whole-loss decoder31 adjoint; not a natural post-update PG'
baseline = cases['old80_before']
rows = []
for name, value in cases.items():
    row = dict(case=name, global_parameter_VJP=value['parameter_VJP_norm'],
        ratio_to_same_session_original80=value['parameter_VJP_norm'] / baseline['parameter_VJP_norm'],
        relative_change_to_same_session_original80=value['parameter_VJP_norm'] / baseline['parameter_VJP_norm'] - 1,
        B4_increment_RSS=value['B4_increment_energy_sqrt'],
        B4_direction_factor=value['B4_directional_accumulation_factor'],
        Gram_reconstruction_difference=value['Gram_reconstruction_difference'],
        hidden_boundary_input_ratios_to_original80={
            str(layer): value['boundaries'][f'model.layers.{layer}']['input'] / baseline['boundaries'][f'model.layers.{layer}']['input']
            for layer in (0, 8, 16, 24, 29, 30, 31)},
        parameter_origin=specs[name]['saved_native_parameter_origin'])
    rows.append(row)
report = dict(recorded_at=datetime.now().astimezone().isoformat(), status='completed_saved_native_update_whole_direction_feedback',
    scope=__doc__, job=status['job'], cases=cases, contrasts=rows, actual_parameter_bindings=bindings,
    actual_seed_checks=seed_checks, source=dict(path=str(source), SHA256=expected),
    new_model_calls=0, new_backward_calls=0, new_optimizer_updates=0, new_DT_calls=0,
    rootcause_repair_verified=False, limits=manifest['limits'] + [
        'Feedback is conditional on an unchanged actual whole-loss direction and the original80 optimizer history. It does not itself prove the from-scratch trajectory.',
        'Native original v47 and v69 outcomes are distinct saved repeats; both remain separate. Their variation is not assigned an invented statistical tolerance.',
        'Parameter/load/seed assertions verify the diagnostic intervention; norm reductions alone are not root-cause acceptance.'
    ])
dest = HERE / 'completed-saved-native-update-whole-direction-v92.json'
dest.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps(dict(path=str(dest.resolve()), contrasts=[{k:v for k,v in r.items() if k != 'parameter_origin'} for r in rows]), ensure_ascii=False))

"""Account the completed actual full-loss 2x2 derivative comparison, CPU only."""
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
source = HERE / 'account_upper_partition_native_v85.py'
expected = '306e89e45abd6f056aae53ff3cc70b57d2982555a222fd411b0915085d60cfa7'
assert hashlib.sha256(source.read_bytes()).hexdigest() == expected
env = {'__file__': __file__}
exec(compile(source.read_text().split('\nmanifest =', 1)[0], str(source), 'exec'), env)
names = ('full40_native', 'full80_native', 'full_weights80_seed40', 'full_weights40_seed80')
status, cases = env['account']('native-full-policy-direction-exchange-v86-status-final.json', names)
assert status['job']['pid'] == 1806691 and status['job']['birth'] == 1792020784.77
bindings = json.loads((HERE / 'completed-full-policy-seed-bindings-v89.json').read_bytes())
checks = []
for name, step in (('full_weights80_seed40', 40), ('full_weights40_seed80', 80)):
    original = bindings[str(step)]['value']
    captured = cases[f'full{step}_native']['seeds']
    replayed = cases[name]['seeds']
    assert len(original['files']) == len(captured) == len(replayed) == 16
    for seed in replayed:
        key = f"{seed['rank']}:{seed['micro']}"
        old = next(v for v in captured if (v['rank'], v['micro']) == (seed['rank'], seed['micro']))
        file = next(v for v in original['files'] if (v['rank'], v['micro']) == (seed['rank'], seed['micro']))
        assert seed['shape'] == old['shape'] == file['shape']
        assert seed['dtype'] == old['dtype'] == file['dtype']
        assert seed['applied_norm'] == old['applied_norm'] == file['applied_norm']
        assert seed['source_SHA256'] == original['hashes'][key] == file['SHA256']
        checks.append(dict(case=name, rank=seed['rank'], micro=seed['micro'], source_SHA256=seed['source_SHA256'], applied_minus_original_norm=0.0))
for name, value in cases.items():
    value['scope'] = 'Original complete first-global64 policy gradient at rho1' if name.endswith('_native') else 'Unchanged whole-loss decoder31-adjoint VJP; not a new natural policy gradient'
    value['decoder31_adjoint_RSS'] = value['boundaries']['model.layers.31']['output']
    value['conditional_parameter_gain'] = value['parameter_VJP_norm'] / value['decoder31_adjoint_RSS']
matrix = [[cases['full40_native']['parameter_VJP_norm'], cases['full_weights40_seed80']['parameter_VJP_norm']],
          [cases['full_weights80_seed40']['parameter_VJP_norm'], cases['full80_native']['parameter_VJP_norm']]]
gains = [matrix[1][i] / matrix[0][i] for i in (0, 1)]
natural = {str(step): {k: cases[f'full{step}_native'][k] for k in ('parameter_VJP_norm', 'decoder31_adjoint_RSS', 'B4_increment_energy_sqrt', 'B4_directional_accumulation_factor', 'Gram_reconstruction_difference')} for step in (40, 80)}
report = dict(recorded_at=datetime.now().astimezone().isoformat(), status='completed_native_full_direction_accounting_not_repair', job=status['job'],
    scope=__doc__, cases=cases, seed_identity_checks=checks, actual_seed_receipts=bindings,
    matrix_rows_weight=[40, 80], matrix_columns_actual_loss_seed=[40, 80], parameter_VJP_norm_matrix=matrix,
    same_whole_loss_direction_weight40_to80_gains=gains, natural_original81_PG=natural,
    natural_parameter_norm_ratio=matrix[1][1] / matrix[0][0],
    source=dict(path=str(source), SHA256=expected), new_model_calls=0, new_backward_calls=0, optimizer_updates=0,
    conclusions=[
        'Internal sensitivity increases for both actual complete loss directions, not only for the earlier selected-target probe.',
        'On this fixed whole batch, B4 directional accumulation weakens at weight80; greater microbatch alignment is contradicted here.',
        'The natural old81 whole PG grows only about1.067 times between these two weights. Changing loss force and propagation counteract, so conditional amplification is not the complete historical training-norm trajectory.'
    ], limits=[
        'The original81 complete first-global64, full256-whitened advantages and original1/16 reduction are fixed; weight40 is not the historical41 rollout.',
        'Each seed remains its actual original BF16 tensor with no normalization or token selection. Directional gains are not operator norms or independent causal shares.',
        'Old-trajectory derivatives cannot establish a from-scratch counterfactual, nor accept a repair on their own.'
    ], rootcause_repair_verified=False, accepted_repair=None, deployed_repair=None)
dest = HERE / 'completed-full-policy-direction-accounting-v89.json'
dest.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps(dict(path=str(dest.resolve()), matrix=matrix, same_direction_gains=gains, natural=natural), ensure_ascii=False))

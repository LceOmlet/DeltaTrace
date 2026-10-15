"""Compare already measured temporal target gradients across all four minibatches.

This is arithmetic on original native readouts, not a new loss, normalizer,
training plan or a complete causal decomposition of the growing gradient.
"""
from collections import defaultdict
from datetime import datetime
import csv
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
source = HERE / 'summarize_historical_target_derivatives_v4.json'
native = json.loads(source.read_bytes())['report']
contract_path = HERE / 'audit_historical_target_contract_v105-dt.json'
contract = json.loads(contract_path.read_bytes())['report']
assert contract['all_native_target_values_equal_max_alpha']
assert contract['max_abs_carrier_coefficient_difference'] == 0.
assert all(contract['original81_input_fields_equal'].values())
rows = []
by_stage = defaultdict(list)
for original in native['rows']:
    assert original['forward_point_matches_between_weight_views_by_rank'] == [True, True]
    row = dict(iteration=original['iteration'], checkpoint=original['checkpoint'], mini=original['mini'], views={})
    for view in ('B4', 'global64'):
        measured = original['parameter_distribution'][view]
        logit = original['output_score']['groups'][view]['output_logit_L2_bounds']
        row['views'][view] = dict(native_target_parameter_gradient=measured['norm'],
            native_logit_gradient_L2_bounds=logit,
            conditional_parameter_per_logit_gain_bounds=measured['effective_parameter_to_logit_gain_bounds'])
    row['B4_over_common_coefficient_target_gradient'] = row['views']['B4']['native_target_parameter_gradient'] / row['views']['global64']['native_target_parameter_gradient']
    rows.append(row)
    by_stage[row['iteration']].append(row)
stages = {}
for stage, items in by_stage.items():
    assert len(items) == 4 and {row['mini'] for row in items} == {0, 1, 2, 3}
    views = {}
    for view in ('B4', 'global64'):
        g = math.fsum(row['views'][view]['native_target_parameter_gradient'] for row in items) / 4
        lo, hi = [math.fsum(row['views'][view]['native_logit_gradient_L2_bounds'][i] for row in items) / 4 for i in (0, 1)]
        views[view] = dict(mean_of_four_native_target_gradient_norms=g,
            mean_logit_norm_bounds=[lo, hi], ratio_of_mean_norms_bounds=[g / hi, g / lo],
            scope='Ratio of mean minibatch norms; not the norm of a single full256 gradient, an operator norm or independent causal shares.')
    stages[str(stage)] = views
comparisons = []
for earlier, later in ((6, 41), (41, 81), (6, 81)):
    for view in ('B4', 'global64'):
        a, b = stages[str(earlier)][view], stages[str(later)][view]
        comparisons.append(dict(iterations=[earlier, later], view=view,
            target_gradient_norm_mean_ratio=b['mean_of_four_native_target_gradient_norms'] / a['mean_of_four_native_target_gradient_norms'],
            mean_logit_norm_ratio_bounds=[b['mean_logit_norm_bounds'][0] / a['mean_logit_norm_bounds'][1], b['mean_logit_norm_bounds'][1] / a['mean_logit_norm_bounds'][0]],
            conditional_gain_ratio_bounds=[b['ratio_of_mean_norms_bounds'][0] / a['ratio_of_mean_norms_bounds'][1], b['ratio_of_mean_norms_bounds'][1] / a['ratio_of_mean_norms_bounds'][0]]))
log_path = Path('F:/DeltaTraceBackups/plots/gradient-rootcause-20261014/native-gradient-growth-85.csv')
with log_path.open(encoding='utf-8-sig', newline='') as f:
    logged = list(csv.DictReader(f))
observed = [row for row in logged if int(row['iteration']) in (6, 41, 80, 81, 82, 83, 84, 85)]
report = dict(recorded_at=datetime.now().astimezone().isoformat(), scope=__doc__,
    sources=[dict(path=str(path.resolve()), SHA256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in (source, contract_path, log_path)],
    original81_target_contract=dict(native_alpha=contract['native_alpha'], target_policy_tokens=contract['target_policy_tokens'],
        target_rows=contract['target_rows'], all_target_advantages_equal_native_alpha=True, native_V_at_targets_zero=True,
        carrier_coefficients_exactly_original=True),
    original_target_loss_at_rho1='L_target=-(1/16)*sum_j alpha/N_j * sum_{i in target policy tokens of B4 j} log pi_i; entropy/KL remain separate original actor terms.',
    rows=rows, stages=stages, comparisons=comparisons, original_logged_rows=observed,
    interpretation=['The original target signal becomes much larger across all four retrospective native minibatches, not just the first one.',
        'The already completed common-token-coefficient diagnostic still shows a large rise. In exact arithmetic its common alpha/N_global factor cancels in parameter-gradient/logit-gradient quotients, so unknown early failure counts cannot by themselves explain that quotient. Native rounding and direction/input differences still remain.',
        'The quotient mixes native weights, actual forward states, incoming score directions and gradient cancellation. It localizes propagation/directional geometry; it does not prove the reason it changed.',
        'Logged native81 is1.401 and native82 is0.502. The completed native old81 after-update PG decline is compatible with this local retreat; it cannot settle the earlier accelerating growth interval.',
        'No claim that the pinned official1/16 is a coding bug, no global-token normalizer deployment and no root-repair acceptance follow from these records.'],
    model_calls=0, backward_calls=0, optimizer_calls=0, rootcause_repair_verified=False,
    limits=native['limits'] + [
        'The global64 view is an already completed diagnostic common-coefficient transport, not the accepted training objective or a proposed normalization fix.',
        'Early historical successful IDs/masks are native but failure denominator counts and printed alpha are approximate. The original81 target coefficients and inputs are now checked exactly.',
        'Minibatch indices across epochs are not task-matched states. Existing matched15-history controls remain a distinct causal input comparison.',
        'Native entropy/KL/clip scalar matches support the paired readout but do not prove every activation value is equal. No invented numerical acceptance threshold is used.',
        'A gain bound or algebraic factorization does not uniquely identify the historical cause. It is combined with the independent native Adam-energy history and original coefficient/source audit.'])
path = HERE / 'historical-target-stage-gain-accounting-v106.json'
path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps(dict(path=str(path.resolve()), stages=stages, comparisons=comparisons), ensure_ascii=False))

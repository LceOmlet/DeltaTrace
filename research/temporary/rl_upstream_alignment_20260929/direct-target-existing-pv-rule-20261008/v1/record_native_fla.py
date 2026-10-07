"""Bind original native-FLA measurements; no credit rule or tolerance."""
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]


def binding(path):
    return dict(path=path.as_posix(), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


records = {}
sources = []
for name, folder, prefix in (
    ('joint', 'native-output-results', 'native-output'),
    ('single', 'native-single-output-results', 'native-single-output'),
    ('orders', 'native-memory-orders-results', 'native-memory-orders'),
):
    path = HERE / folder / 'results/result.json'
    result = json.loads(path.read_bytes())
    launch = json.loads((HERE / folder / 'launch.json').read_bytes())
    observation_path = sorted(HERE.glob(prefix + '-observation-*.json'))[-1]
    observation = json.loads(observation_path.read_bytes())
    assert result['phase'] == 'complete' and len(result['groups']) == 4
    assert not observation['alive_same_birth']
    assert observation['text_same_birth'] and not any(observation['releases'])
    assert result['pid'] == launch['pid'] and result['birth'] == launch['birth']
    assert all(g['initial_state_pair_equal'] for g in result['groups'])
    physical = dict(re.findall(r'\|\s*([45])\s+MetaX C550[^\n]*\n\|[^\n]*?\s(\d+)/65536 MiB', observation['physical']))
    records[name] = dict(launch=launch, native_owner=result['owner'],
        verified_native_source_sha256=result['verified_native_source_sha256'],
        scope=result['scope'], summary=result['summary'], groups=result['groups'],
        max_recorded_pss_bytes=max(g['pss_bytes'] for g in result['groups']),
        group_end_max_live_allocated_bytes=max(g['group_end_allocated_bytes'] for g in result['groups']),
        physical_mib_at_terminal_poll={k:int(v) for k,v in physical.items()},
        driver_exited=True)
    sources.extend([binding(path), binding(HERE / folder / 'launch.json'),
                    binding(HERE / folder / 'transport.json'), binding(observation_path)])

gdn = json.loads((REPO / 'experiments/rl/results_current_extreme_gdn_20261008.json').read_bytes())
branch = gdn['ranks'][0]['gdn_subops']['30']
skip_z = branch['residual_skip_single_delta'] + branch['z_branch_single_delta']
single = records['single']['summary']
joint = records['joint']['summary']
measured = dict(
    joint_finite_minus_native=joint['finite_joint_contraction']-joint['native_output_effect'],
    joint_coefficients_times_actual_single_delta=single['finite_joint_contraction'],
    native_actual_single_output_effect=single['native_output_effect'],
    original_finite_on_actual_single_endpoints=single['original_single_finite_contraction'],
    original_single_finite_minus_native=single['original_single_finite_contraction']-single['native_output_effect'],
    joint_coefficients_single_effect_error=single['finite_joint_contraction']-single['native_output_effect'],
    before_FLA_original_gate_point=branch['points'][0]['coefficient_times_single_delta'],
    after_FLA_saved_joint_point=branch['points'][1]['coefficient_times_single_delta'],
    after_FLA_native_actual_single_point=skip_z+single['native_output_effect'],
    after_FLA_native_actual_single_BF16_point=skip_z+single['native_output_cast_BF16_effect'],
    after_FLA_original_finite_actual_single_point=skip_z+single['original_single_finite_contraction'])
orders=records['orders']['summary']
measured['original_memory_orders_single_effect']=orders['memory_orders']
measured['original_memory_orders_after_skip_and_z']={k:skip_z+v for k,v in orders['memory_orders'].items()}
measured['original_averaged_orders_single_effect']=orders['original_averaged_coefficients_times_single_delta']
measured['original_forward_order_single_effect_error']=orders['memory_orders']['forward']-single['native_output_effect']
measured['original_reversed_order_single_effect_error']=orders['memory_orders']['reversed']-single['native_output_effect']
head_errors = [dict(head_start=g['head_start'],
    joint_coefficients_times_single_delta=g['finite_joint_contraction'],
    native_single_effect=g['native_output_effect'],
    difference=g['finite_joint_contraction']-g['native_output_effect'])
    for g in records['single']['groups']]
post_owner = HERE / 'native-output-post-owner.json'
owner_resolution = json.loads(post_owner.read_bytes())
assert not owner_resolution['cuda_initialized']
sources.append(binding(post_owner))
for name in ('results_current_extreme_gdn_20261008.json',
             'results_current_extreme_fla_precision_20261008.json',
             'results_existing_PV_author_curves_20261008.json',
             'results_memory_capacity_20261008.json'):
    sources.append(binding(REPO / 'experiments/rl' / name))
sources.append(binding(REPO / 'experiments/rl/upstream.lock'))
out = dict(status='Current extreme token: joint finite allocation differs from its actual single-deletion effect at GDN30; no credit repair deployed',
    case_binding=gdn['case_binding'], measured=measured, head_errors=head_errors,
    original_native_operator_diagnostics=records, post_completion_owner_resolution=owner_resolution,
    sources=sources, limitations=[
        'Joint contraction accuracy checks the joint endpoint identity. It does not establish individual token deletion accuracy or official FA/FLA tolerance acceptance.',
        'The single replay uses saved actual single-deletion Q/K/V/raw_g/beta and the captured factual incoming state at the unchanged causal chunk boundary. All retained factual operands match the saved joint factual endpoint exactly. No alternate tool observations or estimated initial states are created.',
        'This isolates the local GDN30 allocation error for the current worst token. It is not a population error rate or a proof that every negative advantage is invalid.',
        'The saved joint coefficients came from the original B4 compiled operation; the native replay uses the selected paired B2/eight-head operands. The prior same-operand eager/compiled control is linked separately.',
        'The author cumulative deletion/RISE/MAS result remains the linked earlier measurement. This supplemental token diagnosis does not replace that evaluation.',
        'Group-end Torch allocation and PSS are observations, not continuously measured physical peaks. Native operator times include first-call compilation and are not production throughput.',
        'Post-completion CPU import resolution is identified as such; it is not presented as an omitted live-process import dump.',
        'The original forward memory order is closer on this local token effect; this does not establish that changing every layer to that existing rule improves full token vectors, author curves or training. No such profile was deployed.',
    ], official_tolerance_claim=False, credit_repaired=False, production_profile_changed=False,
    formal_restart=False, text_update_released=False,
    unchanged_configuration=dict(lora_rank=8,lora_alpha=16,actor_microbatch_per_gpu=4,DT_minibatch_per_gpu=4,context_limit=32768),
    operations=dict(model_load=0, full_DT=0, rollout=0, training_backward=0, optimizer=0, checkpoint_restore=0))
target = REPO / 'experiments/rl/results_current_extreme_native_fla_20261008.json'
target.write_bytes((json.dumps(out, ensure_ascii=False, indent=2)+'\n').encode('utf8'))
print(json.dumps(dict(receipt=binding(target), measured=measured, head_errors=head_errors), indent=2))

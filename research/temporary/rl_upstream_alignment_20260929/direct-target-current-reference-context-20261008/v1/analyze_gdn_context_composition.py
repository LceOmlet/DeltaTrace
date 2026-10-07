"""Account for the current token's measured GDN30 sign change from saved results.

Pure CPU arithmetic on completed, identity-matched original diagnostics.
No reference implementation, numerical threshold or replacement credit.
"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
AUDIT = HERE.parents[1]
BASE = AUDIT/'direct-target-existing-pv-rule-20261008/v1'


def read(path):
    return json.loads(path.read_bytes())


def bind(path):
    raw=path.read_bytes()
    return dict(path=path.relative_to(REPO).as_posix(),bytes=len(raw),
                sha256=hashlib.sha256(raw).hexdigest())


paths = dict(gdn=BASE/'gdn-v2-results/results/rank0.json',
    joint=BASE/'native-output-results/results/result.json',
    single=BASE/'native-single-output-results/results/result.json',
    orders=BASE/'native-memory-orders-results/results/result.json',
    jacobian=REPO/'experiments/rl/results_native_factual_jacobian_20261008.json',
    official=REPO/'experiments/rl/results_current_extreme_official_block_dtype_20261008.json',
    rejected_V=BASE/'factual-v-analysis.json',
    causal=REPO/'experiments/rl/results_credit_causal_diagnosis_20261008.json')
data={key:read(path) for key,path in paths.items()}
gdn=data['gdn']['gdn_subops']['30']
case=data['gdn']['geometry']['candidate']
assert case['packed_slot']==2883 and case['token_id']==198
assert case['traj_uid']=='f0f85f5c-74b4-4670-a074-99a3a09dbb98'
artifacts={v['path']:v['sha256'] for v in gdn['operand_artifacts']}
assert len(artifacts)==4
for key in ('joint','single','orders'):
    assert data[key]['phase']=='complete'
    assert {v['artifact']['path']:v['artifact']['sha256'] for v in data[key]['groups']}==artifacts
    assert not data[key]['production_profile_changed'] and not data[key]['credit_repaired']
assert all(c['equal'] for c in gdn['factual_endpoint_checks'].values())
assert data['official']['official_checks']['total_passed']==22
assert not data['official']['formal_state']['credit_repaired']

skip=gdn['residual_skip_single_delta']
z=gdn['z_branch_single_delta']
constant=skip+z
terms={key:sum(group['terms'][key] for group in gdn['fla_groups'])
       for key in ('q','k','v','g','beta')}
joint_on_single=sum(terms.values())
single=data['single']['summary']
joint=data['joint']['summary']
points={p['name']:p['coefficient_times_single_delta'] for p in gdn['points']}
rows=[dict(name='Recorded gate output, residual and z branches included',
           value=points['after_gdn_norm_and_silu_gate']),
      dict(name='Recorded joint finite FLA coefficients contracted with actual single-deletion operand differences; residual and z included',
           value=points['after_finite_fla_before_qk_l2']),
      dict(name='Same residual and z plus original finite FLA evaluated on the actual single-deletion pair; arithmetic diagnostic only',
           value=constant+single['original_single_finite_contraction']),
      dict(name='Same residual and z plus original native FLA output difference on actual single-deletion pair; arithmetic diagnostic only',
           value=constant+single['native_output_effect'])]

result=dict(
    status='Completed CPU accounting: current GDN30 joint-versus-single context difference explains the measured sign change at this sub-operation; no repair deployed.',
    case=case,source_sha256=data['gdn']['geometry']['source_sha256'],
    original_runner=data['gdn']['finite_owner'],artifacts=gdn['operand_artifacts'],
    unchanged_measured_branches=dict(residual_skip=skip,z_branch=z,sum=constant),
    original_joint_finite_terms_times_actual_single_delta=terms,
    original_joint_finite_terms_sum=joint_on_single,
    source_identity_controls=dict(factual_operands_equal=True,same_four_artifact_hashes=True,
        fixed_original_output_cotangent=True,actual_single_incoming_state_is_shared_factual=True),
    owner_self_comparisons=dict(
        joint_endpoints=dict(original_finite=joint['finite_joint_contraction'],
            native=joint['native_output_effect'],
            difference=joint['finite_joint_contraction']-joint['native_output_effect']),
        actual_single_endpoints=dict(original_finite=single['original_single_finite_contraction'],
            native=single['native_output_effect'],
            difference=single['original_single_finite_contraction']-single['native_output_effect']),
        joint_coefficients_on_single_delta=dict(original_finite=joint_on_single,
            native_actual_single=single['native_output_effect'],
            difference=joint_on_single-single['native_output_effect']),
        scope='Different endpoint contexts, measured with the same original saved output cotangent. These descriptive differences are not official tolerance assertions.'),
    phase_rows=rows,
    diagnostic_closure=dict(
        recorded_after_FLA_minus_reconstructed=points['after_finite_fla_before_qk_l2']-(constant+joint_on_single),
        gate_implied_native_o=points['after_gdn_norm_and_silu_gate']-constant,
        replay_native_o_BF16=single['native_output_cast_BF16_effect'],
        gate_implied_minus_native_replay_BF16=(points['after_gdn_norm_and_silu_gate']-constant)-single['native_output_cast_BF16_effect']),
    interpretation=[
        'The original finite FLA is close to its own native endpoint difference for each measured endpoint pair. The measured large discrepancy appears when a joint-pair coefficient decomposition is used for a factual single-deletion perturbation.',
        'Adding the unchanged measured residual and z contributions locates the positive-to-negative change at finite FLA inside GDN30. The arithmetic single-pair rows are diagnostics, not a new production path or full-model credit.',
        'The new coincident-endpoint official dtype assertions pass independently. They cannot validate this joint-versus-single context approximation.',
        'Replacing only the V branch was already tested and rejected on the full vector. This accounting does not justify another isolated branch substitution.',
        'Do not infer an all-token error rate, a parameter-gradient share, or all learning degradation from this selected point. The original author cumulative deletion/RISE/MAS receipts remain separate.'],
    operations=dict(model=0,DT=0,backward=0,optimizer=0,rollout=0,
        production_modified=False,new_query_policy_implemented=False,credit_repaired=False),
    evidence={key:bind(path) for key,path in paths.items()})
target=REPO/'experiments/rl/results_gdn_context_composition_20261008.json'
assert not target.exists()
target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(receipt=bind(target),owner_self_comparisons=result['owner_self_comparisons'],
    phase_rows=rows,diagnostic_closure=result['diagnostic_closure']),ensure_ascii=False))

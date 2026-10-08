"""Audit composition algebra against preserved owners and real saved vectors.

Symbolic identities explain a possible mechanism; they do not estimate its
frequency or establish a sufficient repair. No new numerical tolerance,
conservation requirement, candidate, model query or training path is added.
"""
import hashlib
import json
from pathlib import Path

import sympy as sp

HERE = Path(__file__).resolve().parent


def ref(path):
    raw = path.read_bytes()
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))


def main():
    owner_path = HERE/'actual-finite-scalar-owner-sources.json'
    owner = next(f for f in json.loads(owner_path.read_bytes())['files']
                 if f['module'] == 'compiled_swiglu_secant')
    assert hashlib.sha256(owner['source'].encode()).hexdigest() == owner['sha256']
    assert 'mg=m_product*(up0+up1)*0.5' in owner['source']
    assert 'mu=m_product*(silu0+silu1)*0.5' in owner['source']
    u, s, du, ds, joint_u, joint_s = sp.symbols('u_F s_F delta_u_i delta_s_i Delta_u_joint Delta_s_joint')
    true = sp.expand(u*s-(u-du)*(s-ds))
    shared = (s-joint_s/2)*du + (u-joint_u/2)*ds
    error = sp.expand(shared-true)
    single_background_error = sp.simplify(error.subs({joint_u: du, joint_s: ds}))
    local_coordinate_effect_sum = s*du + u*ds
    independent_coordinate_error = sp.expand(local_coordinate_effect_sum-true)
    assert single_background_error == 0
    assert independent_coordinate_error == du*ds
    # Source-specific M/H coupling in the existing GDN recurrence; no model
    # assumptions, scalar sampling, allocation or fitted residual is involved.
    M, H, B, dM, dH, dB = sp.symbols('M_F H_F B_F delta_M_i delta_H_i delta_B_i', commutative=False)
    recurrence = sp.expand(M*H+B-((M-dM)*(H-dH)+(B-dB)))
    expected = M*dH+dM*H-dM*dH+dB
    assert sp.expand(recurrence-expected) == 0
    ledger_path = HERE/'conditional-mixers-collection-v1/full-credit-endpoint-ledger.json'
    ledger = json.loads(ledger_path.read_bytes())
    original = {r['traj_uid']: r for r in ledger['rows'] if r['label'] == 'original'}
    summaries = {}
    for label in sorted({r['label'] for r in ledger['rows']}):
        for primary in (True, False):
            rows = [r for r in ledger['rows'] if r['label'] == label and r['primary'] == primary]
            values = [r['endpoint_ledger'] for r in rows]
            assert all(r['factual_logp'] == original[r['traj_uid']]['factual_logp'] for r in rows)
            assert all(r['endpoint_ledger']['reference_target_logp'] ==
                       original[r['traj_uid']]['endpoint_ledger']['reference_target_logp'] for r in rows)
            summaries[label+('_primary' if primary else '_extra_tail')] = dict(
                trajectories=len(rows), states=len({r['state'] for r in rows}),
                unchanged_factual_and_reference_scores=len(rows),
                joint_target_difference_positive=sum(v['joint_target_difference'] > 0 for v in values),
                signed_source_sum_negative=sum(v['recomputed_source_sum_FP64'] < 0 for v in values),
                recorded_finite_source_sum_minus_joint_difference_range=[
                    min(v['source_sum_minus_joint_difference'] for v in values),
                    max(v['source_sum_minus_joint_difference'] for v in values)],
                source_sum_vs_owner_accumulation_maxabs=max(abs(v['recomputed_source_sum_FP64']-
                    v['owner_policy_credit_signed_sum']) for v in values),
                necessary_probability_bound_violations=sum(r['probability_bound_violations'] for r in rows))
    result = dict(scope=__doc__, inputs=[ref(owner_path), ref(ledger_path),
                    ref(HERE/'saved-propagation-order.json'), ref(Path(__file__))],
        owner={k: v for k, v in owner.items() if k != 'source'},
        symbolic_product_audit=dict(
            exact_single_deletion_product_difference=str(true),
            actual_shared_symmetric_owner_product_allocation=str(shared),
            product_only_background_residual=str(error),
            single_background_product_residual=str(single_background_error),
            independent_exact_coordinate_effect_sum=str(local_coordinate_effect_sum),
            independent_coordinate_composition_residual=str(independent_coordinate_error),
            interpretation='This product-only identity uses actual single-deletion delta_s as input. '
                'The full current owner also approximates SiLU and upstream/downstream propagation, so this is '
                'not a decomposition of the measured whole-token error. It explains why a locally exact '
                'coordinate intervention cannot be composed as if all coordinates came from independent source actions.'),
        symbolic_GDN_recurrence_audit=dict(exact_single_source_difference=str(expected),
            noncommutative_identity_residual=str(sp.expand(recurrence-expected)),
            coupled_term='-delta_M_i * delta_H_i',
            interpretation='Existing local conditional windows retain their own coupled terms. The unresolved '
                'question is the coherent source-specific change across downstream positions/layers, not a claim '
                'that the original FLA kernel omitted the recurrence term.'),
        sufficient_composition_condition='At each layer l and source i: B_l(delta_i h_l) = delta_i h_(l+1), '
            'with the same single intervention throughout. A correct local rule for separate coordinate deletions '
            'or a correct joint endpoint secant does not establish this source-specific condition.',
        saved_real_vector_summaries=summaries,
        interpretations=[
            'All compared factual/reference model scores are identical. The candidate changes are therefore '
            'already in the finite attribution vector, before the unchanged expm1 reward conversion.',
            'A sum of individual deletion effects is not generally the joint deletion difference. The ledger '
            'is descriptive; no conservation-based rejection threshold, rescaling or correction is introduced.',
            'The failed composition and source-specific coupling algebra exclude the assertion that exact '
            'local coordinate rules are sufficient for exact whole-token deletion. They do not quantify '
            'which real suboperation dominates, establish a new numerical kernel bug, or validate a repair.',
            'The accepted exact-token-counterfactual Q/V/A premise remains unchanged. This audit concerns '
            'the practical DT finite approximation, not a new value model or an environment simulator.',
            'Do not implement a full source-by-time hidden-state bank. The naive B4*32768*32768*4096 BF16 '
            'bank alone is 32 TiB before causality reduction. This rejects that storage design, not every '
            'possible compressed or structured computation. No such bank has been allocated.'
        ],
        naive_full_source_bank_bytes=4*32768*32768*4096*2,
        new_candidate_derived=False, candidate_accepted=False, production_modified=False,
        extreme_attribution_repaired=False, method_changed=False, official_tolerance_changed=False,
        operations=dict(new_model_queries=0, new_DT=0, optimizer=0, rollout=0, CUDA_initialized=False))
    path = HERE/'coupled-background-audit.json'
    path.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(receipt=ref(path), symbolic_single_background_residual=str(single_background_error),
        symbolic_recurrence_residual=str(sp.expand(recurrence-expected)),
        summaries={k: {n: v for n, v in row.items() if n in ('trajectories', 'signed_source_sum_negative',
            'necessary_probability_bound_violations')} for k, row in summaries.items()}, model_queries=0)))


if __name__ == '__main__':
    main()

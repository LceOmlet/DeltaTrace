"""Record the narrow probability audit, without changing the accepted RL plan."""
import hashlib,json
from datetime import datetime,timezone,timedelta
from pathlib import Path

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
ROOT='/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922'
read=lambda name:json.loads((HERE/name).read_bytes())
point=read('step24-extreme-source-token-CPU.json')
mapping=read('step24-probability-mapping-CPU-audit.json')
saved=read('step24-saved-credit-CPU-audit.json')
loaded=read('step24-existing-passive-reports.json')['version_inspection']['step24-readonly-loaded-workers.json']
logs=read('step24-existing-log-audit.json')
now=datetime.now(timezone(timedelta(hours=8))).isoformat()
owner=HERE/'sources/candidates/textcraft-gate-capture-host-20261012-v1/deltatrace/clean/qwen35'
files=['qwen35_dense_finite_runner.py','qwen35_answer_finite.py','native_target_logit_rows.py',
       'qwen35_decoder_finite.py','compiled_swiglu_secant.py','compiled_logprob_seed.py','compiled_finite_rules.py']
sources={ROOT+'/candidates/textcraft-gate-capture-host-20261012-v1/deltatrace/clean/qwen35/'+name:
         hashlib.sha256((owner/name).read_bytes()).hexdigest() for name in files}
sources[str(HERE/'step24-core-owner-source.py')]=hashlib.sha256((HERE/'step24-core-owner-source.py').read_bytes()).hexdigest()
report=dict(recorded_at=now,version='textcraft-step24-counterfactual-probability-audit-20261015-v1',
    status='single_deletion_interpretation_violates_probability_bound_no_native_illegal_probability_observed',
    authorized_scope='Existing iteration24 logs, passive CPU artifacts and actual imported versions; exact-ratio composition. No later training-effect investigation.',
    active_method='User-approved DTPO v3 f726ee72 / deployment e05e23cb; unchanged by this audit.',
    previous_gradient_rootcause_goal_status='paused_not_resumed',
    driver=logs['actual_driver'],actual_loaded_workers=loaded,
    exact_probability_answer=dict(
        convention='d=log p_factual(Y)-log p_deleted_i(Y)',
        exponential='1-exp(-d)=1-p_deleted_i(Y)/p_factual(Y), an exact identity, not a new approximation.',
        raw_reward_identity='E_{Y~p}[r(Y)(1-q(Y)/p(Y))]=sum_Y r(Y)(p(Y)-q(Y)), under the existing normalized-distribution/support assumptions.',
        policy_gradient_scope='With the accepted action-independent deletion baseline, this has the same expected score-function policy gradient as subtracting the exact value baseline; not a claim of identical finite clipped PPO updates.',
        official_group_multiplier='The pinned VERL GRPO owner is called directly. Conditional on independent group peers, replacing r by g gives the exact importance identity for the group-transformed return g. It is not an unbiased identity for the original raw-return Q/V.',
        variance_only='The pinned official masked_var is used. Scaling by a fixed positive constant only rescales a gradient; using the same random batch standard deviation is not generally an unbiased raw-return policy-gradient identity. It preserves nonzero signs and does not alter the input probability ratio.',
        relevance_to_incident='Neither group normalization nor variance scaling created the invalid ratio: the saved pre-exponential d already violates q<=1.',
        primary_theory='https://papers.neurips.cc/paper/9413-hindsight-credit-assignment.pdf',
        official_GRPO_reference='https://arxiv.org/html/2402.03300v3#S4.SS1.SSS2'),
    incident=dict(
        iteration=24,original_record_time='2026-10-15T14:43:24.440421+08:00',
        backsubstitution_assumption='Assume the returned joint-contrast contribution c_i equals log p_factual(Y)-log p_single_deleted_i(Y), then derive q_implied=exp(factual_logp-c_i).',
        actual_single_deletion_probability_measured=False,
        inference_scope='q_implied>1 rejects the exact single-deletion interpretation of c_i at this point. It does not mean that a native model evaluation returned a probability above1, nor identify a faulty finite operator.',
        source=point['source'],source_SHA256=point['source_SHA256'],trajectory_UID=point['trajectory_UID'],
        local_original_record=str(HERE/'step24-extreme-original-record.pt'),
        actual_context_tokens=point['actual_context_tokens'],target_tokens=point['target_token_count'],
        extreme_source_token=point['extreme_source_token'],
        factual_joint_target_logp=point['factual_joint_target_logp'],
        all_source_EOS_reference_joint_target_logp=point['reference_all_source_EOS_joint_target_logp'],
        group_multiplier=saved['steps']['24']['raw_minimum']['reward_multiplier'],
        **point['probability_consistency']),
    original_credit_reproduction=dict(
        owner=saved['credit_owner'],original_iteration24_rank_stats_exactly_equal=all(
            r['exactly_matches_original_logged_raw_groups'] for r in saved['statistics_checks'] if r['iteration']==24),
        owner_QVA_formula_unchanged=True,new_ratio_formula=False),
    saved_iteration24_mapping=dict(
        record_files=mapping['record_files'],unique_traced_trajectories=mapping['unique_traced_trajectories'],
        checks=mapping['native_packing_checks'],
        scope_limit='CPU reconstruction through the exact installed packing owners on the saved inputs; not a replay of missing historical GPU intermediate tensors.',
        bound_violating_trajectories=mapping['bound_violating_trajectories'],
        bound_violating_tokens=mapping['bound_violating_tokens'],
        count_scope='Exact mathematical necessary-bound observations in saved values; includes small violations, not a numerical tolerance, accuracy score or training gate.',
        extreme_endpoint_reconstruction=mapping['extreme_endpoint_reconstruction']),
    confirmed_call_chain=[
        'DirectActionTargetReadout constructs one factual endpoint and an endpoint replacing all eligible non-target policy source tokens with EOS, preserving target/observation IDs.',
        'PackedAnswerTargets selects factual next-token labels and their causal predictor positions. The runner obtains native full-vocabulary log_softmax endpoint scores.',
        'runner.attribute propagates finite coefficients between those two joint endpoints; _token_effect contracts each embedding displacement with its propagated coefficient.',
        'The readout maps that signed contribution vector into source_log_ratios without measuring each factual-minus-single-deleted endpoint.',
        'The unchanged counterfactual.reward_event_token_credit applies g*(-expm1(-d)).'],
    structural_finding=dict(
        confirmed='An exact allocation of a joint all-source contrast is not generally the individual factual-minus-single-deletion contrast. The current vector is explicitly an estimate of the latter, and this case violates its necessary probability bound.',
        actual_owner_example='compiled_swiglu_secant.swiglu_finite_rule uses two-endpoint means: m_g=m_product*(up0+up1)/2 and m_u=m_product*(silu0+silu1)/2. These divide interaction contributions between jointly changed operands.',
        algebra='For a bilinear uv boundary, mean(v)*Delta_u is its symmetric contrast allocation, whereas changing only u at factual v1 gives v1*Delta_u; their difference is -Delta_u*Delta_v/2. Exact conservation of the joint change does not identify individual deletion changes.',
        case_global_contrast=mapping['extreme_row'],
        not_proven='The original iteration24 per-layer endpoints/coefficient tensors and historical pre-update theta23 were not saved by this passive recorder. No first faulty finite operator or quantitative split into interaction allocation versus numerical error has been identified.',
        historical_scope='Earlier saved probability-bound failures exist, but use different historical weights/readouts; they were not used to prove the current case operator cause.',
        repair_status='No fix has been deployed. No clipping, rescaling of d, alternate reference, learned critic, or shadow probability implementation was introduced.'),
    efficiency_causality=dict(
        deployed_patch='7a33e61a / frozen-head backward owner bb46ebe9cb496a10e13539f8605544c550c892007345a8001bb0ff95f2c425b2',
        deployment_unix=1792045484.9960804,scheduler_before_after=23,
        earliest_affected_policy_backward=24,
        incident_DT_computed_before_that_backward=True,
        removed_branch='Only unused frozen vocabulary-weight gradient; native head forward and necessary hidden-gradient expressions unchanged.',
        DT_endpoint_readout_owner=sources[ROOT+'/candidates/textcraft-gate-capture-host-20261012-v1/deltatrace/clean/qwen35/qwen35_answer_finite.py'],
        separate_head_row_selection_candidate='8c9e99e5 was not deployed, confirmed by actual loaded function signatures/code hashes.'),
    pinned_sources=sources,
    receipts={name:dict(path=str(HERE/name),SHA256=hashlib.sha256((HERE/name).read_bytes()).hexdigest()) for name in
        ['step24-existing-log-audit.json','step24-existing-passive-reports.json','step24-saved-credit-CPU-audit.json',
         'step24-extreme-source-token-CPU.json','step24-probability-mapping-CPU-audit.json']},
    training_changes=0,new_model_forward_backward_DT_optimizer_calls=0,new_numerical_tolerance=None,
    no_hyperparameter_or_credit_changes=True,no_training_restart_or_stop=True)
out=REPO/'experiments/rl/results_textcraft_step24_counterfactual_probability_20261015_v1.json'
out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(path=str(out),status=report['status'],recorded_at=now),ensure_ascii=False))

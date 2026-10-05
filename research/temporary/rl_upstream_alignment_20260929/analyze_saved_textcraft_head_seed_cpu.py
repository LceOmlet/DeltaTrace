"""Describe saved owner seed recomputation; no tolerance or credit repair."""
import json
import statistics
from pathlib import Path

from analyze_textcraft_matched_layout import LOCAL, paired_statistics, reduce_replicas
from analyze_textcraft_native_readout import describe, source


FIELDS = ('dt_d', 'native_single_delete_d', 'owner_head_projected_d',
    'owner_factual_head_projected_d', 'head_conditional_mismatch',
    'remainder_joint_propagation_allocation_numerical', 'total_dt_minus_native',
    'factual_head_conditional_mismatch', 'cpu_log_probability_minus_saved_native')


def summarize(rows):
    comparisons = {name: paired_statistics(rows, left, right)
        for name, left, right in (
            ('dt_vs_native', 'dt_d', 'native_single_delete_d'),
            ('joint_head_on_actual_single_logits_vs_native', 'owner_head_projected_d', 'native_single_delete_d'),
            ('dt_vs_joint_head_on_actual_single_logits', 'dt_d', 'owner_head_projected_d'),
            ('factual_head_on_actual_single_logits_vs_native', 'owner_factual_head_projected_d', 'native_single_delete_d'))}
    head = [row['head_conditional_mismatch'] for row in rows]
    rest = [row['remainder_joint_propagation_allocation_numerical'] for row in rows]
    total = [row['total_dt_minus_native'] for row in rows]
    mse_head = statistics.mean(x*x for x in head)
    mse_rest = statistics.mean(x*x for x in rest)
    cross = 2*statistics.mean(x*y for x, y in zip(head, rest))
    mse_total = statistics.mean(x*x for x in total)
    return dict(comparisons=comparisons,
        errors={field: describe([row[field] for row in rows]) for field in FIELDS[4:]},
        error_MSE_identity=dict(total=mse_total, head=mse_head, remainder=mse_rest,
            cross_term=cross, reconstruction_residual=mse_total-mse_head-mse_rest-cross,
            scope='Algebraic identity, not causal percentage attribution. Retain the cross term.'),
        binary_sign_checks=dict(
            head_native_same_nonzero_sign=sum(row['owner_head_projected_d']*row['native_single_delete_d']>0 for row in rows),
            dt_native_opposite_sign=sum(row['dt_d']*row['native_single_delete_d']<0 for row in rows),
            dt_head_opposite_sign=sum(row['dt_d']*row['owner_head_projected_d']<0 for row in rows),
            n=len(rows)))


def main():
    path = LOCAL / 'head-seed-cpu-observation.json'
    observed = json.loads(path.read_bytes())
    rows = observed['observations']
    assert len(rows)==52
    for row in rows:
        assert row['target_class_index']==1 and row['outcome_token_ids']==[15,16]
        assert row['native_single_delete_d']==row['single_token_eos']['native_target_log_ratio']
        assert row['dt_d']==row['saved_DT_d']
    pool = reduce_replicas(rows, ('traj_uid','source_step','source_position'), FIELDS)
    assert pool['unique_identities']==42
    previous = json.loads((LOCAL/'native-matched-layout-analysis.json').read_bytes())
    before = previous['same_layout_single_delete_vs_saved_DT']['unique_probe_balanced']['saved_dt_vs_matched_single']
    unique = summarize(pool['rows'])
    assert unique['comparisons']['dt_vs_native']['mae']==before['mae']
    assert unique['comparisons']['dt_vs_native']['rmse']==before['rmse']
    sources = [source(p) for p in (path, Path(__file__),
        Path(__file__).with_name('observe_saved_textcraft_head_seed_cpu.py'),
        Path(__file__).with_name('run_saved_textcraft_head_seed_cpu.py'),
        LOCAL/'native-matched-layout-logits-map.json',
        LOCAL/'owner-contract/head-seed-source-contract.json')]
    result = dict(scope=__doc__, status='head_conditional_error_separated_remainder_not_yet_layer_localized',
        sources=sources, actual_imported_owner=observed['imported_owner_sources'],
        recomputation=dict(pid=observed['pid'], wall_seconds=observed['wall_seconds'],
            max_rss_bytes=observed['process_max_rss_bytes'], dtype=observed['dtype'],
            cuda_initialized=observed['cuda_initialized'], distributed_initialized=observed['distributed_initialized'],
            model_forward_calls=0, finite_decoder_calls=0, backward_calls=0, optimizer_steps=0, scheduler_steps=0),
        coverage=dict(transport_probes=52, unique_probes=42, original_first_response_UIDs=21),
        transport_weighted=summarize(rows), unique_probe_balanced=unique, retained_replica_pool=pool,
        max_float64_error_decomposition_residual=max(abs(row['head_conditional_mismatch']+
            row['remainder_joint_propagation_allocation_numerical']-row['total_dt_minus_native']) for row in rows),
        conclusions=[
            'The original joint categorical seed applied to actual single-token native logits has the same sign as the native log-prob effect for every selected unique probe.',
            'All 21 selected DT/native sign disagreements remain before the categorical seed boundary; top conditional curvature alone does not explain them.',
            'The remainder includes projection, lower finite propagation, joint decomposition and native low-precision storage; this evidence does not identify a specific layer or violate FA/FLA acceptance.',
            'Do not change PPO/entropy, rescale credits, or deploy a different head seed on this evidence.',
            'The 2.72-second CPU observation used saved real tensors only; it is not a new model/DT training experiment.'],
        limitations=observed['limitations']+['The earlier full-minibatch gradient imbalance and this selected first-response token analysis are complementary observations, not an end-to-end historical Adam causal proof.'])
    out = LOCAL/'head-seed-cpu-analysis.json'
    out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output=source(out), coverage=result['coverage'],
        unique=unique, recomputation=result['recomputation']), indent=2))


if __name__=='__main__':
    main()

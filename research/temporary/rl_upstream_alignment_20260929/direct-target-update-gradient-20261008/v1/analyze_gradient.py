"""Summarize saved native gradients; never change training or invent a tolerance."""
from pathlib import Path
import datetime
import hashlib
import json
import math

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]


def receipt(path):
    return dict(path=path.resolve().as_posix(), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    ranks = [json.loads((HERE / f'rank{i}-gradients.json').read_bytes()) for i in (0, 1)]
    inspection = json.loads((HERE / 'input-inspection.json').read_bytes())
    launch = json.loads((HERE / 'gradient-launch.json').read_bytes())
    completed = json.loads((HERE / 'completed.json').read_bytes())
    manifest = json.loads((HERE / 'transport-manifest.json').read_bytes())
    for item in manifest:
        record = receipt(HERE / item['name'])
        assert record['bytes'] == item['bytes'] and record['sha256'] == item['sha256']
    assert ranks[0]['gradient_statistics'] == ranks[1]['gradient_statistics']
    assert inspection['rows'] == 64 and inspection['valid_tokens'] == 203153
    assert all(inspection['native_position_checks'])
    for rank in ranks:
        assert rank['labels'] == ['dt_pg', 'grpo_pg']
        assert all(rank['parameters_exact_unchanged'].values())
        assert not rank['optimizer_step_executed'] and not rank['scheduler_step_executed']
        assert rank['native_gradient_clipping_executed']
        assert rank['first_held_source_sha256'] == launch['source_sha256']
        for label in rank['labels']:
            p = rank['passes'][label]
            assert len(p['microbatch_losses']) == 8
            assert p['optimizer_boundary']['optimizer_step_no_op_calls'] == 1
            assert math.isfinite(p['optimizer_boundary']['native_clip_return'])
            assert all(x['pg_clipfrac'] == x['ppo_kl'] == x['pg_clipfrac_lower'] == 0
                       for x in p['microbatch_losses'])

    stats = ranks[0]['gradient_statistics']
    assert all(math.isfinite(v) for v in stats['inner_products'].values())
    full = stats['norms']['dt_pg']
    single = stats['norms']['grpo_pg']
    dot = stats['inner_products']['dt_pg:grpo_pg']
    without = math.sqrt(full * full + single * single - 2 * dot)
    cosine_without = (full * full - dot) / (full * without)
    observed = [json.loads(p.read_bytes()) for p in sorted(HERE.glob('gradient-observation-*.json'))]
    final = max(observed, key=lambda x: x['unix'])
    observations = [dict(unix=x['unix'], driver_same_birth=x['driver_same_birth'],
                         process_count=len(x['processes']),
                         process_tree_pss_bytes=sum(p['pss_bytes'] for p in x['processes']),
                         text_same_birth=x['text_same_birth'], releases=x['text_releases'])
                    for x in observed]
    result = dict(
        observed_unix=final['unix'],
        observed_utc=datetime.datetime.fromtimestamp(final['unix'], datetime.timezone.utc).isoformat(),
        status='Native first-minibatch gradient contribution measured; credit not repaired; formal update held',
        scope='Actual first global64/local32/B4 TextCraft optimizer minibatch at untouched base LoRA weights; two native PG backward passes without optimizer/scheduler writes. No GRPO algorithm was run.',
        code_commit='5fba8de4',
        upstream_lock=receipt(REPO / 'experiments/rl/upstream.lock'),
        imported_sources=ranks[0]['sources'],
        diagnostic_adapter=ranks[0]['gradient_adapter'],
        original_held_source_sha256=launch['source_sha256'],
        input_inspection=inspection,
        sources=[receipt(HERE / item['name']) for item in manifest],
        label_meanings=ranks[0]['actual_label_meaning'],
        effective_config=ranks[0]['effective_config'],
        gradient=dict(full_DT_PG_norm=full, isolated_Format_token_norm=single,
                      norm_ratio=single / full,
                      cosine_with_full=dot / (full * single),
                      projection_on_full_fraction=dot / (full * full),
                      full_minus_isolated_norm=without,
                      full_vs_without_token_cosine=cosine_without,
                      full_vs_without_token_angle_degrees=math.degrees(math.acos(cosine_without)),
                      statistics=stats),
        native_clip_returns_by_rank=[{k: v['optimizer_boundary']['native_clip_return']
                                     for k, v in rank['passes'].items()} for rank in ranks],
        elapsed_seconds_by_rank=[{k: v['elapsed_seconds'] for k, v in rank['passes'].items()}
                                 for rank in ranks],
        parameters_exact_unchanged_by_rank=[all(rank['parameters_exact_unchanged'].values()) for rank in ranks],
        parameter_tensor_count_by_rank=[len(rank['parameters_exact_unchanged']) for rank in ranks],
        job=dict(launch=launch, completed=completed, final_observation=receipt(
            max(HERE.glob('gradient-observation-*.json'), key=lambda p: json.loads(p.read_bytes())['unix'])),
            driver_same_birth=final['driver_same_birth'], observations=observations),
        formal_state=dict(text_same_birth=final['text_same_birth'], text_release_present=final['text_releases'],
                          text_optimizer_updates=0, appworld_formal_terminal=True,
                          formal_restart=False, checkpoint_restore=False),
        interpretation=[
            'The Format token has a measurable contribution. A 21.65% norm ratio is not an additive percentage of the gradient, nor proof that it dominates the minibatch.',
            'Its projection on the full gradient is about 4.00%; subtracting the measured isolated vector from the measured full vector changes the pre-clip PG direction by about 12.50 degrees. No removed-token update was executed.',
            'Observed native ppo_kl and both clipping fractions were zero in all eight microbatches of each pass/rank at unchanged initial weights. These scalar observations are not a saved per-token ratio equality test.',
            'The same official whole-batch whitening coefficients and original token-mean/microbatch accumulation were used. No new whitening, learned critic, alternative objective, coefficient correction, clipping, or parameter change occurred.',
        ],
        limitations=[
            'One real minibatch at initial weights is not the old divergent run or evidence of its complete cause.',
            'The isolated coefficient keeps the original already-whitened vector. Subtracting it is a diagnostic decomposition, not a proposed algorithm or a re-whitened counterfactual batch.',
            'The vector subtraction uses the separately measured native low-precision backward results, so the angle describes those measurements rather than an exact-arithmetic identity or actual optimizer update.',
            'No scalar-gradient norm or direction threshold has been invented; this is measurement, not an official numerical tolerance test.',
            'Scalar branch hooks suppress other original PG/entropy/KL output gradients while their native bodies still execute. Entropy and KL parameters are unchanged; their contributions are not included in the reported PG vectors.',
            'Raw observer legacy grpo_pg/diagnostic_grpo_advantages names mean the isolated Format DT component here. The observer historical provenance text is superseded by the explicit wrapper alias binding; it is not a GRPO baseline.',
            'PSS comes from sampled process trees, including original Ray children. mx-smi samples are retained separately; no physical peak was inferred from the Torch virtual allocator.',
            'Cumulative deletion/RISE/MAS evidence remains separately recorded. This gradient observation cannot replace that attribution assessment.',
        ],
        credit_repaired=False,
        memory_candidate_formally_deployed=False,
    )
    target = REPO / 'experiments/rl/results_update_gradient_20261008.json'
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (HERE / 'analysis.json').write_text(json.dumps(result['gradient'], indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(receipt=receipt(target), gradient=result['gradient'],
                         formal_state=result['formal_state']), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

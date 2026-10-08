"""Finite original-minibatch gradient contributions; no pooled tail/bulk mean."""
import datetime
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
LABELS = ('full_pg', 'development_predicted_tail', 'development_tail_native_sign_flip')


def receipt(path):
    return dict(path=path.resolve().as_posix(), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                bytes=path.stat().st_size)


def geometry(stats, label):
    full = stats['norms']['full_pg']
    part = stats['norms'][label]
    dot = stats['inner_products'][f'full_pg:{label}']
    difference_squared = full * full + part * part - 2 * dot
    difference = math.sqrt(max(0., difference_squared))
    cosine = (full * full - dot) / (full * difference) if full and difference else None
    return dict(norm=part, norm_ratio=part / full if full else None,
                cosine_with_full=dot / (full * part) if full and part else None,
                projection_on_full_fraction=dot / (full * full) if full else None,
                full_minus_component_norm=difference,
                vector_subtraction_angle_degrees=math.degrees(math.acos(max(-1., min(1., cosine))))
                    if cosine is not None else None,
                scope='Pre-clip native backward vectors. A norm ratio is not an additive percentage; the subtraction angle is vector geometry, not a rerun with changed coefficients.')


def main():
    paths = list(HERE.glob('collection-gradient-observation-*.json'))
    assert paths, 'No observation has been transported'
    latest = max(paths, key=lambda p: json.loads(p.read_bytes())['unix'])
    observed = json.loads(latest.read_bytes())
    files = observed['files']
    inspection = files['input-inspection.json']
    minibatches = []
    for index in range(4):
        names = [f'minibatch{index}-rank{rank}.json' for rank in (0, 1)]
        if not all(name in files for name in names):
            continue
        ranks = [files[name] for name in names]
        assert ranks[0]['gradient_statistics'] == ranks[1]['gradient_statistics']
        stats = ranks[0]['gradient_statistics']
        assert all(math.isfinite(v) for v in stats['inner_products'].values())
        for rank in ranks:
            assert rank['labels'] == list(LABELS)
            assert all(rank['parameters_exact_unchanged'].values())
            assert not rank['optimizer_step_executed'] and not rank['scheduler_step_executed']
            assert rank['effective_config']['ppo_micro_batch_size_per_gpu'] == 4
            assert rank['effective_config']['ppo_mini_batch_size'] == 32
            for label in LABELS:
                passed = rank['passes'][label]
                assert len(passed['microbatch_losses']) == 8
                assert passed['optimizer_boundary']['optimizer_step_no_op_calls'] == 1
                assert math.isfinite(passed['optimizer_boundary']['native_clip_return'])
                assert all(m['pg_clipfrac'] == m['ppo_kl'] == m['pg_clipfrac_lower'] == 0
                           for m in passed['microbatch_losses'])
        points = [p for p in inspection['mapping']
                  if any(m['optimizer_minibatch'] == index for m in p['actor_matches'])]
        minibatches.append(dict(index=index, points=len(points),
            states=len({p['initial_state_sha256'] for p in points}),
            raw_native_sign_flip_points=sum(p['native_single_d'] >= 0 for p in points),
            full_pg_norm=stats['norms']['full_pg'],
            all_predicted_tail=geometry(stats, LABELS[1]),
            raw_native_sign_flip_subset=geometry(stats, LABELS[2]),
            statistics=stats,
            elapsed_seconds_by_rank=[{k: v['elapsed_seconds'] for k, v in r['passes'].items()} for r in ranks],
            native_clip_returns_by_rank=[{k:v['optimizer_boundary']['native_clip_return'] for k,v in r['passes'].items()} for r in ranks]))
    # The earlier run did not store initial LoRA_A hashes. The policy is at
    # LoRA_B=0 in both runs, but identical gradient parameterization is unproven.
    # Do not turn this cross-run observation into a tolerance or regression test.
    prior = json.loads((HERE.parents[1] / 'direct-target-update-gradient-20261008/v1/rank0-gradients.json').read_bytes())
    first = files.get('minibatch0-rank0.json')
    parity = None
    if first:
        old = prior['passes']['dt_pg']['microbatch_losses']
        new = first['passes']['full_pg']['microbatch_losses']
        scalar_diffs = [abs(a['dt_pg']['value'] - b['dt_pg']['value']) for a, b in zip(old, new)]
        old_norm = prior['gradient_statistics']['norms']['dt_pg']
        new_norm = first['gradient_statistics']['norms']['full_pg']
        parity = dict(previous_norm=old_norm, current_norm=new_norm,
                      norm_absolute_difference=abs(old_norm - new_norm),
                      max_pg_scalar_absolute_difference=max(scalar_diffs),
                      initial_LoRA_A_cross_run_identity='Unverified; prior run did not retain initial parameter hashes.',
                      scope='Same original first-minibatch coefficients and policy at LoRA_B=0. Cross-run gradients have unverified parameter identity; neither a tolerance pass/failure nor evidence about the optional seam. Within-run component comparisons retain identical parameters.')
    phase = files.get('phase.json', {})
    all_native_returns = (len(minibatches) == 4 and phase.get('phase') == 'original_minibatch_gradients_complete'
                          and phase.get('minibatch') == 3)
    budget_exit = (all_native_returns and 'completed.json' not in files
                   and 'Bounded collection gradient diagnostic exceeded 1800 seconds' in observed['log_tail'])
    complete = all_native_returns and ('completed.json' in files or
                                      (budget_exit and not observed['driver_same_birth']))
    status = ('Complete native measurement; driver budget exit after all four minibatches returned'
              if complete and budget_exit else 'Complete bounded collection gradient measurement'
              if complete else 'Partial measurement; no complete collection claim')
    result = dict(scope=__doc__, status=status,
        observed_unix=observed['unix'], observed_utc=datetime.datetime.fromtimestamp(observed['unix'], datetime.timezone.utc).isoformat(),
        launch=observed['launch'], observation=receipt(latest), inspection=inspection,
        minibatches=minibatches, first_minibatch_cross_run_observation=parity,
        completed=files.get('completed.json'),
        driver_completion=dict(final_native_phase=phase, driver_same_birth=observed['driver_same_birth'],
            completed_json_present='completed.json' in files,
            post_measurement_budget_exit=budget_exit,
            native_elapsed_seconds=phase.get('elapsed_seconds'),
            planned_budget_seconds=1800,
            budget_overrun_seconds=max(0., phase.get('elapsed_seconds', 0.) - 1800.) if budget_exit else None,
            scope='All original actor-group calls and both rank parameter checks returned before the diagnostic-only budget check. The missing completed.json is not fabricated; the driver budget exception remains recorded.'),
        whitening_statistic=json.loads((HERE / 'collection-whitening-statistic.json').read_bytes()),
        effective_configuration=receipt(HERE / 'collection-gradient-effective-config.yaml'),
        frozen_manifest=receipt(HERE / 'manifest.json'),
        baseline_forward_collection=receipt(REPO / 'experiments/rl/results_credit_author_collection_20261008.json'),
        interpretation=[
            'These are gradients of the exact saved actor coefficients through the original owner; no native d/A replaces training credit.',
            'The tail census includes 35 predicted c in (2,10] and 2 in (10,100]; it excludes the bounded bulk. Its vector sum is the real loss contribution, not a statistical mean or moment estimate.',
            'Native d>=0 labels a raw-credit sign crossing. It does not assert that a hypothetical rewhitened native coefficient has positive sign.',
            'Uniformly sampled native tails missed by DT and the bulk error gradient are not measured here. The observed tail norm cannot be extrapolated to them.',
            'Four minibatches use the same untouched base LoRA. This is not replay of four sequential optimizer updates, not the older degradation window, and not GDN causal localization.',
            'The native clipping call executes, while optimizer and scheduler writes are disabled. Geometry uses the saved pre-clip gradients.',
        ],
        resources=dict(host_available=observed['host_available'], process_tree_pss_bytes=sum(p['pss_bytes'] for p in observed['processes']),
                       observation_scope='Point-in-time physical mx-smi and per-process PSS, not continuous peaks.'),
        formal_state=dict(text_same_birth=observed['text_same_birth'], text_releases=observed['text_releases'],
                          new_GDN_candidate=False, optimizer_steps=0, checkpoint_restore=False, formal_restart=False))
    path = HERE / 'collection-gradient-analysis.json'
    path.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    if complete:
        (REPO / 'experiments/rl/results_credit_collection_gradients_20261008.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(status=result['status'], minibatches=[{k:v for k,v in m.items() if k not in ('statistics','elapsed_seconds_by_rank','native_clip_returns_by_rank')} for m in minibatches], parity=parity), ensure_ascii=False))


if __name__ == '__main__':
    main()

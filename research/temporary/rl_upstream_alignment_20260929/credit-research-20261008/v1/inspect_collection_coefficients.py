"""Read-only binding of frozen single-deletion observations to actual actor coefficients.

The original owner supplies reward composition and batch moments. Native
single-deletion values are diagnostic references only. Fixed original moments
separate coefficient error from changing the whole-batch normalization; these
are not newly whitened training advantages or an error-gradient measurement.
"""
import hashlib
import inspect
import json
import os
from pathlib import Path
import time

import psutil
import torch

from inspect_collection_gradients import bind_saved_source_points
from counterfactual import reward_event_token_credit
import verl.utils.torch_functional as verl_F

HERE = Path(__file__).resolve().parent
ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
CAPTURE = ROOT / 'receipts/direct-target-prefix-runtime-20261007-v1/textcraft-first-dt'
SOURCE = ROOT / 'runs/direct-target-prefix-runtime-20261007-v1/textcraft/textcraft-dt/source.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(value):
    path = inspect.getsourcefile(inspect.unwrap(value))
    return dict(path=path, sha256=sha(path))


def band(d):
    # The same frozen diagnostic bins, not a transformation/clipping rule.
    import math
    return ('ratio_le_1' if d >= 0 else 'ratio_1_to_2' if d >= -math.log(2)
            else 'ratio_2_to_10' if d >= -math.log(10)
            else 'ratio_10_to_100' if d >= -math.log(100) else 'ratio_gt_100')


def quantiles(values):
    if not values:
        return None
    data = torch.tensor(values, dtype=torch.float64)
    return dict(zip(('min', 'p25', 'median', 'p75', 'p95', 'max'),
                    torch.quantile(data, torch.tensor([0., .25, .5, .75, .95, 1.],
                                                     dtype=torch.float64)).tolist()))


def credit(d, reward):
    return reward_event_token_credit(
        torch.tensor([[[d]]], dtype=torch.float32),
        torch.tensor([[reward]], dtype=torch.float32),
        torch.ones((1, 1, 1), dtype=torch.bool),
        torch.ones((1, 1), dtype=torch.bool))


def describe(points):
    return dict(points=len(points), trajectories=len({p['traj_uid'] for p in points}),
                initial_states=len({p['initial_state_sha256'] for p in points}),
                actual_actor_matches=sum(len(p['actor_matches']) for p in points),
                raw_sign_disagreements=sum(p['raw_sign_disagreement'] for p in points),
                fixed_moment_actor_sign_disagreements=sum(p['fixed_moment_actor_sign_disagreement'] for p in points),
                abs_raw_A_error=quantiles([abs(p['delta_raw_A']) for p in points]),
                abs_fixed_scale_coefficient_error=quantiles([abs(p['delta_coefficient_fixed_scale']) for p in points]),
                abs_saved_actor_coefficient=quantiles([abs(m['saved_whitened_A']) for p in points for m in p['actor_matches']]))


def main():
    torch.set_num_threads(4)
    started = time.monotonic()
    plan = json.loads((HERE / 'coefficient-inputs.json').read_bytes())
    source = json.loads(SOURCE.read_bytes())
    assert sha(SOURCE) == plan['source_sha256']
    assert identity(reward_event_token_credit)['sha256'] == source['entry_sha256']['counterfactual.py']
    assert identity(verl_F.masked_whiten)['sha256'] == plan['whitening_owner_sha256']
    paths = [CAPTURE / f'rank{rank}-pre-update.pt' for rank in (0, 1)]
    assert [sha(p) for p in paths] == plan['preupdate_sha256']
    assert not any((CAPTURE / f'rank{rank}-release-update').exists() for rank in (0, 1))
    saved = [torch.load(p, map_location='cpu', weights_only=False) for p in paths]
    tensors = {k: torch.cat([s['tensors'][k] for s in saved]) for k in saved[0]['tensors']}
    # Default binding is compared against the complete previous real-artifact
    # receipt, not a synthetic fixture or a new numerical tolerance.
    bound_tail = bind_saved_source_points(plan['tail_points'], saved, tensors)
    assert bound_tail == plan['previous_tail_mapping']
    bound_uniform = bind_saved_source_points(plan['uniform_points'], saved, tensors)
    raw, mask = tensors['dt_token_advantages'], tensors['response_mask']
    mean, variance = verl_F.masked_mean(raw, mask), verl_F.masked_var(raw, mask)
    scale = torch.rsqrt(variance + 1e-8)
    records = []
    for cohort, points in [('uniform', bound_uniform), ('predicted_tail_census', bound_tail)]:
        for point in points:
            factual = credit(point['d'], point['reward'])
            native = credit(point['native_single_d'], point['reward'])
            matches = point['actor_matches']
            assert len({m['actual_raw_A'] for m in matches}) == 1
            raw_A, native_A = matches[0]['actual_raw_A'], native.advantages.item()
            actual_raw = torch.tensor(raw_A, dtype=raw.dtype)
            delta = native.advantages.squeeze() - actual_raw
            fixed_native = (native.advantages.squeeze() - mean) * scale
            original_at_fixed = (actual_raw - mean) * scale
            records.append(dict(**point, cohort=cohort, predicted_band=band(point['d']),
                native_band=band(point['native_single_d']), native_raw_A=native_A,
                delta_raw_A=delta.item(), delta_coefficient_fixed_scale=(delta * scale).item(),
                native_coefficient_at_original_moments=fixed_native.item(),
                actual_raw_A=raw_A,
                CPU_original_composition_difference=factual.advantages.item() - raw_A,
                raw_sign_disagreement=bool(raw_A * native_A < 0),
                fixed_moment_actor_sign_disagreement=bool(original_at_fixed * fixed_native < 0)))
    cohorts = {}
    for cohort in ('uniform', 'predicted_tail_census'):
        rows = [p for p in records if p['cohort'] == cohort]
        cells = {}
        for key in sorted({(p['predicted_band'], p['native_band']) for p in rows}):
            cell = [p for p in rows if (p['predicted_band'], p['native_band']) == key]
            cells[' -> '.join(key)] = dict(**describe(cell), by_state={state: describe(
                [p for p in cell if p['initial_state_sha256'] == state])
                for state in sorted({p['initial_state_sha256'] for p in cell})},
                by_original_minibatch={str(index): describe([p for p in cell if any(
                    m['optimizer_minibatch'] == index for m in p['actor_matches'])])
                    for index in range(4)})
        cohorts[cohort] = dict(points=len(rows), cells=cells)
    result = dict(scope=__doc__, unix=time.time(), inputs_plan_sha256=sha(HERE / 'coefficient-inputs.json'),
        script=identity(main), source_sha256=sha(SOURCE), preupdate_sha256=[sha(p) for p in paths],
        binding=identity(bind_saved_source_points), original_tail_mapping_exact=True,
        owners=dict(reward_composition=identity(reward_event_token_credit),
                    masked_whiten=identity(verl_F.masked_whiten), masked_var=identity(verl_F.masked_var)),
        original_moments=dict(mean=mean.item(), variance=variance.item(), scale=scale.item(),
                              dtype=str(raw.dtype), mask_tokens=int(mask.sum())),
        cohorts=cohorts, records=records,
        limitations=['Fixed original moments; not a rewhitened or corrected batch.',
            'Native single-EOS model scores, not an exact environment oracle.',
            'Uniform sample and tail census are separate designs; no pooled raw mean or population-moment claim.',
            'Coefficient error is not a measured policy gradient error; no inference from squared coefficients to gradient shares.',
            'No unseen test labels are queried; no new model forward, DT, backward, optimizer, environment or restore.'],
        elapsed_seconds=time.monotonic() - started, cuda_initialized=torch.cuda.is_initialized(),
        rss_bytes=psutil.Process().memory_info().rss,
        process_peak_rss_bytes=__import__('resource').getrusage(__import__('resource').RUSAGE_SELF).ru_maxrss * 1024,
        host_available_bytes=psutil.virtual_memory().available,
        formal_release=[(CAPTURE / f'rank{rank}-release-update').exists() for rank in (0, 1)])
    assert not result['cuda_initialized'] and result['formal_release'] == [False, False]
    (HERE / 'coefficients.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(status='complete read-only coefficient binding',
                         counts={k:v['points'] for k,v in cohorts.items()},
                         elapsed_seconds=result['elapsed_seconds'], peak_rss_bytes=result['process_peak_rss_bytes'])))


if __name__ == '__main__':
    main()

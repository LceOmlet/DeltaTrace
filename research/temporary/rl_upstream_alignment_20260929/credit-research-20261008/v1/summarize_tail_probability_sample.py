"""Read complete native sample results; never substitute partial results as TN."""
import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path
import time

from tail_probability_statistics import summarize_task


def ref(path):
    raw = Path(path).read_bytes()
    return dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


def ratio_bin(d):
    if d >= 0:
        return 'ratio_le_1'
    for cutoff,name in ((2,'ratio_1_to_2'),(10,'ratio_2_to_10'),(100,'ratio_10_to_100')):
        if d >= -math.log(cutoff):
            return name
    return 'ratio_gt_100'


def weighted_quantiles(values,weights):
    if not values:
        return None
    ordered = sorted(zip(values,weights))
    result = {'0':ordered[0][0], '1':ordered[-1][0]}
    for p in (.25,.5,.75):
        cut = p*math.fsum(weights)
        total = 0
        for value,weight in ordered:
            total += weight
            if total >= cut:
                result[str(p)] = value
                break
    return result


def main():
    import scipy
    from scipy import stats
    begin = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--task',choices=('textcraft','appworld'))
    parser.add_argument('--prior-reference-ranges',type=Path,
        help='Preserve every prior native range on matching identities, without appending observations or changing sampling weights.')
    args = parser.parse_args()
    directory = Path(args.directory)
    sample_path = directory/'sample.json'
    prepared_path = directory/'native-prepared.json'
    sample = json.loads(sample_path.read_bytes())
    prepared = json.loads(prepared_path.read_bytes())
    assert ref(sample_path)['sha256'] == prepared['sample_sha256']
    prior = json.loads(args.prior_reference_ranges.read_bytes()) if args.prior_reference_ranges else None
    result = dict(scope=__doc__,sources=[ref(sample_path),ref(prepared_path),ref(__file__),
        ref(Path(__file__).with_name('tail_probability_statistics.py'))],
        sampling_owner=dict(name='scipy.stats.hypergeom',version=scipy.__version__,
            source=ref(inspect.getsourcefile(stats.hypergeom.__class__))),
        tasks={},operations=dict(model=0,DT=0,GPU=0,optimizer=0),
        official_FA_FLA_tolerance_changed=False,production_changed=False,
        reference_scope='Native single-EOS outcome probabilities under the saved model, with actual dtype and same-hidden FP32 head sensitivity. '
            'No exact world-oracle claim; sampling intervals do not bound numerical/model error.',
        interpretation='One finite-population design, not a concatenation of earlier diagnostic samples. '
            'All per-state intervals are pointwise, and rare/undefined groups remain visible.')
    if prior is not None:
        result['sources'].append(ref(args.prior_reference_ranges))
    for task,spec in sample['tasks'].items():
        if args.task is not None and task != args.task:
            continue
        observations = []
        cost = []
        prior_by_key = {(p['traj_uid'],p['packed_slot']):p for p in prior['tasks'][task]} if prior else {}
        for job in prepared['jobs']:
            if job['task'] != task:
                continue
            root = Path(job['directory'])/'results'
            assert (root/'completed.json').exists(), 'Do not publish incomplete collection as a recall result'
            for rank in (0,1):
                path = root/f'rank{rank}.json'
                data = json.loads(path.read_bytes())
                assert data['phase'] == 'complete'
                result['sources'].append(ref(path))
                cost.append(dict(chunk=job['chunk'],rank=rank,seconds=data['elapsed_seconds'],
                    native_forward_calls=data['operations']['native_forward'],
                    peak_allocated=data['peak_allocated'],PSS=data['process_pss_bytes']))
                for batch in data['batches']:
                    for point in batch['points']:
                        references = [v['d'] for v in point['scores'].values()]
                        assert all(math.isfinite(x) for x in references)
                        current_range = [min(references),max(references)]
                        previous = prior_by_key.get((point['traj_uid'],point['packed_slot']))
                        if previous is not None:
                            assert previous['token_id'] == point['token_id']
                            references.extend(previous['native_d_interval'])
                        observations.append(dict(point,current_native_d_interval=current_range,
                            prior_native_d_interval=previous['native_d_interval'] if previous else None,
                            native_d_interval=[min(references),max(references)]))
        main_table = {str(c):summarize_task(spec,observations,c) for c in (2,10,100)}
        state_ids = sorted({e['initial_state_sha256'] for e in spec['entries']})
        states = {s:{str(c):summarize_task(spec,observations,c,domain_state=s)
            for c in (2,10,100)} for s in state_ids}
        crossed = {}
        for p in observations:
            native = p['scores']['FP32']['d']
            label = p['prediction_stratum']+':'+ratio_bin(native)
            crossed.setdefault(label,[]).append(p)
        conditional = {}
        for label,points in crossed.items():
            weights = [p['token_total_weight'] for p in points]
            conditional[label] = dict(sampled_positions=len(points),
                sampled_states=len({p['initial_state_sha256'] for p in points}),
                estimated_frame_positions=sum(weights),
                native_d_quantiles=weighted_quantiles([p['scores']['FP32']['d'] for p in points],weights),
                DT_minus_native_d_quantiles=weighted_quantiles([p['saved_d']-p['scores']['FP32']['d'] for p in points],weights),
                scope='Conditional weighted empirical quantiles, no raw unbounded-advantage moments or precision guarantee.')
        result['tasks'][task] = dict(frame={k:spec[k] for k in ('declared_trajectories',
            'completed_trajectories','missing_uids','initial_states','completed_frame_sources',
            'sampled_sources','sampled_trajectories','sampled_states','strata')},
            confusion_by_cumulative_threshold=main_table,by_initial_state=states,
            crossed_magnitude_bins=conditional,cost=cost,observations=observations,
            prior_reference_sensitivity=dict(
                matching_sampled_positions=sum(p['prior_native_d_interval'] is not None for p in observations),
                additional_sampled_positions=0,sampling_weights_changed=False,
                rule='Union all matching recorded reference ranges with the current three head views; no favorable-reference selection. '
                    'This observed numerical range is not a confidence interval or an error bound on exact values.'))
    result['elapsed_seconds'] = time.perf_counter()-begin
    output = Path(args.output)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=ref(output),tasks={t:d['confusion_by_cumulative_threshold']['2']
        for t,d in result['tasks'].items()}),ensure_ascii=False))


if __name__ == '__main__':
    main()

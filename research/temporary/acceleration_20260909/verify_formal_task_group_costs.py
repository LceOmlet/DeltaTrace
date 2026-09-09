"""Validate saved formal-helper timings; no new model or metric calculations."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np

HERE = Path(__file__).resolve().parent
sha = lambda data: hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--quality', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raw = args.results.read_bytes()
    report = json.loads(raw)
    plan = json.loads(args.plan.read_bytes())
    quality = json.loads(args.quality.read_bytes())
    assert report['status'] == 'complete' and report['quality_not_recomputed']
    assert report['metric_calls'] == report['generation_calls'] == 0
    assert report['driver_sha256'] == sha((HERE / 'benchmark_formal_task_groups.py').read_bytes())
    assert report['plan_sha256'] == sha(args.plan.read_bytes())
    assert report['quality_report_sha256'] == plan['quality_report_sha256'] == sha(args.quality.read_bytes())
    assert report['batching_sha256'] == plan['batching_sha256']
    assert report['acceleration_sources'] == plan['acceleration_sources']
    assert report['batches'] == plan['batches']
    assert report['measured_calls_without_new_dynamo_graphs']
    vector_path = args.results.parent / 'vectors.npz'
    assert sha(vector_path.read_bytes()) == report['vectors_sha256']
    quality_vectors = args.quality.parent / 'vectors.npz'
    assert sha(quality_vectors.read_bytes()) == plan['quality_vectors_sha256']
    z = np.load(vector_path, allow_pickle=False)
    qz = np.load(quality_vectors, allow_pickle=False)
    cases = {f"{r['dataset']}_{r['index']}": r for r in quality['cases']}
    calls = {call['name']: call for call in report['calls']}
    expected = {'model_load', 'native_eager_initialization'}
    for batch in range(8):
        expected.add(f'warm_accelerated/{batch}')
        for key in plan['batches'][batch]['cases']:
            expected.add('warm_baseline/' + key)
        for repeat in range(2):
            expected.add(f'measured_r{repeat}_accelerated/{batch}')
            for key in plan['batches'][batch]['cases']:
                expected.add(f'measured_r{repeat}_baseline/' + key)
    assert len(calls) == len(report['calls']) == len(expected) == 74
    assert set(calls) == expected
    order = ['model_load', 'native_eager_initialization']
    order += ['warm_baseline/' + key for batch in plan['batches'] for key in batch['cases']]
    order += [f'warm_accelerated/{batch}' for batch in range(8)]
    for repeat in range(2):
        for batch in (range(8) if repeat == 0 else reversed(range(8))):
            baseline_names = [f'measured_r{repeat}_baseline/' + key for key in plan['batches'][batch]['cases']]
            accelerated_names = [f'measured_r{repeat}_accelerated/{batch}']
            order += baseline_names + accelerated_names if repeat == 0 else accelerated_names + baseline_names
    assert [call['name'] for call in report['calls']] == order
    expected_vectors = set()
    for name, call in calls.items():
        assert call['status'] == 'returned' and call['seconds'] > 0
        if name in ('model_load', 'native_eager_initialization'):
            continue
        accelerated = '_accelerated/' in name
        keys = plan['batches'][int(name.rsplit('/', 1)[1])]['cases'] if accelerated else [name.rsplit('/', 1)[1]]
        assert call['cases'] == keys
        rows = [cases[key] for key in keys]
        lengths = [len(row['input_ids']) for row in rows]
        eos = rows[0]['input_ids'][-1]
        ids = np.full((2*len(rows), max(lengths)), eos, dtype=np.int64)
        mask = np.zeros_like(ids)
        for j, row in enumerate(rows):
            ids[2*j:2*j+2, :lengths[j]] = row['input_ids']
            ids[2*j, [row['user_positions'][k] for k in row['keep']]] = eos
            mask[2*j:2*j+2, :lengths[j]] = 1
        root = call['actual_root']
        assert root == {'input_sha256': sha(ids.tobytes()), 'mask_sha256': sha(mask.tobytes()),
                        'sample_batch': len(rows), 'endpoint_batch': 2*len(rows)}
        detail = call['details']
        assert len(detail['layers']) == 32
        assert detail['norm_gate_rules'] == detail['attention_pv_rules'] == {}
        assert detail['finite_fla_by_layer'] == detail['key_norm_by_layer'] == []
        assert detail.get('checkpoint_device', 'cpu') == ('cuda' if accelerated else 'cpu')
        if name.startswith('measured_'):
            assert call['compiler_before'].get('unique_graphs', 0) == call['compiler_after'].get('unique_graphs', 0)
        for key, length in zip(keys, lengths):
            vector_key = name + '/' + key
            expected_vectors.add(vector_key)
            assert z[vector_key].shape == (length,) and np.isfinite(z[vector_key]).all()
    assert set(z.files) == expected_vectors and len(expected_vectors) == 96
    rounds = []
    for repeat in range(2):
        row = {mode: sum(call['seconds'] for name, call in calls.items()
                         if name.startswith(f'measured_r{repeat}_{mode}/')) for mode in ('baseline', 'accelerated')}
        row['speedup'] = row['baseline'] / row['accelerated']
        rounds.append(row)
    aggregate = {}
    for mode in ('baseline', 'accelerated'):
        selected = [c for name, c in calls.items() if name.startswith('measured_') and f'_{mode}/' in name]
        stages = {}
        for call in selected:
            for stage in call['details']['calls']:
                kind = stage['kind']
                for prefix in ('finite_decoder', 'native_replay', 'public_FA_LSE'):
                    if kind.startswith(prefix):
                        kind = prefix
                        break
                stages[kind] = stages.get(kind, 0) + stage['seconds']/2
        seconds = statistics.mean(row[mode] for row in rounds)
        stages['outside_controller_timed_stages'] = seconds - sum(stages.values())
        aggregate[mode] = {'mean_complete_seconds_per_16': seconds,
                           'peak_allocated_bytes': max(max(c['allocated_before'], c['peak_allocated']) for c in selected),
                           'stage_mean_seconds_per_16': stages}
    aggregate['speedup'] = aggregate['baseline']['mean_complete_seconds_per_16'] / aggregate['accelerated']['mean_complete_seconds_per_16']
    differences = []
    relative = lambda x, y: float(np.linalg.norm(x-y)/max(np.linalg.norm(y), 1e-30))
    for batch, receipt in enumerate(plan['batches']):
        for key in receipt['cases']:
            b0, b1 = [z[f'measured_r{repeat}_baseline/{key}/{key}'] for repeat in range(2)]
            a0, a1 = [z[f'measured_r{repeat}_accelerated/{batch}/{key}'] for repeat in range(2)]
            saved = qz[key + '_DT_signed_full']
            differences.append({'case': key, 'baseline_repeat_relative_L2': relative(b0, b1),
                                'accelerated_repeat_relative_L2': relative(a0, a1),
                                'accelerated_vs_baseline_relative_L2': relative(a1, b1),
                                'accelerated_vs_quality_saved_relative_L2': relative(a1, saved)})
    summary = {'status': 'complete', 'scope': 'Exact formal task-local development16 groups; warmed complete helper API calls',
               'raw_sha256': sha(raw), 'vectors_sha256': report['vectors_sha256'], 'plan_sha256': report['plan_sha256'],
               'quality_report_sha256': plan['quality_report_sha256'], 'all_72_root_inputs_verified': True,
               'all_96_vectors_verified': True, 'all_48_measured_calls_without_new_dynamo_graphs': True,
               'quality_recomputed': False, 'rounds': rounds, 'aggregate': aggregate, 'vector_differences': differences,
               'memory_scope': report['memory_scope'],
               'cold_phases_seconds': {name: sum(c['seconds'] for n, c in calls.items() if n.startswith(prefix))
                   for name, prefix in [('load', 'model_load'), ('initialization', 'native_eager_initialization'),
                                        ('baseline_warmup', 'warm_baseline/'), ('accelerated_warmup', 'warm_accelerated/')]},
               'quality_interpretation': 'Existing quality is for the same frozen backend and exact groups. Vector differences are descriptive, not a new metric score or a bitwise acceptance gate.'}
    args.output.write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'aggregate': aggregate, 'rounds': rounds}, indent=2))


if __name__ == '__main__':
    main()

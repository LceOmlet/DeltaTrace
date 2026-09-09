"""Verify saved native mutation checks, B2 inputs, costs and score-vector identity."""
import csv
import hashlib
import json
from pathlib import Path
import statistics
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sha = lambda b: hashlib.sha256(b).hexdigest()


def main():
    summaries = {}
    for job, driver, count in [('qwen35_retained_pilot', 'benchmark_qwen35_retained.py', 14),
                               ('qwen35_retained16', 'benchmark_qwen35_retained.py', 48),
                               ('qwen35_retained_extra', 'benchmark_qwen35_retained_extra.py', 18),
                               ('qwen35_retained_api', 'benchmark_qwen35_retained_api.py', 12)]:
        folder = HERE / job; r = json.loads((folder / 'results.json').read_bytes())
        assert r['status'] == 'complete'
        assert r.get('driver_sha256', r.get('script_sha256')) == sha((HERE / driver).read_bytes())
        assert r['vectors_sha256'] == sha((folder / 'vectors.npz').read_bytes())
        for n, digest in r['baseline_sources']['files'].items(): assert sha((ROOT / n).read_bytes()) == digest
        candidate = r['candidate_sources']
        if job.endswith('api'):
            assert candidate['base_deferred'] == r['baseline_sources']
            assert candidate['manifest_sha256'] == sha((ROOT / 'deltatrace/accelerated/retained_qwen35_sources.json').read_bytes())
            for n, digest in candidate['files'].items(): assert sha((ROOT / n).read_bytes()) == digest
        else:
            for n, digest in candidate.items(): assert sha((HERE / n).read_bytes()) == digest
        assert r['FT_calls'] == r['metric_calls'] == r['generation_calls'] == 0
        extra = job.endswith('extra'); mode2 = 'candidate' if extra else 'accelerated'
        calls = [c for c in r['calls'] if '/' in c['name']]
        assert len(calls) == count and all(c['status'] == 'returned' for c in calls)
        z = np.load(folder / 'vectors.npz', allow_pickle=False); assert len(z.files) == 2 * count
        groups = list(enumerate(r['groups'])) if extra else [(i, r['batches'][i]['cases']) for i in r['measured_group_indices']]
        for c in calls:
            detail = c['details']; root = c['root'] if extra else c['actual_root']
            assert root['sample_batch'] == 2 and root['endpoint_batch'] == 4
            assert detail['controller_diagnostic_scheduling']['all_32_finite_checks_passed']
            assert len(detail['layers']) == 32
            assert detail['norm_gate_rules'] == detail['attention_pv_rules'] == {}
            assert detail['finite_fla_by_layer'] == detail['key_norm_by_layer'] == []
            if c['name'].startswith(('measured_', 'r0/', 'r1/')): assert c['compiler_before'] == c['compiler_after']
        rows = []
        for i, keys in groups:
            for key in keys:
                names = [n for n in z.files if n.endswith('/' + key)]
                expected = 7 if job.endswith('pilot') else 6
                assert len(names) == expected and all(np.isfinite(z[n]).all() and np.array_equal(z[names[0]], z[n]) for n in names)
            row = {'group': i, 'cases': ','.join(keys), 'all_complete_vectors_equal': True}
            for mode in ('baseline', mode2):
                measured = [c for c in calls if c['name'] in ([f'r0/{mode}/{i}', f'r1/{mode}/{i}'] if extra else [f'measured_r0_{mode}/{i}', f'measured_r1_{mode}/{i}'])]
                assert len(measured) == 2
                row[mode + '_seconds'] = statistics.mean(c['seconds'] for c in measured)
                row[mode + '_peak_bytes'] = max(max(c['peak_allocated'], c['details']['root_peak_allocated']) for c in measured)
                warm_name = f'warm/{mode}/{i}' if extra else f'warm_{mode}/{i}'
                row[mode + '_first_shape_seconds'] = next(c['seconds'] for c in calls if c['name'] == warm_name)
            row['reduction_fraction'] = 1 - row[mode2 + '_seconds'] / row['baseline_seconds']; rows.append(row)
        if extra:
            frozen = json.loads((folder / 'inputs.json').read_bytes())
            assert r['frozen_inputs_sha256'] == sha((folder / 'inputs.json').read_bytes())
            assert frozen['status'] == 'frozen_before_attribution' and frozen['cases'] == r['cases']
            assert r['input_preparation_native_calls'] == 0
            protocol = json.loads((ROOT / 'experiments/official/protocol.json').read_bytes())
            for c in frozen['cache_references']: assert c['cache_sha256'] == protocol['tasks'][c['task']]['cache_sha256']
            refs = {f"{c['dataset']}_{c['index']}": c for c in frozen['cases']}
            for i, keys in groups:
                n = max(len(refs[k]['input_ids']) for k in keys); eos = refs[keys[0]]['input_ids'][-1]
                packed = np.full((4, n), eos, dtype=np.int64); mask = np.zeros_like(packed)
                for b, key in enumerate(keys):
                    c = refs[key]; ids = np.array(c['input_ids'], dtype=np.int64); assert sha(ids.tobytes()) == c['input_sha256']
                    packed[2*b:2*b+2, :len(ids)] = ids; mask[2*b:2*b+2, :len(ids)] = 1
                    packed[2*b, [c['user_positions'][j] for j in c['keep']]] = eos
                roots = [c['root'] for c in calls if c['name'].endswith('/' + str(i))]
                assert all(sha(packed.tobytes()) == x['input_sha256'] and sha(mask.tobytes()) == x['mask_sha256'] for x in roots)
        else:
            for c in calls: assert c['actual_root'] == r['batches'][int(c['name'].split('/')[-1])]['actual_root']
        total = [sum(row[mode + '_seconds'] for row in rows) for mode in ('baseline', mode2)]
        assert abs(total[0] - r['baseline_seconds_per_pass']) < 1e-12 and abs(total[1] - r['candidate_seconds_per_pass']) < 1e-12
        summary = {'status': 'verified', 'raw_sha256': sha((folder / 'results.json').read_bytes()), 'vectors_sha256': r['vectors_sha256'],
                   'full_DT_calls': count, 'sample_vectors': len(z.files), 'sample_batch': 2, 'endpoint_batch': 4,
                   'baseline_seconds': total[0], 'retained_seconds': total[1], 'reduction_fraction': r['reduction_fraction'],
                   'warm_peak_bytes': {mode: max(row[mode + '_peak_bytes'] for row in rows) for mode in ('baseline', mode2)},
                   'rows': rows, 'quality': 'All complete vectors equal within each paired process. Signed RISE ordering and positive MAS/needle inputs preserved; no metric rerun or quality-gain claim.'}
        if job.endswith('pilot'):
            audit = r['capture_mutation_audit']; assert len(audit) == 128 and all(c['all_values_equal_capture_time'] for c in audit)
            assert sum(c['tensors'] for c in audit) == 2080
            summary['native_mutation_audit'] = {'capture_objects': 128, 'tensor_snapshots': 2080, 'all_unchanged_after_native_layer': True}
        (folder / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', newline='\n')
        with (folder / 'costs.csv').open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        summaries[job] = {k: v for k, v in summary.items() if k != 'rows'}
    for n in ('qwen35_retained_capture.py', 'qwen35_retained_controller.py'):
        assert (HERE / n).read_bytes() == (ROOT / 'deltatrace/accelerated/qwen35' / n).read_bytes()
    (HERE / 'qwen35_retained_summary.json').write_text(json.dumps(summaries, indent=2) + '\n', newline='\n')
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__': main()

"""Pair saved DT results with the author's released per-example FT records.

No attribution, model or metric execution. Require original archive identities,
all cached inputs and token maps, and agreement with the published FT CSV means.
"""
import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sha = lambda data: hashlib.sha256(data).hexdigest()
ARCHIVES = {'traces': '20338014c4afe85355a046e6e61d67e688de8fb80304db3653c3422caaa83416',
            'caches': 'cc2ef9bb177b208aa5295a19d2a5076270a83abb1d3b0f49464b5129a799437e'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--publication', type=Path, required=True)
    p.add_argument('--traces', type=Path, required=True)
    p.add_argument('--caches', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    for name, digest in ARCHIVES.items():
        assert sha(getattr(args, name).read_bytes()) == digest, name
    summary_bytes = (args.publication / 'summary.json').read_bytes()
    summary = json.loads(summary_bytes)
    protocol = json.loads((HERE / 'protocol.json').read_bytes())
    tasks = {row['dataset']: row for row in summary['tasks']}
    receipts = {row['dataset']: row for row in summary['receipts']}
    assert set(tasks) == set(receipts) and len(tasks) == summary['completed_tasks']
    caches = {}
    with tarfile.open(args.caches, 'r:gz') as archive:
        for member in archive:
            name = PurePosixPath(member.name)
            if member.isfile() and name.suffix == '.jsonl' and name.stem in tasks:
                raw = archive.extractfile(member).read()
                assert name.stem not in caches
                assert sha(raw) == protocol['tasks'][name.stem]['cache_sha256']
                caches[name.stem] = [json.loads(line) for line in raw.splitlines() if line]
    assert set(caches) == set(tasks)
    trace_data = {}
    with tarfile.open(args.traces, 'r:gz') as archive:
        for member in archive:
            parts = PurePosixPath(member.name).parts
            if (member.isfile() and len(parts) >= 4 and parts[-4].endswith('.jsonl')
                    and parts[-4][:-6] in tasks and parts[-3] == 'qwen-8B'
                    and parts[-2].startswith('ifr_multi_hop_both_n1_')
                    and (parts[-1] == 'manifest.jsonl' or parts[-1].endswith('.npz'))):
                trace_data[member.name] = archive.extractfile(member).read()
    pairs, aggregates, audit = [], [], []
    for task, table in tasks.items():
        receipt = receipts[task]
        count = protocol['tasks'][task]['count']
        assert table['count'] == receipt['count'] == len(caches[task]) == count
        compressed = (args.publication / receipt['results']).read_bytes()
        assert sha(compressed) == receipt['compressed_sha256']
        raw = gzip.decompress(compressed)
        assert sha(raw) == receipt['results_sha256']
        report = json.loads(raw)
        assert report['status'] == 'complete' and report['family'] == 'qwen3'
        dt = {row['index']: row for row in report['cases']}
        manifest_names = [name for name in trace_data if f'/{task}.jsonl/' in name and name.endswith('/manifest.jsonl')]
        assert len(manifest_names) == 1
        manifest_name = manifest_names[0]
        manifest_raw = trace_data[manifest_name]
        records = [json.loads(line) for line in manifest_raw.splitlines() if line]
        ft = {row['example_idx']: row for row in records}
        assert len(records) == len(report['cases']) == count and set(ft) == set(dt) == set(range(count))
        source_rows = []
        values = {key: [] for key in ('rise', 'mas', 'needle')}
        for index in range(count):
            left, right, cache = dt[index], ft[index], caches[task][index]
            assert left['status'] == 'complete' and right['attr_func'] == 'ifr_multi_hop_both'
            for field in ('prompt', 'target'):
                assert right[field + '_sha1'] == hashlib.sha1(cache[field].encode()).hexdigest()
            # Author traces store P in the user-token attribution matrix (P+G),
            # whereas the DT root length includes the chat-template prefix.
            assert len(left['user_positions']) == right['prompt_len']
            assert left['target_length'] == right['gen_len']
            assert sha(np.asarray(left['input_ids'], dtype=np.int64).tobytes()) == left['input_sha256']
            member = str(PurePosixPath(manifest_name).parent / right['file'])
            npz = trace_data[member]
            with np.load(io.BytesIO(npz), allow_pickle=False) as z:
                for name, field in [('user_prompt_indices', 'user_positions'), ('keep_prompt_token_indices', 'keep'),
                                    ('gold_prompt_token_indices', 'gold')]:
                    expected = left[field] if left[field] is not None else []
                    actual = z[name].tolist() if name in z else []
                    assert actual == expected, (task, index, field)
                assert np.array_equal(z['faithfulness_scores'], np.asarray(right['faithfulness_scores']))
                if right.get('recovery_scores') is not None:
                    assert np.array_equal(z['recovery_scores'], np.asarray(right['recovery_scores']))
            refs = {'rise': right['faithfulness_scores'][0][0], 'mas': right['faithfulness_scores'][0][1],
                    'needle': right['recovery_scores'][0] if right.get('recovery_scores') is not None else None}
            item = {'dataset': task, 'index': index, 'input_sha256': left['input_sha256']}
            for key in values:
                a, b = left['metrics']['DT'][key], refs[key]
                assert (a is None) == (b is None), (task, index, key)
                if a is not None:
                    assert np.isfinite([a, b]).all()
                    values[key].append((a, b))
                item.update({f'DT_{key}': a, f'FT_{key}': b, f'DT_minus_FT_{key}': a-b if a is not None else None})
            pairs.append(item)
            source_rows.append({'index': index, 'npz_member': member, 'npz_sha256': sha(npz),
                                'prompt_sha1': right['prompt_sha1'], 'target_sha1': right['target_sha1']})
        references = protocol['tasks'][task]['FT_references']
        reference_rows = {}
        for kind, row in references.items():
            reference_raw = (ROOT / row['path']).read_bytes()
            assert sha(reference_raw) == row['sha256']
            reference_rows[kind] = {r['Method']: r for r in csv.DictReader(io.StringIO(reference_raw.decode('utf-8')))}
        csv_means = {'rise': float(reference_rows['faithfulness']['Seq Attr Scores Mean']['RISE']),
                     'mas': float(reference_rows['faithfulness']['Seq Attr Scores Mean']['MAS']), 'needle': None}
        if 'recovery' in reference_rows:
            recovery = reference_rows['recovery']
            assert int(recovery['Examples Used']['Recovery@10%']) == count
            assert int(recovery['Examples Skipped']['Recovery@10%']) == 0
            csv_means['needle'] = float(recovery['Seq Attr Recovery Mean']['Recovery@10%'])
        for key, rows in values.items():
            if not rows:
                assert table['DT'][key] is None and table['published_FT'][key] is None
                continue
            a, b = np.asarray(rows).T
            assert len(a) == count
            assert abs(float(a.mean()) - table['DT'][key]) <= 1e-12
            assert abs(float(b.mean()) - table['published_FT'][key]) <= 1e-12
            assert abs(float(b.mean()) - csv_means[key]) <= 1e-12
            delta = a-b
            better = delta if key == 'needle' else -delta
            aggregates.append({'dataset': task, 'metric': key, 'count': count,
                               'DT_mean': float(a.mean()), 'FT_mean': float(b.mean()),
                               'mean_DT_minus_FT': float(delta.mean()), 'median_DT_minus_FT': float(np.median(delta)),
                               'p10_DT_minus_FT': float(np.quantile(delta, .1)),
                               'p90_DT_minus_FT': float(np.quantile(delta, .9)),
                               'DT_better_cases': int((better > 0).sum()), 'equal_cases': int((better == 0).sum()),
                               'DT_worse_cases': int((better < 0).sum())})
        audit.append({'dataset': task, 'count': count, 'manifest_member': manifest_name,
                      'manifest_sha256': sha(manifest_raw), 'trace_run_tag': PurePosixPath(manifest_name).parent.name,
                      'DT_results_sha256': receipt['results_sha256'], 'case_sources': source_rows,
                      'all_input_texts_lengths_and_token_maps_verified': True,
                      'all_trace_metric_means_match_published_CSV': True})
    assert len(pairs) == summary['completed_cases']
    args.output.mkdir(parents=True, exist_ok=True)
    for filename, rows in [('paired_cases.csv', pairs), ('paired_summary.csv', aggregates)]:
        temp = args.output / (filename + '.partial')
        with temp.open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        temp.replace(args.output / filename)
    result = {'status': summary['status'], 'completed_tasks': len(tasks), 'completed_cases': len(pairs),
              'publication_summary_sha256': sha(summary_bytes), 'script_sha256': sha(Path(__file__).read_bytes()),
              'archive_sha256': ARCHIVES, 'model_calls': 0, 'metric_calls': 0, 'tasks': audit,
              'output_sha256': {name: sha((args.output/name).read_bytes()) for name in ('paired_cases.csv', 'paired_summary.csv')},
              'interpretation': 'Descriptive paired differences on all completed original tasks. Strict numerical better/equal/worse counts, not significance tests.',
              'recovery_provenance': 'The paired released records and matching recovery CSV means are in n1 trace directories. The release README says K=3 recovery; that documentation differs from these stored run tags. Do not describe reused CSV values as verified K=3 execution.'}
    (args.output / 'paired_audit.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'tasks': len(tasks), 'cases': len(pairs), 'output': str(args.output),
                      'all_sources_inputs_and_published_means_verified': True}))


if __name__ == '__main__':
    main()

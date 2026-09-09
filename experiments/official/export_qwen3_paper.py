"""Validate and package completed original Table 1 DT tasks for paper tables.

This reads saved metrics and calls the existing summarizer; no attribution or
metric is recomputed. Raw curves and vectors are retained in compressed files.
"""
import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from summarize import summarize

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sha = lambda data: hashlib.sha256(data).hexdigest()
VIEWS = {'rise': 'signed', 'mas': 'positive_part', 'needle': 'positive_part'}


def validate_task(report, arrays, task, count, protocol):
    assert report['status'] == 'complete'
    assert report['family'] == 'qwen3' and report['selection'] == 'paper'
    assert report['selected_counts'] == {task: count}
    assert report['dt_backend'] == 'clean' and report['sample_batch'] == 1
    assert report['generation_calls'] == 0 and report['weight_identity']
    assert report['protocol_sha256'] == sha((HERE / 'protocol.json').read_bytes())
    assert report['driver_sha256'] == sha((HERE / 'evaluate.py').read_bytes())
    assert report['score_views_sha256'] == sha((HERE / 'score_views.py').read_bytes())
    assert report['clean_sources_sha256'] == sha((ROOT / 'deltatrace/clean/sources.json').read_bytes())
    assert len(report['cases']) == count
    expected_vectors = set()
    for row in report['cases']:
        key = f"{task}_{row['index']}"
        assert row['status'] == 'complete' and row['dataset'] == task
        assert row['input_matches_unmodified_author_evaluator']
        assert row['metrics']['DT']['views'] == VIEWS
        assert 'FT_K1' not in row['metrics'], 'This run reuses published FT.'
        assert row['published_FT_reference'] == protocol['tasks'][task]['FT_references']
        ids = np.asarray(row['input_ids'], dtype=np.int64)
        assert sha(ids.tobytes()) == row['input_sha256']
        assert len(ids) == row['prompt_length'] + row['target_length']
        signed_key, positive_key = key + '_DT_signed_full', key + '_DT_positive_prompt'
        expected_vectors.update((signed_key, positive_key))
        signed, positive = arrays[signed_key], arrays[positive_key]
        assert signed.shape == ids.shape and np.isfinite(signed).all()
        signed_prompt = signed[row['user_positions']].astype(np.float32)
        assert np.array_equal(positive, np.maximum(signed_prompt, 0))
        assert set(row['keep']).issubset(range(len(row['user_positions'])))
        for method in ('DT_positive', 'DT_signed'):
            if method not in row['metrics']:
                continue
            curve = row['metrics'][method]
            assert len(curve['actual_input_hashes']) == len(curve['deleted_user_indices']) == 21
            assert curve['actual_input_hashes'][0] == row['input_sha256']
            for deleted, actual_hash in zip(curve['deleted_user_indices'], curve['actual_input_hashes']):
                assert len(deleted) == len(set(deleted))
                assert set(deleted).issubset(row['keep'])
                masked = ids.copy()
                # The released complete target is followed by the tokenizer EOS.
                masked[[row['user_positions'][j] for j in deleted]] = ids[-1]
                assert sha(masked.tobytes()) == actual_hash
                assert np.array_equal(masked[row['prompt_length']:], ids[row['prompt_length']:])
            for field in ('scores', 'density', 'normalized_model_response', 'alignment_penalty', 'corrected_scores'):
                assert np.isfinite(np.asarray(curve[field])).all()
        metric = row['metrics']['DT']
        assert metric['mas'] == row['metrics']['DT_positive']['mas']
        assert metric['needle'] == row['metrics']['DT_positive']['needle']
        if metric['signed_RISE_reuse_proof'] is None:
            assert metric['rise'] == row['metrics']['DT_signed']['rise']
            assert row['metrics']['DT_signed']['MAS_is_valid_for_this_view'] is False
        else:
            assert metric['rise'] == row['metrics']['DT_positive']['rise']
    assert set(arrays.files) == expected_vectors
    return summarize(report, protocol)['tasks'][0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-partial', action='store_true', help='Export only complete tasks as explicitly incomplete progress.')
    args = parser.parse_args()
    state_raw = (args.run / 'run.json').read_bytes()
    state = json.loads(state_raw)
    protocol = json.loads((HERE / 'protocol.json').read_bytes())
    assert state['family'] == 'qwen3' and state['selection'] == 'paper'
    assert state['tasks'] == list(protocol['tasks'])
    assert state['expected_cases'] == sum(row['count'] for row in protocol['tasks'].values())
    for name, digest in state['sources'].items():
        path = ROOT / 'deltatrace/clean/sources.json' if name == 'clean_sources.json' else HERE / name
        assert sha(path.read_bytes()) == digest, name
    complete = [a for a in state['attempts'] if a['status'] == 'complete']
    assert len({a['dataset'] for a in complete}) == len(complete)
    full = state['status'] == 'complete' and len(complete) == len(state['tasks'])
    if not full and not args.allow_partial:
        raise ValueError('All 13 complete tasks are required for the final paper table.')
    args.output.mkdir(parents=True, exist_ok=True)
    raw_dir = args.output / 'raw'
    raw_dir.mkdir(exist_ok=True)
    tables, cases, receipts = [], [], []
    for task, spec in protocol['tasks'].items():
        matching = [a for a in complete if a['dataset'] == task]
        if not matching:
            continue
        attempt = matching[0]
        result_path = args.run / attempt['result']
        result_raw = result_path.read_bytes()
        vector_raw = (result_path.parent / 'vectors.npz').read_bytes()
        assert sha(result_raw) == attempt['results_sha256']
        assert sha(vector_raw) == attempt['vectors_sha256']
        report = json.loads(result_raw)
        assert sha(vector_raw) == report['vectors_sha256']
        for kind, reference in spec['FT_references'].items():
            raw = (ROOT / reference['path']).read_bytes()
            assert sha(raw) == reference['sha256']
            rows = list(csv.reader(io.StringIO(raw.decode('utf-8-sig'))))
            label = 'Seq Attr Scores Mean' if kind == 'faithfulness' else 'Seq Attr Recovery Mean'
            reference_row = next(r for r in rows if r[0] == label)
            assert [float(v) for v in reference_row[1:]] == reference['seq_values']
        with np.load(io.BytesIO(vector_raw), allow_pickle=False) as arrays:
            table = validate_task(report, arrays, task, spec['count'], protocol)
        tables.append(table)
        for row in report['cases']:
            metric = row['metrics']['DT']
            cases.append({'dataset': task, 'index': row['index'], 'input_sha256': row['input_sha256'],
                          'prompt_length': row['prompt_length'], 'target_length': row['target_length'],
                          'DT_RISE': metric['rise'], 'DT_MAS': metric['mas'], 'DT_Recall10': metric['needle'],
                          'signed_RISE_source': 'proof' if metric['signed_RISE_reuse_proof'] else 'original_signed_call'})
        gz_name, npz_name = task + '.results.json.gz', task + '.vectors.npz'
        compressed = gzip.compress(result_raw, mtime=0)
        (raw_dir / gz_name).write_bytes(compressed)
        (raw_dir / npz_name).write_bytes(vector_raw)
        receipts.append({'dataset': task, 'count': spec['count'],
                         'results': 'raw/' + gz_name, 'compressed_sha256': sha(compressed),
                         'results_sha256': sha(result_raw), 'vectors': 'raw/' + npz_name,
                         'vectors_sha256': sha(vector_raw), 'input_hashes_saved': True,
                         'all_case_vectors_curves_views_verified': True,
                         'measured_calls_seconds_including_cold': sum(c['seconds'] for c in report['costs']),
                         'peak_allocated_bytes_all_calls': max(c['peak_allocated'] for c in report['costs']),
                         'weight_identity': report['weight_identity']})
    status = 'complete' if full else 'partial'
    stem = 'table' if full else 'table_progress'
    fields = ['dataset', 'count', 'DT_RISE', 'FT_RISE', 'DT_MAS', 'FT_MAS', 'DT_Recall10', 'FT_Recall10']
    with (args.output / (stem + '.csv')).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for table in tables:
            writer.writerow({'dataset': table['dataset'], 'count': table['count'],
                             **{label + '_' + name: table[key][field] for label, key in
                                [('DT', 'DT'), ('FT', 'published_FT')] for name, field in
                                [('RISE', 'rise'), ('MAS', 'mas'), ('Recall10', 'needle')]}})
    if cases:
        with (args.output / 'per_case.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(cases[0]))
            writer.writeheader()
            writer.writerows(cases)
    latex = ['% ' + status.upper() + ': Qwen3-8B clean DT; original full-task FT CSVs.',
             '% RISE/MAS lower; Recall@10% higher. Values are fractions, not percentages.',
             r'\begin{tabular}{lrrrrrrr}', r'\toprule',
             r'Task & $n$ & DT RISE $\downarrow$ & FT RISE $\downarrow$ & DT MAS $\downarrow$ & FT MAS $\downarrow$ & DT Rec. $\uparrow$ & FT Rec. $\uparrow$ \\', r'\midrule']
    for table in tables:
        values = []
        for field in ('rise', 'mas', 'needle'):
            left, right = table['DT'][field], table['published_FT'][field]
            for value, other in ((left, right), (right, left)):
                if value is None:
                    values.append('--')
                else:
                    win = other is not None and (value > other if field == 'needle' else value < other)
                    text = f'{value:.4f}'
                    values.append(r'\textbf{' + text + '}' if win else text)
        latex.append(table['dataset'].replace('_', r'\_') + ' & ' + str(table['count']) + ' & ' + ' & '.join(values) + r' \\')
    latex.extend([r'\bottomrule', r'\end{tabular}', ''])
    (args.output / (stem + '.tex')).write_text('\n'.join(latex), encoding='utf-8')
    package = {'status': status, 'family': 'qwen3', 'model': 'Qwen3-8B', 'selection': 'paper',
               'completed_tasks': len(tables), 'expected_tasks': len(state['tasks']),
               'completed_cases': len(cases), 'expected_cases': state['expected_cases'],
               'method_tag': 'clean-v1-20260909', 'evaluation_base_tag': 'clean-v1-eval3-native-batch-20260909',
               'sources': state['sources'], 'exporter_sha256': sha(Path(__file__).read_bytes()),
               'metric_views': VIEWS, 'FT_source': 'unchanged author full-task CSVs',
               'metric_direction': {'rise': 'lower', 'mas': 'lower', 'needle': 'higher'},
               'tasks': tables, 'receipts': receipts, 'run_state_sha256': sha(state_raw)}
    (args.output / 'summary.json').write_text(json.dumps(package, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    (args.output / 'run.json').write_bytes(state_raw)
    print(json.dumps({'status': status, 'completed_tasks': len(tables), 'completed_cases': len(cases), 'output': str(args.output)}))


if __name__ == '__main__':
    main()

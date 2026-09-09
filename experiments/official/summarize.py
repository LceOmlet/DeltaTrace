"""Aggregate saved original-metric outputs; never reimplement a metric or model."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def summarize(report, protocol):
    if report['status'] != 'complete':
        raise ValueError('An incomplete or failed run cannot produce a completed table.')
    published = report['family'] == 'qwen3' and report['selection'] == 'paper'
    if published and not report.get('weight_identity'):
        raise ValueError('Original-model weight verification is required for a published-number comparison.')
    cases = report['cases']
    keys = [(row['dataset'], row['index']) for row in cases]
    expected = {(task, i) for task, count in report['selected_counts'].items() for i in range(count)}
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError('Missing, duplicated, or unexpected cases.')

    def mean(rows, method, field):
        values = [row['metrics'][method][field] for row in rows]
        if all(value is None for value in values):
            return None
        if any(value is None or not math.isfinite(value) for value in values):
            raise ValueError('Mixed missing or nonfinite metric values.')
        return statistics.mean(values)

    tables = []
    for task, count in report['selected_counts'].items():
        rows = [row for row in cases if row['dataset'] == task]
        if any(row['status'] != 'complete' for row in rows):
            raise ValueError('An unfinished case cannot enter a completed table.')
        table = {'dataset': task, 'count': count,
                 'DT': {field: mean(rows, 'DT', field) for field in ('rise', 'mas', 'needle')}}
        legacy_views = {field: 'positive_part' for field in ('rise', 'mas', 'needle')}
        views = rows[0]['metrics']['DT'].get('views', legacy_views)
        if any(row['metrics']['DT'].get('views', legacy_views) != views for row in rows):
            raise ValueError('Mixed DT metric views cannot enter one task mean.')
        table['DT_score_views'] = views
        live = ['FT_K1' in row['metrics'] for row in rows]
        if any(live) and not all(live):
            raise ValueError('Partial live FT comparison.')
        if all(live):
            table['live_FT_K1'] = {field: mean(rows, 'FT_K1', field) for field in ('rise', 'mas')}
            if table['DT']['needle'] is not None:
                table['live_FT_K3_needle'] = statistics.mean(row['FT_K3_needle'] for row in rows)
        if published:
            reference = protocol['tasks'][task]
            if count != reference['count']:
                raise ValueError('A task subset cannot be compared with a published full-task mean.')
            refs = reference['FT_references']
            rise, mas, _ = refs['faithfulness']['seq_values']
            table['published_FT'] = {'rise': rise, 'mas': mas,
                'needle': refs['recovery']['seq_values'][0] if 'recovery' in refs else None}
            table['reference_files'] = {kind: value['path'] for kind, value in refs.items()}
        tables.append(table)
    return {'family': report['family'], 'selection': report['selection'],
            'published_number_comparison': published, 'tasks': tables,
            'metric_direction': {'rise': 'lower', 'mas': 'lower', 'needle': 'higher'},
            'sample_batch': report['sample_batch'],
            'dt_backend': report.get('dt_backend','clean'),
            'measured_call_seconds_including_cold_calls': sum(row['seconds'] for row in report['costs'])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    args = parser.parse_args()
    report = json.loads(args.results.read_bytes())
    protocol_raw = (HERE / 'protocol.json').read_bytes()
    protocol = json.loads(protocol_raw)
    if report['selection'] == 'paper':
        if report['protocol_sha256'] != hashlib.sha256(protocol_raw).hexdigest():
            raise ValueError('Use the exact protocol release recorded by the paper run.')
    if hashlib.sha256((args.results.parent/'vectors.npz').read_bytes()).hexdigest() != report['vectors_sha256']:
        raise ValueError('Saved attribution vectors do not match the completed run.')
    for task in report['selected_counts']:
        for ref in protocol['tasks'][task]['FT_references'].values():
            if hashlib.sha256((ROOT/ref['path']).read_bytes()).hexdigest() != ref['sha256']:
                raise ValueError('The published reference CSV changed.')
    print(json.dumps(summarize(report, protocol), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()

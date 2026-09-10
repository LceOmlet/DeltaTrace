"""Aggregate versioned evaluation outputs, refusing mixed scopes or budgets."""
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
    evaluation_protocol = report.get('evaluation_protocol', 'released-v1')
    if evaluation_protocol not in ('released-v1', 'source-v2'):
        raise ValueError('Unknown evaluation protocol.')
    source_mode = evaluation_protocol == 'source-v2'
    published = not source_mode and report['family'] == 'qwen3' and report['selection'] == 'paper'
    if report['family'] == 'qwen3' and report['selection'] == 'paper' and not report.get('weight_identity'):
        raise ValueError('Original-model weight verification is required for a full Qwen3 table.')
    if source_mode:
        source_raw = (HERE / 'source_protocol.json').read_bytes()
        if (report.get('evaluation_settings') != json.loads(source_raw)
                or report.get('evaluation_protocol_sha256') != hashlib.sha256(source_raw).hexdigest()):
            raise ValueError('Use the exact source-v2 protocol recorded by the run.')
        if report.get('ft_source') != 'live' or report.get('published_number_comparison'):
            raise ValueError('source-v2 requires live matched FT and cannot use published full-prompt numbers.')
    cases = report['cases']
    keys = [(row['dataset'], row['index']) for row in cases]
    expected = {(task, i) for task, count in report['selected_counts'].items() for i in range(count)}
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError('Missing, duplicated, or unexpected cases.')
    if not cases:
        raise ValueError('An empty run cannot produce a completed table.')
    for row in cases:
        if row.get('evaluation_protocol', 'released-v1') != evaluation_protocol:
            raise ValueError('Cannot mix cases from different evaluation protocols.')
        if source_mode:
            if row.get('published_FT_reference') or 'FT_K1' not in row['metrics']:
                raise ValueError('source-v2 needs a live FT result for every case.')
            if (not row.get('source_span') or row.get('source_eligible_count') != len(row['keep'])
                    or row.get('source_gold_count') != len(set(row['gold']) & set(row['keep']))
                    or not row['keep'] or row['keep'] != sorted(set(row['keep']))
                    or not set(row['keep']) <= set(row.get('author_keep', []))
                    or not (set(row['gold']) & set(row['keep']))
                    or (set(row['gold']) & set(row.get('author_keep', []))) - set(row['keep'])
                    or row.get('source_eligible_positions') != [row['user_positions'][j] for j in row['keep']]):
                raise ValueError('Missing or inconsistent source token mapping.')
            for method in ('DT', 'FT_K1'):
                metric = row['metrics'][method]
                if (not metric.get('reference_matches_final_deletion')
                        or len(metric['actual_input_hashes']) != min(20, len(row['keep'])) + 1
                        or metric['actual_input_hashes'][0] != row['input_sha256']
                        or metric['actual_input_hashes'][-1] != row['reference_input_sha256']):
                    raise ValueError('DT reference and deletion endpoint must match.')

    def mean(rows, method, field):
        values = [row['metrics'][method][field] for row in rows]
        if all(value is None for value in values):
            return None
        if any(value is None or not math.isfinite(value) for value in values):
            raise ValueError('Mixed missing or nonfinite metric values.')
        return statistics.mean(values)

    def mean_curve(curves, *, sentence=False):
        expected_unit = 'sentence_or_line_units' if sentence else 'source_eligible_tokens'
        fractions = report['evaluation_settings']['budget_fractions']
        if any(c['budget_unit'] != expected_unit or [p['fraction'] for p in c['points']] != fractions for c in curves):
            raise ValueError('Mismatched recovery budget units or fractions.')
        for curve in curves:
            if sentence:
                if curve.get('selection') != 'whole_units' or curve.get('score_view') != 'mean_positive_source_token_score':
                    raise ValueError('Sentence recovery must retrieve whole units with the fixed score view.')
            elif curve.get('score_view') != 'positive_part' or curve.get('tie_break') != 'ascending_prompt_token_index':
                raise ValueError('Token recovery must use the fixed score view and tie break.')
            for point in curve['points']:
                if (point['eligible'] <= 0 or not 0 < point['gold'] <= point['eligible']
                        or point['budget'] != max(1, math.ceil(point['fraction'] * point['eligible']))):
                    raise ValueError('Recovery budget must be recomputed from the current eligible units.')
        fields = ['recall', 'precision', 'ceiling', 'random_expected_recall', 'ceiling_adjusted_recall', 'eligible', 'gold', 'budget']
        fields += ['selected_token_count', 'selected_token_fraction'] if sentence else ['chance_adjusted_recall']
        points = []
        for i, fraction in enumerate(fractions):
            point = {'fraction': fraction, 'valid_counts': {}}
            for field in fields:
                values = [c['points'][i][field] for c in curves]
                # Chance-adjustment is undefined if every eligible token is gold.
                valid = [v for v in values if v is not None]
                if any(not math.isfinite(v) for v in valid) or (field != 'chance_adjusted_recall' and len(valid) != len(values)):
                    raise ValueError('Missing or nonfinite recovery curve values.')
                point[field] = statistics.mean(valid) if valid else None
                point['valid_counts'][field] = len(valid)
            points.append(point)
        return {'budget_unit': expected_unit, 'aggregation': 'mean_per_case', 'points': points}

    tables = []
    for task, count in report['selected_counts'].items():
        rows = [row for row in cases if row['dataset'] == task]
        if any(row['status'] != 'complete' for row in rows):
            raise ValueError('An unfinished case cannot enter a completed table.')
        if source_mode and report['selection'] == 'paper' and count != protocol['tasks'][task]['count']:
            raise ValueError('A source-v2 paper task must include its full released cache.')
        table = {'dataset': task, 'count': count,
                 'DT': {field: mean(rows, 'DT', field) for field in ('rise', 'mas', 'needle')}}
        live = ['FT_K1' in row['metrics'] for row in rows]
        if any(live) and not all(live):
            raise ValueError('Partial live FT comparison.')
        if all(live):
            table['live_FT_K1'] = {field: mean(rows, 'FT_K1', field) for field in ('rise', 'mas')}
            if table['DT']['needle'] is not None:
                table['live_FT_K3_needle'] = statistics.mean(row['FT_K3_needle'] for row in rows)
        if source_mode:
            table['recovery'] = {
                'DT': mean_curve([r['metrics']['DT']['recovery'] for r in rows]),
                'FT_K1': mean_curve([r['metrics']['FT_K1']['recovery'] for r in rows]),
                'FT_K3': mean_curve([r['FT_K3_recovery'] for r in rows]),
            }
            for r in rows:
                for metric in [r['metrics']['DT'], r['metrics']['FT_K1'], {'needle': r['FT_K3_needle'], 'recovery': r['FT_K3_recovery']}]:
                    at10 = next(p for p in metric['recovery']['points'] if p['fraction'] == .1)
                    if (at10['recall'] != metric['needle'] or any(
                            point['eligible'] != len(r['keep']) or point['gold'] != r['source_gold_count']
                            for point in metric['recovery']['points'])):
                        raise ValueError('Needle value or budget differs from the source-v2 curve.')
            if report.get('sentence_recovery_enabled'):
                table['sentence_recovery'] = {
                    'DT': mean_curve([r['metrics']['DT']['sentence_recovery'] for r in rows], sentence=True),
                    'FT_K1': mean_curve([r['metrics']['FT_K1']['sentence_recovery'] for r in rows], sentence=True),
                    'FT_K3': mean_curve([r['FT_K3_sentence_recovery'] for r in rows], sentence=True),
                }
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
            'evaluation_protocol': evaluation_protocol,
            'evaluation_protocol_sha256': report.get('evaluation_protocol_sha256', report['protocol_sha256']),
            'sentence_recovery_enabled': report.get('sentence_recovery_enabled', False),
            'published_number_comparison': published, 'tasks': tables,
            'metric_direction': {'rise': 'lower', 'mas': 'lower', 'needle': 'higher'},
            'sample_batch': report['sample_batch'],
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
    if report.get('evaluation_protocol') == 'source-v2':
        if report.get('evaluation_protocol_sha256') != hashlib.sha256((HERE / 'source_protocol.json').read_bytes()).hexdigest():
            raise ValueError('Source evaluation protocol changed; do not mix releases.')
    if hashlib.sha256((args.results.parent/'vectors.npz').read_bytes()).hexdigest() != report['vectors_sha256']:
        raise ValueError('Saved attribution vectors do not match the completed run.')
    for task in report['selected_counts']:
        for ref in protocol['tasks'][task]['FT_references'].values():
            if hashlib.sha256((ROOT/ref['path']).read_bytes()).hexdigest() != ref['sha256']:
                raise ValueError('The published reference CSV changed.')
    print(json.dumps(summarize(report, protocol), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()

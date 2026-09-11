"""Audit all 1,048 released cases with needle labels using frozen DT/FT vectors."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'experiments/official'))
from recovery_diagnostics import recovery_diagnostics, reported_recovery_diagnostics


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--publication', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--traces', type=Path, required=True)
    p.add_argument('--output', type=Path, default=Path(__file__).parent)
    a = p.parse_args()
    protocol = json.loads((ROOT / 'experiments/official/protocol.json').read_text(encoding='utf-8'))
    frozen = json.loads((a.publication / 'summary.json').read_text(encoding='utf-8'))
    receipts = {r['dataset']: r for r in frozen['receipts']}
    rows, sources = [], []
    for task in frozen['tasks']:
        name = task['dataset']
        if task['DT']['needle'] is None:
            continue
        cache_path = a.data / (name + '.jsonl')
        assert sha(cache_path) == protocol['tasks'][name]['cache_sha256'], name
        caches = [json.loads(s) for s in cache_path.read_text(encoding='utf-8').splitlines()]
        result_path = a.publication / 'raw' / (name + '.results.json.gz')
        vector_path = a.publication / 'raw' / (name + '.vectors.npz')
        assert sha(result_path) == receipts[name]['compressed_sha256']
        raw = gzip.decompress(result_path.read_bytes())
        assert hashlib.sha256(raw).hexdigest() == receipts[name]['results_sha256']
        assert sha(vector_path) == receipts[name]['vectors_sha256']
        report = json.loads(raw)
        cases = report['cases']
        assert len(cases) == len(caches) == task['count']
        ft_dirs = list((a.traces / (name + '.jsonl') / 'qwen-8B').glob('ifr_multi_hop_both_n1_*'))
        assert len(ft_dirs) == 1
        ft_dir = ft_dirs[0]
        manifest_path = ft_dir / 'manifest.jsonl'
        manifest = [json.loads(s) for s in manifest_path.read_text(encoding='utf-8').splitlines()]
        assert len(manifest) == len(cases)
        ft_records = {r['example_idx']: r for r in manifest}
        vector_file = np.load(vector_path, allow_pickle=False)
        ft_hashes = []
        for case in cases:
            i = case['index']
            cache, ft_record = caches[i], ft_records[i]
            for f in ['prompt', 'target']:
                assert hashlib.sha1(cache[f].encode()).hexdigest() == ft_record[f + '_sha1']
            assert hashlib.sha256(np.asarray(case['input_ids'], dtype=np.int64).tobytes()).hexdigest() == case['input_sha256']
            key = f'{name}_{i}'
            signed = vector_file[key + '_DT_signed_full'][case['user_positions']].astype(np.float32)
            positive = np.maximum(signed, 0)
            assert np.array_equal(positive, vector_file[key + '_DT_positive_prompt'])
            ft_path = ft_dir / ft_record['file']
            with np.load(ft_path, allow_pickle=False) as ft:
                for field, expected in [('user_prompt_indices', case['user_positions']),
                                        ('keep_prompt_token_indices', case['keep']),
                                        ('gold_prompt_token_indices', case['gold'])]:
                    assert ft[field].tolist() == expected, (name, i, field)
                ft_scores = np.maximum(ft['v_seq_prompt'], 0)
                ft_metric = float(ft['recovery_scores'][0])
            ft_hashes.append({'index': i, 'sha256': sha(ft_path)})
            d = recovery_diagnostics(positive, case['keep'], case['gold'])
            f = recovery_diagnostics(ft_scores, case['keep'], case['gold'])
            original = case['metrics']['DT']['needle']
            reported = reported_recovery_diagnostics(positive, case['keep'], case['gold'], original)
            assert reported['reported_recall'] == original
            assert abs(reported['ceiling_adjusted_recall'] - original / d['ceiling']) < 1e-12
            assert d['recall_tie_low'] - 1e-12 <= original <= d['recall_tie_high'] + 1e-12, (name, i, original, d)
            assert f['recall_tie_low'] - 1e-12 <= ft_metric <= f['recall_tie_high'] + 1e-12
            eligible_gold = sorted(set(case['gold']).intersection(case['keep']))
            gold_scores = signed[eligible_gold]
            detail = case['DT_details']
            r = {'dataset': name, 'index': i, 'input_sha256': case['input_sha256'],
                 'DT': original, 'FT': ft_metric, 'DT_minus_FT': original - ft_metric,
                 **{k: d[k] for k in ['eligible', 'gold', 'budget', 'gold_density', 'ceiling', 'random_expected_recall']},
                 'DT_adjusted': original / d['ceiling'], 'FT_adjusted': ft_metric / d['ceiling'],
                 'DT_precision': original * d['gold'] / d['budget'],
                 'FT_precision': ft_metric * d['gold'] / d['budget'],
                 'DT_tie_width': d['recall_tie_high'] - d['recall_tie_low'],
                 'FT_tie_width': f['recall_tie_high'] - f['recall_tie_low'],
                 'DT_cutoff': d['cutoff'], 'gold_negative_fraction': float((gold_scores < 0).mean()),
                 'gold_negative_abs_mass': float(-gold_scores[gold_scores < 0].sum()),
                 'gold_positive_mass': float(np.maximum(gold_scores, 0).sum()),
                 'positive_gold_ceiling': min(d['budget'], int((gold_scores > 0).sum())) / d['gold'],
                 'target_tokens': case['target_length'], 'judge_true': str(cache['metadata'].get('judge_response')).strip() == 'True',
                 'span_count': len(cache['metadata']['needle_spans']),
                 'boundary_diff_count': len(case['standalone_token_boundary_differences']),
                 'gold_boundary_diff_count': len(set(case['standalone_token_boundary_differences']).intersection(eligible_gold)),
                 'relative_conservation_residual': abs(detail['unassigned_total']) / max(abs(detail['target_delta_score32_sum64']), 1e-12),
                 'absolute_conservation_residual': abs(detail['unassigned_total'])}
            for view, scores in [('signed', signed), ('absolute', abs(signed)), ('negative', np.maximum(-signed, 0))]:
                v = recovery_diagnostics(scores, case['keep'], case['gold'])
                r[view + '_recall'] = v['recall']
                r[view + '_tie_width'] = v['recall_tie_high'] - v['recall_tie_low']
            for fraction in [.05, .2, .3, .5]:
                for label, scores in [('DT', positive), ('FT', ft_scores)]:
                    r[f'{label}_recall_{int(fraction*100)}'] = recovery_diagnostics(scores, case['keep'], case['gold'], fraction)['recall']
            # Equal number of predictions and eligible gold tokens: oracle budget,
            # used only to isolate metric capacity, never as a deployed selector.
            for label, scores in [('DT', positive), ('FT', ft_scores)]:
                order = np.asarray(case['keep'])[np.argsort(-scores[case['keep']], kind='stable')]
                r[label + '_recall_gold_budget'] = len(set(order[:d['gold']]).intersection(eligible_gold)) / d['gold']
            rows.append(r)
        vector_file.close()
        sources.append({'dataset': name, 'count': len(cases), 'cache_sha256': sha(cache_path),
                        'results_gz_sha256': sha(result_path), 'vectors_sha256': sha(vector_path),
                        'ft_manifest_sha256': sha(manifest_path), 'ft_run_tag': ft_dir.name, 'ft_npz': ft_hashes})
        print(name, 'done', flush=True)
    aggregates = []
    for name in dict.fromkeys(r['dataset'] for r in rows):
        group = [r for r in rows if r['dataset'] == name]
        agg = {'dataset': name, 'count': len(group)}
        for k in group[0]:
            if k not in ['dataset', 'index', 'input_sha256']:
                agg[k] = float(np.mean([r[k] for r in group]))
        agg['capacity_limited_cases'] = sum(r['ceiling'] < 1 for r in group)
        agg['DT_better_cases'] = sum(r['DT'] > r['FT'] for r in group)
        agg['DT_worse_cases'] = sum(r['DT'] < r['FT'] for r in group)
        for label in ['DT_minus_FT', 'absolute_recall']:
            values = np.array([r[label] if label == 'DT_minus_FT' else r[label] - r['DT'] for r in group])
            rng = np.random.default_rng(20260910)
            boot = values[rng.integers(0, len(values), size=(10000, len(values)))].mean(axis=1)
            agg[label + '_delta_ci_low'], agg[label + '_delta_ci_high'] = map(float, np.quantile(boot, [.025, .975]))
        aggregates.append(agg)
    a.output.mkdir(parents=True, exist_ok=True)
    write_csv(a.output / 'case_diagnostics.csv', rows)
    write_csv(a.output / 'task_diagnostics.csv', aggregates)
    (a.output / 'source_receipt.json').write_text(json.dumps({'cases': len(rows), 'tasks': len(aggregates),
        'model_calls': 0, 'script_sha256': sha(Path(__file__)), 'diagnostic_module_sha256': sha(ROOT / 'experiments/official/recovery_diagnostics.py'),
        'publication_summary_sha256': sha(a.publication / 'summary.json'), 'sources': sources}, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(aggregates, indent=2))


if __name__ == '__main__':
    main()

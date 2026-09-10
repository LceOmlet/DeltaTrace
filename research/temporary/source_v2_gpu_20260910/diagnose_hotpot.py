"""Post-hoc HotpotQA controls on raw saved deletion curves and ranked evidence."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'experiments/official'))
from recovery_diagnostics import recovery_diagnostics
from analyze import paired_bootstrap


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from tokenizers import Tokenizer
    task = 'hotpotqa_long'
    folder = args.run / task
    verified = json.loads((args.analysis / 'analysis.json').read_bytes())
    receipt = next(r for r in verified['source_receipts'] if r['dataset'] == task)
    assert sha(folder / 'results.json') == receipt['results_sha256']
    assert sha(folder / 'vectors.npz') == receipt['vectors_sha256']
    report = json.loads((folder / 'results.json').read_bytes())
    assert report['status'] == 'complete' and len(report['cases']) == 48
    cache_path = args.data / (task + '.jsonl')
    assert sha(cache_path) == next(r['cache_sha256'] for r in verified['data_caches_verified'] if r['dataset'] == task)
    cache = [json.loads(line) for line in cache_path.read_text(encoding='utf-8').splitlines()]
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    assert sha(args.tokenizer) == verified['tokenizer_sha256']
    methods = ['DT', 'DT_full_reference', 'FT_K1']
    rows = []
    for case in report['cases']:
        i = case['index']
        lines = [line.strip() for line in cache[i]['target'].splitlines() if line.strip()]
        delta = case['metrics']['DT']['scores'][0] - case['metrics']['DT']['scores'][-1]
        row = {'index': i, 'target_length': case['target_length'], 'source_endpoint_delta': delta,
               'nonpositive_endpoint_delta': delta <= 0,
               'exact_two_line_duplicate_target': len(lines) == 2 and lines[0] == lines[1],
               'new_reference_FA_target_delta': case['DT_details']['target_delta_score32_sum64'],
               'full_reference_FA_target_delta': case['DT_full_reference_details']['target_delta_score32_sum64']}
        for method in methods:
            curve = case['metrics'][method]
            scores = np.asarray(curve['scores'])
            assert scores[0] - scores[-1] == delta
            x = np.asarray([len(s) / len(case['keep']) for s in curve['deleted_user_indices']])
            drop = (scores[0] - scores) / case['target_length']
            row[method + '_raw_drop_auc'] = float(np.trapezoid(drop, x=x))
            row[method + '_drop_step1'] = float(drop[1])
            row[method + '_drop_step2'] = float(drop[2])
            row[method + '_step1_token_fraction'] = float(x[1])
            row[method + '_step2_token_fraction'] = float(x[2])
            row[method + '_RISE_first_step_floor'] = bool(curve['normalized_model_response'][0] == 1
                and np.all(np.asarray(curve['normalized_model_response'][1:]) == 0))
            row[method + '_recall10'] = curve['recovery']['points'][1]['recall']
            row[method + '_rise'] = curve['rise']
            row[method + '_mas'] = curve['mas']
        rows.append(row)
    fields = [(method, metric) for method in ('DT', 'DT_full_reference')
              for metric in ('raw_drop_auc', 'drop_step1', 'drop_step2')]
    differences = np.asarray([[r[m + '_' + metric] - r['FT_K1_' + metric] for m, metric in fields] for r in rows])
    boot = paired_bootstrap(differences, np.random.default_rng(np.random.SeedSequence(73, spawn_key=(10, 2))))
    contrasts = {m + '_minus_FT_K1_' + metric: {'mean_difference': float(differences[:, j].mean()),
        'ci95': np.quantile(boot[:, j], [.025, .975]).tolist()} for j, (m, metric) in enumerate(fields)}
    selected_indices = [r['index'] for r in sorted(rows, key=lambda r: r['DT_recall10'] - r['DT_full_reference_recall10'])[:3]]
    examples = []
    with np.load(folder / 'vectors.npz', allow_pickle=False) as vectors:
        for i in selected_indices:
            c, item = report['cases'][i], cache[i]
            encoding = tokenizer.encode(' ' + item['prompt'], add_special_tokens=False)
            scores = {m: vectors[f'{task}_{i}_' + (m + '_positive_prompt' if m.startswith('DT') else m + '_prompt')]
                      for m in ('DT', 'DT_full_reference', 'FT_K1')}
            selected = {m: set(recovery_diagnostics(np.maximum(s, 0), c['keep'], c['gold'])['selected']) for m, s in scores.items()}
            support = []
            for span in item['metadata']['needle_spans']:
                a, b = span['span']
                gold = {j for j in set(c['gold']) & set(c['keep']) if encoding.offsets[j][0] < b + 1 and encoding.offsets[j][1] > a + 1}
                support.append({'sentence': span['sentence'], 'document_number': span['document_number'],
                    'eligible_gold_tokens': len(gold), 'recovered_tokens': {m: len(gold & ids) for m, ids in selected.items()}})
            ranked = sorted(c['keep'], key=lambda j: (-float(scores['DT'][j]), j))
            context_lines = []
            for j in ranked:
                if j in c['gold']:
                    continue
                position = max(0, encoding.offsets[j][0] - 1)
                start = item['prompt'].rfind('\n', 0, position) + 1
                end = item['prompt'].find('\n', position)
                line = item['prompt'][start:end if end >= 0 else None].strip()
                if line and line not in context_lines:
                    context_lines.append(line)
                if len(context_lines) == 3:
                    break
            examples.append({'index': i, 'question': item['prompt'].rsplit('Question:', 1)[-1].strip(),
                'target': item['target'], 'supporting_sentences': support,
                'new_DT_top_non_gold_token_context_lines': context_lines})
    output = {'status': 'verified_posthoc_diagnostic', 'case_count': 48,
        'scope': 'Raw model log-probability drop per fixed response token; no endpoint normalization or clipping',
        'interpretation': 'Higher raw drop indicates a larger deletion effect; diagnostic does not replace the original metrics.',
        'source_results_sha256': receipt['results_sha256'], 'script_sha256': sha(Path(__file__)),
        'nonpositive_endpoint_cases': sum(r['nonpositive_endpoint_delta'] for r in rows),
        'exact_duplicate_target_cases': sum(r['exact_two_line_duplicate_target'] for r in rows),
        'first_step_floor_counts': {m: sum(r[m + '_RISE_first_step_floor'] for r in rows) for m in methods},
        'raw_metric_means': {m: {metric: float(np.mean([r[m + '_' + metric] for r in rows]))
                               for metric in ('raw_drop_auc', 'drop_step1', 'drop_step2')} for m in methods},
        'raw_contrasts_vs_FT_K1': contrasts, 'posthoc_worst_recall_drop_examples': examples}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'diagnostic.json').write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    with (args.output / 'cases.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps({k: v for k, v in output.items() if k != 'posthoc_worst_recall_drop_examples'}, indent=2))


if __name__ == '__main__':
    main()

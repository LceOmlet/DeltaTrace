"""CPU preflight on all released recovery cases; never claims new DT quality.

Checks the production source parser, token mapping, reference construction,
budget curves and optional whole-unit metrics against frozen inputs/vectors.
"""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from evidence_protocol import (SUPPORTED_TASKS, source_span, select_source_tokens,
    recovery_curve, sentence_recovery_curve, reference_token_ids)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--publication', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--tokenizer', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(a.tokenizer))
    release = json.loads((HERE / 'protocol.json').read_bytes())
    spec = json.loads((HERE / 'source_protocol.json').read_bytes())
    frozen = json.loads((a.publication / 'summary.json').read_bytes())
    receipts = {r['dataset']: r for r in frozen['receipts']}
    rows, errors, sources = [], [], []
    for dataset in SUPPORTED_TASKS:
        path = a.data / (dataset + '.jsonl')
        assert digest(path) == release['tasks'][dataset]['cache_sha256']
        caches = [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]
        result_path = a.publication / 'raw' / (dataset + '.results.json.gz')
        vector_path = a.publication / 'raw' / (dataset + '.vectors.npz')
        assert digest(result_path) == receipts[dataset]['compressed_sha256']
        assert digest(vector_path) == receipts[dataset]['vectors_sha256']
        report = json.loads(gzip.decompress(result_path.read_bytes()))
        assert len(caches) == len(report['cases']) == release['tasks'][dataset]['count']
        vectors = np.load(vector_path, allow_pickle=False)
        for case in report['cases']:
            i = case['index']
            try:
                prompt = caches[i]['prompt']
                encoding = tokenizer.encode(' ' + prompt, add_special_tokens=False)
                actual_ids = np.asarray(case['input_ids'])[case['user_positions']]
                assert len(encoding.ids) == len(actual_ids)
                assert np.flatnonzero(np.asarray(encoding.ids) != actual_ids).tolist() == case['standalone_token_boundary_differences']
                span = source_span(dataset, prompt)
                keep = select_source_tokens(span, encoding.offsets, case['keep'], case['gold'])
                assert not set(case['standalone_token_boundary_differences']) & set(keep)
                positions = [case['user_positions'][j] for j in keep]
                eos_id = tokenizer.token_to_id('<|im_end|>')
                assert eos_id is not None
                reference = reference_token_ids(case['input_ids'], positions, eos_id)
                outside = sorted(set(range(len(reference))) - set(positions))
                assert np.array_equal(np.asarray(reference)[outside], np.asarray(case['input_ids'])[outside])
                assert all(reference[j] == eos_id for j in positions)
                assert reference[case['prompt_length']:] == case['input_ids'][case['prompt_length']:]
                scores = vectors[f'{dataset}_{i}_DT_positive_prompt']
                curve = recovery_curve(scores, keep, case['gold'], spec['budget_fractions'])
                assert [r['budget'] for r in curve['points']] == [max(1, int(np.ceil(f * len(keep)))) for f in spec['budget_fractions']]
                sentences = sentence_recovery_curve(prompt, span, encoding.offsets, scores, keep, case['gold'], spec['budget_fractions'])
                assert sentences['budget_unit'] != curve['budget_unit']
                assert all(0 < r['selected_token_count'] <= len(keep) for r in sentences['points'])
                rows.append({'dataset': dataset, 'index': i, 'source_start': span['start'], 'source_end': span['end'],
                    'author_eligible': len(case['keep']), 'source_eligible': len(keep),
                    'eligible_gold': len(set(case['gold']) & set(keep)),
                    **{f'budget_{int(r["fraction"]*100)}': r['budget'] for r in curve['points']},
                    'original_10pct_budget': int(np.ceil(.1 * len(case['keep']))),
                    'reference_input_sha256': hashlib.sha256(np.asarray(reference, dtype=np.int64).tobytes()).hexdigest()})
            except Exception as error:
                errors.append({'dataset': dataset, 'index': i, 'error': f'{type(error).__name__}: {error}'})
        vectors.close()
        sources.append({'dataset': dataset, 'cache_sha256': digest(path), 'result_sha256': digest(result_path), 'vectors_sha256': digest(vector_path)})
        print(json.dumps({'dataset': dataset, 'completed': sum(r['dataset'] == dataset for r in rows),
                          'errors': sum(r['dataset'] == dataset for r in errors)}), flush=True)
    a.output.mkdir(parents=True, exist_ok=True)
    if rows:
        with (a.output / 'source_preflight_cases.csv').open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    receipt = {'status': 'passed' if not errors and len(rows) == 1048 else 'failed', 'checked_cases': len(rows),
        'expected_cases': 1048, 'errors': errors, 'model_calls': 0,
        'interpretation': 'Production parser/budget/reference preflight on frozen vectors; not a source-v2 model evaluation.',
        'new_DT_quality_measured': False, 'tokenizer_sha256': digest(a.tokenizer), 'sources': sources,
        'code_sha256': {name: digest(HERE / name) for name in ['validate_source_protocol.py', 'source_protocol.json', 'evidence_protocol.py', 'recovery_diagnostics.py']}}
    (a.output / 'source_preflight.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': receipt['status'], 'checked_cases': len(rows), 'errors': errors[:12]}, indent=2))
    if receipt['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()

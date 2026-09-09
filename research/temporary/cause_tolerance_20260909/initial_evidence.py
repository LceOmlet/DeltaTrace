"""Audit existing clean development evidence without any model/metric calls."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sha = lambda data: hashlib.sha256(data).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshots', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    path = ROOT / 'research/temporary/acceleration_20260909/clean_signed_views32.json'
    data = path.read_bytes()
    assert sha(data) == '5a21dac6207eb25b60b4c7db98ca7055a3ea46fce62158267f9f8ee0b97cb7c5'
    evidence = json.loads(data)
    assert evidence['views'] == {'rise': 'signed', 'mas': 'positive_part', 'needle': 'positive_part'}
    raw_cases, raw_vectors = {}, {}
    references = []
    for ref in evidence['references']:
        source = args.snapshots / ref['path'].lstrip('/')
        raw, vectors = source.read_bytes(), (source.parent / 'vectors.npz').read_bytes()
        assert sha(raw) == ref['sha256'] and sha(vectors) == ref['vectors_sha256']
        report = json.loads(raw)
        with np.load(source.parent / 'vectors.npz', allow_pickle=False) as z:
            for row in report['cases']:
                if row['status'] != 'complete':
                    continue
                key = (ref['family'], row['dataset'], row['index'])
                assert key not in raw_cases
                raw_cases[key] = row
                name = f"{row['dataset']}_{row['index']}_DT_signed_full"
                raw_vectors[key] = sha(z[name].tobytes())
        references.append(ref)
    means, cases = [], []
    for family, rows in evidence['models'].items():
        assert {(r['dataset'], r['index']) for r in rows} == {
            (task, i) for task in ('niah_mq_q2', 'morehopqa') for i in range(8)}
        for row in rows:
            key = (family, row['dataset'], row['index'])
            original = raw_cases[key]
            assert row['source_vector_sha256'] == raw_vectors[key]
            assert row['input_sha256'] == original['input_sha256'] == sha(np.asarray(original['input_ids'], dtype=np.int64).tobytes())
            assert row['DT']['mas'] == original['metrics']['DT']['mas']
            assert row['DT']['needle'] == original['metrics']['DT']['needle']
            cases.append({'family': family, 'dataset': row['dataset'], 'index': row['index'],
                          'input_sha256': row['input_sha256'], 'prompt_tokens': original['prompt_length'],
                          'target_tokens': original['target_length'], 'eligible_tokens': len(original['keep']),
                          'DT_RISE': row['DT']['rise'], 'FT_K1_RISE': row['FT_K1']['rise'],
                          'DT_minus_FT_RISE': row['DT']['rise']-row['FT_K1']['rise']})
        for task in ('niah_mq_q2', 'morehopqa'):
            selected = [r for r in rows if r['dataset'] == task]
            for metric in ('rise', 'mas', 'needle'):
                if metric == 'needle' and task == 'morehopqa':
                    continue
                left = [r['DT'][metric] for r in selected]
                right = [r['FT_K3_needle'] if metric == 'needle' else r['FT_K1'][metric] for r in selected]
                differences = [x-y for x, y in zip(left, right)]
                means.append({'family': family, 'dataset': task, 'metric': metric, 'n': 8,
                              'DT_mean': statistics.mean(left), 'FT_mean': statistics.mean(right),
                              'DT_minus_FT_mean': statistics.mean(differences),
                              'DT_better_cases': sum(d > 0 if metric == 'needle' else d < 0 for d in differences)})
    args.output.mkdir(parents=True, exist_ok=True)
    for name, rows in [('baseline_means.csv', means), ('baseline_cases.csv', cases)]:
        with (args.output / name).open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    receipt = {'status': 'verified_existing_records', 'new_model_calls': 0, 'new_metric_calls': 0,
               'script_sha256': sha(Path(__file__).read_bytes()), 'assembled_views_sha256': sha(data),
               'raw_references': references, 'cases': len(cases), 'means': means,
               'scope': 'Original clean development16 per model; FT K1 faithfulness and historical FT K3 needle. Not a new run, full paper table, or proof of architectural causation.',
               'signed_RISE_provenance': 'Frozen previously verified 32-case assembly; backing raw inputs and attribution vectors rechecked here.'}
    (args.output / 'initial_evidence.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'cases_verified': len(cases), 'means': means}))


if __name__ == '__main__':
    main()

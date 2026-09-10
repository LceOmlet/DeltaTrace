"""Retrospective NIAH source-only rank intervention at unchanged original k.

The parser uses the benchmark's fixed instruction/query scaffold, not needles,
answer keys, or scores. Its source mask is for evidence retrieval, not an
alteration of the stored attribution target or the original benchmark.
"""
import argparse
import gzip
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE / '.deps'))
sys.path.insert(0, str(ROOT / 'experiments/official'))
from tokenizers import Tokenizer
from retrieval_views import restrict_order_to_span
from recovery_diagnostics import recovery_diagnostics
from audit_saved import write_csv, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--publication', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--traces', type=Path, required=True)
    p.add_argument('--tokenizer', type=Path, required=True)
    a = p.parse_args()
    tok = Tokenizer.from_file(str(a.tokenizer))
    tasks = ['niah_' + suffix for suffix in ['mq_q2', 'mq_q4', 'mq_q8', 'mv_v2', 'mv_v4', 'mv_v8']]
    rows = []
    for task in tasks:
        cache = [json.loads(l) for l in (a.data / (task + '.jsonl')).read_text(encoding='utf-8').splitlines()]
        result = json.loads(gzip.decompress((a.publication / 'raw' / (task + '.results.json.gz')).read_bytes()))
        vectors = np.load(a.publication / 'raw' / (task + '.vectors.npz'))
        ft_dir, = (a.traces / (task + '.jsonl') / 'qwen-8B').glob('ifr_multi_hop_both_n1_*')
        for case in result['cases']:
            i = case['index']
            text = ' ' + cache[i]['prompt']
            assert text.startswith(' Some special magic numbers are hidden within the following text.')
            begin = text.index('\n') + 1
            end = text.rfind('\nWhat are all the special magic numbers for ')
            assert 0 < begin < end < len(text)
            enc = tok.encode(text, add_special_tokens=False)
            keep, gold = case['keep'], set(case['gold']) & set(case['keep'])
            signed = vectors[f'{task}_{i}_DT_signed_full'][case['user_positions']].astype(np.float32)
            with np.load(ft_dir / f'ex_{i:06d}.npz') as f:
                ft_scores = f['v_seq_prompt'].copy()
            r = {'dataset': task, 'index': i, 'source_start': begin, 'source_end': end}
            for label, scores in [('DT', np.maximum(signed, 0)), ('FT', np.maximum(ft_scores, 0))]:
                d = recovery_diagnostics(scores, keep, gold)
                full_order = sorted(keep, key=lambda j: (-float(scores[j]), j))
                scoped = restrict_order_to_span(full_order, enc.offsets, begin, end)
                # Gold is consulted only after the source parser and rank order,
                # to validate the intervention and score its output.
                assert gold <= set(scoped) and len(scoped) >= d['budget']
                k = d['budget']
                selected = set(full_order[:k])
                question_tokens = set(restrict_order_to_span(keep, enc.offsets, end, len(text)))
                instruction_tokens = set(restrict_order_to_span(keep, enc.offsets, 0, begin))
                r[label] = d['recall']
                r[label + '_source'] = len(set(scoped[:k]) & gold) / len(gold)
                r[label + '_query_budget_fraction'] = len(selected & question_tokens) / k
                r[label + '_instruction_budget_fraction'] = len(selected & instruction_tokens) / k
                r[label + '_source_gain'] = r[label + '_source'] - r[label]
                r['source_eligible'] = len(scoped)
                r['budget'] = k
                r['ceiling'] = d['ceiling']
            rows.append(r)
        vectors.close()
        print(task, 'done', flush=True)
    aggregates = []
    for task in tasks:
        for subset in ['all', 'even', 'odd']:
            group = [r for r in rows if r['dataset'] == task and (subset == 'all' or r['index'] % 2 == (subset == 'odd'))]
            r = {'dataset': task, 'subset': subset, 'count': len(group)}
            for field in group[0]:
                if field not in ['dataset', 'index']:
                    r[field] = float(np.mean([g[field] for g in group]))
            aggregates.append(r)
    write_csv(HERE / 'niah_source_cases.csv', rows)
    write_csv(HERE / 'niah_source_tasks.csv', aggregates)
    (HERE / 'niah_source_receipt.json').write_text(json.dumps({'cases': len(rows), 'model_calls': 0,
        'script_sha256': sha(Path(__file__)), 'tokenizer_sha256': sha(a.tokenizer),
        'source_mask_has_no_gold_or_target_input': True,
        'all_gold_verified_within_parsed_source_after_ranking': True,
        'original_token_budget_preserved': True}, indent=2) + '\n', encoding='utf-8')
    print(json.dumps([r for r in aggregates if r['subset'] == 'all'], indent=2))


if __name__ == '__main__':
    main()

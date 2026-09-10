"""Run the fixed, label-free retrieval-view probes and token-level diagnosis."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'experiments/official'))
sys.path.insert(0, str(HERE / '.deps'))
from tokenizers import Tokenizer
from retrieval_views import sentence_density_order, restrict_order_to_span
from recovery_diagnostics import recovery_diagnostics
from audit_saved import write_csv


def token_set(offsets, spans):
    return {i for i, (s, e) in enumerate(offsets) if e > s and any(s < b and e > a for a, b in spans)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--publication', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--traces', type=Path, required=True)
    p.add_argument('--tokenizer', type=Path, required=True)
    p.add_argument('--output', type=Path, default=HERE)
    a = p.parse_args()
    tokenizer = Tokenizer.from_file(str(a.tokenizer))
    frozen = json.loads((a.publication / 'summary.json').read_text(encoding='utf-8'))
    rows, spans_out, examples = [], [], []
    for task in frozen['tasks']:
        name = task['dataset']
        if task['DT']['needle'] is None:
            continue
        caches = [json.loads(s) for s in (a.data / (name + '.jsonl')).read_text(encoding='utf-8').splitlines()]
        report = json.loads(gzip.decompress((a.publication / 'raw' / (name + '.results.json.gz')).read_bytes()))
        vectors = np.load(a.publication / 'raw' / (name + '.vectors.npz'))
        ft_dir, = (a.traces / (name + '.jsonl') / 'qwen-8B').glob('ifr_multi_hop_both_n1_*')
        for case in report['cases']:
            i = case['index']
            cache = caches[i]
            text = ' ' + cache['prompt']
            enc = tokenizer.encode(text, add_special_tokens=False)
            assert len(enc.ids) == len(case['user_positions'])
            actual = np.asarray(case['input_ids'])[case['user_positions']]
            assert np.flatnonzero(np.asarray(enc.ids) != actual).tolist() == case['standalone_token_boundary_differences']
            offsets = enc.offsets
            needles = cache['metadata']['needle_spans']
            char_spans = [(s['span'][0] + 1, s['span'][1] + 1) for s in needles]
            assert token_set(offsets, char_spans) == set(case['gold']), (name, i)
            keep = set(case['keep'])
            gold = set(case['gold']) & keep
            s = vectors[f'{name}_{i}_DT_signed_full'][case['user_positions']].astype(np.float32)
            positive = np.maximum(s, 0)
            with np.load(ft_dir / f'ex_{i:06d}.npz') as ft:
                ft_s = ft['v_seq_prompt'].copy()
            d = recovery_diagnostics(positive, keep, gold)
            f = recovery_diagnostics(np.maximum(ft_s, 0), keep, gold)
            k = d['budget']
            selected = {'DT': set(d['selected']), 'FT': set(f['selected'])}
            r = {'dataset': name, 'index': i, 'subset': 'even' if i % 2 == 0 else 'odd',
                 'DT': case['metrics']['DT']['needle'], 'FT': f['recall'], 'ceiling': d['ceiling']}
            permutation = np.random.default_rng(20260910 + i).permutation(sorted(keep))
            shuffled = positive.copy()
            shuffled[sorted(keep)] = positive[permutation]
            for label, scores, absolute in [('DT_sentence_positive', s, False), ('DT_sentence_absolute', s, True),
                                            ('FT_sentence_positive', ft_s, False), ('shuffled_DT_sentence', shuffled, False)]:
                order = sentence_density_order(text, offsets, scores, keep, absolute=absolute)
                assert len(order) == len(keep) and set(order) == keep
                selected[label] = set(order[:k])
                r[label] = len(selected[label] & gold) / len(gold)
            # This text-only marker detects the repeated task instruction present
            # in VT examples. It is used solely to explain errors, not for ranking.
            marker = 'Memorize and track the chain(s) of variable assignment hidden in the following text.'
            markers = [m.start() for m in re.finditer(re.escape(marker), text)]
            demo = token_set(offsets, [(0, markers[-1])]) & keep if len(markers) > 1 else set()
            question = text.rfind('Question:')
            question_tokens = token_set(offsets, [(question, len(text))]) & keep if question >= 0 else set()
            r['demo_eligible'] = len(demo)
            # Controlled rank intervention: remove only demonstration candidates,
            # keep the ORIGINAL k so its contribution is not mixed with budget.
            # Also report the narrower source's recomputed 10% budget separately.
            scope_start = markers[-1] if len(markers) > 1 else 0
            r['scope_start_character'] = scope_start
            for label, scores in [('DT', positive), ('FT', np.maximum(ft_s, 0))]:
                full_order = sorted(keep, key=lambda j: (-float(scores[j]), j))
                source_order = restrict_order_to_span(full_order, offsets, scope_start, len(text))
                pooled = sentence_density_order(text, offsets, scores, keep)
                scoped_pooled = restrict_order_to_span(pooled, offsets, scope_start, len(text))
                assert len(source_order) >= k
                selected[label + '_scope'] = set(source_order[:k])
                selected[label + '_scope_sentence'] = set(scoped_pooled[:k])
                r[label + '_scope'] = len(selected[label + '_scope'] & gold) / len(gold)
                r[label + '_scope_sentence'] = len(selected[label + '_scope_sentence'] & gold) / len(gold)
                smaller_k = max(1, int(np.ceil(.1 * len(source_order))))
                r[label + '_scope_recomputed_budget'] = len(set(source_order[:smaller_k]) & gold) / len(gold)
                r['scope_eligible'] = len(source_order)
                r['scope_recomputed_budget'] = smaller_k
            for label in ['DT', 'FT', 'DT_sentence_positive', 'DT_sentence_absolute']:
                r[label + '_demo_budget_fraction'] = len(selected[label] & demo) / k
                r[label + '_question_budget_fraction'] = len(selected[label] & question_tokens) / k
            answer_tokens = set()
            for span_i, (needle, (start, end)) in enumerate(zip(needles, char_spans)):
                sp_gold = token_set(offsets, [(start, end)]) & keep
                answer = needle.get('answer')
                answer_chars = [(m.start() + start, m.end() + start) for m in re.finditer(re.escape(answer), text[start:end])] if answer else []
                sp_answer = token_set(offsets, answer_chars) & sp_gold
                answer_tokens |= sp_answer
                sr = {'dataset': name, 'index': i, 'span_index': span_i, 'gold_tokens': len(sp_gold),
                      'answer_tokens': len(sp_answer), 'answer': answer, 'snippet': needle.get('snippet'),
                      'positive_gold_mass': float(positive[sorted(sp_gold)].sum()),
                      'negative_gold_mass': float(np.maximum(-s[sorted(sp_gold)], 0).sum())}
                for label, top in selected.items():
                    sr[label + '_recall'] = len(top & sp_gold) / len(sp_gold) if sp_gold else None
                spans_out.append(sr)
            other_tokens = gold - answer_tokens
            r['answer_gold_tokens'] = len(answer_tokens)
            r['other_gold_tokens'] = len(other_tokens)
            for label in ['DT', 'FT', 'DT_sentence_positive', 'DT_sentence_absolute']:
                for subtype, tokens in [('answer', answer_tokens), ('other', other_tokens)]:
                    r[label + '_' + subtype + '_recall'] = len(selected[label] & tokens) / len(tokens) if tokens else None
                matching_spans = spans_out[-len(needles):]
                r[label + '_span_any'] = float(np.mean([sr[label + '_recall'] > 0 for sr in matching_spans]))
                r[label + '_span_half'] = float(np.mean([sr[label + '_recall'] >= .5 for sr in matching_spans]))
            rows.append(r)
            if i == 0:
                examples.append({'dataset': name, 'index': i, 'summary': r,
                    'top_DT_non_gold': [{'token': enc.tokens[j], 'text': text[offsets[j][0]:offsets[j][1]], 'index': j,
                                         'score': float(s[j]), 'demo': j in demo, 'question': j in question_tokens}
                                        for j in d['selected'] if j not in gold][:25],
                    'gold_tokens': [{'index': j, 'text': text[offsets[j][0]:offsets[j][1]], 'score': float(s[j]),
                                     'DT_selected': j in selected['DT'], 'FT_selected': j in selected['FT']}
                                    for j in sorted(gold)]})
        vectors.close()
        print(name, 'done', flush=True)
    aggregate = []
    numeric = [k for k in rows[0] if k not in ['dataset', 'index', 'subset']]
    for name in dict.fromkeys(r['dataset'] for r in rows):
        for subset in ['all', 'even', 'odd']:
            group = [r for r in rows if r['dataset'] == name and (subset == 'all' or r['subset'] == subset)]
            g = {'dataset': name, 'subset': subset, 'count': len(group)}
            for field in numeric:
                values = [r[field] for r in group if r[field] is not None]
                g[field] = float(np.mean(values)) if values else None
            for label in ['DT_sentence_positive', 'DT_sentence_absolute', 'DT_scope', 'DT_scope_sentence']:
                delta = np.array([r[label] - r['DT'] for r in group])
                rng = np.random.default_rng(20260910)
                means = delta[rng.integers(0, len(delta), size=(10000, len(delta)))].mean(axis=1)
                g[label + '_gain'] = float(delta.mean())
                g[label + '_ci_low'], g[label + '_ci_high'] = map(float, np.quantile(means, [.025, .975]))
                g[label + '_wins'] = int((delta > 0).sum())
                g[label + '_losses'] = int((delta < 0).sum())
            aggregate.append(g)
    a.output.mkdir(parents=True, exist_ok=True)
    write_csv(a.output / 'structure_cases.csv', rows)
    write_csv(a.output / 'structure_tasks.csv', aggregate)
    write_csv(a.output / 'span_diagnostics.csv', spans_out)
    (a.output / 'token_examples.json').write_text(json.dumps(examples, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__), ROOT / 'experiments/official/retrieval_views.py', HERE / 'protocol.md']}
    (a.output / 'structure_receipt.json').write_text(json.dumps({'cases': len(rows), 'gold_offset_checks': len(rows),
        'tokenizer_sha256': hashlib.sha256(a.tokenizer.read_bytes()).hexdigest(), 'tokenizers_version': '0.22.2',
        'model_calls': 0, 'sha256': hashes}, indent=2) + '\n', encoding='utf-8')
    for r in aggregate:
        if r['subset'] == 'all':
            print(json.dumps({k: r[k] for k in ['dataset', 'DT', 'FT', 'DT_sentence_positive', 'DT_sentence_absolute', 'FT_sentence_positive', 'shuffled_DT_sentence', 'DT_demo_budget_fraction', 'FT_demo_budget_fraction']}))


if __name__ == '__main__':
    main()

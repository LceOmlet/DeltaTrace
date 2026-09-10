"""Evaluate the preregistered pooling probe using development vectors only."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'experiments/official'))
from evidence_protocol import recovery_curve
from retrieval_views import sentence_density_order
from sentence_pooling import RULES, sentence_pool_order


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('run', 'verified', 'data', 'tokenizer', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    a = p.parse_args()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(a.tokenizer))
    r = json.loads((a.run / 'results.json').read_bytes())
    checked = json.loads(a.verified.read_bytes())
    assert checked['status'] == 'verified_development'
    assert checked['run_results_sha256'] == digest(a.run / 'results.json')
    assert checked['run_vectors_sha256'] == r['vectors_sha256'] == digest(a.run / 'vectors.npz')
    split = json.loads((HERE / 'answer_split.json').read_bytes())
    tasks = ['vt_h2_c3', 'hotpotqa_long']
    expected = {(t, i) for t in tasks for i in split['tasks'][t]['development']}
    cases = [x for x in r['cases'] if x['target_mode'] == 'answer_only']
    assert len(cases) == len(expected) == 16 and {(x['dataset'], x['index']) for x in cases} == expected
    vectors = np.load(a.run / 'vectors.npz')
    caches = {}
    for t in tasks:
        path = a.data / (t + '.jsonl')
        assert digest(path) == split['tasks'][t]['cache_sha256']
        caches[t] = [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines()]
    rows = []
    identity_count = 0
    for row in cases:
        t, i = row['dataset'], row['index']
        text = ' ' + caches[t][i]['prompt']
        offsets = tok.encode(text, add_special_tokens=False).offsets
        prefix = f'{t}_{i}_answer_only_'
        for method, key in [('DT', 'DT_target_signed_full'), ('FT_K3', 'FT_K3_prompt')]:
            value = vectors[prefix + key]
            if method == 'DT':
                value = value[row['user_positions']]
            for rule in RULES:
                order = sentence_pool_order(text, offsets, value, row['keep'], rule)
                assert len(order) == len(row['keep']) and set(order) == set(row['keep'])
                if rule == 'mean':
                    assert order == sentence_density_order(text, offsets, value, row['keep'])
                    identity_count += 1
                scores = np.zeros(len(value), dtype=np.float32)
                scores[order] = np.arange(len(order), 0, -1)
                curve = recovery_curve(scores, row['keep'], row['gold'], [.05, .1, .2])
                for q in curve['points']:
                    rows.append(dict(dataset=t, index=i, method=method, rule=rule,
                                     fraction=q['fraction'], recall=q['recall'],
                                     budget=q['budget'], gold=q['gold'], ceiling=q['ceiling']))
    means = {m: {rule: {t: float(np.mean([x['recall'] for x in rows if x['method'] == m and
        x['rule'] == rule and x['dataset'] == t and x['fraction'] == .1])) for t in tasks}
        for rule in RULES} for m in ('DT', 'FT_K3')}
    ranked = {m: sorted(RULES, key=lambda rule: (-np.mean(list(means[m][rule].values())),
        -min(means[m][rule].values()), rule)) for m in means}
    selected = {m: ranked[m][0] for m in means}
    diffs = {t: means['DT'][selected['DT']][t] - means['FT_K3'][selected['FT_K3']][t] for t in tasks}
    ceiling = {t: all(x['recall'] == x['ceiling'] for x in rows if x['dataset'] == t and
        x['fraction'] == .1 and x['rule'] == selected[x['method']]) for t in tasks}
    eligible = max(diffs.values()) > 0 and all(diffs[t] > 0 or (diffs[t] == 0 and ceiling[t]) for t in tasks)
    out = dict(status='verified_development', case_count=len(cases), plan_sha256=digest(HERE / 'POOLING_PILOT.md'),
        analyzer_sha256=digest(Path(__file__)), pooling_source_sha256=digest(HERE / 'sentence_pooling.py'),
        results_sha256=digest(a.run / 'results.json'), vectors_sha256=r['vectors_sha256'],
        split_sha256=digest(HERE / 'answer_split.json'), mean_order_identity_checks=identity_count,
        means_at10=means, ranked_rules=ranked, selected_rules=selected, selected_differences=diffs,
        joint_exact_ceiling=ceiling, eligible_for_validation=eligible,
        selection_data='reused development only; no candidate validation outcomes read')
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / 'analysis.json').write_text(json.dumps(out, indent=2) + '\n', encoding='utf-8')
    with (a.output / 'cases.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    if eligible:
        choice = dict(status='frozen_for_validation', selected_rules=selected, plan_sha256=out['plan_sha256'],
            pooling_source_sha256=out['pooling_source_sha256'], development_results_sha256=out['results_sha256'],
            development_vectors_sha256=out['vectors_sha256'], split_sha256=out['split_sha256'])
        path = a.output / 'choice.json'
        if path.exists():
            assert json.loads(path.read_bytes()) == choice
        else:
            path.write_text(json.dumps(choice, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()

"""Verify matched answer-target controls and select using development cases only."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'experiments/official'))
from evidence_protocol import source_span, select_source_tokens, reference_token_ids, recovery_curve
from retrieval_views import sentence_density_order
from analyze import paired_bootstrap, interval
from analyze_recall_pilot import same_tree

TASKS = ['vt_h2_c3', 'vt_h4_c1', 'vt_h6_c1', 'vt_h10_c1', 'hotpotqa_long']
FRACTIONS = [.05, .1, .2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'publication', 'data', 'tokenizer', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--freeze-choice', action='store_true')
    a = p.parse_args()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(a.tokenizer))
    r = json.loads((a.run / 'results.json').read_bytes())
    split = json.loads((HERE / 'answer_split.json').read_bytes())
    assert r['status'] == 'complete' and r['experiment'] == 'answer-pilot-v1'
    assert r['driver_sha256'] == digest(HERE / ('evaluate_answer.py' if r['stage'] == 'development' else 'evaluate_answer_matched.py'))
    assert r['split_sha256'] == digest(HERE / 'answer_split.json')
    assert r['plan_sha256'] == split['plan_sha256'] == digest(HERE / 'ANSWER_PILOT.md')
    assert r['vectors_sha256'] == digest(a.run / 'vectors.npz')
    for name, value in r['weighted_sources'].items():
        assert value == digest(HERE / name)
    correction = r.get('FT_target_correction')
    if r['stage'] == 'development':
        assert correction and correction['version'] == 'initial_seed_full_hops_v1', 'Use the FT control that retains full reasoning hops'
        assert correction['adapter_sha256'] == digest(HERE / 'ft_target_control.py')
        assert correction['script_sha256'] == digest(HERE / 'correct_answer_ft.py')
        assert correction['addendum_sha256'] == digest(HERE / 'ANSWER_HOPS_ADDENDUM.md')
        assert len(correction['identity_checks']) == 4 and all(x['bitwise_equal'] for x in correction['identity_checks'])
    else:
        assert r['FT_initial_target_adapter']['sha256'] == digest(HERE / 'ft_target_control.py')
        assert r['FT_initial_target_adapter']['addendum_sha256'] == digest(HERE / 'ANSWER_HOPS_ADDENDUM.md')
        assert r['choice']['FT_initial_target_adapter_sha256'] == r['FT_initial_target_adapter']['sha256']
    stage = r['stage']
    assert not a.freeze_choice or stage == 'development'
    tasks = ['vt_h2_c3', 'hotpotqa_long'] if stage == 'development' else TASKS
    modes = ['full', 'answer_conditioned', 'answer_only'] if stage == 'development' else list(dict.fromkeys(['full', r['choice']['target_mode']]))
    expected = {(t, i, mode) for t in tasks for i in split['tasks'][t][stage] for mode in modes}
    assert len(r['cases']) == len(expected)
    assert {(x['dataset'], x['index'], x['target_mode']) for x in r['cases']} == expected
    assert r['selected_counts'] == {t: len(split['tasks'][t][stage]) for t in tasks}
    caches, originals = {}, {}
    for t in tasks:
        path = a.data / (t + '.jsonl')
        assert digest(path) == split['tasks'][t]['cache_sha256']
        caches[t] = [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]
        old = json.loads(gzip.decompress((a.publication / 'raw' / (t + '.results.json.gz')).read_bytes()))
        originals[t] = {x['index']: x for x in old['cases']}
    vectors = np.load(a.run / 'vectors.npz')
    rows, diagnostics, residuals = [], [], []
    identity_checks = 0
    for row in r['cases']:
        t, i, mode = row['dataset'], row['index'], row['target_mode']
        cache, old = caches[t][i], originals[t][i]
        assert row['status'] == 'complete'
        assert row['original_target_sha256'] == hashlib.sha256(cache['target'].encode()).hexdigest()
        original_target = tok.encode(cache['target'], add_special_tokens=False)
        start, end = cache['indices_to_explain']
        assert row['original_answer_token_span'] == [start, end]
        cs, ce = original_target.offsets[start][0], original_target.offsets[end][1]
        assert row['original_answer_char_span'] == [cs, ce]
        expected_target = cache['target'][cs:ce] if mode == 'answer_only' else cache['target']
        assert row['target'] == expected_target
        ids = np.asarray(row['input_ids'], dtype=np.int64)
        assert hashlib.sha256(ids.tobytes()).hexdigest() == row['input_sha256']
        positions = row['user_positions']
        for name in ('user_positions', 'gold', 'prompt_length'):
            assert row[name] == old[name]
        assert row['author_keep'] == old['keep']
        assert row['input_ids'][:row['prompt_length']] == old['input_ids'][:old['prompt_length']]
        eos = int(old['input_ids'][-1])
        target_ids = tok.encode(expected_target, add_special_tokens=False).ids + [eos]
        assert row['input_ids'][row['prompt_length']:] == target_ids
        assert row['target_length'] == len(target_ids)
        text = ' ' + cache['prompt']
        enc = tok.encode(text, add_special_tokens=False)
        assert len(enc.ids) == len(positions)
        assert np.flatnonzero(np.asarray(enc.ids) != ids[positions]).tolist() == row['standalone_token_boundary_differences']
        span = source_span(t, cache['prompt'])
        assert span == row['source_span']
        keep = select_source_tokens(span, enc.offsets, row['author_keep'], row['gold'])
        assert keep == row['keep']
        full_ref = np.asarray(reference_token_ids(ids, [positions[j] for j in row['author_keep']], eos), dtype=np.int64)
        assert hashlib.sha256(full_ref.tobytes()).hexdigest() == row['references']['full']
        weights = np.asarray(row['target_weights'], dtype=np.float64)
        assert weights.shape == (len(target_ids),) and set(weights) <= {0., 1.} and weights.sum() > 0
        for hops in (1, 3):
            first = row[f'FT_K{hops}_actual_target_aggregation'][0]
            actual = np.zeros_like(weights)
            lo, hi = first['start'], first['end']
            assert 0 <= lo <= hi < len(weights) - 1
            actual[lo:hi+1] = first['weights'] if first['weights'] is not None else 1.
            assert np.array_equal(actual, weights)
            assert (lo, hi) == (0, len(target_ids)-2)
            if mode == 'answer_conditioned':
                assert all((item['start'], item['end']) == (0, len(target_ids)-2)
                    for item in row[f'FT_K{hops}_actual_target_aggregation']), 'FT must retain all reasoning-hop support'
        prefix = f'{t}_{i}_{mode}_'
        if row.get('weighted_identity_bitwise_equal'):
            assert mode == 'full'
            assert np.array_equal(vectors[prefix+'DT_original_signed_full'], vectors[prefix+'DT_weighted_identity_signed_full'])
            identity_checks += 1
        for method, views in row['metrics'].items():
            if method.startswith('DT_'):
                signed = vectors[prefix + method + '_signed_full']
                assert signed.shape == ids.shape and np.isfinite(signed).all()
                values = signed[positions].astype(np.float32)
                detail = row[method + '_details']
                l0, l1 = np.asarray(detail['endpoint_target_logprobs32'], dtype=np.float64)
                w = weights if method == 'DT_target' else np.ones_like(weights)
                delta = float(((l1 - l0) * w).sum())
                assert np.isclose(delta, detail['target_delta_score32_sum64'], atol=1e-8)
                assert np.isclose(signed.sum(), detail['signed_sum'], atol=1e-10)
                residuals.append({'dataset': t, 'index': i, 'target_mode': mode, 'method': method,
                    'target_delta': delta, 'signed_sum': float(signed.sum()),
                    'residual': detail['unassigned_total']})
            else:
                assert method in ('FT_K1', 'FT_K3')
                values = vectors[prefix + method + '_prompt'].astype(np.float32)
            positive = np.maximum(values, 0)
            order = sentence_density_order(text, enc.offsets, positive, keep)
            assert len(order) == len(keep) and set(order) == set(keep)
            rank = np.zeros_like(positive)
            rank[order] = np.arange(len(keep), 0, -1)
            for view, scores in [('raw', positive), ('density', rank)]:
                curve = recovery_curve(scores, keep, row['gold'], FRACTIONS)
                same_tree(views[view], curve)
                for point in curve['points']:
                    rows.append({'dataset': t, 'index': i, 'target_mode': mode, 'method': method, 'view': view,
                        'fraction': point['fraction'], 'recall': point['recall'], 'budget': point['budget'],
                        'gold': point['gold'], 'ceiling': point['ceiling']})
            if t.startswith('vt_') and method in ('DT_original', 'DT_target', 'FT_K3'):
                assignments = [(m.start(), m.end()) for m in re.finditer(r'\bVAR [A-Z]+\s*=\s*(?:VAR [A-Z]+|\d+)', text)
                    if span['start']+1 <= m.start() < m.end() <= span['end']+1]
                assignment_tokens = {j for j in keep if any(enc.offsets[j][0] < b and enc.offsets[j][1] > a for a, b in assignments)}
                gold = set(row['gold']) & set(keep)
                mass = positive[list(assignment_tokens)].sum()
                k = int(np.ceil(.1 * len(keep)))
                selected = set(order[:k])
                diagnostics.append({'dataset': t, 'index': i, 'target_mode': mode, 'method': method,
                    'gold_share_of_assignment_positive_mass': float(positive[list(gold & assignment_tokens)].sum()/mass) if mass > 0 else None,
                    'distractor_assignment_tokens_in_budget': len(selected & (assignment_tokens - gold)),
                    'assignment_count': len(assignments), 'budget': k})
    assert stage != 'development' or identity_checks == 2
    table = {(x['dataset'],x['index'],x['target_mode'],x['method'],x['view'],x['fraction']):x['recall'] for x in rows}
    means, comparisons, boot_by_mode = {}, {}, {}
    for mode in modes:
        means[mode], comparisons[mode], boot_by_mode[mode] = {}, {}, {}
        for t in tasks:
            indices = split['tasks'][t][stage]
            dt = np.array([table[t,i,mode,'DT_target','density',.1] for i in indices])
            ft = np.array([table[t,i,mode,'FT_K3','density',.1] for i in indices])
            means[mode][t] = {'DT': float(dt.mean()), 'FT_K3': float(ft.mean()), 'difference':float((dt-ft).mean())}
            differences = np.array([[table[t,i,mode,'DT_target','density',f]-table[t,i,mode,'FT_K3','density',f]
                for f in FRACTIONS] for i in indices])
            rng = np.random.default_rng(np.random.SeedSequence(73,spawn_key=(TASKS.index(t),)))
            boot = paired_bootstrap(differences,rng)
            boot_by_mode[mode][t] = boot
            comparisons[mode][t] = {str(f):interval(differences[:,j],boot[:,j]) for j,f in enumerate(FRACTIONS)}
    out = {'status':'verified_'+stage, 'case_count':sum(r['selected_counts'].values()),
        'mode_count':len(r['cases']), 'driver_sha256':r['driver_sha256'], 'analyzer_sha256':digest(Path(__file__)),
        'run_results_sha256':digest(a.run/'results.json'), 'run_vectors_sha256':r['vectors_sha256'],
        'split_sha256':r['split_sha256'], 'weighted_identity_checks':identity_checks,
        'FT_target_correction':correction,
        'actual_DT_FT_target_weights_verified':True, 'original_prompt_gold_and_answer_extraction_verified':True,
        'means_at10':means, 'paired_contrasts':comparisons, 'choice':r['choice'],
        'completed_gpu_operation_seconds':sum(x['seconds'] for x in r['costs'] if x['status']=='returned'),
        'residuals':residuals, 'assignment_diagnostics':diagnostics}
    if stage == 'development':
        ranked = sorted(modes,key=lambda mode:(-min(v['difference'] for v in means[mode].values()),
            -np.mean([v['difference'] for v in means[mode].values()]),mode))
        choice = ranked[0]
        exact = {t:all(x['recall']==x['ceiling'] for x in rows if x['dataset']==t and x['target_mode']==choice
            and x['method'] in ('DT_target','FT_K3') and x['view']=='density' and x['fraction']==.1) for t in tasks}
        eligible = any(v['difference']>0 for v in means[choice].values()) and all(
            v['difference']>0 or (v['difference']==0 and exact[t]) for t,v in means[choice].items())
        out.update(ranked_candidates=ranked,best_candidate=choice,eligible_for_validation=eligible,joint_exact_ceiling=exact)
        if a.freeze_choice:
            assert eligible, 'No candidate qualifies; preserve failed development results'
            frozen = {'status':'frozen_for_validation','target_mode':choice,'split_sha256':r['split_sha256'],
                'development_results_sha256':out['run_results_sha256'],'development_vectors_sha256':r['vectors_sha256'],
                'FT_initial_target_adapter_sha256':digest(HERE/'ft_target_control.py'),
                'selection_rule':'max_min_then_macro_then_name','development_means':means[choice]}
            a.output.mkdir(parents=True,exist_ok=True)
            path=a.output/'choice.json'
            if path.exists():assert json.loads(path.read_bytes())==frozen
            else:path.write_text(json.dumps(frozen,indent=2)+'\n',encoding='utf-8')
    else:
        choice=r['choice']['target_mode']
        groups={}
        for name,members in [('VT',TASKS[:4]),('HotpotQA',TASKS[4:]),('all_five_tasks',TASKS)]:
            boot=np.mean([boot_by_mode[choice][t] for t in members],axis=0)
            groups[name]={'DT':float(np.mean([means[choice][t]['DT'] for t in members])),
                'FT_K3':float(np.mean([means[choice][t]['FT_K3'] for t in members])),
                'differences':{str(f):{'mean_difference':float(np.mean([comparisons[choice][t][str(f)]['mean_difference'] for t in members])),
                    'ci95':np.quantile(boot[:,j],[.025,.975]).tolist()} for j,f in enumerate(FRACTIONS)}}
        out['selected_groups']=groups
        out['predeclared_shared_advantage_criterion_met']=(groups['VT']['differences']['0.1']['mean_difference']>0 and
            groups['HotpotQA']['differences']['0.1']['mean_difference']>0 and groups['all_five_tasks']['differences']['0.1']['ci95'][0]>0)
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'analysis.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    with (a.output/'cases.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    print(json.dumps({k:v for k,v in out.items() if k in ('status','case_count','means_at10','best_candidate',
        'eligible_for_validation','selected_groups','predeclared_shared_advantage_criterion_met')},indent=2))


if __name__=='__main__':
    main()

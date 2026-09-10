"""Produce compact findings and the static comparison plot from saved audits."""
import csv
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def records(name):
    with (HERE / name).open(encoding='utf-8') as f:
        return [{k: (v if k in ['dataset', 'subset', 'input_sha256'] else (v == 'True') if v in ['True', 'False'] else float(v) if v else None)
                 for k, v in r.items()} for r in csv.DictReader(f)]


def main():
    task_rows = records('task_diagnostics.csv')
    tasks = {r['dataset']: r for r in task_rows}
    structure = {r['dataset']: r for r in records('structure_tasks.csv') if r['subset'] == 'all'}
    niah = [r for r in records('niah_source_tasks.csv') if r['subset'] == 'all']
    cases = records('case_diagnostics.csv')
    data = {'scope': '1048 original Qwen3-8B cases, 11 recovery tasks; retrospective CPU reranking only',
            'DT_task_wins_against_released_FT': sum(r['DT'] > r['FT'] for r in task_rows),
            'DT_task_losses_against_released_FT': sum(r['DT'] < r['FT'] for r in task_rows),
            'capacity_decomposition': [], 'VT_gap_decomposition': [], 'chance_adjusted_diagnostics': [],
            'eliminated_explanations': {
                'gold_tokens_at_standalone_boundary_mismatch': sum(r['gold_boundary_diff_count'] for r in cases),
                'cases_with_DT_or_FT_cutoff_ambiguity': sum(r['DT_tie_width'] > 0 or r['FT_tie_width'] > 0 for r in cases),
                'cached_judge_true_cases': sum(r['judge_true'] for r in cases),
                'signed_equals_positive_top10_cases': sum(r['signed_recall'] == r['DT'] for r in cases),
                'absolute_view_task_gains': sum(r['absolute_recall'] > r['DT'] for r in task_rows),
                'relative_conservation_residual_median': float(np.median([r['relative_conservation_residual'] for r in cases])),
                'relative_conservation_residual_max': max(r['relative_conservation_residual'] for r in cases),
            }}
    for name in tasks:
        group = [r for r in cases if r['dataset'] == name]
        row = {'dataset': name, 'random_expected_recall': float(np.mean([r['random_expected_recall'] for r in group]))}
        for method in ['DT', 'FT']:
            row[method + '_chance_adjusted'] = float(np.mean([
                (r[method] - r['random_expected_recall']) / (r['ceiling'] - r['random_expected_recall']) for r in group]))
        data['chance_adjusted_diagnostics'].append(row)
    for low, high in [('niah_mq_q2', 'niah_mq_q8'), ('niah_mv_v2', 'niah_mv_v8')]:
        left, right = tasks[low], tasks[high]
        # Define E = mean(R)/mean(C) so Rbar=Cbar*E is exact. This is not
        # the mean per-case R/C column. Symmetric two-factor decomposition.
        cl, cr = left['ceiling'], right['ceiling']
        el, er = left['DT'] / cl, right['DT'] / cr
        capacity = (cl - cr) * (el + er) / 2
        ranking = (el - er) * (cl + cr) / 2
        gap = left['DT'] - right['DT']
        assert abs(capacity + ranking - gap) < 1e-12
        data['capacity_decomposition'].append({'from': low, 'to': high, 'recall_gap': gap,
            'capacity_component': capacity, 'efficiency_component': ranking,
            'capacity_fraction_of_gap': capacity / gap,
            'interpretation': 'Exact descriptive factorization, not a causal estimate of model behavior.'})
    for name, r in structure.items():
        if not name.startswith('vt_'):
            continue
        gap = r['FT'] - r['DT']
        after = r['FT_scope'] - r['DT_scope']
        data['VT_gap_decomposition'].append({'dataset': name, 'original_FT_minus_DT': gap,
            'scope_FT_minus_DT': after, 'gap_reduction_fraction': (gap - after) / gap,
            'DT_demo_budget_fraction': r['DT_demo_budget_fraction'],
            'FT_demo_budget_fraction': r['FT_demo_budget_fraction'],
            'DT_original': r['DT'], 'FT_original': r['FT'], 'DT_scoped': r['DT_scope'],
            'FT_scoped': r['FT_scope'], 'DT_scoped_sentence': r['DT_scope_sentence'],
            'FT_scoped_sentence': r['FT_scope_sentence'], 'DT_gold_span_any_hit': r['DT_span_any'],
            'DT_gold_answer_token_recall': r['DT_answer_recall']})
    data['VT_mean_gap_before'] = float(np.mean([r['original_FT_minus_DT'] for r in data['VT_gap_decomposition']]))
    data['VT_mean_gap_after_scope'] = float(np.mean([r['scope_FT_minus_DT'] for r in data['VT_gap_decomposition']]))
    data['VT_macro_gap_reduction_fraction'] = 1 - data['VT_mean_gap_after_scope'] / data['VT_mean_gap_before']
    data['NIAH_source_intervention'] = niah
    data['NIAH_mean_DT_advantage_before'] = float(np.mean([r['DT'] - r['FT'] for r in niah]))
    data['NIAH_mean_DT_advantage_after_source'] = float(np.mean([r['DT_source'] - r['FT_source'] for r in niah]))
    data['NIAH_advantage_reduction_fraction'] = 1 - data['NIAH_mean_DT_advantage_after_source'] / data['NIAH_mean_DT_advantage_before']
    (HERE / 'findings.json').write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(data, indent=2))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none', 'svg.hashsalt': 'needle-gap-audit-20260910',
                         'savefig.facecolor': 'white'})
    names = list(tasks)
    labels = ['MQ-Q2', 'MQ-Q4', 'MQ-Q8', 'MV-V2', 'MV-V4', 'MV-V8', 'VT-H2-C3', 'VT-H4', 'VT-H6', 'VT-H10', 'HotpotQA']
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.5, 7.4), gridspec_kw={'width_ratios': [1, 1]})
    fig.subplots_adjust(left=.10, right=.97, bottom=.26, top=.84, wspace=.37)
    y = np.arange(len(names))
    ax.barh(y, [tasks[n]['ceiling'] for n in names], color='#e7ebef', height=.64, label='Budget ceiling')
    ax.barh(y, [tasks[n]['DT'] for n in names], color='#187e94', height=.36, label='DT original')
    ax.plot([tasks[n]['FT'] for n in names], y, 'o', color='#e29945', markersize=5, label='FT original')
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.02)
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xlabel('Original token Recall@10%')
    ax.set_title('A. Low recall can have a low ceiling', loc='left', weight='bold', pad=16)
    ax.legend(loc='upper left', bbox_to_anchor=(0, -.12), frameon=False, fontsize=9)
    vn = [n for n in names if n.startswith('vt_')]
    vy = np.arange(len(vn))
    for j, name in enumerate(vn):
        r = structure[name]
        bx.plot([r['DT'], r['DT_scope'], r['DT_scope_sentence']], [j, j, j], color='#a1cbd1', lw=3, zorder=1)
    for key, marker, color, label in [('DT', 'o', '#7b8994', 'DT original'),
                                       ('DT_scope', 's', '#187e94', 'DT: current problem only'),
                                       ('DT_scope_sentence', 'D', '#133e4a', 'DT: scope + sentence density')]:
        bx.scatter([structure[n][key] for n in vn], vy, s=52, marker=marker, color=color, label=label, zorder=3)
    bx.scatter([structure[n]['FT_scope_sentence'] for n in vn], vy+.14, s=46, marker='D',
               facecolors='none', edgecolors='#d78a36', label='FT: same scope + density', zorder=3)
    bx.set_yticks(vy, ['VT-H2-C3', 'VT-H4', 'VT-H6', 'VT-H10'])
    bx.set_ylim(3.6, -.6)
    bx.set_xlim(.35, 1.02)
    bx.xaxis.set_major_formatter(PercentFormatter(1))
    bx.set_xlabel('Recall at unchanged original token budget')
    bx.set_title('B. Demonstrations crowd out current evidence', loc='left', weight='bold', pad=16)
    bx.legend(loc='upper left', bbox_to_anchor=(0, -.12), frameon=False, fontsize=9)
    for axis in (ax, bx):
        axis.grid(axis='x', color='#edf0f3', lw=.7)
        axis.set_axisbelow(True)
    fig.suptitle('DeltaTrace needle gap: budget, source scope, and evidence granularity',
                 fontsize=17, weight='bold', x=.10, ha='left', y=.96)
    fig.text(.10, .90, '1,048 frozen Qwen3-8B examples across all 11 recovery tasks', color='#52616b', fontsize=11)
    fig.text(.10, .025, 'Panel B is retrospective reranking, not a new attribution run. Original benchmark scores remain unchanged.', fontsize=9, color='#52616b')
    fig.savefig(HERE / 'needle_gap.png', dpi=180)
    svg = HERE / 'needle_gap.svg'
    fig.savefig(svg, metadata={'Date': None})
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines()) + '\n', encoding='utf-8')
    plt.close(fig)


if __name__ == '__main__':
    main()

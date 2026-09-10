"""Build the fixed Recall validation report, including failures and baselines."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
TASKS = ['vt_h2_c3', 'vt_h4_c1', 'vt_h6_c1', 'vt_h10_c1', 'hotpotqa_long']
LABELS = ['VT H2-C3', 'VT H4-C1', 'VT H6-C1', 'VT H10-C1', 'HotpotQA']


def pct(x):
    return f'{100*x:.2f}%'


def pp(x):
    return f'{100*x:+.2f}'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--analysis', type=Path, required=True)
    p.add_argument('--plot', action='store_true')
    a = p.parse_args()
    report = json.loads((a.analysis / 'analysis.json').read_bytes())
    if report['stage'] == 'development':
        assert report['status'] == 'verified_development' and report['case_count'] == 16
        lines = ['# Recall 开发试测：全部候选', '',
            '16 例（VT H4-C1、HotpotQA 各 8 例），用于选择方案。下表均为相对 live FT K3、相同排序方式下的 Recall@10% 差值，单位为百分点。', '',
            '| Reference / 方向 / 排序 | VT H4-C1 差值 | HotpotQA 差值 |', '| --- | ---: | ---: |']
        for candidate in report['ranked_candidates']:
            means = report['means_at10'][candidate]
            lines.append(f'| {candidate} | {pp(means["vt_h4_c1"]["difference"])} | {pp(means["hotpotqa_long"]["difference"])} |')
        lines += ['', '选择 `full/forward/density`：原 full-prompt reference、原方向、句内正分数均值排序。', '',
            'VT 两者 8/8 例达到同一 token 预算上限，均为 70.8357%；HotpotQA DT 86.1688%，FT 80.1550%。VT 上限使严格正差值不可能，因此仅允许每例均已到上限的精确平局通过开发准入；没有改动样本、候选排序规则或验证集。见 [上限修正说明](../RECALL_CEILING_ADDENDUM.md)。', '',
            '这是开发结果，不能作为验证优势。80 例验证前已固定方案；完整负向结果、原始向量和计算成本均保留。', '',
            '原方向的 DT/FT 对照与前轮 GPU 运行逐位一致。交换端点的实验在部分原生批次上有数值不对称，最大目标差偏差 0.480 nats；不能把它的变化全部归因于单一有限交互规则。所选方案不使用端点交换。', '',
            '[全部统计](analysis.json) · [逐例表](cases.csv) · [固定方案](choice.json) · [对照复现](control_verification.json)', '']
        (a.analysis / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
        print(str(a.analysis / 'RESULTS.md'))
        return
    assert report['status'] == 'verified_validation' and report['case_count'] == 80
    choice = report['choice']['candidate']
    assert choice == 'full/forward/density', 'Rewrite interpretation if a different candidate is frozen'
    rows = list(csv.DictReader((a.analysis / 'cases.csv').open(encoding='utf-8')))
    means = report['means_at10'][choice]
    def mean(task, method, view):
        selected = [r for r in rows if r['dataset'] == task and r['method'] == method
            and r['view'] == view and float(r['fraction']) == .1]
        assert len(selected) == 16
        return np.mean([float(r['recall']) for r in selected])
    lines = ['# VT 与 HotpotQA：Recall 小样本修复验证', '',
        '固定方案：**原 full-prompt reference + 正分数句内均值排序**。句子排序后仍只取正文候选 token 的 10%，DT 与 live FT K3 使用完全相同的来源、聚合规则和 token 预算。没有扩大 k，也没有用 gold 定义候选句。', '',
        '16 例开发选择方案后固定；80 例留出验证，每个任务 16 例。验证集来自此前审计过的发布缓存，不是外部新数据集。', '',
        '| 验证任务 | DT 原 token 排序 | DT 句聚合 | FT K3 同样句聚合 | DT−FT，百分点 [95% CI] |',
        '| --- | ---: | ---: | ---: | ---: |']
    for t, label in zip(TASKS, LABELS):
        c = report['paired_contrasts'][choice][t]['0.1']
        lo, hi = c['ci95']
        lines.append(f'| {label} | {pct(mean(t, "DT_full_forward", "raw"))} | {pct(means[t]["DT"])} | {pct(means[t]["FT_K3"])} | {pp(c["mean_difference"])} [{pp(lo)}, {pp(hi)}] |')
    lines += ['', '| 汇总（任务等权） | DT | FT K3 | 差值，百分点 [95% CI] |', '| --- | ---: | ---: | ---: |']
    for name, group in report['selected_groups'].items():
        c = group['differences']['0.1']
        lines.append(f'| {name} | {pct(group["DT_recall_at10"])} | {pct(group["FT_K3_recall_at10"])} | {pp(c["mean_difference"])} [{pp(c["ci95"][0])}, {pp(c["ci95"][1])}] |')
    lines += ['', '**预定的共同优势判据：' + ('达到。**' if report['predeclared_shared_advantage_criterion_met'] else '未达到。**') +
        '要求 VT 与 HotpotQA 的均值均优于 FT，并且五任务等权差值的 95% 区间下界大于 0。单任务或分组区间跨 0 时不宣称该任务已建立可靠优势。', '',
        '## 原因与被排除的修复', '',
        '- 原始检索范围会让示例、题目和格式占用名额。显式限定正文能消除这项竞争；固定 token 预算与可达上限必须同时记录。',
        '- 保留正文外内容的 EOS reference 并未修好 Recall：完整 VT H4-C1 100 例下降 4.55 点，HotpotQA 48 例下降 8.69 点。因此本方案保留原 reference，仅修复检索排序。',
        '- DT 常给支持句中的少数词高分，而 gold 覆盖整句话。句内均值把这些信号用于支持句排序；FT 同样受益，表中比较已经给 FT 相同处理。',
        '- 开发集的端点反向和两方向平均均未胜过原方向。原生半精度模型在部分交换批次后还有数值差异（最大目标差不对称 0.480 nats），因此该诊断也不能被当成仅改变交互公式的无混杂因果实验。最终方案不使用这些变体。', '',
        '## 预算敏感性', '', '| 等权汇总 | Recall@5% 差值 | Recall@10% 差值 | Recall@20% 差值 |', '| --- | ---: | ---: | ---: |']
    for name, group in report['selected_groups'].items():
        parts = []
        for f in (.05, .1, .2):
            c = group['differences'][str(f)]
            parts.append(f'{pp(c["mean_difference"])} [{pp(c["ci95"][0])}, {pp(c["ci95"][1])}]')
        lines.append(f'| {name} | ' + ' | '.join(parts) + ' |')
    lines += ['', '区间为逐任务独立、逐例配对 bootstrap（10,000 次，seed 73）。5% 和 20% 是预定敏感性分析，不能替换 10% 主结果。句聚合是检索规则，不改变 signed attribution 或声称新的 conservation。', '',
        '## 可复核记录', '',
        '- [固定实验计划](../RECALL_PILOT.md)、[开发集预算上限修正](../RECALL_CEILING_ADDENDUM.md)、[预先固定的样本](../recall_split.json)。',
        '- [全部开发候选](../recall_development/analysis.json)、[验证前固定的方案](../recall_development/choice.json)、[开发对照向量复现检查](../recall_development/control_verification.json)。',
        '- [验证分析与区间](analysis.json)、[逐例结果](cases.csv)。两种方法的输入、目标、范围、reference、向量与 Recall 已独立重建核对。',
        f'- 本次验证的已完成模型操作累计 {report["completed_gpu_operation_seconds"]:.2f} 秒，另计开发与此前被中断的全量实验。', '']
    if a.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={'width_ratios': [1.2, 1]})
        y = np.arange(5)
        d = np.array([means[t]['DT'] for t in TASKS]) * 100
        f = np.array([means[t]['FT_K3'] for t in TASKS]) * 100
        axes[0].barh(y-.17, d, .32, label='DT + sentence density', color='#246a73')
        axes[0].barh(y+.17, f, .32, label='FT K3 + sentence density', color='#bb6841')
        axes[0].set(yticks=y, yticklabels=LABELS, xlabel='Recall at 10% source-token budget (%)', xlim=(0, 100))
        axes[0].invert_yaxis()
        axes[0].legend(loc='lower right', fontsize=8)
        contrasts = [report['paired_contrasts'][choice][t]['0.1'] for t in TASKS]
        x = np.array([c['mean_difference'] for c in contrasts]) * 100
        low = np.array([c['ci95'][0] for c in contrasts]) * 100
        high = np.array([c['ci95'][1] for c in contrasts]) * 100
        axes[1].errorbar(x, y, xerr=[x-low, high-x], fmt='o', color='#246a73', capsize=4)
        axes[1].axvline(0, color='#666', linewidth=1)
        axes[1].set(yticks=y, yticklabels=LABELS, xlabel='DT minus FT K3 (percentage points)')
        axes[1].invert_yaxis()
        for ax in axes:
            ax.spines[['top', 'right']].set_visible(False)
            ax.grid(axis='x', alpha=.2)
            ax.set_axisbelow(True)
        fig.suptitle('Frozen Recall repair: 80 reserved cases, 16 per task', fontsize=13)
        fig.tight_layout()
        fig.savefig(a.analysis / 'recall_validation.png', dpi=180)
        plt.close(fig)
        lines.insert(6, '![Reserved-case Recall comparison](recall_validation.png)\n')
    (a.analysis / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    print(str(a.analysis / 'RESULTS.md'))


if __name__ == '__main__':
    main()

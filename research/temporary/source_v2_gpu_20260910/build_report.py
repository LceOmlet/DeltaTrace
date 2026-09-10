"""Present all predeclared tasks and paired intervals from a complete GPU run."""
import argparse
import csv
import json
from pathlib import Path
import statistics


def ci_text(point, scale=1., digits=3):
    mean = point['mean_difference'] * scale
    low, high = [v * scale for v in point['ci95']]
    return f'{mean:+.{digits}f} [{low:+.{digits}f}, {high:+.{digits}f}]'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plots', action='store_true')
    args = parser.parse_args()
    analysis = json.loads((args.analysis / 'analysis.json').read_bytes())
    assert analysis['quality_conclusion_allowed'] and analysis['case_count'] == 1048
    with (args.analysis / 'paired_cases.csv').open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    tasks = list(analysis['tasks'])
    groups = analysis['groups']
    primary = groups['all_tasks']['contrasts']['reference_change']['recall@10']
    description = ('有明确提高' if primary['ci95'][0] > 0 else
                   '有明确下降' if primary['ci95'][1] < 0 else '未显示明确提高或下降')
    means = {task: {method: statistics.mean(float(r[method + '_recall@10']) for r in rows if r['dataset'] == task)
                    for method in ('DT_full_reference', 'DT', 'FT_K3')} for task in tasks}
    lines = ['# source-v2 GPU 配对补跑结果', '',
        f'完整运行覆盖 11 个任务、1,048 例。相同正文范围与预算下，新 reference 相对旧 reference 的 DT Recall@10% 按任务等权平均差为 **{ci_text(primary, 100, 2)} 个百分点**（方括号为 95% 配对 bootstrap 区间），主比较{description}。', '',
        '两种 reference 的 DT 均为本次重新归因，使用相同模型、输入、固定完整响应和方法实现；FT K1/K3 也在同例 live 运行。指标范围和预算完全相同，差值隔离 reference 的影响。', '',
        '## 各任务 Recall@10%', '',
        '数值为百分比。FT 是本次 K3，旧 DT 是本次全 prompt reference 归因后放到相同正文范围评分，不能把这一列当作旧论文原始指标。', '',
        '| 任务 | 例数 | 旧 reference DT | 新 reference DT | live FT K3 | 新−旧差值与 95% CI（百分点） |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for task in tasks:
        report = analysis['tasks'][task]
        m = means[task]
        point = report['contrasts']['reference_change']['recall@10']
        lines.append(f'| {task} | {report["count"]} | {m["DT_full_reference"]*100:.2f} | {m["DT"]*100:.2f} | {m["FT_K3"]*100:.2f} | {ci_text(point, 100, 2)} |')
    lines += ['', '## 配对恢复率与删除指标', '',
        '差值均为新 DT 减去旧 reference DT：recovery 正值较好，RISE/MAS 负值较好。所有删除曲线都使用共同正文范围与同一终点，不比较两个不同删除问题的数值。分组按任务等权。', '',
        '| 分组 | Recall@10% 差值（百分点） | RISE 差值 | MAS 差值 |',
        '| --- | ---: | ---: | ---: |']
    for name, group in groups.items():
        c = group['contrasts']['reference_change']
        lines.append(f'| {name} | {ci_text(c["recall@10"], 100, 2)} | {ci_text(c["rise"])} | {ci_text(c["mas"])} |')
    lines += ['', '## 多预算结果', '',
        '同一正文候选集上重新计算每个预算。下表是全部 11 个任务等权的新−旧 reference 差值；不从次要预算中选择最好的数字作为主结论。', '',
        '| token 预算 | Recall 差值（百分点） | Precision 差值（百分点） | 上限归一化 recovery 差值（百分点） |',
        '| --- | ---: | ---: | ---: |']
    for percent in (5, 10, 20, 30, 50):
        c = groups['all_tasks']['contrasts']['reference_change']
        values = [ci_text(c[f'{field}@{percent:02d}'], 100, 2)
                  for field in ('recall', 'precision', 'ceiling_adjusted_recall')]
        lines.append(f'| {percent}% | ' + ' | '.join(values) + ' |')
    lines += ['', '## 与本次 live FT 的比较', '',
        'Recovery 对照是 K3，忠实度对照是 K1。以下是新 DT−FT，保留正负差值及不确定性。', '',
        '| 分组 | Recall@10% 差值（百分点） | RISE 差值 | MAS 差值 |',
        '| --- | ---: | ---: | ---: |']
    for name, group in groups.items():
        c = group['contrasts']['new_DT_minus_live_FT']
        lines.append(f'| {name} | {ci_text(c["recall@10"], 100, 2)} | {ci_text(c["rise"])} | {ci_text(c["mas"])} |')
    lines += ['', '## 执行与解释范围', '',
        '- 本地从保存向量重算所有 token recovery 点，并重建每例两个 reference 和全部实际删除输入的哈希。原输入、target、gold 与原缓存逐项一致。',
        '- 95% 区间使用预定的 10,000 次配对 bootstrap、seed=73；分组在任务内重采样，任务权重固定。次要列区间仅作描述，不能据单个最优列断言成功。',
        '- 新 reference 是否改善排序由上面的配对结果判断。正文范围与预算修正的定义合理性，不能代替归因质量提升的证据。',
        '- 旧冻结运行的向量与本次旧 reference 重跑存在数值差异；主比较使用本次两份新向量，不复用旧向量充当配对对照。逐例最大绝对差另存。',
        f'- 已记录模型操作耗时合计 {analysis["measured_call_seconds"]/3600:.2f} 小时，峰值分配显存 {analysis["peak_allocated_bytes"]/2**30:.2f} GiB。该成本包含补充旧 reference 对照、FT 与评测，不是单次 DT 归因成本。',
        '- 固定设计见 [PROTOCOL.md](PROTOCOL.md)，逐例数据见 [paired_cases.csv](analysis/paired_cases.csv)，完整统计及输入回执见 [analysis.json](analysis/analysis.json)。', '']
    args.output.mkdir(parents=True, exist_ok=True)
    if args.plots:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        labels = [t.replace('niah_', 'NI ').replace('vt_', 'VT ').replace('hotpotqa_long', 'HotpotQA') for t in tasks] + ['All tasks (equal weight)']
        data = [analysis['tasks'][t]['contrasts']['reference_change'] for t in tasks]
        data.append(groups['all_tasks']['contrasts']['reference_change'])
        fig, axes = plt.subplots(1, 3, figsize=(11.6, 6.1), sharey=True)
        for ax, field, title, scale, better in zip(axes, ['recall@10', 'rise', 'mas'],
                ['Recall@10% (pp)\nHigher is better', 'RISE\nLower is better', 'MAS\nLower is better'],
                [100, 1, 1], [1, -1, -1]):
            for i, row in enumerate(data):
                p = row[field]
                center = p['mean_difference'] * scale
                low, high = [v * scale for v in p['ci95']]
                color = '#087F6B' if (low > 0 if better > 0 else high < 0) else (
                        '#B74735' if (high < 0 if better > 0 else low > 0) else '#64748B')
                ax.plot([low, high], [i, i], color=color, linewidth=1.8)
                ax.plot(center, i, 'o', color=color, markersize=5 if i < len(tasks) else 7)
            ax.axvline(0, color='#9AA4AF', linewidth=.9, linestyle='--')
            ax.axhline(len(tasks) - .5, color='#D9DFE5', linewidth=.8)
            ax.set_title(title, fontsize=10, loc='left', pad=12)
            ax.set_xlabel('New minus full-prompt reference', fontsize=9)
            ax.spines[['top', 'right']].set_visible(False)
            ax.grid(axis='x', color='#E8ECF0', linewidth=.6)
            ax.set_axisbelow(True)
            ax.tick_params(labelsize=9)
        axes[0].set_yticks(range(len(labels)), labels)
        axes[0].invert_yaxis()
        fig.suptitle('Paired reference comparison on identical evidence-body budgets', fontsize=13, x=.02, ha='left')
        fig.text(.02, .015, '1,048 cases | 11 fixed tasks | 95% paired bootstrap intervals | Green: improvement; red: decline; gray: interval crosses zero', fontsize=8, color='#475569')
        fig.tight_layout(rect=(0, .04, 1, .94))
        fig.savefig(args.output / 'paired_reference_effects.png', dpi=200, facecolor='white')
        fig.savefig(args.output / 'paired_reference_effects.svg', facecolor='white')
        plt.close(fig)
        lines[4:4] = ['![Paired reference effects](paired_reference_effects.png)', '']
    (args.output / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    print(args.output / 'RESULTS.md')


if __name__ == '__main__':
    main()

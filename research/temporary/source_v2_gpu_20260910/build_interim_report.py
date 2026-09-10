"""Report complete task shards without presenting an unfinished suite as final."""
import argparse
import json
from pathlib import Path

from build_report import ci_text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    analysis = json.loads((args.analysis / 'analysis.json').read_bytes())
    assert analysis['status'] == 'verified_complete_tasks_partial_suite'
    assert analysis['scope'] == 'complete_task_interim' and not analysis['full_suite_complete']
    lines = ['# GPU 配对补跑：完整任务的中途结果', '',
        f'已完成并独立复核 {analysis["task_count"]} 个完整任务，共 {analysis["case_count"]} 例；当前范围的 {analysis.get("expected_case_count", 1048):,} 例全集尚未完成。以下不包含任何任务的样本前缀。', '',
        '两种 DT 都在本次实际重新归因，使用相同正文范围、相同预算和相同删除规则。Recovery 越高越好，RISE/MAS 越低越好。区间为 10,000 次配对 bootstrap 的 95% 区间。', '',
        '| 任务 | 例数 | 旧 reference DT | 新 reference DT | live FT K3 | 新−旧 Recall@10%（百分点，95% CI） |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for name, task in analysis['tasks'].items():
        means = task['method_means']
        values = [f'{means[m]["recall@10"]*100:.2f}%' for m in ('DT_full_reference', 'DT', 'FT')]
        point = task['contrasts']['reference_change']['recall@10']
        lines.append(f'| {name} | {task["count"]} | ' + ' | '.join(values) + f' | {ci_text(point, 100, 2)} |')
    lines += ['', '## 配对删除指标', '',
        '| 任务 | 新−旧 RISE（95% CI） | 新−旧 MAS（95% CI） |',
        '| --- | ---: | ---: |']
    for name, task in analysis['tasks'].items():
        c = task['contrasts']['reference_change']
        lines.append(f'| {name} | {ci_text(c["rise"], digits=5)} | {ci_text(c["mas"], digits=5)} |')
    lines += ['', '## 预定的句子/行级次要指标', '',
        '预算为 10% 的句子/行单元，选中后取回整个单元；与 10% token 预算不同。各方法单元格依次为单元 recall / 实际取回 token 占比。', '',
        '| 任务 | 旧 reference DT | 新 reference DT | live FT K3 | 新−旧单元 recall（百分点，95% CI） |',
        '| --- | ---: | ---: | ---: | ---: |']
    for name, task in analysis['tasks'].items():
        cells = [f'{task["method_means"][m]["unit_recall@10"]*100:.2f}% / '
                 f'{task["method_means"][m]["unit_selected_token_fraction@10"]*100:.2f}%'
                 for m in ('DT_full_reference', 'DT', 'FT')]
        lines.append(f'| {name} | ' + ' | '.join(cells) +
                     f' | {ci_text(task["contrasts"]["reference_change"]["unit_recall@10"], 100, 2)} |')
    lines += ['', '## 结果覆盖与限制', '']
    for name, task in analysis['tasks'].items():
        point = task['contrasts']['reference_change']['recall@10']
        finding = '显示提高' if point['ci95'][0] > 0 else '显示下降' if point['ci95'][1] < 0 else '尚未显示明确提高或下降'
        ties = task['tie10_diagnostics']
        lines.append(f'- {name}：主指标{finding}；新/旧 DT 在截止处存在影响 recall 的并列样本数分别为 '
                     f'{ties["DT"]["cases_with_recall_sensitive_tie"]}/{ties["DT_full_reference"]["cases_with_recall_sensitive_tie"]}。')
        lines.append(f'  Recall 提高 {point["positive_cases"]} 例、下降 {point["negative_cases"]} 例、持平 {point["tied_cases"]} 例。'
                     f'旧 reference DT−FT 为 {ci_text(task["contrasts"]["old_DT_minus_live_FT"]["recall@10"], 100, 2)} 个百分点。')
    lines += ['- 这些结果仅适用于已完成的任务，不能推断全部任务的平均收益。其余预定任务：' + '、'.join(analysis['remaining_tasks']) + '。',
        '- 所有 recovery 点由保存向量重新计算；每例两个 reference、全部实际删除输入、原输入与固定目标均已验回。',
        '- 对比对象和预算与旧论文指标不同；这里的新−旧差值隔离 reference 的影响，不能用新旧协议原始分数相减来声称改进。',
        '- 多预算、逐例值和完整区间分别保存在同目录的 analysis.json、paired_cases.csv。', '']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text('\n'.join(lines), encoding='utf-8')
    print(args.output)


if __name__ == '__main__':
    main()

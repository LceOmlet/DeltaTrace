"""Build a fully numeric report from the verified experiment artifacts."""
import csv
import hashlib
import json
from pathlib import Path

OWN = Path(__file__).resolve().parent
REPO = OWN.parents[2]


def read(path):
    return json.loads((OWN / path).read_bytes())


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(str(x) for x in row) + ' |' for row in rows])


def label(task):
    return {'macro': '等任务平均', 'math': 'MATH', 'morehopqa': 'MoreHopQA'}.get(task, task.replace('niah_', '').replace('_', '-').upper())


def main():
    a = read('raw/stage_a_v1/causal_analysis.json')
    b = read('raw/stage_b_v2/variant_analysis.json')
    c = read('raw/validation_v1/validation_analysis.json')
    d = read('raw/generalization_fresh_v1/generalization_analysis.json')
    role = read('token_role_diagnosis.json')
    audits = {stage: read(f'raw/{stage}/independent_verification.json')
              for stage in ('stage_a_v1', 'stage_b_v2', 'validation_v1', 'generalization_fresh_v1')}
    assert all(v['status'] == 'passed' for v in audits.values())
    assert read('raw/validation_v1/input_verification.json')['status'] == 'passed'
    assert c['status'] == d['status'] == 'complete'
    cm = {(r['dataset'], r['method']): r for r in c['means']}
    cp = {(r['dataset'], r['metric']): r for r in c['paired']}
    dm = {(r['dataset'], r['method']): r for r in d['means']}
    old, new = cm['macro', 'DT_original'], cm['macro', 'DT_gdn_symmetric']
    reduction = (old['rise'] - new['rise']) / old['rise'] * 100
    recall_gain = 100 * (new['recall'] - old['recall'])
    q4 = {r['method']: r for r in role['means'] if r['dataset'] == 'niah_mq_q4' and r['group'] == 'query_key'}
    factor = [r for r in a['query_factorization_task_means'] if r['target'] == 'full']
    screen = [r for r in b['means'] if r['dataset'] == 'macro']
    task_rows = []
    task_regressions = []
    ft_remaining = []
    tasks = list(read('validation_protocol.json')['selection'])
    for task in tasks + ['macro']:
        x, y, f = (cm[task, method] for method in ('DT_original', 'DT_gdn_symmetric', 'FT_K1'))
        task_rows.append([label(task), f"{x['rise']:.5f} → {y['rise']:.5f}", f"{f['rise']:.5f}",
                          f"{x['mas']:.5f} → {y['mas']:.5f}", f"{100*x['recall']:.2f}% → {100*y['recall']:.2f}%"])
        if task != 'macro':
            bad = [metric for metric in ('rise', 'mas', 'recall')
                   if (y[metric] < x[metric] if metric == 'recall' else y[metric] > x[metric])]
            if bad:
                task_regressions.append(label(task) + '：' + '、'.join(bad))
            if y['rise'] >= f['rise']:
                ft_remaining.append(label(task))
    interval_rows = [[metric.upper(), f"{cp['macro', metric]['delta']:+.6f}",
                      f"[{cp['macro', metric]['ci95'][0]:+.6f}, {cp['macro', metric]['ci95'][1]:+.6f}]"]
                     for metric in ('rise', 'mas', 'recall')]
    case_counts = []
    for metric in ('rise', 'mas', 'recall'):
        better = sum(cp[task, metric]['better'] for task in tasks)
        equal = sum(cp[task, metric]['equal'] for task in tasks)
        case_counts.append([metric.upper(), better, equal, 60 - better - equal])
    broad_rows = []
    broad_bad = []
    for task in ('math', 'morehopqa', 'macro'):
        x, y, f = (dm[task, method] for method in ('DT_original', 'DT_gdn_symmetric', 'FT_K1'))
        broad_rows.append([label(task), f"{x['rise']:.5f} → {y['rise']:.5f}", f"{f['rise']:.5f}",
                           f"{x['mas']:.5f} → {y['mas']:.5f}"])
        if task != 'macro':
            for metric in ('rise', 'mas'):
                if y[metric] > x[metric]:
                    broad_bad.append(label(task) + ' ' + metric.upper())
    cost_rows = [[r['method'], f"{r['median_seconds']:.4f} s", f"{r['mean_seconds']:.4f} s",
                  f"{r['peak_allocated_bytes']/1e9:.3f} GB"] for r in d['cost']]
    passed = '通过' if c['passed'] else '未通过'
    text = f"""# Qwen3.5 NIAH RISE：控制解释与统一 GDN 改进

2026-09-13。冻结候选 **DT-GDN-Symmetric {passed}了预先规定的 60 例验证条件**。
等任务平均 RISE 从 **{old['rise']:.5f} 降到 {new['rise']:.5f}**（相对降低 {reduction:.2f}%），
MAS 从 **{old['mas']:.5f} 到 {new['mas']:.5f}**，Recall@10% 从
**{old['recall']:.2%} 到 {new['recall']:.2%}**（{recall_gain:+.2f} 个百分点）。
这里改进的是归因；模型权重与生成能力没有改变。该候选是本次四项统一规则候选中的筛选最优，不能称为全局最优。

## 证据最支持的解释

原始 RISE 衡量删除输入 token 后，完整固定推理及答案的总对数概率下降得多快。
问题中的查询键也可以被删除。因此，优先找到针的位置与更快降低这个完整响应分数并不等价。
原始 DT 对两端点总变化的有限分配，与 RISE 要求的逐步删除排名，是不同的对象；总和守恒不保证删除排名最优。

在 600 例原始观察之后，预先固定 MQ-Q2、MQ-Q4、MV-V2 各 4 个索引（11、37、63、89），
执行实际模型干预，分别保留问题、只删除问题，以及原来的完整删除路径。
下表所有分解项共用原始全保留/全删除分母，并在两个删除路径的组合上做两因素 Shapley 分解。
这是**删除路径差异的因果分解**，不是将差距直接归因给某个内部算子的证明。

{table(['任务（各 4 例）', '原 DT−FT RISE', '问题删除路径贡献', '其余删除路径贡献', '保留问题后的差距（共同分母）'],
       [[label(r['dataset'])] + [f"{r[k]:+.6f}" for k in ('dt_minus_ft', 'query_schedule_contribution', 'nonquery_schedule_contribution', 'fixed_original_denominator_restore_query_gap')] for r in factor])}

MQ-Q2 的差距主要来自问题删除路径；MQ-Q4 的问题路径约占原差距的 52%，其余路径也有贡献。
MV-V2 虽有相同的问题路径劣势，但证据路径优势足以抵消它。
保留问题并相应重建 DT 参考端点后，三个诊断任务中 DT 的新范围 RISE 都低于 FT；这些是另一个干预范围，不能替换论文原指标。

控制试验还限制了另外两种解释：

- 统一到完整响应中非停用 token 的归因目标后，MQ-Q2/MQ-Q4 的差距仍在，FT 初始答案目标与 DT 全响应目标的差异不是本组 MQ 差距的主要解释。
- 只对最终答案内容评分时，8 个 MQ 诊断样本中有 4 个出现“全输入得分低于全删除得分”的端点反转，且 FP64 求和诊断仍存在。固定推理前缀已经包含答案，答案条件概率可退化；这样的答案单独 RISE 不能拿来证明原问题已修好。只用答案归因再按原完整响应评分也明显变差。

这 12 例诊断中的原始数值重放和归因向量均通过历史核验；先前 600 例的负值裁剪效应也远小于 MQ 差距。
目前证据支持**交互贡献的分配与删除排序不吻合**这一解释，而非将原始 RISE 表现直接视为数值实现错误。

## 实际改法与候选筛选

全部 24 个 GDN 层统一做两处有限规则改变，没有按任务、层号或 gold 位置选择规则。
对输出乘积 n·s，原 content1 分配为 s₁Δn+n₀Δs；改为
((s₀+s₁)/2)Δn+((n₀+n₁)/2)Δs。
对记忆递推，使用同一个上游系数，平均现有两端点捕获在正序和逆序下得到的局部有限系数。
两种局部次序均满足相应有限差分恒等式，在精确算术下平均仍保持该恒等式；真实精度残差继续记录。
原生 forward、完整响应加 EOS 的目标、EOS 参考、FA/QK/MLP 规则和原始评分函数保持固定。

筛选集合就是上面的 12 个已用诊断样本；它不构成独立验证。
按预先规则，要求两个 MQ 与总体 RISE 降低，同时 MAS 最多退步 0.01、Recall 最多退步 0.02，再在合格者中取平均 RISE 最低者。
所有候选均保留：

{table(['方法（12 例）', 'RISE ↓', 'MAS ↓', 'Recall@10% ↑'],
       [[r['method'], '—' if r['rise'] is None else f"{r['rise']:.6f}", '—' if r['mas'] is None else f"{r['mas']:.6f}", '—' if r['recall'] is None else f"{r['recall']:.2%}"] for r in screen])}

对 MQ-Q4 的 token 角色分析说明，这不是简单提高整个问题的总权重：
问题总贡献比例略降，而查询键的比例从 13.67% 升到 15.51%。
前 20% 删除中，被删查询键比例从 {q4['DT_original']['fraction_deleted_4']:.2%}
升到 {q4['DT_gdn_symmetric']['fraction_deleted_4']:.2%}；前 10% 的针数值删除比例从 73.21% 升到 96.43%。
这支持问题内部及证据 token 之间的分配改善；该角色分析是筛选样本的描述结果，不证明唯一的中介机制。

## 冻结后的 60 例验证

先冻结候选及源码哈希，再以种子 13720260913 构造六个 NIAH 任务各 10 个新键和值。
模板来自预先选定的原始索引 5、15、…、95，与筛选索引分离；提示、固定响应、答案和 gold 字符跨度一致替换。
没有重新生成模型响应。这检验的是**原模板上的新键值泛化**，不是新自然任务或未见过的模板。

预定通过条件：等任务平均配对 RISE 差值的任务内分层 bootstrap 95% 区间上界小于零，
同时平均 MAS 与 Recall 不退步。10,000 次，种子 20260913，每个任务内部重采样配对样本，再等权平均六个任务。
结果：**{passed}**。逐任务及次要指标区间只作描述，没有多重比较校正。
这些区间条件于本次执行环境，不包含跨进程原生后端差异。

{table(['任务（各 10 例）', 'DT → 新规则 RISE ↓', 'FT K1 RISE ↓', 'DT → 新规则 MAS ↓', 'DT → 新规则 Recall@10% ↑'], task_rows)}

{table(['指标', '新规则−原 DT', '任务内分层 bootstrap 95% 区间'], interval_rows)}

{table(['指标', '逐例改善', '持平', '退步'], case_counts)}

逐任务均值退步：{'；'.join(task_regressions) if task_regressions else '无（RISE、MAS、Recall 三项）'}。
新规则 RISE 仍未低于 FT K1 的任务：{'、'.join(ft_remaining) if ft_remaining else '无'}。
FT 忠实度使用 K1，检索同时保存 K1 与 K3；上表的 Recall 箭头只比较两种 DT。全部 FT 及逐例值见 JSON/CSV。

## 更广任务、入口一致性与代价

MATH、MoreHopQA 各固定 6 个原始索引（10、26、42、58、74、90），检查新规则的任务迁移。
原始基准先前已经被检查过，这部分是固定索引的描述性迁移检查，不是未见测试集，也没有据此重选规则。
本表中的原 DT、新规则与 FT 均在同一个新进程中重新归因和评分，不复用历史向量或指标。

{table(['任务（各 6 例）', 'DT → 新规则 RISE ↓', 'FT K1 RISE ↓', 'DT → 新规则 MAS ↓'], broad_rows)}

迁移检查的均值退步：{'；'.join(broad_bad) if broad_bad else '无'}。
新入口与冻结研究原型在实际模型上逐位一致：**{d['deployment_bitwise']}**。
60 例新输入的原 DT 重复归因最大相对 L2 为 {c['max_baseline_repeat_relative_l2']:.1g}；
12 例迁移输入在同一进程内的原 DT 重复归因最大相对 L2 为 {d['max_baseline_repeat_relative_l2']:.1g}。
迁移输入与历史存档的实际 token IDs、完整目标、eligible/gold 均独立核对一致。

采用新基线的原因：前两次历史向量复用检查在产生质量指标前中止。
MATH 首例历史原生根效应为 73.18754，新进程分别为 73.53494 和 72.64418，
向量相对 L2 差异分别为 0.016657、0.014765；各进程内部重复结果均逐位一致。
差异发生在归因规则之前，具体的原生后端原因尚未定位，不能归功或归咎于候选改动。
保留原有历史一致性阈值和全部失败记录，另立[同进程新基线协议](generalization_fresh_protocol.json)进行本表比较。
历史漂移另存于迁移分析的 `historical_diagnostics`，不计作改进。

成本在 {label(d['cost_case']['dataset'])} 索引 {d['cost_case']['index']} 的同一实际输入上测量
（总长 {d['cost_case']['input_tokens']} token，目标 {d['cost_case']['target_tokens']} token），
两种路径运行后额外预热一轮，再交替测量 5 对完整归因。
包括当前 CPU checkpoint 和同步诊断，不含模型加载和删除评分；不能外推到批处理生产速度。

{table(['方法', '中位时间', '平均时间', '峰值 allocated 显存'], cost_rows)}

中位时间比为 {d['cost_time_ratio']:.3f}×。每次完整归因的有限记忆调用由 24 次增至 48 次，
不增加原生模型根 forward。该代价必须和质量改善一起考虑。

## 核验、记录与复现

原始评分使用同一模型与 BF16 原生 scorer；独立 CPU 程序重新核验实际输入哈希、每个输出 token 的原生分数求和、
删除预算与排序、RISE、MAS 和 Recall。Recall 边界并列时保留合法区间，不人为指定更有利的 tie break。
60 个新输入另外逐字节解码实际 prompt 与完整 target，并重算 gold 与 eligible 映射。
60 例中新规则的最大绝对守恒相对残差为 {100*c['max_absolute_relative_residual']['DT_gdn_symmetric']:.3f}%
（原 DT 为 {100*c['max_absolute_relative_residual']['DT_original']:.3f}%）；守恒仍不等于删除排名最优。

{table(['阶段', '完成样本', '独立核验'], [[stage, v['cases'], v['status']] for stage, v in audits.items()])}

一次候选筛选预运行在产生指标前被原 DT 向量一致性保护中止，原因尚未定位。
失败日志完整保留于 `raw/stage_b_v1.log`；没有降低 1e-4 阈值。
随后完整筛选运行与后续逐例重复检查用于核实稳定性；不能把那次保护失败写成“已定位并修复的 bug”。

- 方法入口：[qwen35_gdn_symmetric.py](../../../deltatrace/profiles/qwen35_gdn_symmetric.py)，用法见[profiles README](../../../deltatrace/profiles/README.md)。
- 筛选冻结凭据：[candidate_receipt.json](candidate_receipt.json)；新输入及条件：[validation_protocol.json](validation_protocol.json)、[validation_cases.json](validation_cases.json)。
- 因果分析：[causal_analysis.json](raw/stage_a_v1/causal_analysis.json)；所有候选：[variant_analysis.json](raw/stage_b_v2/variant_analysis.json)。
- 验证分析：[validation_analysis.json](raw/validation_v1/validation_analysis.json)；迁移与成本：[generalization_analysis.json](raw/generalization_fresh_v1/generalization_analysis.json)。
- 全部新验证逐例表：[validation_cases_metrics.csv](validation_cases_metrics.csv)；文件身份：[artifact_manifest.json](artifact_manifest.json)。

在同一已核验的 Qwen3.5/MetaX 运行环境中，先运行 `run_causal.py` 与 `run_variants.py`（对应各自协议），
再使用冻结输入运行 `run_validation.py --run-root RUN_ROOT --protocol validation_protocol.json --cases validation_cases.json --output NEW_OUTPUT`。
迁移运行使用 `run_generalization.py --run-root RUN_ROOT --protocol generalization_fresh_protocol.json --cases generalization_fresh_cases.json --output NEW_OUTPUT`；
将独立方法模块原字节复制到驱动旁的 `qwen35_gdn_symmetric.py`。
驱动验证既有模型、运行库与干净源码身份，并拒绝占用中的 GPU。
本次依赖和实际源码副本保留在每个 raw 阶段目录及原有环境归档中。

CPU 复核依次运行 `verify_variants.py RUN`、`analyze_validation.py RUN`；输入核验为
`verify_validation_inputs.py RUN --source SOURCE_DATA --tokenizer TOKENIZER_JSON`。
迁移汇总运行 `analyze_generalization.py RUN --archive-root HISTORICAL_FULL_RUN`。最后运行 `build_report.py` 重建本报告。
原论文表格继续引用原始冻结方法；本报告的候选值不混入旧表。
"""
    (OWN / 'RESULTS_zh.md').write_text(text, encoding='utf-8')
    with (OWN / 'validation_cases_metrics.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(c['rows'][0]))
        writer.writeheader()
        writer.writerows(c['rows'])
    paths = list(OWN.glob('*.py')) + list(OWN.glob('*protocol.json')) + [OWN / 'candidate_receipt.json', OWN / 'validation_cases.json',
             OWN / 'generalization_fresh_cases.json', OWN / 'RESULTS_zh.md', OWN / 'validation_cases_metrics.csv', REPO / 'deltatrace/profiles/qwen35_gdn_symmetric.py']
    paths += [path for path in (OWN / 'raw').glob('*/*.json') if path.name != 'input_cases.json']
    manifest = dict(version='qwen35-gdn-symmetric-controlled-improvement-v1', validation_passed=c['passed'],
                    files=[dict(path=str(path.relative_to(REPO)).replace('\\', '/'), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                           for path in sorted(set(paths))])
    (OWN / 'artifact_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(dict(report=str(OWN / 'RESULTS_zh.md'), passed=c['passed'], rise_relative_reduction_percent=reduction,
                         recall_gain_percentage_points=recall_gain, task_regressions=task_regressions,
                         ft_remaining=ft_remaining, broad_regressions=broad_bad, cost_ratio=d['cost_time_ratio']), ensure_ascii=False))


if __name__ == '__main__':
    main()

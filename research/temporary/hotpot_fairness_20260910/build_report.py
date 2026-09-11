"""Render reviewable tables directly from the verified correction analysis."""
import csv
import json
from common import HERE,sha

TARGET={'answer_only':'删前缀后答案','full':'完整回答'}
LABEL={'official_restored':'官方 ID，恢复正文坐标','review_corrected':'另修订两处关系标签'}
METHOD={'DT_target':'DT','FT_K1':'FT K1','FT_K3':'FT K3'}
VIEWS={'raw_cached':'原始词元排序 + 缓存标签','raw_restored':'原始排序 + 正文坐标修正',
       'regex_cached':'旧正则句均值 + 缓存标签','regex_restored':'旧正则句均值 + 正文坐标修正',
       'regex_anchor_restored':'再修正前导空格归属','native_token_restored':'再使用原生句边界（仍按词元截断）'}
pct=lambda x:f'{100*x:.2f}'
sign=lambda x:f'{100*x:+.2f}'
ci=lambda r:'['+', '.join(sign(x) for x in r['interval'])+']'

def csv_file(name,rows):
    with (HERE/name).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    a=json.loads((HERE/'analysis.json').read_bytes());audit=json.loads((HERE/'source_audit.json').read_bytes())
    assert a['status']=='verified_complete_hotpot_fairness_correction'
    lines=['# HotpotQA 定位修正与探索性评测：完整 48 例', '',
        '**复审更正：不能将整套 v2 协议称为已验证的公平性修复。** 标题/正文坐标和首词归属修正有直接依据；答案实验删掉了原推理前缀，整句表收费的是过滤标点/空白后的有效词元，聚合与装箱策略也未证明中立。详见 [复审及最小修正结果](REVIEW.md)。', '',
        '下列数值保留为原 v2 预先指定的探索性对照。删除推理前缀后的答案重建实验为 **DT 68.58%、FT K3 66.15%**；另修订两处语义标签后为 **68.58% 对 67.19%**。它解释的是新构造的直接答案条件，不能当作保留原推理的最终答案归因。四项原定比较的区间均跨零。', '',
        '本次新增重跑全部 48 例的答案重建条件 DT、live FT K1/K3，完整回答使用已验证的 48 例原始向量。候选选择不依赖归因值或 gold；规则和五个待审例索引在修正结果计算前固定。属于回顾性实验，包含既有开发样本，不是新的独立留出实验。', '',
        '## 找到的问题及修正', '',
        '| 问题 | 直接证据 | 已实施的修正 |',
        '| --- | --- | --- |',
        '| 同名标题被当作正文 gold | 第 8 例 `Why Is There Air?` 第 0 句旧坐标 `[762,779)` 落在标题；正文应为 `[780,797)`。仅核对文本和句子 ID 会漏检。 | 用完整原文文档定位，再对原生句子数组做前缀和；主表保留官方句子 ID，恢复正确正文位置。 |',
        '| 前导空格使词元落到上一句 | 124 条官方支持事实中 59 条受影响；全部候选内有 1,673 个合资格词元的正则组号因此变化。 | 以词元第一个非空白字符定位，用原始 context 数组作为统一句子边界。 |',
        '| 按词元 Recall 会使长事实权重更大 | 同例 gold 句子的有效词元长度最大/最小比中位数为 2.06，最高 7 倍。 | 增加每条支持事实等权的 P/R/F1/EM，主排序双方都用正值词元均值；检索整句并支付实际合资格词元成本。 |',
        '| 两处原生支持标签缺失问题所需关系 | 第 36 例标建筑用途却漏了建筑师身份；第 40 例标商场面积/设施却漏了 Wilmorite 到 Macerich 的所有权转移。 | 在明确标识的敏感性表中，对双方将第 36 例句 1 换为句 2、第 40 例句 2 换为句 1；官方标签主表仍保留。 |',
        '| 解释目标与证据检索问题容易混淆 | 平均完整回答 57.35 个词元，最终答案 13.56 个；其余 43.79 个词元属于答案跨度外内容。 | 全部 48 例同时报告最终答案与完整回答目标；每个目标内 DT/FT 接收完全相同的输入和目标权重。 |', '',
        '完整回答衡量解释整个回答；删除前缀后的答案实验衡量另一个条件下的答案。前两项是定位问题；按词元或按支持事实计分是不同指标定义，不能仅凭长短加权就认定前者不公平。语义标签修订是有原文依据、但未经多人独立复核的敏感性注释。', '',
        '**更正此前判断：** 早先“gold 映射没有问题”的说法只核对了文本和句子 ID，不充分；第 8 例的位置确实错误。修正它本身没有给 DT 带来收益：完整回答的原始词元 Recall 从 40.6337% 降为 40.6017%，FT K3 不变。这一错误应修，无论对哪种方法有利。', '',
        '源文核验覆盖 **480/480 文档、2,031 个原生句子、48/48 词元映射**。三个跨原生边界的不可分 BPE 词元（第 12、31、44 例）全部按首个非空白字符归属，并只计一次成本，见 [边界说明](TOKEN_BOUNDARY_NOTE.md) 和 [逐例源文审计](source_audit.json)。', '',
        '## 共同句聚合与预算', '',
        '候选是所有原生正文句子，不是只取 gold 句。标题和文档编号保留在模型输入中，但不作为原生支持事实候选。对每句的原有合资格词元，双方采用同一个正值算术均值，平分按源文位置排序。负向 DT 原始向量保存在归档中，本表采用共同的正值检索视图。', '',
        '词元预算为 `ceil(f × N_body)`，其中 `N_body` 为原正文句子的合资格词元数；沿排序访问整句，能够放入剩余预算才选入，否则跳过。逗号、句点及纯空白 stop-token 过滤沿用作者规则，因此这里的收费是“合资格词元”，不是整句全部词元成本。另报告固定 top 2/4/8 句及有效词元收费。相同装箱规则并不保证中立，复审发现 66/288 组 10% 结果超过相应全部正文词元预算，见复审报告。', '',
        '原生支持事实是 `(title, sentence_id)` 集合，P/R/F1/EM 按 [HotpotQA 官方评测器](https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py) 的集合定义计算。各例先评分再宏平均；完整支持命中允许额外误检，EM 则要求预测集合与 gold 完全相同。10% 预算下，官方标签的平均最大可达事实 Recall 为 **93.06%**，不是 100%。', '',
        '## 原 v2 指定比较：10% 正文有效词元预算的整句支持事实 Recall', '',
        '单位为百分比；差值与区间单位为百分点。FT K3 是固定主对照，FT K1 全表另报。两种目标 × 两种标签共四项比较，使用配对 bootstrap 10,000 次、seed 73、Bonferroni 98.75% 描述区间。', '',
        '| 目标 | 标签 | n | DT | FT K3 | DT−FT | 98.75% 配对区间 |',
        '| --- | --- | ---: | ---: | ---: | ---: | --- |']
    primary=[]
    for r in a['primary_comparisons']:
        lines.append(f"| {TARGET[r['target_mode']]} | {LABEL[r['labels']]} | 48 | {pct(r['dt'])} | {pct(r['ft'])} | {sign(r['dt_minus_ft'])} | {ci(r)} |")
        primary.append(dict(target_mode=r['target_mode'],labels=r['labels'],n=48,dt_recall=r['dt'],ft_k3_recall=r['ft'],
            dt_minus_ft=r['dt_minus_ft'],ci_low=r['interval'][0],ci_high=r['interval'][1],confidence=r['confidence']))
    lines+=['','## 逐项隔离修正：保持原候选集与原 10% 词元预算', '',
        '下表继续计算旧式词元 Recall，因而可以隔离坐标、空格与句边界的影响。它包含标题等旧候选元数据，且允许在词元预算处截断，分母也与上面的原生整句主表不同；两表绝对值不可直接当成同一指标的提升。', '',
        '| 固定目标 | 诊断视图 | DT | FT K3 | DT−FT |',
        '| --- | --- | ---: | ---: | ---: |']
    for target in ('full','answer_only'):
        for view in VIEWS:
            r=next(x for x in a['diagnostic_summary'] if x['target_mode']==target and x['view']==view and x['fraction']==.1)
            lines.append(f"| {TARGET[target]} | {VIEWS[view]} | {pct(r['methods']['DT_target']['recall'])} | {pct(r['methods']['FT_K3']['recall'])} | {sign(r['paired']['FT_K3']['recall']['dt_minus_ft'])} |")
    lines+=['','完整回答目标下，仅换成原生句边界后，旧式词元 Recall 的 DT−FT 从旧正则视图的 +1.73 点变成 −0.42 点。可见旧分句方式确实会改变比较，不能把旧句聚合的优势直接当成方法优势。', '',
        '## 同预算的质量与成本明细', '',
        '下表为完整 48 例、官方恢复坐标标签、10% 正文词元预算。P/R/F1/EM 和完整支持为百分比；成本和句数为逐例均值。', '',
        '| 目标 | 方法 | P | R | F1 | EM | 完整支持 | 有效词元收费 | 未用有效词元 | 选中句数 |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for target in ('answer_only','full'):
        r=next(x for x in a['native_summary'] if x['target_mode']==target and x['labels']=='official_restored' and x['subset']=='all48' and x['budget_unit']=='body_tokens' and x['budget_value']==.1)
        for method in METHOD:
            m=r['methods'][method]
            lines.append('| '+' | '.join([TARGET[target],METHOD[method]]+[pct(m[k]) for k in ('precision','recall','f1','exact_match','complete_support')]+[f"{m[k]:.2f}" for k in ('spent_tokens','unused_tokens','selected_sentences')])+' |')
    lines+=['','答案目标的 DT Recall 略高，但其 F1（47.58%）低于 FT K3（47.78%）；修订两处标签后 FT K3 的 F1 为 48.30%。因此不把单列 Recall 均值当作整体优胜。', '',
        '## 固定 45 例敏感性子集', '',
        '第 15 例人口所属城市指代含糊，第 16 例题目与源文南北方向相反，第 30 例的刊物频率证据不足以证明“非小报格式”。主表不删任何一例；下面对双方固定排除同样三个索引。修订标签来自 [明确的审阅记录](semantic_review.json)，完整支持文档见 [源文摘录](review_source_excerpt.json)。', '',
        '| 目标 | 标签 | DT Recall | FT K3 Recall | DT−FT | 95% 描述区间 |',
        '| --- | --- | ---: | ---: | ---: | --- |']
    for r in a['native_summary']:
        if r['subset']=='unambiguous45' and r['budget_unit']=='body_tokens' and r['budget_value']==.1:
            lines.append(f"| {TARGET[r['target_mode']]} | {LABEL[r['labels']]} | {pct(r['methods']['DT_target']['recall'])} | {pct(r['methods']['FT_K3']['recall'])} | {sign(r['paired']['FT_K3']['recall']['dt_minus_ft'])} | {ci(r['paired']['FT_K3']['recall'])} |")
    lines+=['','## 验证与可复现文件', '',
        '- 新增答案目标 48/48 例完成；8 例既有开发对照的 DT、FT K1、FT K3 共 24 份向量逐位一致。',
        '- 全部新例从原始向量重算并核对输入、最终答案跨度、实际 FT 初始目标权重、reference 哈希、旧 gold 与旧预算分数。完整回答逐例来源绑定旧完整基准的已验证文件哈希。',
        '- 288 组旧正则排序精确重现；5,184 条诊断分数和 3,456 条原生事实分数覆盖所有样本、方法、目标与预算。标签变体复用完全相同的选择结果。',
        '- 8 项错误回归测试通过；另从归档的检索集合独立核算全部 P/R/F1/EM、成本与预算，验证回执见 [verification.json](verification.json)。',
        '- 原 GPU 运行代码、原目标构造、原 reference 和 27 项冻结方法文件保留；新增的是显式选择的 `hotpot-evidence-v2` 评分入口。历史 [448 例表](../source_v2_gpu_20260910/full_recall/RESULTS.md) 继续作为旧 HotpotQA 标签/分句口径的可复现实验。', '',
        '| 文件 | 内容 |', '| --- | --- |',
        '| [主比较 CSV](primary_recall10.csv) | 四项 Recall、配对差及调整区间 |',
        '| [所有原生预算汇总](native_summary.csv) | 5/10/20% 词元与 top 2/4/8 句、全部目标/标签/方法/子集、质量和实际成本 |',
        '| [诊断汇总](diagnostic_summary.csv) | 各项实现修正的 5/10/20% 对照 |',
        '| [原生逐例 CSV](native_per_case.csv) / [诊断逐例 CSV](diagnostic_per_case.csv) | 所有完整逐例数值 |',
        '| [检索集合](selections.json) / [候选](candidates.json) / [标签](labels.json) | 排序结果与评分标签分离，可直接重计集合分数和成本 |',
        '| [分析与配对区间](analysis.json) | 代码与来源哈希、48 例验证、全部统计 |',
        '| [答案目标原始归档清单](raw/answer_v2/manifest.json) | 七份文件的原始及压缩字节哈希，gzip 无损 |',
        '| [运行前协议](PROTOCOL.md) / [运行回执](execution_receipt.json) | 预先固定规则及 GPU 执行身份 |', '',
        'CPU 复现需要 `numpy` 和 `tokenizers`。在仓库根目录运行，源数据位置可按本机目录替换；源文件和缓存哈希由协议强制核验。', '',
        '```bash',
        'python research/temporary/hotpot_fairness_20260910/test_evidence.py',
        'python research/temporary/hotpot_fairness_20260910/prepare_inputs.py --source ../audit/primary_sources/hotpot_dev_distractor_v1.json --cache ../audit/published_flashtrace/table1-data-v1/extracted/data/hotpotqa_long.jsonl --tokenizer ../audit/deltatrace-figures-20260909/qwen3-tokenizer.json',
        'python research/temporary/hotpot_fairness_20260910/analyze_fairness.py --cache ../audit/published_flashtrace/table1-data-v1/extracted/data/hotpotqa_long.jsonl --tokenizer ../audit/deltatrace-figures-20260909/qwen3-tokenizer.json --publication ../DeltaTrace-paper-qwen3/experiments/official/results/qwen3_8b_table1_20260909',
        'python research/temporary/hotpot_fairness_20260910/build_report.py',
        'python research/temporary/hotpot_fairness_20260910/verify_outputs.py',
        '```', '',
        '如需重新运行 GPU，先用旧 `archive_results.py --restore` 将 `raw/answer_dev_corrected_v1` 对照恢复到新目录，再执行 `run_answer.py --environment /path/environment.json --control /path/restored-control --output /path/new-answer-run`。驱动拒绝覆盖既有输出，验证权重、共同目标与每个开发重叠对照；用 `archive_answer.py` 生成有哈希清单的归档。评分默认读取这里保存的版本化归档。', '',
        '确定修复的是两个定位问题。其余协议选择没有获得“公平性已验证”的结论，见 [复审](REVIEW.md)。原生支持事实仍可能省略替代证明、标题或指代信息，单个助手的语义修订也不能视为官方真值；证据重合率不能单独证明归因的因果忠实度。', '']
    csv_file('primary_recall10.csv',primary)
    native=[];diagnostic=[]
    for r in a['native_summary']:
        for method,m in r['methods'].items():
            row={k:r[k] for k in ('target_mode','labels','subset','n','budget_unit','budget_value')}
            native.append(dict(row,method=method,**m))
    for r in a['diagnostic_summary']:
        for method,m in r['methods'].items():
            diagnostic.append(dict(target_mode=r['target_mode'],view=r['view'],fraction=r['fraction'],n=r['n'],method=method,**m))
    csv_file('native_summary.csv',native);csv_file('diagnostic_summary.csv',diagnostic)
    (HERE/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(dict(primary_rows=len(primary),native_summary_rows=len(native),diagnostic_rows=len(diagnostic),report_sha256=sha(HERE/'RESULTS.md'))))

if __name__=='__main__':main()

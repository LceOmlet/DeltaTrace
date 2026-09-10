"""Render every frozen v3 comparison, including unfavorable outcomes."""
import csv
import json
from artifacts_v3 import HERE

TARGET={'full':'完整回答','answer_conditioned':'保留原推理的答案'}
POOL={'signed_sum':'有符号总和','positive_mean_eligible':'正值有效词元均值'}
METHOD={'DT_target':'DT','FT_K1':'FT K1','FT_K3':'FT K3'}
POLICY={'eligible_skip':'旧有效词元计费＋跳过放不下的句子','all_tokens_skip':'全部正文词元计费＋跳过','all_tokens_prefix':'全部正文词元计费＋排序前缀（v3）'}
pct=lambda x:f'{100*x:.2f}'
sign=lambda x:f'{100*x:+.2f}'
ci=lambda r:'['+', '.join(sign(x) for x in r['interval'])+']'

def write_csv(name,rows):
    with (HERE/name).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    a=json.loads((HERE/'analysis.json').read_bytes());audit=json.loads((HERE/'source_audit.json').read_bytes())
    assert a['status']=='verified_complete_context_preserving_full_cost_evaluation'
    assert len(a['primary_comparisons'])==4 and not a['neutral_protocol_proven'] and not a['new_holdout']
    native=[];primary=[];packing=[];legacy=[]
    for r in a['native_summary']:
        for method,m in r['methods'].items():
            native.append({k:r[k] for k in ('target_mode','pooling','budget_unit','budget_value','n')}|dict(method=method,**m))
    for r in a['primary_comparisons']:
        primary.append(dict(target_mode=r['target_mode'],pooling=r['pooling'],n=48,dt_recall=r['dt'],ft_k3_recall=r['ft'],
            dt_minus_ft=r['dt_minus_ft'],ci_low=r['interval'][0],ci_high=r['interval'][1],confidence=r['confidence']))
    for r in a['packing_summary']:
        for method,m in r['methods'].items():packing.append({k:r[k] for k in ('target_mode','policy','fraction','n')}|dict(method=method,**m))
    for r in a['legacy_summary']:
        for method,m in r['methods'].items():legacy.append({k:r[k] for k in ('target_mode','view','fraction','n')}|dict(method=method,**m))
    for name,rows in [('native_summary.csv',native),('primary_recall10.csv',primary),('packing_summary.csv',packing),('legacy_summary.csv',legacy)]:write_csv(name,rows)
    def summary(target,pool,unit,value):return next(r for r in a['native_summary'] if r['target_mode']==target and r['pooling']==pool and r['budget_unit']==unit and r['budget_value']==value)
    lines=['# HotpotQA v3：保留完整上下文、计入全部正文词元成本', '',
        '本轮完成了可具体检验的修正：新增 **48/48 例**保留原推理的答案归因；整句收费计入逗号、句点和空白；按固定排序的可支付前缀选句。全部原官方支持事实保留。完整回答与答案条件、有符号总和与旧正值均值四组结果均报告，未依据结果选择目标、池化或标签。', '',
        '**结果不支持 DT 在保留原推理的答案条件下胜过 FT K3。** 10% 全部正文词元预算下，有符号总和为 **DT 35.59%、FT K3 51.04%**；正值有效词元均值为 **48.96% 对 64.76%**。两项 DT−FT 的调整描述区间均低于零。完整回答的两项均值差为正，但区间均跨零。', '',
        '这修复了输入条件的混淆和对实际收费范围的过度表述，并使选择随预算嵌套；不等于证明整套测量中立，也不证明一种归因方法普遍更忠实。48 例来自反复检查过的发布缓存，含既有开发样本；区间是描述性统计，不是独立留出确认。', '',
        '## 本次实际改变了什么', '',
        '| 项目 | v3 实现与核验 |', '| --- | --- |',
        '| 输入与目标 | 保留每例原始完整 `input_ids`、推理前缀、最终答案和 EOS reference，仅把初始目标权重限制到原答案跨度。48 例完整输入、reference 哈希及所有目标位置的原输入端点 logprob 逐位匹配旧完整回答。 |',
        '| FT 回溯 | 使用冻结的 InitialTargetFT：初始权重与 DT 相同，后续跳数保留完整生成跨度。8 例既有 conditioned 开发对照的 DT、FT K1/K3 共 24 份向量逐位复现。 |',
        '| 成本 | 总正文词元由旧有效词元 56,273 增至全部词元 62,415。被旧逗号/句点/空白过滤排除的词元虽在全部 288 个目标×方法×样本向量中贡献为零，检索时仍支付其完整成本。 |',
        '| 预算选择 | 预算为 `ceil(f × N_all_body)`；沿固定句子排序选连续前缀，遇到第一句放不下即停止。没有跳过长句后再补短句。空选、未用预算与真实成本全部记录。 |',
        '| 句子分数 | 同时报告原始有符号归因总和与旧正值有效词元均值。前者保留负贡献抵消，后者是明确标注的敏感性视图；二者支付相同全部正文词元成本。 |',
        '| 标签与边界 | 480 份文档、2,031 个原生句子、124 条官方事实。保留第 8 例正文坐标修正及首个非空白字符归属。主结果不使用 v2 的两处语义标签替换或 45 例子集。 |', '',
        '标题和文档编号仍在模型输入中，作为已提供的文档元数据，不计入此处明确限定的**正文词元预算**。因此本预算不是包含元数据、序列化 ID 与全部提示内容的传输成本。三个跨原生边界的不可分 BPE 词元按首个非空白字符所在句归属一次并收费，属于原生句的 token 分配，不声称独立分词后的句子边界完全可分。', '',
        '以前的 `answer_only` 删除推理前缀，是曾明确声明的另一个答案重建实验；它的 68.58% 对 66.15% 不能替代这里的保留上下文实验。原始结果继续作为历史记录，不改写成 v3。', '',
        '## 预先固定的四项主比较', '',
        '官方支持事实 Recall，10% 全部正文词元预算，48 例宏平均。差值与区间单位为百分点。四项 DT−FT K3 配对比较采用 10,000 次 bootstrap、seed 73、Bonferroni 98.75% 描述区间。协议与 GPU 代码在计算新质量分数之前以提交 `258a745` 固定。', '',
        '| 目标 | 句子分数 | DT | FT K3 | DT−FT | 98.75% 配对区间 |',
        '| --- | --- | ---: | ---: | ---: | --- |']
    for target in TARGET:
        for pool in POOL:
            r=next(x for x in a['primary_comparisons'] if x['target_mode']==target and x['pooling']==pool)
            lines.append(f"| {TARGET[target]} | {POOL[pool]} | {pct(r['dt'])} | {pct(r['ft'])} | {sign(r['dt_minus_ft'])} | {ci(r)} |")
    lines+=['',
        '完整回答与答案跨度对应不同解释目标。原推理内容可能承接提示中证据的影响，答案只播种时的证据检索下降不能单凭本表判为实现错误；本表检验的是与官方支持事实的重合，不能单独区分推理中介、模型答案正确性及归因忠实性。', '',
        '## 同一 10% 预算的质量、实际成本与空选', '',
        '所有质量值为百分比；成本、句数为逐例均值。P/R/F1/EM 从 `(title, sentence_id)` 集合的 TP/FP/FN 计算。完整支持允许额外误检，EM 则要求集合完全相等。', '',
        '| 目标 | 句子分数 | 方法 | P | R | F1 | EM | 完整支持 | 全部正文词元 | 未用词元 | 句数 | 空选率 |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for target in TARGET:
        for pool in POOL:
            r=summary(target,pool,'all_body_tokens',.1)
            for method in METHOD:
                m=r['methods'][method]
                lines.append('| '+' | '.join([TARGET[target],POOL[pool],METHOD[method]]+[pct(m[k]) for k in ('precision','recall','f1','exact_match','complete_support')]+[f'{m[k]:.2f}' for k in ('spent_all_tokens','unused_all_tokens','selected_sentences')]+[pct(m['empty_selection'])])+' |')
    ceiling=summary('full','signed_sum','all_body_tokens',.1)['methods']['DT_target']['fact_recall_ceiling']
    lines+=['',f'同预算下，已知 gold 并任意选择其可支付子集的平均事实 Recall 上限为 **{pct(ceiling)}%**；该上限独立于方法排序，用来区分预算限制和排序失误，不是 v3 前缀策略本身保证能达到的上限。', '',
        '## 预算敏感性：5%、10%、20%', '',
        '| 目标 | 句子分数 | 正文预算 | DT Recall | FT K3 Recall | DT 平均收费 | FT K3 平均收费 |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for target in TARGET:
        for pool in POOL:
            for b in (.05,.1,.2):
                r=summary(target,pool,'all_body_tokens',b);d=r['methods']['DT_target'];f=r['methods']['FT_K3']
                lines.append(f"| {TARGET[target]} | {POOL[pool]} | {100*b:.0f}% | {pct(d['recall'])} | {pct(f['recall'])} | {d['spent_all_tokens']:.2f} | {f['spent_all_tokens']:.2f} |")
    lines+=['','## 固定句数：单独列出真实正文词元成本', '',
        'top 2/4/8 是句数限制，与 10% token 限制不同。全方法的 P/R/F1/EM 与费用见 CSV。', '',
        '| 目标 | 句子分数 | 句数 | DT Recall | FT K3 Recall | DT 平均收费 | FT K3 平均收费 |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for target in TARGET:
        for pool in POOL:
            for b in (2,4,8):
                r=summary(target,pool,'sentences',b);d=r['methods']['DT_target'];f=r['methods']['FT_K3']
                lines.append(f"| {TARGET[target]} | {POOL[pool]} | {b} | {pct(d['recall'])} | {pct(f['recall'])} | {d['spent_all_tokens']:.2f} | {f['spent_all_tokens']:.2f} |")
    lines+=['','## 隔离计费与选择策略的影响', '',
        '下表保持同一份归因向量、原生句排名及正值有效词元均值。第一行沿用旧有效词元分母和收费；第二行只换成全部正文词元分母和收费；第三行再改为排序前缀。每行都是其相应定义的 10%，不能把变化全部归因于修正一处代码。前两种跳过策略只作诊断。', '',
        '| 目标 | 计费与选择 | DT Recall | FT K3 Recall | DT 真实收费 | FT K3 真实收费 |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    for target in TARGET:
        for policy in POLICY:
            r=next(x for x in a['packing_summary'] if x['target_mode']==target and x['policy']==policy and x['fraction']==.1)
            d=r['methods']['DT_target'];f=r['methods']['FT_K3']
            lines.append(f"| {TARGET[target]} | {POLICY[policy]} | {pct(d['recall'])} | {pct(f['recall'])} | {d['spent_all_tokens']:.2f} | {f['spent_all_tokens']:.2f} |")
    lines+=['','所有旧完整回答的正值均值＋有效词元跳过选择在 5/10/20% 上精确复现，共 432 组。v3 的 1,152 次相邻 token 预算变化全部集合嵌套、Recall 不下降。旧规则的非嵌套问题与实际是否发生 Recall 下降不同；先前 v2 审计中 198 次非嵌套，但观察到的 Recall 下降次数为零。', '',
        '## 最小定位修正：保留旧候选与旧词元预算', '',
        '这是旧式 token Recall，允许按词元截断，旧候选还包含标题等元数据；不能把其绝对值与上方整句事实 Recall 直接比较。完整回答原始缓存坐标结果为 DT 40.63%、FT K3 45.35%；恢复正文坐标后如下。原正则分句结果为 73.81% 对 72.09%，同时修正文坐标与前导空格归属后的对照也如下。', '',
        '| 目标 | 定位视图 | DT | FT K3 | DT−FT |', '| --- | --- | ---: | ---: | ---: |']
    for target in TARGET:
        for view,label in [('raw_restored','原始词元排序＋恢复正文坐标'),('regex_anchor_restored','旧正则均值＋恢复坐标＋首词归属修正')]:
            r=next(x for x in a['legacy_summary'] if x['target_mode']==target and x['view']==view and x['fraction']==.1)
            lines.append(f"| {TARGET[target]} | {label} | {pct(r['methods']['DT_target']['recall'])} | {pct(r['methods']['FT_K3']['recall'])} | {sign(r['paired']['FT_K3']['recall']['dt_minus_ft'])} |")
    lines+=['','## 验证与复现', '',
        '- 48 例完整输入和 reference 保留；所有目标位置的原输入端点 logprob 与完整回答逐位一致；8 个开发对照共 24 份原始向量逐位一致。',
        '- 保存全部新 GPU 输出、分片哈希、运行前协议及代码提交，不删除任何失败样本或不利数值。',
        '- 3,456 组原生检索与评分，2,592 条计费/策略隔离记录，1,728 条旧词元诊断；独立验证器重算排序、选择集合、全部成本、支持事实指标和 gold 子集预算上限。',
        '- 八项回归测试覆盖目标不改输入、标点/空白收费、旧非嵌套反例、总和保留负值等；27 份冻结方法文件保持原字节哈希。详细回执见 [verification.json](verification.json)。', '',
        '| 文件 | 内容 |', '| --- | --- |',
        '| [主比较](primary_recall10.csv) / [全部预算汇总](native_summary.csv) | 4 项主比较、72 行全目标/池化/方法/预算汇总 |',
        '| [逐例评分](native_per_case.csv) / [检索集合](selections.json) | 全部候选排名、实际检索集合和质量/费用 |',
        '| [策略隔离汇总](packing_summary.csv) / [逐例](packing_per_case.csv) | 同向量只改变计费或跳过/前缀选择 |',
        '| [旧指标汇总](legacy_summary.csv) / [逐例](legacy_per_case.csv) | 原始与正则排序下的最小定位修正 |',
        '| [候选](candidates.json) / [官方标签](labels.json) / [源文审计](source_audit.json) | 候选生成与 gold 评分分离 |',
        '| [分析](analysis.json) / [分数诊断](score_diagnostics.csv) | 全部区间、代码/原始记录哈希、负贡献统计 |',
        '| [完整输入身份](full_input_identity.json) / [运行回执](execution_receipt.json) | 与原完整回答逐例绑定、运行前提交及归档哈希 |',
        '| [GPU 原始归档清单](raw/conditioned_v3/manifest.json) | 7 份无损 gzip 文件的原始与压缩字节哈希 |',
        '| [运行前协议](PROTOCOL.md) / [机器协议](protocol.json) | 固定目标、计费、排序和四项比较 |', '',
        '在仓库根目录、安装 numpy 与 tokenizers 后执行：', '', '```text',
        'python research/temporary/hotpot_context_v3_20260910/prepare_candidates.py --source ../audit/primary_sources/hotpot_dev_distractor_v1.json --cache ../audit/published_flashtrace/table1-data-v1/extracted/data/hotpotqa_long.jsonl --tokenizer ../audit/deltatrace-figures-20260909/qwen3-tokenizer.json',
        'python research/temporary/hotpot_context_v3_20260910/analyze_v3.py --cache ../audit/published_flashtrace/table1-data-v1/extracted/data/hotpotqa_long.jsonl --tokenizer ../audit/deltatrace-figures-20260909/qwen3-tokenizer.json --publication ../DeltaTrace-paper-qwen3/experiments/official/results/qwen3_8b_table1_20260909',
        'python research/temporary/hotpot_context_v3_20260910/build_report.py',
        'python research/temporary/hotpot_context_v3_20260910/verify_outputs.py', '```', '',
        'GPU 重跑使用 `run_conditioned.py --environment /path/environment.json --control /path/restored-answer-dev-control --output /path/new-run`，先将 [旧对照归档](../source_v2_gpu_20260910/raw/answer_dev_corrected_v1/manifest.json) 恢复到新目录。环境路径须指向原模型、缓存与冻结代码；驱动会验证权重身份及全部输入身份。输出必须是不存在的新目录。新结果用冻结的 [归档程序](../hotpot_fairness_20260910/archive_answer.py) 打包，`import_archive.py` 验证传回的完整 ZIP 哈希及每个文件后导入。', '',
        '历史追溯：[v2 方法学复审](../hotpot_fairness_20260910/REVIEW.md)、[v2 原实验](../hotpot_fairness_20260910/RESULTS.md)、[原完整 448 例基准](../source_v2_gpu_20260910/full_recall/RESULTS.md)。这些结果、原归因方法和 VT 数值均保留。', '']
    (HERE/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(dict(status='report_written',native_summary_rows=len(native),primary_comparisons=len(primary))))

if __name__=='__main__':main()

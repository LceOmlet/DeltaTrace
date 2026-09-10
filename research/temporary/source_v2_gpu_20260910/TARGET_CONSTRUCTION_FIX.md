# 三链 VT：目标构造错误与已验证的修复

**是，48.11% 对 74.67% 那一轮存在解释目标与检索目的不一致的问题。** gold 正确地只覆盖被问到的那条链；实际归因目标却包含讨论三条链的完整生成响应。

旧入口 `evaluate_recall.py` 先保留作者缓存中的完整 `ex.target`，再将它传给 `_ensure_generation(ex.prompt, ex.target)`，并将 `ex.target + EOS` 构造成目标 token。DT 的有限归因调用没有最终答案权重掩码，因此解释完整目标。`indices_to_explain` 的存在不等于它已限制 DT 的目标种子。FT 原 Both 实现也从完整非停止生成 token 聚合初始归因；这不是“DT 解释全文而 FT 只解释答案”的不公平对照。

修复在新版 `evaluate_target_scope.py` 中明确完成：

1. 仅使用缓存中**模型生成的 target**、既有答案 token span 与 tokenizer offsets 截出最终答案。提取过程不读取 gold 或参考答案。
2. VT 先把 `ex.target` 换成该答案子串，再构造实际模型输入。原 prompt 保持不变，先前生成的推理文本不再进入输入。
3. DT 与 FT 在这个相同输入上解释相同的非停止、非 EOS 答案 token。运行时逐次检查 FT 的真实输入、初始权重与完整可用多跳范围。
4. gold、正文候选集和双方的 `ceil(0.10*N_source)` token 预算保持原定义；同时报告原始 token 排序与相同的句均值聚合。

仅改变最终答案种子、仍保留已生成推理的控制在开发集上未能解决问题（DT 67.72%、FT 100%）。移除先前推理后双方句聚合才达到 100%。这区分了“输出种子选在哪里”和“计算目标时模型还条件于哪些文本”。修复改变了解释问题，不代表模型推理能力提高。

冻结策略后，另取 80 例内部留出验证，四组 VT 和 HotpotQA 各 16 例。三链 VT 原始 token Recall 为 **84.85% 对 71.27%**，句均值聚合为 **100% 对 100%**。四组 VT 原始排序的等权差值 **+4.09 点**，四项预定比较校正后的区间为 **[+2.78, +5.56]**。句聚合后的 VT 全部达到各自预算上限，属于持平。详见 [完整结果](target_scope/RESULTS.md)。

HotpotQA 使用运行前固定的完整回答目标，DT/FT 输入和目标仍相同。句聚合为 75.94% 对 69.22%，差值区间 [−4.55, +17.92] 点，尚未证实稳定优势。这一任务设置不能推广成统一目标下所有任务都胜出。

旧 `evaluate_recall.py`、旧 `source-v2` 协议及失败结果保留为实验快照。复现修正后的这批比较应使用下面的新版入口，而不是用旧入口的完整响应结果代替最终答案检索结果。

```bash
python research/temporary/source_v2_gpu_20260910/evaluate_target_scope.py \
  --environment /path/environment.json --stage validation \
  --choice research/temporary/source_v2_gpu_20260910/scope_choice.json \
  --datasets vt_h2_c3 vt_h4_c1 vt_h6_c1 vt_h10_c1 hotpotqa_long \
  --output /path/new-target-scope-run

python research/temporary/source_v2_gpu_20260910/analyze_target_scope.py \
  --run /path/new-target-scope-run --publication /path/original-publication \
  --data /path/original-data --tokenizer /path/qwen3-tokenizer.json \
  --output /path/new-target-scope-analysis
```

输入配置与原始依赖沿用 `experiments/official/environment.example.json`。新版入口按固定 split 重跑这 80 例，并核对模型、原作者实现与归因代码哈希；正式方法的 27 项冻结依赖没有修改。

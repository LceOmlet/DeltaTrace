# 独立加速实现

Qwen3新增显式 [`graphed_qwen3.make_graphed_qwen3`](graphed_qwen3.py) 入口，逐次执行真实B2根前向，再刷新全部输入并重放原有限GPU程序。固定短输入B1四档的完整热调用均快于FT Both和原FT multi-hop，两轮均值及冻结的95% bootstrap门槛通过。完整向量、数值诊断和11条原始短样本一致；显存峰值20.879–40.019GB，空编译缓存第一调用31.794秒。见[API与成本](GRAPHED_QWEN3.md)、[全部重复、失败候选和适用范围](../../research/temporary/qwen3_short_efficiency_20260910/README.md)。

短输入 B1 新增显式 Qwen3.5 入口 `code_local_qwen35.make_code_local_qwen35(root, model, finite_fa, finite_fla)`，返回与 retained 入口相同接口的 runner 和来源记录。它用 CPython 3.12+ 的局部事件接口减少被动捕获开销；原生计算、有限规则及完整诊断保持。实际总长度157/264/477的三档完整延迟降低35.5%/37.2%/20.2%，904-token档增加0.47%、无已证实收益；四档预热显存相同。4个合成格和11个原始短样本的完整向量/逐层诊断一致。实现要求单次使用捕获上下文、可用monitoring tool ID及无已有 `sys.setprofile` profiler。见[本次核验与限制](../../research/temporary/cause_tolerance_20260909/CODE_LOCAL_CAPTURE.md)。Qwen3未采用本次试验候选，原正式评测默认入口保持。

固定基线仍在`deltatrace/clean`，原文件逐字节保留。这里的`qwen35/controller.py`只扩展DT控制器的检查点存储与原生变长batch接线；有限传播公式全部导入固定基线。`qwen35/dynamic_finite.py`只把同一有限函数交给官方`torch.compile(dynamic=True, fullgraph=True)`。不改写FT、模型、原生注意力或原生反向，也没有编译失败回退。

来源与文件哈希见[sources.json](sources.json)。两例成本测试和16例原协议质量回归已完成，原结果见[临时研究记录](../../research/temporary/acceleration_20260909/README.md)。固定原表评测入口仍默认调用干净基线，不会自动换到这里。

已支持CPU/GPU检查点、原生FA变长路径、连续右padding和每例独立目标。正式入口已提供显式`--dt-backend accelerated_qwen35 --sample-batch 2`，实际完成原16例及有符号RISE/正值MAS核验；默认仍是干净B1。额外显存保持随batch、序列和层数线性增长，未引入全局序列平方矩阵。默认FA/FLA精度与原有限规则保持；动态编译和native batch可能产生数值及排序差异。

相同16例预热后，完整归因16.684秒降至12.331秒，峰值21.644GB增至23.882GB。当前厂商持久化自动调优缓存在原生伴随中遇到过长文件名错误，完整DT验收明确关闭该开关；没有修改框架。新入口的RISE相对干净基线有8例退步、MAS有4例退步、needle无逐例退步，全部保留，不作为新方法质量提升宣传。

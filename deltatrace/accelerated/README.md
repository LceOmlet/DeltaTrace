# 独立加速实现

固定基线仍在`deltatrace/clean`，原文件逐字节保留。这里的`qwen35/controller.py`只扩展DT控制器的检查点存储与原生变长batch接线；有限传播公式全部导入固定基线。`qwen35/dynamic_finite.py`只把同一有限函数交给官方`torch.compile(dynamic=True, fullgraph=True)`。不改写FT、模型、原生注意力或原生反向，也没有编译失败回退。

来源与文件哈希见[sources.json](sources.json)。两例成本测试和16例原协议质量回归已完成，原结果见[临时研究记录](../../research/temporary/acceleration_20260909/README.md)。固定原表评测入口仍默认调用干净基线，不会自动换到这里。

已支持CPU/GPU检查点、原生FA变长路径、连续右padding和每例独立目标。正式入口已提供显式`--dt-backend accelerated_qwen35 --sample-batch 2`，实际完成原16例及有符号RISE/正值MAS核验；默认仍是干净B1。额外显存保持随batch、序列和层数线性增长，未引入全局序列平方矩阵。默认FA/FLA精度与原有限规则保持；动态编译和native batch可能产生数值及排序差异。

相同16例预热后，完整归因16.684秒降至12.331秒，峰值21.644GB增至23.882GB。当前厂商持久化自动调优缓存在原生伴随中遇到过长文件名错误，完整DT验收明确关闭该开关；没有修改框架。新入口的RISE相对干净基线有8例退步、MAS有4例退步、needle无逐例退步，全部保留，不作为新方法质量提升宣传。

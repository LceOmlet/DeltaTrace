# 独立加速实现

固定基线仍在`deltatrace/clean`，原文件逐字节保留。这里的`qwen35/controller.py`只扩展DT控制器的检查点存储与原生变长batch接线；有限传播公式全部导入固定基线。`qwen35/dynamic_finite.py`只把同一有限函数交给官方`torch.compile(dynamic=True, fullgraph=True)`。不改写FT、模型、原生注意力或原生反向，也没有编译失败回退。

来源与文件哈希见[sources.json](sources.json)。两例成本测试和16例原协议质量回归已完成，原结果见[临时研究记录](../../research/temporary/acceleration_20260909/README.md)。固定原表评测入口仍默认调用干净基线，不会自动换到这里。

已支持CPU/GPU检查点、原生FA变长路径、连续右padding和每例独立目标。当前仅验证真实B2、固定16例；正式入口的batch参数尚未集成。额外显存保持随batch、序列和层数线性增长，未引入全局序列平方矩阵。默认FA/FLA精度与原有限规则保持；动态编译和native batch可能产生数值及排序差异。

# Qwen3 原生有限传播图入口

`graphed_qwen3.make_graphed_qwen3(root, model, library, library_sha256)` 返回显式 B1/E2 DT 控制器和来源记录。固定短输入协议的四档完整热调用均快于两版固定 FT，见[实测、全部重复和限制](../../research/temporary/qwen3_short_efficiency_20260910/README.md)。

```python
from pathlib import Path
import sys

root = Path("/path/to/DeltaTrace").resolve()
sys.path.insert(0, str(root / "deltatrace/accelerated"))
from graphed_qwen3 import make_graphed_qwen3

model.eval().requires_grad_(False)
model.set_attn_implementation("flash_attention_2")
runner, receipt = make_graphed_qwen3(
    root, model, finite_library_path, finite_library_sha256
)
try:
    result = runner.attribute(before_ids, after_ids, attention_mask, prompt_len)
    signed = result["signed_full_sequence"]
finally:
    runner.close()
```

使用与固定 clean Qwen3 相同的 FP16 模型、公开 FA、有限 FA 库及其 SHA-256。两个输入均为 `[1, N]`；`before_ids` 按原方法将可归因位置替换为 EOS，生成目标保持相同；attention mask 全为 1。返回整个序列的有符号向量、原目标差、逐层诊断及未分配残差。`mutation_audit=True` 额外验证 720 个实际根张量快照和 883 个图输入视图，成本在调用内计入。

每次调用都执行一次原模型 B2 根前向，将新根张量写入图输入，然后重放全部有限传播计算。图保留原生 GEMM、36 次额外公开 FA 和原有限 FA；没有注意力矩阵的全局平方存储。每个根张量视图仍参与计算和检查，共享存储只拷贝一次。829 个布尔检查包含原种子检查，另保留最终有符号向量有限性检查；所有检查在 API 返回前完成。

控制器缓存一个输入形状、prompt 长度和目标 logits 形状对应的图。首次调用包含两次有限程序预热、一次图记录和一次真实重放；形状改变会替换旧图。每次输入内容都更新，没有归因结果缓存。用固定的 eval 模型复用控制器；模型结构或配置改变后重新构造。控制器不可重入，`close()` 释放图和静态输入。

实测 C550 上完整序列 158/265/478/905 的预热峰值 allocated 显存为 20.879/23.611/28.963/40.019 GB，FT 为 16.919/17.118/17.666/19.437 GB。完全空编译缓存的第一调用为 31.794 秒，随后换形状首次调用约 2.8–3.6 秒；新进程首次构造 905-token 图为 10.565 秒。这些成本保存在原始记录中。高频更换形状时应据实计入构图时间。

捕获使用已验证的 CPython 3.12+ `sys.monitoring` 局部事件，需要一个可用 monitoring tool ID，且不能已有 `sys.setprofile` profiler。构造函数检查[图入口哈希](graphed_qwen3_sources.json)、根捕获哈希、retained/deferred 基线及实际导入位置。验证范围为固定的 Qwen3-8B 环境与输入协议；11 条原始短样本另外通过了完整向量和逐层数值一致性检查，未据此作额外速度或质量指标声明。

保留的 `root_retained_qwen3.make_root_retained_qwen3(...)` 入口只复用实际根张量、执行有限传播并集中处理严格检查，不缓存图。它是本次图入口的构成与核验基础，早期独立短档速度验收未通过；不将其描述为四档都快于 FT。

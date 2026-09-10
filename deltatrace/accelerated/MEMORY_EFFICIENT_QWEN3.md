# Qwen3 完整图与显存复用入口

显式工厂 [`make_memory_efficient_qwen3`](memory_efficient_qwen3.py) 为固定的 Qwen3-8B FP16、eval、FA2 模型提供 B1 样本、B2 端点归因。每次调用刷新真实输入，执行原模型根前向和完整有限传播，返回完整有符号向量及所有原数学诊断。速度、显存和冷启动的适用范围以[本次研究报告](../../research/temporary/qwen3_stream_memory_20260910/README.md)的实际结果为准。

```python
from pathlib import Path
import sys

root = Path('/path/to/DeltaTrace')
sys.path.insert(0, str(root / 'deltatrace/accelerated'))
from memory_efficient_qwen3 import make_memory_efficient_qwen3

model.eval().requires_grad_(False)
model.set_attn_implementation('flash_attention_2')
runner, sources = make_memory_efficient_qwen3(
    root, model, '/path/to/libdeltatrace_fa_finite_shared_mean_reuse.so',
    '28b99bea512211c33887ac773ced6a5b68b058b3f3b30638bd7128183b4b2a86',
)
try:
    result = runner.attribute(baseline_ids, input_ids, attention_mask, prompt_len)
    signed_scores = result['signed_full_sequence']
finally:
    runner.close()
```

输入为同设备的 `[1, N]` token IDs 和合法 mask；基线只对可归因输入位置替换 EOS，完整响应及目标边界与原实验一致。`mutation_audit=True` 增加真实输入快照审计，其额外成本应单列。每次正常调用仍执行原有状态检查、829 个严格谓词、36 个逐层统计、完整有符号向量有限性检查；启用原生投影复用时另执行 36 个同输入谓词。返回的嵌套元数据容器每次重新创建，调用方修改上次返回不影响下次结果。

一个 runner 只保留当前几何形状的一张图。形状改变要重新构图，训练状态、配置、参数身份/版本、模型方法或 hook 等契约变化会被拒绝。参数更新后应创建新 runner。该入口针对固定 eval 模型，不支持并发或重入调用；显式 `close()` 释放控制器图及引用，模型权重和框架自身分配器缓存仍由调用方管理。

## 计算和存储

根前向调用原生完整 Qwen3 模型、全词表 head 和默认 FA2。有限 seed 的完整形状、FP32 归一化边界及原生 GEMM 形状保持。目标 log-prob 使用原生算子分块计算，沿用原始目标求和顺序；已完成样本以完整向量和全部数学诊断的精确相等为准。

生命周期管理按层释放有限传播不再需要的引用；同一次真实根前向与 decoder 重算通过未改动的公开 PyTorch SAC 复用部分原生投影输出。选择依据是完整序列长度：≤300 不复用，301–700 复用 down 投影，>700 复用 q/k/v/o/down，其中 q 只保存前16层。它不在两次归因之间保存或复用激活值、分数或归因结果。

独立的有限 FA 扩展复用固定厂商 FA2.5.3 框架的 MMA 和数据布局。phase0 维持32×32/2warp，phase1/2 为64×32/4warp；三个阶段均将所属输入 tile 留在原生 MMA 寄存器布局中，调用原厂商 `flash::gemm<true,false>`。完成载入后，A 的共享存储供 B 复用，三个阶段共享存储分别为8192/16384/16384字节。原 K 循环、列累加顺序、半精度操作数与有限数学规则保持。此扩展不是模型安装的 FA2.6.3，也没有替换模型原生注意力或反向。源文件、动态库哈希和依赖均由[版本清单](memory_efficient_qwen3_sources.json)固定，构建证据在研究目录的 `raw/evidence_fa_mma_owner_v2.zip`。

## 完整成本

第一调用包含两次原模型/有限程序预热、一次图记录、一次图重放及编译。后续每次调用仍包含输入复制、完整 GPU 图执行、原检查、当前分数和完整向量传回 CPU，以及返回对象处理；热调用不能代表频繁切换形状或一次性使用的总成本。

GPU 图内存池会使热调用 allocated 计数低于实际保留容量。因此容量比较同时报告 reserved、冷调用 allocated，以及模型加载与初始化峰值；只看热调用 allocated 不足以证明显存优势。主机记录包含调用后的当前 RSS 和进程生命周期高水位。构图中的历史分配器快照不是逐次重放峰值，主机子阶段时间也可能重叠排队中的 GPU 工作，不能重复相加。

本轮对 DT 和两种 FT 统一使用作者原加载器先载入 CPU，再执行原生 `model.to('cuda:0')`。全部400个参数与 buffer 的数值、形状、步长、精度及设备已与直接 GPU 加载逐字节核对；模型计算代码不变。该路径降低加载期间的 GPU 临时峰值，但主机 RSS 高水位约28GB，须计入容量规划。完整加载成本保存在每个进程的原始记录中。

固定四档短输入的独立几何容量验证，不代表任意长 rollout、形状收缩过程或其他设备的显存保证。长 rollout 的失败/OOM及实际成本保存在同一曲线数据中。

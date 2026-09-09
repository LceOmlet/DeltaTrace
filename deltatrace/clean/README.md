# 干净 DeltaTrace：clean-v1-20260909

这个目录只保存方法代码，不包含实验循环、评分器或临时诊断。

| 模型 | 固定入口 | 固定规则 |
|---|---|---|
| Qwen3-8B | `qwen3/qwen_signed_secant_paired_vendor_fa.py:propagate_paired_secant` | `pv_rule='content_P1'`，原FP16有限传播与可追溯FA扩展 |
| Qwen3.5-9B | `qwen35/qwen35_clean_runner.py:make_qwen35_clean_runner` | 四个逐层规则映射均为空，原content1/P1，BF16默认FA/FLA |

所有27个依赖文件从已保存的DT源码逐字节固定，来源与SHA256在[sources.json](sources.json)。没有复制或改写模型、原生注意力、原生反向或FT评分器。有限传播是明确命名的归因扩展；不冒充原生反向。两个模型在各自依赖环境、各自进程中运行。

Qwen3.5每次归因仍为8次有限FA、24次有限FLA和48个原生FLA伴随阶段；不包含第0层端点平均、逐层norm-gate、PV19或K修补。保留带符号输出；正值视图在[正式评测入口](../../experiments/official/evaluate.py)生成。

编译库由运行环境提供，必须匹配环境清单中的哈希。构建来源保留在仓库原[FA扩展说明](../../docs/fa_two_endpoint_extension.md)及`research/reproduction_templates/`的已归档构建记录；归因入口不静默回退到另一套注意力。

该目录固定后，加速工作从独立分支或独立版本开展。修改方法或精度须生成新版本，不覆盖论文与实验已引用的版本。

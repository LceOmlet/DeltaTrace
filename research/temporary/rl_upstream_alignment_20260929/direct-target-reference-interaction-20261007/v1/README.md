# 真实极端 token：参照上下文交互诊断

这是实际测量记录，不是另一份方法计划。信用定义仍以 `experiments/rl/PLAN.md` 为准。

AppWorld 原生批次 rank1/batch6 的 row0，UID
`b98de8ee-995e-49e7-acb8-1510b4433a6a`，response4498/packed7260，token198
是 `Code:` 后、代码围栏前的换行。原正式联合 DT 给它 d=-3.11176191，
奖励1时对应样本优势约 -21.46058；AppWorld 未完成该轮 QVA，因此不是已消费的更新。

使用相同的原 VERL actor、fresh LoRA8/16、真实 B4、HF/PEFT/FA/FLA、原目标选择器，
补测“所有原 prior-source 为 EOS，仅恢复这个 token”的两个原生端点。正式轨迹、
target 身份和其他三行不变。这次每卡仅一次成对原生前向；DT、采样、反向、
optimizer、恢复检查点均为零。PID3902140/birth1791380517.62 已完成退出。

| 上下文 | 保留该 token 的联合 logp | 删除该 token 的联合 logp | 保留减删除 |
| --- | ---: | ---: | ---: |
| 真实上下文，前次实际端点 | -54.19597564 | -75.02560897 | +20.82963333 |
| 其余 prior-source 全为 EOS，本次端点 | -159.24135409 | -153.91202829 | -5.32932580 |

两种上下文的边际影响相差26.15895913。紧邻代码围栏 target52451 的变化最明显：
真实上下文为 +21.50193501，EOS 上下文为 -3.26080132。
这直接证实上下文交互足以反转符号。联合参考分解中的负贡献，不能据此解释为
真实轨迹中该 token 的单独删除效应；也不能把这两种边际中的任何一个直接等同于
原 DT 联合有限分配的 -3.11。尚未把全部有限分配误差归结为一个原因。

两 rank 分数相同，其他三行差值0，更早68个 target 差值0；实际 LoRA_B 分片全0。
原 BF16 logits、FP32 目标 logp、FP64 求和均保留。这不是 FA/FLA 容差验收；
完整原生前向相对旧缓存事实分数的 -.28363 漂移仍独立保留。单例对照也不替代
累计删除曲线、RISE、MAS，不据此判断 DT 的整体证据恢复能力。

`actual-results/transport.json` 保存逐文件 SHA256；`analysis.json` 绑定之前的真实
F/D 端点和本次 B/C 端点。CPU 测试使用保存的真实 B4 和原 owner 定义，只核查
输入运输、原 target 保留和身份；不能代替模型计算。测试与运行状态另存回执。

没有更改 DT 参照、Q/V/A、白化、PPO、奖励、环境、LoRA 或 microbatch。
TextCraft 首次更新继续 hold；AppWorld 正式任务仍终止。
显存生命周期修复和失败 B4 共存验证见前一份
`direct-target-native-mlp-memory-20261007/v2/candidate-analysis-v6/analysis.json`，
它不构成信用已修复或精确32768已验收的证明。

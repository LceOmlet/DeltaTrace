# 基于 FA 框架扩展双端点有限传播

状态：**尚未实现和验收生产 FA 后端扩展**。当前已保存的是显式整网方法和独立 Triton 分块原型。旧原型采用类似 key-block 的调度，并不等于修改了实际厂商 FA 的标准 backward。

后续用户明确要求最小适配在架构上不依赖 FA 版本，并再次强调效率和批处理。当前执行方向以[后端契约](backend_contract.md)为准：默认已安装 FA 和稳定公开接口优先；下面的源码扩展讨论保留作研究记录，不再把移植旧版内核作为默认路线。

用户优先级：真实性优先；小幅耗时增加可以如实记录，不能成为构造或切换影子实现的理由。当前 PV 实验的旧冻结时间门槛仍如实报告，但不把它当作禁止继续实现真实 FA 扩展的门槛。

## 为什么不能直接调用普通 FA backward

每行令 `L = logmean(P0, P1)`，`T = M_out @ V_route.T`，`c = sum(L*T)/sum(L)`。有限 softmax 乘子为 `M_score = L*(T-c)`。标准反向使用 `P*(dP-dO·O)`；这里的 L、中心化 c 和 V 分支概率一般均不同，不能更名后当作同一算子。

|PV 规则|V_route|V 分支概率|
|---|---|---|
|对称|mean(V0,V1)|mean(P0,P1)|
|内容 P1|V0|P1|
|内容 P0|V1|P0|

Q/K 分支仍采用原对称有限乘积：`M_Q = M_score @ mean(K0,K1)*scale`，`M_K = M_score.T @ mean(Q0,Q1)*scale`；`M_V = P_value.T @ M_out`。不能把 V 分支概率偷换成 L。

## 扩展边界

1. 明确实际运行 FA 二进制、接口、厂商源码对应版本和许可证。安装包中的实验 Triton 文件不等于当前生产 `.so` 的源码。
2. 从可追溯源码保留分块访存、因果 mask、stride、GQA 与梯度归并框架；用独立命名的有限传播入口修改必要公式。保留上游文件与最小补丁，记录源码差异；不覆盖模型前向或标准反向。
3. 第一遍计算有限中心化所需的行统计，随后复用分块 Q/K/V 传播框架；额外遍历、布局转换、同步与缓冲区全部计费。是否能减少遍历必须先证明，不能把普通 `dO·O` 偷换成 c。
4. 使用用户允许的 FA 默认精度策略，准确记录操作数和累加类型。允许非逐位相同；保留误差、符号和目标残差，禁止偷偷裁掉负数或硬归一化总残差。
5. 在原模型实际捕获操作数上核对完整输出与实际执行内核，再接入同一整网方法跑原 FlashTrace 基准。局部原型不是新质量基准，局部耗时也不等于完整方法耗时。

截至本仓库建立时，已直接检查官方 v2.6.3 backward 入口以及实际安装的 `flash_attn_triton.py`。后者自述为旧版实验实现，并使用旧 Triton API；当前生产后端是厂商编译的 FA 二进制。尚未确认其完整可构建源码对应关系，因此不能声称当前已有“官方生产框架扩展”。

上游参考：[FA backward kernel](https://github.com/Dao-AILab/flash-attention/blob/v2.6.3/csrc/flash_attn/src/flash_bwd_kernel.h)、[FA backward launch](https://github.com/Dao-AILab/flash-attention/blob/v2.6.3/csrc/flash_attn/src/flash_bwd_launch_template.h)。

## 已有失败原型为什么保留

历史 v5 在三个实际层输入上约12.17—12.24ms，显式参考约7.10—8.17ms；局部额外峰值约59.49MB对880.04MB。它证明了一个局部分块表达的内存收益，并未完成整网集成或生产 FA 扩展。该结果保留在 `evidence/finite_attention_fusion_summary_20260906.json`，后续不因较慢而删除其代码，也不把该失败当作换用影子实现的理由。

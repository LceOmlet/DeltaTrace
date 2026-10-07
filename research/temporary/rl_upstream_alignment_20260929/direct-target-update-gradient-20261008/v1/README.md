# 实际 actor 位置与白化核对（只读）

本目录复用当前冻结 TextCraft 首次 DT 后、首次 update 前的两份真实保存张量，
没有生成新轨迹、重启训练、运行反向、恢复检查点或改写信用。
`actual-actor-mapping.json` 保存原 VERL `masked_whiten` 的实际导入路径、SHA 和源代码。

12 个先前做过 native 单 EOS 端点对照的位置全部通过 traj_uid 和
retained_response_positions 找到，原 token IDs 相同，全部在有效 actor mask 中。
` Format` 的 token ID 是14606、回复位置是351；没有因10240训练截断消失。
其原始 A=-22.816926956，原 actor 系数=-140.340957642。
整批已保存 actor advantages 与原 masked_whiten 一次计算逐值相同；
response_mask 与原 loss_mask 的 response 切片逐值相同。
这排除了该样本的映射/白化实现错误；不能因此宣布 DT 估计正确。

713539 个真实有效 token，原 A 均值0.039150731、最小-22.816927、最大1；
原 actor 系数最小-140.340958、最大5.899810，全都有限。
这些是系数统计，不是参数梯度占比，不采用自定接受容差。
12 点是先前选出的偏向高影响样本，不声称其反号比例是全局发生率。
原生端点、作者累计删除/RISE/MAS、算子 dtype 容差和存储回归分别保留各自回执。

CPU 分析结束时 CUDA 未初始化，进程 RSS 约9.00 GiB；没有新增 GPU 作业。
TextCraft 两 rank release 文件仍不存在，AppWorld 正式训练仍未重启。

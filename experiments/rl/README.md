# DeltaTrace RL

唯一方法规范是 [PLAN.md](PLAN.md)。环境复用见
[REMOTE_ENVIRONMENT.md](REMOTE_ENVIRONMENT.md)；本页只记录实现与测试状态。

**2026-09-23 06:41 更新：正式 v3 三组作业均未完成一次更新，随后发生主机
OOM，现已停止失败作业。** WebShop 完成约 6.5 小时 rollout 后，在向 DT
传递轨迹的 Ray RPC 边界，TaskRunner 内存达到 373 GiB，容器 883/900 GiB，
导致三个 policy worker 被 Ray 杀掉。先前“有 GPU 活动、尚无 OOM”的采样不能
证明训练健康。已修复逐行 tensor 视图重复序列化整个 batch storage；
跳过已结束轨迹的生成改动已与同输入形状的原 HF owner 逐 token 对齐。
真实 32k 夹具中，4 条生成预热后 78.745 秒、2 条有效输入为 68.011 秒；
全无效微批次为 0.020 秒。生成阶段的整轮吞吐仍未验收，不能自动恢复正式规模。
后续先查现有进程栈和阶段耗时，再对已定位问题测试。实际恢复以新回执为准，
不能沿用 v3 启动状态。
四轨迹 pilot 的成功记录仍有效，但没有覆盖这个正式规模传输缺陷。
诊断、规模传输峰值和当前修复验证见
[results_rollout_failure_fix.json](results_rollout_failure_fix.json)。

2026-09-23 已通过 `ROLLOUT_BACKEND=vllm` 接入固定 VERL 的原 rollout、LoRA
同步和 sleep/wake；没有独立服务 client 或第二套权重同步。原生 vLLM 试跑中，
Sokoban 两轮均有非零优势/更新，成功率均为 100%；WebShop、AppWorld 分别完成 1、3 次零奖励
更新后，在首次非零奖励 DT 处遇到 FSDP 根模块初始化错误。三任务连续训练
尚未通过，正式规模未恢复，定时检查和备份保持暂停。
LoRA API 兼容回补来自 VERL v0.7.0；vLLM 0.15 的 weights pool 上下文遗漏
回补官方 v0.17.0 修复后，9B 休眠物理占用从 19.056 降到 2.285 GiB。
真实 native loader 已绑定 200 层 LoRA。相同 32256+512、batch4 的生成
预热后为 67.189 秒（包含官方同步与休眠；并发 CPU 负载不完全相同），
修复前同接口为 64.447 秒，输出相同。vLLM 完整容量夹具已完成：每条 DT
恰好 32768 tokens（含 1024 action tokens 和读出目标），四次串行 DT、两次
非零原 PPO 更新及更新后的 LoRA 同步完成，总计 851.49 秒。这不代表 DT
同时 batch=4 已通过。证据见
[results_vllm_integration.json](results_vllm_integration.json)。

最新 `origin/main=6ca8dc07` 的 MetaX 路径已核对并在使用：模型前向走已安装
FA 2.6.3 / FLA 0.4.1；DT 有限传播复用 main 的 MetaX FA 扩展和 FLA 原反向
子内核。32k、batch=4 的 FA 算子诊断中，原有限传播 49.03 秒，原生 FA
反向 1.008 秒。优化候选为 31.07 秒，尚未达到效率要求；不能把这个算子
比值当作完整 DT/完整模型反向比值。
候选在两端重合时通过原 FA 测试（长度 128/447），非零端点与原有限内核
另做数值对照。DT minibatch 接口的同批次复制/复用张量结果相同，但串行/
批量仍有归因差异，真实 32k×4 候选仍在重算阶段 OOM。因此没有切换正式
训练到这些候选。记录见 [results_dt_minibatch_candidate.json](results_dt_minibatch_candidate.json)。

FSDP 修复恢复官方 VERL 对完整 PEFT actor 的包裹位置，DT 注册在同一根
模块的公共 forward 接口。已验证先 actor 后 DT、先 DT 后 actor 的两种
顺序，并在真实 9B 上复现先 actor 后非零 DT。原生 vLLM 仍绑定 200 个 LoRA
层、无未绑定键，sleep 后占 2.285 GiB；这不代替三任务重新试跑。

此前 HF 参照为 78.745 秒处理四条
32256-token prompt、各生成 512 tokens，约 26.0 输出 tokens/s（包含 prefill）；
这个绝对耗时本身不是性能缺陷。后端比较须使用同卡、同权重/LoRA、原始输入
IDs、精度、生成配置和缓存条件，分别报告首轮与预热耗时；原生后端与接入后
的额外开销须单独比较。只修复已定位的差异；收益未超出重复测量波动时，不
继续凭感觉调参。整轮耗时还须分别核算环境、生成、逐事件 DT、PPO 和传输。
原生生成的调度、缓存和内核由 vLLM 拥有，既定 DT/QVA 和原 PPO 保持不变。

## 当前实现：EOS DT → 奖励事件 → token PPO

2026-09-22 已移除额外参考 token 采样和逐 token 前后奖励询问。
`reward_readout.py` 现在调用正式 DT `attribute`，对每个 response / 非零未来
奖励事件返回各 token 的 signed 归因，再由 `counterfactual.py` 分事件完成
`r * (-expm1(-d))` 和 Q/V 组合。没有复制有限传播、环境评分或 PPO。

原始 token IDs、奖励事件身份及 mask 由固定 VERL collector 提供。
O/padding 不参与 actor loss；实际 EOS action 保留 action 身份，其 EOS 替换
对比可以为零。各事件分别变换后相加，不广播 span，不归一化 credit。

归因阶段通过 HF 公共接口临时切换到 FA；正常和异常退出都恢复原 actor
后端。当前按用户指令只使用 MetaX，复用其已记录的动态 shape 执行配置和持久缓存。启动脚本按实际
tokenizer 为事件询问及 target 预留空间，总上限仍为 32768，不截断已生成 action。

## 当前验证范围

- 三任务原生训练均已完成连续迭代，且各有真实非零奖励、DT token 优势与原 PPO
  非零更新。Sokoban 两轮成功率 100%/75%；WebShop 两轮 0%/25%；AppWorld
  seed=1 四轮 25%/0%/0%/75%。AppWorld 第四轮 44 次 DT、7251 个非零 token
  优势、grad_norm=0.199。这些每轮仅四条轨迹，是接线/连续训练验证，不能作为
  测试集性能。见 [results_native_training_metax.json](results_native_training_metax.json)。
- 32k 主动容量检查：DT 两端各 32768 有效 token、1024 个 action token、
  minibatch=4，四次正式 DT 与两次非零原 PPO 更新均完成。allocated 60.596 GiB，
  reserved 62.250 GiB（reshard=False），总耗时 743.78 秒；32769 被明确拒绝。
  合成容量夹具不是自然任务长度。见 [results_dt_context_capacity.json](results_dt_context_capacity.json)。
- 显存修复复用 FSDP2 公共分层释放接口；同一 16k 输入的完整 DT/QVA 零误差一致，
  allocated 54.187→39.806 GiB，预热后 40.521→40.698 秒（+0.44%）。原 HF 32k
  batch4 生成在 reshard=False 下预热后 77.174 秒，比 True 快 17.3%，生成 IDs 相同。
  见 [results_runtime_efficiency.json](results_runtime_efficiency.json)。
- 原 PPO loss/core_algos 未被替换。Qwen batch1 padding mask 回补官方修复；
  共有 padding 裁剪保持 FLA 64-token 分块边界。已有短链同权重/输入/DT 优势的
  FP32/普通 BF16 对拍通过本地借鉴 FA 的误差界；该整网对拍不是官方 PPO 容差认证。
  见 [results_short_owner_parity.json](results_short_owner_parity.json)。
- 按用户最新要求，官方容差在对应算子上直接调用原测试：FA 2.6.3 四个 BF16
  forward/backward 用例通过（output 2×普通低精度误差，dQ/dK/dV 3×）；FLA 0.4.1
  原参数列表中的 FP16 chunk 用例通过。额外把 FLA 用例扩展到 BF16 时，dq RMS
  比值 0.008965/0.008983 超过原函数的 0.008；失败保留。固定版及当前官方
  test_chunk 参数列表均使用 FP16，不能把这两个扩展说成官方 BF16 用例通过，
  也没有修改阈值或训练内核。见 [results_official_kernel_tolerances.json](results_official_kernel_tolerances.json)。
- 多步数值差异保留为诊断，不再添加方向余弦、clipping 分支逐位相同等新验收门槛。
  默认 16k math backward OOM、额外 BF16 reduction 参考及确定性重复结果见
  [results_policy_effects.json](results_policy_effects.json)。
- DT 的整网有限分解仍有 conservation residual 超阈的实测样本，报告保留 signed
  原值，不隐式归一化或裁剪；接口组合正确、训练可执行与归因估计精度分别报告。

## 原文训练规模、检查点和异机备份

正式预算见 [paper_scale.json](paper_scale.json)：WebShop 150×128，Sokoban
官方脚本 150×256，AppWorld LOOP 200×240；AppWorld 训练子集由官方
load_task_ids(difficulty=1/2) 取得 72 个任务，与论文的 24 个 scenario 对齐。
这是每任务一组实验的采样预算，不是三随机种子结果，也不替换固定 DT 方法。
具体论文/脚本来源以及模型、观测、奖励、上下文和评估差异均列在配置中。
正式前两次启动触发主机内存问题：Ray 的 900 GiB 容器上限、95% 阈值产生明确的
worker-killed-by-memory-pressure 记录。已停止失败的自有作业；验证 batch 分别
改为 4/4/3，数据条数仍为 WebShop 256、Sokoban 128、AppWorld 57，训练
batch/轨迹总预算完全不变。验证仍按原环境 reset 的采样方式，不声称小批次
采样与一次抽取整个验证集得到相同任务集合。没有关闭 Ray 内存保护。
阶段探针进一步定位到 Pyserini Lucene 导入：环境进程继承 GPU 可见性时，
额外加载加速库，单 WebShop worker RSS 9.383 GiB、RssAnon 5.203 GiB。
使用原 Ray runtime_env 仅屏蔽 CPU 环境 worker 的 CUDA/MACA 后，分别降至
1.235/0.709 GiB；按 132 个环境外推，匿名内存差约 593 GiB。原 worker 的 reset、
搜索、商品/选项点击、购买终止和官方评分完全相同，见
[results_environment_memory.json](results_environment_memory.json)。未修改环境、奖励或采样批量。
v3 三组原文预算作业曾进入首轮 rollout，容器内存抽样约 542–550 GiB；
06:41 在后续 DT RPC 阶段失败，尚未完成首轮更新。历史启动身份和阶段采样见
[results_paper_scale_launch.json](results_paper_scale_launch.json)；不把启动等同于实验完成。
00:59 的 `/proc/*/smaps_rollup` 核算中，WebShop/Sokoban/AppWorld 环境进程
PSS 分别为 96.7/74.7/79.7 GiB。99 个编译子进程 RSS 总和为 470.1 GiB，
但 PSS 仅 14.6 GiB、私有脏页 0.19 GiB；不能把 fork 共享页重复算作内存泄漏。
此时容器使用 549.4/900 GiB；采样只覆盖 rollout，后续 DT/PPO/checkpoint
峰值由每小时检查继续记录，不提前声称正式规模全阶段峰值已通过。

[run_verl_agent.sh](run_verl_agent.sh) 只透传上游 checkpoint save/resume/retention
与任务配置。检查点仍由原 FSDPCheckpointManager 保存完整 model、optimizer、
extra state，trainer 保存 dataloader 和 latest marker。正式启动与恢复状态以
远端运行根目录的 formal-training.json 为准；尚未生成该记录时不能称作已启动。

[backup_metax_to_restic.py](backup_metax_to_restic.py) 让 MetaX 的原 restic 经
4090 跳板直接写入异机仓库，复用其加密、压缩、去重。备份机执行原
restore --verify；检查点还逐文件核对源端/恢复端 SHA256。只选上游完成标记
覆盖的检查点，不改变检查点格式、不删除历史备份。原 roundtrip 检查点的
9 个文件（17.945 GB）已异机恢复并逐文件 SHA256 一致；借助已有权重块，
新增压缩数据约 495.5 MB。当前实验数据、AppWorld 官方输出、三组原 Ray
日志共 13,725 个文件也已备份并恢复验证，仓库 check 无错误。记录见
[results_backup_metax.json](results_backup_metax.json)。正式训练尚未产生首个
检查点，每小时检查会继续备份原生完成标记覆盖的检查点；不将探针检查点
说成正式实验结果。早期中转与校验包装错误均保留，未删除历史快照。

## 入口与历史结果

- `run_verl_agent.sh`：固定上游 VERL 训练入口，DT/PPO/GRPO 共用原训练基建。
- `verify_short_owner_parity.py` / `compare_short_owner_parity.py`：固定原 actor
  的真实短链对拍，独立 FP32 校准；只用于测试，不接管训练。
- `inspect_owner_policy_effects.py`：读取已保存的 log-prob/梯度/权重，调用原
  PPO 函数检查实际概率比、clipping 分支和更新方向；只报告测量，不新设门槛。
- `patch_verl_agent2.py`：固定上游的薄接口补丁与共有左 padding 裁剪；默认上游路径
  保持原行为。环境、rollout、optimizer 和 PPO clipping 都由上游实现。
- `verify_upstream_actor_update.py`：复用真实 worker 的非零更新回归，可复现旧
  FSDP1 参数配置、对照上游参数配置，并保存初始/每次更新后的可训练权重。
  明确使用测试优势与脚本动作；不产生 DT 估计，不作为任务训练数据。
- `results_env_smoke.json`：历史环境 readiness。
- `results_reward_events.json`、`results_event_transport.json`：奖励代数和传输历史检查。
- `results_reward_readout.json`：已撤下的逐前缀读出历史结果，非当前 EOS 方法结果。
- `results_counterfactual_32k_a6000.json`、其他旧归因/训练记录：只适用于各自历史
  实现，不能作为当前 PLAN 的验收。

实际累计长度、DT 调用次数、首轮/后续时间及显存必须来自新日志；不将配置上限
当作真实任务长度，不将所有非零 reward 项的多个 DT 调用报告为一次。

# 远端环境记录与复用入口

**当前执行平台（2026-09-22 用户最新指令）：只用 MetaX，不再使用 A6000。**
**最新显存约束：MetaX 使用物理 64 GB 上限、不 OOM；不再要求 48 GB。
32k 须主动构造已知长度验证，显存修复须同时对比预热后的速度。**
下方 A6000 内容仅保留为已有环境与历史验证记录。当前任务没有在 A6000
运行的自有训练/探针作业；后续从本文的 MetaX 复用入口恢复。

整理日期：2026-09-21。**这台机器已经配置过环境；恢复工作应从复用开始。**
本文件记录环境、路径和已知问题。RL 方法只以 [PLAN.md](PLAN.md) 为准；
本文件和历史运行记录不代表该计划已实现或 DT 多轮训练已经验收通过。

## 2026-09-24 rollout 修复与复用

- 三组旧正式作业主动停止，状态是 `stopped_for_rollout_throughput_repair`；
  WebShop 完成的 step 1 检查点保留，未完成轨迹按用户指令丢弃。恢复时读取
  当前 manifest，不复用已终止 PID，不重新安装环境/权重/缓存。
- MetaX `ROLLOUT_MAX_NUM_SEQS` 默认 32，独立于 actor/DT minibatch4。
  同卡重复生成调用 92.713→29.948 秒；并发 32 的 32768 DT/PPO 容量复核通过。
  `VERL_TRIM_RESPONSE_HEAD=1` 调用 Qwen 原选择性 logits 接口；真实短链有效
  log-prob、两次 PPO 梯度/参数与修改前相同。完整任务耗时尚需实际检查。
- 数值对照复用 `CUBLAS_WORKSPACE_CONFIG=:4096:8` 和
  `FLASH_ATTENTION_DETERMINISTIC=1`，满足原测试的确定性设置；不是新的训练
  参数要求。候选目录仅加 PYTHONPATH 不保证加载，因为工厂会插入发布路径；
  检查实际 `__file__`，正式运行使用带哈希 manifest 的统一发布目录。
- `receipts/vllm-active-rows/` 保留并发、物理显存、32k 容量、接口、真实 head
  数值对照与 B1 编译回归结果。早期测试启动/配置/旧模块加载失败也保留。

## 2026-09-23 已有加速分支的复用

- 当前正式预算运行使用已推送提交 `6d1a946`，完整发布目录
  `$DT_RUNTIME_ROOT/releases/6d1a946`。其 94 个 Git 文件及 10 个实际导入路径
  已校验，部署边界测试 6 passed；`environment.json` 原样复用前一发布目录，
  未重配环境/权重/缓存。新修复只给现有动态编译标记补上 batch 维，消除
  B4→B2 时的 7 次图编译，未改有限传播公式、Q/V/A 或 PPO。
  2026-09-23 20:35:24 +08 启动，运行目录
  `runs/native-prefix-6d1a946-formal`。任务、PID、检查点、Ray 目录与预算均
  记录在当前 `formal-training.json` / `active-training.json`，源目录由
  `active-source.json` 指定。启动时 GPU 4/5/6 空闲，分别运行
  Sokoban 150×256、WebShop 150×128、AppWorld 200×240；GPU 编号不是预约。
  备份脚本已按该 manifest 增加实际发布目录，不能只备份历史 `repo/`。
- 已完成的三任务 pilot 使用已推送提交 `d6b7351` 的完整发布目录
  `$DT_RUNTIME_ROOT/releases/d6b7351`，其 `source-manifest.json` 记录 93 个
  文件的 Git 来源和 SHA256。设 `DT_ROOT` 为此目录、`DT_ENVIRONMENT_JSON`
  为目录内 `environment.json`，再 source 原 `experiments/rl/environments/metax.env.sh`。
  `PYTHONPATH` 使用该发布目录的 `clean/qwen35`、`experiments/rl` 和目录本身，
  不再叠加多层历史候选。实际导入回执 `import-receipt.json` 已核对；配置/
  Ray 传输边界 6 项通过。未重新配置 Python、模型或持久缓存。
- 启动记录为 `runs/native-prefix-d6b7351-pilots/jobs.json`
  （2026-09-23 18:59:31 +08）。Sokoban/WebShop/AppWorld
  分别使用当前空闲 GPU 4/5/6，均为两迭代验收、32k 上限、actor/DT batch4；
  AppWorld 保留 40 轮。`active-source.json`/`active-training.json` 已更新，
  三个 pilot 已两迭代退出 0，每个任务均有真实非零奖励/DT/PPO 更新；旧失败
  manifest 作为 `previous-formal-training.json` 保存在新运行目录。原 PPO 核心
  SHA256 仍为 `5043f97b87b00ab5c5d907022ea6cf148b935fdb71c37222a3f9c33ccfaf1dc6`。
- `candidates/dt-minibatch/native-prefix` 复用原模型混合缓存，新增可选
  `dt_reuse_native_prefix=true`。已编译库 `libfinite_cached_queries.so` 的
  SHA256 为 `5d2af760abb2684401682029e41ac68aefc58ed2796082d435874a2b74bbcebb`。
  不重建缓存/环境。原有限 FA 只扩展 Q 后缀、完整 K/V 的输入布局，有限公式
  不变；原生 GDN 使用真实缓存状态和卷积左窗口，已纳入上述完整发布目录。
- `capacity32k.json` 状态 passed，退出码 0，总计 337.772 秒；完整热 DT
  27.814 秒，原版官方反向主参照 44.007 秒。B4/32768、1024 action 槽位、
  vLLM 驻留、actor microbatch4、原 activation offload 和 FSDP2 参数卸载。
  两次 PPO 更新及原生 LoRA 同步完成，1,441,630 个可训练元素改变。
  32769 明确拒绝。该输入是容量夹具，不能称为真实任务的成功率/吞吐验收。
  DT 结束物理占用约 31.44 GiB，PPO 阶段快照约 51.27 GiB；这不是物理
  峰值的连续采样。完整作业无 OOM，进程 RSS 约 11.1 GiB，不代表整机用量。
- 原生/有限 FA 的官方输出和 dQKV 断言，以及 FLA 的原 FP32 recurrence/
  assert_close 已用于缓存续算对照。短链累计差异单独保留，不另造整网门槛：
  A 相对 L2 约 1.78，V 约 0.0526；原生 GDN 前向首先出现差异，逐层重放
  与对应根前向相同。后续部署须保留这些诊断，不把守恒或算子通过冒充整网
  逐值一致。当前三任务正式训练与监控/备份仍未恢复。

- 隔离候选 `candidates/dt-minibatch/compact-gdn` 增加
  `dt_compact_gdn_captures=true`；`pinned-root` 叠加
  `dt_pin_root_host=true`。沿用所有下层候选、原模型和持久缓存，未重装环境。
  前者只捕获原始 GDN 末段，包含原 FLA h 和原生卷积左窗口；后者复用原
  pinned copy 接口保存根前向检查点，不改变模型或有限公式。
- `compact-copies32k.json`：相同实际张量的 D2H 22.109375→0.820831 GiB，
  GPU 保存量 9→0.333984 GiB；拷贝预热合计 0.417647/0.417222→
  0.016649/0.020149 秒，打包另计 0.034629 秒。18 项捕获值与原切片相同。
  完整短链 A 相对 L2 差 0.008337，单独记录，不创造整网容差。
- `short-root-pin-v2.json`：原位置 ID 的零步幅视图在拷贝边界先实体化，
  随后完整短链 signed/target/Q/V/A 与原搬运路径逐值相同。首个失败日志
  `short-root-pin.log` 保留。`root-copy32k.json` 的实际 2 GiB 输入拷贝预热
  0.351250/0.341406→0.037766/0.037749 秒，值和步幅相同。
- 原生反向的主要参照保持 `native-backward32k-v3-vllm-ppo.json` 的
  44.007159 秒；当前相同参数卸载选项的 `native-policy-offload32k.json`
  为 44.231252 秒，仅作诊断，不替换原参照。双方可以采用各自可行、高效
  的卸载方式；模型、卡、输入、B4/32768、vLLM 驻留和物理容量条件须对应。
- `pinned-root/capacity-compact-pinned32k.json`：DT 输入精确 32768，B4
  成对内部 batch8，1024 action 槽位，actor microbatch4。两次 DT 耗时
  158.108/121.332 秒；两次原 PPO 更新梯度非零，1,440,439 个元素改变，
  200 层 LoRA 通过原 vLLM manager 同步，无 OOM。记录各阶段物理显存与
  进程 RSS；RSS 不是驱动固定页内存和整个主机占用的总和。
- 热调用根前向 34.624、native replay 44.801、finite decoder 31.134 秒；
  decoder 已包含恢复 5.068 和公开 FA LSE 5.395 秒。相对原 44.007 秒
  反向仍约 2.76 倍，完整效率未达标。未选择为生产、未恢复正式训练或检查。
  对应代码与原始回执哈希见 `results_dt_minibatch_candidate.json` 的
  `compact_gdn_captures` 和 `pinned_root_checkpoints`。

- 最新 GDN 隔离候选为 `candidates/dt-minibatch/gdn-suffix`，叠加于下述
  `coefficient-suffix` 等已记录目录；未改生产选择。`dt_gdn_coefficient_suffix`
  使原 FLA 有限回调使用从首个变化 token 所在 64-token 块开始的实际保存
  张量，保留进入该块的原 h 状态。没有复写 FLA 递推或改变 EOS/Q/V/PPO。
- `fla-suffix-official-v2.json`：128/447 的原 FP32 recurrence 与原 FLA
  assert_close 均通过；完整短链另记录 Q/V/A 差异，不将逐值相同作为新增门槛。
  `short-suffix-diagnostic-v2.json` 已完成，A 相对 L2 差 0.01175、最大差
  0.016401。此前整网零容差失败和精简测试漏读基准的 KeyError 日志均保留。
- `same-gdn-operands32k.json`：原 actor/休眠 vLLM，同一次实际首层 GDN
  捕获和 upstream、相同驻留，完整预热 3.361/3.349 秒，范围受限
  0.819/0.820 秒，所需输出最大差 3.49e-8、相对 L2 差 4.89e-6。
  随后主动退出，未运行其余 DT/PPO，未验收完整效率或恢复三任务。
  原始回执哈希见 `results_dt_minibatch_candidate.json::gdn_coefficient_suffix`。

- 后续有限 FA 候选在 `candidates/dt-minibatch/coefficient-suffix`，库
  `libfinite_d256_suffix.so` 的 SHA256 为
  `63ad5dde0b2e2f02275e4960ee1f654e076c55fc9963b15a760d30f853c722fa`。
  可选 `dt_fa_coefficient_suffix=true` 由同一 runner 找到每对输入首个变化
  位置，仅省略共同因果前缀的系数输出计算，保留完整 K/V、输入、EOS 端点
  和原有限公式。原库导出和默认行为保留，尚未选择为生产配置。
- 新库全量路径通过原 FA 128/447 测试，与之前候选的非零端点五项输出
  逐值一致；范围受限路径在 128/447（含非整块起点、不同有效长度）和
  32768 上与同库全量路径的所需系数逐值一致。完整短链 signed/target/Q/V/A
  也逐值一致；既有估计守恒误差仍保留，未当作消失。
- `same-fa-operands32k.json`：原 actor、休眠 vLLM 和同显存起点，首个实际
  FA 输入仍为 B4/32768，各样本系数起点 31573。全量预热 13.381/13.377 秒，
  范围受限 0.585/0.587 秒（约 22.8 倍），所需系数逐值一致；随后主动退出。
  这只证明该有限 FA 阶段收益，未重跑全部 DT/PPO、未恢复三任务正式训练。
  回执和哈希位于 `results_dt_minibatch_candidate.json::fa_coefficient_suffix`。

- 最新隔离候选在 `candidates/dt-minibatch/selective-capture`，生产目录仍未
  选择。原 VERL `fsdp_config.offload_policy` 启用 Torch FSDP2
  `CPUOffloadPolicy(pin_memory=True)` 后，初始化物理占用约 18.95→2.285 GiB。
  修复的是 DT 读出用 CPU 参数存储位置建立索引的接口错误：计算设备读取
  DTensor 的原 device mesh；未实现另一套参数卸载。
- `root-policy-comparison.json`：开启/关闭该 policy 的真实 9B，923 个参数
  指纹、输入 IDs/mask 一致，实际 `[8,635,4096]` 的 32 层输入和最终 logits
  全部逐值一致。历史跨运行目标分数差异没有在此受控对照中复现成 policy
  差异；不据此宣称所有历史波动都已解释。原生根接口及奖励组合回归 35 passed。
- 通过原 capture 可选目的设备保留 `h/w/v_new/A` 共 9 GiB，首层 GDN 的
  D2H 从 31.109375 降至 22.109375 GiB；24 层推算减少累计双向搬运 432 GiB。
  `selective-same-capture32k-v2.json` 使用同一次实际捕获和 upstream，双方
  保持相同 GPU 驻留；预热 CPU 路径 5.601/5.813 秒，GPU 路径 3.342/3.356 秒，
  输出逐值一致，约减少 41.3%。此计时只覆盖首个 GDN 有限阶段。
- 同分组完整短归因的 signed/target/Q/V/A 仍逐值一致。32k 诊断在首个 FA
  和 GDN 后主动停止，未跑其余层/PPO；replay 的物理快照达到设备总量
  68,283,269,120 bytes，不能声称有剩余空间或完整容量已通过。第一条对照
  命令缺少必填 `--artifacts`，在模型初始化前退出，日志保留；更正后通过。
  原始回执和哈希索引见 `results_dt_minibatch_candidate.json` 的
  `parameter_offload_and_selective_capture`。完整 DT 效率仍未达标。

- 分支头已通过 origin 复核：clean-v1-acceleration 为 `2b36c4ec`，
  qwen35-cause-and-tolerance 为 `e1e37bb4`。复用源码及迁移边界见
  `deltatrace/accelerated/README.md`；方法仍只以 PLAN.md 为准。
- 当前 MetaX 生产 Python 仍为 `eda0e231` 的根模块修复版本；新 minibatch、
  capture、延后同步和 GDN 分阶段内存候选均在
  `$DT_RUNTIME_ROOT/candidates/dt-minibatch`。不把候选测试当作正式部署。
- `execution-reuse-v1.json`：同一 9B actor、B4/635 tokens 的 12 次执行
  全 signed/target/QVA 一致。预热均值 borrowed 3.293、retained 3.280、
  local-events 3.083、deferred 3.045 秒；不是 32k 或完整反向成本结论。
- `minibatch-parity-v10-staging-isolated.json` 分开测试 head 分组和 CPU
  搬运，同分组的搬运前后逐值一致。`fla-coincident-head-partition.json`
  为完整/head8 的两端重合算子检查；使用官方 FLA FP32 recurrence 和原阈值。
  完整原生 BF16/L2 复合测试的历史失败没有被此局部通过覆盖。
- `gdn-norm-official-tests.json` 通过原 `test_rmsnorm_gated` 的 8 个
  原生/有限对照；有限端点重合后对应输入梯度，权重梯度仍由原生算子负责。
  公式只从原 GDN 函数中提取，再交给既有 FiniteBoundaryOps 编译接口。
- `reuse-interface-regression-v2.log`：27 passed，已回显实际导入候选路径。
  第一次错误使用历史串行 test double 的 4 个失败保留在旧日志。
- `batch-capacity-v11-reused-staged-gdn.json`：精确 32768、B4/1024 action
  槽位约 182 秒时，在 GDN 有限 RMS 的 2 GiB 临时分配处 OOM。v12 融合
  原 RMS/SiLU 有限规则后通过该处，随后卷积 SiLU 的 4 GiB 临时张量 OOM。
  v13 通过相同编译接口融合这段原公式后完整通过：DT B4 一次调用
  671.653 秒；两次非零 PPO 梯度范数 0.030762/0.024170，改变 1,439,802
  个参数元素，200 层 LoRA 原生同步，无 OOM。整体 917.039 秒。
  回执 `batch-capacity-v13-compiled-conv.json`；只验收容量与调用链，速度未达标。
- 候选 receipt 的 `qwen35.dt_offload_replay_mixer=true`、
  `dt_gdn_head_batch_size=8`、`dt_compile_gdn_scalar_rules=true` 通过原
  make_qwen35_runner 选项接入；生产 receipt 未修改。v14 用这些配置复测，
  不通过测试 wrapper 注入参数。
- `measure_native_backward.py` 使用原 VERL actor、同 B4/32768 事实输入和
  checkpoint，单独计时原生前向/反向，不含 PPO/optimizer。无 activation
  offload 的 v1 在原生 FLA 前向 OOM；v2 显式启用已有 VERL activation
  offload，结果另存，不能隐藏失败或将 PPO 更新总耗时冒充纯反向。
- `fa-key32-cost32k.json`：64×32 tile 候选 16.187 秒，慢于保留的
  64×64 候选 14.616 秒，未采用。未修改生产 finite FA 库或持久缓存。

- v14 配置入口容量复测通过，完整 DT 725.507 秒、总计 973.723 秒；
  不是提速。此前 v13/v14 均为 PPO minibatch4/microbatch1。
- 原生反向对照 `native-backward32k-v3-vllm-ppo.json` 启用原 VERL
  activation offload，包含原生 vLLM 同步/休眠残留。B4/32768 预热反向
  44.007 秒；PPO 实际两次 `[4,32597]`、68.353/67.520 秒，非零梯度，
  更新后的 vLLM 同步通过。无 offload OOM 不作为速度基线。
- 新搬运候选单独位于 `candidates/dt-minibatch/pinned-capture`；主候选
  v13/v14 文件未覆盖。`dt_pin_replay_host=true` 只启用 Torch 的 pinned
  GDN 捕获，退出作用域前同步。去除未读的 norm_output/output 后，每层
  复制 37.109→33.109 GiB；单原生层成本对照和短链搬运逐值对照已通过。
  完整 v15 已通过 actor microbatch4、activation offload 和 vLLM：DT
  487.649 秒、总计 742.741 秒，两次非零 PPO 更新和原生 LoRA 同步完成。
  GPU5 进程已退出，卡号不是预留；完整效率仍未通过。


- 后续搬运候选在 `candidates/dt-minibatch/pinned-restore`；生产目录仍未
  切换。它将同一 pinned capture 接到 FA，并允许 Torch 在当前流内直接
  恢复 pinned GDN 切片。`restore-minibatch-parity.json` 中，同一 head8
  分组的完整短归因、目标 log-prob、Q/V/A 与原非 offload 路径逐值相同。
- `profile_dt_replay_memory.py` 可设 `DT_PROFILE_GDN_LAYERS=2`、
  `DT_PROFILE_OUTPUT=<JSON>`，观察指定 GDN 阶段后主动退出。输出明确属于
  部分运行诊断，不作完整容量/训练通过；`DT_CAPACITY_SCRIPT` 指向已暂存的
  同版容量入口。vLLM 启动仍须 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False`，
  回到 DT/PPO 后由原 sharding manager 恢复 True。首次诊断漏设该启动参数
  被原 allocator 拒绝，失败日志保留，未据此重装或改 allocator。
- 实际首两个 GDN 层诊断在 `in-context-gdn32k-v2.json` 后主动结束：首层
  包含编译约 12.615 秒，下一层约 5.010 秒。CPU profiler 显示主要仍是搬运；
  不能用孤立单层 2.206 秒取代真实上下文下的耗时。随后同一次实际捕获、
  同一 upstream 的对拍为旧路径 5.589 秒、新路径 2.289 秒，输出逐值一致；
  `in-context-transport-compare32k.json` 保存该单次对拍，不能扩大为整网稳态收益。
- 原生 mcTracer 的 `fa-native-trace32k-v2/` 确认当前有限 FA 三个 GPU 阶段
  分别约 1.934/5.035/7.719 秒，主要成本在 Q 与 K/V 传播。trace 原时间单位
  按纳秒换算后与同步墙钟约 14.7 秒相符；CPU 同步等待不另加到 GPU 时间。


- 复用历史 Qwen3 的固定左矩阵寄存器缓存方式，仍调用厂商
  `flash::gemm_opt<true,false>`；D256 保持原 64×64/4-warps 布局。
  `libfinite_d256_cached_owner.so` 的 B4/32768 预热为 13.797 秒，当前
  14.616 秒参照另存；改进有限，不能据此报告整体达标。候选通过固定 FA
  128/447 原断言，32768 非零端点 dq/dk/dv/tau/center 与原候选全部逐值相同。
  新源文件已保留，生产库仍未选择；尝试 M32 的厂商布局编译拒绝也保留。


- 实際捕获清单已与 layer30 的 19 次 copy 记录对齐：每层 D2H 为
  35,550,920,704 bytes（33.109375 GiB）。input 2 GiB 只用于 shape 检查，
  在该历史快照尚未删除，后续 shape-only 候选已删除；不能宣称全部搬运必要。当前 restore 调用按实际形状
  推得 H2D 31.15625 GiB，旧 head 切片额外 pageable CPU 重排 15.078125 GiB
  已由同一 Torch copy 接口的 pinned 源路径省去。24 个 GDN 层的累计 D2H/H2D
  约 1.506 TiB 是搬运量，不是驻留内存。此清单索引在
  `results_dt_minibatch_candidate.json::gdn_transfer_inventory`。

- 最新捕获候选目录 `candidates/dt-minibatch/shape-only` 同时包含 input
  形状元数据和 FA 转置视图共享。`shape-metadata-parity.json`、
  `dense-alias-minibatch-parity.json` 均通过相同 head 分组的完整短链搬运
  逐值对照；前者还确认 cached finite FA 的 Q/V/A 与当前候选相同。
- `DT_PROFILE_DENSE_ALIASES=1`、`DT_PROFILE_OUTPUT=<JSON>` 在原 32k
  容量入口观察首个 FA 的真实 Q/K/V 拷贝，随后主动退出。回执
  `dense-alias-in-context32k.json`：旧拷贝总计 0.438 秒，新共享视图约
  0.00016 秒，值与 stride 全等。属于 D2H 捕获局部比较，不作完整 DT
  速度或训练验收。两项删除按当前形状合计减少 128 GiB 累计双向搬运。
- `fa-kernel-attributes-separated.json` 从 MetaX 公共 runtime 查询已编译
  内核资源；每个库使用独立进程，避免相同导出符号的注册冲突。当前 Q/KV
  阶段各 256 registers/thread、156/764 local bytes/thread。backward warp
  布局探针被快速转置接口编译拒绝，日志保留；未运行、未替换生产有限 FA。

## 2026-09-22 正式规模启动与监控

- 2026-09-23 MetaX 有限 FA 候选继续复用同一厂商 framework。`gemm_rs` 快速
  转置接口按两个 128 列视图处理 D256；64×64、4 warps 的预热 batch4/32k
  为 14.616 秒，原有限核约 48.771 秒，原生 FA backward 1.009 秒。
  候选库位于 `candidates/dt-minibatch/libfinite_d256_owner_optimized.so`，SHA256
  `21c55032fc04d08c5fb8346dd961cf31bea772320b38f5c5751d2bf8800f35d1`。
  device fatbin 与已计时/长数值检查的 split-rs 候选逐字节相同。
  原 environment.json 未切换；恢复环境时不可把候选状态当正式训练已通过。
  官方短算子断言、32k 非零端点诊断、完整短 DT 差异及失败结果见
  [results_dt_minibatch_candidate.json](results_dt_minibatch_candidate.json)。
  CPU capture 的跨设备复制需要保留 strides；MetaX 默认复制曾改变四维布局，
  修正后同批次短链 Q/V/A 逐位相同。最新 v13 容量通过，速度仍未验收。
- FSDP root 修复后的 WebShop/AppWorld 恢复作业正常结束，完成 step 2/4，
  但这轮均为零奖励/零优势/零梯度。新回执为
  `receipts/training-setup/vllm/task-pilots-v2-results.json`；不是非零 DT 验收。
- 2026-09-23 FSDP 接入修正：vLLM 的 WebShop/AppWorld pilot 在先做零奖励
  PPO、后首次进入非零奖励 DT 时，报 `FSDP state has already been lazily
  initialized for model.embed_tokens`。原本地补丁把 FSDP2 放在 PEFT 内层，
  而 PEFT `BaseTuner.forward` 直接调用内层 `.forward`，绕过根模块 hook。
  现恢复固定 VERL 原 `apply_fsdp2(actor_module, ...)`，与官方 v0.7.0 相同；
  DT 使用同一 FSDP 根的 `register_fsdp_forward_method`。已有 Python/模型/
  缓存均复用。小模型两种调用顺序及真实 9B actor-first→DT 通过，后续三任务
  仍须重新验证，不以此前失败进程或零奖励更新作为成功。
- main 路径审计：`origin/main=6ca8dc07` 已有并正在复用 MetaX 有限 FA/FLA
  实现。`candidates/dt-minibatch/` 内是隔离数值/性能候选；原 environment.json
  的有限 FA 库没有切换。32k batch4 的归因内存与速度未验收，不自动重启正式规模。
- 2026-09-23 vLLM 复用检查：系统已有 `vllm==0.15.0` 和
  `vllm-metax==0.15.0+g24fb31.d20260310.maca3.5.3.20.torch2.8`，位于
  `/opt/conda/lib/python3.12/site-packages`；现有训练 Python 可直接导入。
  厂商 `vllm_metax.models.qwen3_5` 注册了当前权重的
  `Qwen3_5ForConditionalGeneration`。原 `EngineArgs.create_model_config()`
  已对当前 MODEL_PATH、BF16、max_model_len=32768 成功解析；原 registry
  识别为 hybrid text-generation 模型，原 LLM 有 sleep/wake_up/collective_rpc。
  接入使用原 `ActorRolloutRefWorker`、`vLLMRollout`、
  `FSDPVLLMShardingManager`、`TensorLoRARequest` 和 sleep/wake。通过
  `ROLLOUT_BACKEND=vllm` 选择，默认仍为 HF；PLAN 的 Q/V/PPO 不变。
  已回补 VERL v0.7.0 原 `VLLMHijack` 的 0.15 API 兼容（来源和 SHA 见
  `patches/verl-v0.7.0-vllm-lora.patch`），FSDP2 使用同一 PEFT actor 的原
  load/offload；Qwen 文本 actor 到完整 checkpoint 仅映射已有 LoRA 键前缀。
  实际 native loader 已绑定 200 个 LoRA 层，没有未使用的 adapter 键。
  完整 DT/PPO/三任务训练仍须以本次运行回执为准。
- vLLM 0.15.0 `v1/worker/gpu_worker.py::load_model` 的 `with A and B`
  实际未进入 weights pool；原生 sleep 后仍占 19.056 GiB。仅回移官方
  [v0.17.0 的双上下文写法](https://github.com/vllm-project/vllm/blob/v0.17.0/vllm/v1/worker/gpu_worker.py)，
  不改 MetaX allocator。已执行 `patch_vllm_sleep.py`，目标为
  `/opt/conda/lib/python3.12/site-packages/vllm/v1/worker/gpu_worker.py`。
  真实 9B sleep/wake 后占用降至 2.285 GiB，163 个 weights allocation 得到
  host backup；唤醒及正常退出完成。回执 `vllm/owner-sleep-v4.json/.log`。
  修复为显式一次性操作，恢复运行时不要重装 vLLM 或重建缓存。
  同机早期独立 256 MiB allocator 小探针在退出时中止，保留为
  `vllm/native-pool-check.*`；不能把它当完整 engine 的退出结果。
  CuMem 初始化使用 `expandable_segments:False`，原 VERL 阶段接口返回
  DT/PPO 时恢复 True；休眠后 torch allocated 包含未映射的 pool，不能直接
  作为物理占用。报告需同时读取 device memory 或 mx-smi。

- **2026-09-23 06:41，v3 正式训练失败并已停止。** WebShop 首轮 rollout 后
  `compute_dt_token_advantages` 的 Ray 参数序列化将单行 tensor 视图所引用的
  整批 storage 重复打包；TaskRunner 达 373 GiB、容器 883/900 GiB，Ray
  杀掉三组 policy worker。不要自动重启旧代码。记录见运行根目录
  `receipts/training-setup/formal-v3-oom-stop.json` 和 `rollout-fix/`。
  原 `to_list_of_dict` 只在 DT 调用启用行级 clone，值/顺序/mask 不变，默认
  owner 路径保留。原 HF microbatch 同时跳过 collector 标记已结束的行。
  原规模、最长轮数、每行 32768 的独立序列化夹具峰值分别为 WebShop
  6.512 GiB、Sokoban 12.157 GiB、AppWorld 29.091 GiB；这只证明传输容量，
  不代替完整训练吞吐或所有环境常驻内存验收。每小时检查须回显真实阶段、
  轮次/有效轨迹/token 吞吐；不能把进程存活或 GPU 忙当作训练健康。

- 2026-09-23 按 phase 定位前两次内存失败：Pyserini 的 Lucene 导入会间接
  加载加速依赖。GPU 可见的 WebShop Ray worker 为 RSS 9.383/RssAnon 5.203 GiB；
  原 Ray `runtime_env.env_vars` 将 CPU 环境的 CUDA/MACA 可见性置空后为
  1.235/0.709 GiB。只改原资源配置；policy actor 的 GPU 配置保留。
  原 worker 实际 reset/search/click/options/购买和 task_score 对拍一致。
  证据：`webshop-memory-phase-visible.json`、`webshop-ray-cpu-memory-v2.json`。
  第一版对拍没有走到购买终止，不用它代替第二版的奖励路径检查。
- v3 首轮 rollout 的 PSS 证据保存在
  `receipts/training-setup/formal-v3-rollout-memory-pss.json`：容器 549.4/900 GiB，
  环境进程 PSS 合计约 251.1 GiB；编译子进程 RSS 合计 470.1 GiB，但 PSS
  14.6 GiB、Private_Dirty 0.19 GiB。用 smaps_rollup 区分 fork 共享页，不能
  简单加 RSS/RssAnon 判断真实物理占用。此时三组均在实际 rollout 生成，
  正式规模的 DT/PPO/checkpoint 阶段峰值尚待记录。
- 异机备份改为 MetaX 的 restic 直接经 4090 到 A6000 存储。MetaX
  `backup-access/` 保存专用出站私钥、严格 host-key 配置、repo 密码和原 restic。
  跳板公钥只允许到目标端口的转发，存储端公钥强制 SFTP；凭据不入仓库或日志。
  A6000 的原 Qwen3.5-9B 文件已加入同一 restic 仓库作为去重种子，避免再次
  传输能由相同块复用的权重；实际节省量以 restic summary 为准。
  每次备份在存储端执行 restore --verify，检查点另外对源/恢复文件 SHA256。
  本机旧 tar 中转仅完成元数据验证，完整检查点流已停止，不能称为成功。
- 2026-09-23 直接备份已通过完整校验：原 roundtrip 检查点 9 文件 / 17.945 GB，
  异机 SHA256 全部一致，已验证快照为
  `17463d66fdc2acfc56a4b42bcafa3715efeb46f8576e241832de4b3b95d52ebe`。
  最新数据/日志快照为
  `32f46d86cf76b30a29c08bf5985cb9b2c293d0a0160bbd0c1271be86d904ee6c`，
  包含 AppWorld 原输出及三个作业各自的 Ray 日志，恢复/check 均通过。
  原低速中转留下的失效锁在核对 owner PID 已消失后用 restic unlock 清理。
  备份机 Python 3.8 使用流式 hashlib.sha256；不要使用 file_digest。
  逐文件校验清单若通过本机 SSH stdin 传递，使用 bytes，避免 Windows 自动
  把 LF 转成 CRLF。正式训练首个检查点尚待原 trainer 完成，不能混淆两者。
- 训练预算配置：`experiments/rl/paper_scale.json`；当前进程与精确启动时间以
  MetaX 运行根目录 `formal-training.json` 为准，不把旧 PID 当作当前进程。
  第一版并发验证环境过多，Ray 的 900 GiB 容器内存阈值触发杀 worker；证据
  `receipts/training-setup/formal-v1-memory-evidence.json`、
  `formal-v1-host-memory-stop.json`。只按三个确切 RAY_TMPDIR 停止自有进程。
  使用原 `data.val_batch_size` 分批验证，验证数据条数与训练预算保留。
- 原检查点路径已完成真实保存/清空 LoRA/恢复，参数逐项相同。修复仅在
  原 checkpoint metadata 与原 FSDP1 附加 adapter 导出边界；FSDP2 完整
  model/optimizer/extra 检查点未换格式。记录 `checkpoint-owner-roundtrip-v3.json`。
- AppWorld 官方 load_task_ids 的 difficulty 1/2 并集为 72 个任务，生成
  `data/datasets/train_difficulty_1_and_train_difficulty_2.txt`；无新采样器。
  既有20服务补充277时官方随机端口重复4个，保留日志并另外补4个服务，
  去重后297个端口的 /openapi.json 全部200。正式端口表是 AppWorld 根下
  `appworld_ports_formal.ports`，不会覆盖旧端口表。没有重装环境。
- 早期连接时本机 SSH 密钥未被接受。2026-09-23 复查时，原生
  `ssh -oBatchMode=yes -p 32036 root@ssh.v5000-prod-gw.nhss.zhejianglab.com`
  已以现有公钥认证成功，原生 SFTP 也可复用；本次没有新配密钥。
  原 Windows DPAPI 记录（仓库外）及
  `C:/Users/Administrator/.codex/private/metax_exec.py` 仍保留。握手曾短暂超时，
  随后同一入口恢复；不能据此判断训练结束或重新配置环境，不输出密码。
- 每小时线程检查已创建，automation id=`deltatrace`。2026-09-23 用户已取消
  Codex 剩余额度 20% 停工条件，不再查询额度或按该条件停止操作。
  三任务 pilot 通过且正式预算作业进入生成后，已恢复该检查并保留每次回显；
  不因一小时未结束大批次就重启任务。首轮 metadata 异机备份/restore 已通过，
  回执为本机 `research/temporary/rl_backup_20260922/formal-6d1a946-initial.json`，
  快照 `90fa82b72894b4c2b25bee6f2bb68dcc27a037fe489e82eb523a490d34e50c10`。
  当时新正式检查点尚未产生，不能将该 metadata 回执当作新检查点备份成功。
  环境状态/调用审计另见 `receipts/environment-state-20260923.json`，未更改
  正式训练状态；定时检查须区分原格式有效性标记与实际动作/API 执行结果。
- 异机备份目标为 `liangchen@10.70.5.230:2501`（经4090），目录
  `/data/liangchen/deltatrace_rl_metax_backup`；只用于存储，A6000不运行训练。
  restic 0.19.1 来自官方发布，压缩包 SHA256 为
  `f415415624dcc452f2a02b8c33641791a8c6d6d3b65bbb3543fcf9a25151585c`。
  repository.password 位于该私有目录（0600），本机另有 DPAPI 加密恢复副本，
  均不入 Git。每份备份的 restore/check 结果另报，不以仓库创建冒充备份完成。

## 2026-09-22 最新短链对拍与 padding 修复

- 数值探针启用 `torch.use_deterministic_algorithms(True)` 时须在进程启动前
  传入 `CUBLAS_WORKSPACE_CONFIG=:4096:8`。16k 探针首次缺少该变量，失败
  记录保存为 `long-parity-16k-missing-workspace.*`，未重装环境。16k 的
  head/padding 分离对照完成后发现相同输入/初值/DT 信号的跨次更新波动，
  以 HF 原有 `FLASH_ATTENTION_DETERMINISTIC=1` 完成同路径重复，两次
  log-prob/原梯度/参数更新零误差一致。优化版对原版的长链差异仍在，测量见
  `long-policy-effects-deterministic.json`，不把重复一致说成两实现一致。
  原版 16k 默认 math 对照先在 MCBLASLT 内中止；添加阶段日志与 Torch 原生
  `save_on_cpu(pin_memory=True)` 后，定位到 backward 申请额外 16.36 GiB 时
  OOM。记录 `long-parity-16k-math-owner*`；没有重装依赖或清缓存。
  额外 BF16 math 诊断使用 Torch 现有 `allow_fp16_bf16_reduction_math_sdp(True)`，
  保留同一输入/LoRA 初值/DT QVA，明确区别于默认 math 中间精度，不作为新的
  FP32 基准或放宽验收。记录 `long-parity-16k-math-bf16*`。CPU saved-tensor
  offload、阶段日志和确定性选项仅用于探针，生产训练未因此改变。
- 继续复用 MetaX 原 Python、权重和缓存。安装版 Qwen3.5 的
  `apply_mask_to_padding_states` 在 batch=1 时跳过 mask，实际有效 token
  log-prob 偏差超过 4；PPO microbatch=1 正好触发。只回补官方提交
  `59eed1a6ba3566d5d4e8bec23bc1a21bfbda3e84` 的条件修复，没有升级依赖栈。
  [官方源码](https://github.com/huggingface/transformers/blob/59eed1a6ba3566d5d4e8bec23bc1a21bfbda3e84/src/transformers/models/qwen3_5/modeling_qwen3_5.py)，
  [相关缺陷说明](https://github.com/huggingface/transformers/pull/46773)。
- 实际模型源码从 `3ef131eadfaf519937e2b78e3122549b50952e9c55f3553f908a4755ce34edb4`
  变为 `f7e1a804fa12684bd1cc225c85cdf5f0b5996f30d66263f11de13449f53272be`。
  记录在 `receipts/training-setup/qwen35-padding-mask-patch.json`，旧源码保存在
  `qwen35-before-padding-mask.py`。环境 JSON 的 `qwen35.rl_runtime_model_sha256`
  与 `rl_runtime_model_patch_receipt` 记录当前 RL 补丁；原 native_model_sha256
  保留为历史 owner 快照，不能误当成安装后当前文件校验值。
- Qwen 的共有 padding 裁剪按已固定 FLA 的 64-token chunk 对齐，最多保留
  63 个 padding；其他模型及关闭裁剪时仍保持原行为。`qwen35-padding-mask-tests.log`
  为 2 passed，`chunk-align-interface-tests.log` 为 12 passed，重复应用补丁
  后模型源码 SHA 不变。没有改动 PPO loss、DT 公式、模型权重或 FLA 算法。
- 原 actor 基线来自 VERL 提交 `732f37acd7684b8c24d14ba3ededfe9fab1ed472`，
  源码 SHA `38cdd5545a62aeb8e17364ec24b245f4cc449aedce99d9b78f1e4be497504918`。
  `core_algos.py` 仍为 `5043f97b87b00ab5c5d907022ea6cf148b935fdb71c37222a3f9c33ccfaf1dc6`。
  两个 actor 共享同一实际模型、原 optimizer、输入和 DT 信号；每组前恢复初值
  及 optimizer/scheduler/RNG，分别执行两次真实更新，不复制 PPO 计算。
- `short-parity-fa-standard.json` 是短链误差界检查的历史文件名：相同 165-token 左 padding、
  447 个有效输入 token、14 个 policy token（batch 4 共 56），DT 实际长度 618，
  配置上限仍为 32768。原 actor 的 FP32 math SDPA 与 HF 原 GDN fallback 是
  独立数值参考；BF16 训练路径不变。PPO loss/clipping、有效 token log-prob、
  原梯度和参数增量均满足借鉴 [FA 2.6.3 测试](https://github.com/Dao-AILab/flash-attention/blob/v2.6.3/tests/test_flash_attn.py) 的本地误差界：
  相对 FP32 的最大误差不超过原 BF16 math 路径误差的两倍。该判据在本次验证
  中延伸到 PPO 的梯度/参数增量。官方检查同一 Q/K/V 的 attention；本地分母
  还包含整网权重精度、其他层和更新累计误差，不能把该界单独作为整条训练
  可靠性的证明。2 是借用的工程常数，不是 PPO 数值误差的理论保证。
  两边读取同一 checkpoint，LoRA 初值和 DT 输入逐元素核对；checkpoint 中
  727 个 BF16、48 个 FP32 参数张量的加载精度差异计入普通 BF16 基线误差，
  不把整个 FP32/BF16 模型说成字节相同。清单 `checkpoint-dtype-inventory.json`。
- `short-parity-comparison.json` 记录关闭裁剪时的零误差对照。
  `short-parity-trimmed-comparison.json`、`short-parity-aligned-comparison.json`
  及 `short-parity-maskfixed-trimmed-*` 的失败保留：前两者采用的 BF16 路径差值
  1x 门槛不等于 FA 的 FP32 参考判据，后者还包含额外的严格零误差连续性检查。
  不覆盖这些历史状态，也不拿默认路径零误差替代真实优化路径验收。
- 1024-action 的精确 32768 容量补充：`dt-capacity-32k-response1024-{reshard,noreshard}.json`，
  均完成四次 DT 和两次非零 PPO 更新。allocated 60.596 GiB，reserved
  62.289/62.250 GiB，总耗时 748.51/743.78 秒。这是显式容量夹具，包含合成
  action/filler，不能当成真实任务。原始 JSON/PT 和各失败记录都保留在远端。
- v3 三任务结果另存 `native-training-v3.json`；Sokoban 第二轮旧作业仅按
  `/tmp/dt-mx-soko-v3` 的确切自有环境变量筛选后停止，182 个进程的清单在
  `soko-v3-stopped-for-parity-fix.json`。Ray 驱动随后显示 aborted 是这次主动
  停止的结果，不是另一次训练失败；其他用户 GPU 0–2 的进程未操作。
  新版本训练的 PID、启动版本和设置以运行根目录 `active-training.json` 为准。

## 连接与加载配置

本机 PowerShell：

```powershell
ssh -o BatchMode=yes -J 4090 -p 2501 liangchen@10.70.5.230
```

不使用本机别名的同一入口：

```powershell
ssh -J chen@e1b4acb6c4c142d5a00d326cec07fb11.hn.takin.cc:10179 -p 2501 liangchen@10.70.5.230
```

2026-09-21 用户更新 A6000 内网地址为 `10.70.5.230`。已通过旧地址记录的
SSH 主机密钥验证并成功登录，主机名 `hit`、账号 `liangchen`；新地址的
known_hosts 记录复用该已验证密钥。该变更只适用于 A6000，不改 3090 记录。

远端 Bash：

```bash
cd /data/liangchen/deltatrace_resume_20260917/repo/deltatrace
source experiments/rl/environments/a6000.env.sh
```

该文件只导出配置，不安装依赖、不编译、不下载、不启动训练。无需新建 venv
或反复激活 conda；直接调用 `"$VENV_PYTHON"`。其他机器应覆盖自己的路径，
不能直接套用此主机的编译产物。

## 固定资源

| 资源 | 已有位置 |
|---|---|
| 运行根目录 | `/data/liangchen/deltatrace_resume_20260917` |
| 远端 DT 源码及 RL 脚本 | `/data/liangchen/deltatrace_resume_20260917/repo/deltatrace` |
| Python 入口 | `/data/liangchen/deltatrace_resume_20260917/env/bin/python` |
| 日志中解析到的 Python 包目录 | `/data/liangchen/conda_envs/mem1_train/lib/python3.12/site-packages` |
| 模型 | `/data/liangchen/models/Qwen3.5-9B` |
| 运行时权威配置 | `/data/liangchen/deltatrace_resume_20260917/environment.json` |
| VERL | `/data/liangchen/deltatrace_resume_20260917/third_party/verl-agent2` |
| WebShop | `/data/liangchen/deltatrace_resume_20260917/third_party/webshop` |
| AppWorld | `/data/liangchen/deltatrace_resume_20260917/third_party/appworld` |
| DT official root | `/data/liangchen/deltatrace_resume_20260917/flashtrace_paper` |
| DT finite FA 库 | `/data/liangchen/deltatrace_resume_20260917/build/libdeltatrace_fa_finite_bf16_d256.so` |
| Triton 缓存 | `/data/liangchen/deltatrace_resume_20260917/cache/triton` |
| Inductor 缓存 | `/data/liangchen/deltatrace_resume_20260917/cache/inductor` |
| Java | `/usr/lib/jvm/java-17-openjdk-amd64` |

远端 DT 目录是同步的源码目录，不应假设它有 Git 元数据。它的 `profiles/`
和 `clean/` 位于目录顶层；本地仓库则在 `deltatrace/` 子目录下。

`environment.json` 已记录模型配置/tokenizer 哈希、native-stage 源码哈希、
finite 库哈希、扩展目录、`dt_dynamic_shapes=true` 和运行变量。
[历史快照](environments/a6000.owner-environment.snapshot.json)来自本地已保存的
`research/temporary/qwen35_fleet_20260917/runtime-receipts-v4/environment.json`，
用于离线查阅；**不要用此历史快照覆盖远端当前配置**。后续兼容性修改可能使
部分源码哈希变化，应记录差异，不能靠重装抹掉差异。

## 版本与兼容性

- VERL 固定基线：`732f37acd7684b8c24d14ba3ededfe9fab1ed472`。
- AppWorld 固定基线：`42b5bcf3cd334fee33f0c37c02070a9f5807add5`。
- 上游来源见 [upstream.lock](upstream.lock)；附加 RL 包版本见
  [requirements-upstream.txt](requirements-upstream.txt)。这些是安装规格，不是
  最新实测的完整 `pip freeze`，不能据此宣称当前环境完全一致。
- 2026-09-21 的模型加载日志显示 Transformers `5.13.0`、Python `3.12`；
  其他已安装包应直接通过下方只读命令查询，不重装来“确认版本”。
- VERL/HF 兼容性修改由 [patch_verl_agent2.py](patch_verl_agent2.py)维护。
  这是打过补丁的上游代码；启动脚本会调用该补丁脚本，但不会执行 pip、
  重新下载权重或重新编译 CUDA 扩展。
- `PARAM_OFFLOAD=False`：当前 FSDP2 路径曾进入只接受 FSDP1 的参数卸载
  helper 并触发 assertion。保留此设置；该错误不意味着环境需要重建。
- 提前加载本配置，让 Ray/模型首次 import 时即看到持久缓存路径和
  `FLA_BOUNDED_NORM_TUNING=1`、`FLASHTRACE_HYBRID_ATTENTION_CPU_CACHE=1`。
  保留已有缓存；首次新图编译与每次从空缓存编译是不同问题。

FlashAttention 的已知情况：主机 glibc 为 `2.31`，曾使用的预编译 wheel 要求
`GLIBC_2.32`，不能再次安装那个不兼容 wheel。已构建并安装的产物位于：

```text
/data/liangchen/deltatrace_resume_20260917/downloads/flash-build2/flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl
```

这是基于 FlashAttention `2.8.3` 源码、修改构建范围后得到的 **BF16、head_dim=256、
causal、sm80 前向专用构建**，不是完整通用 wheel；不能据其前向通过推断 actor
反向已受支持。相关工作目录曾为 `/tmp/flash_attn_build2`，不是持久依赖入口。
后续应优先复用上述 wheel 和已有 finite 库，先定位具体缺失能力，再决定是否修复。

## 三个任务的现成资产

- WebShop 已有 1000 商品数据和 Lucene `indexes_1k`；VERL 内通过链接访问。
  这是 small 配置，不是完整商品库。
- Sokoban 使用现有上游环境；当前 text-only 启动设置为 `tiny_rgb_array`、
  `6×6`、一个箱子。
- AppWorld 已有官方数据。训练使用服务时，查
  `"$APPWORLD_ROOT/appworld_ports.ports"` 及 VERL 目录同名文件/链接，并检查
  对应服务是否存活。文件存在不等于服务存活，不要重复启动第二套服务。
- `prepare_task_assets.sh` 在 AppWorld 数据缺失时会运行安装/下载命令；
  **它不是日常恢复工作的入口**，不能把它当成纯只读检查。

## 恢复前的轻量只读检查

```bash
source experiments/rl/environments/a6000.env.sh
test -x "$VENV_PYTHON"
test -f "$MODEL_PATH/config.json"
test -f "$DT_ENVIRONMENT_JSON"
test -d "$VERL_ROOT/verl"
"$VENV_PYTHON" -c 'import sys; print(sys.executable); print(sys.version)'
"$VENV_PYTHON" -m pip list --format=json
readlink -f "$VENV_PYTHON"
ls -ld "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu --format=csv
pgrep -a -u "$USER" -f 'verl.trainer.main_ppo|appworld'
```

这些检查不加载模型、不占用训练显存。按当前占用选择空闲卡；历史 GPU 1/2/4
都不是预留卡。断线后先确认已有训练/服务是否仍在运行，再发起下一次测试。
若 SSH banner 超时，先排查连接，不创建新环境。

训练脚本路径为 `experiments/rl/run_verl_agent.sh`。当前 DT 适配代码尚未按
[PLAN.md](PLAN.md) 完成核对与修正，故此处撤下旧 `METHOD=dt` 启动示例，
避免将旧算法入口当作固定计划的可用实现。环境复用和上述只读检查仍然有效。

`32768` 是要求的总上下文上限，实际轨迹长度必须从日志另行报告。

## 已验证范围与尚未通过的部分

- [已有任务环境记录](results_env_smoke.json)：WebShop search、Sokoban step、
  本地 AppWorld `help()` 曾通过，支持复用现有资产，不代表多轮 RL 已通过。
- 2026-09-21 独立 owner DT 前向/归因曾运行；它与 PPO actor 反向验收分开。
- 2026-09-21 11:47 的 WebShop DT 联调在 finite decoder 的 `bmm` 报
  `mixed torch.Tensor and DTensor`。后续重试曾打印
  `owner FSDP2 lifecycle modules=34 remaining_dtensor=0`，但目前没有取得其
  最终成功退出记录。恢复时先查该次 Ray 日志，不应从安装依赖重新开始。
- 当时 Ray 日志入口：`/backup01/ztl/tmp/bash/ray/session_latest/logs`；
  后续 session 会改变这个链接，定位后应保留具体 session 路径。
- 旧 `results_counterfactual_32k_a6000.json` 使用旧信用方案，不能作为当前
  token DT 方案成功的证据。

## 2026-09-21 新地址恢复检查

- 已通过 `10.70.5.230:2501` 经 4090 登录，并验证它与旧地址的 SSH 主机密钥相同。
- 已同步本地唯一 PLAN、状态 README、本记录和 `a6000.env.sh`；旧文档快照保存在
  运行根目录的 `receipts/docs-before-sync-*` 下，不是活动方法规范。
- 已实际 source `a6000.env.sh` 并调用原有 Python，没有安装、下载或编译步骤。
  Python 解析位置为 `/backup01/liangchen/conda_envs/mem1_train/bin/python3.12`。
- 已查询安装版本：Torch `2.8.0+cu128`、Transformers `5.13.0`、
  VERL `0.3.1.dev0`、AppWorld `0.2.0.dev0`；VERL Git HEAD 与上述固定基线一致。
- 模型 `config.json` 及两个持久缓存目录存在。已有 AppWorld 7000--7004 进程存在；
  本次未重启服务，也没有据进程存在宣称任务评估通过。
- GPU 占用检查发现所有六张卡都有已有进程。GPU 1/2/5 约各占 1.8 GiB，
  GPU 4 约占 40.7 GiB，其他卡也有进程；这些不是本任务的资源预留。
  未终止这些进程，未启动新的训练。
- `*.sh` 同步时使用 LF。PowerShell 直接将 CRLF 脚本管入远端 Bash 会导致
  `true\\r` 一类解析错误；应修复传输换行，不重配 Python/CUDA 环境。

上述是连接与环境复用检查。当前固定 Q/V 方法的三个任务训练仍未验收。

## 2026-09-22 目标读出与运行记录

- 继续复用上述 Python、权重和缓存，没有安装或下载步骤。
- 新增 categorical target 支持位于正式 `clean/qwen35/qwen35_answer_finite.py`
  与 `qwen35_dense_finite_runner.py`；同步时保留远端已有动态 shape 执行兼容修改。
  修改前源码保存在 `receipts/readout-before-20260922T084710`。
- 真实模型探针及 CPU 检查记录在运行根目录的 `receipts/readout-20260922/`。
  当次使用 GPU 3 前确认其空闲；该编号不构成后续预留。
- 历史生成前后读出使用 actor 的 SDPA 和原始 head；该读出方法已撤下，
  其探针不能证明当前 EOS finite 归因路径通过。
- `VERL_TRIM_SHARED_PADDING=1` 在 HF rollout 和 actor 内移除共有左 padding
  的无效计算。其改动前源码保存在 `receipts/readout-20260922/before-padding/`。
  `patch_verl_agent2.py` 维护该补丁，不重新安装 VERL 或 Transformers。

## 2026-09-22 EOS 事件接入修订

- 当前 producer 调用正式 finite `attribute`，临时通过 HF 公共接口切换到 FA，
  完成或异常后恢复 actor 原后端。FA wheel 仍只用于归因前向/重放，PPO 反向保留 SDPA。
- 显式复用 `environment.json.qwen35.dt_dynamic_shapes` 和 `dt_compiler_options`，
  不重新编译/清空持久缓存作为恢复步骤。对应远端 execution overlay 已存在，
  不用本地基础源码直接覆盖它。
- 本次只同步 RL 适配代码、测试和文档，未安装软件、下载权重、重启 AppWorld 服务。
- 新记录目录：`/data/liangchen/deltatrace_resume_20260917/receipts/eos-events-20260922/`。
  GPU 训练结果以该方法的新日志为准；CPU 验证不能替代。

## 2026-09-22 MetaX 资源检查

- 用户提供入口：`ssh -p 32036 root@ssh.v5000-prod-gw.nhss.zhejianglab.com`。
  不在任何仓库文件或运行配置中存放登录密码。
- 查询时 8 张 MetaX C550，各 65536 MiB。GPU 0–2 有 vLLM 进程；GPU 3–7
  无计算进程、利用率 0%、各约 864 MiB 基础占用。此记录不是预留，启动前重查。
- 权重已存在：`/mnt/si0021787ci2/default/models/Qwen3.5-9B`。
- 已有专用运行根目录：
  `/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_qwen35_20260912`。
  `env/bin/python` 可运行，复用系统 MetaX Torch，不新建第二套依赖环境。
- 该专用环境：Torch `2.8.0+metax3.5.3.9`、Transformers `5.13.0`、
  Triton `3.0.0+metax3.5.3.9`、FA `2.6.3+metax3.5.3.9torch2.8`、FLA `0.4.1`；
  当前 Python 尚未发现 VERL 模块。不能把默认 `/opt/conda/bin/python`
  的 Transformers `4.57.3` 当成这套专用环境。
- 原 `build/libdeltatrace_fa_finite_bf16_d256.so` 的 ELF 依赖是 MetaX 的
  `libruntime_cu.so`、`libmccompiler.so`、`libmcruntime.so` 等，已有平台产物。
- 旧环境 JSON 中路径仍为 `/mnt/geogpt-doc-new/...`，恢复时须核对当前挂载映射；
  路径变化不是重装依据。旧运行目录与其他服务不覆盖、不清缓存。
- 原 runtime 设置含 `MACA_TORCH_COMPILE_CONF=maca.disable_maca_triton_heuristics:1`
  和 `FLA_BOUNDED_NORM_TUNING=1`，动态编译选项也已记录。
- 以上只证明资源与已有环境存在；不证明本轮 RL/DT 已在 MetaX 跑通。

## 2026-09-22 MetaX 复用入口与实测故障

- 当前隔离源码/日志根目录：
  `/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922`。
  源码在 `repo`，运行配置在 `environment.json`，记录在 `receipts`。
  旧 20260912 目录仍提供 Python、有限传播库及持久缓存，没有被覆盖。
- 在新源码目录执行 `source experiments/rl/environments/metax.env.sh`。
  该文件只导出已核验路径和 `MACA_PATH=/opt/maca` 等已有运行变量；不安装。
  原 JSON 的旧挂载路径只在新运行配置中修正，没有修改旧配置。
- 使用前查询 `mx-smi`，再同时设置 `CUDA_VISIBLE_DEVICES` 和
  `MACA_VISIBLE_DEVICES`。本轮探针使用 GPU 3，结束后已释放；没有资源预留。
- 该 Python 尚缺 VERL、hydra-core、tensordict、gym/gym-sokoban、codetiming、
  wandb 和 AppWorld。本轮没有安装这些包。三个官方正奖励轨迹从 A6000 导出，
  在 MetaX 复用原始 token/reward 夹具，只检查模型归因，不冒充当地任务服务或训练。
- 实测 MetaX 编译后的 categorical `FiniteAnswerOps` 触发非法地址；独立小算子
  在 no-grad 下复现。使用同一正式 seed 的 eager 执行可越过该错误。新增显式
  `qwen35.dt_answer_compiled=false`，仅用于此运行配置；owner 默认仍为 true，
  其他有限传播图保持既有动态编译。未复制 seed 算法，也未清编译缓存。
- 修复前的真实 Qwen 探针未通过已有归因守恒阈值。当时记录：原生目标差及 seed
  log-prob 差均为 `0.015408515930175781`，head 输入端有限贡献为
  `0.08401765790792126`，最终 norm 输入端为 `0.08037153781038953`，
  全部 source token 归因和为 `0.05778632130990635`。32 层原生重放 L2 差为 0。
  这将差异定位到 head 有限拉回边界；该项已由下面的舍入边界修复处理，
  后续层的整网误差仍单独记录，不改变 PLAN。
- 该次失败 trace 实际长度 266（保持总上限 32768），含诊断用时约 19.79 秒，
  peak allocated 为 19426050048 字节。这不是 32k 容量、训练吞吐或已通过的
  信用分配结果。首次/后续编译成本尚未完成受控测量。
- `receipts/native-eos.json` 保留失败状态和完整逐层诊断；
  `receipts/metax-runtime.json` 保留版本、路径及源码哈希；
  `receipts/probe_answer_seed.py` 和对应日志保留编译器复现。
  本地结果索引为 `results_eos_events.json`。阈值未放宽，归因未归一化，未启动 PPO 更新。

## 2026-09-22 head 舍入边界修复

- 原生 BF16 head 的 `delta(logits)` 与未舍入线性运算 `W delta(hidden)` 不同。
  在捕获的真实操作数上，单纯提高转置乘法精度仍有约 0.087 的差额，主要来自
  原生输出舍入。正式 head 有限规则现已使用已有 `_secant` 计入该操作，再做
  类别行的 FP32 转置乘法；没有改变模型原生输出或把最终归因重新缩放。
- 所需隐藏行由已有 final norm hook 按正式 target selection 打包，只保留
  目标 predictor 行；不复制完整序列，不增加原生模型前向。默认完整词表
  文本 seed 路径保留；categorical seed 必须提供对应的原生隐藏行。
- MetaX 继续使用 `dt_answer_compiled=false`；A6000 保持默认编译设置。
  没有安装依赖、下载权重、重建缓存或重启已有服务。
- A6000 CPU 回归 54 项通过；MetaX 重跑其中 12 项 head 回归通过。
  A6000 GPU 3 确认空闲后，9 项宽度 4096 的编译/eager 对照通过，最大有限
  边界误差 `1.1920928955078125e-7`。编号仍不代表资源预留。
- MetaX 三任务首个真实 head 检查误差分别为：Sokoban `5.57e-8`、WebShop
  `3.41e-8`、AppWorld `7.52e-8`。A6000 真实 Qwen 首个 head 检查误差约 `1.79e-7`。
  三任务的整网归因仍未通过原有守恒阈值；head 修复通过不等于完整 DT/训练通过。
- MetaX 新记录：运行根目录下 `receipts/head-fix/`，包含捕获的 `head.pt`、
  修改前后诊断、三个任务的独立日志和 `task-summary.json`。
  A6000 新记录：`/data/liangchen/deltatrace_resume_20260917/receipts/head-fix-20260922/`，
  包含 CPU、编译对照及原生整网结果。修改前 owner 源码在其 `before/` 中。
- A6000 仅在已有 owner overlay 上应用相应差分；保留此前去除 head shape hook
  的执行兼容修改。原始隐藏行通过已有 norm hook 获取，避免依赖该旧 hook。

## 2026-09-22 MetaX 训练组件接入（当前入口）

- 用户指定只使用 MetaX；不再在 A6000 启动测试或训练。Goal 保持 active，
  三任务完整 DT→PPO 尚未通过，不能把本节的环境准备记成训练成功。
- 已在原 `deltatrace_qwen35_20260912/env` 补齐缺失组件，没有新建环境。
  固定 VERL 源码：`$DT_RUNTIME_ROOT/third_party/verl-agent2-732f37acd7684b8c24d14ba3ededfe9fab1ed472`，
  editable 安装时使用 `--no-deps`，随后应用现有 `patch_verl_agent2.py`。
  AppWorld 源码同层：`appworld-42b5bcf3cd334fee33f0c37c02070a9f5807add5`。
  路径已写入 `environments/metax.env.sh`，恢复时不要重复下载或安装。
- 新增 hydra-core 1.3.2、tensordict 0.10.0、gym 0.26.2、gym-sokoban 0.0.6、
  codetiming 1.4.0、torchdata 0.11.0、wandb 0.30.0；AppWorld 及 WebShop
  必需包亦已安装。安装时用版本约束保护原 Torch、Transformers、Triton、
  FlashAttention、NumPy、PEFT、Ray、Accelerate 和 torchvision。
- 记录目录 `$DT_RUNTIME_ROOT/receipts/training-setup/` 保存安装前后 freeze、
  保护约束、pip 安装报告、固定源码来源、CPU 回归和 Sokoban 奖励记录。
  42 项既有组合/collector/PPO 回归通过，Sokoban 官方脚本奖励接口通过。
- WebShop 新增 Java 17.0.20；apt 原有索引过期导致 404，更新索引后安装，
  未升级模型依赖。断线后的 dpkg 配置已完成；`java -version` 已通过。
  商品数据下载日志为 `receipts/webshop-assets.log`；首次 Google Drive 下载
  因该主机到 `drive.google.com:443` 的网络不可达而失败。包安装已成功，
  此问题需要取得官方资产，不应重新安装包。之后还需构建官方 1k Lucene
  index。不能运行完整旧 `setup.sh` 重装依赖。
- AppWorld GitHub 源码归档含 Git LFS 指针。必需 bundle 从该固定 commit 的
  GitHub media URL 获取，并核对指针中的 SHA256 和字节数后替换。
  已验证的文件不重复获取；余下安装/数据下载见 `receipts/appworld-remaining-lfs.log`
  及同名 `.pid`，恢复时先查看进程和输出，再决定下一步。任务服务尚未启动。
- 数值诊断及未采用候选位于 `receipts/boundary-audit/`。残差/归一化原生
  舍入候选通过局部检查，但整网系数放大，未合入。当地正式两个 owner 文件
  已恢复 `4df76746`；没有放宽阈值或更改奖励公式。FP32 线性转置仅作为
  诊断比较，未接入训练；完整 DT 数值问题继续处理。
- 用户再次确认 token PPO 目标后，核对了当地实际 `DELTATRACE` 分支直接
  传递 `dt_token_advantages`，actor 使用 `vanilla` token PPO loss。
  新增“调用第二次 GAE 即失败、传入优势逐元素不变”的断言，仍复用原有
  collector→trainer→PPO 梯度测试；该检查 1 项通过，日志为
  `receipts/training-setup/no-second-gae.log`。这不替代尚未通过的整网 DT。

## 2026-09-22 MetaX 三任务资产与官方服务就绪

- AppWorld 官方 `install --repo` 和 `download data --root "$APPWORLD_ROOT"`
  已完成。当前 `data/datasets` 含 train/dev/test_normal/test_challenge；不再下载。
  原本地 worker 执行官方训练答案返回 `10, done=True`。随后直接使用已安装
  CLI 的 `serve environment --num-servers 20 --port -1 --no-show-usage
  --without-setup --root "$APPWORLD_ROOT"` 启动服务，未执行旧 VERL 脚本中的
  全局 kill 或新建 conda 环境。20 个 `/openapi.json` 均通过检查，原远程
  `AppWorldWorker.reset/step` 执行同一官方答案返回 `10, done=True`。
- 服务地址来自官方 CLI 输出 `receipts/appworld-services.log`；端口文件在
  `$APPWORLD_ROOT/appworld_ports.ports`，VERL 同名文件链接到它。服务进程使用
  现有 Python、CPU，不占 GPU；恢复时先检查这份日志中的地址是否仍存活，
  不重复启动。机器/服务重启后端口可能变化，历史端口不构成预留。
- WebShop 原 Google Drive 链接在 MetaX 不可达、本机返回 404。缺失的三个
  JSON 改从公开镜像取得：`HongbangYuan/webshop` 固定 commit
  `0129d4a81dbdb827e76afd20a1e2c38b61098613`，并与
  `sparklabutah/timewarp-env-data/webshop` 逐文件核对 SHA256 相同。
  来源、原 Drive ID、字节数和哈希保存在 `$WEBSHOP_ROOT/data/receipt.json`
  及 `results_eos_events.json`。这是数据镜像核验，不声称重新验证了不可达的
  原 Drive 文件。三个文件已经在 MetaX，不再下载。
- 已使用固定上游 `convert_product_file_format.py` 和官方
  `python -m pyserini.index.lucene` 构建 1k 索引，日志确认 1,000 documents、
  0 errors。`search_engine/indexes` 链接到 `indexes_1k`；原 worker 实际搜索、
  选项点击、购买的奖励为 `[0,0,0,0,0,10]`，最终 done=True。
- 现有 `en_core_web_sm` 已确认可用，没有重复安装。索引启动暴露缺失 `faiss`，
  只补装 `faiss-cpu==1.15.1`，使用已有 protected-constraints，未改变模型栈。
  新增安装报告 `receipts/training-setup/faiss-installed.json`。
- 奖励/服务日志：`receipts/training-setup/webshop-index-reward.log`、
  `appworld-official-reward.log`、`appworld-remote-reward.log/.json`。
  这些使用脚本或训练答案验证环境，不能当作模型任务成功率或 DT 训练结果。
- GRPO 在 MetaX 的独立运行检查保持 minibatch=4、group=4、总上限 32768、
  max_steps=15。首次 Ray 初始化因运行根目录过长超过 107-byte Unix socket
  限制失败；短 IPC 目录改为 `/tmp/dt-rl-mx-20260922`，已写入 env.sh。
  日志为 `receipts/training-setup/grpo-Sokoban-short-ray.log`，尚待训练结果。
- DT 控制检查：同一张空闲卡、同一对输入的三次原生类别 log-prob 完全相同；
  之后仍在第 2 个事件复现 `root=0.10085296630859375`、
  `signed_sum=0.04567181524601055`。`receipts/boundary-audit/repeat-native*`
  及 `controlled*` 保存结果。没有修改有限规则、信用公式或验收阈值。
- MetaX SDPA 在真实 actor 启动时提示没有编译 memory-efficient attention。
  对已安装 FA2 做了模型实际头布局（16 Q heads、4 KV heads、head_dim=256、
  BF16）的前向/反向核验：对 FP32 math SDPA 的前向相对 L2 为 0.001858，
  Q/K/V 梯度相对 L2 为 `[0.002546,0.003051,0.002844]`，全部有限。
  `receipts/training-setup/metax-fa-backward.log/.json` 保存结果。当地 env.sh
  因此默认 `VERL_ATTN_IMPLEMENTATION=flash_attention_2`，复用已安装且具备
  backward 的库；没有新增注意力实现。这不是 32k 训练通过的证据。
- 后续 least-change 局部舍入候选仍未通过整网检查，已撤回并恢复正式 owner。
  候选与诊断保留在 `receipts/boundary-audit/least-change*`；未调整验收阈值。
  `verify_reward_readout.py` 现在也会在整体检查失败时保存已有的单 token EOS
  数值对照；保持原失败状态，额外前向只用于诊断，不作为训练 credit fallback。

## 2026-09-22 原始 DT 数值诊断与训练接口分离

- 以实际部署的 `profiles/qwen35_gdn_symmetric.py` 和 dense runner 为准，
  原生舍入残差是正式接口的诊断输出；`0.02` 阻断此前由 RL 适配层加入。
  按 PLAN 对近似和接口的区分，保留同一阈值及 `conservation_verified=False`，
  原始 signed 向量仍按既定逐事件 expm1 公式接入 Q/V。数值 verifier 仍会
  保存全部结果并以失败退出；这不是修好或放宽了数值验收。
- 三任务当地官方正奖励轨迹已全部完成原生 DT/Q/V 组合。文件为
  `receipts/training-setup/native-{Sokoban,Webshop,AppWorld}-events.json/.log`。
  Sokoban 的 15 次 trace 合计约 30.79 秒：首次 20.43 秒，随后 0.66–0.92 秒；
  WebShop 6 次约 36.00 秒；AppWorld 1 次约 20.20 秒。首次时间含运行初始化
  和编译，不应当逐次相乘估计稳态卡时。实际上下文远短于 32768，上限未改。
- 三任务非零 token 优势数分别为 65、46、341；Q−V、mask 和有限值检查通过。
  原守恒审计分别失败 6/15、3/6、1/1；这些失败保留在结果中，不能作为精度
  或真实模型训练验收通过。相关 38 项 CPU 回归通过，日志
  `receipts/training-setup/audit-contract-tests.log`。
- 原 HF rollout 的 micro_batch_size=1 会串行生成 16 条 rollout；当地入口
  使用其已有参数 `ROLLOUT_MICRO_BATCH_SIZE=4`。GRPO 当前运行记录
  `receipts/training-setup/grpo-Sokoban-fa-b4.log`；此前 b1 运行已主动中止，
  不据其未完成状态声称吞吐比较或参数更新成功。
- 更正：固定 VERL 读取 `ray_init.num_cpus=8`（环境变量
  `DT_RAY_NUM_CPUS`），避免按宿主全部 CPU 预启动大量 worker。
  之前用 `+ray_kwargs.ray_init.num_cpus` 只证明 Hydra 接受了新增键，实际 trainer
  并不读取它。当前键在原 config 中已存在，不加 `+`；用真实 Hydra compose
  和原 `run_ppo` 验证它最终传入 `ray.init`，不再将配置显示当成生效证据。
- 同时启动多个任务会同时写固定上游补丁，实际复现了 shared-padding anchor
  检查失败。启动脚本现用系统 `/usr/bin/flock` 对
  `$VERL_ROOT/.deltatrace-patch.lock` 加锁，串行调用原补丁程序；单次串行复核
  通过，日志 `receipts/training-setup/serialized-patch-check.log`。没有复制
  补丁算法或重新安装库。并行任务使用各自的 Ray 短目录和已有数据目录。

## 2026-09-22 生成 token 配置修复与训练采样

- AppWorld、WebShop 模型生成提前结束后，上游 HF rollout 补齐 response 时
  复现 `NoneType * Tensor`。固定 VERL 的 FSDP worker 仅判断
  `GenerationConfig` 对象是否存在，没有判断其 `pad_token_id` 字段是否为空。
  原模型目录没有独立的 generation_config.json。已在该元数据拥有者处修复：
  EOS/PAD 未设置时使用原 tokenizer，明确配置的值（包括 0 和 EOS 列表）保留。
  8 项原上游元数据/补齐接口测试通过（17.80 秒），日志为
  `receipts/training-setup/generation-fix-tests.log`。没有重新安装模型或库。
- 10 秒 py-spy 生成采样中，主线程 874 个样本有 282 个包含 FSDP 栈。
  文件 `receipts/training-setup/grpo-generation-profile.json` 及 summary.json。
  这只是 CPU 栈采样，不能当作 GPU 总耗时比例或实测速比。
  当前通过上游已有 `fsdp_config.reshard_after_forward=False` 测量重复收集
  参数的开销；启动器提供 `FSDP_RESHARD_AFTER_FORWARD`，通用默认值仍为 True。
  显存/速度验收未完成，不提前把这个候选配置写为当地最优设置。
- 原进程使用了缺失 EOS/PAD 的旧元数据；已中止尚未完成的 Sokoban/GRPO
  运行，保存原日志。修复后的三任务运行记录为
  `receipts/training-setup/dt-{Sokoban,Webshop,AppWorld}-tokenids.log`。
  另有 `grpo-capacity-32k.log`，显式使用 31800 个 filler token、max_steps=1，
  只检查长输入资源容量；不是原始任务长度、完整任务训练或成功率实验。

## 2026-09-22 LoRA 线性边界核查与容量候选

- 对已安装 PEFT 的实际 Linear 做非零适配器输入梯度核对，确认 `.weight`
  仅返回 base 权重。旧 DT 读取方式在该边界的最大误差为 1；使用 PEFT
  `get_delta_weight` 提供的增量后误差为 2.22e-16（FP64 边界检查）。
  记录 `receipts/training-setup/peft-linear-contract.json`。
- 修复候选保存在 `receipts/training-setup/peft-linear-candidate/`；完成下述
  原生 Qwen 检查后已同步到 active owner，旧文件保存在 `peft-linear-before/`。它直接读取 PEFT 的 delta，不复制
  LoRA 公式或 merge/unmerge 参数。base/delta 分别走已有线性转置，避免
  先加到 BF16 base 时把小更新舍入消失；这会增加适配器相关矩阵计算，
  其整网开销仍需测量。当前只接入训练所用 vanilla LoRA。
- 原 head 及新增原生/编译线性边界测试 17 项通过（25.28 秒），包括适配器
  激活/关闭/合并状态和原权重不变。日志 `peft-linear-tests.log`；同模型
  非零 LoRA 的正式 DT 检查见 `peft-native-Sokoban.log/.json`，不是模型训练。
- `reshard_after_forward=False` 的 32k 长输入候选实际板卡占用达到
  55749 MiB（包含设备基础占用），已经超过 48 GiB 目标，主动停止，
  不能报告为合格的省显存配置。长输入首次因 prompt 超出 5 tokens 被
  原接口拒绝，filler 改为 31750；总上限和 minibatch 均未改。
  下一项使用上游既有 `ACTOR_STRATEGY=fsdp, HF_FSDP_WRAP=True`，
  日志 `grpo-capacity-32k-fsdp1.log`。仍只属于显式扩长的容量检查。
- 非零 LoRA 的完整原生 Sokoban Q/V 组合已经完成：15 次正式 DT、65 个
  非零 token 优势，约 39.47 秒（首个 27.03 秒，其余 0.77–1.18 秒），
  peak allocated 19792867328 bytes。原数值审计仍有 8/15 项失败；不把
  接口修复说成整网精度通过。该夹具修改了适配器权重，与基础模型结果
  不是同一受控速度/精度对照，不能从两者的差值声称收益。
- FSDP1 的长输入测试已完成 rollout/旧 log-prob，进入 actor 更新后失败：
  `Cannot writeback when the parameter shape changes`，期望扁平
  `[1017118720]`、实际 `[248320,4096]`。这是 FSDP 生命周期错误，不能
  当作容量或 PPO 更新通过。当前容量运行回到 FSDP2、重分片开启、rollout
  microbatch=2；PPO minibatch/group 仍为 4，context cap 仍为 32768。
  日志 `grpo-capacity-32k-b2.log`，结果尚待完成。
- 当前 Qwen tokenizer 的默认模板以未闭合 `<think>` 开始生成。实际 Sokoban
  rollout 的多轮行动有较多无效项，尚未确认原因。启动器现可显式传入
  `ENABLE_THINKING=False`，直接透传上游已有
  `data.apply_chat_template_kwargs.enable_thinking`；未设置时保留默认行为。
  独立对照运行 `dt-Sokoban-nonthinking.log`；不改变 DT 信用公式、token mask
  或 PPO。默认思考配置的三个任务继续保留各自运行与日志，不能提前宣称
  关闭思考改善了成功率或吞吐。

## 2026-09-22 FSDP1 上游行为对照与参数恢复修复

- 原错误已在真实 Qwen3.5-9B、原 VERL worker 的 eval log-prob→train 更新
  转换中复现，不依赖 DT 或生成。原因是本地补丁强制 LoRA 使用
  `use_orig_params=True`；有上游 LoRA auto-wrap policy 时原设置为 False。
  只恢复这一项即可连续完成两次非零更新。修复保留无 auto-wrap 的旧混合
  requires_grad 配置；本次不将该未测试路径宣称为通过。FSDP2 没有改动。
- 正式修订在有 auto-wrap policy 时使用上游 False；兼容补丁可将旧 True
  改动迁移，重复运行保持幂等。未改 Torch writeback、张量形状或参数存储。
- 远端 `core_algos.py` SHA256 与固定上游文件相同；`update_policy` 的 AST
  也相同。对照记录 `active-ppo-provenance.json`，结果索引
  [results_fsdp_owner.json](results_fsdp_owner.json)。没有改 PPO loss、Q/V、mask。
- 旧错误：`owner-update-before.json/.log`；仅恢复上游配置：
  `owner-update-upstream.json/.log`。修复后独立运行：`owner-fixed.json/.log`。
  普通内核的独立运行旧 log-prob 相同，但更新后权重的逐位比较失败，未放宽
  阈值。随后在同一 GPU 5 使用 `PYTHONHASHSEED=0`、
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`、`FLASH_ATTENTION_DETERMINISTIC=1` 和
  Torch `use_deterministic_algorithms(True)`；参考与修复版的初值、旧 log-prob、
  两次更新后的全部可训练权重完全一致，rtol=atol=0。
  记录 `owner-{reference,fixed}-deterministic.json/.pt/.log`；确定性配置只用于
  这项对照，没有悄悄改变训练默认设置。20 项 owner 接口回归通过（14.27 秒），
  日志 `origparam-owner-regression.log`。
- `owner-fixed-long.json/.log` 使用显式合成长 prompt、512 response 槽位、
  总宽度 32768、minibatch=4、microbatch=1，执行原 worker 两次非零更新：
  梯度范数 0.7421875、0.72265625，1,448,885 个可训练参数元素发生变化，
  peak allocated 33.080 GiB / reserved 36.160 GiB。输入动作和优势是明确的
  测试夹具；这是长输入更新验证，不是自然任务长度或 DT 学习有效性证明。
- 原 32k FSDP1 Ray rollout 配置已正常完成：
  `grpo-capacity-32k-fsdp1-fixed.log/.json`。实际 prompt 32211，response
  最大 512，minibatch/group=4，rollout microbatch=4；allocated 37.631 /
  reserved 38.834 GiB。生成 467.789 秒、旧 log-prob 62.091 秒、actor 更新
  253.903 秒，整个训练 step 783.927 秒（不含启动）。这批 GRPO 优势为零，
  只证明原报错路径与容量检查通过；非零参数更新证据来自上一项。
  另一个 FSDP2、rollout microbatch=2
  对照已走完 rollout/log-prob/update，allocated 33.866 / reserved 36.389 GiB；
  其 GRPO 优势为零，不能把这次容量检查当作有效任务学习。

## 2026-09-22 精确 32k DT 边界与 FSDP2 生命周期

- 记录目录仍为 `$DT_RUNTIME_ROOT/receipts/training-setup/`，无新环境或缓存。
  `dt-capacity-exact32k.json` 记录 reshard=True 时 replay 后权重又变回
  DTensor 的失败；`dt-capacity-exact32k-noreshard.json` 记录保留整模型参数
  时的真实 32k OOM。均保留，未改成通过。
- 修复只在原 runner 的层边界调用 actor 公共 `unshard()` / `reshard()`；
  参数数值、前向、有限规则及 PPO 均不改。正常和异常退出仍统一 release。
  旧适配层保存为 `pre-layerwise/deltatrace_rollout.py`，供数值对照使用。
- `verify_dt_owner_lifecycle.py --previous-adapter <上述文件> --context-length
  16384 --output <JSON>`：同一个真实 Qwen actor，LoRA B 注入非零测试值，
  旧/新各执行三次，第一轮不计入速度对比。其余中位耗时 40.521/40.698 秒，
  allocated 54.187/39.806 GiB，完整 signed attribution 和 Q/V/A 零容差
  一致。结果 `dt-lifecycle-speed16k.json`；短原始夹具结果另存
  `dt-lifecycle-comparison.json`，不能用其冷/热耗时宣称提速。
- `verify_dt_context_capacity.py --output <JSON> --artifacts <PT>`：FSDP2
  reshard=True、LoRA r=1/alpha=2、actor minibatch=4/microbatch=1；四次真实
  DT 输入各 32768 有效 token，32769 明确拒绝。actor 输入 32597，含官方
  脚本夹具的 14 个 action token；其余增加的是明确的前文容量 filler。
  不把该 fixture 当作自然任务 rollout 或 512-token 生成速度。
  四次读出为 157.03/131.63/131.67/131.68 秒，两次原 PPO
  更新梯度范数 2.4375、1.0234375，改变 1,448,475 个可训练参数元素。
  总过程 717.22 秒，allocated/reserved 60.627/62.328 GiB，无 OOM。
  原始 owner ledger 保存在 `dt-capacity-exact32k-layerwise.json`。

## 2026-09-22 聊天停止边界与完整生成成本

- 当前 checkpoint 的 `text_config.eos_token_id=248044` 是 `<|endoftext|>`，
  官方 tokenizer 的聊天 EOS 是 `<|im_end|>`（248046），且本地与该模型主分支
  都没有 generation_config.json。实际 rollout 曾在一次动作后继续生成伪造的
  user/assistant 对话。仅在固定 worker 的 Qwen3.5 模型配置回退路径加入
  tokenizer EOS，同时保留原 EOS；不修改共享模型文件，不套用其他模型的
  温度/top_p 等参数。显式 generation config 和其他模型路径不变。
- `chat-stop-regression.log`：11 项通过，含实际 tokenizer、HF EosTokenCriteria、
  VERL get_response_mask 及既有 padding 接口；真实权重速度另由
  `verify_fsdp_generation_cost.py` 测试。该脚本复用原 worker，使用公共 FSDP
  setter 对比相同长输入与生成结果，不替代训练或任务成功率评估。
- 官方来源核对：[文件树](https://huggingface.co/Qwen/Qwen3.5-9B/tree/main)、
  [tokenizer 配置](https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/tokenizer_config.json)、
  [模型配置](https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/config.json)。未采用其他
  用户的模型 PR 或其他模型的采样参数配置。
- `fsdp-generation-cost32k.json`：同输入 32256+512、batch=4，True/False 的四次
  生成逐 token 一致；预热后分别 93.355/77.174 秒，peak reserved 为
  44.588/59.293 GiB。默认分层释放的 DT 修复在 False 下也完成精确 32768
  输入的四次 DT 与两次非零原 PPO 更新（14-action 脚本容量夹具），记录在
  `dt-capacity-exact32k-layerwise-noreshard.json`；1024-action 上限测试另报。
- `ray-config-regression.log`：1 项通过，读取实际 launcher 的 CPU 参数键，
  经原 Hydra 配置和 run_ppo 验证传给原 ray.init 的 num_cpus=8。旧错误键的
  Hydra 配置可显示但从未被 trainer 使用，故以前大量 idle worker 仍被启动。
- 单独容量脚本也必须复用训练入口的 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。
  默认 allocator 下曾出现 32k OOM，仍有 4.35/11.32 GiB reserved 未分配；失败
  保留为 `dt-capacity-noreshard-default-allocator.*` 与
  `fsdp-generation-cost32k-old-stop-default-allocator.*`。配置已写入 metax.env.sh，
  不清空编译缓存，不通过重装处理碎片问题。
- 旧 v2 三任务的第一轮和 worker DT 报告保存在 `pre-chat-stop-v2.json` 及对应
  `.log`。旧作业只按确切自有 RAY_TMPDIR 停止；现有 AppWorld 服务和其他用户
  的 GPU 作业保留。运行根目录 `active-training.json` 才是最新作业记录，
  receipts/training-setup 下同名文件仅是更早历史快照，不能用旧 PID 判断占用。
  本地结果及原记录 SHA256：[results_dt_context_capacity.json](results_dt_context_capacity.json)。
- 上述独立 worker 验证需设置空闲 CUDA_VISIBLE_DEVICES，并以单进程环境
  `RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=<空闲端口>`
  运行，另设 `DT_TASK=Sokoban DT_MAX_STEPS=15 DT_MAX_LENGTH=32768`。
  数值对照使用 `FLASH_ATTENTION_DETERMINISTIC=1`、
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`、`PYTHONHASHSEED=0`；不修改训练默认值。
- `lifecycle-interface-regression.log`：50 passed；
  `observations-regression.log`：4 passed（与固定上游原始 manager 源码对照）。
  原始 manager 保存于 `pinned-env_manager.py`，测试使用环境变量
  `PINNED_ENV_MANAGER_SOURCE` 指向它；不要求远端安装 Git。
- 三任务新运行使用 `full_chat_observations=True`；DT 配置 entropy_coeff=0，
  use_invalid_action_penalty=False，官方事件奖励保持原值。使用原
  `trainer.rollout_data_dir` 保存可检查的 rollout，不另建日志器。
  新配置为 TRAIN_SIZE=1/GROUP_SIZE=4/MINI_BATCH_SIZE=4、MAX_STEPS=15、
  MAX_RESPONSE=1024、MAX_TOTAL_TOKENS=32768、TOTAL_EPOCHS=2、
  ENABLE_THINKING=False、FSDP_RESHARD_AFTER_FORWARD=True、DT_RAY_NUM_CPUS=8。
  4 号卡 Sokoban、5 号卡 WebShop、6 号卡 AppWorld；日志分别
  `dt-{Sokoban,Webshop,AppWorld}-v2.log`，Ray 目录
  `/tmp/dt-mx-{soko,shop,app}-v2`。这是启动记录，不是完成声明或卡号预留。
  旧作业结果保存在 `superseded-*-reports.json` 与 `superseded-*-worker.log`；
  只停止已核对 RAY_TMPDIR 的本任务进程，其他用户的 GPU 0–2 作业未操作。

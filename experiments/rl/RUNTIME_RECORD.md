# 当前运行版本与修复记录

本文件记录运行与修复事实，不定义信用方法；信用规范仍是 [PLAN.md](PLAN.md)。
最近部署核对于北京时间2026-10-02 01:53；精确采集时间见
[current_runtime.json](current_runtime.json)。阶段会继续推进；本文件记录该次核对结果。

固定编号对应：DT发布`c9cd147`、DT数值参考`fc2e6c2`、VERL官方提交`20bd331`、
输出头/B4修复`dc4e4d7`。完整40位提交号和文件SHA保存在快照；记录文档的Git提交
不是新的算法版本，也不表示远端执行了该提交的全部文件。

SQL旧作业在原生mcTracer结束附加采样后退出，已按退出前实际生效的padding版本
冻结重启；没有可恢复的正式检查点。AppWorld旧作业完成step4后，已通过原生恢复部署
生成上下文修复。**三个任务的实际代码组合并不相同**，具体见下表；历史段落不能代替
当前PID绑定的完成回执。

每项修复按“代码提交 → 实际文件SHA → 原测试及适用范围 → 部署路径/PID/时间 →
替代的旧版本”对应记录。已验证、已准备、已部署和已退役分别标明；未提交完成回执
的运行时修改不计入生效版本。原回执和冻结源码保留，不靠目录名或日期推断。

05:25的原只读复核没有发现源码漂移。05:39的SQL采样事故及随后重启保留在历史账本。
当前三组有效actor均为`1f862e8bbdaa…`；TextCraft旧冻结文件`2b80b938fee4…`已由
19:28完成的worker覆盖替代。不能把旧SQL完成步数带到新进程，也不能按TextCraft旧launch重启。

## 哪份记录回答哪个问题

| 要确认的事实 | 原始事实来源 | 本机入口 |
| --- | --- | --- |
| 当前三组PID、创建时间、卡号、预算、日志和检查点 | 远端根目录 `active-training.json`，其中 `manifest` 指向当前正式清单 | `current_runtime.json` 的 `jobs`；重新采集后才称当前 |
| 启动时到底用了什么文件 | 各作业 `source.json`、`launch.json`，冻结entry/VERL目录 | 快照中的路径、SHA256、`entry_files` 和 `owner_files` |
| 启动后在原worker上应用了什么 | 对应PID/rank的原RPC完成回执 | `runtime_overrides`，包括生效代码路径和再次核对的文件SHA |
| 已准备的哪个版本将用于后续恢复 | 冻结候选的 `prepared.json` 与对应测试回执 | `prepared_versions`；含替代关系、逐文件SHA及当前清单中是否存在该entry |
| 哪些数值对照曾经通过 | [verified_runtime.json](verified_runtime.json) 中的固定数值基线与原回执；后续改动另见下表 | 基线不能冒充整个当前部署；最新输出头对照见 [results_actor_b8.json](results_actor_b8.json) |
| 哪次提交解决了哪个问题 | 下方修复表及对应代码/测试 | Git提交是修复来源，不能代替远端已部署证明 |

远端根目录：`/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922`，
SSH端口30821。当前只运行SQL、AppWorld、TextCraft三组DTPO；GPU6/7不再提交GRPO。

## 当前代码组合

| 部分 | 固定来源及实际组成 | 已验证的范围 |
| --- | --- | --- |
| DT数值核心 | release `c9cd147`，数值参考revision `fc2e6c2`，源文件逐项SHA在基线及快照中 | 保留既有FA/FLA实际dtype、原参考和原断言；本次输出头修复没有改DT传播、Q/V/A或容差 |
| PPO训练所有者 | VERL-agent `20bd331bdbc9026a5668e11362178e10ab7400c8`，各任务冻结候选目录 | 原loss、optimizer、update；仅信用及已记录的接口/显存补丁。不是未经修改的官方仓库 |
| Actor输出头 | 原VERL `FusedLinearForPPO`和原Qwen torch wrapper；`dc4e4d7`中的 [patch_actor_fused_head.py](patch_actor_fused_head.py) | 原 `test_linear_cross_entropy.py` 指定用例及实际BF16/FP32、temperature=1/.9对照；不称整网PPO官方容差验收 |
| vLLM | v0.15.0 / MetaX `0.15.0+g24fb31.d20260310.maca3.5.3.20.torch2.8`，沿用已验证的两处owner修复 | [results_vllm_owner_alignment.json](results_vllm_owner_alignment.json) 的原模型/批量/sleep测试；不另造scalar阈值 |
| SQL环境/配方 | SkyRL固定官方SQL配方与官方环境，配置revision `7d94ccf`，数据源回执在数据目录 | 原轨迹、环境、奖励；既有VERL/vLLM承接训练/生成，不启动SkyRL训练器 |
| TextCraft环境/配方 | AgentGym-RL `82402a99c62a293735a3f412fb8ac9a600673bc0` | 原完整rollout、模板、状态机、mask和评测；引擎依赖注入到现有VERL/vLLM |
| AppWorld环境/配方 | LOOP `f14107a976e5793990329d3193df4742076c5a1d` 训练基准 | 原sampler、runner池、完成比例、取消与奖励；completion接口接现有VERL/vLLM，不运行LOOP训练器 |

`v4/v6/v7/v8` 是部署目录编号，**不是DT或VERL提交号**。三组可以使用不同
冻结目录而复用同一数值核心；目录名、提交时间和“最新”字样都不能替代文件SHA。

### 三组实际部署

目录均相对于远端根目录；完整SHA、创建时间、原source/launch回执在快照中。

| 作业 | 当前PID / GPU | 实际执行的版本 | 生效边界与旧版本 |
| --- | --- | --- | --- |
| SQL-DT | 552842 / 0,1 | entry=`runs/sql-padding-restart-20261001/sql-entry`；VERL=`candidates/official-verl-20bd331-sql-padding-20261001`；actor SHA `1f862e8bbdaa…` | 05:44以`bc68687`恢复脚本启动；完成步数见带采集时间的快照。旧PID1876409在mcTracer附加采样后终止，无正式检查点；旧更新不计入新进程。新目录冻结旧进程已生效的padding覆盖，任务参数与初始评估未改 |
| AppWorld-DT | 2479539 / 2,3 | entry/VERL=`candidates/appworld-rollout-scope-20261001/{entry,verl}`；保留head `dc4e4d7`、DT分发 `4c0cbdd`、请求分发 `64e6377`、padding `1f862e8bbdaa…`，新增scope `2a32d00` | 18:30由原提交助手从PID285580的完整step4恢复；启动源码`6e8fbe9`。原loader与双rank加载记录已确认，已进入正式采样；旧step4不是新更新 |
| TextCraft-DT | 212110 / 4,5 | entry/VERL仍为v7；有效actor已由原RPC绑定为`1f862e8bbdaa…`；head继续使用`dc4e4d7`的PID绑定覆盖 | 19:28双rank完成padding接入，原优化器各64步、模型/配置不变；DT分发与生成上下文尚未部署。冻结v7启动命令仍是旧microbatch1，不能直接按旧命令重启 |

两种actor完整SHA：

- 旧forward：`2b80b938fee442ea5d9273523b6cce2bcc0511b7729aa6cea90333fef5de7cd3`。
- padding修复：`1f862e8bbdaad6fa116d0670772ad41269529a3a1e4a5b1eb383352d0372e9bd`。
  源码封存提交`0c80b41`，对应原VERL padding断言证据提交`44e1149`，
  详细范围见[原对照回执](results_actor_response_padding.json)。该回执中的
  `not_deployed`描述测量当时；后续部署以本节和PID绑定完成回执为准。

SQL完成回执为`receipts/owner-b8-dispatch-20260930/actor-response-padding/sql-live/complete.json`。
TextCraft完成回执为同级`textcraft-live/complete.json`；两rank实际方法、旧/新SHA和
原优化器步数见[原回执](../../research/temporary/rl_upstream_alignment_20260929/textcraft-padding-live-20261001/complete.json)。
AppWorld启动`source.json`记录原检查点、旧PID、提交代码及脚本SHA；新worker不继承旧PID回执。
本次原回执内容、远端文件SHA及实际日志统一保存在
[部署转换记录](../../research/temporary/rl_upstream_alignment_20260929/deployment-transitions-20261001/observed.json)。

当前SQL padding已写入新冻结owner，不再依靠旧PID的RPC覆盖；不重复提交该RPC。旧AppWorld仅分发修复候选
`appworld-balanced-resume-20261001`从未启动，已被当前组合候选替代，不能误用。

### 原trainer实际计算的训练规模

2026-10-01核对当前TaskRunner原日志，原配置均未调整。下列是原trainer打印的
`Size of train dataloader`和`Total training steps`，快照的`native_training_workload`
保存具体日志路径。这里的step是一轮采样加训练，不是一次B4反向或一次优化器更新。

| 任务 | 原dataloader批数/epoch | 正式采样/训练迭代总数 | 原优化器更新单位 |
| --- | --- | --- | --- |
| SkyRL-SQL | 2 | 30 epochs × 2 = 60 | 满批1280条轨迹，global mini1280，PPO epoch1：1次更新/迭代 |
| AppWorld | 1 | 200 | 按实际完成轨迹数、global mini32、PPO epochs2计算；首批224条对应14次更新 |
| TextCraft | 11 | 30 epochs × 11 = 330 | 满批256条轨迹，global mini64，PPO epoch1：4次更新/迭代 |

每卡实际microbatch4仅规定原更新内部的处理批量；不能据此把global mini改成4或8，
也不能把TextCraft的30个epoch写成只有30次正式采样迭代。

### 官方负载到实际入口的回归对应（2026-10-01）

此前配方、launcher、轨迹运输和部署记录分别通过，容易漏掉它们之间的单位换算。
已将这条链补成CPU回归：执行作者原脚本/README配置，调用项目实际正式launcher，
再执行固定VERL原worker的两条minibatch归一化语句。环境重复由`env.rollout.n`负责；
engine的`rollout.n=1`，不能再次把完整轨迹重复扩增。

| 已修复的不一致 | 代码来源 | 当前验证与部署 |
| --- | --- | --- |
| 每轮response误作PPO训练行，放大更新数 | SQL `37938d0`，TextCraft `cb8e569`，AppWorld `cf145b2` | 三组走作者完整轨迹；DT response只作归因运输，通过索引回到同一轨迹token |
| prompt单位的mini未换算为trajectory单位 | SQL `37938d0` / `launch_sql_native.py` | 官方256组×5与global mini1280对应；VERL单卡mini640，满批1次联合更新 |
| TextCraft按未使用的配置值多跑一遍PPO | `cb8e569` / `launch_textcraft_native.py` | 作者实际update_policy不读取ppo_epochs；保留单遍，global mini64，满批4次联合更新 |
| AppWorld的采样/完成/取消与作者不同 | `cf145b2`，请求运输 `64e6377` | 复用LOOP原sampler/runner池；40×6请求、原完成规则、mini32、2遍、200迭代 |
| 已部署micro4/head/padding未写进旧launch | `dc4e4d7`，padding源`0c80b41`，原比较`44e1149`，部署`3a46cec` | 三组实际B4与LoRA8/16有完成回执；TextCraft完整恢复候选固化这些设置，旧launch保持历史原貌 |

新增测试是[SQL/TextCraft负载对照](test_official_workload_local.py)及
[AppWorld负载对照](test_loop_model_entry_local.py)。负载文件7项（新增SQL/TextCraft各1项，
其余为已有单位回归）、新增AppWorld 1项通过；
测试代码固定为`1a1e945`；最终独立回执中，两组进程分别7.359/11.250秒，
采样RSS峰值146.7/254.3MiB，物理显存前后均0MiB。
这是配置与原单位换算检查，不替代DT、PPO或vLLM的数值验收。
AppWorld初次缺少本机APPWORLD_ROOT；测试夹具使用作者原dev清单补全临时布局后通过，
初始失败与最终结果均保存在
[workload-regression-20261001](../../research/temporary/rl_upstream_alignment_20260929/workload-regression-20261001/summary.json)。

**尚未全部对齐的是执行效率路径。** SQL/TextCraft的整段rollout上下文与DT跨卡均衡
候选已验证、尚未部署；AppWorld的整批同步返回仍导致已完成请求等待尾部。
AppWorld已新增只处理跨卡返回顺序的候选，原接口CPU测试9项通过，已冻结但未部署；
其版本与初始失败见下方2026-10-02记录。不能将候选测试通过写成正式训练已提速。
当前证据不足以称三组整体耗时已达到官方同等计算负载的水平。不能缩小任务预算、
改minibatch或部署未对照的异步实现来消去这些待完成项。

已核对的head文件SHA前12位（完整值见快照）：

| 文件 | SHA前缀 |
| --- | --- |
| `verl/utils/experimental/torch_functional.py` | `e285c3353bdd` |
| `verl/models/transformers/monkey_patch.py` | `3c78654e0eba` |
| 原 `verl/models/transformers/qwen3_vl.py` | `ebc52fb35812` |

## 修复账本

### 2026-10-02：跨卡返回候选与版本封存

此前负载单位修复没有同时消除执行路径的额外等待，而且冻结启动文件、PID绑定覆盖、
未部署候选容易被当成同一个版本。当前记录将这三种来源分别保存；新的恢复准备从
实际作业及完成回执继承，不从旧目录名、最新Git提交或历史launch推断。

AppWorld候选仅改 `LoopOwner.collect_native_trajectories` 的回复运输：由原Ray
`ActorPool`负责排队与完成顺序，调用同一VERL的
`RayWorkerGroup._execute_remote_single_worker`，原 `ObjectRef.future` 唤醒现有队列。
官方LOOP sampler、runner、完成阈值、取消事件、token artifact、环境reward、原PPO和
数值代码不变。原先测到快卡完成后还等待另一卡39.301秒；候选针对这个跨卡边界，
**不声称已消除一张卡内部的批次尾部，也没有正式迭代提速结果。**

| 对应版本 | 文件/验证事实 | 部署状态 |
| --- | --- | --- |
| 原AppWorld bridge | SHA `22ff649007bf6d98584e6ebb1d19f72cd6010d68e6f40ac082418ee50ee0dce5` | PID2479539仍在执行，检查点5已由原loader保存 |
| 初始候选 `17dc9ef` | 原Ray/VERL RPC的7项CPU运输检查通过；尚未覆盖排队后取消与原生成器B0 | 未部署，不能替代完整9项结果 |
| 空批修复 `ff28f35` | 候选SHA `8e93ec140303a88e35d8ff2a9a50aa1fc8012215e979bb9d8d5967da2a95b030`；仅在运输为空时以原Ray对象完成，不调用不支持B0的原VERL生成器 | 不进入默认入口 |
| 完整CPU回执 `33b074e` | 9项通过，原Ray2.53.0 ActorPool、原VERL RPC；73.800秒进程墙钟、采样进程树PSS峰值2.510GiB、Ray GPU资源0 | 接口验证，不是模型数值或正式速度验收 |
| 冻结准备 `7140cc4` | 实际entry只替换上述bridge；原VERL目录及全部其他entry、DT、输出头SHA保持；原恢复助手增加对应回执字段 | `candidates/appworld-rank-completion-20261002/entry`，prepared-only；尚未切换PID |

完整测试与准备记录见
[appworld-rank-completion-20261002](../../research/temporary/rl_upstream_alignment_20260929/appworld-rank-completion-20261002/prepared.json)
和[最终CPU回执](../../research/temporary/rl_upstream_alignment_20260929/appworld-rank-completion-20261002/final-cpu-tests.json)。
原始日志/XML及失败尝试一起保存：`367cb7f`的 `inspect.unwrap` 未去掉原日志器，
CPU检查仍调用GPU显存API；`25319d1`绕过该日志器后实际复现原B0的
`max() iterable argument is empty`，由 `ff28f35`修复运输边界。没有改生成器、隐藏失败、
放宽数值容差或修改官方空批约束。初始7项文件是历史证据，只有最后9项回执对应最终候选。

现有原恢复提交助手逐项核对准备回执的entry/owner/DT SHA、旧PID创建时间和原
`latest_checkpointed_iteration.txt`及data/model/optim/extra_state文件，才交给VERL loader。
新的完成运输版本也记录到 `source.json` 和带采集时间的快照。原checkpoint之外不补造
reader状态，不把历史回执当新PID完成证明。SQL仍无完整检查点；TextCraft原保存边界25
尚未到达，这两组的效率候选仍未部署。

### 2026-10-01 20时：AppWorld正式生成的原生profile与整批等待

本次只观察当前PID2479539已经在跑的请求，没有新增轨迹、引擎、模型或GPU任务。
`3e8d6f7`经原`collective_rpc`注入vLLM原`ProfilerConfig/TorchProfilerWrapper`，
由原worker的`max_iterations=16`自动停止；每rank仅一个现有29请求批次。
原GPU annotation给出3个包含context的步骤、13个只有generation的步骤。
两个rank均已清理profiler并恢复原generate绑定。原trace保留在远端，路径与完整SHA见
[原生profile汇总](../../research/temporary/rl_upstream_alignment_20260929/appworld-rollout-scope-20261001/native-profiler/summary.json)。

| 16步局部profile，秒 | rank0 | rank1 |
| --- | ---: | ---: |
| GPU kernel区间并集 | 5.065 | 5.575 |
| GPU memcpy区间并集 | 0.058 | 0.061 |
| CPU `aten::copy_`各线程时间直接求和 | 8.734 | 9.737 |
| 同一CPU copy事件跨线程合并后的区间 | 4.408 | 4.914 |

这些行存在重叠，不能相加。CPU copy包含等待，并且多个线程会等待同一段GPU计算；
不能把8–10秒解释成实际搬运时间。此处真正的GPU memcpy约0.06秒，FA/FLA和矩阵乘法
均出现在原trace中，本次没有依据去改这些数值内核或删掉原同步。

随后用`45ba936`读取两次**未启用profiler**的原`engine.step`与
`get_num_unfinished_requests()`，只记原返回和时间；step及generate绑定均已恢复。
其中每卡29请求的同一个批次为：

| 原生成调用，秒 | rank0 | rank1 |
| --- | ---: | ---: |
| 完整`generate`调用 | 48.266 | 87.567 |
| 尚未完成请求数≤4时的原step时间 | 28.039 | 64.484 |
| 占原step时间比例 | 58.1% | 73.7% |
| 只剩1个请求时的原step时间 | 9.150 | 47.361 |

两卡开始时间相差0.030秒，快卡约提前39.301秒完成；外层原VERL同步RPC仍等整批返回，
LOOP只能在之后收到回复并推进这些轨迹。另一个每卡3请求批次用时7.786/16.825秒。
这里的计数属于原output processor，不冒充GPU利用率或纯decode时间；
也不把两次调用外推成整个训练的提速倍数。原记录见
[等待计数](../../research/temporary/rl_upstream_alignment_20260929/appworld-rollout-scope-20261001/formal-inflight/summary.json)，
简表见[阶段报告](results_formal_generation_profile.json)。

当前vLLM内部`async_scheduling=True`，剩余大项是外层批次返回边界；仅再打开这个
开关不会解决它。后续应接原逐请求异步completion及原LoRA同步，不能自行复制调度器、
减少任务步数或缩短生成来宣称修复。已对实际冻结owner做只读检查：其两个async模块
与本地参考文件SHA相同，而当前sync模块有已验证补丁，不能整体换回临时checkout。
实际async模块的独立导入在现有vLLM0.15报`vllm.entrypoints.openai.protocol`不存在；
这证明旧async入口还不能直接切换，不表示正在运行的sync入口失败。
[源文件与导入回执](../../research/temporary/rl_upstream_alignment_20260929/appworld-rollout-scope-20261001/native-profiler/async-owner-interface-audit.json)
保留精确路径、SHA及错误；本次没有换依赖或部署未对照的异步路径。

### 2026-10-01 19:28：TextCraft部署同一份padding修复

复用已由原VERL padding断言验证的`1f862e8bbdaa…`文件，不重写forward，不再跑一套
GPU测试。部署助手复用历史SQL脚本，`c462a3f`增加显式任务、双rank预检查和`func=`
原RPC调用；`3a46cec`完成已通过预检查的绑定。完成脚本SHA
`0244a0c7d547e65628f3a7c3e9900ac64faec0205540dd8af688e696081eff48`，与该提交的文件一致。

- 两rank原优化器均为64步，即16轮×每轮4次联合更新，不是双卡各算一次后相加为128。
- 只在第16轮原更新结束、第17轮旧概率尚未计算前应用；模型和优化器对象、配置均未替换。
  LoRA8/16、每卡actor/DT4、完整轨迹和任务预算保留。
- 第一次预检查成功时，TaskRunner还在更新后的`batch_decode`日志输出，过窄的
  `multi_turn_loop`阶段断言退出；当时没有绑定任何新方法。原栈和失败原因保留，
  随后在已观测到的下一轮采样阶段完成同一预检查的部署，没有重新训练或重跑数值对照。
- 当前三组有效actor都为同一padding SHA；TextCraft的完整恢复候选已含相同文件。
  本次没有切换TaskRunner中的DT分发和生成上下文，因此仍不能称三组效率路径已全部统一。

远端`active-training.json`、当前formal清单和`active-source.json`已索引该完成回执；
旧内容在回执目录封存。快照分别保留冻结v7、head覆盖和padding覆盖，避免把旧启动参数
误当当前参数，或恢复时漏掉任何覆盖。只读快照进一步保存提交与完成脚本的不同版本。

### 防止重启回旧行为的版本对应

此前返工不只是漏写版本号：旧launch、实际worker覆盖、已测但未部署的候选被混成了
一个“最新版本”。现在按以下证据恢复，而不是重新挑选同名目录或复述历史通过结果：

1. `active-training.json`确认PID、创建时间与冻结entry/owner；源码SHA对应原`source.json`。
2. worker覆盖必须有两个当前PID的完成回执；排队或旧PID回执不能算生效。
3. 每个修复保留原官方对照的文件SHA和适用范围。DT/FA/FLA、VERL、vLLM各自的
   对照不互相替代，不增加整网两次更新阈值。
4. 恢复候选直接固化当前有效head、LoRA8/16、每卡micro4及已有修复；原恢复接口
   只增加`resume_mode/resume_from_path`。已通过的CPU配置对照和候选SHA保持关联。

这些检查针对已经发生的旧版本回退和部署漏项，不新加训练参数或额外GPU验收。
当前仍未部署的修复必须继续列明；记录齐全不等于所有修复已经生效。

### 2026-10-01 19时：正式采样缓存与剩余等待的实测

在AppWorld当前PID2479539的原worker RPC边界读取随后8次正式`engine.generate`返回的
RequestOutput元数据，没有增加生成请求、环境轨迹、模型或GPU，也没有改采样参数。
观测源码`89455a9`，运行脚本SHA与登记回执一致。两个rank均已记录8次并自动恢复原
generate绑定；原始回执和汇总见
[正式缓存观测](../../research/temporary/rl_upstream_alignment_20260929/appworld-rollout-scope-20261001/formal-cache/summary.json)。
另调用原`LLM.get_metrics()`读取原生分阶段指标，两rank均明确返回
`Stat logging disabled`。未改运行参数或把缺失指标填0；这次只能报告原生成调用耗时，
不能进一步称为纯prefill或纯decode时间。原返回保存在同目录`native-metrics.json`。

| 项目 | rank0 | rank1 |
| --- | --- | --- |
| 原生成调用数 | 8 | 8 |
| prompt tokens | 1,076,702 | 1,098,078 |
| cached tokens | 825,856 | 843,264 |
| 缓存命中比例 | 76.70% | 76.79% |
| 生成tokens（含原DP padding计算） | 25,213 | 35,102 |
| 引擎调用累计秒数 | 407.04 | 519.24 |

每次观测的原上下文均开启，同一rank的LoRA ID保持一致；这证明生成上下文修复已在正式
工具交互中产生实际缓存复用，不代表全轮已提速某个倍数。计时仍含prefill和decode。

**新的实测大项是同步批次等待。** 两卡请求数都为`32,1,32,1,32,1,32,1`；
按对齐调用的最大rank耗时计算，4次单请求批次共125.48秒，占8次的540.49秒的23.22%。
这是引擎关键路径占比，不是完整RPC或整轮占比。第一次32请求调用的两卡耗时为
59.85/100.88秒，相差41.04秒。同步接口必须等整批结果，不能因缓存已修复就认为剩余
等待消失；也不能把这段等待与DT跨卡分区问题混成同一阶段。

已只读检查固定VERL `20bd331` 的异步路径：trainer中manager生成调用被注释，实际仍进
`traj_collector.multi_turn_loop`；`AsyncActorRolloutRefWorker.generate_sequences`明确
抛出NotImplementedError。因此仅设置`mode=async`不构成可用接入。本次没有切换后端、
升级依赖、添加凑批延时或自行复制异步调度器；后续改动须针对这笔已测成本复用原接口。

观察登记最初两次被Ray客户端参数签名拒绝，均发生在RPC提交前。读取当前actor元数据后，
确认其句柄只接收`**kwargs`，改为原方法的`func=`调用后成功。两份失败回执保留；
没有训练中断、GPU试跑或在worker中留下失败包装。

### 2026-10-01 18:30：AppWorld部署生成上下文修复

原PID285580已完整保存step4，原marker=4；`data.pt`与双rank的model/optim/extra_state
均存在且非空。核对准备回执及所有候选SHA后，仅终止该AppWorld进程树，SQL PID552842、
TextCraft PID212110及创建时间均未改变。停止回执为
`receipts/owner-b8-dispatch-20260930/appworld-rollout-scope/completed-checkpoint-stop.json`。

用既有`submit_prepared_appworld_resume.py`及原VERL loader提交PID2479539，启动源码
`6e8fbe90ac1dd27f4f880598d3fe3b19b0d5b319`。新manifest为
`runs/appworld-rollout-scope-20261001/formal-training.json`。只增加已测的整段采样上下文；
原任务预算、采样/评估、LoRA8/16、micro4、PPO与DT数值版本保持。
18:37原TaskRunner已记录`Setting global step to 4`及原恢复路径，两个rank加载各自的
model/optim/extra_state后进入原`actor_rollout_generate_sequences`。观测回执为
`appworld-rollout-scope-20261001/restore-observation.json`；尚未完成新完整迭代。
旧step4用时18129.757秒，gen9737.716、DT4908.032、actor3028.621、保存52.922秒；
这是旧版本基线，不能当作新修复的耗时。新的完整迭代尚未结束，不预报整轮提速倍数。

### 2026-10-01：把未部署修复组成完整恢复版本

反复返工有两个已证实的来源：旧接入把response数当成轨迹数，放大原PPO学习负载；
后续修复又分布在启动目录、worker运行时覆盖和未部署候选中。只记录“测试通过”或
本机HEAD，没有同时记录实际生效PID和完整恢复组合，就可能重启回旧行为。
本节不把代码准备完成当成部署完成。

| 不一致/恢复风险 | 修好的代码与证据 | 当前状态 |
| --- | --- | --- |
| response粒度误作PPO轨迹单位 | SQL `37938d0`、TextCraft `cb8e569`、AppWorld `cf145b2`；原trainer负载核对 | 三组已修复。SQL满批1次、TextCraft满批4次optimizer更新；不是每个DT请求一次更新 |
| TextCraft启动文件仍是micro1，运行中已B4，重启会漏掉head覆盖 | `dc4e4d7`及两个实际worker PID的完成回执；恢复候选固化这些相同文件SHA和micro4 | `textcraft-rollout-scope-20261001`已准备，未部署；旧启动文件保留为历史证据，禁止直接重用 |
| TextCraft无效padding计算 | padding `0c80b41`，原对照`44e1149`；原RPC部署`3a46cec` | 19:28双rank已应用，三组有效actor SHA相同 |
| SQL/TextCraft DT双卡分发不均 | `4c0cbdd`调用原VERL分区；实际输入的原FLA断言回执 | 已组合进恢复候选；这两组尚未部署，不提前记为生效 |
| 工具轮次之间重复退出原生成上下文 | `2a32d00`，原vLLM实际prompt比较及原collector默认路径测试 | AppWorld已从完整step4提交部署；SQL/TextCraft候选已准备、未部署 |
| 恢复入口另造训练设置的风险 | SQL/TextCraft仅转交原`trainer.resume_mode/resume_from_path`；AppWorld沿用原入口 | 原验证器和逐配置比较通过；没有另写保存/恢复算法 |

恢复候选均以**当前作业的实际entry/owner**为底，加上已有通过对照的文件；不从旧
通用部署目录重新拼装。准备回执位于远端
`receipts/owner-b8-dispatch-20260930/{appworld,sql,textcraft}-rollout-scope/prepared.json`，
包含被替代PID、来源回执SHA、候选逐文件SHA、验证回执与配置差异。
`current_runtime.json`同时列出这三个候选和实际作业，明确区分。

- AppWorld：7项CPU检查，原任务/模型参数不变，仅新增已测生成上下文边界。
- SQL：16项CPU检查；相同正式参数构造后，仅数据类文件和模板所在目录变动，
  两份文件内容SHA一致。原完整检查点恢复不改变任务负载。
- TextCraft：49项CPU检查，耗时16.19秒；相对旧launch仅固化已经在原worker上生效的
  actor/logprob/reference micro4、原fused head及torch后端。其他任务、采样、损失、预算
  参数逐项相同。LoRA reference走原`compute_ref_log_prob -> compute_log_prob`，
  不新建reference model，不走非LoRA分支的独立ref microbatch。

准备中的两次检查错误也保留：SQL首次pytest从运输目录导入，缺少旁边的配置资产，
修正为从冻结entry运行后16项通过；TextCraft首次已通过49项，随后记录比较误把旧launch
中省略的fused选项当作显式false。修正记录比较后复用相同候选及原测试回执，未修改模型
或再跑GPU验证。它们不是正式训练故障，也不隐藏为“首次全部通过”。

SQL当前没有作者保存周期内的完整检查点；关于使用原VERL保存/恢复模型、优化器、RNG，
但在缺少`data.pt`时按原行为重置数据读取顺序的选择，仍待用户答复。未擅自执行。
AppWorld已在原step4检查点完成后提交恢复，TextCraft原保存周期为25。未停止SQL/TextCraft未保存的更新，
未使用SIGSTOP、mcTracer、伪造`data.pt`或自建checkpoint逻辑。

**数值残差补查已结束，不再把非零残差当新门槛。** 同一事实B4、同一原RPC内比较
前缀704/320/704，首个观测差异位于GDN层0；Q/K/V/beta等相同，缓存状态和其派生量
随分块变化。对记录的两份实际FLA输入，实际FP16与显式BF16对照共44项原FLA断言通过，
参考与阈值未改；共73.04秒，独立算子进程最大allocated 2.584GiB、RSS 9.525GiB。
范围是原forward/backward与有限传播重合端点极限，不是整网归因验收，也不覆盖其他历史
BF16失败样本。没有新增纠偏、放宽容差或修改数值核。原始输入路径、逐项结果和脚本SHA见
[DT分发诊断及原容差回执](results_dt_dispatch_profile_20261001.json)。

<details>
<summary>历史检查与候选记录：其中“当前、未部署”等状态只描述记录时刻，继续工作以本页顶部和带时间的current_runtime.json为准</summary>

### 2026-10-01 17:44：实测分发成本与恢复候选

观测、准备助手和原始回执封存于提交`8e07dd3`；17:45只读快照再次核对当前三组
entry/owner、固定数值核心与已准备候选，未发现SHA漂移。快照记录的提交是采集时
的代码来源，不代表新候选已部署。

本次没有改变正式训练的模型、算法、参数或数值容差，也没有停止尚未保存的更新。
AppWorld已冻结完整恢复候选 `candidates/appworld-rollout-scope-20261001`：
在当前实际entry/owner上只加入`2a32d00`已对照的整段采样上下文，保留请求分发、
DT均衡分发、padding和head修复。7项远端CPU测试通过；准备回执含原PID285580、
原检查点根目录、逐文件SHA和原vLLM比较回执SHA。**这是已准备版本，尚未启动**；
提交助手只把原完成检查点交给VERL恢复，不负责停止或另写保存/恢复逻辑。

原worker上的有界DT观测使用保存的真实token和回报，沿用原readout/FSDP/DT调用，
没有optimizer更新。两卡一次B4各收集参数101次，CPU参数来源字节各24.19GiB。
短输入卡的collective事件累计6.82秒，长输入卡0.42秒，而两卡调用均约10秒：
主要暴露两卡工作量差异造成的等待。不同流上的事件包含重叠与等待，不能相加成墙钟
或把这些秒数都写成搬运耗时。

随后用同一组8条真实输入各重复一次，固定每卡2次B4，比较原连续分片与原VERL
`get_seqlen_balanced_partitions`。最慢rank耗时19.717→9.641秒，有限小样本比值
2.045；参数收集次数仍为202，CPU参数来源字节仍为48.38GiB/卡。**不是整轮提速比**。
Q逐值相同；A最大绝对差0.049801、均值差0.000772。恢复样本身份后，共同前缀
由每批704/320变为320/2496，批内padding和native分块路径也随之改变；同一模式内
重复输入亦有非零差异。该结果是定位材料，**不构成整网DT通过官方容差的声明**；
未添加新阈值、归一化或纠偏，也未据此修改FA/FLA或撤销既有原算子容差。

原始数据、输入身份、阶段事件、源码SHA、数值差异和恢复标记见
[本次分发观测](results_dt_dispatch_profile_20261001.json)。观测包装在每次原RPC返回前
恢复；正式worker没有保留新的FSDP方法。诊断夹具最初漏传原reward manager必读的
`episode_lengths`，在任何模型调用前失败；随后补传原配置不使用的`None`元数据，
没有伪造轨迹长度或奖励。此失败属于诊断夹具，不能写成正式训练故障。

当时未闭合的是：SQL/TextCraft的均衡分发与整段rollout上下文尚未部署，TextCraft的
padding尚未部署；分批变化的数值差异尚待按实际计算路径解释（后续原容差补查见上节）。这些状态与已通过的
测试、已完成的正式更新分开记录，不能因本机提交变新就把远端标成“全部对齐”。

### 2026-10-01 17时：负载、版本与未闭合项

最新原TaskRunner日志见[负载核对回执](results_workload_alignment_20261001.json)。
SQL本进程完成3轮，AppWorld从step2恢复后完成新step3，TextCraft完成14轮。
SQL第3轮190.17分钟、AppWorld第3轮251.95分钟、TextCraft第14轮79.50分钟。
这些是当前实现的实测，不能称官方速度；不同批次不能作受控提速比较。

训练单位已经改为原完整轨迹：SQL global mini1280、每轮1次更新；TextCraft
global mini64、每轮4次更新；AppWorld global mini32、2遍。DT仍按实际response
线性归因，不能把DT请求数说成PPO更新数。原作者单位换算和生命周期6项本机回归
通过（3.92秒含启动，进程树RSS峰值128.87MiB，前后显存均0MiB）。

| 已发现的不一致 | 修复来源与验证 | 此次核对的实际生效情况 |
| --- | --- | --- |
| response行被误当完整轨迹，放大PPO更新数 | SQL `37938d0`、TextCraft `cb8e569`、AppWorld `cf145b2`；复用原轨迹/采样接口和原VERL更新，原单位单测及正式更新回执 | 三组已使用完整轨迹；不再采用旧120次/轮口径 |
| AppWorld逐轨迹建runner及完成/取消规则偏离作者 | `cf145b2`恢复原LOOP sampler/runner池，`5724fe1`修复队列运输 | 当前AppWorld继承；不恢复已退役manager |
| 两卡之前又加总请求32上限 | `64e6377`，原队列与方法对照 | AppWorld PID285580已应用，原日志出现62项提交；每卡vLLM32仍不变 |
| 无效padding计算 | actor `1f862e8bbdaa…`，来源`0c80b41`、原VERL比较证据`44e1149` | SQL、AppWorld已应用；TextCraft未应用 |
| DT双卡长度分配不均 | `4c0cbdd`，原分区接口及顺序恢复测试 | AppWorld已应用；SQL、TextCraft未应用；不把长度代理收益称实测速度 |
| 每个工具轮次重复退出生成上下文、换LoRA ID、失去前缀缓存 | `2a32d002cffb6a16a1be89e835c3f43ba70b471c`；原manager外移到完整采样边界，原collector正文不变；4项远端CPU与双rank真实prompt原vLLM比较通过 | **三组均未部署**；仅候选文件及有界回放，正式worker方法已经恢复 |

生命周期候选的完整SHA和原对照证据见[results_rollout_scope.json](results_rollout_scope.json)：
worker从`9278dc651af5…`到`807e51856f99…`，collector从`90c525ace13c…`到
`8e4e3372f790…`；wrapper为`98bc4d3dd899…`。沿用原vLLM token/top-k比较，
没有自造scalar容差或改采样参数。长输入同一组prompt的原热调用engine耗时6.58秒，
候选第二次4.80秒，缓存命中6144→24064；生成token数256→253仍通过原比较。
候选call计时不含外层enter/exit，故不作整段生成总耗时或确定提速比。

六份正式DT原生阶段计时也已保存。B4四对端点的张量batch为8，不能误称每卡B8。
短序列643–665时，端点前向与逐层重算合计约占完整DT的78%–81%，有限传播约9%–11%。
阶段CUDA事件包含流上等待，尚未分离搬运/collective/计算，不能直接写成拷贝瓶颈。
当前重点仍是root/replay与双卡等待，未改DT数值核、Q/V/A、FA/FLA或容差。

`README.md`不再重复维护步数，`LOCAL_TESTING.md`已更正过期的“当前120次更新”。
源码快照增加原collector、trainer和sharding manager的实际文件SHA，并单列未部署候选。
TextCraft冻结启动参数中的micro1仍如实保留；有效actor/logprob4由当前PID的完成覆盖回执
证明，不能把启动参数直接改写成4后假装它原来如此。后续恢复必须包含已生效覆盖，
不通过停止未保存的正式更新来强行上线候选。

### 2026-10-01 09:07 AppWorld请求分发修复上线

旧PID2680224的原trainer完成step2，`latest_checkpointed_iteration.txt=2`，
两rank的model/optim/extra_state和data.pt已存在且非空。原完整迭代17709.505秒，
生成10345.341秒、DT4244.875秒、actor2619.001秒；它是当前实现实测，非官方效率。
核对准备回执的entry/owner/DT全部SHA后，终止该作业自己的进程树并用原VERL恢复入口
提交PID285580。原检查点与冻结目录保留，没有SIGSTOP或修改其他作业。

唯一新增生效代码是此前验证的`loop_owner_rollout.py`请求分发修复`64e6377`：
SHA从`cb00a657de71af5afcf386f0ab520377746172152a61a328af9068d4147da537`
变为`22ff649007bf6d98584e6ebb1d19f72cd6010d68e6f40ac082418ee50ee0dce5`。
去掉接口在两张卡之前额外施加的总请求数32限制，保留每卡vLLM原max_num_seqs=32，
让已排队请求由原VERL分发。采样、环境、全局预算、LoRA8/16、每卡micro4均未改变。
测试仍仅为原接口/配置对照；未声称整轮或生成提速已经验证。

原回执为`receipts/owner-b8-dispatch-20260930/appworld-request-dispatch/stop-after-step2.json`，
新启动source/job在`runs/appworld-request-dispatch-20261001/appworld-dt/`。
随后原TaskRunner与两rank日志均记录读取step2的模型、优化器、extra_state，
并进入新rollout；原运输日志已产生生成结果，实际单次提交62个已排队请求，
超过旧接口32上限，证明分发修复已经走到实际调用。尚不能据此宣称整轮加速比。回执见
[恢复后生成记录](../../research/temporary/rl_upstream_alignment_20260929/appworld-request-dispatch-20261001/restored-generation.json)。
09:11重新核对DT/FLA/vLLM文件未发现相对固定回执的SHA漂移。

同次性能调查在TextCraft原worker RPC边界安装了三次后自移除的观测包装：
只保存正式runner已经返回的CUDA阶段计时，以及原vLLM输出中的长度/缓存元数据；
输入和返回对象不变，不添加模型调用、CUDA事件、同步、参数或数值校正。
AppWorld旧worker上的DT观测包装在step2后退役前尚未产生样本，随进程退役，
不能把它写成已取得DT逐层数据。原始CPU采样也不能把等待栈解释成GPU拷贝耗时。

TextCraft三次/引擎的正式生成观测已完成并恢复原方法：482个请求，
1347912个prompt token，仅8704个缓存命中（0.646%），生成104555个token。
原vLLM每请求时序因log stats关闭而缺失，不能据此拆出纯prefill或decode耗时。
实际源码核对到每轮重新同步LoRA、创建新ID，以及LLM.sleep重置前缀缓存；
AgentGym原版在整段rollout外管理引擎。尚未修改这处生命周期，不把修复方向当提速结果。
正式各阶段耗时、官方预算依据及观测边界见[性能记录](results_phase_cost_20261001.json)。

### 2026-10-01 05:39 SQL采样事故

对SQL rank1使用MetaX `mcTracer --attach`，05:39:18发送其官方停止指令Ctrl+T，
05:39:33工具完成输出后该worker退出，Ray随后结束整个SQL作业。没有捕获到DT
非有限值异常；容器`oom_kill=0`、`memory.failcnt=0`。具体底层退出原因未确定，
按与本次采样相关的事故处理，不归咎于DT数值或PPO。**不再向正式worker附加mcTracer。**

旧SQL已完成1轮，但原`save_freq=60`尚未产生检查点；第2轮DT到219/572批。
首轮内存中的模型/优化器更新无法恢复。05:44使用原正式入口重新启动，保留初始
验证、预算、LoRA8/16、每卡4及全部原训练选项；只将已在旧worker生效的actor
`1f862e8bbdaa…`写入新冻结owner。原68个entry文件未变；原生launch选项逐项比较，
除部署/输出路径迁移外一致。AppWorld、TextCraft没有重启或改动。

原日志片段、工具/trace SHA、退出与新PID对应关系见
[事故回执](../../research/temporary/rl_upstream_alignment_20260929/sql-native-trace-incident-20261001/sql-native-trace-incident.json)。
该记录不是训练通过回执。新source中继承的旧`numerical_override`重复字段已按实际
新owner同步，修正前文件及SHA另存，执行代码和参数没有因此改变。

每条区分“修复代码”“部署位置”和“证据覆盖”；没有部署回执不能从提交推断已生效。
下表保留三任务接入及其依赖的历史修复链；行中的部署位置描述修复当时，当前适用范围以上方三组表和快照为准。

| 修复提交 / 问题 | 实际改动及代码 | 当前适用范围 / 验证证据 | 已失效的用法 |
| --- | --- | --- | --- |
| `8cbb29d`、`c9cd147`：vLLM更新后生成出现非有限值，权重池/sleep恢复遗漏 | 修复MetaX owner映射页初始化，恢复原vLLM权重池上下文；没有自写采样器 | 基线 `installed_restored_files` 两文件SHA；[vLLM回执](results_vllm_owner_alignment.json) | 不能只看包版本0.15；它不反映已安装源码补丁。FP16断言不能直接充当BF16整模型门槛 |
| `8e4af7a`：官方环境与Qwen3.5入口缺乏本机对照 | 官方环境、模型入口、配方单测；[LOCAL_TESTING.md](LOCAL_TESTING.md) | 83项本机单测的明确覆盖范围 | 本机接口测试不证明GPU训练或任务性能 |
| `99fb5c2`：新环境重复引入训练/推理框架、信用与原奖励边界不清 | [owner_environment_transport.py](owner_environment_transport.py)、原奖励后DT接线、既有VERL/vLLM复用 | [results_environment_entry.json](results_environment_entry.json)及固定数值基线 | 不能重新启动SkyRL/AgentGym/LOOP的训练器或推理服务 |
| `37938d0`：SQL轨迹与官方训练载体不一致；LoRA配置错误 | [sql_owner_rollout.py](sql_owner_rollout.py)、[owner_trajectory_batch.py](owner_trajectory_batch.py)、固定LoRA8/16 | 原SQL状态机/轨迹/环境单测；SQL当前v8继承 | 历史rank1/alpha2的容量记录失效于当前模型配置 |
| `cb8e569`：TextCraft自写session偏离官方；原验证直接cat异宽reward失败 | 复用AgentGym原完整rollout；在原验证处先按原维度sum再cat；[patch_owner_trajectory_entry.py](patch_owner_trajectory_entry.py) | 原环境固定动作/异常对照、[test_native_validation_batches.py](test_native_validation_batches.py) | 自写TextCraftSession不再使用；不得按旧异宽cat回退 |
| `cf145b2`：共有左padding裁剪越过response边界 | [patch_actor_response_boundary.py](patch_actor_response_boundary.py)，保留原response全部列及前驱logit | 28项有效logprob/梯度/索引对照，native-trajectory-v5回执 | 仅按attention共同左空白裁剪可能删掉需要的训练列 |
| `cf145b2`：TextCraft原作者重编码后token与最初生成ID不同 | DT使用原作者实际训练token；[textcraft_owner_rollout.py](textcraft_owner_rollout.py) | 10项真实服务/异常/特殊token检查，native-trajectory-v5回执 | 不再要求官方重编码后的IDs等于最初vLLM IDs，也不复制作者token处理 |
| `cf145b2`：AppWorld逐轨迹创建runner、完成/取消行为偏离作者 | 复用LOOP原sampler和runner池；[patch_loop_external_completion.py](patch_loop_external_completion.py)、[loop_owner_worker.py](loop_owner_worker.py) | 原方法AST、train/eval、尾部取消、双进程Gloo等5项，native-trajectory-v5回执 | 已废弃逐轨迹manager；不能把单个runner.run对照冒充完整采样器对照 |
| `cf145b2`：MetaX C层隐藏设备掩码污染Ray worker | AppWorld launcher以显式env执行原入口 | v5双worker通过原失败位置；当前v6继承 | 不再用继承C层隐藏环境的旧execv路径 |
| `5724fe1`：AppWorld队列已有60项但每批只取4–7项 | 在completion运输层按已有Queue计数收齐，仍最多32，不等待新任务；[loop_owner_rollout.py](loop_owner_rollout.py) | 真实延迟feeder回归及原方法AST共2项，native-trajectory-v6回执；AppWorld当前v6 | 不再用`get_nowait()`把feeder尚未送达的数据误作队列已空 |
| `dc4e4d7`：原dense actor绕过自身编译熵接口，首反向OOM | [patch_actor_entropy_dispatch.py](patch_actor_entropy_dispatch.py)，调用原 `self.compute_entropy_from_logits` | [results_actor_entropy.json](results_actor_entropy.json)，30项接口/边界测试；已覆盖三组 | micro1容量回执仅为历史，不能满足当前每卡4 |
| `dc4e4d7`：B4长response词表输出占用过大；原chunked head未接Qwen3.5且dtype对照超差 | 接原Qwen torch wrapper和原FusedLinearForPPO；FP32输出存储、BF16梯度分支和temperature除法边界；编译原chunk，保留原chunk循环 | 原VERL head用例5次、实际BF16/FP32的1/.9温度对照、11项最终CPU检查；实际B8×32768更新98.039秒、物理采样49.86GiB/卡。详见[新回执](results_actor_b8.json) | 不得以降低rank/alpha/microbatch、放宽容差、乘性纠偏或重新引入整网两次更新阈值来替代 |
| 2026-09-30运行事故：SIGSTOP恢复后SQL通信超时 | 停止的是旧失效作业；原正式入口重启SQL v8，没有修改通信超时或跳过官方初始验证 | [sql-pause-timeout.json](../../research/temporary/rl_upstream_alignment_20260929/owner-b8-dispatch-20260930/sql-pause-timeout.json)及当前新PID | 不再对通信中的分布式训练用SIGSTOP保留现场；旧内存中DT进度没有成为完成检查点 |

## 后续继续与恢复

1. 先读远端当前manifest，并用PID创建时间确认身份。终止/超时/旧日志不能混作新作业状态。
2. 不覆盖冻结运行目录。AppWorld使用当前balanced-padding冻结入口及原恢复参数；
   SQL使用新冻结padding入口；TextCraft启动后有PID绑定覆盖，不能直接重放旧冻结命令遗漏它们。确需重启时
   将已验证的实际组成冻结到新目录，沿用原入口、预算、检查点，并另留部署回执。
3. 保持每卡actor/DT4、LoRA8/16；任务global minibatch及PPO epochs按原配置。
   B8是双卡实际microbatch总数，不是把SQL1280、TextCraft64或AppWorld32的global minibatch改成8。
4. 修改时沿用本账本记录“原错误→最小修复→对应原测试→实际生效文件/回执”。
   未改动且SHA一致的数值叶子复用已验证结果；有新错误才定位并做对应范围的对照。
5. 用下面的只读命令更新快照；它不启动、暂停、热修改训练或下载模型，也不导入训练模型。
   `matches_workspace_bytes=false`明确表示本机文件与冻结文件不同，不能自动解释为部署损坏：
   同时查看该任务实际入口以及`runtime_overrides`。`entry_files.matches=false`才表示冻结
   文件已偏离其原source回执，需要调查，不能静默把新哈希写成“已验证”。

```powershell
& C:/Users/Administrator/miniconda3/python.exe -X utf8 research/temporary/rl_upstream_alignment_20260929/record_current_runtime.py
```

## 本次训练证据与版本边界

- 旧SQL首轮原更新完成，两rank原AdamW步数均1、状态有限且非零；随后的padding
  覆盖完成，原TaskRunner已进入下一轮采样。首轮日志16142.743秒（4.48小时），
  其中actor4758.024秒；**这些耗时属于旧actor，不能用来判断新padding版本的速度**。
  该作业已因下述采样事故退出，无正式检查点；新PID552842从原模型重启，不继承此步数。
- AppWorld旧作业首轮完成，两rank各14次原AdamW更新，状态有限且非零，原
  `global_step_1`检查点及完成标记存在。旧作业已在该边界停止。新PID2680224原日志
  明确记载恢复路径、global_step=1及后续生成；尚未完成新版本的一整轮更新，
  不把成功恢复说成已测得完整提速。旧首轮29969.050秒属于此前运行过程。
- TextCraft已完成5个正式迭代并进入第6轮生成；第5轮4718.347秒、训练奖励均值0.535，
  见[连续性记录](results_formal_continuity_20261001.json)。它仍使用上表旧actor和旧DT分发。

原训练日志的`rollout_probs_diff_*`使用response attention mask，会包含观测位置；
运输接口在这些非action位置的rollout_log_probs填0。这些原指标不能当作仅policy token
的vLLM数值对拍。本次保留原指标，不改变算法、不新增对拍门槛。

04:20后的原状态观察确认：AppWorld恢复后的新worker两rank均保留AdamW步数14，
状态有限且非零，每卡4、LoRA8/16。SQL第2迭代的首次交互已完成1280条轨迹的生成，
947554 token、1428.793秒，整次生成RPC约663.185 token/s；第1迭代相同阶段为
972237 token、1550.218秒、627.161 token/s。这是不同采样工作的正式观察，
不是纯decode或受控提速对照。恢复和后续生成原回执见
[连续性记录](results_formal_continuity_20261001.json)。

SQL首批原训练分数为1269条−1、8条0、3条1；−1来自原SQL格式评分。
不能把有限梯度或原更新完成等同于任务效果良好。当前未改变prompt、stop、长度或评分。
新padding的首个正式更新耗时及AppWorld新分发的实际DT耗时仍待正在运行的作业给出，
不以候选夹具速度替代。三组继续运行，没有为本次观察重跑训练测试。

05:06原TaskRunner调用栈已确认SQL第二轮`old_log_prob=374.7883407473564`秒，
并进入DT；上一轮原日志为1310.274秒。两轮均1280条轨迹，有效上下文总量从
4,157,129变为4,064,835 token（约少2.2%），阶段耗时少约71.4%。两批采样内容
及长度不同，这不是同输入受控速度比较；它提供了新padding在正式前向阶段生效的
实际证据，不能代替本轮尚未完成的actor更新测量。
来源、完整actor SHA与原计时见[SQL正式padding观察](results_sql_padding_formal.json)。

随后TextCraft原日志确认第5轮完成：整轮4718.347秒，生成2666.679秒、DT1177.208秒、
原actor更新575.956秒，训练批奖励均值0.535。参数和执行版本未变，没有追加优化器
状态读取或数值测试；不从批间奖励差异推断效果趋势。原行保存在
[连续性记录](results_formal_continuity_20261001.json)的最新TextCraft回执中。

### 新候选 `64e6377`：移除AppWorld运输层的重复限流（未部署）

正式日志反复出现`batch_requests=32 queued_requests=32`。运输层误把每个vLLM
引擎的`max_num_seqs=32`当成全局RPC上限，原VERL再等分后每卡只有16个请求。
修复仅删除运输层上限，把已经排队的请求交给原VERL分发及原vLLM调度。
64个请求的CPU回归中，原实现两次`[16,16]`，候选一次`[32,32]`。
每个引擎的32上限、原LOOP runner数量/取消/完成规则、采样与训练参数均未改。

候选`loop_owner_rollout.py` SHA为
`22ff649007bf6d98584e6ebb1d19f72cd6010d68e6f40ac082418ee50ee0dce5`，
当前PID2680224仍使用SHA
`cb00a657de71af5afcf386f0ab520377746172152a61a328af9068d4147da537`。
5项CPU接口测试通过，覆盖原DP分发/收集、奇数pad/unpad、取消与返回身份，
以及已有Queue feeder和LOOP原方法对照。首次候选测试因pytest路径顺序加载了旧模块，
修正测试导入路径并复用已装extras后通过；原失败回执保留。
详见[候选与原始测试对应](results_loop_request_dispatch.json)。
尚未测量真实生成提速，不将请求数变化宣称为两倍速度；保留当前轨迹，
后续在原检查点边界恢复时另记部署回执，不能把本机修复提交当成已生效。

05:16对当前两worker的30秒只读采样进一步定位到批次尾部等待：rank0无活跃
Python采样，随后栈为Ray空闲主循环；rank1仍在原vLLM解码，原提交batch=16，
随后栈中只剩3个请求、各调度1个token。这支持上述分发修复，不能推断整轮空闲
比例或候选提速。原采样与调用栈的路径、SHA及覆盖范围已归入同一候选回执。

该修复现已冻结为`candidates/appworld-request-dispatch-20261001/entry`，复用当前
`appworld-balanced-padding-resume-20261001/verl`。与当前entry逐文件比较，只有
上述`loop_owner_rollout.py`变化；DT、actor和任务作者目录均沿用原版本。
[准备回执](../../research/temporary/rl_upstream_alignment_20260929/appworld-request-dispatch-20261001/prepared.json)
绑定当前PID2680224、原检查点目录、修复完整提交号、来源和逐文件SHA。
新冻结入口的原训练配置、数据载体和原恢复选项3项CPU检查通过。
已有提交助手增加可选`--prepared`和`--run-dir`来选择这份回执和独立输出目录，
原模型/优化器加载仍由VERL负责；助手不会停止作业，并拒绝向仍运行的旧作业重提交。
默认参数保留旧提交的可复查行为，**不能不带候选参数直接运行旧命令恢复当前作业**。
准备完成没有改变`active-training.json`，没有提交新训练进程。

### SQL首批格式失败定位（只读，未改任务行为）

对原保存的1280条轨迹，1279条能精确匹配唯一的官方模板提示。去除这个明确
前缀后调用固定SkyRL原`verify_format_and_extract`，1279条判定全部与原分数一致。
通过追踪原函数的返回行定位首个失败条件：903条在观测之后没有重新以`<think>`
开始；364条`<solution>`数量不等于1；1条没有闭合的think块；11条格式通过。
剩余1条未匹配，不推测其格式原因。它们是训练样本诊断，不是验证集指标。

源码、原轨迹、模板和数据SHA以及示例见
[原诊断回执](../../research/temporary/rl_upstream_alignment_20260929/sql-formal-format-20261001/observed.json)。
检查脚本复用原评分函数，没有另写解析器、执行数据库、生成新轨迹、补造标签或修改奖励。
当前证据定位了低分的主要触发条件，不将原训练接口测试扩大成模型能遵守格式的证明。

## 2026-10-01 观察口径修正

- AppWorld 的 DT 按官方 `num_tests` 所决定的回报类别分组；`batch=N/M` 是当前
  worker RPC 组的进度，不是整轮总进度。已在实际 TaskRunner 栈确认存在后续组。
  只读 `collect_native_progress_snapshot.py` 现保留每个已观察组的 plan、最近批次、
  累计批次耗时和最终 report；尚未提交的组不猜成已知总数。没有改变训练代码。
- 固定 VERL `_validate`（`ray_trainer.py:795,806`）先收集每个验证批次的
  `success_rate`，再平均各批次；这不是逐例加权成功率。SQL 初始日志的
  `val/success_rate=0.01416015625` 保留原名与原口径，不能直接当作 SkyRL 的
  `eval/all/pass_at_1`。固定 SkyRL `generators/utils.py:get_metrics_from_generator_output`
  按原 trajectory rewards/UID 聚合；当前没有重写任何评估公式或改训练配置。
- 当前固定 VERL 验证函数没有写出 `validation_data_dir` 的逐例文件，不能声称已经
  从该目录检查过 SQL 的具体失败文本。后续使用实际保存的轨迹再判断失败原因，
  不因缺少该材料重新生成整套初始验证。

## 2026-10-01 待部署的分发修复：`4c0cbdd`

**此提交是候选，不是当前三组训练已部署的版本。** 正在执行的TaskRunner仍使用
其冻结entry的`dt_training_batch.py`；本机源码与冻结文件不同是已记录的候选差异。
没有停止、重启或热改正式任务，也没有丢弃已采轨迹。继续运行时不能把下面的
CPU测试或负载估算报告为实际提速；后续部署必须另留新启动/生效回执。

- 已完成的真实DT报告显示：AppWorld首个回报类别组两卡分别处理5,387,368与
  2,594,254个上下文token；TextCraft首轮分别788,904与1,719,819。
  当前接口将全局长度排序后直接等分，造成长短请求分别集中到两张卡。
- 修复只调用原VERL `get_seqlen_balanced_partitions(equal_size=True)`，在原DP分发前
  重排，返回后先恢复原顺序再调用原unpad。每卡B4、类别分组、完整回报、token身份、
  Q/V/A、PPO以及所有数值容差不改，没有重写分区算法。
- 9项CPU接口测试通过，覆盖原DP分发/收集、重复行、补齐还原、两卡完整未来回报、
  零回报跳过与类别分组。夹具中的信用比值用于检查运输，不能当作模型数值对拍。
- 用真实已完成请求长度调用原分区函数后，每个同步批次的较长卡长度累计减少
  AppWorld25.60%、TextCraft26.82%；总padding计算量分别增加约0.36%、0.29%。
  这是工作量代理，**不是墙钟提速测量**。12秒Python栈采样主要落在CUDA同步，
  不能从中分离真实内核时间与跨卡等待时间。

代码SHA、原分区函数SHA、日志路径与测试回执见
[results_dt_owner_balance.json](results_dt_owner_balance.json)。

### 历史AppWorld恢复候选：`2036246`（未启动；已被下面组合候选替代）

已冻结`candidates/appworld-balanced-resume-20261001/{entry,verl}`，复用原v6的
LOOP环境目录；它只供AppWorld使用，不能拿其中历史SQL/TextCraft启动文件启动其他任务。
新的AppWorld入口已包含每卡B4、LoRA8/16、`dc4e4d7`输出头及`4c0cbdd`原生分区调用。
原head三个文件与actor文件均与已验证B8候选逐字节相同，DT数值文件匹配固定基线。

`--resume-from`仅转交原VERL的`resume_mode=resume_path`和`resume_from_path`；
原加载器负责模型、优化器、global step及dataloader，未复制恢复算法。
指定恢复路径前后的其他配置、采样参数完全相同。原配置/载体/恢复参数3项及
分发9项CPU测试通过；尚无该候选的实际恢复或提速结果。准备过程没有修改原任务，
也未改变任何原训练预算、存档频率或运行中的文件。
源码SHA、原目录和未启动状态见
[prepared.json](../../research/temporary/rl_upstream_alignment_20260929/appworld-balanced-resume-20261001/prepared.json)。

准备时的第一次`-k appworld`误匹配了父目录名，从而执行了其他任务的历史启动器；
其中SQL/TextCraft旧micro1断言失败，AppWorld通过。已改为三个明确node ID；原失败
回执保留在远端`appworld-balanced-resume/launch-cpu-tests-unscoped.xml`，不把它们
描述为已修复其他任务，也不把本候选当作通用三任务部署。

后续提交助手为
[submit_prepared_appworld_resume.py](../../research/temporary/rl_upstream_alignment_20260929/submit_prepared_appworld_resume.py)
并显式传入原完成检查点路径。它不停止任务、不加载或改写模型；原进程未退出时直接拒绝，
随后核对原完成标记、原状态文件、候选SHA及GPU2/3占用，再调用上述冻结入口。
它现在明确选择下面的balanced-padding组合候选，不再选择本节旧候选。
新source回执保存旧PID、恢复步数、候选SHA、提交脚本SHA和提交号；旧事实源归档保留，
SQL/TextCraft的作业条目不改。已通过本机与内嵌远端Python语法检查，尚未执行提交，
不能把该检查或脚本存在称为检查点恢复成功。

### SQL padding 调查（正式训练路径未改；候选单独封存）

首批正式采样为1280条轨迹、4850条response、1,385,232个policy token、
4,157,129个有效上下文token；原trainer记录采样3505.147秒。随后实际栈位于
原`compute_log_prob`，每卡B4；一次采样的`seqlen=20937`、
`response_length=head_response_length=18220`。这说明输出头仍保留完整response列，
但该单次栈不能给出整批padding比例或可节省的墙钟时间，不据此承诺提速。
这些单批数值来自当时的工具观察输出；旧`SkyRL-SQL-1884758-counts.txt`是滚动
计数文件，后来被覆盖，不能再把它当作该时刻的原始栈。现有检查脚本改为按时间保存
到`receipts/owner-b8-dispatch-20260930/phase-observations/`，优化器观察也使用独立
时间戳文件，保留首轮原回执。没有修改训练路径。

该阶段后来完成，原trainer记录`old_log_prob=1310.273639176041`秒，已进入DT；
当前固定栈回执为`phase-observations/1790787928-SkyRL-SQL-1880429-counts.txt`。

已检查原VERL remove-padding路径和当前Transformers Qwen3.5源码。后者的原
causal-conv接口读取`seq_idx`，原GDN接口读取`cu_seq_lens_q`；不能仅切开关就
假定跨轨迹边界正确。当前没有改这些参数、接口或数值路径，也没有撤回`cf145b2`
的response边界修复。后续若处理此处，先核实原owner如何传递这些边界及实际padding工作量。

2026-10-01 进一步完成一个独立候选的同权重诊断。原actor文件SHA为
`2b80b938fee442ea5d9273523b6cce2bcc0511b7729aa6cea90333fef5de7cd3`，
候选为`1f862e8bbdaad6fa116d0670772ad41269529a3a1e4a5b1eb383352d0372e9bd`。
它只修改共有左padding的运输边界；34项CPU接口测试通过，真实Qwen有效
token log-prob曾出现待定位差异（最大绝对差17.750082），当时未接入正式训练。
后续已复用固定VERL的原padding比较断言并通过，详见下节；候选仍**未部署**。
不能把其运行速度、一次更新完成或CPU测试通过本身写成数值验收通过。

候选源码、精确diff、测试与真实模型诊断统一封存在
[actor-response-padding-20261001](../../research/temporary/rl_upstream_alignment_20260929/actor-response-padding-20261001/README.md)。
`candidate.json`中的`CPU_only_candidate_not_deployed`是较早的阶段回执，
后续结论以同目录`status.json`和`result.json`为准；旧回执未覆盖。
默认`experiments/rl/patch_actor_response_boundary.py`及其测试已恢复为`bf5f4ab`
提交中的原内容，避免后续发布误带候选。远端正式目录、worker、训练参数没有改动。
上面的历史AppWorld恢复候选使用原actor哈希；下面的新组合候选已包含该padding候选。

### 版本状态速查（2026-10-01 03:46部署核对）

| 标识 | 状态 | 能证明什么 / 不能混成什么 |
| --- | --- | --- |
| DT `c9cd147` / 数值参考 `fc2e6c2` | 三组正在复用 | 固定DT文件身份与既有对应算子回执，不是新候选的验收 |
| VERL `20bd331` + 输出头修复 `dc4e4d7` | 三组已生效；AppWorld/TextCraft含PID绑定的覆盖回执 | 原head容差与B8×32768容量，不是未经修改的官方整个仓库 |
| 分发修复 `4c0cbdd` | CPU测试通过，正式任务未部署 | 原分区接口及顺序恢复，不是实测墙钟提速 |
| AppWorld恢复入口 `2036246` + 旧提交助手 `bf5f4ab` | 历史准备候选，未启动；被下方组合候选替代 | 不再重放旧提交助手选择该候选 |
| padding actor SHA `1f862e8bbdaa…` | 原VERL padding比较已通过，未部署 | 完整调查及适用范围见下节；默认补丁仍保持现有正式版本 |
| `appworld-balanced-padding-resume-20261001` / 准备及提交代码 `1ac9323` | 组合候选已冻结，46项CPU检查通过，未启动 | 复用 `2036246` 恢复入口、`4c0cbdd` 分发及 `1f862e8bbdaa…` actor；不是新DT数值版本 |

提交前后的文档版本、实际部署版本和数值参考版本分别记录。快照的
`code_repository_commit_at_collection`记录采集时本机HEAD，`recorder_source`记录
采集脚本的确切SHA、最后修改提交和是否存在未提交差异；它们不表示远端三组运行了
该提交的全部文件。每次继续工作先读本表，
再根据快照中的原manifest、实际路径、SHA、PID创建时间和运行时覆盖确认适用关系。

### padding 后续定位与原框架容差（2026-10-01）

完整来源和数据见[padding回执](results_actor_response_padding.json)。候选、默认
补丁及三组正式作业仍分开记录；本轮没有改变任何正式训练文件或运行时方法。

- 两卡、同权重、同输入重现原差异；取样的前512个有效token在输入及前3个GDN层
  输出逐值相同，第4层（索引3，第一个完整attention层）开始不同。
- 用原Qwen层及保存的前缀独立定位，再缩小为原Qwen RMSNorm与原`nn.Linear`：
  输入、归一化输出逐值相同，BF16 V投影随矩阵长度变化出现最大0.00390625、
  RMS约1.0841e-5的差异，与单层回放中的V逐值对应。两布局相对同一FP32投影
  参考的最大误差均0.0207186、RMS均0.00109877。它定位了形状相关低精度差异；
  没有修改线性算子、提升正式训练精度或添加纠偏。
- 固定VERL `tests/models/test_transformer.py::test_hf_casual_models`对padding
  使用的是masked-mean log-prob断言，`atol=1e-2, rtol=1e-5`。直接提取并执行
  原断言后，保存样本的两均值为−0.2779143和−0.2829209，**通过**。
  原函数测试的是所列单层模型；这里复用其比较方法检查Qwen3.5保存输出，
  不能声称官方原测试已覆盖Qwen3.5全训练，也不能把均值判据改说成逐token阈值。
- 首次逐层观察脚本按类名字符串找层，漏掉原FSDP2动态子类，因而失败。
  已按原继承关系改为`isinstance`并完成观察；这属于诊断脚本问题，未改正式模型。
- 后续应用候选须使用明确版本及原worker/恢复边界，保持同一训练迭代的旧概率
  重算和actor更新使用同一实现。不能在已经算完旧概率、尚未完成更新时切换。
  单层与投影参考只作定位；未新增整网容差要求，也不以定位耗时代替正式吞吐。

### 已冻结的新AppWorld组合候选（未部署）

远端目录为`candidates/appworld-balanced-padding-resume-20261001/{entry,verl}`。
旧候选、当前v6任务、原LOOP目录和DT发布均保留原样。新候选只将旧恢复候选的
actor换成已对照的padding版本，并保存其对应补丁和测试；不是重新实现训练或恢复。
准备与提交脚本固定在`1ac9323fe169bc36d58756611d5fe569ca8c7d7d`；
只读记录脚本的采集版本另见快照`recorder_source`，它的更新不改变已冻结的候选。
快照逐项记录脚本SHA与Git内容一致性，不能仅凭候选目录名重新生成另一份代码。

| 对应对象 | 精确来源 |
| --- | --- |
| DT分发 | `4c0cbdd`；`dt_training_batch.py` SHA `da9b8a01c3bb9ba00fe3388fd95f93961d33c44bdf0eccf32e7b5846e2ffcae2` |
| 原恢复入口 | `2036246`；`launch_appworld_native.py` SHA `53ddbb5fcb834fd8d700649fbfc734588757dcbc2095718e93429d6ea1778ac3` |
| 新actor | `dp_actor.py` SHA `1f862e8bbdaad6fa116d0670772ad41269529a3a1e4a5b1eb383352d0372e9bd`；源码封存在 `0c80b41`，后续原框架对照证据记录于 `44e1149` |
| 原actor | SHA `2b80b938fee442ea5d9273523b6cce2bcc0511b7729aa6cea90333fef5de7cd3`；当前三任务仍用此版本 |
| 数值核心与head | DT `c9cd147` / 参考 `fc2e6c2`，head `dc4e4d7` 三文件SHA不变 |
| 参数 | 原任务配置、LoRA rank8/alpha16、每卡actor/DT4不变；未新增训练组合 |

[准备回执](../../research/temporary/rl_upstream_alignment_20260929/appworld-balanced-padding-resume-20261001/prepared.json)
记录全部文件SHA、旧候选回执SHA和原padding对照回执SHA。
[CPU原始结果](../../research/temporary/rl_upstream_alignment_20260929/appworld-balanced-padding-resume-20261001/cpu-tests.xml)
共46项通过：3项AppWorld配置/载体/恢复参数、9项DT运输、34项padding接口。
这是组成和接口验证；未声称该候选已恢复检查点或已测得正式训练提速。

SQL的`apply_sql_padding_at_update_boundary.py`已于北京时间2026-10-01 03:17提交，
排在首轮原`update_actor`后；提交时两rank仍在该调用内。
[原提交回执](../../research/temporary/rl_upstream_alignment_20260929/sql-padding-live-20261001/submitted.json)
记录提交代码`a01c43c`、脚本SHA、原worker PID和候选SHA。
当前尚未收到完成回执，不能称已应用。生效仍以原worker完成回执
`actor-response-padding/sql-live/complete.json`中两个rank的实际PID、源路径、SHA及
应用时原优化器步数为准，不能把`submitted.json`或本机提交号当成完成证明。
只读记录分别收集待完成提交和完成回执；只有提交时不会将候选标成当前运行代码。
提交记录也不代替进程健康检查。当前提交进程及其原RPC等待已用实际调用栈确认。

## 尚未被证明的结论

2026-10-01：TextCraft已完成两个正式迭代并进入第三轮采样。首轮证据保留在
[首轮回执](results_first_formal_update.json)；后续证据另存于
[第二轮回执](results_textcraft_continuity.json)，没有覆盖首轮优化器观察。
两rank的原AdamW步数各从4增加到8，状态有限且非零；每卡microbatch4、LoRA8/16。
第二轮DT有177773个非零token优势。原日志整轮4546.147秒，其中采样2604.801秒、
DT1066.727秒、actor更新576.073秒、旧概率159.338秒、reference138.754秒；
这是一批256条轨迹、4次原优化器更新，不是一条轨迹或一次B4的耗时。
奖励均值0.586是该训练批次分数；不从两个批次推出效果改善。
console的`grad_norm=0.000`经过三位小数舍入，不能称真实梯度全零；原优化器步数
与状态回执单独记录。该记录不是新增的整网两次更新数值容差测试。

TextCraft随后完成第三轮，原AdamW两rank步数均12，状态有限且非零，每卡4、
LoRA8/16不变。第三轮耗时4681.597秒，其中gen2465.382、DT1343.447、
actor574.862、old_log_prob159.042、reference138.419秒，训练批奖励均值0.660，
优势范围−0.282到0.237。它是正式训练连续性证据，不是新的多步数值阈值。
详见[第三轮原日志及优化器回执](results_textcraft_third_update.json)。

TextCraft第四轮也已完成并进入第五轮生成，原日志耗时4753.725秒：gen2447.970、
DT1433.381、actor574.521、old_log_prob159.013、reference138.382秒，训练批奖励
均值0.633。见[第四轮原日志与后续生成阶段](results_textcraft_fourth_update.json)。
本次没有增加优化器状态读取或数值测试；不从四个训练批次推断奖励提升。

SQL已完成首轮DT并进入原`update_actor`反向，来源为
`phase-observations/1790794409-SkyRL-SQL-1884758-counts.txt`；更新尚未完成。
2026-10-01 03:17：AppWorld六个DT回报类别组均已完成并进入原`update_actor`；
TextCraft进入第四轮DT。SQL和AppWorld首个完整更新、当前正式检查点与更长连续性
仍待实际日志确认。已在原AppWorld worker队列提交只读优化器观察，未重跑训练测试。
B8夹具证明固定参数下的原更新容量，head原容差证明对应算子范围；两者都不能代替
正式任务收益或整条DT训练数值准确性。暂停期间计入的历史elapsed不用于估算当前吞吐。

已撤回的整网两次更新阈值不再作为验收要求；历史零信用、守恒或进程存活也不作为成功证据。

</details>

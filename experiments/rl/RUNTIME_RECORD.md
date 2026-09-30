# 当前运行版本与修复记录

本文件记录运行与修复事实，不定义信用方法；信用规范仍是 [PLAN.md](PLAN.md)。
部署核对于北京时间2026-10-01 04:39；精确采集时间见
[current_runtime.json](current_runtime.json)。阶段会继续推进；本文件记录该次核对结果。

固定编号对应：DT发布`c9cd147`、DT数值参考`fc2e6c2`、VERL官方提交`20bd331`、
输出头/B4修复`dc4e4d7`。完整40位提交号和文件SHA保存在快照；记录文档的Git提交
不是新的算法版本，也不表示远端执行了该提交的全部文件。

本次三组冻结entry、固定DT/FLA和平台补丁均匹配原SHA。SQL已收到两rank的padding
应用完成回执；AppWorld已从旧作业step1恢复到新冻结入口；TextCraft仍为原入口加
已完成的B4 head覆盖。**三个任务的实际代码组合并不相同**，具体见下表。

每项修复按“代码提交 → 实际文件SHA → 原测试及适用范围 → 部署路径/PID/时间 →
替代的旧版本”对应记录。已验证、已准备、已部署和已退役分别标明；未提交完成回执
的运行时修改不计入生效版本。原回执和冻结源码保留，不靠目录名或日期推断。

04:39只读复核：三组PID及创建时间、冻结entry、启动owner文件、已完成覆盖的
文件SHA，以及固定DT/FLA/平台补丁均与各自回执相符。SQL两rank的actor为
`1f862e8bbdaa…`，AppWorld启动actor相同，TextCraft仍为`2b80b938fee4…`。
本次只刷新版本记录，没有更换训练代码或重跑数值测试；版本吻合不扩大原测试覆盖。

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
| SQL-DT | 1876409 / 0,1 | 启动entry=`runs/official-trajectory-20260930-v8/sql-entry`；VERL为SQL v8；当前actor forward为`candidates/official-verl-20bd331-response-padding-20261001/verl/workers/actor/dp_actor.py`，SHA `1f862e8bbdaa…` | 首轮使用SHA `2b80b938fee4…`完成；04:02两rank在原update_actor结束后应用新forward，原AdamW步数均1。模型/优化器保留。旧冻结启动目录没有包含此覆盖，不能直接重放它当作最新版本 |
| AppWorld-DT | 2680224 / 2,3 | `candidates/appworld-balanced-padding-resume-20261001/{entry,verl}`；head `dc4e4d7`、分发 `4c0cbdd`、恢复入口 `2036246`、actor SHA `1f862e8bbdaa…` | 04:01以提交助手`1ac9323`启动；提交时本机HEAD `6a076ec`。旧PID2027456在完成step1检查点后停止；原VERL已确认恢复global_step1并开始生成 |
| TextCraft-DT | 212110 / 4,5 | entry/VERL仍为v7；actor SHA `2b80b938fee4…`；head来自`candidates/official-verl-20bd331-fused-head-b4-20260930`的已完成PID绑定回执 | 尚未部署padding和DT分发候选。冻结v7启动命令仍是旧microbatch1，不能直接按旧命令重启 |

两种actor完整SHA：

- 旧forward：`2b80b938fee442ea5d9273523b6cce2bcc0511b7729aa6cea90333fef5de7cd3`。
- padding修复：`1f862e8bbdaad6fa116d0670772ad41269529a3a1e4a5b1eb383352d0372e9bd`。
  源码封存提交`0c80b41`，对应原VERL padding断言证据提交`44e1149`，
  详细范围见[原对照回执](results_actor_response_padding.json)。该回执中的
  `not_deployed`描述测量当时；后续部署以本节和PID绑定完成回执为准。

SQL完成回执为`receipts/owner-b8-dispatch-20260930/actor-response-padding/sql-live/complete.json`。
AppWorld启动`source.json`记录原检查点、旧PID、提交代码及脚本SHA；新worker不继承旧PID回执。
本次原回执内容、远端文件SHA及实际日志统一保存在
[部署转换记录](../../research/temporary/rl_upstream_alignment_20260929/deployment-transitions-20261001/observed.json)。

当前没有待完成的SQL padding操作；不重复提交该RPC。旧AppWorld仅分发修复候选
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

已核对的head文件SHA前12位（完整值见快照）：

| 文件 | SHA前缀 |
| --- | --- |
| `verl/utils/experimental/torch_functional.py` | `e285c3353bdd` |
| `verl/models/transformers/monkey_patch.py` | `3c78654e0eba` |
| 原 `verl/models/transformers/qwen3_vl.py` | `ebc52fb35812` |

## 修复账本

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
   SQL/TextCraft启动后有PID绑定覆盖，不能直接重放旧冻结命令遗漏它们。确需重启时
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

- SQL首轮原更新完成，两rank原AdamW步数均1、状态有限且非零；随后的padding
  覆盖完成，原TaskRunner已进入下一轮采样。首轮日志16142.743秒（4.48小时），
  其中actor4758.024秒；**这些耗时属于旧actor，不能用来判断新padding版本的速度**。
- AppWorld旧作业首轮完成，两rank各14次原AdamW更新，状态有限且非零，原
  `global_step_1`检查点及完成标记存在。旧作业已在该边界停止。新PID2680224原日志
  明确记载恢复路径、global_step=1及后续生成；尚未完成新版本的一整轮更新，
  不把成功恢复说成已测得完整提速。旧首轮29969.050秒属于此前运行过程。
- TextCraft已完成4个正式迭代并进入第5轮生成；第4轮4753.725秒、训练奖励均值0.633，
  见[原记录](results_textcraft_fourth_update.json)。它仍使用上表旧actor和旧DT分发。

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

<details>
<summary>历史调查与阶段记录：以下“当前、未部署、未完成”仅指各记录当时；当前版本以上方表及带时间快照为准</summary>

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

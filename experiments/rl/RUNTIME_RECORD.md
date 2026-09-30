# 当前运行版本与修复记录

本文件记录运行与修复事实，不定义信用方法；信用规范仍是 [PLAN.md](PLAN.md)。
代码修复基准为 `dc4e4d7`。以下部署快照最近核对于北京时间2026-10-01 00:44，具体时间见
[current_runtime.json](current_runtime.json)。任务会继续运行，快照中的阶段不是实时状态。

固定编号对应：DT发布`c9cd147`、DT数值参考`fc2e6c2`、VERL官方提交`20bd331`、
输出头/B4修复提交`dc4e4d7`。账本引入提交`071b751`只整理记录，不是新算法版本。
完整40位提交号和每组部署组合保存在快照的`version_mapping`中；输出头版本标签
仅在实际文件SHA与对应修复一致时写入，未匹配时明确留空，不猜测版本。

本次核对：三组冻结entry文件、固定DT/FLA及平台补丁文件均匹配原SHA；
AppWorld两份、TextCraft一份运行时完成回执仍对应当前worker PID，生效文件SHA匹配。
这是版本身份核对，不是重新运行数值测试。正式TextCraft更新证据来自`a97bb15`，
观察口径修正来自`a9ca81c`；两者均未改变训练算法，不能当作新的DT数值版本。

## 哪份记录回答哪个问题

| 要确认的事实 | 原始事实来源 | 本机入口 |
| --- | --- | --- |
| 当前三组PID、创建时间、卡号、预算、日志和检查点 | 远端根目录 `active-training.json`，其中 `manifest` 指向当前正式清单 | `current_runtime.json` 的 `jobs`；重新采集后才称当前 |
| 启动时到底用了什么文件 | 各作业 `source.json`、`launch.json`，冻结entry/VERL目录 | 快照中的路径、SHA256、`entry_files` 和 `owner_files` |
| 启动后在原worker上应用了什么 | 对应PID/rank的原RPC完成回执 | `runtime_overrides`，包括生效代码路径和再次核对的文件SHA |
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

以下目录均相对于远端根目录；完整路径和SHA保存在快照。

| 作业 | 当前PID / GPU | 启动目录及运行中覆盖 | 重新启动时必须注意 |
| --- | --- | --- | --- |
| SQL-DT | 1876409 / 0,1 | entry=`runs/official-trajectory-20260930-v8/sql-entry`；VERL=`candidates/official-verl-20bd331-sql-formal-20260930-v8`；B4及新head已写入启动版本 | 旧v4 PID1374630已退出，无完成检查点；不能把旧DT进度当本次进度 |
| AppWorld-DT | 2027456 / 2,3 | entry/LOOP/VERL分别为v6；原worker先应用v7 entropy dispatch，再应用验证过的B4 head | 冻结的v6启动命令仍是旧microbatch1；直接重放旧命令会丢掉运行时修复 |
| TextCraft-DT | 212110 / 4,5 | entry/VERL为v7，环境仍复用冻结的官方TextCraft；原worker应用B4 head | 冻结的v7启动命令仍是旧microbatch1；直接重放旧命令会丢掉运行时修复 |

后两组当前head来源均为
`candidates/official-verl-20bd331-fused-head-b4-20260930`，
完成回执是 `receipts/owner-b8-dispatch-20260930/live-<task>-complete.json`。
其中两rank记录实际microbatch4、rank8/alpha16、原loss配置及生效源码位置。
回执只适用于记录的原worker PID；重启后的新worker必须由新的启动文件或实际回执证明。

已核对的head文件SHA前12位（完整值见快照）：

| 文件 | SHA前缀 |
| --- | --- |
| `verl/utils/experimental/torch_functional.py` | `e285c3353bdd` |
| `verl/models/transformers/monkey_patch.py` | `3c78654e0eba` |
| 原 `verl/models/transformers/qwen3_vl.py` | `ebc52fb35812` |

## 修复账本

每条区分“修复代码”“部署位置”和“证据覆盖”；没有部署回执不能从提交推断已生效。
下表覆盖当前三任务接入及其依赖的修复链，早期探索保留在历史README和原始回执中。

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
2. 当前不重启、不覆盖冻结运行目录；正式任务继续探测。后续确需重启AppWorld/TextCraft时，
   将`dc4e4d7`的对应launcher、`owner_runtime_options.py`及两个owner补丁落实到新的冻结
   启动目录，使用原入口；不能只复用v6/v7旧命令而遗漏运行时覆盖。保留原预算和原检查点。
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

### AppWorld 恢复候选：`2036246`（尚未启动）

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

### SQL padding 调查（未修改实现）

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

SQL和AppWorld首个正式更新、当前正式检查点与更长连续性仍待实际日志确认。
B8夹具证明固定参数下的原更新容量，head原容差证明对应算子范围；两者都不能代替
正式任务收益或整条DT训练数值准确性。暂停期间计入的历史elapsed不用于估算当前吞吐。

已撤回的整网两次更新阈值不再作为验收要求；历史零信用、守恒或进程存活也不作为成功证据。

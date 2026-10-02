# 当前运行版本与修复记录

本文件记录运行与修复事实，不定义信用方法；信用规范仍是 [PLAN.md](PLAN.md)。
最近三组只读源码快照与原阶段观察分别记录，见下方定向回执。
[current_runtime.json](current_runtime.json)保留精确采集时间，
不能把文档更新时间当作三组阶段都已重新采集。

固定编号对应：DT发布`c9cd147`、DT数值参考`fc2e6c2`、VERL官方提交`20bd331`、
输出头/B4修复`dc4e4d7`。完整40位提交号和文件SHA保存在快照；记录文档的Git提交
不是新的算法版本，也不表示远端执行了该提交的全部文件。

SQL PID552842已因整机global OOM退出；最后完成step11，原save_freq60尚未形成
可恢复检查点，没有提交重启。AppWorld PID3232113、TextCraft PID3218909在
21:29只读源码快照仍存活并由原worker日志分别确认完成step10、step40；存活本身
不是训练健康证明。三个任务代码组合不同，具体见下表与PID绑定回执；历史段落
不能代替当前终止状态或新部署。

每项修复按“代码提交 → 实际文件SHA → 原测试及适用范围 → 部署路径/PID/时间 →
替代的旧版本”对应记录。已验证、已准备、已部署和已退役分别标明；未提交完成回执
的运行时修改不计入生效版本。原回执和冻结源码保留，不靠目录名或日期推断。

2026-10-01 05:25的原只读复核没有发现源码漂移。同日05:39的SQL采样事故及随后重启保留在历史账本。
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
SSH端口30821。当前清单仅三组DTPO，SQL已退出，AppWorld和TextCraft继续；GPU6/7不再提交GRPO。

## 当前代码组合

### 2026-10-02 AppWorld 第10次迭代实际工作量复核

[原日志工作量对照](results_appworld_workload_accounting_20261002.json)仅统计同一正式
PID3232113的原日志，不是新的数值/效率实现或已部署版本。原采样1579.803秒
（26.33分钟），DT5215.893秒（86.93分钟），actor2706.845秒（45.11分钟）；
第10次迭代还含2902.114秒独立评估。LOOP图7的42小时和约90次迭代只能推出
约28分钟整轮平均，不能当作官方单独LLM采样计时。

两rank第10次DT各492个B4调用、1968个response对照；累计事实请求context
38,654,496 token，补齐后dense槽位39,053,844 token。批内额外补齐1.0225%；
原PPO完整轨迹输入3,188,772 token，事实请求累计context是其12.1221倍。
这是重复历史的工作量证据，不是FLOPs倍数或提速承诺；现有runner已将端点
共同前缀在一对内只前向一次。原逐response的O(T)调用不再展开未来事件。
跨调用复用需继续核对原HF Cache与正式B4组合，当前没有增加影子Cache。

随后第11次DT的前4个原worker RPC结束处，已执行部署的原host缓存释放，
两rank各次释放约0.48–1.38秒、reserved均回到32,858,176,416 bytes。
这不是第11次完整DT/PPO更新完成；原数值版本、任务参数、LoRA8/16和每卡4不变。
SQL仍停止、没有重启。之前的21:26 TextCraft事实保持，原只读全源码快照仍21:29；
本项记录以其单独原日志回执采集时间为准。


### 2026-10-02 21:26：TextCraft首次正式DT与PPO清理均已执行

[原worker日志回执](results_native_host_cache_boundary_20261002.json)已记录两rank原阶段末
实际调用，不再只是挂载状态：DT释放耗时1.5746/1.6470秒，PPO结束释放耗时
0.3896/0.3889秒；释放后各rank pinned reserved回到32,858,176,440 bytes
（30.602GiB）。这是两次不同phase的读数，不合并成某个同时存活峰值。
原TextCraft step40已完整结束、随后进入正式step41采样；AppWorld step11原DT
正在运行，尚未经过本次阶段末；SQL仍停止。21:29源码快照无DT/FA/FLA漂移。
原actor forward仍1f862e8b…，原卸载/优化器/任务配置没有改。此次原接口已经
在TextCraft连续DT→PPO→生成路径上运行，但原输入工作量各轮不同，不能据此
宣布整轮提速或排除未来峰值；仍按原计时/实际token数比较。

### 2026-10-02 21:06：原阶段结束处释放闲置 pinned 缓存

资源修复代码`6781bdd`，记录器`e945acb`。仅在原 ActorRolloutRefWorker 的整次
compute_dt_token_advantages finally/offload 之后、update_actor 原卸载之后，增加
PyTorch 现有 _host_emptyCache；不在 B4 循环内清理，也没有新分配器、卸载策略或数值
纠偏。VERL 原代码和任务设置保持，默认 VERL_RELEASE_UNUSED_HOST_CACHE 未开启
时原路径不变。三项实际 owner 源码接口检查通过：除资源语句外整份 AST 相同，
DT 原返回/default/异常清理顺序不变。不是新的 DT/PPO 数值对照或容量验收。

[阶段释放与成本回执](results_native_host_cache_boundary_20261002.json)记录：

- 20:50 AppWorld worker3250426/3254727、20:52 TextCraft worker3236008/3240133
  在原 execute_with_func_generator 边界完成挂载；driver PID/创建时间不变。实际
  方法源为 candidates/native-host-cache-phase-20261002/e5eb4afc42f1/fsdp_workers.py，
  SHA e5eb4afc42f10fb4608b3ac43046c906d6a2d21a5387ee02e176bc395f1c6f39，
  原源807e51856f99…；原register分发元数据、model/optimizer对象及config均保持。
- 当前 worker 的资源环境开关为 VERL_RELEASE_UNUSED_HOST_CACHE=1。冻结启动源码
  没有被覆盖；恢复时须同时复用此 owner 补丁和开关，不能仅设置开关就声称生效。
  current_runtime.json 将这些完成回执记为 owner_worker_method_sources；原 actor
  _forward_micro_batch 仍1f862e8b…，不是被该fsdp文件替换。原激活回执的legacy
  effective_forward_source字段名保留，methods字段说明真实替换对象。
- 空闲GPU6只作原生接口小测，最大live pinned为256MiB，不加载模型或训练。4次
  重新分配28.35–38.69毫秒，缓存复用9.51–15.48微秒，释放1.37–1.79毫秒。
  不据此外推整阶段开销，更不能逐minibatch清理。第一次独立probe缺少CUDA init，
  因原host_memory_stats返回空字典而失败；按该API初始化约定补上init后完成，
  正式worker原本已初始化，未为此改任何训练实现。
- 21:06已部署两组四rank，但本次读取的原日志尾部还没有新的阶段释放行，不能
  称下一次正式DT/PPO清理或长期OOM修复已验收。当前MemAvailable590.342GiB；
  原mx-smi物理AppWorld53,638/53,114MiB、TextCraft17,111/17,111MiB。
- 最新原完整指标AppWorld step10：采样26.33、DT86.93、actor45.11、原评估48.37、
  整轮213.67分钟；检查点原完成标记10，训练score0.800、独立val score0.482。
  TextCraft step39：采样31.65、DT9.51、actor9.66、整轮55.81分钟，score0.512。
  这些计算阶段发生在新清理hook前，不能拿来宣传hook提速。
- SQL仍因global OOM停止，没有重启；不把目录存在、记录的新Git号或两组挂载当作
  三组训练目标完成。

### 2026-10-02 20:19：XTT来源已定位，原生闲置缓存释放已实测

本次未改DT/FA/FLA、PPO、vLLM、LoRA8/16、每卡B4或任务预算。使用当前
MetaX PyTorch 2.8.0+metax3.5.3.9原有host_memory_stats及_host_emptyCache，
经现有VERL execute_with_func_generator接口观察并各调用一次；没有安装分配器、
重装库、清理编译缓存或部署持久训练hook。诊断源码be7e4b7，记录器b68c0f4；
实际四worker/PID创建时间、API共享库路径/SHA、原始回执见
[原生主机内存结果](results_native_host_memory_20261002.json)。这两个提交不是新算法版本。

- 四worker原pinned reserved合计321.507GiB（AppWorld每卡97.588、TextCraft每卡
  63.166），与原驱动XTT322.421GiB基本吻合；各worker PSS仅约8GiB，不能用PSS
  替代此项驱动/allocator占用。XTT八设备同值仍不得逐卡相加。
- 9MiB有界原API测试释放8MiB闲置缓冲，保留1MiB仍被引用的张量及异步GPU拷贝
  数值。但allocated_bytes.current增加1、freed=-1；正式counter的数TiB不是物理
  用量。安装header的process_events_for_specific_size(-1)按size而非block->size_
  扣统计，同一行也在PyTorch v2.8.0原源码。没有改计数器或以纠偏掩盖此缺陷。
- 同一原API在正式worker上释放：AppWorld每卡97.588→10.289GiB，用时
  5.084/5.100秒；TextCraft每卡63.166→30.602GiB，用时2.355/2.349秒。
  四次原reserved减少合计239.726GiB。随后全局XTT114.265GiB、MemAvailable
  634.400GiB；期间原任务已继续分配，不能把不同采样时间的净变化强行等同。
- 四次完成回执均为一次性内存操作，记录在current_runtime.json的native_memory_actions，
  不属于runtime_overrides或新部署。还没有持久阶段释放hook，不称内存问题长期修好，
  更不将此次释放称为DT/actor提速。后续需在原阶段边界复用该API，并量化下一次分配
  开销；不能逐minibatch清空缓存或改卸载策略。
- 20:11原非阻塞栈显示AppWorld已退出第10次actor，当前原_validate及LOOP采样；
  迭代末日志/检查点仍为9，不先称第10次完整迭代完成。TextCraft完成step38：
  采样39.486、DT8.654、actor9.623、整轮62.755分钟，原平均奖励0.418。
  SQL仍已退出、无可恢复检查点，未重启；用户恢复决策仍待答复。


### 2026-10-02 19:25：整机OOM与官方卸载候选结果

SSH恢复后只读取原诊断PID3125131及同一回执，没有重新提交。官方分阶段卸载
候选`6ac1fdd`完成原B8x32768夹具；LoRA8/16、每卡B4、原PPO/head/FA/FLA
不变。原old-logprob44.176秒、update480.768秒，两rank各248个可训练张量发生
更新并完成原生LoRA同步。原98.039秒容量记录保留；本次更新明显更慢，且同一
观察时段出现整机OOM，不能归因于单一卸载开关或称为稳定提速。未做新的DT容量
比较，也没有部署该候选。原结果、源码SHA及失败时段见
[候选完成记录](../../research/temporary/rl_upstream_alignment_20260929/phase-offload-20261002-pending.json)。

82个原被动样本中GPU6/7最大各58877MiB，诊断进程树PSS最大54.70GiB，
cgroup最大304.28GiB。这些数字没有覆盖驱动占用及整机可用内存，不能据此
认定1TiB主机安全。原内核明确记录global_oom杀死SQL TaskRunner558046，
driver552842随后退出；cgroup failcnt为0、oom_kill为1，属于整机耗尽。
SQL完成step11后无检查点目录（原save_freq60），未重启，恢复选择待用户答复。
[原内核与资源回执](../../research/temporary/rl_upstream_alignment_20260929/phase-offload-resource-and-oom-1790938987.json)
保留原始时间；dmesg墙钟与Ray日志不一致，不据转换时间认定精确先后或单一原因。

本次只读MetaX原sysfs及`mx-smi --show-memory`，得到相同的XTT读数
338083408KiB（约322.42GiB），八设备显示同值，不逐卡求和。TTM原
kernel/used_memory为341157198KiB；这些驱动计数补充了此前只看cgroup/PSS
的缺口，尚未将其全部归入某个进程或证明具体分配调用。
[原驱动读数](../../research/temporary/rl_upstream_alignment_20260929/native-xtt-memory-owner-1790940064.json)、
[原TTM读数](../../research/temporary/rl_upstream_alignment_20260929/native-ttm-accounting-1790940264.json)。
[MetaX官方指标说明](https://developer.metax-tech.com/api/client/document/preview/%E9%9B%86%E7%BE%A4%E9%83%A8%E7%BD%B2/%E4%BA%91%E5%8E%9F%E7%94%9F%E5%8F%82%E8%80%83%E6%89%8B%E5%86%8C/%E6%9B%A6%E4%BA%91C500%E7%B3%BB%E5%88%97/0.16.0/k8s/03_component.html)
将xtt定义为系统内存；该定义不是本地创造的显存口径。

只读observer修复`f3d6500`、SHA256
`b5ce0035f0332c4113760fe4c566221dcecf913839597eb1dc7cf1ce2e71c91b`，在原
物理显存/PSS/cgroup字段之外读取/proc/meminfo、cgroup OOM计数、原XTT/VRAM
sysfs和原worker进程名。仅在receipts独立诊断目录执行150秒，不修改默认入口、
训练作业、kernel参数或内存策略；语法检查通过，实际采样结果单独记录。
不创建新的资源容差、自动重启阈值或训练健康门槛。

150秒观察已自行结束，14个实际样本跨134.39秒；原observer无错误日志，未留下
长期检查进程。当前AppWorld actor/TextCraft原更新阶段中，整机MemAvailable
452.59–452.67GiB、cgroup174.59–174.60GiB，XTT恒为322.42GiB，原OOM计数
保持1。GPU2/3峰值46836/44350MiB，GPU4/5峰值24213/24213MiB。
这是SQL和容量进程退出后的样本，不外推为三组并行或下一32k测试的安全预算。
[采样摘要与原文件SHA](../../research/temporary/rl_upstream_alignment_20260929/host-memory-observer-20261002-summary.json)。
19:37非阻塞原worker栈确认AppWorld在原FSDP反向、TextCraft已进入原生成；
该栈不能单独拆出具体GPU内核或证明吞吐达标，没有附加mcTracer或暂停进程。

19:25源码快照确认当前entry和固定DT/FA/FLA没有SHA漂移。AppWorld仍原actor
更新，TextCraft完成新step37（采样33.10分钟、DT7.63分钟、actor9.63分钟、
整轮55.36分钟，原平均奖励0.473）。没有把TextCraft加载的旧step25或AppWorld
加载的旧step8算成本次新更新。当前三组尚未全部健康，不能据容量完成关闭目标。

### 2026-10-02 18:20：真实 DT 分段耗时与前缀复用接口核查

本次仅保留正式 runner 原来已经返回的计时，不增加前向重放、CUDA event、同步、
profiler、优化器更新或数值处理。诊断源码提交`8edef25e5b2a8220b62c080f2743cca31edf6d1e`，
脚本SHA256为`955822046103df5737e354800e7e4131a4510daf1dee49b53d6b99233ea5291c`。
正式PID3232113、创建时间1790924078.76，以及两worker3250426/3254727保持；
冻结owner/entry、LoRA8/16、actor/DT每卡4及数值发布c9cd147/fc2e6c2未变。
原调用位于`releases/c9cd147/clean/qwen35/qwen35_dense_finite_runner.py`，
SHA256为`c7fc969f9f521993f2449ea5f364adcb3e0fdac5b01c38c103963639551516c1`。

[分段结果与完整SHA索引](results_appworld_dt_stage_times_20261002.json)保存两个rank各3次
真实>=8k调用及原生Cache源码核查。两rank均已恢复原attribute；原始安装/恢复回执
保留。此诊断不是新的正式训练版本，也不表示已实现缓存加速。

- 真实输入约8.2–8.4k，每次仍是4对事实/EOS端点。6次完整DT平均8.1447秒，
  原历史前缀阶段stream计时平均3.1788秒，是此次返回的最大单阶段；有限decoder
  为1.49–2.23秒。stream时间包含原计算、传输和等待，不等于纯内核或纯拷贝，
  不能把这些计时相加当作独立墙钟成本；样本也不代表整轮或32k。
- 上一完整step9两rank均412次B4；批次耗时和分别4006.47/4005.10秒，
  两卡对应批长平均差36.28 token。本次主要大头不能再归因于明显两卡失衡。
  dense输入槽位15368240/15366964仅作原始工作量计数，不称FLOPs或唯一token。
- 保存的不同轨迹样本共同前缀只有349 token；同一traj_uid不同轮次可共享10k以上
  原始历史。当前每个response重复原历史前向，逐轨迹复用值得进一步核查。
  不根据这一小样本宣称整轮缓存命中率或承诺提速倍数。
- 已查实际安装的HF Cache：原reorder_cache和layer offload/prefetch可处理线性状态；
  通用batch_select/repeat并未覆盖所有线性layer，未提供batch_split/from_batch_splits，
  且GDN缓存不可crop回退。不能自行拼一份影子Cache或把合成奖励询问后的状态用于
  真实后续历史。当前未新增或部署缓存实现，原正式作业继续。
- 18:20只读原日志仍以step9作为最新完整指标；下一批原采样已经保留233条轨迹、
  3914条response、808920个policy token，最长26365，当前DT继续推进。
  官方90%完成门槛可以同时收到多项完成结果，因此233不是自行增加采样配置。
  原尾日志及时间保存在索引中，不把DT进行中称为新完成更新。

效率比较口径保持：[LOOP图7及附录D](https://arxiv.org/pdf/2502.01600)的42小时/
约90迭代是约28分钟的完整迭代均值，作者是8卡生成加8卡学习，并未单独给出采样
半小时。当前step9采样21.07分钟，DT66.92分钟、actor41.59分钟、整轮136.29分钟；
卡时相同不是跨模型/硬件效率证明。本记录不更改官方容差、任务/采样/训练参数或
EOS Q/V/A；SQL候选切换仍待原恢复行为裁定。

### 2026-10-02 17:27：AppWorld原生完整更新9、检查点9及更新后生成已完成核查

正式PID3232113的新step9原日志、AdamW状态、原检查点标记及下一批生成绑定在
[完整迭代回执](../../research/temporary/rl_upstream_alignment_20260929/appworld-official-async-20261002/native-futures/first-complete-formal-update.json)，
SHA256为`e6e4549582c9c4c8e828e28ca86cbe5d6d9cf46b1a7911a605fe66f94651c9de`。
该回执观察于17:21；源码快照单独采集于17:27，三组PID身份、入口及固定数值文件哈希
对应不变。此记录不发布新的训练代码，不更改参数、Q/V/A或任何官方容差。

- 原step9：采样1264.195秒（21.07分钟），旧log-prob350.001秒，DT4014.917秒
  （66.92分钟），actor2495.232秒（41.59分钟），保存51.996秒，整轮8177.214秒
  （136.29分钟）。采样运输的1207.6秒不是这里完整`timing_s/gen`，两者保留原口径。
- 官方早停保留216条轨迹、3278条response、648607个policy token，最长上下文27005。
  原日志平均训练reward0.823是测试通过比例，不是独立评估TGC；grad_norm0.001是
  console舍入值。advantage范围为-1.087至0.566，未以裁剪或缩放纠偏。
- 从检查点8的原AdamW step114恢复后，两rank实际状态均为step128，新增14次
  优化器更新。原状态496项、21639168个moment元素均有限且非零；实际microbatch4、
  LoRA rank8/alpha16未变。只读RPC复用正式owner的`execute_with_func_generator`，
  诊断修正仅使客户端导入同一冻结owner并用其原`func=`签名，不是新的训练模块。
- 原`latest_checkpointed_iteration.txt`为9，global_step_9含data.pt、两rank模型/
  优化器/RNG状态及原配置文件。检查的是原完成标记和非空文件大小，没有声称异机备份、
  全文件SHA核验或恢复测试。随后原正式生成已交付1233次回复、259581个生成token/
  423.3秒，证明更新后已继续下一轮，不把这些token混入step9采样工作量。
- 同一17:21物理mx-smi显示AppWorld两卡55867/55777MiB，容器cgroup约276.45GiB；
  console虚拟allocator78.532/87.117GiB和整机CPU872.255GiB不是该作业物理占用。

作者记录为16张H100（8张采样、8张训练）、Qwen2.5-32B、42小时及图中约90次完整
迭代，约28分钟/完整迭代；没有发布采样单独半小时。当前采样已降至21.07分钟，不能
据此或用卡时算式宣称整轮效率等同作者。当前整轮仍以DT和actor为大头。
SQL的已验证均衡分发/完整rollout候选尚未部署；其原save_freq60在当前step10没有
完整driver data.pt。提前通过原worker API保存并恢复将改变数据顺序，该具体选择
仍待用户裁定，不以候选成绩冒充当前SQL效率。

### 2026-10-02 16:45：AppWorld完成本批DT，PPO更新仍在进行

本次只读原worker日志及PID身份，没有模型回放、运行时覆盖、参数调整或新的容差。
原始观察见[阶段回执](../../research/temporary/rl_upstream_alignment_20260929/phase-observation-20261002/formal-phase-observation-1790930745.json)，
SHA256为`9879a765b1bf1fc801b8a7978ea67a85fd08f407f4877ee3971de87cf85910cb`。
源码快照单独采集于16:37，三个PID身份及固定数值文件对应不变；不能把16:45阶段观察
写成16:45重新核对了全部导入文件。

- AppWorld PID3232113：采样运输1207.6秒/738406生成token；随后7个奖励类别分组均完成，
  两rank各412次B4归因、1648个实际含原DP补齐的response请求，原readout累计
  4009.924/4008.500秒。7个分组不是7次训练迭代。当前原`update_actor`，尚无本次
  新完成迭代或检查点；readout累计秒数也不冒充迭代末的`timing_s/adv`。
- TextCraft PID3218909：已完成本进程新迭代26–35；step35采样1919.486秒、DT538.489秒、
  actor577.473秒、整轮3334.612秒，原平均奖励0.488，随后进入下一次生成。
- SQL PID552842：完成step10、当前下一轮DT。step10采样2721.968秒、DT6418.249秒、
  actor1290.156秒，另含官方验证1200.061秒，整轮11995.090秒。保存的1280条原评分为
  `-1:1262、0:3、1:15`。原SQL reward明确将格式不符记为-1；检查保存的失败文本，
  发现工具观测后未重新输出`<think>`。原Qwen的think标记不是special token，不能将
  此结果解释为`skip_special_tokens`删掉标记；尚未据此修改官方提示、解析或奖励。

DT短批次的非阻塞Python栈停在最终`.cpu()`等队列完成，不能仅靠此栈判定具体GPU内核
或拷贝耗时。已保存的原FSDP对照测出连续长度分片会放大跨卡等待；SQL对应的原VERL
均衡分发及完整rollout上下文候选仍未部署，不把候选实测写成当前SQL提速。
本次物理mx-smi均低于64GiB，cgroup约279GiB/900GiB；原console整机CPU用量和虚拟
allocator总量不当作当前作业物理用量。现有作者时间证据只支持明确列出模型、GPU、
工作量和阶段差异，仍不能称三组已达到作者记录的整体实验效率。

### 2026-10-02 15:32：首批原生生成完成，官方生成比较通过，当前DT

同一PID3232113的正式采样已由原LOOP规则保留216条轨迹、3278条response、
648607个policy token，最长上下文27005。原采样运输循环交付3590次完成回复、
738406个生成token，最后计时1207.6秒（20.13分钟）。它包含prefill、decode、
环境等待及RPC；不是纯decode或迭代末才公布的`timing_s/gen`。原计划、任务、
采样和资源参数未改。只读观察时已进入DT，两rank最近均为B4，未完成新的PPO迭代。
见[正式生成阶段回执](../../research/temporary/rl_upstream_alignment_20260929/appworld-official-async-20261002/native-futures/first-formal-native-generation.json)。

诊断源码`88b5723`只连接本次已唤醒的原引擎和实际同步的LoRA，4个原保存prompt
分别串行及并发生成64 token。原vLLM0.15 `test_batching`的生成助手和
`check_logprobs_close`不改，27.169秒通过，8个输出均使用同一非零正式LoRA ID。
原检查产生一处Test0的top-k分歧告警；通过意味着满足该原检查，不能称逐值相同、
所有log-prob的allclose、PPO更新验收或固定负载吞吐验收。没有HF比较或新容差。
见[原生成比较回执](../../research/temporary/rl_upstream_alignment_20260929/appworld-official-async-20261002/native-futures/native-batching-deployed-owner-final.json)。

先前诊断继承基础环境的旧VERL路径，发送请求前因签名绑定失败；`81d5617`让诊断
解析同一个冻结owner。随后一次额外的Ray内部属性读取也在生成前失败；`88b5723`
删掉这项非必要读取并记录初始化失败。两次都没有发出生成请求或修改训练作业，
旧回执保留，不能作为引擎数值失败或有效速度数据。

对照旧正式step8：原`gen`6217.597秒、运输6207.2秒、681046生成token；
本批运输1207.6秒、738406生成token。两批模型状态/轨迹不同，不能声称严格固定输入
加速比。作者论文42小时/约90次完整迭代也不是采样单独耗时；资源换算不能代替
实际工作量对照。

### 2026-10-02 14:54：原生Future调用契约修复后，从检查点8恢复

PID2463700已在`fit()`加载原检查点8，但首次生成于14:17失败：
原Ray `collective_rpc(non_block=True)`返回批量`FutureWrapper`，原抽象
`execute_model`却执行`output[0]`。没有完成生成回复、DT或新更新。
退出清理日志中的单消息episode不是有效采样或提速证据。14:52确认其已记录的
driver、TaskRunner、两rank、生成服务和LOOP子进程均不存在，GPU2/3空闲；
未为这次恢复停止任何仍运行的进程，SQL/TextCraft保持原身份。

修复`a1ad7b15269d14ef2771684764c392bd7a974d7d`复用安装版本的
`UniProcExecutor.execute_model/sample_tokens`、`RayDistributedExecutor.collective_rpc`
和`FutureWrapper`；仅将原批量结果选为原单输出契约要求的首个结果。
不复制调度器、Future、模型运算或Ray传输，非0.15路径不变。
实际owner文件SHA为`8cb6c3821ac2962e84d91247f244dc4226db2c08a00ec6839f8a2cbe906ac201`。
旧CPU调用复现同一个TypeError；新CPU调用与原UniProc执行入口比较，
阻塞/非阻塞、execute/sample、值/None共8项逐值通过，15.517秒。
这是接口对照，不是GPU数值容差或吞吐验收；初次夹具缺少MM缓存成员的失败也保留。
见[原始对照与失败记录](../../research/temporary/rl_upstream_alignment_20260929/appworld-official-async-20261002/native-futures/probe-after/interfaces.json)。

新PID3232113、创建时间1790924078.76，GPU2/3；冻结入口/owner为
`candidates/appworld-native-async-015-native-futures-20261002`，输出为
`runs/appworld-native-async-native-futures-20261002/appworld-dt`。
仍由原VERL恢复完整检查点8；本次候选相对上一份配置没有参数差异，
LoRA8/16、actor/DT每卡4、任务预算、DT和PPO数值版本保持。
当前状态为恢复已提交；实际首次生成、官方vLLM生成对照和效率结果尚待验证。

### 2026-10-02 14:11：补齐原生manager所需的官方scheduler配置

PID2120244已经越过前两处失败，原GPU引擎完成初始化后，manager却仍等待
`chat_scheduler_ready`。非阻塞原TaskRunner栈确认它停在
`async_server.py:281`，尚未进入`fit()`；后台线程因原sync默认的
`chat_scheduler=None`报错。错误在接入时启用async而没有补齐该官方配置。

固定owner的`ChatCompletionScheduler`已经提供所需构造器与就绪流程；
修复`d3b762b3ae7bd2dad2489c2d609f22c7371f1899`只选择这个原类，不复制或修改
manager、scheduler算法。LOOP任务仍直接使用原生生成服务；不改变任务、奖励、
采样、训练或资源参数。实际候选launcher组合对照的唯一差别为
`rollout.chat_scheduler: None → verl.workers.rollout.async_server.ChatCompletionScheduler`。
CPU测试实际运行原线程入口、原scheduler构造器及就绪信号，33.318秒通过；
无推理请求或模型权重加载。见
[官方scheduler配置回执](../../research/temporary/rl_upstream_alignment_20260929/appworld-official-async-20261002/native-scheduler-config/interfaces.json)。

只停止经创建时间确认的未完成初始化进程树，原检查点8完整保留，另两组PID未变。
新PID2463700、创建时间1790921486.96，GPU2/3，冻结入口/owner为
`candidates/appworld-native-async-015-owner-scheduler-20261002`，输出为
`runs/appworld-native-async-owner-scheduler-20261002/appworld-dt`。
仍由原VERL恢复检查点8；实际GPU LoRA同步、生成数值与吞吐须以新作业验证，
不能由这次CPU配置校验宣布效率对齐。

### 2026-10-02 13:52：AppWorld从原检查点8提交原生异步修复版本

原AppWorld PID1199302已完成第8次更新和双rank的model/optimizer/RNG/reader
检查点，原标记8确认后于13:05停止。一次性观察助手已退出，不重复启动。
本机SSH通道在远端助手退出后未返回；按实际进程创建时间终止了仅该通道及其
助手，修复`6dbf281`使用OpenSSH原生keepalive，不添加重试实现。

随后两次原生异步启动均在原`trainer.init_workers()`失败，尚未进入`fit()`，
未加载检查点或产生新更新。PID1414890将`swap_space=None`传给原生配置；
`b31b695`恢复原同步owner已使用的None表示未覆盖规则，默认值由vLLM负责。
PID1800022暴露兼容分支遗漏`is_version_ge`导入；`332427104fa7e88b6c16dda612a9a3b61faf4288`
只增加原VERL版本助手的局部导入。旧候选、失败日志及哈希均保留。

扩展现有CPU接口测试，实际执行原`execute_method → init_worker`，在原生
`WorkerWrapperBase.init_worker`设备初始化入口停止。旧源码复现NameError，
只增加该导入后rank0/1通过，32.916秒；未创建GPU引擎或加载模型。
实际原生配置仍为BF16、32768、max_num_seqs32、LoRA rank8，训练alpha16和
actor/DT每卡4保持。对照回执见
[导入修复记录](../../research/temporary/rl_upstream_alignment_20260929/appworld-official-async-20261002/worker-import-fix/draft.json)。

新PID2120244，创建时间1790920325.04，GPU2/3，冻结入口与owner位于
`candidates/appworld-native-async-015-worker-import-20261002`；输出为
`runs/appworld-native-async-worker-import-20261002/appworld-dt`。
只由原VERL恢复完整检查点8；SQL PID552842、TextCraft PID3218909未改变。
当前只是正式恢复已提交，实际引擎初始化、GPU LoRA同步、生成数值和吞吐尚待
原作业证据。CPU通过、目录或进程存活不作为效率对齐或训练健康证明。

### 2026-10-02 12:48：原生异步数值对照入口已准备，未执行GPU生成

诊断源码`d983a94a760a577de76352e1249e12e201636422`，脚本SHA `de747e6a0757…`。它只向已唤醒的正式原生
AsyncLLM提交原保存prompt IDs，不创建引擎、模型或唤醒/休眠上下文。
仍执行vLLM0.15原`test_batching`的生成方法及`check_logprobs_close`，
保留64输出token、top5和原断言；比较串行与并发提交，正式请求可以同时存在，
不能称孤立batch1对照或吞吐验收。原默认独立测试行为保留。
CPU检查通过实际`SamplingParams`与当前原运输构造器的所有字段；仅诊断
阶段/失败标签变动后的carrier AST完全相同，原始回执保留。CPU结果不表示
实际生成数值通过。源码、fixture与原测试文件哈希见候选的
`native-batching-diagnostic/ready.json`、`sampling-payload-cpu.json`和
`source-identity-cpu.json`。正式更新未停止、冻结候选owner/entry未修改。

12:48非阻塞栈确认当前AppWorld两rank都在原`dp_actor.update_policy`的
`loss.backward()`；原完成检查点标记7，一次性助手PID61821创建时间
1790913374.59仍存活。第8次DT已完成：各400个B4、各1600个含原padding的
contrast，累计归因计时3717.371/3715.487秒，输入槽位14,669,888/
14,667,296。这些是现有日志计时，不是新性能测试，也不作为数值精度证明。
原worker同一stdout有两个打开描述符；计数按唯一路径读取一次，避免把
重复观察误报为DT重复计算。原错误观察与更正指针均保留，见
`checkpoint-observer/dt-workload-step8.json`；与step7不同输入的耗时差不能
称固定工作量提速。

### 2026-10-02：已冻结原生异步恢复入口，尚未部署

准备源码`b6e470b7428ff4ac4a1968d2ba247b1a7ffe4e28`从当前PID1199302的
冻结来源继承，目录为`candidates/appworld-native-async-resume-20261002/entry`；
owner复用已检查的`appworld-native-async-015-transport-20261002/verl`，
未覆盖原CPU候选、默认入口或当前正式文件。逐文件SHA、实际配置组合对照和
原测试回执见
[恢复准备回执](../../research/temporary/rl_upstream_alignment_20260929/appworld-official-async-20261002/resume-preparation/prepared.json)。

执行两个实际launcher的`options_for`后，排除部署路径本身，唯一配置差别是原
`rollout.mode`从继承的`sync`选择为官方`async`。采样参数、任务预算、LOOP
完成/取消行为、LoRA8/16、actor/DT每卡4、PPO及DT数值文件均保留。
新入口显式选择原`AsyncActorRolloutRefWorker`、原manager/server及原vLLM调度。
原第8次采样运输已结束：3553次完成请求、681046生成token、6207.2秒；
当前原检查点标记仍7。只有原第8次或之后的完整model/optimizer/RNG/reader
检查点完成，才使用既有恢复助手切换；不重放第7次、不丢弃本轮更新。
准备和CPU接口通过不作为实际引擎初始化、GPU LoRA同步、数值或提速验收。

11:57已核实一次性检查点观察助手存活：本机PID7940，远端PID61821、
创建时间1790913374.59，代码`e699459d50a08ce95f7dcdee5732074206e11ad7`。
原AppWorld PID1199302仍在运行、检查点标记7；助手仅等待原标记至少8后调用
已有停止/提交入口，不立即停止当前更新。见同候选的
`checkpoint-observer/submitted.json`及原远端`waiting.json`。
它是排队中的一次性转换，不是已经部署；后续以`completed-stop.json`、新
`active-training.json`、原loader日志及实际新worker回执确认。不要重复启动助手。

### 2026-10-02 11:00：原生异步运输候选，仅 CPU 接口验证

11:32追加的原生接口检查通过（8.987秒）：原FSDP sharder经原生
`WorkerWrapperBase`传递同一LoRA张量、rank8/alpha16和adapter ID；
原server/rollout的sleep/wake调用同一sharder上下文。实际Ray CPU actor的
完成future经原collector返回：同服务短请求已交付时，长请求仍未完成。
这些载体使用记录的`RequestOutput`，不是实际模型生成或GPU LoRA同步验证。
回执见候选的`transport-cpu-probe/probe-lifecycle/lora-delivery.json`。
原生分布式初始化源码和SHA保存在同目录`installed-lifecycle-sources.json`；
不根据`/proc/environ`的初始mask猜测运行时GPU分配，不改原FSDP通信组。
检查助手PSS末值5.29GiB，不作为峰值。正式默认入口和当前PID均未改变。

正式三组未切换异步路径。11:00只读阶段回执确认SQL PID552842仍为第9次迭代，
AppWorld PID1199302仍为第8次采样（3124次完成请求、608442生成token、5457.6秒），
TextCraft PID3218909已完成第29次迭代。这个阶段采集不刷新上方10:04的源码快照时间。

候选源码提交`e3230713478a10434f30dbd06607e01df6a8d7d1`，冻结目录
`candidates/appworld-native-async-015-transport-20261002`，状态是**未接受、未部署**。
候选从当前AppWorld冻结owner/entry复制，原始四文件保存在`originals/`；
只适配固定VERL原AsyncvLLMServer/WorkerWrapperBase与已安装vLLM0.15的API，
复用原Ray RPC、原AsyncLLM生成/取消及原manager生命周期。
输出转换从原`vLLMRollout.generate_sequences`移为同模块的共享函数，
原同步方法调用同一组语句；没有再实现一套token、位置、mask或log-prob转换。
现有配置的seed和engine_kwargs传入原生参数，LoRA8/16、max_model_len32768、
max_num_seqs32及所有训练/环境参数不改。

远端原生接口检查18.114秒通过；运输对照51组、实际RequestOutput的三组
停止/长度/空回复检查、原生异步方法签名和分别完成的接口检查通过（0.172秒，
这是记录输出的CPU测试，**不是实际LLM生成时间或数值容差**）。同步上下文
4项本机测试通过。CPU助手未构造AsyncLLM引擎或载入模型；接口助手末端PSS
5.34GiB、运输助手0.57GiB，不作为峰值或物理显存测量。

正式默认`loop_owner_rollout.py`和`owner_rollout_scope.py`保持已部署版本。
未接受的三个entry文件只保存在
`research/temporary/rl_upstream_alignment_20260929/appworld-official-async-20261002/transport-candidate-entry/`；
兼容补丁不在默认launcher/patch调用链中。准备和检查回执见同级
`transport-cpu-probe/`，原始第一版接口候选及其回执仍保留。
实际原生引擎初始化、LoRA同步、正式采样吞吐尚未在该候选上测量，
不能把CPU运输验证称为已部署、已提速或训练验收。DT数值核心、Q/V/A、
PPO及官方容差没有变化。

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
| AppWorld-DT | 1199302 / 2,3 | entry=`candidates/appworld-batch-coalescing-20261002/entry`；VERL仍为`candidates/appworld-rollout-scope-20261001/verl`；bridge SHA `036977b4eb1f…`，其余entry、VERL和数值源未变 | 05:06在旧PID150275未完成的第7轮采样中切换，05:07仍从PID2479539的原完整step6恢复；原loader和两rank原model/optim/extra_state路径已确认。原Ray ActorPool空闲接口恢复请求合批；第7次完整迭代及原检查点标记7已完成，整轮259.819分钟，当前第8轮采样。恢复的6不计新更新；不同任务批次不能当固定输入加速比 |
| TextCraft-DT | 3218909 / 4,5 | entry=`candidates/textcraft-rollout-scope-20261001/entry`；VERL为同目录`verl`；head/padding、每卡B4、整段生成上下文及原DT均衡分发均固化在启动源码 | 07:00:51从旧PID212110的原完整检查点25恢复；双rank原model/optim/extra_state加载及设置step25已确认。10:04原trainer已打印完成26–28，恢复的25不计新更新；第28次整轮56.310分钟。旧v7/运行覆盖只保留历史证据 |

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
| AppWorld合批修复已部署，但仓库默认入口仍是旧同步文件 | 源码`4fa72d6`，SHA `036977b4eb1f…` | 06:12默认`loop_owner_rollout.py`同步为同一已通过10项接口对照、已部署到PID1199302的文件；逐字节一致，不引入第三份实现或重启作业 |

新增测试是[SQL/TextCraft负载对照](test_official_workload_local.py)及
[AppWorld负载对照](test_loop_model_entry_local.py)。负载文件7项（新增SQL/TextCraft各1项，
其余为已有单位回归）、新增AppWorld 1项通过；
测试代码固定为`1a1e945`；最终独立回执中，两组进程分别7.359/11.250秒，
采样RSS峰值146.7/254.3MiB，物理显存前后均0MiB。
这是配置与原单位换算检查，不替代DT、PPO或vLLM的数值验收。
AppWorld初次缺少本机APPWORLD_ROOT；测试夹具使用作者原dev清单补全临时布局后通过，
初始失败与最终结果均保存在
[workload-regression-20261001](../../research/temporary/rl_upstream_alignment_20260929/workload-regression-20261001/summary.json)。

**尚未全部对齐的是执行效率路径。** SQL的整段rollout上下文与DT跨卡均衡
候选已验证、尚未部署；TextCraft已在原检查点25恢复部署该组合。AppWorld跨卡返回顺序修复
之后的正式日志又暴露单条RPC累积，已用原ActorPool.has_free恢复合批；新接口CPU测试
10项、逐文件/参数对照、原恢复证据与实际B31分别记录。TextCraft新组合第26次完整迭代
实测62.005分钟（DT11.384分钟），旧第25次81.192分钟（DT22.064分钟）；批次内容不同，
不能称为固定输入的加速比。AppWorld新完成第7次迭代为259.819分钟，采样128.555、
DT75.869、actor47.730分钟；仍不能将成功恢复或批量变大写成正式训练已全面提速。
效率标准是对照作者记录的实验耗时与吞吐，并明确模型、资源、实际token工作量及执行
方式的差异；同卡原生调用profile用于定位差距，不能替代作者实验参照或另设分钟门槛。
当前证据不足以称三组整体耗时已达到官方同等计算负载的水平。不能缩小任务预算、
改minibatch或部署未对照的异步实现来消去这些待完成项。

### 作者实验时间与当前正式采样（2026-10-02）

来源、单位和实测原日志绑定在[时间口径回执](results_official_efficiency_reference.json)。
LOOP论文附录D/图7记录Qwen2.5-32B、16张H100（8张采样、另8张训练）共42小时。
图中约90次训练迭代，折算约28分钟/完整迭代；这是据图估算，不是作者单独公布的采样
耗时，也不是仓库200次预算的平均。当前AppWorld Qwen3.5-9B两张MetaX C550采样
128.555分钟、整轮259.819分钟。未经工作量与硬件吞吐对照，不能据此认定相近或把
整轮约9.3倍的原始时间比当作相同资源下的实现慢倍数。SQL作者公开W&B报告当前
匿名读取失败；尚未取得TextCraft固定配方的可匹配绝对耗时。这两项不填猜测值。

当前AppWorld PID1199302的原vLLM一次性profile已完成并恢复两rank原调用，
[原回执](../../research/temporary/rl_upstream_alignment_20260929/appworld-batch-coalescing-20261002/native-profiler/installed.json)
绑定观察脚本`3242ca1`、实际worker1213516/1217549、原ProfilerConfig/Wrapper SHA。
每rank只观察已有16请求批次的前16个worker迭代，不新增生成、修改采样或数值。
全调用130.779/68.163秒包含profile开销，不能当无profile吞吐；缓存命中104448/121408
与112128/142902。原profiler完整调用计数1502/538步，生成4318/3032 token，
平均2.875/5.636 token/worker步（包含prefill，是工作量代理，不是精确decode活跃数）。
当前原LLM.generate整批完成才返回，同批短请求的环境推进等待长请求；当前bridge已
避免跨rank按顺序取结果，但这不等于作者在线服务在批次内部持续接收新请求。
仍需通过兼容的上游原生异步接口处理，未部署新scheduler、异步候选或推理版本。

AppWorld第7轮DT各rank的所有返回类别共448批，而不是末类别的19批；
[原日志汇总](../../research/temporary/rl_upstream_alignment_20260929/checkpoint-boundary-20261002/appworld-step7-dt-workload.json)
计时和为4543.012/4541.535秒，解释75.869分钟总DT阶段。两卡完整输入槽约35.82M，
既有fact/ref共同前缀复用、FA/FLA后缀传播已开启，因此该槽数不能直接当全部反向FLOPs。
信用、原PPO、LoRA8/16、每卡B4、任务预算与官方容差均未改变。

已核对的head文件SHA前12位（完整值见快照）：

| 文件 | SHA前缀 |
| --- | --- |
| `verl/utils/experimental/torch_functional.py` | `e285c3353bdd` |
| `verl/models/transformers/monkey_patch.py` | `3c78654e0eba` |
| 原 `verl/models/transformers/qwen3_vl.py` | `ebc52fb35812` |

## 修复账本

### 2026-10-02 07:00：TextCraft从原检查点25完成恢复

旧PID212110已完成原保存点25；原边界助手06:54完成停止，随后SSH控制连接重置，
导致本机助手在提交新作业前退出。此处是控制传输失败，不冒充训练、保存或模型故障；
原stderr和completed-stop保留。确认旧进程已停止且新目标目录尚未创建后，仅调用一次
原`submit_prepared_appworld_resume.py`提交现有完整候选，没有重新停止或重跑GPU测试。

- 提交仓库commit：`7831caf8f019eec1a0f069cb5a05b1ccddc60e0b`；提交脚本SHA
  `b64fe522bba9829e63155a23c9b0b8b4b92b9e4f33f9957d7f822f30b512d504`。
- 新PID3218909，创建时间1790895651.16；启动时间1790895651.6807766。
  原检查点为`runs/official-trajectory-20260930-v7/textcraft-dt/checkpoints/global_step_25`。
- 候选prepared SHA：`d6ced8cba268b9d2a055010f9d62d9a0af12c636f94c956b9b4849d8d1ae8dc9`；
  launcher SHA：`67487af4de55233946ff80eb2d136b4afa9c46184d9cfdec791d39c3f95590e8`。
- 修复来源仍为head/B4 `dc4e4d7`、DT分发`4c0cbdd`、原rollout上下文`2a32d00`、
  padding源码`0c80b41`及原比较`44e1149`；DT数值发布`c9cd147`未改。
- 实际启动LoRA8/16、actor/DT4、32组×8条、global mini64、PPO epoch1、30 epochs、
  30次交互、原保存25及原评测设置保持。原恢复参数仅resume_path和原检查点路径。
  已有49项接口结果不重跑；[启动前配置回执](../../research/temporary/rl_upstream_alignment_20260929/checkpoint-boundary-20261002/textcraft-pre-resume-configuration-20261002.json)
  仅证明配置、源码与原prepared一致，不当作新数值验收。
- 双rank原加载路径及设置step25见
  [07:11观察](../../research/temporary/rl_upstream_alignment_20260929/checkpoint-boundary-20261002/speed-observation-1790896266.json)；
  第26次完整迭代及原阶段计时见
  [09:00观察](../../research/temporary/rl_upstream_alignment_20260929/checkpoint-boundary-20261002/current-phase-1790902771.json)。
  新目录尚无新的原保存点；加载的25和完成的26严格区分。

快照现额外保留原配置文件/model配置/tokenizer配置SHA、完整启动options及已配置环境脚本SHA。
AppWorld记录实际LOOP README、ConfigStore及Hydra配置树；可选YAML被记录不表示全部被消费。
这些只读指纹不增加启动门槛或算法验证标准，也不把记录脚本的commit当成训练版本。

### 2026-10-02 06:13：同步默认源码，保留真实验证及部署边界

发现默认`experiments/rl/loop_owner_rollout.py`仍为旧SHA `22ff649007bf…`，而正式
AppWorld已运行`4fa72d6`中的SHA `036977b4eb1f…`。06:12将默认文件直接同步为
那份已测试、已部署的相同字节，避免以后从默认入口重新带回旧RPC行为。
`__init__`、`close`、`to_batch`的AST与旧默认文件相同；只有请求运输方法不同。
原Ray/VERL接口10项通过的回执继续对应相同完整SHA，没有重跑模型、扩展原数值
容差或把运输测试称为整条训练验收。运行中的冻结entry没有修改。
[默认源码晋升对应记录](../../research/temporary/rl_upstream_alignment_20260929/appworld-batch-coalescing-20261002/canonical-promotion.json)
保留旧SHA、原代码提交、测试回执SHA、部署PID/创建时间及不变方法。

本次06:08非阻塞原栈显示：SQL第8轮已经进入DT，AppWorld恢复6后的第7轮仍在
生成，TextCraft第25轮仍在生成；06:13原TextCraft日志推进到28/30轮。原保存点25
助手452582仍存活、创建时间匹配，尚无completed-stop，候选未部署。SQL也尚无
完整reader检查点；不补造data.pt或静默改变恢复行为。三组实际entry与原数值源
SHA仍匹配，当前snapshot没有新数值版本。
[原阶段回执](../../research/temporary/rl_upstream_alignment_20260929/checkpoint-boundary-20261002/phase-observation-1790892519.json)
SHA=`3279489c73c68c89aad635a2a4dc583aa76cc6c2c089d5ec9817969fb5a85726`。

另核对SQL真实step7的原生奖励：1280条中1264条为-1、9条为0、7条为1，均值
-0.98203125。原SkyRL评分器的-1是格式未通过，0是格式有效但SQL结果未通过；
额外VERL无效动作惩罚关闭。实际Qwen tokenizer的think标签不是special token，
skip_special_tokens不会删除它们；实际官方qwen3_acc_thinking模板的assistant
前缀也没有注入空think。这个结果排除了两项接入假设，没有修改原奖励、模板、
采样参数或训练信号；低任务得分也不被宣称为数值容差故障或训练健康证明。
[原奖励来源回执](../../research/temporary/rl_upstream_alignment_20260929/rollout-scope-20261001/native-score-cause-20261002.json)
SHA=`c038a84b3523a1a0dee8a61c0ae198baf74134ce5c0840e07cbf6fd1f461dd2d`；
[CPU tokenizer边界回执](../../research/temporary/rl_upstream_alignment_20260929/rollout-scope-20261001/native-tokenizer-boundary-20261002.json)
SHA=`692acb34ce0db3fe31f83ab35a605c273cd0bacdb73236d8cd9dc7e8718894f7`。

### 2026-10-02 05:32：真实阶段核对，避免把观测统计误作推理容差失败

三组原TaskRunner的非阻塞栈已确认：SQL第8轮生成、AppWorld恢复6后的第7轮
生成、TextCraft第24轮actor更新。源码与原数值文件没有漂移。
[原日志与栈回执](../../research/temporary/rl_upstream_alignment_20260929/checkpoint-boundary-20261002/phase-observation-1790890330.json)
同时核对TextCraft原保存点25助手452582存活、创建时间匹配且尚无completed-stop。
后续05:40原日志已显示TextCraft完成24（奖励0.516、迭代4731.084秒），进入25采样；
这个完成值晚于05:30源码快照，不能把快照里的23当作新的停滞。效率候选尚未部署。
AppWorld继续合批，05:40已返回1011个请求、221064生成token，采集墙钟1807.5秒；
没有新完成迭代，这些请求数不是完成的轨迹数。

SQL完成step7的`training/rollout_probs_diff_mean=0.283`不能直接当作vLLM动作概率
容差失败。实际固定VERL的原指标使用整段response的attention_mask；官方SkyRL
给观测位置填0.0 rollout logprob，这些位置指数后为1，原指标会包含其与模型概率
的差。初始prompt若落入完整轨迹response区域也被该指标纳入。
当前原多轮actor的策略、熵和KL损失使用loss_mask，排除这些非动作位置；既有
原环境/轨迹对照也覆盖该mask。没有改变原统计代码、容差或训练信号，也没有由
这个统计口径推断动作token的数值误差为零或推理完全无误。

[实际owner源码与统计口径回执](../../research/temporary/rl_upstream_alignment_20260929/rollout-scope-20261001/native-probability-metric-scope-20261002.json)
SHA=`6da73bbeb3dde7c708a671c425a37b2143bfba2b8cff03335439b403f4db259f`。
trainer仍`8816ea4e…`，actor仍`1f862e8b…`，generator仍`1e727204…`；这次调查
没有新的生产代码版本或GPU测试，也没有改动已通过的官方数值验收范围。

### 2026-10-02 05:23：修复忙卡时过早拆分RPC，保留原学习负载

上一版`8e93ec14…`已通过返回身份、取消、异常和跨卡完成顺序测试，却没有覆盖
两张原actor忙时连续到达请求的工作量。正式旧PID150275的只读记录中，非空返回
290次为B1、3次为B2，仅4次为B9–12；请求被过早拆为固定单条RPC，进入原ActorPool
待执行队列后无法合批。这是接线引入的实际回归，不是作者任务预算造成。

`4fa72d6`只调用原`ActorPool.has_free()`：原actor忙时保留原请求；actor可用时再由
原VERL preprocessing/padding/chunk和原单worker RPC提交。没有新增等待计时器、
batch上限或调度器；原LOOP采样、完成/取消、结果身份及任务/训练参数保持。
同一34条请求、完全相同token/logprob/artifact的CPU对照中，旧路径34次B1，修复后
4次调用（两次B1、两次B16）。原Ray2.53.0及实际VERL RPC边界共10项通过；
pytest73.38秒、整个监测进程90.90秒，进程树PSS采样峰值4.36GiB，0 GPU。
这是运输与调用工作量证据，不是8.5倍模型加速或新的算法数值容差验收。

版本对应固定为：代码`4fa72d6` → bridge完整SHA
`036977b4eb1f5cb11a9fb0370b387d207cb1c8d131df9bef4fbaf03c1b976498` →
[10项CPU回执](../../research/temporary/rl_upstream_alignment_20260929/appworld-batch-coalescing-20261002/final-cpu-tests.json) →
远端`candidates/appworld-batch-coalescing-20261002/entry` → PID1199302，
创建时间1790888812.77。准备时`source.json/prepared.json`的prepared标记保留为历史，
后续是否部署以本次原`source.json`、PID及恢复证据为准。

05:06非阻塞原TaskRunner栈确认旧PID150275仍在原trainer第1097行
`traj_collector.multi_turn_loop(is_train=True)`，尚未进入这轮更新。只丢弃未完成
第7轮轨迹，重新交给原loader加载原完整检查点6；没有补造reader、复制loader或
丢弃新完成更新。转换助手首次`0ac2076`误查不存在的接口名，停止前即失败；
失败回执保留，`80a2742`按实际原调用修正后完成转换。

新旧实际launch只有输出/观察/数据入口路径不同；数据入口内容SHA相同，恢复路径仍
是同一原检查点6。其他entry及全部VERL SHA一致，DT仍`c9cd147/fc2e6c2`，输出头仍
`dc4e4d7`，LoRA8/16和actor/DT4、PPO/loss/优化器及作者完整负载均未改。
原TaskRunner设置step6，两rank分别读取原model/optim/extra_state文件。
05:22正式采样已返回483个请求、100727个生成token，采集墙钟720.1秒，已出现B31，
但尚未完成新迭代。`queued_requests`只统计原IPC队列，不含本地保留的pending；
B0是取消/padding运输空项，没有调用vLLM。

[实际部署与参数差异](../../research/temporary/rl_upstream_alignment_20260929/appworld-batch-coalescing-20261002/deployed-observation-20261002-0523.json)
SHA=`874e63fb13507ab02b3b67cff5fc0ebbefb448e99ef6f5ea0a5fbb498e55b2d2`；
[双rank原加载行](../../research/temporary/rl_upstream_alignment_20260929/appworld-batch-coalescing-20261002/original-load-lines-20261002.json)
SHA=`d1aeb750a744e7c4d7a19b2a0769d6c3d739b6ab2a2dabbbdffd32b602f70387`。
首次只读加载过滤器漏掉原日志大写`Loading from`，后一个回执补录原行；前一个原回执
保留。原始`initial-evaluation-stack.json`命名不准，实际栈是第7轮训练采样，不是评估。
05:22物理GPU2/3为54896/50756MiB；全容器memory usage约248.5GiB，不是本作业RSS，
这些瞬时值不冒充峰值或训练健康证明。

SQL PID552842最近完成7，TextCraft PID212110最近完成23（05:12原快照）；两作业
身份保持。TextCraft原保存点25的助手仍为本机18596/远端452582，已执行脚本仍为
`ddc3a57/b3210b11…`，不能用磁盘上已更新的助手源码替代其实际加载身份；
原完整切换候选尚未部署。SQL的完整上下文候选也尚未部署，不能称三组效率已全对齐。

### 2026-10-02 04:28：原检查点6恢复与训练参数逐项核对

原检查点6保存完成后，`ddc3a57`助手停止已绑定身份的AppWorld旧PID2479539；
现有提交助手调用同一VERL的原loader，04:08提交PID150275，创建时间1790885285.16。
实际entry的`loop_owner_rollout.py`由`22ff6490…`替换为`8e93ec14…`，其余entry及
VERL文件SHA与旧source逐项相同；候选仅复用原Ray ActorPool/VERL RPC改变返回运输。
不复制原checkpoint loader，也不改变DT、Q/V/A、PPO、LoRA8/16或每卡B4。

新旧实际`launch.json`的完整options逐项比较，差异仅为输出/日志目录、内容相同的
dataset入口路径、worker观察目录与恢复检查点路径。采样组数、完成/取消规则、训练
mini/epochs、任务/生成限制、loss和优化器设置没有变化。原TaskRunner明确打印
`Setting global step to 6`；worker169930/173822分别打印原rank0/1的model、optim、
extra_state加载路径，随后进入原生成方法。恢复的6不是本次完成的新更新。

本次原始来源、逐项配置差异和双rank加载行保存在
[恢复与负载核对回执](../../research/temporary/rl_upstream_alignment_20260929/checkpoint-boundary-20261002/observed-transition-20261002-0428.json)，
SHA256=`8c1d253c42f7cee0b7d35bd3262396a9a78f97e674805e4553db697acb5e30bd`。
旧等待进程4044066已在完成转换后退出，原等待回执和`completed-stop.json`保留；
不再把它的退出报告为训练故障，也不重复提交同一转换。

TextCraft原PID212110继续运行，最近完成23。其完整候选的entry/owner/DT SHA再次与
prepared回执核对一致，复用原49项CPU结果，没有重跑GPU测试。04:24启动同一经过4项
真实OS进程检查的助手：本机18596、远端452582/创建时间1790886289.58；只等待
**原保存点25**，不改保存频率。等待回执SHA为`c043bd0641e67ba03dd810311eba65df621e205566e202e95e3e06ee34818dbd`；
当次核对无completed-stop，不能称TextCraft效率候选已部署。
助手源码仍为`ddc3a57`/`b3210b11…`；其运行时仓库记录`d640835`是文档提交，
不是新的助手或数值实现。

SQL仍为PID552842，最近完成6；它的上下文/分区候选尚未部署。当前原保存点60之前
无完整reader检查点，不能用自己补造的data.pt或静默重置reader来恢复。
三组官方学习负载的8项单位/launcher回归仍对应`1a1e945`；本次只读恢复核对没有
扩大原数值验收范围。全局学习负载已修复与效率修复全部生效是两个不同结论。

### 2026-10-02 03:51：删除部署助手对其他作业的额外停止条件

原助手`e87c593`在等待开始时记录其他作业PID，停止本组之后又要求那些PID仍相同。
另一组合法恢复或独立退出就会让本组已经停止、却无法继续提交。这不是VERL的恢复
条件，也不是用户要求。`ddc3a57`改为在本次停止边界读取权威清单的其他作业身份，
仅作观察记录；删除这项提交前断言。当前目标作业的PID创建时间、准备源码SHA、
原完整checkpoint文件和剩余进程检查保持不变，仍调用原VERL保存/加载行为。

本机复用Python3.11和已有`rl_local_test_deps`，实际执行助手生成的Python体，
用隔离的真实OS进程验证4种情形：其他作业不变、等待中正常替换、独立退出、
准备源码SHA错误。4项通过，原JUnit进程时间5.595秒；其中合法替换一项先执行
旧助手，确实复现“本组已停止但因其他旧PID变化而拒绝提交”。这是进程生命周期
检查，**夹具checkpoint只是占位文件，不能当成VERL恢复、训练或数值验收**。
测试源码、SHA、原失败及最终回执见
[checkpoint-observer-local-20261002](../../research/temporary/rl_upstream_alignment_20260929/checkpoint-observer-local-20261002/summary.json)。

03:46只退役旧远端等待进程2374706，原AppWorld PID2479539及两个worker均保留，
原marker仍为5、两个worker正在原`update_actor`中。旧等待回执及停止助手的记录
保留在
[退役回执](../../research/temporary/rl_upstream_alignment_20260929/checkpoint-boundary-20261002/retired-e87c593-observer.json)。
旧本机控制句柄17944及其SSH子进程随后关闭；这不是训练停止或训练失败。

03:51新本机助手PID16024启动，同一AppWorld候选、同一原检查点6；远端等待进程
4044066，创建时间1790884266.51。快照核对其源码SHA
`b3210b11fd8f0235c92ca521bba316e5d63625512c85729493d83d475127e318`
与`ddc3a57`一致，实际存活；尚无`completed-stop.json`。三组训练PID、数值源SHA、
LoRA8/16和实际B4均未变。原TaskRunner最近完成迭代为SQL6、AppWorld5、TextCraft22。
这次只更换一次性部署助手；AppWorld跨卡回复候选仍未部署，TextCraft候选也未部署。

### 2026-10-02 02:42：正式DT批次等待与版本复核

对当前六个worker做了一次20秒、非阻塞的只读Python栈采样，并读取它们已产生的
原DT批次记录；没有重放模型、生成新轨迹、暂停worker或修改运行绑定。六个profiler
均正常退出，原worker继续运行。原始source回执、PID创建时间、日志路径和六份栈
文件的SHA均在
[本次原始观察](../../research/temporary/rl_upstream_alignment_20260929/checkpoint-boundary-20261002/native-phase-profile-1790880137/observed.json)
中；[精简索引](../../research/temporary/rl_upstream_alignment_20260929/checkpoint-boundary-20261002/native-phase-profile-1790880137/compact.json)
保留批次与调用栈对应。它们是阶段诊断，不是新数值验收或候选提速回执。

| 正式记录 | rank0 / rank1的实际B4长度 | rank0 / rank1的批次墙钟 | 对应修复与生效状态 |
| --- | --- | --- | --- |
| SQL最近完成DT的588/588批 | 3311 / 22299 | 79.848 / 80.050秒 | `4c0cbdd`复用原VERL长度分区器；SQL恢复候选已验证，尚未部署 |
| TextCraft本次DT的100/149批 | 1679 / 3443 | 9.207 / 9.191秒 | 同一分区修复；TextCraft完整恢复候选已准备，尚未到原保存点25 |
| AppWorld本次回报类别组的13/107批 | 4913 / 4901 | 5.249 / 5.203秒 | 当前AppWorld已部署该分区修复；107只属于这一类别组，不能当成整轮总批数 |

SQL、TextCraft两卡长度明显不齐而耗时相近，与既有FSDP同步等待的诊断一致。
不能拿短卡批次墙钟当作它独立完成该长度所需的计算时间。已验证的原分区修复
应先落入对应正式版本，不能继续用未部署的旧路径评估修复效果；原小规模
19.717→9.641秒对照仍只证明那组8例，不外推为正式迭代加速比。

采样时SQL已进入原旧概率计算，Python样本主要落在HF的`_get_unpad_data`；
AppWorld/TextCraft在正式DT中，样本主要落在有限传播及流同步处。Python栈可能在
等待GPU，不能把样本占比写成GPU内核耗时，或据此复制HF的mask/unpadding实现。
本次没有新增此类优化或改动官方容差。

02:58快照确认原TaskRunner最近完成迭代仍是SQL6、AppWorld5、TextCraft21；
三组冻结entry及数值文件没有哈希漂移。AppWorld原检查点marker仍为5，一次性等待
助手PID2374706的创建时间仍匹配且存活，尚无完成停止回执。跨卡回复候选仍属于
prepared-only，不能把等待助手的存活称为新版本部署或训练健康。

### 2026-10-02：跨卡返回候选与版本封存

02:16已将AppWorld候选安排在**原作业完整检查点6**切换：一次性助手
`e87c593`，本机PID17944，远端等待进程PID2374706/创建时间1790878602.62。
02:17实际复核两进程存活，原训练PID2479539仍在运行，marker仍为5，尚未停止或部署。
助手只等原marker及data/model/optim/extra_state文件，然后停止已绑定创建时间的该作业
进程树，调用现有提交助手及原VERL loader；不修改保存频率、任务预算或模型参数。
原等待回执在 `receipts/owner-b8-dispatch-20260930/checkpoint-boundary/appworld-completion/waiting.json`。
快照单独保存等待PID、创建时间和实际存活状态，不能把“已安排”写成“已部署”。

同一提交助手已支持TextCraft原检查点恢复，显式传 `--task TextCraft`；正式数据路径和
AgentGym owner从当前原launch/argv继承，B4/head/padding来自冻结候选；没有复制loader。
`--prepared`与`--run-dir`现在必须显式指定，撤销旧AppWorld候选的默认值。
两任务实际生成的提交/切换脚本已在CPU解析核对；这项检查不是保存/恢复运行验收。
TextCraft尚未到原保存点25，当前没有安排其自动停止或新训练提交。

进度记录同时修复了原先仅看driver日志尾部的缺陷：该尾部可能被生成文本占满。
现保留原TaskRunner最近的完整 `step:` 指标行及运输行，并绑定原PID/文件路径；
不把正在采样的下一轮当已完成更新。02:00的原日志确认SQL6、AppWorld5、TextCraft21完成。

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

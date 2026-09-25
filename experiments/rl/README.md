# DeltaTrace RL

唯一方法规范是 [PLAN.md](PLAN.md)。环境复用见
[REMOTE_ENVIRONMENT.md](REMOTE_ENVIRONMENT.md)；本页只记录实现与测试状态。

**当前修订（2026-09-25）：按用户要求移除平方归因展开。**
19:30已通过原入口启动当前发布`64e5876`的正式预算：Sokoban GPU4 150×256，
WebShop GPU5 150×128，AppWorld GPU6 200×240。运行目录为远端
`runs/author-64e5876-paper/{Sokoban,Webshop,AppWorld}`；启动信息和实际Ray
session已记录在`formal-training.json`。19:42三组均已完成首轮实际环境交互并
继续下一轮；首次生成RPC的Sokoban/WebShop/AppWorld吞吐约965/465/514 tokens/s，
包含完整生成调用开销，不是纯decode。正式训练迭代和完整预算尚未完成。
三任务此前均完成两轮当前方法的真实训练，原检查点1→2分别有
1,433,186 / 1,452,754 / 1,436,548个LoRA元素改变，参数有限。
汇总见[当前验收与启动记录](results_linear_return.json)，包含各证据的范围和原始回执。

18:52完成当前发布的同输入精确32768/B4直接耗时对照：完整DT冷/热为
58.56/27.31秒，原生纯反向冷/热为82.27/57.58秒，原生前向另为32.60/15.18秒。
热DT为热纯反向的0.474倍；双方均含原CPUOffloadPolicy、activation offload、
checkpointing和驻留休眠的vLLM graph32。梯度有限、退出0；一次冷/热观测，
不是自然任务平均耗时或所有长度的普遍结论。见[同输入原生反向对照](../../research/temporary/rl_recovery_20260925/receipts/linear-return-cost32k-summary.json)。
当前真实训练验证：三个任务均两轮完成且退出0。AppWorld首轮有1/4官方
成功、21请求/6次DT，产生9241个非零优势，原PPO梯度约0.005。
WebShop最初两轮8轨迹均为零奖励，未将其当作非零路径通过；原采样器扩大到
4个任务组×8轨迹后，首轮有3个官方成功、30请求/8次DT/81.93秒，产生16695个
非零优势；原PPO97次B4更新为466.96秒，梯度约0.002，checkpoint1完成，
整轮1180.42秒。第二轮29请求/8次DT/124.84秒，原PPO114次B4更新583.16秒，
梯度约0.003，整轮1375.86秒；checkpoint2完整、optimizer步数97→211。
未按奖励挑选任务，actor/DT minibatch仍为4。
AppWorld第二轮2/4官方成功，32请求/8次DT/81.17秒，11522个非零优势，
原PPO336.69秒、梯度约0.009，整轮1230.49秒。checkpoint1→2的1,436,548个
LoRA元素变化，参数有限，optimizer步数31→58。见
[AppWorld两轮](../../research/temporary/rl_recovery_20260925/receipts/linear-return-appworld-completed.json)、
[实际参数更新](../../research/temporary/rl_recovery_20260925/receipts/linear-return-appworld-saved-updates.json)。
这些pilot不计入新正式预算。见
[AppWorld首轮更新](../../research/temporary/rl_recovery_20260925/receipts/linear-appworld-first-update.json)、
[WebShop非零路径](../../research/temporary/rl_recovery_20260925/receipts/linear-webshop-diverse-first-dt.json)。
当前同输入缓存诊断确认：原GDN完整/缓存前向相对原FLA FP32参考的误差比为
0.001716/0.001826（固定原断言0.005），条件于实际缓存状态的误差为0.001708。
三个检查均通过。没有根据整网守恒残差另设零误差门槛或修改原缓存；联合有限
分解与单token删除的差别仍属于单独记录的估计近似，不冒充精确反事实验证。
见[当前原生缓存核验](../../research/temporary/rl_recovery_20260925/receipts/linear-return-cached-owner.json)。

每个 response 只读出完整未来回报，所有官方过程 reward 进入累计值，再复用
原 DT/QVA/PPO 接口。81项 CPU 组合与接口/输出层测试通过（1项GPU专用测试跳过）；15步的成功/失败夹具均为
15请求、4次batch4调用。真实 tokenizer 的 Sokoban 31类均为单token，
询问加目标最多412tokens；WebShop/AppWorld为193tokens。新目标的真实模型
接入与精确32k容量已通过：三任务官方奖励夹具产生有限非零token优势，DT B4
连续两次为37.12/27.41秒，原PPO两次非零更新和原生LoRA同步完成；1,441,283个
参数元素改变。阶段物理显存快照最高约51.19GiB，不是连续峰值；旧虚拟allocator
统计不代表物理用量。完整用时634.23秒，含253.87秒原owner初始化及首次编译。
原DT守恒诊断仍有5条超原阈值，最大绝对残差0.04443，保留失败标记，没有缩放
或裁剪信用；接口/容量通过不能扩大为逐token反事实精度通过。
回执见[线性目标与32k](../../research/temporary/rl_recovery_20260925/receipts/linear-return-capacity-compact.json)
和[组合/输出层测试](../../research/temporary/rl_recovery_20260925/receipts/linear-return-cpu-tests.json)。
Sokoban新鲜四轨迹的首轮已完成：轨迹长度9/15/15/9，48个response只产生48个
请求、12次DT，共119.80秒；旧展开对相同步数需330请求，这个旧数是解析计数，
不是另一条实际耗时测试。原PPO更新47.54秒，非零梯度约0.041，完成checkpoint1；
整轮564.53秒。第二轮也已完成：60请求/15次DT/114.67秒，原PPO46.18秒，
整轮545.86秒；退出0、checkpoint2完整。直接读取两个原生检查点确认
1,433,186个LoRA元素改变、参数有限、optimizer步数12→27、学习率1e-6。
第二轮四轨迹均未成功，不能把小规模运行当作任务性能提升。首轮30项、第二轮
40项原守恒诊断超阈值也保留，不能称数值
误差全面解决。见[首轮真实训练](../../research/temporary/rl_recovery_20260925/receipts/linear-return-sokoban-first-update.json)。
实现提交`64e5876`已推送，远端`releases/64e5876`的111项源码SHA已核验；
已完成的Sokoban小规模验证使用已记录的候选目录，没有运行中热替换。
旧Sokoban正式作业已在1118/3847个DT批次时定向停止，未完成迭代不保留作训练
结果，原日志保留；WebShop已完成旧方法正式step1。
18:19停止WebShop/AppWorld旧未完成迭代，保留检查点及原始日志，使用发布
`64e5876`启动各四轨迹、两次迭代的原入口验证。两者已进入真实交互，仍需
完成新版本更新；未据启动就宣称通过。见[两轮Sokoban](../../research/temporary/rl_recovery_20260925/receipts/linear-return-sokoban-completed.json)、
[已保存参数](../../research/temporary/rl_recovery_20260925/receipts/linear-return-sokoban-saved-updates.json)。
短数值诊断已将残差拆到原加法/norm/mixer边界；完整DT端点与原生完整前向差
最大3.82e-6，缓存与完整端点差最大0.16990，单token重放两次逐值一致。
这四条实际输入也仍有联合分解与单token删除异号，不能称逐token精度通过；
未修改有限规则、FA/FLA或PPO来强迫守恒。见[原始边界诊断](../../research/temporary/rl_recovery_20260925/receipts/linear-return-boundary-summary.json)。
下方按日期记录历史状态。

**历史状态（2026-09-25 15:56 +08）：已重现并修复一处完整链路的 head 放大；三个任务均完成实际非零 DT/PPO 小规模验证。正式作业中 Sokoban/AppWorld 继续运行；WebShop 在原 PPO 更新期间退出，保留失败现场后通过原 owner 重新启动，尚未完成正式首个更新。**
同一异常样本的旧 head 操作数逐值重现：首 token 的 `d=-3.21144` 隐含
反事实概率1.28169；只替换为既有 FP32 head 修复后，该 token `d=+0.187565`，
概率0.0407164，−0.1步罚的优势从+2.38148变为−0.0171025。
原因是逐类别舍入倍率破坏 log-softmax seed 的零和结构，并放大随后相互抵消
的贡献；不是 PPO 重复乘奖励。两条仍使用旧 head 的正式作业已定向停止，
保留已完成检查点及日志；AppWorld 此前已在完整40轮 rollout之后因 TaskRunner
退出而停止；内核日志现已确认12:54:06由主机global OOM杀掉该TaskRunner，
不能仅根据历史cgroup峰值就称为本次触发了容器限额。未以这些记录宣称训练健康。
完整证据、失败的诊断尾部及尚未解决的逐 token 估计偏差见
[信用放大定位](../../research/temporary/rl_recovery_20260925/CREDIT_AMPLIFICATION.md)。
历史最大 `d=-13.016` 尚未逐值重现，不能声称 head 是所有极端值的唯一来源。

15:44:52 WebShop Worker 3642631 在 native mcTracer 脱离约十秒后断开，
原 Ray 报 SYSTEM_ERROR/EOF，未完成检查点。未发现该时段内核 OOM/崩溃
记录，cgroup OOM 计数仍34；因果尚未确认。独立小矩阵进程在相同工具正常
脱离后继续运行并退出0，不能据此排除真实 PPO 路径上的观察器问题。
已停止在正式作业附加该工具，保留177 MB原trace与全部失败日志。
WebShop 新目录为 `runs/author-9785786-webshop/Webshop`，shell PID 4185212；
111项源码SHA和原环境、原PPO core均核验。训练计算与5a3cda0一致，仅提前
写出已有最负归因的原始重放数据；42项既有接口/QVA测试通过。Sokoban与
AppWorld仍使用5a3cda0，未重启或热替换。启动不代表恢复健康，下一次完整
更新仍待确认。见[退出现场](../../research/temporary/rl_recovery_20260925/receipts/formal-webshop-observer-incident.json)
和[原入口重启](../../research/temporary/rl_recovery_20260925/receipts/webshop-native-restart-9785786.json)。

同形状 B8/1479 对照已完成，包含单 token 原模型删除、单 token DT 与联合
EOS 分解。原前向缓存路径差异和有限传播偏差已分开记录；未把整网守恒阈值
当成FA/FLA容差。新FP32 head与DT内native FLA FP16组合的精确32768/B4、
原vLLM驻留、两次原PPO更新与LoRA同步容量检查通过。Sokoban已完成两次
非零更新及checkpoint2。14:24 WebShop后续两轮验证已退出0、checkpoint2完整，
第二轮3个真实成功事件经过36个DT对照及原PPO更新，梯度约0.013、token优势
范围[-0.712,+1.988]；此前全零小样本不能覆盖的路径现已实际运行。
AppWorld两轮已退出0、checkpoint2完整；首轮奖励全0，第二轮有1个官方成功，
17个DT对照/5次调用，d范围[-0.072029,+0.753094]，原PPO梯度约0.011、
token优势范围[-0.747,+5.291]。这些是实际小规模训练，不能当作任务性能提升。
随后直接读取三任务原检查点：第1到第2次更新分别有1425978/1442025/1430209个
LoRA元素改变，参数有限，优化器步数增加，学习率均为1e-6。控制台lr显示0.0是
格式化舍入；这次核验不是仅凭非零梯度推断参数更新。见
[已保存参数对照](../../research/temporary/rl_recovery_20260925/receipts/head-pilot-saved-lora-updates.json)。
新诊断记录保留原端点分数及最负归因所在
minibatch的原始IDs，41项CPU接口测试通过，无额外模型前向或信用裁剪。

14:44发布`5a3cda0`后，三任务从原权重在新目录`runs/author-5a3cda0-paper`
启动，保留旧异常运行及检查点。预算依次为Sokoban 150×256、WebShop 150×128、
AppWorld 200×240；模型与DT等明确适配仍见`paper_scale.json`，不称原文得分复现。
14:56已进入第3/5/4轮交互，正式训练迭代均为0。Sokoban前两次生成调用为
236.081/197.475秒、226477/205808个token；这次较长调用有实际工作量对应，
不是纯decode内核计时。内存约671.2GiB，主机OOM计数未增加；物理GPU约49–52GiB。
事实与来源见[启动后回执](../../research/temporary/rl_recovery_20260925/receipts/formal-head-fixed-initial-rollout.json)。

15:14更新：WebShop正式首轮15次生成调用完成，894098个token共1236.87秒；
7个非零奖励事件经过51个对照、13次DT调用（含B3尾批），DT共116.553秒。
d范围[-0.108941,+0.275745]；51条trace的最大隐含删除概率0.893370，没有越过1。
原守恒诊断17项未通过仍保留，不扩大成全部归因精度通过。现场栈已转入原
VERL compute_log_prob；Sokoban/AppWorld仍在第11/8轮交互。见
[正式首轮DT回执](../../research/temporary/rl_recovery_20260925/receipts/formal-webshop-first-dt-summary.json)。

绘图来源修复：缺少实际Ray session时，旧工具读取共享session_latest，导致
三任务状态串台。已补全原session身份并改为只读各自driver的缺省路径，两个
回归测试和三个现场日志来源核验通过；错误图隔离，正确图重新发布。
另有逐批极值日志候选：42项CPU接口测试通过，保留发生后续失败前的原始重放
输入；该候选尚未热替换进当前训练。当前实际作业与诊断候选须分开阅读。

以下为此前有时间戳的训练状态：
截至 10:33 +08，`runs/author-2ee7ab4-paper` 的 WebShop 已完成 checkpoint9/150，
Sokoban 完成 checkpoint1/150、第二轮正在 DT；AppWorld 本次重启完成第六轮
交互生成，尚无新检查点。AppWorld 使用已推送的 `c88a749`，其余沿用 `2ee7ab4`。
固定原作者源码与活动目录的差异、接口检查及未解决数值问题见
[本次审计](results_recovery_audit.json)。核心 PPO 和任务 worker 的奖励/状态更新
保持原实现；活动目录包含兼容及 DT 接入补丁，并非未经修改的官方仓库。
模型、模态、response 长度、LoRA 和 AppWorld 历史设置存在明确适配，不能
把预算相同称为原文实验复现，也尚不能声称任务性能无下降。
定时检查和备份未因本次恢复而重新启动。当前事实来源是远端三个 active/formal
manifest；下方所有旧状态均须结合其日期阅读。

12:30 数值边界修复候选：保存的 Sokoban head 操作数确认，BF16 输出舍入
再经逐类别 secant 校正，会把该事件的 head 系数范数放大约5.63倍。相同 h/W
的 FP64 参考事件分数差为−0.1935527；BF16 读出为−0.0448978，FP32 为
−0.1935556。候选改为原 head 选中类别的 FP32 投影，端点和有限 seed 共用
这份读出，去掉额外舍入倍率；Q/V、reward、PPO 不变。40项边界/接口测试
通过，真实 B4 完整对照完成：同模型两路 head 输入逐值相同，新路径 root
与 seed 读出一致；但这轮旧路径也没有再现历史尖峰，不宣称全网尖峰已消除。
正式任务未热替换，不把此处 head 改善扩大为整条 DT 单 token 精度已经通过。

独立的 FLA FP16 候选已完成120例原文本证据恢复：88例 paper_reference
平均71.456%→71.506%，32例 source_reference_cache_check 为84.453%→84.536%。
前者9升8降71同，后者3升4降25同；不是任务成功率，也不覆盖 categorical head。
仅在 DT 内转换 FLA 操作数、在 PPO 前恢复24个原绑定的32768/B4容量检查通过，
两次原 PPO 更新非零；该容量结果使用修复前的 categorical head，不覆盖新 head。

11:28 数值复核：真实缓存 GDN 操作数上的 BF16 dK 超过 FLA 原0.008阈值；
隔离的原 FLA FP16 调用边界使 native/DT有限 dK 分别降至0.00240/0.00217，
五项梯度断言通过，默认 BF16 六个有限算子系数字段逐值不变。候选又完成32768/B4、原 vLLM
驻留、两次原 PPO 非零更新和 LoRA 同步；热 DT28.26秒，第二次更新72.61秒。
但整条 prefix-on 归因差异仍在，候选未启用到正式训练，不能称数值问题全修复。
详细回执、默认路径对照及测试范围见上述审计。Sokoban仍在第二轮DT；WebShop
第10轮更新后评估中、最近完成检查点9；AppWorld本次恢复已进入第18轮交互。

以下为此前停止正式训练时的核查记录，不能替代上述当前状态。
Sokoban 完成的 step 1、WebShop 完成的 step 3 检查点保留；AppWorld 因超限
退出，未重启。作者仓库 `langfengQ/verl-agent@20bd331` 已实际获取并与当前
fork 核对：PPO/任务环境核心并非全部被改写，但 fork collector 的全历史累积
和本地观测重写不等于作者的每步输入机制。完整证据与未完成项在
[results_upstream_audit.json](results_upstream_audit.json)。没有声称已经全部替换
或任务性能不下降；这些结论需要原接口及同配置任务对照后才能给出。

隔离的完整作者源码候选 `candidates/official-verl-20bd331` 已接入现有兼容
补丁和 DT 接口，18 项 tokenizer/collector/配置边界测试通过；三个真实任务
的状态、重置、奖励与终止检查通过。这些是接口/环境验收，不是模型成功率。
本地完整历史观测重写已移除，尚未部署正式训练。AppWorld 作者默认最近两轮、
历史文本最多 10000 字符，与此前批准的完整历史预算结束策略冲突，已单独
提请用户确认；未擅自选择裁短历史。WebShop/Sokoban 各 4 条真实模型轨迹的
前两轮对照已完成，输入/回复 token、奖励与步数逐值相同；这不是完整任务
成功率验收。Sokoban 同一批首轮输入下，512 上限四条均在 action 前截断；
保留既有 1024 上限时两条给出可执行动作，两条仍截断。1024 的原作者/候选
对照也逐 token 相同。不能用“与作者接口相同”代替任务质量检查。

2026-09-24 上下文核查纠正：原文不能支持“三任务的完整历史轨迹必然都能
放进 32k”。GiGPO 的 WebShop 使用每步 4096 prompt / 512 response、两步
历史；Sokoban 的固定 fork 脚本是视觉输入、1024/512，并非当前文本全历史。
LOOP 的官方 AppWorld 例程设置 32000 上限、单次 API 输出最多 3000 tokens，
遇到 `MaxSeqLenExceeded` 结束单条 episode；当前接入此前没有后两项。
原文与固定代码链接记录在 [paper_scale.json](paper_scale.json)。
此次 AppWorld 首批到第 26 轮之后，下一轮 prompt 31748 超过预留
1024 response 和 146 DT 读出后的 31598 上限，异常退出整批。此前的精确
32768 容量测试没有覆盖这种多轮超限停止路径，不能据此声称完整任务可容纳。
用户已批准单条轨迹预算结束、保留此前官方奖励、不补造终局奖励；对应
owner 补丁和实际 tokenizer/collector 接口测试在 `patch_verl_agent2.py`、
`test_context_budget.py`。不自动加入 LOOP 的观测截短或修改 DT/PPO 公式。
定时检查已按用户指令暂停，当前专注修复；下列启动说明为历史记录。

2026-09-24 04:54:46 +08：修复发布 `88685db` 的 99 个源码文件哈希已核对，
三组正式预算作业已恢复，运行目录为 `runs/rollout-repair-88685db`。WebShop
由原 owner 自动恢复已有 step 1，Sokoban/AppWorld 重采未完成的首轮。
生成并发 32、actor/DT minibatch4、总上限32768，预算与 Q/V/A/PPO 均保持。
当前处于初始化/真实 rollout 验证，不把启动当成完整迭代健康；每小时检查
已更新到新 manifest，备份等待修复后完整 rollout/更新确认。

此前三组正式作业主动停止以修复 rollout 吞吐，保留 WebShop
已完成的第 1 次更新和检查点；未完成采集不再用于更新。当前运行状态以
`formal-training.json` 为准；历史主动停止状态不是当前运行故障。
正式规模暴露了 rollout 入口漏接：collector 传出的
`rollout_active_mask` 只在 HF 后端使用，vLLM 仍为已结束的轨迹生成文本。
Sokoban 前 9 轮的 2,304 个请求中，939 个无效；AppWorld 前 25 轮的
6,000 个请求中，1,743 个无效。不能把进程存活、无 OOM 当作效率正常。
`patch_verl_agent2.py` 已补上固定 vLLM owner 的请求过滤与原行序恢复；
`test_vllm_active_rows.py` 的 48 项官方接口对照通过（显式 engine double，
核验请求、LoRA、返回索引、零 mask、无 mask 原路径，活跃行逐值一致）。
这项测试不是实际模型吞吐或 FA/FLA 数值验收。运行中 worker 的加载状态及
生效后的耗时须另看远端 `receipts/vllm-active-rows/`，源码落盘不表示已生效。
WebShop 第 1 轮 1,920 个请求中有 156 个无效；请求比例不直接等于耗时比例。
原第 1 次更新：生成 RPC 总耗时 13,730.372 秒，DT RPC 235.493 秒，
总迭代 22,441.507 秒。生成 RPC 包含 prefill、decode、唤醒、权重同步和搬运，
不能将 46.75 output token/s 称为 decode 内核吞吐。

[同卡并发对照](benchmark_rollout_concurrency.py) 复用原 VERL/vLLM worker、
32k 上限、actor minibatch4 和卸载设置。32 条相同输入共 199,420 prompt
tokens，实际最长 16,489，每条强制生成 128 tokens；预热后重复调用如下：

| vLLM max_num_seqs | 生成调用秒数 | output token/s | 相对并发 4 |
| --- | ---: | ---: | ---: |
| 4 | 92.713 | 44.18 | 1.00× |
| 16 | 39.930 | 102.58 | 2.32× |
| 32 | 29.948 | 136.77 | 3.10× |

三组均无 OOM；2 秒间隔物理显存采样最大 58,327 / 65,536 MiB，非连续峰值。
输入来自 owner 导出的任务文本（导出时已移除特殊 token），因此是等输入吞吐
回放，不是原 on-policy token 轨迹复现、任务效果评估或完整迭代提速结论。
并发 4 下仅提交 8 个活跃请求后耗时 23.241 秒，原 32 行运输顺序保持不变。

并发 32 的重复调用中，引擎 `generate` 占 29.050 秒，外层边界占 0.898 秒。
实际 `RequestOutput.num_cached_tokens` 合计仅 2,048 / 199,420；相同输入
重复提交也未复用上一调用的前缀。已核实每个工具轮次退出 sharding manager
都会调用 `LLM.sleep(level=1)`，其内部先 `reset_prefix_cache()`。这是缓存
生命周期事实；尚未隔离量出重复 prefill 的耗时，暂不因此扩展异步架构改造。
完整记录在远端 `receipts/vllm-active-rows/concurrency-{4,16,32}.json`。
已有容量脚本在并发 32 下复核通过：DT 输入恰好 32768、B4、1024 action
槽位、休眠 vLLM 驻留、两次原 PPO 更新及 200 层 LoRA 同步均完成，无 OOM。
32769 明确拒绝，总计 356.327 秒。MetaX 环境默认生成并发已设为 32，actor/DT
minibatch 仍为 4；真实完整 rollout 的修复后耗时尚待验证。

response padding 修复使用 Qwen 原 `logits_to_keep` 的位置选择接口，只跳过
B4 中所有行都无效的右侧词表投影；decoder 输入、FA/FLA、response 形状和
PPO mask 保持原样。开关 `VERL_TRIM_RESPONSE_HEAD` 在 MetaX 默认为 1。
24 项接口测试通过；真实 9B、B4、128 有效 action + 896 padding 的短链中，
有效 log-prob、两次 PPO 更新的梯度及参数逐值一致，省去 3,584 个无效槽位的
log-prob 计算。热更新分别为原版 2.835 秒、选择头 2.734 秒；首次调用包含
编译/预热，不用于提速比较。这不是完整训练吞吐或官方 FA/FLA 的整网认证。
已保存 WebShop 文本按 B4 顺序估算，共同右侧空白约 47%，不是按所有回复
均值算出的 67%；文本导出移除了特殊 token，精确训练统计仍以原张量为准。

对照同时复现并修复 DT B1 尾批编译错误：成对 batch=2 与单条 adjoint 的
特殊化不能强制为动态符号。仅将 batch 标记改为官方 `maybe_mark_dynamic`，
时间维及有限传播公式不变，不增加 eager 回退。B4/B2/B1 的 Dynamo 回归及
真实 Qwen B1 Inductor 路径已通过。候选验证时须核对实际模块路径；工厂会
插入发布目录，不能仅凭 PYTHONPATH 写了候选路径就认为已加载新代码。

每小时检查同时运行 [日志绘图](plot_training_progress.py)：
`python -X utf8 experiments/rl/plot_training_progress.py --source-root /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922 --publish`。
它只读当前正式 manifest 的原 Ray worker 日志，生成进度/效果 PNG、指标 CSV
和带来源行号的 JSON 快照，不改训练、奖励或记录器。输出按运行标识保存在
`research/temporary/rl_training_plots/`，保留每小时版本；`--publish` 同步到远端
`receipts/training-plots/`，纳入原 restic 备份。训练与独立评估成功率分开，
缺失不填零、不混入 pilot；原 console 指标仅有三位小数。资源是定时快照，
不是连续峰值。已用真实上游两迭代日志核对解析，并生成正式运行首版图。

2026-09-23 20:35:24 +08：三个两迭代 pilot 均已退出 0，各任务至少有一轮
真实非零奖励、DT 优势和 PPO 梯度。AppWorld 第二轮为零奖励、零优势/梯度，
不冒充非零更新。尾批次修复及回归验证后，已推送的 `6d1a946` 以 94 个文件
的独立发布目录启动三组正式预算作业：WebShop 150×128、Sokoban 150×256、
AppWorld 200×240。`formal-training.json` / `active-training.json` 指向
`runs/native-prefix-6d1a946-formal`；本次启动不代表已完成正式训练或收敛。
训练继续复用原 VERL/vLLM、token Q/V/A、32k、actor/DT minibatch4，原 PPO
核心哈希未改，部署导入核验与 6 项原配置/传输边界测试通过。论文预算来源及
模型、算法等差异仍见 [paper_scale.json](paper_scale.json)。
以下为当时的启动记录；当前主动停止状态见本页开头。已有每小时检查
`deltatrace` 每次回显，且删除了额度查询/
20% 停工规则。首轮源码、配置、数据和日志的异机 restic 备份及 restore 校验
已通过（快照 `90fa82b7`）；新检查点须等待原 owner 完成标记后再备份。

用户要求额外检查环境状态与调用行为。`verify_environment_state.py` 直接调用
固定上游 worker / projection，复用现有资产；检查已通过：Sokoban 两实例
棋盘隔离、重置和 `−0.1×4 → 10.9`；WebShop 独立会话、搜索/选项/购买与
原 task_score=1→reward=10；AppWorld 两个未被正式任务占用的现有服务上的
REPL 隔离、重置清理、40 步终止、`complete_task()` 状态隔离和重置。
WebShop 正例使用已标明的训练目标 fixture，未混入策略样本；没有重置或
写入当前训练服务。结果见 [环境检查](results_environment_state.json)。

652 条 pilot 输出的原解析器检查表明，651 条缺完整 `<think>` 标签，因而
原 `is_action_valid` 格式标记为假；该标记不表示环境没有执行动作。当前 DT
配置关闭原 invalid-action penalty，源码确认它未进入其他奖励/损失路径。
三个环境不填 `tool_calling`，故 trainer 的 tool_call_count=0 也不能解释为
调用次数。AppWorld 真实任务 `229360a_3` 的 4 份原 API 日志分别有
32/32/19/58 次调用；第一条轨迹的官方评估为 6/6 通过，与 pilot 成功一致。
这项检查覆盖代表性状态/调用链和现有日志，不宣称穷尽所有环境行为。

最新 `native-prefix` 候选直接复用 Qwen 原生混合缓存：共同前缀只计算一次，
原缓存复制到两个端点，后续原生前向/重放只计算变化后的部分，完整 K/V 历史
仍然保留。精确 32768、DT B4、actor microbatch4、休眠 vLLM 驻留下，完整
DT 热调用 **27.814 秒**，主参照仍是原版官方反向 **44.007 秒**。两次原 PPO
更新及原生 LoRA 同步通过，无 OOM；PPO 源码哈希未变。该容量夹具含 1024
个 action 槽位和较长共同前缀，不冒充真实任务轨迹或所有长度的速度结论。

FA/FLA 仍使用固定官方测试的原 FP32 参考和断言，包含原生缓存续算的算子
对照。完整短链缓存/全量前向的 A 相对 L2 差约 1.78，V 约 0.0526，单独
保留：首个差异发生在原生 GDN 前向，每层重放与自己的根前向相同。这不是
逐值一致或整网容差通过的声明。算子结果不能替代三任务验收；三任务后续运行
状态见上方更新。正式启动入口已补齐官方 activation offload、actor/log-prob
microbatch4，避免与容量验收设置脱节。

提交 `d6b7351` 已推送并发布到 MetaX `releases/d6b7351`。三任务两迭代 pilot
于 2026-09-23 18:59:31 +08 启动，使用当时空闲的 GPU 4/5/6；AppWorld
保留 40 轮，均使用原 VERL/vLLM。实际导入路径已核对，部署边界 6 项通过。
启动和数值通过不等于三任务已完成；以各任务实际 DT/PPO 结果为准。

本轮 pilot 已观察到 Sokoban 首个非零 PPO 更新：4 条轨迹成功率 1.0，
33 个非零奖励事件、154 个事件对照经 39 次 DT minibatch 计算，DT 合计
494.17 秒；原 PPO 更新 108.23 秒，日志梯度范数 0.009。DT 最长真实输入
7965，不是 32k 容量夹具。154 个对照中 73 个原诊断的守恒残差超过 0.02，
最大绝对残差 0.14286；不把非零优势或轨迹成功宣称为精确归因证明。
AppWorld 首轮已读出 1 个非零奖励事件、20 个对照，5 次 DT 共 146.20 秒，
有 4601 个非零 token 优势，原 PPO 更新梯度范数 0.006。WebShop 首轮
8 条轨迹奖励全零，DT/梯度也为零；第二轮 3/8 成功，34 个事件对照经 9 次
DT 得到 12198 个非零 token 优势，原 PPO 梯度范数 0.012。Sokoban 第二轮
完成 451 个过程/终局事件对照，PPO 梯度范数 0.002。2026-09-23 20:10 +08
Sokoban/WebShop 均已两迭代完成、退出码 0；AppWorld 随后也两迭代完成、
退出码 0，第二轮原 PPO 更新 635.14 秒，整轮 2115.26 秒，无 OOM。

Sokoban 首轮最后的 B2 尾批次用了 71.86 秒，前面的热 B4 约 9–14 秒。
额外 `verify_dt_context_capacity.py --tail-batch-probe` 复用同一官方 worker、
休眠 vLLM 与精确 32768 输入，记录 B4/B2 的原编译器计数和 DT 阶段成本。
它只诊断实际出现的慢批次，不替换 44.007 秒的官方反向主基准，也不重复宣称
PPO 更新已验收；未改动运行中的三个 pilot 或生产 DT 规则。该诊断已结束：
首次 B2 为 61.24 秒，7 个有限图因 batch 维度 guard 重新编译；第二次 B2
为 15.56 秒、恢复 B4 为 28.72 秒，两者没有新增编译计数。动态 batch 修订
在隔离候选中沿用原 Torch `mark_dynamic` 接口标记 batch 维，有限公式未改。
相同精确 32768、原 actor/休眠 vLLM 下，首次 B2 降到 15.99 秒，再次 B2
15.16 秒、恢复 B4 27.72 秒，三个阶段编译计数均无增量。初始 LoRA 参数
相同，B4 的 Q 逐值相同，V/A 相对 L2 差分别为 0.0000617/0.00814；
这只记录编译调度带来的整链数值变化，不新增数值阈值。

动态 batch 候选的原 FLA FP32 参考/断言已在 T=128/447、非零初始状态、
全 head/8-head/缓存后缀上通过。再次跑同样 120 例证据恢复，与修订前加速
路径比较：原协议 88 例为 86 同/1 升/1 降，平均 −0.00785 个百分点，
最大单例下降 1.31579 个百分点；缓存续算 32 例为 31 同/1 升/0 降，
平均 +0.10776 个百分点。signed 向量相对 L2 最大差分别为 0.001848/
0.001320。该回归仍是自然长度 590–3141，不冒充 32k 恢复率。
原始结果见 `dynamic_batch_fix`；运行中的 pilot 保持原提交，没有热改代码。

用户另要求确认加速后的证据恢复能力。`verify_dt_recovery.py` 直接复用现有
Qwen3.5 缓存的输入 IDs、targets、gold 与评分函数，对比同一模型、同一
`gdn-symmetric-v1` profile 的默认执行和当前 RL 加速执行。11 个原评测任务
各 8 例共 88 例，另有 32 例 VT 正文 reference 对照，实际触发共享前缀缓存。
两种 reference 分开报告，不混成一张原论文表。MetaX GPU 3 的配对运行
`candidates/dt-minibatch/recovery/matched-8` 已完成、退出码 0：

| 证据恢复对照 | 样本数 | 恢复率相同 / 提高 / 下降 | 平均变化（百分点） | 最大单例下降（百分点） |
| --- | ---: | ---: | ---: | ---: |
| 原协议、原 reference | 88 | 86 / 1 / 1 | +0.0040 | 0.625 |
| 正文 reference、实际缓存续算 | 32 | 27 / 3 / 2 | −0.0026 | 3.125 |

缓存续算实际使用 512-token 原生混合状态；signed 向量相对 L2 最大差分别为
0.001789、0.043702，作为整链诊断保留，不套用自定阈值。输入自然长度
590–3141，未补长；这不是 32k 证据恢复或全量论文重跑。原始向量 SHA256
为 `66d43ee6df29ef49112c2ebc65fc1386e4c06b2975ff976c1a5841d6ab04ae1b`，
本机下载后复核一致。逐任务数值及两种协议见结果索引的 `recovery_regression`。
当前证据支持这批样本上的恢复能力基本保持；算子容差仍由原 FA/FLA 断言判定，
两类证据分别记录，均不冒充整条 PPO 的官方数值标准。

以下为前序诊断记录。此前隔离候选已接通原 VERL FSDP2 参数卸载，并修复 DT 读出索引的计算设备。
真实 9B 同权重/输入下，开启和关闭卸载的 32 层输入与 logits 逐值一致。
保留 9 GiB 原生 GDN 中间量后，同一次 32k 捕获、同显存条件的预热有限阶段
由 5.60/5.81 秒降至 3.34/3.36 秒，输出逐值一致。该结果只覆盖首个 GDN；
这项局部测试当时尚未覆盖完整容量；下方记录后续结果，三任务正式训练仍未恢复。
具体回执见 [minibatch 结果](results_dt_minibatch_candidate.json) 的
`parameter_offload_and_selective_capture`。

后续在原有限 FA 内加入可选的系数输出范围：完整 K/V 保留，省略共同因果
前缀的系数计算。真实 actor/休眠 vLLM、同一份 B4/32768 首层张量对照为
13.38 秒→0.586 秒，所需系数逐值一致；完整短归因及 Q/V/A 也逐值一致。
这项局部结果见同文件 `fa_coefficient_suffix`，不代表完整 DT 效率已达标。

GDN 隔离候选复用原 FLA 保存的块边界状态，只计算共同前缀之后的系数。
同一次 B4/32768 捕获、同显存驻留下，首个 GDN 从 3.35 秒降至 0.819 秒；
所需输出最大差值 3.49e-8。128/447 算子用原 FLA FP32 参考和原阈值通过。
完整短链记录的 advantage 相对 L2 差为 0.01175；这是诊断，不另造整网容差，
也不声称逐值一致。见 `gdn_coefficient_suffix`。全量捕获的搬运仍待削减。

后续 `compact_gdn_captures` 已在捕获前选取真实末段，保留原 FLA 边界状态和
卷积左侧 3 个位置。首个实际 32k GDN 的 D2H 为 22.109→0.821 GiB，GPU
保存量为 9→0.334 GiB；同张量预热拷贝合计约 0.417→0.018 秒，打包另计
0.035 秒，切片值全部一致。检查点另复用固定页内存搬运：同一 2 GiB 原始
张量预热约 0.346→0.0378 秒，值和步幅一致；完整短链 signed/target/Q/V/A
逐值一致。原 position IDs 广播视图导致的首次失败已修复并保留原日志。

性能主参照仍是原来可行的官方反向 **44.01 秒**；不要求双方内部卸载策略
完全相同，不通过换慢参照验收。追加的原生参数卸载选项测试为 44.23 秒，
仅是诊断。三个任务正式训练和自动检查仍未恢复。

`capacity-compact-pinned32k.json` 已完成合并候选的精确 32768、DT B4 和
actor microbatch4 检查：两次 DT 为 158.108/121.332 秒，两次非零原 PPO
更新及 200 层原生 LoRA 同步通过，无 OOM；改变 1,440,439 个可训练元素。
热 DT 仍是原官方反向的约 2.76 倍，**效率未达标**。热调用中根前向
34.624 秒、逐层重放 44.801 秒、有限 decoder 31.134 秒；后者已包含
5.068 秒恢复和 5.395 秒 FA LSE，不能重复相加。剩余大头是原始前向/重放
对共同前缀的重复计算，须沿现有原生接口继续处理。此容量夹具不是三任务训练验收。

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

2026-09-23 补查确认：仅核对 main 的 FA/FLA 内核不足以确认完整加速路径。
`codex/clean-v1-acceleration=2b36c4ec` 和
`codex/qwen35-cause-and-tolerance=e1e37bb4` 的远端分支头已核对。候选现在
复用后者的 Qwen3.5 retained/code-local capture，并把原 deferred 控制器的
延后同步接入当前 official runner；既有动态编译继续使用，没有另建控制器。
真实 9B、相同 B4/635-token 输入的 12 次调用，完整 signed 向量、目标分数和
Q/V/A 逐值一致；两轮预热后均值 3.293 → 3.045 秒，降低 7.52%。这些是短链
执行开销结果，不是 32k 或一次反向效率验收。迁移范围和未直接采用的 Qwen3
冻结图、投影缓存、D128/FP16 库见 [加速来源](../../deltatrace/accelerated/README.md)。

32k 候选按 head 分组调用原 FLA，仍保留所有 4 条轨迹、完整序列和两端对称
规则。搬运/分组分开核验；同分组下搬运前后逐值一致。两端重合的原生/完整
有限/8-head 有限算子，在 T=128/447 的 dq/dk/dv/db/dg 上通过 FLA 原误差函数
和阈值。这不覆盖先前完整 BF16 `test_chunk` 的 L2-normalization 复合路径
失败，旧失败仍保留。GDN 输出 RMS/SiLU 有限规则另通过官方
`test_rmsnorm_gated` 的 8 个原生/有限对照，未改变 1e-3 阈值。
最新 `batch-capacity-v13-compiled-conv.json` 已通过精确 32768、DT batch4，
一次批量归因后完成两次非零原 PPO 更新及 200 层 LoRA 同步；共改变
1,439,802 个可训练参数元素。输入仍是明确的容量夹具，包含 1024 action
槽位和 170-token 事件询问；不是自然任务成功率。完整 DT 671.653 秒，
整次含初始化/PPO/同步 917.039 秒，无 OOM。Torch 105.18 GiB allocated
包含休眠的未映射 vLLM pool，不能报告为物理显存占用。
原 PPO 源码 SHA 未变。近一次原生反向的效率要求尚未通过；正式训练、
检查和备份仍暂停。配置路径的预热复测及原生反向参照单独记录。

对比条件另已核清：上述 v13/v14 的 PPO 为 minibatch4、microbatch1；
不能据此声称四条同时反向通过。`native-backward32k-v3-vllm-ppo.json`
在原 vLLM 同步/休眠后启用官方 activation offload，四条 32768-token 原生
反向预热后 44.007 秒，随后原 PPO 实际两次接收 `[4,32597]`，耗时
68.353/67.520 秒，非零梯度与更新后同步完成。vLLM 残留算在同一卡内。
无 offload 的失败仅保留为容量诊断，不作为速度基准。DT/原生反向测试
现共用原容量输入构造函数；CPU 逐值对照证明与已有输入完全相同。

GDN 搬运大头已单独定位：原捕获每层复制 37.109 GiB。两个未被有限传播
读取的模块输出占 4 GiB，候选去除后为 33.109 GiB，并使用 Torch pinned
allocator/异步复制，在 capture 退出前统一等待。相同真实 9B 层、同一
B8/32768 隐状态容量夹具，全部共同捕获值和 stride 完全一致；两轮预热
记录为旧捕获 6.584/11.546 秒，新捕获 0.958/2.766 秒。该局部改进不被
当作完整 DT 达标。新路径完整短归因和 Q/V/A 的搬运对照也逐值一致。
`pinned-capture/batch-capacity-v15-pinned-micro4.json` 已通过完整 DT B4、
PPO microbatch4、官方 offload、vLLM 同进程组合：DT 487.649 秒，整次
742.741 秒，两次非零 PPO 更新，改变 1,441,808 个参数元素，200 层 LoRA
原生同步通过。这里确认容量和调用链；仍未满足接近原生反向的效率要求。
分阶段主开销为 root 67.134 秒、重放 107.569 秒、有限 decoder 300.555 秒；
有限 decoder 包含嵌套 restore/LSE，不能重复相加。

后续已用前两个 GDN 层完成真实上下文的早停诊断，保留 vLLM 休眠占用。
同一层、同一实际捕获及 upstream 的搬运对拍为旧 5.589 秒、新 2.289 秒，
输出逐值一致；短链完整归因、目标 log-prob 和 Q/V/A 搬运对照也逐值一致。
该局部结果不代替完整 DT 速度；诊断到指定阶段即退出，不等整轮跑完。

本轮继续删除两处有确切调用证据的冗余。GDN/FA 的完整 input 副本只被读取
shape，现保留实际 shape 元数据；按 B8/32768 计算，省去 64 GiB D2H 和
16 GiB H2D。FA 原生 dense Q/K/V 与接口 Q/K/V 是转置视图，实际指针、
dtype、shape、stride 一致时共享捕获，并在回传后重建视图，另省去双向各
24 GiB。合计 128 GiB 是单次完整 DT 的累计搬运减少量，不是常驻内存。
32k 第一层实际回放保留原 actor/vLLM 休眠状态；三次重复 D2H 原耗时合计
0.438 秒，共享视图约 0.00016 秒，共同值和 stride 全等。观察到指定拷贝后
主动退出，没有继续剩余层/PPO。完整短链归因、目标 log-prob、Q/V/A 与
相同 head 分组的原路径逐值一致；实际短链 618–635 tokens，不作新32k
全链验收。环境未重建，生产目录未切换，固定 PLAN 和 PPO 无改动。

资源核查还表明：D256 有限 FA 的 Q/KV 阶段各使用 256 registers/thread；
当前寄存器复用候选分别报告 156/764 bytes local memory/thread。它解释了
进一步检查寄存器压力的方向，尚未量化局部存储对耗时的贡献。原生 backward
warp 分布的编译探针不兼容现有快速转置接口，已拒绝，不进入 GPU 长跑。
全部原始回执和失败记录索引在 `results_dt_minibatch_candidate.json`。

最新 `origin/main=6ca8dc07` 的 MetaX 路径已核对并在使用：模型前向走已安装
FA 2.6.3 / FLA 0.4.1；DT 有限传播复用 main 的 MetaX FA 扩展和 FLA 原反向
子内核。32k、batch=4 的 FA 算子诊断中，原有限传播约 48.77 秒，原生 FA
反向 1.009 秒。最新候选为 14.62 秒，尚未达到效率要求；不能把这个算子
比值当作完整 DT/完整模型反向比值。
候选在两端重合时通过原 FA 测试（长度 128/447），非零端点与原有限内核
另做数值对照。DT minibatch 接口的同批次复制/复用张量结果相同，但串行/
批量仍有归因差异；此前 32k×4 重算 OOM 已在上述 v13 容量候选中消除。
完整速度和三任务连续训练尚未验收，因此没有切换正式训练到候选。记录见 [results_dt_minibatch_candidate.json](results_dt_minibatch_candidate.json)。

候选复用厂商 `gemm_opt` 和 `gemm_rs`：后者的快速转置加载接口每次接受
128 列，因此 D256 使用两个视图调用，保留原有限公式；不是重写 GEMM。
32k 非零端点对照中 dV 相同，dQ/dK 差异另存原始结果。扩大输出并行度的
另一候选为 21.35 秒，比 14.62 秒更慢，未采用。训练环境的有限 FA 库尚未切换。
CPU capture 候选必须保留原 strides；默认跨设备复制曾改变四维布局。
修正后短链 Q/V/A 与未 offload 的同批次结果逐位一致。候选已消除诊断 FP64
大临时张量、base/LoRA GEMM 重复 BF16 副本，并释放已消费的 FA 缓存；
早期候选在 GDN 同时恢复全部缓存后 OOM；v13 已按原 head 独立递推分组，
并编译原 RMS/SiLU 有限公式后通过完整容量检查。候选尚未接入正式环境，
容量通过不代表完整 DT 效率达标。

FSDP 修复后的 WebShop/AppWorld vLLM 恢复作业已分别完成 step 2/4，耗时
1134.59/980.84 秒；两轮实际 reward、DT 优势、梯度均为零，不能作为非零
DT 更新或任务学习验收。正式实验、每小时检查和异机备份仍未恢复。

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

## 当前实现：EOS DT → 完整未来回报 → token PPO

2026-09-22 已移除额外参考 token 采样和逐 token 前后奖励询问。
2026-09-25 按用户要求，`reward_readout.py` 对每个 response 至多调用一次正式
DT归因请求，目标是其后完整累计回报G；`counterfactual.py` 复用已有稳定计算
`G * (-expm1(-d))` 和 Q/V 组合。没有复制有限传播、环境评分或 PPO。

原始 token IDs、奖励事件身份及 mask 由固定 VERL collector 提供。
O/padding 不参与 actor loss；实际 EOS action 保留 action 身份，其 EOS 替换
对比可以为零。每个token仍有自己的d和advantage，不广播span，不归一化credit。

归因阶段通过 HF 公共接口临时切换到 FA；正常和异常退出都恢复原 actor
后端。当前按用户指令只使用 MetaX，复用其已记录的动态 shape 执行配置和持久缓存。启动脚本按实际
tokenizer 为事件询问及 target 预留空间，总上限仍为 32768，不截断已生成 action。

## 历史验证范围（不自动覆盖2026-09-25完整回报目标）

- 此前 HF 后端三任务完成过连续迭代，且各有真实非零奖励、DT token 优势与原 PPO
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

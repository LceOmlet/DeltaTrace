**2026-10-16 最新补充：PPO的32k必须按对应VERL数值标准验证，且不能降低效率。**
保持B4/32768、原GAE/critic/actor/optimizer/精度与更新配置；先定位已发生的
critic backward OOM及历史原生activation-offload取回错误，复用官方接口。
数值对照必须对应被改变的计算和原同状态输入，不能挪用生成top5一致性或
其他项目某处的容差；性能对照同输入/状态，不能用更短样本或缩批冒称加速。
新修正先放独立候选验证，不改变现有0/1 DTPO与4/5原PPO；旧梯度goal仍paused。

**2026-10-16 用户补充授权物理ID2/3可用于调试和训练。**
最新核对两卡空闲，允许SQL专项调试/有限原生效率试跑与6/7 AppWorld并行；
0/1 TextCraft DTPO v5及4/5原PPO保持。此授权不改变信用方法、奖励、采样/更新
超参或正式训练身份；先查原生日志/接口阻断实际DT负载的直接原因，再决定试跑。
AppWorld补充原生PPO两轮效率试跑：复用同一LOOP环境/4组x6及32k/B4，
由原固定VERL的GAE/critic执行，不实现另一套PPO。critic采用已运行原TextCraft
PPO的实际兼容配置，其他critic选项保留官方默认；原AppWorld actor配置保持。
只比较记录的阶段成本和有效负载，不把不同样本、两轮或小批量称质量对照。
SQL两轮全为原生格式-1、g=0，不能得到DT时间；仅补一轮原样配置的
原生逐turn文本/观察/ID被动记录以定位格式失败。记录不改原生成、奖励、
终止、token/mask或更新返回，不通过造非零奖励/乘子来强行启动DT。
原生逐turn记录确认：SQL40条中18条未输出solution，22条有solution但观察后
未重开think，被原utils.py的23/41行拒绝；不是tokenizer剥去think。独立
sql-native-multiturn-speed-20261016-v1只测试SkyRL已有use_conversation_multi_turn
模式是否改善此直接格式边界，保留原SQL采样/奖励和v5 DT/PPO。原seam仅
给原多轮state更新super调用补齐同样的turn记录，先与原生多轮完整轨迹/最后
结算target作CPU接口核对，再跑原两轮预算；候选不修改默认入口或正式manifest。

**2026-10-16 SQL/AppWorld v2实际启动失败后的最小接口修复，执行版本v3。**
SQL原生生成完成后，DirectActionTargetReadout任务注册只允许TextCraft/AppWorld而
拒绝SkyRL-SQL；仅在原owner任务注册中接纳已有SQL原始artifact，Q/V/A、EOS、
指数和mask均不变。AppWorld沿用的async选项触发固定VERL另建HTTP server，
其旧vLLM协议导入失败；使用现有原生sync rollout接口承接LOOP actor RPC，
LOOP自身并行采样/取消策略不变。v3保持v2全部训练/采样数值配置及两轮预算，
只修这两个已复现接口问题，独立candidate/run；v2的0完整迭代失败记录保留。

**2026-10-16 用户进一步授权停止物理ID6/7上的作业，继续SQL/AppWorld效率研究。**
原GPU2/3已被maxk项目占用，未启动的v1试跑回执保留；本次停止已核对身份的
GPU6/7 Qwen3-32B推理服务后，将相同已通过CPU检查的两任务试跑配置迁到6/7。
新执行版本为appworld-sql-native-speed-20261016-v2，只改设备、独立输出和版本标识；
原v5 DT/优势/PPO、单轨迹限制、32k/B4及两轮预算保持。先SQL后AppWorld，
复用原生生成、环境、DT及更新和原日志解析；按实际轨迹/token及阶段区分耗时。
DTPO与PPO的actor更新、PPO额外critic更新分开比较，不用总轮数或不同token量
冒称策略更新内核加速。当前TextCraft DTPO/PPO、小时绘图及GPU2/3其他项目保持。

**2026-10-16 用户授权 AppWorld、SQL 训练速度试跑，明确第3/4张为物理GPU2/3，保留现有PPO。**
仅为这两个任务做独立、有限的原生训练速度试跑；当前TextCraft DTPO/PPO、
小时绘图和已停止GRPO身份保持。沿用v5官方GRPO乘子、逐target原EOS DT及
指数后Q/V/A组合、官方masked_var仅方差缩放和原官方PPO；不恢复旧梯度goal。
AppWorld复用LOOP的原任务/环境/采样/取消流程，target只取终局结算对应的
原已执行代码片段；此前代码转为source。SQL原SQLEnv中间步骤奖励固定0，
仅done时结算；target取原SkyRL generator记录的该终局Action的原生成token
片段，包含格式失败-1、执行失败0及成功1的结算，之前查询为source。
观察原文保留但policy系数为0；无重分词或另一套任务parser/reward实现。
测速复用已验收DT数值/内存、原生vLLM图和冻结head更新路径；每卡B4、
LoRA8/16、32k与原任务单轨迹限制保持。先用两个完整原生迭代，SQL每批8组×5，
AppWorld每批4组×6（仍用原取消策略）；这是明确标记的小批量速度试跑，
不是原正式SQL1280轨迹或AppWorld240轨迹的整轮吞吐结论。原任务PPO epochs/
minibatch由原入口保留；仅试跑预算、组数、输出路径与关闭评估/保存改变，
不造诊断checkpoint。分开报告初始化/生成环境交互、old/ref前向、DT及更新，
按实际有效token/轨迹/RPC数量归一；不相加两rank并行耗时或嵌套设备窗口。
版本、原来源/SHA、有效配置、PID出生、实际导入及测试/测速事实写入运行记录。

**2026-10-16 用户批准实现、提交推送并从基础权重训练 DTPO v5。**
本次采用刚讨论完成的同一官方GRPO组相对信用目标：固定VERL原GRPO计算g，
原逐事件EOS有限传播保留未缩放的d_ik，先计算u_ik=-expm1(-d_ik)，
再在值域合并A_i=sum_k b_k*u_ik，Q_i=sum_k b_k，V_i=sum_k b_k*exp(-d_ik)。
不把带符号信用取log，不将信用放进有限seed，不除以端点差，不额外归一化u。
此处精确critic PPO等价比较使用同一个组相对目标；不再以原回报幅度的显然
尺度差异替代组合论证。理论同一期望策略梯度范围仍是原始优势/on-policy；
不扩称有限批次方差缩放、clipped PPO更新或未来训练轨迹逐值相同。
正式TextCraft原owner每条rollout只有一个真实终局结算，结算0也保留，
故b_1=g，目标为该结算动作的原parser payload联合token集合。多个文本片段
不重复计入同一结算；早先非结算动作从target移为历史source，观察mask保持。
其余多个实际奖励事件的组合仍由原reward_event_token_credit逐事件指数后求和，
不将全部target的log信用先合并后指数；本次不制造TextCraft不存在的奖励事件。
原self-target每token Q=g/V=0/A=g，原整批官方masked_var仅方差缩放，原官方
PPO actor/core及L=(1/16)*sum(S_j/N_j)、32k/LoRA8/16/B4/global64/330/save5/keep2保持。
使用实际验收的最快DT数值/内存路径、原生vLLM图和已部署冻结head更新优化；
未接受的head-selection、旧分母候选及v4仿射seed不部署。v4撤回记录保留，
新版本标记v5、独立run，不拼接v3/v4曲线。现有原生PPO继续，不重跑已停GRPO；
旧梯度根因goal保持暂停。本条解除本次新版本的先讨论不部署限制，版本与行为、
容量、实际导入、进程出生及推送事实由RUNTIME_RECORD/current_runtime登记。

**v5多target的实际接口、执行顺序与边界（同一方法规范，不是另一份方案）。**
`dt_direct_target_artifact.dt_reward_target_events`可承接多个已经确定信用的事件。
每项包含`event_id`、原完整response中的`target_offsets`、带符号`credit=b_k`，
以及原完整response位置上的bool `future_policy_mask`。多片段属于同一次结算时，
只放入同一项的offset并集；同一event_id禁止重复。b_k是值域的稳定信用，不能
用rollout的同一个g给每项各乘一次，不能对每项单独标准化，更不对b取log。
原official outcome GRPO提供rollout g；现有TextCraft一结算直接使用b_1=g。
多结算接口接收调用方已经确定的b_k分解，保持同一官方总回报标准差尺度，
不声称现有TextCraft产生了不存在的中间奖励，也不重写官方组均值/方差算法。

调用方按信用分量是否已经由当前prefix固定提供future mask；不能仅按原始
reward时间或target文本位置猜测，尤其共享组标准差仍依赖后续结果时。
每个事件单独保留原target mask，EOS同位替换该事件尚需归因的较早policy source，
其他事件的较早Action在该请求中仍可作为source。观察原文保留、policy系数为0。
每项得到未缩放原d_ik后，由原Q/V primitive计算b_k、b_k*exp(-d_ik)、
b_k*(-expm1(-d_ik))；同一target的self-token仅替代该事件项为Q=b_k/V=0/A=b_k，
同时仍可接收其他未来事件的source贡献。不把不同事件的d先合并再指数。

事件必须在driver上先展平，再复用原DataProto去重/pad、原长度平衡器与DP RPC；
沿用已部署readout完整历史长度的原生平衡。所有rank保持同数B4调用，不能到
worker内部各自展开不同数量的target而破坏FSDP collective顺序。返回后仅对
原token位置上的各事件Q/V/A作加法，再复用同一个原retained-position scatter。
整批官方masked_var缩放只在全部事件汇总之后执行一次。没有source u分母。
外部总信用B=0时也不能跳过非零b_k：例如(-1,1)仍可产生非零token优势。
单事件无扩展字段时仍走原native高效路径，不增加额外DT请求。

验收覆盖带符号/抵消信用、变长事件数、重复轨迹与不同保留位置、每rank等数B4、
明确future/观察/self mask、多片段一结算及重复身份拒绝；另用原parser的两个
实际payload验证真正DT逐事件调用、指数后的汇总和每rank B4精确32768容量。
用户例(-0.1,1)仅作为多target接口信用输入，不冒称TextCraft额外发了两次奖励。
具体已完成项与原始失败/修正记录以验证回执为准，不把准备或代码shape称验收。

**2026-10-16 完整组合推导结果：当时未改训练代码，未启动新训练。**
按用户前提接受EOS精确反事实。官方GRPO组内信用g_j=(R_j-mu_G)/(s_G+epsilon)
作为输入，固定其余组rollout C_j后，将g_j视为当前rollout结果的确定函数；
这保留当前样本对官方组均值/方差的影响，不把detach伪称为统计独立。
若真实结算已给出r_jk分解，可在同一官方总回报尺度下作线性分解
b_jk=(r_jk-mu_Gk)/(s_G+epsilon)，sum_k b_jk=g_j；不是各span独立标准化。
同一次结算的多个文本片段仍是一个联合target，不重复乘同一g。
原逐target EOS有限传播给出d_ik，先逐项u_ik=-expm1(-d_ik)，再以b_jk在
值域合并A_i=sum_k b_jk*u_ik。优势可负，不取log；不把d乘c或除DeltaF，
不归一化u，也不在合并log向量之后才指数。B=sum b非零时B*sum(b/B)*u
只是相同组合的重写；B=0时保留直接sum b*u，不引入epsilon或丢弃抵消分量。
精确change-of-measure使该估计的条件均值为Q_g(h,a,C)-V_cf,g(h,C)。
因此它与同一官方组信用目标下的精确critic PPO具有相同on-policy期望梯度；
无需V_cf,g逐值等于V_g，也不因有限归因守恒扩大到全部clipped更新相同。
若用户比较的是原环境回报R的精确critic PPO，官方有限组标准化一般使
该强等价不成立，不能偷偷改PPO目标来凑等价。原官方8条组/epsilon1e-6
CPU全计数枚举：奖励a与2a对应GRPO期望梯度0.4339509313/0.4339513654，
原回报精确PPO为0.25/0.5；EOS精确前提完全未改变。该结果只定位数学范围，
0模型/0optimizer/0正式进程信号。目标仍保留原用户要求，不能宣称已证明
与现有原回报PPO完全相同，也不能据此部署新方法。细节在同名推导结果JSON。

**2026-10-16 再次明确已接受前提：EOS有限传播预测精确反事实，不重新审理其准确性。**
用户再次指出这一理想化前提早已确定。不得把EOS是否精确、是否逐值等于PPO
的V重新设为实现前置问题。沿用已约定的当前动作无关稳定反事实基线时，
A_DT-A_PPO=V^pi(h)-V_cf(h)，其score梯度期望为零；不需要两个V逐值相等。
后续只检查给定target相对优势/信用、log读出、指数变换与Q/V的组合是否保持
上述等价，不转去研究DT误差或要求世界模拟器。下条关于不同V的区分仅作
数学说明，不得作为重新检验EOS预测能力或阻止既定理想化推导的门槛。

**2026-10-16 最新要求：先论证稳定相对价值基线，PPO critic按完全精确处理；不部署。**
用户明确讨论的是与PPO相同期望策略梯度的等价性，同时要求稳定的基线，
不能因任意环境奖励/EOS差可写成Q-V就认定满足要求。精确PPO的基线是V^pi(h)。
原官方GRPO组均值/尺度或其他已明确的相对价值基准须在事实与参考端保持同一
定义；冻结的是这次比较的基线与尺度，不能把所有反事实结果的相对价值都固定
为事实样本的那个数。优势可为负，不对优势取log；指数转换仍作用于模型log比。
固定、对当前动作无关的mu/s下，g(Y)=(R(Y)-mu)/s，精确概率比组合的条件期望
为(Q^pi(h,a)-V_ref(h))/s。mu在两个端点之差中抵消，所以中心化可改变采样方差，
但不会自动令V_ref等于精确V^pi。逐值优势等价需要对应值语义一致；同一期望
策略梯度允许差一个动作无关的状态基线，归一化/状态权重亦须按同一标准比较。
动作无关本身不保证低方差。实际官方有限组mu/s含当前样本，detach不等于
统计上动作无关，不能把固定mu/s的理想论证直接冒充实际有限组严格无偏证明。
保留官方组件，不制造组均值/方差/critic影子实现，不改变超参或重启训练。
rollout内target系数b_k/B的分母B与上述基线尺度s不同；外部B保持原总量。
不额外加入sum_source(1-exp(-d))归一化；它不是等价性要求。v4继续已撤回，
旧梯度goal继续paused。本段仅收紧研究问题和记录组合条件，不宣称新方法已验收。

**2026-10-16 最新纠正：v4线性优势替代已撤回，原Q/V指数构造不能被默默取消。**
用户指出取消指数可能破坏原等价性质。此前把“带权multi-target种子合并的
线性归因等价”扩展为可用g*C直接替代原g*(1-exp(-d))的决定不成立；总信用
守恒不能证明原Q/V或期望策略梯度等价。原公式V=Q*exp(-d)、A=Q-V必须
在其既定值语义下检查，不能用V=Q-A的形式恒等式证明任意新A都正确。
把已归一的C重新当d进入指数，也不会自动恢复原概率比/价值组合。
已提交的v4在首轮采样、原日志完成0轮时撤下，无正式更新或检查点；
源码804d2f67及行为/容量/停止回执保留，明确为已撤回候选，不作默认训练版本。
只保留真实奖励结算动作target（含结算0）的用户要求仍有效；原DT有限传播
扩展可以保留源码，但不得据线性归因测试宣称原优势等价已保持或再次启动。
须先把multi-target输出读出、各span信用与原Q/V指数组合的等价链条说明清楚。
PPO续训不变；旧v3停止45轮、GRPO停止191轮的历史保留。旧梯度goal仍paused。
本条覆盖以下v4已批准部署表述；下方内容作为此次已撤回实现的历史记录。

**2026-10-15 历史v4实施决定（已被上条撤回）。**
保留下述带权multi-target线性有限传播构造；target只取原owner奖励结算对应的
动作payload，结算值0仍保留。TextCraft原owner每条轨迹只输出一笔终局score，
按其原task_rounds对应的动作身份选择，而不是按非零奖励或最后文本片段猜测。
较早已执行Action移出target后成为历史source；观察仍保留原文且policy mask为0。
TextCraft这一笔结算的内部信用份额为1，外部总量使用原官方GRPO组内乘子g，
不是再乘一次环境原奖励。source原指数变换改为g乘原DT单位有限分配；既有
self-target处理保持A=g、Q=g、V=0，其余policy处Q=g、V=Q-A，观察/padding为0。
历史source的单位分配守恒不应混称为包含self-target policy动作的全token预算。
原v3整批官方masked_var仅方差缩放保持；以上分配量均指该缩放之前的量。
复用原DT局部有限规则，仅在原输出seed/读出加入默认不启用的逐span仿射权重；
同一事实/EOS端点上的多个种子合并走一次原反传。没有新的概率比指数、epsilon、
裁剪、学习价值网络、观察路由、官方GRPO/PPO或optimizer实现。
原32k、LoRA8/16、每卡B4、global64、330完整迭代、save5/keep2保持。新v4验证后
从基础权重重训，不恢复v3；原PPO续训和已停止GRPO历史保持，v3日志不拼入v4。
新版本与候选实际SHA/验证范围/部署状态以RUNTIME_RECORD和current_runtime记录。
本条覆盖下方历史“未批准部署/暂不修复”的范围限制，未恢复旧梯度根因goal。

**2026-10-15 带权multi-target研究构造（现已按上条获准实现和验证）。**
本次按用户给定的带符号target信用b_k讨论，不继续争论这些输入是否必须来自
字面删除概率；例b=(-0.1,1)，总量B=0.9，单位信用c=(-1/9,10/9)。
在同一事实/原EOS参考端点和原source/target mask下，F_k沿用原生span logp和，
DeltaF_k=F_k(h1)-F_k(h0)。若各DeltaF_k非零，可定义单位读出
U_k(h)=(F_k(h)-F_k(h0))/DeltaF_k，两个端点常数在传播时固定；联合读出
G=sum_k c_k U_k的端点差为1。原DT有限反传的输出斜率为m_Fk=c_k/DeltaF_k，
source信用C_i=sum_k c_k*d_ik/DeltaF_k，理想守恒下sum_i C_i=1。
直接以c_k乘原斜率分出的是c_k*DeltaF_k，不能说已经保持给定span信用预算。
固定端点下原有限转置传播对种子线性，可合并种子后走一次原反传；这不是新增
官方实现已支持的宣称，原FiniteAnswerOps目前没有外部逐span种子权重参数。
若用于分配原优势量B，须保留外部B，即B*C_i；内部单位化不等于抹掉rollout幅度。
DeltaF_k=0且c_k非零时，此标量读出不能用有限斜率完成给定分配；近零差及正负
抵消可放大source信用。不得自动插epsilon、取绝对值、裁剪或另换读出掩盖该边界。
这是待审查的信用路由构造，守恒不证明预测准确、PPO无偏或异常已修复；不将
分完的C_i再当概率比作指数变换。后续若恢复检验，原作者部分EOS干预曲线须固定
上述权重/分母，比较同一带权multi-target读出的原生变化，不能拿全删端点自证。
本条不修改已批准v3、Q/V/A、target选择、正式训练或原PPO，当前只作定义推导。

**2026-10-15 最新收紧：先定义RL所需反事实，再设计multi-target检验。**
用户指出“最后一个span”只是位置特例，不能等同奖励结算target，也不能单独验明
multi-target优势或新方案更好。该单span对照候选取消，不据此继续GPU测试。
必须先区分一次奖励结算的多个target片段与多个实际奖励结算事件；同一奖励不因
片段多而重复计权，结算值0仍是事件。EOS是DT同位缺失参照，不是令环境收到EOS
命令或新增提前终止规则。RL反事实需要对应同一决策前状态下的未来回报变化。
下面的固定历史条件评分只是一种模型干预诊断，不能自动等同RL所需的联合结果
分布或优势；须明确其与该结果变量的预测关系，而非把两者混为同一个量。
保留精确模型估计token价值等于对应世界变量的理想化，不要求精确世界模拟器或
零估计误差，不重新要求DT逐项等同字面单删，不擅改正式target/Q/V/A或PPO。

**2026-10-15 最新专门goal：只审查multi-target span定义和删除干预预测能力。**
用户已授权本问题的有限研究与检验；此前“先解释、暂不修复”仍约束正式方法与训练，
不再禁止为本问题补充直接相关的诊断。旧梯度增长goal保持暂停，不部署、重启或改动
正式训练，不研究训练质量、超参或其他未建立直接联系的问题。
多span读出沿用原接口：T为固定target offset并集，F_T(h)=sum_{t in T}log p(h_t|h_<t)。
本次删除集合D限定为现有source集合的子集，EOS同位替换，固定原target/其余历史/观察；
测量Delta_D(T)=F_T(h)-F_T(h_without_D)。它是固定历史的条件评分干预，不冒充重跑环境。
一次联合DT的sum_{i in D}d_i(T)作为该局部干预的预测接受实测检验，联合守恒不是
局部预测准确性证据；不要求所有单token都等同字面单删，也不以概率反推越界判无效。
同一事实/参考端点、source集合下，联合读出与逐span读出对应同一加和目标；改变
target读出时不能同时偷偷改变source/EOS背景。复用FlashTrace原作者累计删除/RISE/MAS
与DT原target读出，保留原始log评分和逐span变化；作者正向归一化指标不能代替负尾
预测方向/幅度检查。先明确预测对象再测，不制造影子归因、评分或评测实现。
本条不更改Q/V/A、现有指数组合、官方GRPO乘子、方差缩放、观察mask或官方PPO。

**2026-10-15 用户再次明确研究前提：利用DT的预测能力，不再重新审理DT是否本身等同删除基线。**
后续分析必须遵守此前已接受的这一前提。不得把实际DT逐token分量必须等于原生
单EOS删除log概率比，重新设为研究目标、合法性判据或修复前置条件；也不得仅因
将DT分量按字面概率比反推得到越界值，就宣称DT预测信用已经失效。
此前据此作出的“本例DT估计已失效”结论撤回，不删除原始数值和历史诊断证据。
如评价预测能力，参照必须对应DT实际预测的对象，并与当前异常建立直接关系，
不能默认原生单删就是所有DT分量的真值。当前要解释的是该换行token的大负DT值
如何形成、其预测依据以及进入既定优势组合后的实际作用，不转成删除基线等价性研究。
保留用户的精确token价值理想化假设、既有Q/V/A与官方PPO；本条不修改DT计算、
指数优势公式、官方GRPO组乘子或方差缩放。用户仍要求先解释、暂不修复，未授权
据本次澄清部署新方法、重启训练或运行新诊断；当前实际版本以RUNTIME_RECORD为准。

**2026-10-15 用户更新运行安排：停止GRPO，继续既有官方PPO。**
GRPO保留已完成191轮历史与完整190检查点，不再训练。既有原生PPO从其自身完整30检查点迁移至释放的GPU4/5，保留模型、optimizer、scheduler、RNG/rollout RNG、数据游标与原预算；这不是将GRPO检查点转换成PPO。DTPO v3方法不变，根因排查暂停。仅运行安排覆盖下方历史“GRPO保持”的说明；具体提交/实际恢复状态以RUNTIME_RECORD与current_runtime为准。

**2026-10-14 用户批准 DTPO v3：原官方归约、官方组乘子、仅标准化方差。**
仍使用固定 VERL 的原 actor/core，`token-mean` 及 B4/双卡/global64 的
`L=(1/16)*sum_j(S_j/N_j)` 保持，不部署自编的全局 token 分母候选。
在下述官方 GRPO 组乘子与原 DT 组合之后，v3 对完整 rollout batch 的有效
policy mask 调用原 `verl.utils.torch_functional.masked_var`，使用与原白化相同的
`1e-8`，仅执行 `A / sqrt(masked_var(A, mask) + 1e-8)`，不减均值。
原 `masked_whiten(shift_mean=False)` 会先减均值再加回原均值，不符合本次
指定语义，因此复用官方方差统计与原生 Torch 缩放，不复制方差算法。
零组乘子、观察与 padding 的优势仍为零，非零优势的符号不因该缩放改变；
Q/V 原字段、原环境奖励、原 DT 数值/内存路径及 PPO/optimizer/scheduler 不变。
版本标记 v3，从基础权重重新启动 DTPO，原 GRPO/PPO 对照保持，不恢复旧 DTPO。
32k/LoRA8/16/B4/global64/330/save5/keep2 保持。本段覆盖下一段历史 no-whiten
安排；方法更新与真实导入核验不能直接当作历史梯度加速增长已修复。

**2026-10-14 历史 v1：官方 GRPO 组乘子且不做 DT 白化。**
在原完整 rollout batch 上直接调用固定 VERL 的官方 GRPO 优势组件，使用原
`uid`、`traj_uid`、奖励、mask 及原组内标准差设置，得到每条轨迹的组内优势乘子
`g_Y`；不复制组均值、标准差、epsilon 或组处理算法。新的 source 系数为
`g_Y[-expm1(-d_i(Y))]`，self-target 系数为 `g_Y`，O/padding 为零。
这是用官方 GRPO 乘子替换旧的原始回报乘子 `r_Y`，不是额外再乘一次 `r_Y`。
原奖励为零而组乘子非零的轨迹仍须按同一 DT 接口计算，不能沿用零奖励跳过。
取消 DT 系数之后的整批 `masked_whiten`，不增加其他中心化、缩放或裁剪。
真实联合 target、逐 token signed DT、原模型/数值路径、官方 PPO/归约/optimizer/
scheduler、32k/LoRA8/16/B4/global64/330 预算/save5 均保持。
组乘子版本的 Q/V 字段是相应 surrogate 分解，不能继续声称等于原始环境回报
的 Q/V；环境奖励原值保留。原始回报 Q/V 的历史推导不因本改动变成新 surrogate
的证明。新的训练从基础权重开始，旧有缺陷 DTPO 不恢复，原 GRPO/PPO 对照不重做。
这是用户批准的方法改动，不是梯度范数加速上升根因已证明或修复已验收；继续研究
实际更新模式及累积内部传播增益，不研究共同分量是否应被归类为噪声。

**2026-10-11 22:54 历史运行安排，当时信用方法不变。** 用户明确要求DT从基础权重重新训练，
原25恢复分支已停止并保留。新DT PID2638949物理2/3；原生GRPO PID2590429物理4/5，
两组均无旧检查点、原330预算、LoRA8/16及每卡B4，当前首轮采样。实际版本与回执见RUNTIME_RECORD。
该段覆盖下方历史续训安排，不改Q/V/A、观测mask、白化或官方PPO/GRPO目标。

**2026-10-11 21:05 实际恢复确认，方法不变。** 原VERL已完成第25轮完整检查点
模型/optimizer/scheduler/RNG/data恢复；原trainer显示25/330，两正式worker进入
第26轮generate_sequences。已部署FSDP事件等待修复2534a670，LoRA8/16、每卡B4、
原预算保持；尚未完成恢复后的新完整更新。此状态覆盖下方较早“初始化待恢复”。

**2026-10-11 最新用户指令与运行状态，信用方法不变。** 第37轮已定位原FSDP
CUDA事件被通用Stream错误等待、参数通信缓冲区过早复用，导致输出层数据污染及
随后原KL反向NaN。最小owner查找修复已通过原事故回放和原Ray启动hook，源2534a670。
用户明确跳过NaN步不构成重启理由；第25轮完整检查点CPU检查通过，已提交原VERL
完整resume_path续训，恢复模型/optimizer/scheduler/RNG/data，保持原330预算、
LoRA8/16和每卡B4。driver1949625/出生1791723586.64，物理2/3；当前初始化，
检查点恢复及后续更新尚待实际日志确认。该段覆盖下方历史“不恢复/停止”的运行安排；
不改变Q/V/A、真实联合Y、白化或官方PPO。详见RUNTIME_RECORD与current_runtime。

# DeltaTrace token 信用：唯一固定计划

**2026-10-10 23:59 运行状态更新，方法不变。** TextCraft第37轮发生真实KL反向NaN，
原VERL跳过首笔optimizer更新；第38轮采样期间已停止原driver与两worker并保存现场。
当前不恢复检查点或部署未验收修复。第37轮已定位原low_var_kl指数溢出造成NaN梯度，
其前置actor/old logp异常差异仍在调查；第14轮首个异常算子仍未知。
该段覆盖下方“正式持续运行”的状态叙述，不修改Q/V/A、官方目标或正式预算。
源版本、有效参数与事故回执见RUNTIME_RECORD.md和current_runtime.json。

**2026-10-09 最新人类指令：新稳定版直接部署 TextCraft 正式预算。**
使用已验收 `fla-early-output-scale-20261009-v1`（数值源26bef6c8），实际GDN
SHA7c06d5e0…与验收绑定一致；不是未接受研究候选，也不退回旧FP16溢出路径。
原正式入口、真实联合action target、Q/V/A、整批白化、VERL/vLLM均保持；
仅修正启动器历史数值版本文字标签。物理GPU2/3，PID982372/出生1791553809.84，
2026-10-09 21:50:09起从基础权重启动，不恢复检查点，无诊断停止、首更新hold或timeout。
原30 epoch，原trainer实际显示330次完整迭代；每迭代32组×8条轨迹、4次全局64条
optimizer更新，不是只运行4次更新。LoRA8/16、每卡B4/双卡B8保持，参数逐项与原正式配置一致。
21:53:10原两worker已进入首轮generate_sequences，日志0/330；尚无完整更新验证。
2026-10-09 22:35只读核验：真实两worker已加载上述release与SHA，首DT各22个B4完成；
原PPO更新中，尚无完整更新返回。详见results_textcraft_live_dt_version_20261009.json。
22:41原step1完整返回，1/330并已进入下一次采样；原奖励均值0.664、grad_norm0.035，
无nonfinite skip/OOM。正式预算未截停，见results_textcraft_first_formal_iteration_20261009.json。
AppWorld仍停止，4/5用于已授权调试。该段覆盖下方较早“两正式任务停止”等状态，
不更改方法定义或宣称原PPO NaN、极端归因准确性已修复。源/配置/进程/验收关联见
results_textcraft_formal_stable_20261009.json及RUNTIME_RECORD.md。


**2026-10-09 用户要求统一负尾混淆矩阵与召回采集。**
此前各128个均匀source及预测尾部全数的采样框不同，阳性状态太少，不能拼接成
总体召回率。本轮只重新设计诊断采样，不改变任何信用、官方PPO或正式入口。
冻结开发/测试身份保持；以已完成的开发capture全部source为明确有限总体，
TextCraft85条、AppWorld122条（另2条缺失继续记录，不填阴性）。原保存DT输出
定义固定预测，c<=1和1<c<=2两层按总体比例、固定新seed无放回抽取合计1024位置；
c>2的37位置全数保留。抽样不读native反事实结果。每个位置保存n_h/N_h、总体
状态/轨迹/source权重；TP/FP/FN/TN用同一采样框及逆纳入概率计算，不追加旧点。
主报告每任务有限总体token计数及比率，另报状态/轨迹等权结果；不混淆两个estimand。
沿用原幅度分层及c>2、>10、>100累计阈值，主体/尾部分开，不混原始无界优势矩。
原native读出及同隐藏FP32/回舍入head结果分别保存；数值跨阈值单列未定，不当作TN。
抽样不确定性使用现有SciPy超几何分布反演；零命中不得报零宽度区间，数值参考
范围不是置信区间。1024仅为有界采集预算，不保证稀少分组精度；报告实际阳性数、
状态覆盖和区间。原作者累计删除/RISE/MAS继续独立负责方法质量，不由此替代。
使用原VERL初始化及原native/target读出，四个删除/事实pair组成既有B8诊断输入。
新manifest只增加任意概率样本的批次query列表，不复制模型、读出或算法。
两正式训练仍停止，不恢复检查点；已接受FP16范围修复保持，RMS和FA研究候选不部署。


**2026-10-08 最新用户指令：保存 PPO debug 现场，随后返回极端负信用的准确性修复。**
保留原 NaN 事故及已完成诊断的输入、源码身份、有效配置、随机状态和梯度；
当前一次原首轮诊断只补缺失现场，不重复有限回放、不据此宣称 NaN 修复。
极端归因研究恢复下述冻结集合、主体/负尾分层及原作者评测；不混入历史训练
退化原因，不新增外部裁剪、倍率纠偏、top-k 单删除精化或替代 Q/V。
本段覆盖下方较早的排查优先级，方法及正式入口不变；两组正式任务仍停止。

**2026-10-08 最新优先级：先 debug TextCraft PPO 非有限梯度。**
暂停新增极端归因候选研究，调试物理 GPU4/5；两项正式作业当前均停止。
本段只更新排查优先级与实际运行状态，不改变下述 Q/V/A、白化或官方 PPO。

**2026-10-08 最新运行安排：用户允许 TextCraft 跑起来，剩下两卡用于调试。**
物理GPU2/3的原TextCraft作业解除一次性首更新hold，沿用原版本、基础权重起点、
原Q/V/A、整批白化、官方PPO及既定参数，不部署研究候选、不恢复检查点。
物理GPU4/5用于固定开发集合上的极端归因对照；AppWorld正式作业不重启。
本段覆盖下方较早研究安排中的“TextCraft继续hold”，不改变方法定义。

**2026-10-08 用户再次明确本轮研究对象：极端 token 归因的来源与可信度。**
当前真实联合动作 target 版本是否训练退化尚未判断。此前熵损失主导的问题已找到
原因并修复，属于已处理的历史事项，不与本轮归因诊断混用。冻结集合、单删除、
逐层传播、概率边界和原作者累计删除/RISE/MAS用于解释及检验极端归因；
候选算子通过不代表极端归因已经修复，也不代表训练效果改善。下述 Q/V/A 与
正式入口不因这一范围澄清改变。


**2026-10-08 用户授权继续有限研究，并要求集合级开发与独立评测。**
本段只约束研究，不修改下述正式 Q/V/A、真实联合 Y、PPO 或部署入口。
复用当前首次真实 action-target rollout 的原始保存集：TextCraft 170 条去重轨迹，
AppWorld 224 条，其中 8 条尚无完成的 DT capture。它们来自原非零奖励 DT 请求，
不冒充全任务成功率样本；零奖励项仍按原公式处理。复制 UID 不重复计权。
任务分组使用原第一枚 policy token 之前的完整状态 IDs（含首个环境观测），
不能仅用 TextCraft 通用系统 prompt。当前有 32/33 个初始状态组。
冻结 manifest 将完整状态组分为开发/测试；已分析过的异常批次及同状态轨迹
留在开发侧。TextCraft 开发/测试为 85/85 条，AppWorld 为 124/100 条。
分组与抽取不读取 d、优势、成功排序或长度；缺失 capture 保留，不换样本。
第一阶段每个开发状态组按固定散列顺序取最多两条，共 32+32 条；不是手选异常点。
全开发与测试身份一并冻结，测试侧不随开发结果改选。

**同日用户补充：主体与尾部必须分层，不混淆矩与总体结论。**
对非零奖励的 source，以 `c=exp(-d)=p_deleted(Y)/p_factual(Y)`、
`A/r=1-c` 报告 `[0,1]`、`(1,2]`、`(2,10]`、`(10,100]`、`(100,inf)`
的频率与条件幅度，分别保留任务、初始状态组及已查看/未查看组。
这些只是诊断区间，不是裁剪规则、训练参数或官方容差。有限保存样本不能
证明总体期望/方差存在或不存在；不将主体与无界负尾部混成原始优势/误差均值。
第一阶段冻结的64条轨迹各均匀抽4个source，抽取不读d或优势；另对完整开发
capture内所有`c>2`位置作单删除对照，按上述尾部区间分别报告，不把尾部全数
和主体抽样直接合并计权。单删除只诊断估计误差，不替换任何训练信用。
取得原native单删对照后，用相同区间同时划分预测c与对照c，误差仅在两者
交叉单元内汇总；DT漏估的对照尾部不能混入“有界主体”的误差统计。
原作者累计删除/RISE/MAS仍按下段原接口与状态组聚合，与尾部误差分开报告。
先完成这一固定集合的基线测量，不据其中单个点新增候选或反复扩充样本。

原作者 `faithfulness_test_skip_tokens`（SHA583f4b7d...、原 k=20）继续负责
累计删除、RISE/MAS、密度和归一化；signed RISE 与仅评测用 positive MAS
分别保持原入口。每条轨迹先得原分数，按初始状态组内轨迹平均、再按状态组
等权平均；另报全部去重轨迹算术平均。两个环境与两个指标分别报告，
RISE/MAS均为删除曲线面积、越低越好（原README明确RISE↓/MAS↓）；上升不得报改善。
不自造合成分、不按 token 数或奖励加权，不把缺失/非有限结果填零。
候选/基线只比较同一 UID、同一 Y、同一权重、同一精度/资源设置，保留配对差
和状态组间统计不确定性。单点数值、守恒、核容差、累计删除质量各自报告，
不能互相代替；累计删除曲线也不等于逐 token 单删除真值。

**同日用户否决“精化四个最负 source”的候选：它处理输出症状，未修复已测
异常点的有限传播问题，且增加多次整网前向。该候选撤回，未部署、未启动新增模型查询。**
准备过的查询和源文件仅保留为未接受的历史草案，入口禁用，不构成研究/训练路径。
开发集、测试集及原作者指标保留，不因候选撤回重新挑选。

**同日用户要求回查以点带面的决策错误。** 联合背景/事实单删除背景之间的
有限传播误差，已在当前 AppWorld 异常 token 上定位；它在集合中的频率、
误差规模与实际更新梯度影响尚未建立。不得把该局部定位写成整体训练退化
的已定位原因，也不得在只附加“仅该点”限定后仍据此独占后续修复方向。
已完成的 GDN Q/K/V/gate/state 耦合推导只保留为局部机制的研究结果；暂停
新增该候选实现，先回查既有结论与优先级，并读取冻结开发集的原始保存值。
负 d/负优势的发生率不是估计错误率，训练系数的平方和不是实际梯度占比；
当前真实 action-target 首批数据也不能冒充旧辅助读出训练退化窗口的数据。
原作者累计删除/RISE/MAS、单删误差和原 PPO 梯度各自回答不同的问题。
后续若形成原 owner 候选，应明确支持它的集合证据、保留的项及全 token
计算/存储量；不得以单个异常点的局部改善推断整体有效。
不能用极值筛选、额外单删除覆盖、信用裁剪/缩放或单独替换 V 来绕过原因。
不因某个点改变方向就认为修复；集合级累计删除、RISE/MAS 和实际 dtype 的
原官方算子容差仍分别验证。没有推导与成本依据不新增 GPU 候选测试。
不扫参数、不反复重抽开发集、不在看过测试后继续调参；已失败候选不重复试。
只让推导支持的候选进入固定开发集，不支持则淘汰该推导，记录原因而非堆补丁。
累计删除的同一原生 B4 前向应同时计入四条轨迹，不能为每条轨迹重复另外
三条控制行的整网计算。独立测试只评被冻结的最终候选，不能把首个 B4 或
少量已知异常点当作集合结论。正式训练的一次 DT/B4 成本目标保持不变。
该安排记录时物理 GPU4/5 仅作本研究、TextCraft 原训练继续 hold；最新运行安排见顶部。
AppWorld 不重启，
不恢复检查点。原核若被改动，仍使用对应 FA/FLA 原参考、实际 dtype 与原断言；
本段统计比较不是新的算子容差。

**2026-10-07 最新用户确认：实现并正式运行真实 action target 接法。**
target 必须是实际 LLM 输出中由任务官方 parser 执行/验证的 Action 或代码片段，
包含原 parser 识别该动作所需的生成语法；原归一化后的 Action/代码另存作执行核对。
不再使用额外询问和人为奖励类别标签。TextCraft、AppWorld 都在 episode 末结算
一次奖励；同一奖励对应一次联合动作目标 Y，不按每段动作重复乘同一奖励。
对 target 自身的每个生成 token，按用户指定的 literal 删除定义
`p_without_i(Y)=0`，该事件的直接项为 `Q_hat=r_Y,V_hat=0,A_hat=r_Y`。
这项替代同一事件的 DT 项，不再叠加其对同一 Y 内后续 logits 的归因。
其他生成 token 使用正式 DT 的逐 token signed 删除效应及下述采样修正；
O 仅作为原状态，ratio/loss mask 仍为零。每个 token 仍乘自己的 PPO ratio。
不按 target 长度分奖，不额外补倍率，不分别白化 source/target。
沿用原 trainer 的整批 masked_whiten 和原 PPO；从基础权重新开实验，不恢复
已停止的辅助标签实验检查点。现有两个任务各两卡，物理2/3 TextCraft、4/5 AppWorld。
本文件下文均按这一当前方法解释；旧类别标签、逐 response 回报读出和旧任务安排
只属于历史版本，不再构成当前实现要求。其他官方环境和训练参数不因该目标修改。

固定 VERL 官方损失默认设置（entropy=0.001、dual-clip=3）保持不变。
依据用户既定约束，禁止 O(n²) 的归因调用。直接使用 EOS DT 对同一轨迹的
真实联合动作目标 Y 做一次归因，不再逐 response 重放越来越长的前缀，也不展开
response × 未来奖励事件。不逐 token 询问，不额外采样参考 token。
本文件是唯一方法规范；README 只记录实现状态，环境记录只规定复用。

**2026-10-06 用户批准：在原 trainer 的 DT 优势出口复用固定 VERL 的
`verl_F.masked_whiten`，以整批已采集的有效 action-token mask 计算一次，
然后交给原 actor 分发、microbatch 与 PPO loss。保留原始 d、Q_hat、V_hat、
A_hat；白化只产生 actor 使用的训练系数，不改写反事实价值定义。
不在每卡或每个 B4 内重新白化，不另做 GAE。
不能仅因任务梯度变大而宣称学习退化已修复。**

**当前范围：仅 TextCraft、AppWorld 两组 DTPO，各两卡；设备以最新运行记录为准。
只复用任务作者的环境、数据、采样/评测接口；不引入 SkyRL、
AgentGym-RL 或 LOOP 的训练器及推理服务。AppWorld 调用 LOOP f14107a 的训练基准
环境，不混入历史 VERL AppWorld 奖励。训练由已验证的 VERL-agent / vLLM 承接；
数值版本以 verified_runtime.json 的官方对照回执、哈希及运行开关为准。
以下 DT Q/V/A 不变。环境比较、启动、容量检查分别报告，不能代替训练验收。
SkyRL-SQL、Sokoban、WebShop 及 GRPO 均不属于当前运行范围。**

**资源约束：实际 actor microbatch=4、DT minibatch=4、LoRA rank=8、
alpha=16 固定，不用更小 microbatch 配合梯度累积代替 B4。原官方 PPO 的
optimizer minibatch 和梯度累积保留。总上下文32768的更新容量、对应
官方容差与效率分别核实。此前microbatch1的容量回执不满足此约束。**
双卡容量按最新要求实际提交B8，由原VERL分发为每卡B4；不把原任务的
全局PPO minibatch或更新次数改为8。

**同日最新用户指令优先：本文件只固定信用分配；信用以外使用官方实现与配置，
不得用旧 PLAN 阻止恢复官方行为。** 原 PPO loss、无效动作惩罚、任务历史窗口、
采样与更新由固定 VERL-agent 的原代码负责。训练所需的机型、模型、内存兼容
改动须逐项记录，不能称为原论文同配置。DT 容差复用对应 FA/FLA 原参考和断言，
不把整网守恒、短链逐值一致或自己设置的倍率变成官方验收标准。

## 前提与固定接口

保留用户的理想化假设：**DT 估计的 token 反事实价值完全准确时，就是对应
真实世界变量的反事实价值。** 实际有限传播允许估计误差，不要求新的环境
反事实模拟器，也不要求开发前达到零误差。组合规则、估计误差、接口错误、
资源效率分别核验；不因其中一项近似而撤回其他已经成立的部分。

每个 reasoning、工具调用、final token 都是独立 policy action。工具观测 O
属于状态，不是 action；O/padding 的 ratio 和 actor loss mask 为零。不把
O 的 token 链当作工具的自回归生成过程，不用任意 span 标量代替逐 token 计算，不做
O credit 路由，不引入 value model/value loss 或第二套 PPO。
上述已知 self-target 边界使多个 token 的直接项恰好相等，不是另外引入 span action。

## 终局奖励到轨迹 token 的计算

当前 TextCraft、AppWorld 都由官方环境在 episode 结束时结算一次奖励，
当前官方 PPO 配置 gamma=1。直接读取原 trainer 奖励处理后的整条轨迹
`token_level_rewards` 之和 r_Y；不在信用接线中重算奖励或处罚。
一条轨迹的 Y 是其实际执行过的 Action/代码片段构成的联合目标，
使用原完整执行史、原 token 身份及官方 parser 的 target 位置。
不按 response 重复分配同一 r_Y，不追加回报询问，不编码人为奖励类别。
预算终止保留已经发生的官方奖励，不补造终局奖励。
对不属于 target 的 policy source token，DT 估计：

\[
d_i(Y)=\log p_i(Y)-\log p_i^{\setminus i}(Y),
\]

其中第二项是 EOS 删除 source token i 后同一真实联合目标的反事实概率。
精确假设作用于
这个逐 token 的量；不会把一条轨迹归因向量的总和守恒当作逐 token
反事实准确性的证明。

Y 和 r_Y 来自事实 rollout。source token 的采样组合为：

\[
\widehat Q_i=r_Y,\qquad \widehat V_i=r_Y e^{-d_i(Y)},
\]
\[
\boxed{\widehat A_i=r_Y\bigl[-\operatorname{expm1}(-d_i(Y))\bigr]}.
\]

target 自身按已接受的 literal 删除边界使用
`Q_hat=r_Y,V_hat=0,A_hat=r_Y`，替代同一事件的 DT 项。
联合 Y 的多个 target offset 由原 DT 目标接口求联合 log-prob；
**不能对各个片段分别完成奖励修正后再相加**，也不假设各步奖励独立。
不能用 r_Y*d 替代，不能再次乘 reward、取绝对值，不能对原始
d/Q_hat/V_hat/A_hat 归一化或裁剪指数。批准的 actor 优势白化见下节。

在参考结果分布被事实结果分布覆盖时，精确 d 满足：

\[
\mathbb E_{Y\sim p_i}\left[r(Y)(1-e^{-d_i(Y)})\right]
=\sum_Y r(Y)\left[p_i(Y)-p_i^{\setminus i}(Y)\right].
\]

外部 reward 是数值系数，无需对环境求导。实际 r_Y=0 的样本项为零，
可省去 DT；不因此从目标分布删除零回报结果。提前停止和预算终止按官方行为处理。
该概率比与实际奖励的组合对应 HCA 的采样形式
（[原文定理2、式6](https://papers.neurips.cc/paper/9413-hindsight-credit-assignment.pdf)）。
对奖励结果分组与对完整结果求和，在对应精确分布下可有相同的期望差；
当前实现固定使用真实联合 Y，不重新启用旧奖励类别读出。

## 与 PPO 的对应及等价范围

| 项目 | PPO | 本方案 |
| --- | --- | --- |
| 状态/action | token 前缀 / 生成 token | 相同 |
| Q | 条件未来累加回报 | 事实回报的采样估计 Q_hat |
| V | 当前状态的价值 | 由 DT 反事实比值得到的 V_hat，不训练 value model |
| Advantage | Q-V 或其采样估计 | A_hat=Q_hat-V_hat，稳定实现用 expm1 |
| 工具调用 | 价值含调用结果对后续奖励的影响 | 同一事件反事实定义包含该链路 |
| O | 状态，不参与 actor loss | 相同 |
| Actor loss | 原 token PPO ratio/clipping、原熵项 | 完全复用固定 VERL 的原实现与默认设置 |

这里的 A_hat（也记作 C_i^{DT}）已经汇总该 token 之后的奖励事件。原始值保留；
actor 的 `advantages` 由官方 `masked_whiten(A_hat, response_mask)` 产生，
mask 外存为零。每个生成 token 的训练系数进入自己的 PPO ratio 和逐 token
clipping。**不再对 A_hat 做 GAE、第二次
return-to-go 累加或 span 平均。** O/padding 仍不参与策略损失。

白化使用官方默认 `shift_mean=True`、官方 masked variance 与 epsilon，
不复制公式或自定容差。它是优化预处理，白化后系数不再逐值等于 Q_hat−V_hat。
原始 Q/V/A 的等价范围保持如下；有限 batch 的中心化、缩放可能改变 PPO
clipping 的分支和 PG/entropy/KL 配比，不能宣称整个 clipped 更新逐值不变。

保留既定精确 PPO 对照目标：

\[
Q^\pi(h_i,a_i)=\mathbb E[G_i\mid h_i,a_i],\qquad
V^\pi(h_i)=\mathbb E[G_i\mid h_i],\qquad A_i^\pi=Q^\pi-V^\pi.
\]

当 DT 提供的事实和反事实 token 价值分别对应这里的 Q 和 V 时，上述采样
组合在条件期望上等于精确 Q-V；相同 advantage 进入相同 PPO loss，给出
相同更新信号。**单条蒙特卡洛样本不被宣称为精确期望值；无偏 advantage
也不自动证明有限采样的 clipped loss 无偏。**

EOS 删除端点首先定义删除反事实。仅“删除反事实精确”不能额外证明该端点
天然等于 PPO 的动作边缘 V；两者的价值语义必须对齐，才使用 PPO 等价结论。
这项范围说明不引入额外参考采样、逐前缀预测或替代算法；实现保留用户选择的
EOS DT 和上述 Q/V 数值接口，验证中分别记录组合正确性与 DT 估计语义。

**同一期望策略梯度不要求删除参照等于 PPO 的 V。** 若同一前缀的统一删除
参照价值 B(h_i) 不依赖实际选中的 token，则精确 DT 组合的条件期望为
Q(h_i,a_i)-B(h_i)。因为 E_pi[grad log pi(a|h_i) B(h_i)]=0，它与减去
精确 V(h_i) 的期望 actor 梯度相同。这是当前 EOS 方法的基线消去性质，
不是另增一个 B 模块，也不需要额外计算或采样策略参照。等价范围是采样策略
处、相同状态/token 权重下的期望梯度；不能扩大成所有 clipping 更新相同。
不能固定由实际 token 导致的事实后缀，再把该删除参照宣称为动作无关。
实际 DT 分解对这个理想干预的近似质量仍单独核验。

## 真实动作目标与正式 DT 接入（2026-10-07 替代旧类别读出）

正式 runner 使用原全词表文本目标分支，不增加参数、训练头、预测询问或类别标签。
当前正式任务的事件是一次官方终局结算；Y 是实际执行过的动作片段的联合目标。
原环境 parser 负责片段选择，薄接线只保留其位置与原 token 身份。

1. 复用官方 rollout 的原始 prompt、response IDs、attention mask、traj_uid、
   env_step 和实际每步 reward；在官方 reward/penalty 之后读取训练奖励。
   接线不对原 owner 的 action IDs 做 decode/encode 重建；TextCraft 作者原本的
   decode/encode 训练表示保留，不能冒充 raw sampled IDs。
   原 `adjust_batch` 可能复制行，
   balance 可能改变顺序；以原 traj_uid 对齐，每条实际轨迹归因一次，
   再按原训练行顺序返回，不把复制行当成新环境奖励。
2. 一条轨迹至多一个联合文本目标请求。使用原完整执行历史、原 observation
   和原 policy mask；原训练截断若只保留部分 action，归因后仅映射回原保留位置，
   不删除实际发生的后续奖励或动作。总长度仍必须在既定32768内。
3. target offsets 来自官方实际解析的 Action/代码片段。调用原
   PackedAnswerTargets 全词表分支，以多个 offset 形成同一联合目标的
   log-prob 总和，再只做一次奖励修正。不能逐片段各乘同一个终局奖励后相加。
4. 事实端点为原完整 token IDs；参考端点仅把非 target 的 policy source token
   换为 EOS。target 与 O 保留原值；不会把 O 当成模型生成的目标。
   target 自身的删除概率零直接项走上述已知边界，不计算 log(0)/无穷大。
5. 调用正式 `PackedAnswerTargets` / `FiniteAnswerOps` / `runner.attribute`，
   一次有限归因返回该轨迹 source token 各自的 signed log-prob 贡献。
   该向量是 DT 对理想逐 token 删除效应的实际估计，不是已经逐 token
   穷举测量的删除差。保留 signed 值，按原 action 索引接入前述采样公式。
6. 单一终局事件的组合为 source 的 r_Y[-expm1(-d_i(Y))]，target 的 r_Y。
   两者调用同一 Q/V/A 组合接口，self_target_mask 只替换同一事件分支。
   O/padding 为零；没有重复的未来事件矩阵或独立奖励分账模块。

这里 source 的 DT 输入是已发生的真实完整执行历史；工具返回保持原状态，
不被重新解释成自回归生成，也不另造 O→A 分账。实际联合有限分解对理想
逐 token 删除效应的近似质量需要单独测量；
其输出守恒、接口形状正确，均不能代替单 token 反事实精度检查。

按原分布式入口打包，每卡 DT minibatch4；一条轨迹一个请求，不按轮次重放
越来越长的前缀。source 保留独立 signed 值，self-target 使用已知直接项。
多 token target 的实际 dtype 数值对照和输出头工作量须单独核验，不能套用
旧类别 FP32 head 的回执，也不能因旧 kernel 通过而宣称新整条路径已验收。

## 当前两个任务的真实奖励

奖励来自固定作者环境和原 trainer，不在信用适配层重新计算评分：

- TextCraft：使用 AgentGym 原环境的配方、库存和目标检查。成功时原 reward=1，
  其他交互为0；原 rollout 将最终 episode score 结算一次。Y 取真正执行过的
  Action 及其识别语法，不把 observation 当 target。
- AppWorld：使用 LOOP f14107a 的训练基准，读取原生 rollout 的 `ret`，
  当前为终止时官方测试通过比例。原 `eval_result.num_tests` 仅作环境评分元数据，
  不再用它生成奖励类别或标签。原配置两项动作/执行失败惩罚均为0；
  独立评估仍保留 LOOP 的原生稀疏成功评分，不能用评估分数覆盖训练回报。

当前两组未启用 VERL 的额外 invalid-action penalty；如果官方实际配置改变，
信用入口读取原 trainer 处理后的奖励，不复制处罚公式。
历史 WebShop、Sokoban 过程奖励类别、逐 response 累计回报以及 SQL/GRPO 资源安排
均已退出当前方法和运行范围；保留在历史版本及运行记录中，不作为当前配置要求。

## 实现与验收约束

- 2026-09-29 用户澄清：存在数值残差本身正常，主要判断是否超出对应官方容差。
  整网守恒残差只作定位诊断，不另设归零或百分比门槛；已通过的算子不因
  此诊断再加补偿。实际超差、非有限值与普通低精度误差分别记录处理。
- 2026-09-29 用户再次要求：DT 有数值问题时，补齐保存的实际操作数 dtype
  对照并定位残差来源；记录 q/k/v、beta、g、缓存状态、上游系数及参考的
  dtype，不能用不同 dtype 的随机测试冒充实际路径通过。FA/FLA 原参考和
  原容差保持不变，不添加强制守恒缩放、裁剪或其他“纠偏”来让结果通过。
- 2026-09-29 用户要求：PPO 除信用来源外，须对齐同一个强化学习框架的官方
  行为，并采用该框架对应测试原有的容差。此处 owner 仍为固定 VERL-agent，
  任务工作量继承作者配置，PPO 实现与测试仍由 VERL 所有，不用 LOOP 的训练器
  或测试冒充 VERL 行为。对照传入完全相同的
  token IDs、mask、advantages、初始模型/optimizer 状态及配置，只检查信用
  接口之后的原 PPO 路径。原测试未覆盖的量须明确说明，不能自创或放宽容差，
  也不能把 FA/FLA 算子阈值扩大成整条 PPO 阈值。以下 FA/FLA 要求继续只用于
  各自对应的算子；本条不改变上面的 EOS DT Q/V/A、reward 或 token mask。
- 同日用户明确选择“恢复 VERL 官方默认损失设置”：删除启动器对
  `entropy_coeff=0`、`clip_ratio_c=inf` 的覆盖，继承固定 owner 的
  `entropy_coeff=0.001`、`clip_ratio_c=3.0`。PPO 对照与 DT 使用相同设置。
  熵项不是 DT 信用；报告更新时分别保留原 actor 的 policy/entropy 指标，
  不能将仅由熵项造成的参数变化称为非零 DT 更新。环境 reward 与 Q/V/A
  继承官方奖励处理，包括原生无效动作惩罚；由上述信用接口读取结果。
- 正式 DT 负责有限传播，现有 VERL 负责 rollout、PPO clipping、optimizer，
  官方 task worker 负责 reward。禁止复制这些实现。
- 用户固定 LoRA rank=8、alpha=16，不得更改。旧rank=1/alpha=2
  的容量记录不代表当前配置通过；资源调整只能保留该LoRA设置后处理。
- 当前两个 DT 任务各用两张卡。Qwen3.5-9B、总长度上限32768；
  实际 minibatch 通过官方分布式配置安排，每卡 actor microbatch4/DT batch4。
  信用公式不依赖单卡假设。原单卡
  验收不能冒充双卡请求分发、FSDP collective、奖励/优势对齐已通过。
  PPO optimizer minibatch、epochs 和梯度累积继承各任务已对齐的官方配置，
  不把双卡 B8 的容量测试批量改成正式 optimizer minibatch。
  smoke 同样保持 32768 上限，实际长度另报，不能用 padding
  冒充原始任务上下文。真实 target 已在原轨迹中，不再追加询问或类别 token。
- 2026-09-23 用户明确要求 DT 本身使用 minibatch，效率应接近一次反向或更快。
  调用正式 runner 的 batch 接口，保留每个样本/奖励事件的独立归因。
  用同模型、同卡、同长度、同 batch 的原生反向作为实测参照；分别报告完整
  DT（含端点前向、重算、传播）和各阶段成本，不能用 PPO 更新耗时代替纯反向。
  没有完成该对照、32k 容量和数值校验前，不报告效率要求已达标。
  加速优先参考已验证的官方反向设置及执行路径，包括卸载、重算、保存张量、
  FA/FLA 内核与批处理；先比较已暴露的主要阶段，尽早确认 DT 多出的计算和
  搬运是否必要。原版可行反向保持为主基准，不用新的较慢变体替换它，也不
  为统一内部卸载方式而削弱官方路径；确需不同策略时记录实际成本与原因。
- 2026-09-22 用户指定后续全部改用 MetaX，不再使用 A6000。先读环境记录并
  复用 Python/权重/持久缓存；先确认空闲显卡。平台变更不改变上述 Q/V、
  信用或上下文约束。2026-09-22 用户进一步明确：MetaX 按物理 64 GB
  显存、不 OOM 验收，不再要求 48 GB；物理容量与实测占用仍分别报告。
  32k 容量须用已知长度的输入主动测试：DT 原执行史及目标位置所在输入恰好
  32768 tokens，并验证原始 PPO 更新；不能等待真实任务偶然接近上限。
  容量夹具与原始任务长度分开报告，越界必须明确拒绝而非静默截断。
  预算终止保留已经执行的 action 和官方 reward，不再生成或执行工具，
  不补造终局 reward。当前 AppWorld 的环境、历史渲染、采样长度及停止行为
  继承固定 LOOP 训练基准；历史 VERL AppWorld 的 history_length=2、字符窗口
  和额外 DT query 预留不再作为当前要求，不覆盖作者的原生行为。
  显存修复必须同时对比相同输入的预热后耗时，不能以严重拖慢训练换取通过。
  归因阶段使用正式 runner 要求的 FA，结束后恢复原 actor attention 后端，
  包括异常路径；不能让前向专用 FA wheel 接管 PPO backward。
- 复用已有动态 shape 执行配置，记录首轮/后续耗时和编译；不能清缓存假装恢复。
- 测试必须分清：数学组合、正式 DT 调用及单 token 对照、真实任务 reward、
  上游 actor 更新、32k 容量、完整训练/评估。历史结果不自动覆盖当前实现。
- 相同权重/输入/随机种子下的 PPO 短链保留为对照诊断；不再用 FA/FLA
  算子误差替整网梯度或多步更新设阈值。信用以外调用同一框架原代码与原测试，
  原测试未覆盖的量如实记录；既不据差值自动判错，也不据差值小就宣称通过。
  用户进一步确认：数值容差直接采用固定版本 FA/FLA 官方测试的计算方式，
  比较同一初值下单步前向/反向；允许与这些低精度实现相同的多步累计误差，
  不另设多步逐位相同、更新方向或 clipping 分支完全一致的验收门槛。
  FA 2.6.3 `test_flash_attn_output`/`test_flash_attn_varlen_output` 比较同一
  Q/K/V 的 FP32 参考与普通低精度基线，输出最大绝对误差为基线的 2 倍、
  dQ/dK/dV 为 3 倍，不能笼统写成所有项目 2 倍；其他测试的例外须按原函数。
  FLA 0.4.1 `test_chunk` 调用原 `assert_close`：误差 RMS / (参考 RMS+1e-8)，
  o/ht 阈值 0.005，dq/dk/dv/dh0 为 0.008，db/dg 为 0.02；绝对误差 <=1e-6
  直接通过。不启用 `FLA_CI_ENV` 的 warning 豁免冒充严格通过。上述断言在
  各自对应的算子量上复用，不把权重梯度任意改名成 dq，不虚构官方 PPO 阈值。
  已有多步结果保留为诊断，当前两任务仍须实际连续运行且无 OOM/非有限值。
- 同步、提交时保留他人改动。遇到实际无法继续的问题，停止受影响部分并给出
  具体事实，等待用户处理；不得悄悄换用被否决的方法。

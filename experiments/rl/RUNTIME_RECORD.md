# 当前运行版本与修复记录

## 2026-10-09 保存 PPO 调试现场后，返回极端负信用：原卷积接入已完成局部对照

实际执行数值源已与提交92fb84301cea314db126a104029a1fbfe3b708a7的Git blob逐字节SHA核对；
仍是隔离研究接入，未部署/未接受完整DT修复。

PPO原NaN调试现场44文件、1,089,145,135字节已有逐文件SHA核验的本机副本，
无新增PPO回放、不称原事故修复。回执results_preserved_actor_debug_20261008.json。
两项正式任务仍停止，不恢复检查点；本次继续联合背景有限传播研究，Q/V/A、
白化及原VERL PPO没有修改。

隔离候选conditional_conv_windows SHA cb336dc1，复用实际安装的原
causal_conv1d_fn SHA7286f939和FLA l2norm_fwd SHAfca8a850；只打包单个
source及其原3行历史、后3行事实输入。每个tile两次原卷积调用；训练路径
没有逐source整网前向。原FLA forward源码已核对使用同一个l2norm_fwd。
实际GPU4检查PID2738633/birth1791489789已退出，启动基线1d908254，
实际源码按launch SHA绑定；尚未部署默认GDN或正式训练。

历史真实B4算子输入的199位置全数对照，两个dtype各796行中实际改变99行；
其余为零变化控制，不能把796行全报作非零反事实。四位置卷积pre/fused输出、
Q/K/V与完整原生调用逐值一致。原卷积v1.5.0 test_causal_conv1d的dtype
rtol/atol由原AST提取：FP16(0.003,0.005)、BF16(0.01,0.05)，52项原
public fn/ref前向对照通过；没有改阈值、不冒称执行完整官方测试或反向测试。
两case含诊断参考耗时9.94/9.25秒，峰值allocated2,728,526,848字节，
reserved3,141,533,696字节；不是完整DT成本或32k容量结论。
这些历史算子输入不替代冻结开发集/测试集的原作者累计删除、RISE/MAS。

原只读检查曾因CPU入口关闭可选kernel返回None；换用原任务source环境后
入口可用，未重装。其递归receipt glob又阻塞在NFS getattr，原Python/内核
调用栈、源码、PID已保存，仅停止本检查进程并去掉该递归扫描；未动训练。
本次8文件的新张量/源码/结果已在远端归档304,514,612字节并逐文件SHA记录；
本机已保存源码、结果和manifest，新大张量未异机复制，不能称本机全量备份。
完整回执results_conditional_conv_20261009.json。下一步才是与条件状态core
合并、原chunk有界tile调度及固定集合质量检查；完整归因修复尚未验收。


## 2026-10-09 条件窗口有限传播：原核组合与导数极限已验证，非零及整网尚未验收

代码及回执已提交8d38802bb4da9c6834be338861b4cc07f0a5692f；原运行launch仍记录启动时
Git基线30bb2ded，实际执行文件按SHA与此提交逐字节核对，不能混作部署状态。

原FLA chunk_fwd_o SHA548cd026直接复用，不复制原状态递推/反向。真实B4、
128token、8头的两dtype原o/dq/dv组合12项通过；刻意保留的4项非连续GPU
布局反例仍失败。根因是新诊断低层读出漏用原input_guard给出的连续布局，
不是原正式训练路径的缺陷。v3诊断启动时误替换了操作数目录，CPU加载即
失败；原命令、源码、错误均保留，未发生数值调用。

后续组合SHA9f8e8c59在原B4、199token、非零初始/未来状态、末尾部分chunk
上，FP16/BF16原o/dq/dk/dv/db共10项通过。PID2507832/birth1791487569.39
已退出，两case11.83/9.99秒。原原生梯度公式/FP32参考和容差不变。

研究内条件四位置传播SHA8e4c0502同时保留两个端点方向、卷积位置之间的
状态耦合、首位alpha/beta和固定事实未来。原核组合计算40个readout/源tile，
不是40次DT或整网前向。PID2548305/birth1791487938.73已退出，两case
3.97/2.71秒；原FP32 dq/dk/dv/db/dg导数极限28项全部通过，实际readout
各40次。尚未对非零有限变化或整网信用作通过声明。实际默认owner仍为
TextCraft ef55ce08、AppWorld448ef32c，训练路径、Q/V/A/PPO均未部署此候选。

真实B4x7999x8头、同GPU驻留的原有限FLA热调用46.42-46.50ms，原readout
0.89-0.93ms，不能外推完整候选速度。所有查询/因子全序列铺开的当前实现
算术上需39GiB，已排除；4096源+原64chunk halo的保守额外张量上界约
7.24GiB仍不包含原模型/捕获/allocator，也不是实测容量通过。当前core
仅有界诊断，尚未完成生产tile调度、原因果卷积接入与集合评测。

非零有限检查PID2572344/birth1791488186.74已退出，两case15.28/11.39秒；
真实B4x199位置全数诊断，无极值筛选。局部有限变化的相对RMS残差FP16
0.000374689、BF16 0.00215830，实际非零对照795/604项；不是原FLA官方
有限标量容差通过。其26项原native输出o仍通过原0.005阈值。详见
results_conditional_memory_20261009.json，隔离core SHA8e4c0502未部署。
三个新诊断的精确张量/源码/日志共在远端完成归档并记录逐文件SHA；大SCP
传输超时，本机只保存元数据/源码/结果及未验证的部分归档，不冒称已完整
异机复制。保存工具现先落盘远端manifest，避免传输失败丢失哈希现场。
两个正式训练均停止，PPO原NaN现场已保存且未宣称修复；不恢复检查点。


## 2026-10-09 冻结单背景诊断两任务全部完成，联合背景误差有集合证据

TextCraft与AppWorld各165个冻结source位置完成；AppWorld严格合并原74项、
原失败B4重放8项和缺失查询83项，无重复或换样本。原失败记录仍保留为失败。
补齐PID2032920/birth1791482964.86已退出；两rank各20次B4 DT，分别
1012.36/1012.51秒；无新增rollout、optimizer、checkpoint restore或native单删查询。
实际worker SHAb6c4c341、runner628006b6、AppWorld原source58209daa与
隔离数值owner33b169b3绑定记录；代码已提交3852dbb6，启动时Git基线26a6ba76
不能误当成worker源码身份。默认owner和正式启动入口未部署研究候选。

单背景诊断中，原原生单删对照不支持的负尾TextCraft18项、AppWorld11项
均不再落入c>2，每任务涉及6个初始状态。这支持联合背景分配是这批误报的
重要来源；不是GDN整体主因结论，也不是训练信用已修复。TextCraft仍有1项
轻微反号，AppWorld均匀抽样4项漏报尾部中仍有1项未进入尾部；有限传播和
端点精度残差分别保存，主体/尾部不混合汇总。原作者累计删除/RISE/MAS
仍是方法验收，不由这个诊断或官方核容差替代。

补齐现场61项25,370,477字节，本机归档2,819,506字节，直接核验归档内
每个文件SHA256，未因Windows长路径解包失败丢弃现场。两组正式训练保持停止，
原PPO NaN现场已保存，未宣称原事故修好。完整来源、分层结果和资源观察见
results_single_background_completed_20261009.json；末态主机可用约1034GB，
该数不是运行峰值。原FLA只读readout API映射属于推导，未实现/接受信用候选。


## 2026-10-09 原 FLA 数值表示修复通过原容差与失败 B4 重放；归因修复仍分开检验

原 FP16 seed 溢出修复在隔离诊断中验证，未部署默认入口。对应
results_fla_range_owner_20261009.json。保存的真实溢出 block3712:3776、
B4×8头、非零初始状态，调用原 FLA 0.4.1 递归参考与 assert_close；
阈值直接读取 SHA35f28bf6 原 test_chunk，不改容差。FP16/BF16各11项全部通过。
参考使用原低精度操作数的FP32拷贝及原BF16种子值，避免参考梯度的FP16输出
本身再溢出；这是实际dtype局部导数极限对照，不是非零有限归因的官方验收。

隔离候选仍基于实际 AppWorld SHA448ef32c，候选SHA33b169b3；
原runner628006b6、FLA核、Q/V/A、白化、PPO、LoRA8/16和每卡B4不变。
原失败batch6/7 round0重放 PID2003658/birth1791482696.9，两rank各一次完整DT，
81.84/82.00秒结束，全部信用有限。24项精确信用向量/源码/配置/日志共2,797,054
字节，本机491,427字节归档，逐SHA校验；另6项原精确输入和官方检查现场共
2,221,325字节本机逐SHA校验。没有将这项成功称为极端归因准确或原PPO NaN修复。

首份诊断launcher缺少plan变量，在创建driver前失败，未发生模型/DT调用；
原失败命令保留。候选以独立diagnostic_range_owner.py显式加载，只绑定
诊断runner的原GDN函数，不遮蔽默认qwen35_gdn_finite导入，退出时还原。
正式训练继续停止，不恢复检查点。

剩余AppWorld查询 PID2032920/birth1791482964.86、物理4/5，只补83个冻结位置，
复用原74和失败调用重放8个，额外20次B4 DT/rank；不重做已完成查询。
实际脚本SHAb6c4c341、候选33b169b3和原source58209daa分别记录；原44项PPO
现场不变。仍须读取该诊断终态并按原交叉单元汇总，不能把部分结果冒充165项完成。
极端负信用的主线仍是联合删除背景分配误差，未用裁剪、倍率或额外训练单删绕过。


## 2026-10-09 原 FLA 输入转换溢出定位；最小数值候选仅准备，未部署

原单背景AppWorld诊断不是OOM。失败batch6/round0的两次原owner有界复现
PID1695831/birth1791479850.34、1740550/birth1791480262.55均已结束。
被动观察先在layer4 conv-SiLU发现非有限返回；精确输入证实其upstream已经含NaN，
因此没有误判SiLU为根因。补查实际逐层FLA回调后，保存全部原FLA操作数：
各端点均有限，原FP16 seed只有一个-Inf，位置[0,3735,4,79]。
原FP32 norm-gate输出有限。PID1793952/birth1791480746.55只补存转换前mo，
两rank分别80.57/81.84秒后按预设采集完成退出；不是完成DT或训练更新。
原BF16 seed值-76800在FP16转换后为-Inf；重算转换与失败输入逐值完全一致。

三次现场分别本机保存26/26/28项源码、配置、精确输入及phase元数据，逐SHA校验。
4.19GB首个SiLU操作数、1.45GB FLA操作数和两rank转换前mo仍原样保留远端，
本机manifest绑定每项路径、字节数及SHA；没有称这些大张量已经异机复制。
对应results_fla_seed_nonfinite_20261009.json，原PPO44项已保存现场不变。

仅复用保存操作数、零模型/DT/optimizer调用，测试原FLA的线性齐次数值表示。
PID1818800/birth1791480963.38的单位幅度缩放虽然全部有限，但产生最高约0.00627
相对L2的二倍线性误差；未接受。PID1840886/birth1791481160.11改用torch.finfo
决定的最小二幂，所有原系数有限；相同输入再半幅/还原的相对L2为2.0e-8至1.8e-5。
31个不需缩放的头全部q/k/v/beta/alpha/g与原值逐值一致；只有1头原输入溢出。
这不是原FLA官方容差通过证据；未修改容差、核数学、Q/V/A、白化或PPO。
第二调用热耗时约0.0695秒，仅该原FLA组合算子，不冒充整DT速度。

静态实际owner检查拦截了一份尚未运行的候选底稿：TextCraft SHAef55ce08缺少
AppWorld的consume_captures释放。该底稿及否决原因保留在rejected-wrong-base，
没有部署、运行或据此重跑。新候选严格基于已取回的实际AppWorld SHA448ef32c，
保留全部生命周期释放；AST检查只有原gdn_finite_pullback函数改变。
候选SHA33b169b3位于native-fla-range-owner-prepared，仅准备、未接受。
FP16转换前按每样本每头、跨时间共同的最小二幂表示cotangent，原FLA所有返回
系数还原，BF16路径保留。没有信用裁剪或依据已输出信用补倍率，没有新增FLA/
模型/DT调用，不进入默认导入/启动路径；仍需对应原dtype官方对照及整体调用验证。

极端误报的集合主线仍是联合背景分配误差，不因这个独立dtype故障替换成仅修NaN。
原作者集合评测不被单操作数或不溢出头的逐值一致替代。两组正式训练保持停止；
最后物理观察诊断driver均退出、8卡空闲，主机可用约1034GB。


## 2026-10-09 AppWorld 单背景诊断非有限退出，失败现场已保留

原诊断 PID1571788/birth1791478672.23 已退出；rank0在batch6/round0的第17次
DT中触发原runner的延迟非有限检查，rank1终态停在batch7/round0，未冒充完成。
已完成74/165位置；原两rank各16次DT完成。该错误不是OOM：失败记录仍有
10.65GB runtime free；最终物理观察确认作业退出。未根据报错末端猜测首个坏算子。
53项精确现场共14,602,845字节、本机归档1,380,458字节，逐项及归档SHA已核验；
包括原完整有符号向量、两rank日志、实际owner源码和配置。对应
results_single_background_appworld_20261009.json为failed_partial_diagnostic。
本批尚未完成11个原稳健误报尾的完整对照，不扩大部分结果。

只用同一原B4失败输入做有界复现：PID1695831/birth1791479850.34、物理4/5，
每rank最多一次DT、零optimizer/rollout/恢复；脚本SHA731134bf…、观察器1358ccbd…，
base commit b47b7133和实际新增脚本SHA分别记录。观察器返回原有限对象，发现
非有限返回才保存精确操作数并停止；不修改公式、Q/V/A、容差或正式入口。
正式TextCraft/AppWorld保持停止，原PPO44项debug保存不变，NaN没有修复声明。


## 2026-10-09 非零 FA 独立计算对照完成，单删除背景集合诊断开始

PPO现场保留不变，原NaN未定位；两组正式任务仍停止，不恢复检查点。
本轮没有部署候选、改Q/V/A、PPO、白化或FA/FLA，也没有重试已否决输出规则。

原FA官方容差回执只覆盖重合端点；真实非零端点旧检查只对比同内核的row/scalar
表示，不能当独立有限公式对照。本轮组合原signed_secant_rules的softmax和matmul
有限接口、原FA因果mask，检查已保存真实B4的全部4行×16头；只作诊断，不新造
官方容差。原操作数SHAed804b5c…、原wrapper3e1d6103…、实际库4f42c391…均绑定。
PID1467448/birth1791477673.83已退出：32.614秒，FP32/FP64各64头，零模型/DT/
反向/优化器调用。Q/K/V最大相对L2误差0.004384/0.003533/0.002726；tau/center
最大绝对误差9.44e-6/2.20e-6。没有因此声称全方法通过、发现根因或信用已修复。
最大torch allocated1.682GB、reserved3.072GB；完整来源/逐头数值见
results_nonzero_finite_fa_20261009.json。没有证据支持据此修改该核。

继续检查联合删除背景与单token删除背景的差别：固定原165个TextCraft查询、原B4
及原producer/trace接口，仅诊断时令每行一个被查询token为EOS，其余ID保持事实。
原联合DT和native单删除值复用；额外DT不进入训练路径、不替换正式信用。
CPU实际165项reference检查通过，未初始化CUDA；它只验证这个输入适配。
基于e1ff6b7d，实际脚本SHA147cf425…及完整launch保存；TextCraft诊断PID1514892/
birth1791478145.06，物理4/5，计划每rank30次B4 DT，沿用1800秒有界诊断预算。
原actor初始化、LoRA8/16、每卡B4及官方算子未改。AppWorld尚未启动此项诊断。
本次源码是显式SHA记录的研究文件，不把基准commit当作它已经被提交的证据。


### 同轮 TextCraft 完成，AppWorld 已启动同一有界诊断

TextCraft两rank各30次原B4 DT、386.80/386.75秒，固定165位置完成。
原18个稳健误报负尾在单删除背景下均不再达到c>2，其中1处仍轻微反号；
这不是“全部准确”或已修复。预测2<c<=10、native c<=1交叉单元中，未查看/已查看
组的背景差中位数分别-1.0131/-0.9580，单背景有限传播残差中位数
-0.000351/+0.012869；10<c<=100的1点独立保留，不混进前组幅度。
原模型端点评分差另列，不把它们混作核超差。结果支持联合背景的分配为这批
极端误报的重要误差来源，不支持给正式信用扣除残差或增加逐source调用。
原累计删除/RISE/MAS没有因本诊断重跑，也没有宣称方法质量通过。
74项精确artifact、9,151,091字节已本机保存，882,603字节归档及各SHA均核验。
结果在results_single_background_textcraft_20261009.json及其绑定的分层analysis。

AppWorld PID1571788/birth1791478672.23，物理4/5，原165位置、同一脚本SHA147cf425…
及原plan SHAe835680d…，计划每rank37次B4 DT；当前刚进入首项调用，未称完成。
原actor和DT实际导入SHA随worker回执记录，不使用TextCraft的路径冒充AppWorld。
正式任务仍停止，本轮无候选部署、无优化器、无rollout、无恢复。

## 2026-10-09 输出规则候选完成并否决，调试数据已保留

候选基于8861da161ad486955fff8c4d73647c3e85f87ef5，实际数学源endpoint_head_seed.py
SHA593b7ebddfff594844c228b67b8dcf9e6b411bed485e248987dda7f41e872014。
PID1257614/birth1791475669.40，物理4/5，两卡约467.71/467.95秒完成；
每卡6次B4 DT、84次原native累计删除前向，零optimizer/rollout/恢复。
没有重跑原FA失败候选或重复native单删除；32轨迹/16状态的原基线直接复用，
本次32条事实评分与原基线逐值相同。原作者函数、输入、原k=20及聚合不变。

实际运行覆盖只在隔离module research_endpoint_head_owner内把seed_with_checks
绑定到上述新数学函数，并替换浅复制runner的answer实例。实际FiniteAnswerOps、
_linear_transpose、原runner、FA/FLA、模型对象、Q/V/A和PPO均保留原owner。
正式导入路径没有改变。两卡实际来源及覆盖写入results_endpoint_head_20261009.json。
首个实际B4的128个target行做FP32/FP64原语诊断：种子最大差约7e-7，端点
残差约5.4e-6；它不是FA/FLA官方容差或整网通过证明，没有新增容差。

候选未显示集合改善：状态等权signed RISE0.37831975→0.37943135，
positive MAS0.63088382→0.63379717（均越低越好）；不称统计显著变差。
原37个预测负尾中已确认的18个误报仍全部为负尾且方向错误，均匀集合中的
2个已确认漏报也仍漏报。主体、负尾、状态和已查看/未查看组分开保存。
因此否决这条具体候选，不扩展AppWorld、不部署、不裁剪补偿、不改Q/V/A。
不据此断言所有输出规则都无效，也不判定某个FA/FLA核为剩余根因。

本次完整DT向量、原基线向量、原始rank/phase日志和源码身份共35项、9,187,111
字节另存本机preserved-artifacts.tar.gz（2,730,139字节），归档及每项SHA均通过。
精确原始字节与Windows JSON重序列化分开记录，不把换行差异当数值差异。
终端观察1791476292确认driver已退出，8卡物理均858MiB、无作业进程，主机
可用约1034.15GB。原PPO debug44项保留不变，NaN仍未定位；正式作业仍停止。
极端负信用尚未修复，下一步继续针对冻结集合支持的有限传播误差研究，
不重新采样开发集、不用单个点的改善替代原作者集合指标。

## 2026-10-09 集合诊断完成，输出有限规则候选尚未测试

AppWorld PID990446/birth1791473090.64 已完成并退出，两卡约866.6/867.5秒，
12次原DT、每卡37次native；165位置DT与旧值逐值一致，native最大重复差1.14e-13。
最大观察CPU bank33.08GiB，额外读出约30.10/27.95秒；未见OOM/非有限。
最终物理观察1791474250确认研究driver退出、GPU空闲，主机可用约1034.3GB。
完整两卡结果、实际导入SHA和分层分析绑定在results_suboperations_20261008.json。
这完成了被动定位，没有修复信用；两组正式作业仍停止，不恢复检查点。

两个任务均显示最后attention、最终norm与log-softmax背景项存在相反的大项；
种子重算及输出线性投影的差异小得多，不能把某个残差直接减掉。
此前坐标分组输出规则草案已在CPU代数审查中否决：保持总和并不保证竞争logit
单调性。未进行该草案的GPU测试，不引入事后投影、倍率或信用裁剪补救。

新的未接受研究仅考察输出层有限log-softmax seed：以两端实际概率分布的KL确定
唯一线段混合权重，使seed仍满足原端点差，同时各类别权重位于实际端点概率之间。
原归一化logarithmic-mean也是有效有限规则，不称其缺失归一化或数值bug。
推导在endpoint-head-derivation.json；只是原因相关假设，尚未测试或部署。
复用实际FiniteAnswerOps，只注入该数学seed；原runner、FA/FLA、native forward、
Q/V/A、PPO、LoRA8/16及B4均不变，一次DT，无额外模型前向。
拟在原TextCraft32轨迹/16状态、相同均匀/负尾集合上做一次有界比较；直接调用原
作者累计删除/RISE/MAS，原基线及native单删使用已保存的同权重同输入结果，
不重试失败FA候选、不重做PPO回放、不看测试集调参。守恒或数值残差不替代质量。

## 2026-10-08 已保存 PPO 现场，继续极端负信用的算子分解

PPO debug 的44项本机现场及其逐文件SHA校验已完成，继续使用下节的
results_preserved_actor_debug_20261008.json；原NaN未定位，不再重复有限回放。
正式TextCraft/AppWorld均停止，无检查点恢复。当前只诊断原冻结的各165位置，
Q/V/A、整批白化、PPO、LoRA8/16、每卡B4及所有核均未改。

新观察器仍调用原decoder/answer/norm owner并返回原对象。按此前各任务、cohort、
状态等权的定位频率，取层18/30/31；不重选token。分别观察norm/mixer/MLP、
两次BF16残差相加，以及最终norm、原seed的独立重算、线性投影、log-softmax。
残差相加与seed重算差异显式保留，不假定它们为零，不扣除任何项作信用修复。
原consume_captures保持启用，没有因观察而保留原MLP宽张量的诊断分支。

TextCraft v1观察器在首个DT及native前向之后，报表遍历误读已格式化的字典，
触发KeyError('matched')；这是本地观察代码错误，不是DT/PPO非有限或OOM。
原源码从发布tar单独保存，失败PID924321/birth1791472490.78已退出。
v2修正该遍历，并在可选格式化前保存原始point；CPU报表检查不冒充方法质量测试。

v2基于f7381669和显式源码SHA发布：inspect_layer_collection为5f946fab78b0，
passive_suboperations为2933cb5f7b13。TextCraft PID951632/birth1791472744.49，
12次DT、每卡30次native配对前向已完成，各165位置DT及native值均与旧记录
逐值相同；约220.7秒，额外读出每卡约3.5秒，最大保留CPU bank约9.58GiB。
这是实际集合的被动诊断，不是新的官方容差、32k容量或候选质量验收。

原稳定误报中的17点/6状态落在预测2<c<=10、对照c<=1同一交叉单元：最后一层
attention mixer的匹配残差中位数-3.4403，输出log-softmax +3.2587，最终norm
-1.2022，输出线性投影-0.1043；大项相互抵消。不能只改输出层或把某项扣掉。
其余尾部交叉单元单独报告，未把主体/尾部、任务和cohort合成原始均值。

AppWorld同一v2观察器在物理GPU4/5运行，PID990446/birth1791473090.64，
远端receipts/credit-layer-suboperations-appworld-20261008-v2；当前尚未完成，
不宣称已经修复极端信用。launch保存实际source路径/SHA与环境，完成后再绑定
两卡实际import和完整集合结果。状态、失败版本、协议、原始逐位置结果及已完成
分析见results_suboperations_20261008.json和current_runtime的latest_suboperations。
没有新PPO回放、rollout、optimizer、正式恢复或核改动。

## 2026-10-08 PPO debug 现场已本机保存，归因研究完成事实端点控制

按用户“保存 debug 信息，随后返回极端负信用准确性修复”的指令，v5 原首轮
30次交互及每卡4次原 optimizer 更新已结束；两卡各32个B4的观察输出与原始
梯度有限，未复现原NaN，未修复或重启正式训练。此次停止来自已记录的诊断
边界，不当作训练失败。原事故中未保存的 old/ref/初始LoRA仍不能补造。

本机 preserved-debug/1791469222 保存44项、1,089,145,135字节，全部逐文件
SHA256校验通过：两卡初始LoRA/RNG、实际DataProto（old/ref、position及mask）、
4次更新前未裁剪梯度及logprob/entropy反向记录、更新后LoRA/RNG、实际源码身份、
配置与日志。直接SCP在126MB文件上触发120秒传输超时；复用同一不可变快照，
用标准库tar/gzip无损压缩后完成复制，没有重新采样或更新。归档580,627,856字节，
SHA1d339eaf33318adff6011bf00b7554209cee484fe576928d9453b93282557f7b。
远端快照目录时间1791469220与采集结束时间1791469222分别记录，不再混用目录身份。
清单与CPU检查见results_preserved_actor_debug_20261008.json；这不是训练检查点恢复。

极端归因已回到原冻结集合，TextCraft/AppWorld各165位置（128均匀抽样、37预测
负尾全数），没有重新挑点。原逐层观察器只增加同一原_token_effect下的控制量
`m*(DT事实hidden-native事实hidden)`，不改变其返回对象或任何信用。observer源
SHA1c05353fffaf883c419b7b43816baccc906a6324b9348a716a9c3bbffda1255d，
基线提交5dd910a7；实际任务source/导入路径/SHA及原VERL20bd331绑定在
results_factual_controls_20261008.json。物理4/5、DT每卡B4、native配对B8、
LoRA8/16不变；TextCraft约223秒、AppWorld约859秒，零rollout/optimizer/恢复。

两项新观察的DT值均165/165与旧观察逐值相同；native TextCraft165/165相同，
AppWorld149/165相同，其余差值最大1.14e-13。这里只报告重复观察差值，未创造
容差或改动FA/FLA。事实端点对齐后的残差仍有相反的大项：在此前跨多次对照
确认的误报负尾中，TextCraft18点/6状态的输出norm前事实差异加权绝对值中位数
0.09657，而对齐后的norm/head残差绝对值中位数1.99890；AppWorld11点/6状态
分别0.25888和8.00621。decoder与head残差互相抵消，不能将其中一项直接扣掉
当修复，也不能据whole-decoder残差判定某个FA/FLA核错误。主体/负尾、任务、
状态和cohort继续分组；该测量未产生通过集合评测的修复，不替代原作者RISE/MAS。

1791470818.72终端观察：两诊断已退出，mx-smi全8卡858MiB/65536MiB、0%、
无GPU进程；主机可用1,034,458,435,584字节，cgroup97,346,322,432字节。
这是结束时资源快照，不是32k峰值容量验收。Q/V/A、原整批白化、PPO及正式
入口未改，两个正式任务保持停止，不恢复检查点。后续工作仍针对有限传播
对单删除效应的估计误差，不重复PPO有限回放，不加端点补偿或信用裁剪。

## 2026-10-08 用户要求保存 debug 现场并返回极端负信用修复

优先级已按用户最新指令写入 PLAN 顶部，Q/V/A、白化、PPO、官方参数及入口不变。
两组正式作业仍停止；GPU4/5只保留已有 v5 首轮诊断，不再新增有限 PPO 回放。
原 NaN 未定位、未部署修复。原事故缺失的 old/ref、位置及初始 LoRA 不能补造。

检查实际落盘发现 v5 trainer 观察并未安装，初始状态文件缺失；复用该进程的
原 worker generic RPC 在首轮 rollout 中补挂，没有重新采样或重启。两个原
worker264116/265959均已保存首更新前 LoRA/RNG。安装 RPC 完成后，客户端写
receipt 时因 Ray ActorDeathCause 无法 JSON 序列化退出；错误和实际部署的
attach 源已保存，不能据客户端失败重复安装。当前源有安装后捕获原 update
并停止诊断的边界；原更新实现、optimizer、credit 不变。修正的本机 receipt
序列化代码尚未重新部署，不冒充已运行版本。

本机 preserved-debug/1791467516 已保存18项、87,730,454字节，逐文件 SHA256
验证通过：两卡初始 LoRA/RNG、有效配置、原 source、诊断源码、阶段日志及
安装错误。清单 preserved-debug-latest.json 明确实际 update 输入和梯度尚未
产生；不能称完整故障复现现场已齐。v5 driver258568/birth1791466135.87，
观察1791467905.60仍在原 rollout 第25/30交互，尚无 actor update 文件。
新观察器会保存实际 DataProto（old/ref/position/masks）、各原 minibatch 的
未裁剪梯度、logprob/entropy及其反向梯度；以实际 sidecar 发布为准。

归因研究已返回冻结集合的330个既有点，无新增模型调用：合并原 native、
逐层 native、重测 native 与同 hidden 的 FP32 head 对照，保留任务/状态/
cohort/预测与对照负尾分组。在37个预测负尾位置中，TextCraft18点/13轨迹/
6状态、AppWorld11点/7轨迹/6状态在DT重算仍属负尾，而所有保存对照均非负。
TextCraft的该37点中另有5点跨全部保存端点仍违反 p_deleted<=1 的必要界。均匀样本
中另有2/128、4/128个漏估负尾。以上不是总体错误率或官方容差验收，也不是
训练退化判断；FP32 head不是全FP32模型。来源及逐点身份见
results_stable_negative_credit_20261008.json。该复核确认仍有待修的估计误差，
未证明某一层为集合主因，不据此部署候选、裁剪信用或改下游Q/V/PPO。

## 2026-10-08 TextCraft NaN：完整原首轮诊断与原输入一致性核查

原故障保存的256行actor输入，与对应原环境轨迹保留的830,939个response ID及
policy mask逐值一致；prompt也一致，无未映射policy token。原owner位置构造与
此前诊断重建的有效位置一致，差别仅右padding；原实际position_ids未保存，
不能将重建扩大为原整个输入完全相同。原startup_options的multi_turn.enable=True，
与trainer更新时写入的开关一致，未发现观测loss mask在该处被关闭。
结果original-actor-artifact-audit.json，CPU观察不构成数值容差验收。

为保留此前缺失的现场，新增一次完整原环境首轮诊断：原TaskRunner/采样/old/ref/
DT/全批白化/PPO不变，原330预算及scheduler不变；仅诊断输出目录、GPU4/5与
完成原step1日志后的停止点不同。记录实际update DataProto全部字段、初始LoRA/
RNG、每个原optimizer minibatch的未裁剪梯度、logprob/entropy及其反向梯度。
不启用detect_anomaly，不在各B4间新增item/CPU数值检查；原norm检查处落盘，
保留native非有限时跳过更新的行为。观察仍有克隆/落盘成本，不宣称完全不扰动。

v4诊断233497/birth1791465895.67因替换worker类造成Ray序列化super绑定错误，
在原init_model前退出，零模型/DT/PPO计算。失败源及日志完整保留，不算原NaN。
v5保留原worker类，通过原execute_with_func_generator安装实例观察；CPU已核对
native worker cloudpickle身份。新driver258568/birth1791466135.87，SHA078df501…，
启动基线78dd0d97；原source2796233e…、actor3a65e173…、core fc2f992b…不变。
启动不代表已复现或已修好，当前正式TextCraft/AppWorld均停止，不恢复检查点。
诊断文件在actor-nonfinite-20261008/v5；最新阶段以current_runtime.json及观察为准。

## 2026-10-08 TextCraft NaN 继续定位：补入一次原 vLLM 交接，四次更新仍有限

新增诊断 driver8327/birth1791463785.66、实际Python8328，仅物理GPU4/5；
启动基线78dd0d97，脚本启动时未提交，SHA ef2cf1f0…单独记录。沿用原source
2796233e…、VERL20bd331和原参数。复用上一诊断的原更新观察方法，新增原
actor_rollout初始化、一次原LoRA同步/生成/休眠；输入为原保存的32条初始prompt。
生成结果仅用于交接诊断，不充当环境轨迹或训练样本。随后原176条DT carrier
与256条actor carrier分别保持原顺序及原优势，执行原四次连续PPO更新。
不启动正式任务、不恢复检查点、不部署修复，不把诊断耗时当正式吞吐。

目前原vLLM交接已完成，前后全部本地LoRA张量SHA相同，dtype/精度标志未变；
FSDP记录的变化仅为原出口train()引起的子层模式变化。CPU只读核对当前
PyTorch2.8.0+metax3.5.3.9的FSDP2 offload已有梯度搬运event等待，尚无缺失
等待的故障证据，不据假设改它。

1791465038.22诊断完成，两rank各32个B4、4次原optimizer step及所观察输出/
梯度全部有限，无原PyTorch异常栈；未复现NaN，因此没有可部署修复。阶段耗时
init109.07、生成24.49、old150.23、ref119.48、DT219.71、update604.50秒；
这是带原异常检测及同步观察的诊断，不是正式吞吐。观察最大进程树PSS56.85GB，
物理单卡37,218MiB；1791465111.50原进程树已退出，GPU释放。此次只含一次原
VERL/vLLM基础采样调用，非原30轮完整环境采样；没有拿32条诊断生成代替任务评测。

补回原TaskRunner日志：故障首轮原汇总ppo_kl=0.288，而上一轮完整诊断按
同口径汇总约-0.00004174，本轮约+0.00004011。这说明原概率变化尚未被重现，
不代表KL已定位为根因。原old/ref、actor position_ids和初始LoRA未保存的限制
保留；本次position_ids通过原helper重建。参数SHA只检查trainable本地shard，
不能扩大成全部frozen base权重校验。原22小时update计时包含首更新
hold，不能当实际PPO计算耗时。详细回执在actor-nonfinite-20261008/v3，最新状态
见current_runtime.json的latest_native_actor_nonfinite_rollout_lifecycle_20261008。
汇总results_rollout_dt_to_native_actor_nonfinite_20261008.json；正式两组继续停止，
未恢复检查点、未改DT/PPO/训练参数/容差。下一步需要在完整原采样→DT→更新
链路保留首个故障的old/ref、位置、初始/更新前LoRA及未裁剪梯度，减少同步观察
对时序的影响；不继续用相同的重放有限结果冒充根因定位或修复。

## 2026-10-08 优先 debug TextCraft PPO NaN：真实 DT 后完整四次原更新有限，根因未定位

按用户最新指令暂停新增极端归因研究，先查独立的原 actor 非有限梯度。
先澄清旧回执：前三次更新有限、第四次未完成是 900 秒诊断上限触发，不是第四次
复现 NaN。原正式两 rank 各记录一次非有限 norm；具体 optimizer minibatch 未保存。
原 VERL 遇到该值清梯度并跳过对应 step，不能称全部四次都失败或全部都健康。

本次源绑定诊断 driver3979707/birth1791461643.35，实际 Python driver3979708；
基线提交89776c46，观察脚本启动时尚未提交、SHA998698ec…单独绑定。仅GPU4/5。
原 source SHA2796233e…、actor3a65e173…、core fc2f992b…及真实 DT producer
2aa5f552…不变。核对固定 VERL20bd331：policy loss、KL、update_policy、
optimizer_step 和 fsdp2_clip_grad_norm_ 五个函数 AST 相同；输出头和调度文件也
与原 source 的已记录 SHA 相同。这是版本/函数身份核查，不是新数值容差验收。

复用原保存的完整176条 DT carrier，经原 compute_dt_token_advantages；随后用原
256条 actor carrier 与原全批白化优势做完整四次连续更新。DT/actor 的行数、顺序、
padding本来不同，分别保持原 artifact，不人为重建配对。首次 CPU 准备误把两者
行数当成相同，断言在 GPU/model/optimizer 初始化前失败；失败源码/日志另存。
修正的是这个诊断断言，没有改变实际训练输入、DT 或 PPO。原 B4、全局optimizer
minibatch64、LoRA8/16、损失、dtype、scheduler 均不改，未替换 actor 保存优势。

原 DT 完成；两卡各32个B4、4次optimizer step全部完成，裁剪前各496个梯度张量
及原生norm均有限。rank0 norm为0.0307812/0.0501252/0.0302992/0.0488105；
rank1为0.0352183/0.0585729/0.0349105/0.0554691（完整精确值以JSON为准）。
DT 前后已记录的训练模式、FSDP状态、精度标志一致。原 PyTorch detect_anomaly
未报错；没有修核、改公式、加倍率/裁剪、扩大容差或部署研究候选。

old/ref/DT/update阶段分别149.94/117.64/218.74/601.88秒，异常检测和同步观察
有额外开销，不作为正式吞吐。进程树观察PSS最大56,442,462,208字节；物理单卡
显存观察最大38,544MiB，无OOM。1791462836.64确认本诊断进程树已退出。
本次保留初始本地LoRA/RNG张量和真实old/ref输入供后续重放，不作为训练检查点。
原native ECC计数读出全8卡为0且启用；这不证明硬件计算完全正确。

本次补齐了旧诊断缺失的 DT→完整四次更新，仍没有复现原 NaN，不能称修复。
历史初始LoRA A和old/ref未保存，且本次没有重放前置vLLM采样；原故障GPU2/3、
本次4/5，这些证据限制明确保留。未重启正式训练、未恢复检查点，目标未完成。
原旁路观察/hold代码另作CPU审查，不因其存在就断言其导致NaN。

结果：experiments/rl/results_dt_to_native_actor_nonfinite_20261008.json。
具体输入、源码、PID/birth、有效配置和原始阶段观察：
research/temporary/rl_upstream_alignment_20260929/actor-nonfinite-20261008/v2。

## 2026-10-08 极端归因：330点参考精度复核与45轨迹状态隔离完成，尚未修复

本轮只分析极端归因，不把它改称旧熵主导退化的原因。冻结集合不变：每任务128个
均匀source、37个原预测尾部source，共330点；主体与尾部、任务与状态分别记录。
新增三次有界原模型诊断，不训练、不采样、不恢复检查点、不改默认入口或PLAN。
VERL owner为20bd331，启动基线提交1a7676c6；新增脚本启动时未提交，实际SHA、
原source SHA、实际导入路径、有效配置、driver PID/birth、完成标记一并绑定。

| 问题 | 本轮证据 | 能支持的判断 |
| --- | --- | --- |
| 输出层舍入能否解释尾部 | 同一原生hidden/weight，FP32投影再舍入为BF16；全330点 | TextCraft主体d差中位数0.00543、最大0.13648；AppWorld为0.11302、1.09253。部分AppWorld判定敏感，不能把旧BF16对照直接称为稳定真值 |
| 先前漏报点是否保留 | 对原先TextCraft2点、AppWorld11点逐身份复核 | TextCraft仍2点；AppWorld在本次native为7点，FP32输出层为4点。这是原漏报子集的复核，不是新的总体发生率 |
| 稳定负尾的实际幅度 | TextCraft同一状态两个冻结位置765/800 | FP32输出层d为−2.44143/−2.53597，A/r为−10.4895/−11.6287，原DT却为正；该处舍入d仅−0.09520/−0.03215，不能解释反号 |
| source位置接错 | 330点的原DT source contraction与所选embedding边界对照 | 全部逐值相同，只排除该边界的错位，不扩大成整条actor接口验收 |
| DT或原观察钩子污染状态 | TextCraft全部45条冻结UID、16状态、12个B4；DT前两次、DT后、原钩子启用及移除 | 所测事实分数与单删d全部逐值相同，记录精度标志未变；在此范围未复现污染，不修改状态恢复模块 |

输出层FP32参考不是完整FP32模型，也不是FA/FLA/VERL官方容差验收；没有添加容差、
倍率、剪裁或残差纠偏。AppWorld实际answer owner SHA1e20956a…与TextCraft
d47333ea…分别保留，不因同名文件而混成同一运行版本。旧TextCraft layer对照与
此次native存在的差异仍未定位；本次状态隔离没有解释它，不能随意归因为batch形状。

三个诊断driver分别3616016/birth1791458234.81、3650124/birth1791458528.17、
3752986/birth1791459490.25，均仅GPU4/5；最大rank测量阶段分别114.79、467.41、
205.26秒，初始化不计入这些阶段数。head诊断每rank分别30/37次原生paired B8，
零DT；状态隔离每rank6次原DT与30次原生paired B8。三项均完成、无optimizer。
head最大单rankallocated分别10,713,222,144与32,251,317,248字节；状态隔离观察
进程树PSS最大46,327,587,840字节，未见OOM。这些不是32k训练容量结论。

远端1791460203.9453628终态观察：8卡均858/65536MiB、0%、无GPU进程。
TextCraft正式仍因原actor非有限梯度停止，AppWorld正式仍停止；没有恢复或重启。
极端归因有限传播的集合主因尚未确定，没有可部署修复。已否决的条件注意力候选
仍不部署；此前RISE/MAS结果保持原判定。本条覆盖下方较早的运行阶段，不改历史。

汇总：experiments/rl/results_extreme_attribution_evidence_20261008.json。
分项：results_native_reference_drift_20261008.json、results_native_reference_repeat_20261008.json、
results_native_head_textcraft_20261008.json、results_native_head_appworld_20261008.json、
results_native_dt_lifecycle_20261008.json。原始身份、分组、分位数、配置和资源观察均保留。

## 2026-10-08 原actor有界诊断部分完成：未复现NaN，不构成修复

仅在研究GPU4/5启动一次原VERL观察作业，driver包装PID3316643/birth1791455383.0，
原source SHA2796233e…、actor SHA3a65e173…；具体观察脚本SHA9dab61c1…，
基于提交36263569但脚本当时未提交，实际文件SHA另行绑定，不冒充正式部署。
原256行输入/优势、全局optimizer minibatch64、每卡microbatch4、LoRA8/16、原损失
及连续optimizer更新不变。只增加PyTorch原detect_anomaly和返回原梯度的被动记录，
零DT、采样、检查点恢复。两rank各完成3/4次optimizer step，step前梯度元素均有限；
第四次仅启动第29/32个microbatch，没有整批完成或正式健康更新结论。

原900秒诊断上限触发SIGTERM；1791456457.52观察本树进程0、两卡各858MiB。
观察到全树PSS最大45,893,045,248字节，物理单卡显存最大30,632MiB，未见OOM。
old/ref logprob阶段150.118/116.595秒；异常检测与同步记录有额外开销，不能用作正式
训练吞吐。原进程的初始LoRA A与old/ref输出未保存，本次fresh actor没有经历前置DT，
所以不是历史逐位重放；不能据前三次有限便宣称NaN已修复或完整DT→PPO接线正确。
TextCraft正式作业继续保持停止；不为扩大计数自动重跑相同整批诊断。
结果：experiments/rl/results_native_actor_nonfinite_20261008.json。

另用现有两任务各165个配对点，分任务/主体抽样/尾部普查/预测与对照区间，记录DT
cached-prefix事实target分数与native完整前向事实分数的漂移。两侧已有相同 scoped
FA/FLA精度与target读出，但调用形状/前缀重用不同；不把漂移称为核超差或归因主因，
不减残差、不加纠偏、不改容差。新增模型调用0。
结果：experiments/rl/results_extreme_attribution_endpoint_drift_20261008.json。

## 2026-10-08 固定集合完成：条件注意力候选被否决，未部署

TextCraft v3已完成32条主评测轨迹/16个初始状态、128个均匀source与37个原预测尾部
单删配对。每rank12次DT（原/候选各6）、186次native paired B8前向，
阶段耗时983.83/984.13秒；零optimizer/backward/rollout/检查点恢复。
执行worker SHA0fe976d0…，基准source SHA2796233e…，基线提交7de13a34，
运行时尚未提交的具体文件SHA及实际导入路径已绑定结果回执，不能冒充其已部署。

必须纠正中途汇报：原作者函数SHA583f4b7d…返回AUC，README明确RISE↓/MAS↓，
越低越好。此次同一fresh原模型/参考下，状态等权signed RISE 0.378320→0.402637，
positive MAS 0.630884→0.667248，分别12/16、11/16状态变差，不能报收益。
128个均匀source中，两处原DT漏估的native c>2仍未修好；不能用个别预测尾部
幅度下降盖过整体质量结果。候选淘汰、不部署、不跑测试侧、不围绕失败候选调参。

已立即停止刚启动的AppWorld候选driver3168041/birth1791453938.36，
1791454041.96确认remaining_non_zombie=[]，两rank完成DT/native/optimizer均0。
只停止本任务研究进程，正式入口和其它用户进程未改；物理GPU4/5释放。
未将候选局部FA导数容差通过宣传为整网归因验收。

旧保存基线与本次fresh B4重算有差异，回执保留分组幅度；其来源尚未定位，
只使用本次相同原模型/候选/native配对判定，不混旧尾部计数、不声称官方容差通过。
当前“极端归因的集合主因”仍未定位；这一阶段的有效结果是淘汰候选。
TextCraft原正式任务因原actor非有限梯度已停止，详见下一条，不能报告仍健康训练。

完整结果：experiments/rl/results_conditional_attention_collection_20261008.json；
原始轨迹/状态分数、尾部分组、身份、命令、owner SHA、PID/birth均保留。
本条覆盖下方v3仍待测/候选收益/作业仍运行的历史状态，不更改PLAN的方法公式。


## 2026-10-08 恢复后出现原actor非有限梯度：停止受影响TextCraft，独立保留现场

原rank0/1分别记录一次grad_norm=nan。已核对原dp_actor._optimizer_step：
非有限时zero_grad并跳过该minibatch的optimizer.step；不能声称已完成健康更新，
也不能因一个告警就声称其余minibatch均未更新。采样、DT、原actor的场景分开记录。
保存的两rank原pre-update输入、DT Q/V/A、白化advantages、mask均有限，
白化advantages最小值分别−140.341/−47.240。这只排除了已保存输入直接含NaN，
没有定位梯度生成位置，不把它归因于当前归因候选或旧熵主导问题。

用既有resume_at_native_checkpoint.py --stop-only --stop-now停止原2833207树；
远端1791451908.09确认remaining_non_zombie=[]，无checkpoint marker。
不创建、导出、恢复检查点。原worker日志、py-spy、原actor源码、两rank输入快照保留。
active-training/formal-training/active-source状态已对齐，当前正式TextCraft停止。
现场与停止回执：receipts/textcraft-first-update-nonfinite-20261008-v1。
本条覆盖下方“已解除hold进入更新”的先前状态，不宣称训练修复。

隔离归因集合测试v1因原作者ft_ifr_improve搜索路径遗漏退出，零DT/更新；
v2完成原DT后，候选wrapper的layout类身份不同而断言退出，零策略更新。
已分别补回已有官方模块路径、依赖注入原runner的同一RightPaddedLengths类，
未抄写layout、改变张量或跳过断言。失败源码/启动/SHA/错误栈均保留，
并非原训练的数值故障。v3 GPU4/5 driver2979183/birth1791452179.28，
workerSHA0fe976d0…，原/候选整网质量仍待实测，不进入正式入口。


## 2026-10-08 最新用户安排：TextCraft原作业解除hold；4/5继续极端归因调试

原driver2833207/birth1791370325.16、workers2838967/2840776创建时间逐一核对，
source.json SHA2796233e…不变。远端1791451326.23写入原hold接口的两枚release文件，
两worker均于1791451326.96记录hold_released_unix并进入原pending PPO更新。
未重启、未恢复检查点、未修改参数、未部署研究候选。正式GPU2/3，研究GPU4/5。
此条覆盖下方较早“TextCraft继续hold”的运行安排；首次完整更新另以原日志核实。

极端归因量级补充见results_extreme_attribution_magnitude_20261008.json：
均匀抽样漏报的单删负尾TextCraft2/128（1状态）、AppWorld11/128（8状态），
native A/r范围分别−10.51～−8.53、−7.70～−1.09，原DT对应均为正值。
原开发预测尾部37项中跨到native非负的分别24/14项；分组、身份和分位数保留，
不混成总体均值，不把原模型对照称为真实世界因果或官方容差。

条件注意力组合已过原FA重合导数断言与256非零局部对照；完整owner只在隔离
研究目录加载，不进入默认PYTHONPATH。固定TextCraft开发集合比较已提交，
driver2908360/birth1791451554.49、原source SHA2796233e…、两卡4/5。
原模型、GDN、真实Y、原作者累计删除/RISE/MAS、B4/LoRA8/16沿用。
每rank计划12次DT（原/候选各6）及186次native paired前向，1800秒有界阶段。
启动时尚无完整模型/集合结果，不能称修复；AppWorld正式作业仍未重启。
原始源码SHA与命令见conditional-attention-owner-v1/textcraft-launch.json。


## 2026-10-08 用户澄清：当前诊断极端归因，历史熵主导问题已修

本轮目的是解释极端 token 归因的来源、可信度及有限传播误差，当前真实联合
动作 target 版本的训练效果尚未判断。历史熵损失主导的问题已定位并修复，
不得将本轮集合或局部候选结果写成对该历史问题的原因调查。
近期已完成集合分层单删除、概率边界、整decoder残差及原PG影响测量；
GDN30 联合背景与单删除背景的差额仅解释被选中的一个局部反号，未建立
集合主因。最近数次新增的是条件注意力候选的接口/数值验证，并没有新增
“极端归因的集合主因已定位”的证据。完整候选集合质量仍未测。
最新局部组合回执见 results_conditional_attention_composition_20261008.json；
其原 FA 断言通过和非零局部误差诊断不冒充整网归因修复。
TextCraft hold、AppWorld 不重启、不恢复检查点的安排保持。


## 2026-10-08 条件注意力隔离 owner 候选：原 FA 导数断言通过，未部署

候选库 SHA78678504b0c87e26f9a242faff2b077b3c7174c9c85a9655a9e1d2ab58569271；
生成 CUDA SHAd7f6bd57…、包装器 SHA544d1795…、行统计头文件 SHAbd098ce0…。
原 owner CUDA SHA9ebcef18…、原包装器 SHA3e1d6103…；实际路径、完整 SHA、
编译命令和检查回执绑定在 results_conditional_attention_owner_20261008.json。
代码由3043a0bd基线上的差异补丁/生成器确定；此记录的提交不代表生产部署。
原导出及 conditional=False 路径保持，候选只由显式研究导出/库路径调用；
原 public FA、模型、训练器、Q/V/A、白化、PPO、LoRA 和正式入口未改。

先前候选 NaN 是新增实现错用 CUDA 相邻 lane 行归约：当前 MetaX 原
quad_allreduce_ 使用 Allreduce<64> xor48/32/16，候选原用 xor2/1 混了行。
依据原 utils.h/softmax.h 改回其行映射；两次编译文本错误和失败数值回执
均保留，不加输出纠偏/裁剪、不改原参考或容差。这不证明原训练存在同一错误。

真实已存 Q[4,527,16,256]、K/V[4,911,4,256]，BF16 输入及原 FP32 上游，
按原有限核 BF16 上游口径验证；关闭候选的 dq/dk/dv/tau/center 五项逐值
等于原库。开启后重合端点导数的 dq/dk/dv 满足固定 FA v2.6.3 原断言，
原 out 为2倍低精度基线误差、原梯度为3倍；未自行替换成统一2倍。
全返回张量未见非有限值。该检查只支持导数边界，不能称为非零有限归因、
整网质量、32k容量或训练验收；完整 runner/gate 尚未接入，无候选集合评分。

本次成功编译61.03秒，采样进程树PSS约0.953GiB；单算子检查15.59秒，
采样PSS约5.845GiB，allocator allocated/reserved高水1.058/1.340GiB；
这些不是模型或32k运行峰值。空闲物理GPU4仅执行这个小算子检查，无模型
前向、optimizer、rollout、恢复或正式重启。TextCraft原PID2833207、
birth1791370325.16仍hold且release false/false；AppWorld未由本次重启。

集合结论仍按冻结任务/状态/轨迹/查看身份/原minibatch、均匀抽样与尾部
全数分开，保留DT/native概率比交叉区间。原累计删除/RISE/MAS是质量主指标；
局部原算子容差通过不替代它们。未选择生产修法，也未恢复单点GDN主因推断。
候选的非零局部语义与完整组合属于下一研究阶段；不得拿本回执先行部署。


## 2026-10-08 回查单点推断与分层结论；条件注意力仅CPU可行性研究

已撤回从一个GDN异常token决定整体修法的推断；原冻结集合和测试侧身份不变。
均匀抽样的native c>2而DT未预测到的尾部，TextCraft为2/128位置、1初始状态；
AppWorld为11/128位置、8状态。完整开发侧预测尾部37项分别有24/14项
在native单删对照落到c<=1，涉及6/7状态。均匀抽样与尾部全数的分母分开，
按DT/native交叉区间保留条件分位数和状态分布，不混成总体原始优势/误差均值。
有限集合不证明无界尾部或总体矩存在；空区间也不证明总体没有该尾部。
原作者累计删除、RISE/MAS仍为方法主指标，本次未声称候选改善。

TextCraft四个原optimizer minibatch的预测尾部误差梯度/原完整PG梯度范数
为20.54%、2.01%、0.0091%、3.37%；几何加入对应误差后的角度
11.84、1.15、0.0052、1.93度。沿用原白化尺度和原损失分母，没有生成修正更新，
不把均匀抽样误差外推到未测位置。AppWorld实际梯度仍未测。
整decoder残差含MLP/norm/gate等并存在相反项，不能称作FA/FLA核误差或因果份额，
不能用这些数字宣布GDN或完整注意力已是历史退化的主因。

条件注意力仅完成完整局部Q/K/V/对角/gate代数与当前owner工作量/生命周期核对；
局部恒等式不证明整网逐source准确。原生FA已经计算LSE，当前被动capture未保留；
复用该原输出仅属有源码依据、尚未实现/验证的机会。相对当前含成对LSE重放的
逻辑收缩计数，拟议规则约0.999倍；相对潜在原生LSE复用基线则约1.378/1.392倍。
不能将前一比较称为无代价精度改善。额外归约/SFU/流量与物理显存仍需实测；
此前62.63GiB容量来自独立memory候选，不能为本数学候选背书。
未实现新核、未选择生产修法、未改变原FA/FLA参考/实际dtype/断言。

本次新工作全为CPU分析，零model/DT/GPU/gradient/update/rollout/restore。
最新只读源码核对中TextCraft原PID2833207/birth1791370325.16同出生且release
仍false/false；AppWorld未由本次重启。计划、Q/V/A、原PPO和正式入口未改。
记录以results_credit_collection_scope_review_20261008.json及current_runtime
对应字段绑定源码/SHA；提交本记录不代表候选已部署或已通过官方容差。



## 2026-10-08 分组残差与完整开发集概率边界：CPU研究完成，未选择修法

只分析原冻结集合，未新增GPU/model/DT/gradient/update/rollout/restore。
TextCraft原PID2833207/birth1791370325.16再次核对同出生，release仍false/false；
AppWorld未恢复，正式路径及Q/V/A、self-target、observation mask、白化和PPO未改。
实际decoder、有限FA库、公共FA接口的路径/SHA与原测量绑定一致；本次研究源码
按SHA绑定，记录提交不代表重新部署。原GPU诊断执行源码仍2951f70e。

逐层项按模型原配置分成24个线性注意力整decoder、8个完整注意力整decoder、
最终norm+head，仍保留任务、cohort、DT/native交叉区间、状态及查看身份。
这些是包含MLP/norm/残差等的整层，不称作FA/FLA核误差，不按最大项选主因。
两个任务的完整注意力层与最终norm/head均可出现大幅相反项，GDN单点结论
不能上升为共同修法；分组只描述，不是组件干预。原累计删除/RISE/MAS仍为主指标。

完整开发侧已存capture：TextCraft85条、121248个source位置；
AppWorld124条中122条已存、204865个位置，2条缺失不补0。
TextCraft推算删除后logp>0的位置229个/14状态：223/22263位于1<c<=2，
6/35位于2<c<=10；其他区间0。AppWorld已存位置未出现此越界。
各区间、状态、查看身份单列；不能解释两任务共同退化，也不把有限集合
上观察的频率称总体发生率。正logp差额完整保留，不设裁剪、门禁或新容差。
已有复测包含上述6个较大负信用位置：fresh DT有5个仍越界、1个近边界消失；
另1个均匀样本的轻微越界也消失。原actor位置/系数与批形漂移分别保留，
不能把系数大小或平方质量当作未测过的单点梯度。

条件注意力仅推导内部完整Q/K/V、对角交互与gate要求，代数恒等式已核验。
公共FA源码确认bottom-right mask与dropout0下可返回LSE而不请求概率大矩阵，
但未作新GPU调用、packed/suffix映射或官方容差验收。B4/32k单个BF16注意力
输出为1GiB，FP32全注意力矩阵为256GiB，禁止构造；完整tile成本、活跃张量
与整网组合行为仍未实现，因此没有新候选、质量改进或修复成功的宣称。
不得只改KV而留下query/gate背景，或把条件效应缩放回联合端点差。

完整来源、分组、旧/新绑定和执行范围见
results_credit_background_research_20261008.json及current_runtime对应字段。
独立测试侧未调参/评分，不自动扩样或追加GPU；不恢复单点GDN候选，
不增加top-k重算/信用纠偏，不变更官方容差。

## 2026-10-08 冻结集合逐层定位实测记录：两任务全部完成、正式训练仍hold

执行源码2951f70e，脚本SHA8e9b1c8b…，固定输入SHAe835680d…。
未改正式DT/FA/FLA、PPO或训练配置；不传observer，不替换任何有限规则/返回张量。
TextCraft PID1576668/birth1791438687.36已正常完成并退出；AppWorld
PID1622404/birth1791439120.18接着在已空出的物理GPU4/5启动同一协议。
TextCraft原训练PID2833207/birth1791370325.16仍hold、release false/false。
本次均零optimizer/scheduler/rollout/restore；不是恢复或启动正式训练。

TextCraft45条轨迹，原均匀128source及预测尾部全数37source全部返回，
共165个不同位置；每rank6个B4 DT、30个paired8 native forward。
worker实测220.01/220.13秒，包含初始化的launch到完成约308秒。
阶段采样PSS最大22.34/24.11GiB，实际单批bank最大14.72/16.23GiB，
batch结束全部清空；没有保留所有查询的整网hidden states。
这不是连续峰值或32k容量验收；最新TextCraft只读快照里的物理GPU已属于
随后AppWorld诊断，不能误记作TextCraft显存。

全部165位置的输入层乘积与本次fresh DT d观测差为0；仅验证本次
层轴/source映射，不冒充官方核容差。新旧DT/native批形差与native/DT事实
端点差分别记录，没有把这些差额补回或缩放训练信用。

逐层项有明显相消，不能按“绝对值最大层”选择修法。例如均匀样本的
原预测/原native都c<=1交叉单元有95点、16状态，fresh |d误差|中位数
0.04267，而最终norm+head项绝对值中位数3.1710，其他层有相反项。
预测c在(2,10]、原native c<=1的单元23点/6状态，fresh |d误差|中位数
1.04443；原native c在(1,2]的单元11点/7状态为0.95128。
两漏估native负尾仍来自同一状态，fresh差额为2.5503/2.8022；
不把它们合成两个独立样本。原始层值、跨层相消与事实漂移全部保留。
这些量没有证明GDN、输出层或FA为整体主因，也不替代原累计删除/RISE/MAS。

AppWorld48条轨迹，均匀128source及预测尾部37source全部返回，共165位置。
每rank6个B4 DT、37个paired8 native forward，worker实测846.70/847.24秒；
launch至完成约920秒。原completed标记及两rank complete均已核对，
观察1791440122.734确认同出生driver已退出、GPU4/5已释放。
阶段采样PSS最大47.63/53.86GiB、单批bank最大39.09/46.05GiB，
批后释放；无OOM，无backward/optimizer/scheduler/rollout/restore。
原训练仍hold、release false/false；未新增GDN候选或信用纠偏。

AppWorld原均匀样本漏估的native c>2有11位置、8状态，独立保留；
预测c在(2,10]、原native c<=1的12位置/6状态，fresh |d误差|中位数3.63515。
各分组保留新旧批形漂移和native/DT事实端点差，不混算重尾信用均值。
这仍是条件描述，不能证明总体期望存在，也不能把最大层项当作GDN主因。

完整命令、导入路径/SHA、source、本次两rank操作计数及原记录见
results_credit_layer_localization_20261008.json与current_runtime对应条目。
两任务本次测量均已终止；不自动扩样、重跑或追加GPU候选。

## 2026-10-08 集合逐层定位准备完成；不是GDN候选或正式训练

沿用同一冻结查询，TextCraft45条/AppWorld48条轨迹，均包含原32条均匀
样本和完整预测尾部的额外轨迹；每任务128个均匀source、37个尾部source。
CPU调用各任务实际import的原DirectActionTargetReadout._prepare_row，
全部selected IDs、target offsets、suffix/prior/target掩码与保存值相同，
0.338/0.552秒，CUDA未初始化。未重新抽样、未读取独立测试侧。

复用原readout的长度顺序按B4测量，每任务12个原DT调用；各rank6个，
随后原native paired8单删除前向。为匹配FSDP调用数，两卡按相邻批次
最大查询次数同步，TextCraft30/AppWorld37个native forward/rank；
耗尽的行只是显式identity控制，不额外计作样本。不是逐token重跑DT。

只旁观原decoder_finite_pullback返回系数及原_token_effect的顶层输入，
返回对象保持不变；不传observer参数，保留原prefix/capture/offload设置。
逐层单删状态与联合有限系数的乘积用于分解差额；整decoder残差不冒充
GDN-only或FA/FLA核误差。头部/最终norm另列，事实端点漂移及批形变化另记，
不补偿、不更改训练信用。主体、预测尾部、漏估尾部和状态/查看身份仍分开，
原作者累计删除/RISE/MAS不被替代。

只保留一个B4的CPU系数及事实端点bank，不保存每个查询的整网隐藏状态。
按实际最长6013/17994、H4096、33边界计算的保守FP64+BF16 bank上界
30.28/90.61GiB每rank，实际可更少；这不是实测峰值。已有torch虚拟allocated
历史值甚至超过卡容量，不把它当物理VRAM；新诊断必须读mx-smi与phase/PSS。
脚本已准备、语法检查，尚未运行GPU。worker墙钟预算1800秒，不自动重试，
不启动候选、更新或恢复。源码与CPU回执见current_runtime对应准备记录。

## 2026-10-08 分组误差梯度四组完成；不据首组锁定GDN或全方法主因

同一PID1202123/birth1791435098.07完成原13pass/rank诊断，normal completed.json
返回，native elapsed1896.983秒，最终观察1791437177.364确认原进程退出、诊断
worker清理、GPU4/5各859MiB/0%，诊断树PSS为0。TextCraft原PID仍同出生、
两release仍false，AppWorld未重启。本次optimizer/scheduler/DT/rollout/恢复均0。
实际执行代码仍7f948141、脚本SHAa8b25a4c…；后续记录/分析提交不是新部署。

四个原global64/local32、每卡B4在同一未更新基础LoRA上分别测量；每组496个
参数分片逐值未变，两rank梯度统计一致，原clipfrac/ppo_kl/lower_clipfrac均0，
梯度有限。此处是裁剪前原loss梯度诊断，不是FA/FLA或算法容差的新标准。

预测尾部完整37位置按原minibatch拆为19/10/3/5点：
误差梯度范数/完整PG范数分别20.5433%、2.01465%、0.00911488%、3.36630%；
native-minus-DT差分加回完整PG后的几何夹角11.8410、1.15387、0.005222、
1.92805度。沿原方向投影分别-5.17580%、-0.101176%、+0.000128067%、
-0.00157944%；不能据首组代表四组，更不能据此定位误差产生于GDN。

均匀抽样中对照有界126位置为32/36/20/38点，所抽位置误差范数比
0.266616%、0.750072%、0.154655%、0.327295%，夹角0.152757、0.429718、
0.0885484、0.187524度。没有逆抽样权重，不能外推全主体梯度误差很小。
另两个均匀抽样漏估native负尾同属一个此前未查看的状态/轨迹、均在末组：
误差范数比5.95834%、加回夹角3.40327度。其DT raw信用原为正；
不能只按DT大负输出筛选问题，也不能将这两个位置当两个独立状态。

任务、状态、已查看身份、预测尾部与对照漏估尾部继续分开；不合并原始
重尾均值，不从有限样本宣称总体矩存在性。差分保持原整批白化尺度，
没有重白化或覆盖训练值。所测向量和仅是选中位置的几何量，不是全误差
估计/已纠正更新。原作者累计删除、RISE/MAS仍是另一项方法质量证据，
不被单删除/本次梯度测量替代；本数据也不是旧辅助标签退化窗口。

已只读核对后续观察接口的实际import路径、resolve和SHA5f14bb3c…，
与source.json记录一致。先前误拼的额外嵌套源码c7fc969f…明确保留为
未导入的初步检查，不作为运行版本。原observer会关闭prefix复用与mixer
offload并增加capture复制；尚未安装callback、没有新增DT查询或部署候选。
保存的native批次只含最终source信用和层汇总，不能据此离线恢复逐层单删误差。

完整原位置、源SHA、配对几何与资源来源见current_runtime的
latest_credit_collection_error_gradients及results_credit_collection_error_gradients_20261008.json。
GDN新候选仍暂停，正式训练不放行；需要集合级传播定位才能选择对应修法。

## 2026-10-08 原PPO系数误差梯度诊断启动；不是正式训练或GDN修复

代码7f948141、脚本SHAa8b25a4c…，PID1202123/birth1791435098.07，
物理GPU4/5，目录credit-collection-error-gradients-20261008-v1。
复用原固定权重、actor/core/worker与观察器；原B4、LoRA8/16、掩码、
分母、累积、裁剪调用不变。参数/optimizer/scheduler不写，DT/rollout/恢复0。

原完整PG与三种native-minus-DT系数差分开反向：预测尾部37点全数、
均匀主体126点、均匀抽样漏估native尾部2点。保持原整批白化尺度；
不重白化、不覆盖训练值，不加逆抽样权重，不从抽样梯度外推全主体。
原四个global64/local32按原行序；首三组无漏估尾部，跳过结构上空的分量，
不把未查询的零当数值验收。各组3/3/3/4，共每卡13个native pass，预算2700秒。
这是误差梯度测量，与先前现有尾部的贡献梯度不同；尚无完成或GDN主因结论。

准备时诊断函数抽取引入NameError(uids)，在CPU、加载模型前拦住；
已补回UID，保留失败源/命令/回执，未影响原训练版本。初次准备可见GPU
导致CUDA上下文初始化，CPU检查已用不可见配置，未修改官方导入或模型。
另外直接检查原DataProto.select_idxs源码确认它共享meta_info，薄诊断接口
改用原select(deepcopy=True)保留各组元数据；真实四组CPU检查确认末组
两个漏估尾部未被前三组过滤丢失。两条真实DataProto准备路径均通过，
完整37位置仍与旧回执一致，未以映射测试冒充PPO/归因/核容差通过。

精确launch/PID创建时刻、源码SHA、两条CPU回执及失败范围见current_runtime。
TextCraft原两release仍false、AppWorld不重启；完整反向/有限值与资源尚待原日志。
用read_collection_gradients.py --error-gradients读取同一PID，不重启/重复提交。
用analyze_collection_error_gradients.py分别分析三组原loss梯度，不合并尾部均值。

## 2026-10-08 冻结集合信用误差已绑定原训练系数；只读CPU、不放行训练

本次诊断代码发布3c5157f3；实际执行时为ec0a0e69上的未提交诊断补丁，
已逐文件核对发布Git内容与远端执行SHA一致，不冒充原训练已验证版本。
正式DT/FA/FLA、PPO/worker、LoRA8/16、每卡B4与原数据均未修改/部署。
远端同步CPU助手在credit-collection-coefficients-20261008-v1完成，
CPU计算1.296秒、进程峰值RSS1481875456字节、CUDA未初始化；原训练两release
仍false。此同步短进程未留PID，不声称是新的存活/正式作业。

复用既有source→原response→retained actor位置绑定，无decode/encode。
抽出共享绑定函数后，完整37个旧尾部映射与此前回执逐值一致，另外128个
冻结均匀位置均绑定实际token/源掩码/保留位置。原FP32奖励组合在165个
位置复算与保存raw A的观测最大差为0；这仅检查组合与映射，不证明DT准确。
原VERL masked_mean/masked_var提供原批次矩：mean0.0391507298、
var0.0265237167、scale6.140203，未按子集重白化。差值乘固定原尺度只作
分析；不是完整native重白化结果，更不是训练中覆盖信用。

原预测尾部24点raw反号，在固定原矩下20点actor系数反号；不能把两种
计数混同。均匀样本的95个DT/native均为正raw信用的位置中31个跨原均值，
这是已接受中心化下的系数比较，不据此宣称白化接口有错。
两个漏估native负尾来自同一状态/同一轨迹，固定尺度系数差-52.5509/-65.9024；
保留其均匀抽样身份，不据两点外推整批主体误差或梯度占比。
主体交叉单元、漏估尾部、预测尾部全数按原诊断区间/状态/原minibatch分开，
无合并raw均值、总体矩结论或新数值容差。

已有原模型端点允许额外检查必要概率边界：logp_deleted=logp_factual-d应<=0。
TextCraft均匀1/128、预测尾部6/37（5状态）推出正logp；尾部正差为
0.00357/0.21818/0.22333/0.43342/0.75694/2.08115。AppWorld两集合均0。
保留所有差额；小差额未用既有数值参考排除舍入，不把0阈值冒充FA/FLA
容差，也不把不越界当反事实准确。不以边界裁剪/修正DT，不据此定位GDN。

原尾部贡献梯度与“native单删系数差的梯度”继续区分；后者本轮未测。
共享原PG observer已有必要接口，不新写PPO；没有新增GDN候选或模型查询。
AppWorld无完整pre-update，未伪造其训练系数/梯度结果。来源、各交叉单元、
实际dtype/owner SHA、资源与限制见results_credit_coefficient_errors_20261008.json。

## 2026-10-08 四个原minibatch的集合尾部梯度测量完成；不据此认定GDN主因

代码b7bc74bd、PID768614/birth1791430928.98的GPU4/5诊断已退出释放。
四次原actor-group调用均返回、八个rank回执均有参数逐值未变；每rank12个
原PG反向pass，完整global256按原四个global64/local32分组、实际B4，
LoRA8/16、原PPO/core/worker、掩码、分母及保存的整批白化系数保留。
没有优化器/调度器写入、DT/rollout/恢复、新GDN候选或正式训练放行。

冻结开发37个预测c>2位置（7状态组）在四个minibatch为19/10/3/5个。
其当前PG分量在完整PG方向上的投影依次6.5325%、0.1114%、
-0.0001623%、-0.0002868%；完整向量减去该分量的夹角为
15.393/1.160/0.00703/1.981度。这是现有系数的实际反向贡献，不是与精确
信用之间的误差梯度、不是范数占比当加法占比，也不是重新白化/更新后的效果。
其中24个raw native单删反号位置的投影为1.7894%、0.04995%、
0.000000875%、0.02146%，分别保留，不跨minibatch混成总体均值。
本数据仅限当前action-target首批/同一诊断基础LoRA初始化，不证明旧训练
退化主因；原跨run没有保存LoRA_A初始哈希，不能将不同初始化的梯度范数差
说成官方容差失败或通过。同run所有分量参数不变。

原作者曲线/RISE/MAS、均匀主体及漏估native尾部继续各自按冻结集合报告，
本轮没有测它们的误差梯度。预测尾部的梯度贡献跨batch高度不均，
不能据该量宣称整个DT有害/无害，不能直接归因于GDN。GDN新候选仍暂停，
不回到极值筛选后覆盖单删输出的路线，不因本次结果重抽开发/测试集。

重用原保存系数与此前原owner白化回执计算当前有限batch中心化平方和：
35个c(2,10]位置贡献1.0317%、2个c(10,100]贡献3.7942%；24点raw反号
子集为1.3409%。此量仅解释现有白化统计，不是梯度占比或总体矩。
额外CPU脚本曾不必要地把2线程重算白化的逐值相等设为断言，未通过；它没有
改变训练，也不构成原owner缺陷证据。该重算不再重复，实际统计复用已经保存
且此前验证过的原系数及全batch标量，保留失败命令来源，未加数值纠偏。

本次原native流程1817.057秒；1800秒自定预算检查在最后原调用返回之后触发，
所以保留TimeoutError与缺失completed.json事实。全部12pass测量已齐，
不把预算退出伪装成正常退出，不据此重跑，也不当作PPO/数值错误或goal阻塞。
1791432895最终观察GPU4/5空闲、host available907217387520B；TextCraft原
PID/birth未变、两release仍false，AppWorld未重启。资源是阶段观察，不是连续峰值。
完整输入/导入路径SHA、有效配置、phase、各rank梯度、原dtype、预算退出与
解释范围见results_credit_collection_gradients_20261008.json/current_runtime.json；
原训练数值版本和方法未改，新的提交仅记录本诊断与分析，不是训练部署/验收。

## 2026-10-08 集合尾部原PPO梯度诊断已启动；训练仍hold

实际代码b7bc74bd、PID768614/birth1791430928.98，物理GPU4/5。原owner双卡
初始化完成，1791431089观察正在对原完整256行计算old logprob；尚无完成
minibatch，不把准备/启动称梯度验收。37点与24点子集、四个global64/local32、
每卡B4、LoRA8/16、原loss/反向/分母/白化均按准备回执；预算12次原PG反向
pass/rank、1800秒。没有native信用替换、新GDN候选、DT或采样/优化器写入。
旧prepare中fc1dd9e7仅是未提交patch时的父提交；本次实际部署b7bc74bd包含
新observer6ed5cd29和driver cfe1cc2f。准备输入和源哈希不变，启动提交另记，
没有为提交号变化重做模型计算。实际源及PID见collection-gradient-launch.json。
当时GPU4/5物理约21.46/18.64GiB，host available840140083200B；
诊断进程树PSS43687496704B；这只是old-logprob阶段观察，不是连续峰值。
TextCraft原PID/birth未变、两release仍false；AppWorld未重启。

## 2026-10-08 集合尾部的原PPO梯度诊断 prepared-only

冻结TextCraft开发集合37个c>2位置、其中native单删d>=0的24个位置，全部通过
原token ID、native suffix位置与actor retained_response_positions绑定，无缺失/
重复；37个位置在四个原global64/local32 optimizer minibatch中为19/10/3/5个。
新诊断只将原整批白化系数做只读视图，原分母、掩码、B4、loss与反向保留。
每个minibatch测原完整PG、37点对应贡献、24点对应贡献；不重新白化子集，
不将native单删回填训练，不把稀疏主体/漏估尾部与全数尾部相加成总体均值。
四个minibatch均在同一未更新基础LoRA上测，不冒称连续四次真实更新回放。

observe_native_optimizer_minibatch.py增加默认不启用的named coefficient views，
继续调用原VERL update_policy/loss/backward/reducer；默认GRPO诊断入口保留。
prepared observer SHA6ed5cd29、driver cfe1cc2f、原梯度统计94219328、endpoint
8a72887a；原正式actor3a65e173/core fc2f992b及训练入口未改。
本次准备时HEAD fc1dd9e7是父提交，诊断patch当时未提交；源以记录的实际SHA为准，
不能把父提交称为包含新代码的部署版本。启动后另记实际提交与PID/birth。
原owner导入初始化了CUDA，但此准备未创建模型、前向、反向或更新，不能称为
纯CPU零CUDA检查。GPU4/5准备前空闲，TextCraft仍hold、AppWorld不重启。
AppWorld无完整pre-update，不能推造其PPO梯度。完整准备回执为
credit-research-20261008/v1/collection-gradient-prepare.json；尚未完成native验收。

## 2026-10-08 冻结集合基线全部完成；主体与两侧尾部交叉分层，信用尚未修复

7f8f53e8/PID432447/birth1791427765.32的GPU4/5诊断完成并退出，原actor group
两rank都返回completed。每rank194次原native forward，1750.15/1750.83秒；
零DT/backward/optimizer/rollout/恢复，未放行正式训练。TextCraft原PID2833207
出生未变、两release仍不存在；AppWorld未重启。没有新增GDN或信用候选。

冻结32+32条、各16状态组的原作者k20累计删除/RISE/MAS与256个均匀单删完成；
完整开发已保存capture中的c>2尾部74个位置全数完成，不回填训练。误差按
预测c与native单删c的相同五档交叉分层，漏估尾部独立；不将它混入有界主体，
不把尾部全数与主体抽样直接合并，不用总均值或有限样本推断总体矩。

DT预测A/r<-1的37个尾部位置中，TextCraft24个、AppWorld14个的native单删
对照A/r>=0，分别覆盖6/7个状态组，含各3个此前未查看的状态组。这证明当前
开发集合问题超出原单点，不证明这些误差都源于GDN或导致旧训练退化。
均匀128点/任务中另有2/11个native c>2而DT c<=2的漏估尾部，与上述全数
集合分别报告，不能把预测极端点精化当成覆盖全部误差的方法。
原作者状态组等权RISE/MAS为TextCraft0.381832/0.637945、AppWorld
0.305489/0.473539；这是当前集合基线，不是修复通过阈值或总体质量结论。
集合误差的实际PPO梯度影响尚未量化，暂停新增GDN候选的决定保持。

实际有效配置SHA5f7a9026与此前AppWorld原owner配置一致；冻结输入SHA86186d74，
原作者/target reader/VERL的实际导入路径及SHA、完整逐前向phase、尾部逐token
原始d和native对照、缺失capture与已查看/未查看分组均保存。诊断无OOM，
阶段worker PSS约7.8/8.7GB，结束主机available907361386496B；不把阶段观察
称连续物理显存峰值或正式DT/PPO容量验收。完整回执见
results_credit_author_collection_20261008.json及current_runtime.json的
latest_credit_stratified_collection；保留原Q/V/A、PPO、LoRA8/16与每卡B4。

## 2026-10-08 原作者集合诊断已启动；训练未放行

诊断代码7f8f53e8，PID432447/birth1791427765.32，物理GPU4/5。复用最新已测
helper7277fade、endpoint8a72887a、原作者metric583f4b7d，正式source58209daa
与冻结manifest2096e03e不变。CPU以旧原生实测分数回放21批/168回调，全部
IDs与原owner的RISE/MAS返回值逐值一致，约0.807秒；它只验证边界，不是模型
容差或集合质量结果。第一次本机哈希错误未产生上传/远端调用，记录见下。

原VERL双卡actor初始化后，真实native B4配对前向首调用约19.49/19.26秒，
后续同批约3.38秒。1791428028观察两rank各完成47次native forward，已完成
16/32条TextCraft曲线与64/128个均匀单删；AppWorld和全数尾部尚未开始。
这是阶段进展，不能将半个集合写成全方法结论。原SHA/导入路径、PID、预算、
CPU回放和阶段PSS/allocated/reserved等见current_runtime.json的
latest_credit_stratified_collection及author-collection-launch.json。
原source中的路径别名与实际target reader路径不同但SHA一致，两者均保留。
没有新GDN候选、DT/backward/optimizer/rollout/恢复；TextCraft hold，AppWorld不重启。
每rank194次native forward的固定诊断与1800秒预算保留，不增加训练查询。

## 2026-10-08 开发主体与负尾部分层；原作者集合诊断 prepared-only

用户要求将长尾与主体分别聚合。只读完整冻结开发capture，以c=exp(-d)分为
<=1、(1,2]、(2,10]、(10,100]、>100五档，区间不是新的容差/裁剪或训练设置。
TextCraft121248个source中35/2个落在(2,10]/(10,100]；AppWorld204865个中
同样35/2个；两者>100观察数均0。缺失的两条AppWorld开发capture不补零。
不据有限样本认定总体矩存在/不存在，不把频率当估计错误率或梯度占比。

准备对冻结32+32条轨迹调用原作者k20累计删除/RISE/MAS；仅将原同步score
callback的真实IDs合为B4配对前向，排序、删除、归一化和指标保持原owner。
另作每轨迹4个均匀source对照及完整开发c>2的74位置全数对照，三种口径
分开；不会把单删除结果回填训练。每rank预计194次原native forward，
无DT/backward/optimizer/采样/恢复。脚本尚未启动；CPU原保存分数回放也未
执行，不把语法检查或prepared源称作接口通过。实际启动/结果另补回执。
训练源、Q/V/A/PPO与LoRA8/16、每卡B4均未改，TextCraft仍hold，AppWorld不重启。
来源为credit-research-20261008/v1/{manifest,corpus,credit-strata}.json；
候选GDN实现仍暂停，本项是集合级基线诊断，不能证明整体退化原因已定位。
首次本机提交在上传前发现helper哈希仍指旧AppWorld单点版本55f88c76，零远端
调用。逐项diff确认实际复用的是已测TextCraft版本7277fade（回执
results_textcraft_actual_author_curves_20261008.json），metric/carrier未变，
新增的是原TaskRunner步骤标量传递与case选择；已纠正绑定，未回退旧helper。

## 2026-10-08 回查确认以点带面的优先级错误；暂停新增GDN候选实现

用户指出后回查PLAN：“研究仅回到已定位的……误差”把已测AppWorld单token
机制变成了排他的修复方向；此前附加“仅该点”没有约束实际决策。已纠正该段，
下方“下一步形成GDN候选”的研究优先级由本条取代；局部实测与一般代数结果
仍保留，不改历史回执，不据此认定整体训练退化主因。正式Q/V/A/PPO未改。

只读冻结开发集：TextCraft85条/121248个source token、AppWorld124条中122条
有capture/204865个source token，2条缺失明确列出。状态组等权的负d比例为
14.019%/17.440%；这是负值发生率，不是反事实估计错误率。已查看状态组与
未查看状态组分别报告，测试集不读值、不重选。TextCraft直接读取原pre-update
中已白化训练系数；AppWorld无完整actor保存，未推造。系数平方和不称梯度占比。
原Format极端token在一个真实minibatch的任务梯度投影约4.00%，去掉该向量
方向变化12.50度；只能量化该点，不支持“极值必主导”，也不证明旧退化根因。

CPU约1.678秒、peakRSS770179072字节、CUDA未初始化；零新增模型/DT/backward/
optimizer/采样，无训练放行、重启或恢复。代表性单删误差和实际梯度影响仍未建立，
不以新负值统计替代它们。完整来源、旧结论纠正及分组结果见
`results_credit_evidence_scope_audit_20261008.json`；当前候选实现优先级已暂停。

## 2026-10-08 GDN 条件窗口有限系数推导与成本核算；没有新增模型/GPU作业

只读核对远端原HF/FLA源码，SHA与已记录owner一致：模型卷积宽度确为4、
value heads32、head维度128。单个层输入位置改变会同时改变后续四个位置的
Q/K/V；将它们分别放进事实背景后独立相加仍会漏掉窗口内状态交互。
已推导完整Q/K/V/alpha/beta有限系数与条件窗口递推，不用事实导数、V-only、
极值筛选、输出缩放或额外整网单删除覆盖。两种端点方向可共用窗口后的事实
future adjoint，但反向方向仍须保留窗口内的条件反向递推。原native h/v_new、
dh_end/dU_WY提供分块数据；不是复制原模型forward或新增奖励模型。

10项非交换符号恒等式展开残差为0，包括耦合状态差、四步窗口、低秩因子递推
和完整有限memory系数。CPU约1.60秒、RSS约64MiB，未导入Torch/模型/FLA；
这是一般代数核算，绝非原官方数值容差或集合归因质量通过。公式仅证明给定
memory输出cotangent的条件窗口有限变化，不冒充整网原token精确反事实。

按真实B4/32768几何，逐位置一份FP32完整状态为256GiB；即使只存四个低秩
因子，全32头仍为32GiB、8头组为8GiB，不能直接铺满序列。原64-token分块
状态一份FP32为4GiB。原mixed代码每端点方向/头组18次GEMM，当前两方向、
四个8头组共144次，纯GEMM算术量约3.436TFLOP/该GDN层；这是源码工作量，
不含native adjoint、投影、搬运与编译，不能换算为已测墙钟速度。
下一步把条件窗口的有限系数化为这些原分块数据上的收缩，核算跨块窗口、
两端点方向的实际工作量及live tensors，再形成owner层候选；尚未实现或部署。
冻结开发/测试manifest未变，原作者累计删除/RISE/MAS仍未取得新集合结果。

本次只读核实TextCraft PID2833207出生1791370325.16仍在且两rank update
release均不存在；AppWorld原PID2786671不存在。未放行训练、重启或恢复。
原始源码路径、SHA、推导、存储算术及资源见
`results_gdn_conditional_context_algebra_20261008.json`；信用修复未完成。

## 2026-10-08 集合级研究数据冻结；四点精化候选已由用户否决，零新增 GPU 作业

已只读原首次 DT 的 prepared/capture：TextCraft 176 行去重为170 UID，
AppWorld224 UID，216有完成capture、8缺失。按第一枚policy token之前的原状态
IDs归组为32/33组；TextCraft的通用系统prompt不能作任务身份。冻结开发/测试
85/85、124/100；已查看异常批次及同状态组仅在开发侧。第一阶段每开发组固定
散列最多2条，共64条，未按信用极值、成功排序或长度挑轨迹。原非零奖励请求集
不冒充全任务采样成功率。原作者k20/RISE/MAS与环境分开聚合，缺失不填零。
CPU收集1.363秒、peakRSS678866944B、CUDA未初始化；分组/去重/汇总接口检查通过，
不是新的归因质量结果。完整SHA及来源见results_credit_research_collection_20261008.json。
用户否决了本轮提出的“选四个最负source追加原生单删除替换”候选，因为它覆盖
输出症状而非修复有限传播背景交互，并增加多次整网前向。草案及CPU准备的768个
查询只作未接受历史保存；三个相关本机入口已禁用，未上传或启动模型诊断入口，
零新增模型/DT/backward/optimizer/采样。正式源与配置完全未改，不放行TextCraft，
不重启AppWorld、不恢复检查点。研究回到Q/K/V/gate/state完整耦合项的原owner
公式与一次DT成本；已有事实VJP20.063 vs原生15.828也排除直接把事实梯度当精确
有限修法。此研究授权取代旧记录“精化裁定待答”，不意味着任何候选已通过。

## 2026-10-08 当前GDN30反号的同操作数分阶段核算完成，零新增GPU调用

仅CPU核算已经完成的原结果，以相同四份artifact SHA、当前newline2883/UID、
原output cotangent及共享实际single-prefix state对齐；没有新模型/DT/backward/采样。
原joint FLA自己的有限/原生差为107.287136/107.289270，差-0.002134；
原single FLA自己的有限/原生差为15.825014/15.827504，差-0.002490。
原joint系数乘actual single位移只得9.671126，与原生single差-6.156378。
加同一已保存residual_skip=-16.636698、z=1.341236之后，记录的gate处+0.543606，
记录的joint finite FLA处-5.624335；用已测single finite/native作算术诊断分别
为+0.529553/+0.532043。不是新训练系数、全模型替换或官方有限归因容差。
记录值的分支重构残差0；gate implied o与原native BF16重放差-0.001754，
此普通数值差与跨端点背景的6.156378差分别保留，不新设误差门槛。
这把当前点反号的主要实测差定位到joint分解近似single的环节，不能证明全部
误差均来自该处。原V-only替换已全向量失败，不由此重试；原作者RISE/MAS独立保留。
1791415904.0715535只读核实TextCraft同PID/birth仍在、双rank release不存在，
AppWorld原正式PID不存在。79922486显存补丁保持verified/un-deployed；信用未修复。
既有单删除精化裁定仍待用户回复；未增加查询策略、修改PLAN或放行训练。



## 2026-10-08 当前AppWorld极端点GDN30实际块完成原FLA dtype对照，信用未修复

9bb32def/PID3281630/birth1791414714.78，物理GPU4。
使用原保存的当前newline2883、GDN30首个single-intervention 64-token块，
四个8-head拼为32-head；实际q/k/v/beta FP16、raw_g FP32、原非零initial state，
原saved upstream，无随机替代。原verify_saved_fla_dtypes.py SHA386a389f...未修改，
调用原FLA 0.4.1 reference/assert_close，FP16与BF16对照各11项通过，未改原阈值。
这是重合端点的o/dq/dk/dv/dbeta/dg检查，不含ht/dh0，不证明非零有限整网归因准确。
FP16原native adjoint部分RMS比显示0但max_abs非零，不能称逐位相同。
此输入通过不覆盖历史不同输入的dk超差，也不覆盖整条PPO或归因质量。
约60.09秒含准备/冷编译；torch peak allocated 564818432B，结束PSS6415631360B，
live物理GPU4观测1996MiB，结束859MiB，物理观测不是连续峰值。
原phase字段absolute_block_start=128实际是capture-local offset；原回执逐字保留，
仅未来helper改名capture_local_block_start。未改输入/结果，也未为元数据重跑GPU。
首次launch引号导致远端Python parse失败，未创建测试进程/GPU；修正后只启动一次。
1791414791.8797884 terminal观测：TextCraft原PID/birth仍hold、双rank release不存在；
AppWorld正式terminal。信用问题仍未修复，无新精化/倍率/裁剪，无训练重启/恢复。
原AppWorld consumed-FA-cache生命周期修复和failed-B4/精确32k验收单独保留。



## 2026-10-08 极端点原生对照覆盖范围量化完成，无新增GPU调用

以原population的actual native file SHA、source SHA、UID、token ID和位置，
对齐已有三个原生单删除样本。没有新增前向、DT、backward、optimizer或采样。
AppWorld已完成54个native capture/216请求行/403866 source slots中的两个A<=-5，
恰好均为已测的newline198，均紧邻下一个target。原A=-53.979836/-21.460581，
原生事实删除d=+23.089031/+20.829633，两个负方向均不受事实单删除对照支持。
两点覆盖89.6333%的负source系数平方，但只占全部已存policy系数平方2.50288%；
不能据此声称占89.6%参数梯度或解释了全部学习退化。App正式DT未完成。
TextCraft已测Format覆盖43.6555%的负source平方/2.57717%的全部policy平方，
方向有据、幅度高估；已有真实global64原生PG对照中其full-gradient投影约4.0002%、
移除该项方向夹角12.4954度。此原生PG证据与平方占比分开保留，不运行GRPO。
TextCraft统计按176已保存请求行加权，含六个重复UID，不冒充原actor全局batch。
-5继承原描述bin，未改成新的查询门限、裁剪或验收标准。原RISE/MAS记录不变。
1791414096.140核实Text原PID/birth仍hold，release不存在；App正式terminal。
信用未修复、未增加精化接法。用户关于精化的裁定仍待回复；未改PLAN或正式入口。



## 2026-10-08 实际DT接口审计：联合分解与单删除误差分别记录

本轮仅CPU AST读取此前已保存的实际导入源码，runner SHA7d6f57f6未改变。
原类仅__init__/forward_prefix/read_outcomes/attribute四个方法；attribute接收
调用者给定的交错端点，末端为同一组finite系数乘各token embedding位移。
原接口支持指定single-EOS pair，已有真实对照已验证；但所审计类没有单独的
一次全向量事实单删除估计入口。联合守恒不能代替逐token删除精度。
当前App真实两组（换行/其他source）的事实删除差之和731.217797，joint差
698.657982，非加性32.559815。这不是全部individual差之和或总体错误率。
Text Format的d误差-0.955824使概率比高估2.600812倍；App新鲜重放的d误差
-27.504585使A由原native约+0.75变-61.295728。仅描述原样本误差放大，不新设容差。
精确假设和Q/V/A组合未撤回；当前问题是实际joint有限估计的近似质量。
已向用户请求裁定是否允许研究一次DT之外的有界native单删除精化；选择规则、
阈值、替换系数和额外查询均未实现，未改PLAN。原作者RISE/MAS与官方算子容差
继续分开；不以单点结论判定整体归因或整个学习退化原因。
1791413421.829只读核实原诊断已退出，Text原PID/birth仍hold、release均不存在，
App正式terminal。零模型/DT/backward/optimizer/采样/恢复/重启。
原79922486显存生命周期修复及失败B4/双次32768回执不变，仍未正式部署。



## 2026-10-08 当前AppWorld极端换行的原生背景交互已量出

3c17af2f/PID3063340/birth1791412620.43已完成退出，物理4/5各回到859MiB。
当前真实B4、row3/packed2883/newline198及原joint Y不变。复用原native scorer、
pair builder和原actor，每rank只增加一次27.82/27.85秒的native paired forward；
零DT/backward/optimizer/rollout/恢复。两rank每target结果一致，其他三行差0、
此前target差0，LoRA B均0。新all-EOS端点与此前实际作者曲线末端逐值相同，
旧事实端点与作者曲线首端逐值相同。
原native四端点F=-230.145626,D=-253.234657,B=-928.803608,C=-938.274391。
完整事实背景的单删除效应F-D=+23.089031；其他source全EOS背景下恢复该token的
效应C-B=-9.470783，背景交互差32.559815。原DT鲜重放d=-4.415554既不是F-D
也不是C-B。因此当前大负优势不能解释为该换行在事实轨迹中有负作用；至少存在
显著背景交互分摊的近似误差。不能据此声称已定位全部误差或官方算子超差。
原DT接口明确分解joint端点变化；PLAN已经将其作为逐token删除效应估计，
此次结果量出该近似在实际极端点的失效，不改变既定精确假设或Q/V/PPO公式。
不把C-B或端点平均塞回训练，不再试无机制依据的kernel变体或数值纠偏。
TextCraft Format原生负作用仍有根据但幅度被高估；其原作者累计删除/RISE/MAS
与原single诊断同时保留，不被当前AppWorld四点对照取代。
PSS记录9.111/8.223GB，未连续采样本次物理峰值。Text原birth继续hold，
release均不存在；App正式terminal。信用仍未修复。79922486缓存生命周期补丁
已另行通过真实失败B4及双次32768容量，未部署到正式作业，也不冒充PPO整网验收。
实际导入、SHA、固定配置、PID创建时间、transport及回执写入current_runtime末字段。



## 2026-10-08 TextCraft实际Format轨迹的原作者累计删除完成

2b98d7f9/PID2967962/birth1791411696.44，物理4/5诊断已完成退出。
同一正式真实B4、3068个原source位置及原joint动作Y，原作者函数583f4b7d/k20，
每rank42次原native forward，零DT/采样/backward/optimizer/恢复，约244.18秒。
只给既有诊断增加task选择，并透传原TaskRunner已解析的330给原actor初始化；
无新训练预算、metric/scorer/sorting实现。真实ID transport原2项CPU检查通过。
两rank原数组/分数一致、paired twins差0、其他三行分数恒定，LoRA B均零。
原signed RISE=0.408195828；positive-only评估MAS=0.694071027。
Format在signed第20组删除。该组153个实际改变source均负、DT总和-13.990249，
native kept-minus-deleted却+56.012404。这是前19组已删除背景下的联合效应，
不能把56归给Format个人，也不替代它的原single删除A=-8.157497对照。
原始非单调logp、作者归一化/penalty数组、CSV、图和完整回执均保留；
这些是单条实际轨迹质量证据，不新设容差/总体质量结论或以排名证明指数幅度准确。
PSS记录峰值7.612/8.558GB，无OOM；本诊断未连续采样物理峰值。
Text原birth仍hold、release均不存在；App正式terminal，无正式重启/恢复。
信用仍未修复。79922486的consumed-FA-cache生命周期修复单独已验收，未正式部署。
导入路径/SHA、原配置、启动commit/PID创建时间与回执已绑定current_runtime末字段。



## 2026-10-08 原生事实梯度诊断结束：普通梯度不等于此有限删除效应

3c9c7081/PID2889955/birth1791410961.25已完成退出。只对保存的实际GDN30
单删除操作数调用原FLA，4个八head分组各一次forward/backward，未加载模型或更新。
事实输入逐值相同，incoming state固定原值，FP16 q/k/v/beta及FP32 raw_g保留；
原cotangent只在事实行，参考行梯度均零，所有梯度有限。
原生有限输出效应+15.827504；原生事实梯度乘真实差+20.063134，差+4.235630。
之前同一single端点的原finite为+15.825014；联合参考系数乘该single差+9.671126。
这些是固定原输出cotangent的局部分量，不是完整advantage或官方容差通过。
不能据此改用普通梯度，也不能局部替换或加倍率强行修正。信用仍未修复。
Torch live峰值0.948GiB、PSS约6.27GB；未连续采样物理峰值，首group含编译。
Text原birth仍hold、release均不存在；App正式terminal，无恢复/正式重启。
实际源码SHA、启动commit/PID出生、原算子和回执写入current_runtime末字段。



## 2026-10-08 条件V对照和完整向量结束：局部替换未修复信用，拒绝部署

原生FLA条件V诊断9738653b/PID2700340/birth1791409158.24已结束。
保存的真实GDN30单删除V端点，其余输入固定事实值、incoming state相同。
原生条件效应+14.116234，原forward V系数乘真实差+14.112987，差-0.003247；
对称V项+5.508959，差-8.607275。输入FP16 q/k/v/beta、FP32 raw_g均已记录。
这是条件V，不是独立token奖励或完整advantage，不另设finite容差/通过判据。

54b398d8/PID2758971/birth1791409734.01的完整B4诊断也结束，4/5已回到859MiB。
仅保留原forward V系数、其他原对称分量不动；基线全向量与端点逐值复现。
换行d从-4.415554变-6.153260，A从-61.295715变-351.936005，仍与原生单删除
d=+23.089031/A=+0.75相反。联合残差从-1.350164变-182.973199，未保持原有限恒等式。
因此候选拒绝，未校正、未部署，也不继续花费新增作者曲线来宣称它有效。
此前作者累计删除/RISE/MAS回执保留；本次没有总体归因质量结论。
两rank各2次DT；105.26/105.32秒冷基线与66.88/66.92秒暖候选不称加速比。
物理峰值54196MiB，PSS峰值9.050/8.333GB，无OOM。

CPU原保存元数据核对确认TextCraft Format所影响的目标来自官方parser已经POST的
动作，归一化为n format，源自模型复述的Action格式示例。不是桥中新造target，
不自行删除或重写。该token原生删除使附近format目标概率0.1054升至0.9708，
完整d=-2.214573/A=-8.157497；正式DT为-3.170397/-22.816927，方向有据、幅度偏大。
按traj_uid查原actor rank0 row5；DT rank1不等于balance后的actor rank1。

79922486显存生命周期修复的真实失败B4/双次32768容量回执独立保留，尚未正式部署。
Text原birth继续hold、release不存在；App正式仍terminal，无optimizer/恢复/重启。
当前信用未修复；不能把条件分支局部接近原生当成全链修复，也不能用倍率掩盖残差。
实际源码路径/SHA、启动commit/PID创建时间、固定配置及回执绑定在current_runtime末字段。



## 2026-10-08 完整clean-v1 GDN数值规则对照完成：未修复极端信用，不部署

b5b7b391/PID2575252/birth1791407996.54两rank已完成，物理4/5回到各859MiB。
同一实际B4、当前存储/内核/分块/target/QVA/PPO不变，仅选择作者保留clean-v1的
空norm/gate与memory map。完整基线向量逐值复现，所有原target端点数组逐值相同。
换行符2883的原对称d=-4.415554/A=-61.295715，clean d=-8.588811/A=-4027.667480；
先前原生单删除d=+23.089031/A=+0.75，实际奖励0.75。因此未修复，候选拒绝。
此前偏置12点中的反号4变5，不是总体错误率。两rank全向量与QVA逐值相同。
完整DT冷基线105.63/105.52秒、暖clean70.21/70.25秒，不称加速比。
物理峰值均54196MiB，PSS峰值8.151/8.375GB，原79922486缓存释放仍生效。

此候选未跑新增作者曲线：它已经未能修复本次明确异常，不继续花费曲线计算；
不因此声称总体归因质量变差。原对称及memory-only的作者曲线回执保留，不被单token
对照替换。辅助clean曲线入口只是复用原owner的诊断选项，未启动/未部署。
没有裁剪、倍率、改target、改Q/V/PPO、修改容差或恢复检查点。Text仍hold，
App正式仍terminal。当前信用问题未修复，显存生命周期补丁验收单独保留。
所有源码SHA、实际helper、原任务配置及回执绑定见current_runtime末字段。



## 2026-10-08 实际下一代码围栏隔离完成：后续目标加入时出现负信用

5699ac34/PID2401809/birth1791406322.4已完成退出，单次完整DT/rank约105秒，
物理峰值各43420MiB。原实际B4、reference source IDs和所有target输入均保留；
只在原目标接口选择row3下一代码围栏71093的score，原2656个score变为1个。
这不是训练Y修改；诊断不导出完整事件Q/V/A，原producer日志也不能当作训练量。
换行符2883的完整DT d=-4.415554，所选围栏d=+4.272880；两者代数差-8.688434。
先前原生单删除围栏d=+21.029863，其余目标净d=+2.059169，总d=+23.089031。
故完整负值在后续目标加入时出现；不能据此删target、裁剪、加倍率或修改Q/V/PPO。
两rank隔离向量逐值相同，围栏端点logp跨运行逐值相同；其他三行归因max差0.014422，
端点logp max差4.7e-7，head输入形状变化。残差未定位，无全DT官方容差结论，
-8.688434只能称两次运行代数差，不能称精确独立剩余目标归因。

下一诊断b5b7b391/PID2575252/birth1791407996.54只在4/5核对作者保留clean-v1的
完整GDN数值规则（原norm/gate map和memory map均空），保留当前已验证存储生命周期、
内核、分块、真实target、Q/V/PPO。不是恢复历史runtime，也未切换正式profile；
源码SHA/CPU身份/启动实参见clean-profile-owner.json和clean-gdn-launch.json。
Text仍原birth hold，无optimizer release；App正式不重启、不恢复检查点。
当前信用未修复，79922486显存容量修复与此分开验收。



## 2026-10-08 完整memory对照与原作者曲线结束：单方向规则未被接受

完整DT诊断99ad9596/PID2175801/birth1791404173.84、作者曲线8987deac/
PID2233742/birth1791404715.28均已完成退出，物理4/5回到各859MiB。
同一真实B4、原对称向量逐值复现；仅改为原已有forward memory callback时，
newline的d从-4.415554变-4.698849，原始A从-61.295715变-81.615494。
先前原生单删除d=+23.089031，对应A约+0.75；主要来自下一code fence的概率
0.998526降至7.3486e-10。此大负值不符合当前定义的单删除文本反事实方向。
原作者函数SHA583f4b7d、k20和所有原数组保留；原对称/forward的signed RISE
为0.393269/0.403295，positive-view MAS为0.631998/0.644110，均未改善。
这是选定真实轨迹，不能当总体质量结论；两rank各42个原生forward，无新DT/
backward/optimizer/rollout/恢复。完整DT两模式各一次/rank的冷暖时差不称提速比。

已有原生FLA张量的CPU分项汇总进一步定位：真实single端点上原finite的V分支
为+12.016079；joint正向/反向系数乘同一single差为+14.112987/-3.095070。
反向V项差-15.111150，其余项合计补偿约+2.686，不能把分项称独立V干预或
全token优势。全native single输出效应+15.827504，原single finite为+15.825014；
未发明非零finite官方容差。已有单方向局部改善没有传递到完整向量与作者曲线，
故不部署，不组合扫描，不裁剪或加倍率，不修改Q/V/PPO与原白化。

App OOM的79922486存储生命周期补丁与信用质量分开：实际失败B4和包含原
async-vLLM生命周期的两次精确32768 DT容量已通过，峰值60.803/62.627GiB；
此为容量回执，不是32k任务/PPO更新/信用准确性验收，也尚未正式部署。
Text原PID2833207/birth1791370325.16继续hold，release不存在；App正式未重启。
所有实际路径/SHA、启动commit/PID创建时间、配置与回执绑定见current_runtime
末字段、results_existing_memory_rule_20261008.json及
results_existing_memory_author_curves_20261008.json。信用尚未修复。


## 2026-10-08 原已有memory callback完整B4对照已启动，尚未信用修复

99ad9596/PID2175801/birth1791404173.84使用物理4/5；原真实B4四行、target IDs/
offsets CPU逐值检查通过且CUDA未初始化。原VERL fresh actor、LoRA8/16、每卡B4不变。
原对称memory与原forward memory各一次完整DT/rank，原finite_fla_by_layer空map选择
已有compiled callback；norm/gate仍sym、FA仍content1，原head/奖励/QVA/PPO均不变。
没有新公式、裁剪、倍率、逐token重算、rollout、optimizer、恢复或正式profile发布。
初次原对称模式两rank112.04/112.14秒完成，极值d=-4.415553734907168复现；
最新旁观单方向模式到decoder4、物理4/5各54196MiB。当前运行源/脚本SHA/phase/PID
及实际输入SHA在memory-launch.json、memory-observation系列和current_runtime末字段。
Text同birth hold、release不存在；App正式未重启。待完整向量及原作者曲线结果。


## 2026-10-08 原native FLA对照：当前反号来自联合memory分配，反向端点是主要来源

原保存张量的三个算子诊断均已完成退出，仅物理4；没有加载模型、完整DT、训练
backward、optimizer、rollout、恢复或正式发布。f004170b/PID2003987/birth1791402505.21
核对原joint；a34e2228/PID2047919/birth1791402934.07核对真实single删除；
4e7faa79/PID2107699/birth1791403500.67旁观原memory callback两端方向，原返回不变。
每组8 head，原Q/K已归一化、raw_g FP32、Q/K/V/beta/do FP16。原public
chunk_gated_delta_rule读实际incoming h；single在唯一删除之前的原chunk边界复用
实际factual h，不估计初始状态、不置零、不构造工具返回。所有保留factual操作数逐值相同。

联合finite收缩107.287136，原native输出差107.289270，相差-0.002134。
真实single的原native输出效应15.827504，同端点原finite为15.825014，差-0.002490；
原joint系数乘真实single差却只有9.671126，差-6.156378。加回相同skip与z后，
原joint点-5.624335，native single点+0.532043（原BF16边界+0.545360）；原gate点+0.543606。
最大差额来自head24..31：joint10.038809，native17.026803，差-6.987994。
因此这次首次反号不能由该局部运算的舍入解释；joint总贡献近似一致不代表每token准确。

原两方向分别15.942119、3.400138，原平均9.671128。正向对该single差+0.114615，
反向差-12.427367；反向端点拉低是该点分配误差主要来源。仅据此不部署单方向规则，
下一步是原已有单方向memory callback的完整向量和原作者累计删除/RISE/MAS对照；
不新增公式、不修正倍率、不裁剪、不改Q/V/PPO。前次attention PV改法没有改善整条
作者曲线的结果继续保留，不能混为本次memory对照。此诊断也不替代作者曲线。

原native/finite/profile路径、SHA、三个code commit、PID/birth、原数据SHA及小文件运输
回执已保存在results_current_extreme_native_fla_20261008.json。非零finite没有发明官方
容差，此处不称FA/FLA官方验收通过。后补CPU同source导入解析与实际运行记录分开。
Text同birth hold、release仍不存在，App正式未重启。App内存修复79922486已完成
原失败B4及两次32768 DT有界回归，physical60.803/62.627GiB；仍非正式部署/更新健康性证明。
LoRA8/16、每卡B4及32768不变。三个算子诊断终态4/5各859MiB；组末allocation/PSS非连续峰值。


## 2026-10-08 已排除当前反号由中间GEMM低精度舍入解释

9863bd84、算子重放PID1874736/birth1791401248.61完成退出，只用物理4，
不加载模型/不运行完整DT/无训练backward、optimizer、rollout或恢复。
原四组8-head FLA张量直接调用原finite_fla_pullback与原两端平均callback。
原native-GEMM eager收缩-5.624333245，与原B4 compiled保存值-5.624335441
相差+0.000002196；仅取消中间GEMM的FP16操作数舍入后为-5.624418073，
相较eager只变-0.000084828。TF32=False、float32_matmul_precision=highest。
这与前面+0.543606到-5.624335的6.16794变化不在同一量级，不能靠这一部分
提高精度修复信用反号。native FLA adjoint阶段/捕获值/elementwise/Triton仍保持
原样，因此没有排除其他数值来源或公式错误，也未发明非零finite的FA/FLA容差。

原四组算子累计4.749秒，FP32对照0.1385秒有先冷后热差异，不作速度比较。
每组结束live allocated约0.516GB、PSS最大7.237GB；这些不是连续物理峰值。
终态4/5各859MiB。完整原始结果与SHA在results_current_extreme_fla_precision_20261008.json。
正式Text同birth hold、App未重启；Q/V/PPO/LoRA8/16/B4与生产后端没有改动。
内存修复已完成有界原失败B4+两次32768容量回归，但仍未部署到正式作业。
下一步核查原有限FLA公式/联合删除分配与原native输出，而不是再调head或MLP。


## 2026-10-08 GDN30 首次反号在有限FLA；原算子输入已留存

e1bc175d、PID1788375/birth1791400430.70完成退出。原B4/两rank/31与30边界
逐值等于此前完整旁观；内层七类事实端点在原保留区间逐值相同。single原
capture_start=128、joint=0，按原坐标对齐，无pad/伪造端点。GDN进入5.39004，
norm/gate后仍+0.543606，有限FLA后-5.624335，完整GDN后-5.598628。因此
首次反号已缩小到有限FLA；还不是FLA数值内核出错的证明，未部署任何纠偏。
实际原profile同时有symmetric norm/gate与两端平均memory callback，已核对
实际constructor回退到top-level profiles及该源SHA dd6bbfff，不能重做已有对称化。

每rank两partial DT；single49.59秒，joint rank0含3.712GB原算子输入/输出落盘
45.23秒，rank1无落盘19.74秒；总启动至最后worker完成165.54秒。记录PSS最大
13.213/12.715GB，终态4/5各859MiB。原张量分四个8-head文件留在远端，SHA/字节
见results_current_extreme_gdn_20261008.json，后续可直接重放原算子，避免模型重跑。
没有full signed/QVA、rollout、PPO更新或恢复。Text同birth hold、App正式不重启。

下一有界诊断已准备：用保存的原算子张量直接调用原eager finite_fla+原两端平均，
与只取消中间GEMM的FP16操作数舍入作FP32对照；FLA adjoint阶段/原捕获值不变。
这是部分精度来源诊断，不能标成完整FP32参考或官方容差通过，不是生产后端。


## 2026-10-08 GDN细分v1失败保留；修正旁观capture_start对齐，v2准备

6f712988、PID1732100/birth1791399903.96退出，未完成。新增旁观器错误地直接
比较原single compact捕获9723与joint9851的位置维；错误在旁观比较、尚未进入
joint GDN算子。不把此错误归于生产DT或OOM。完整栈/原源/launch/config已按SHA
保存到gdn-results；4/5回859MiB。此前31/30子操作结果仍成立，没有被覆盖。

v2仅使用原GDN capture_start把两次实际保留区间对齐，在公共时间坐标收缩；
不pad、不伪造端点、不中途改原capture范围。被省略区间在唯一删除token之前，
沿用原因果前缀语义。所有原callback输入/返回值仍不变。v2独立-v2目录，CPU原
输入身份验证完成，尚未GPU启动。Text仍hold、App正式不重启、参数与信用未变。


## 2026-10-08 GDN30 norm/gate 与 FLA 细分诊断已准备

沿用同一真实B4与原两partial DT到decoder30；只把现有GDN norm_gate_pullback
及finite_fla callback套上返回原值的旁观器。记录原mo/mz及q/k/v/raw_g/beta
系数与真实single端点差的收缩，分清反号是否在norm/gate或FLA。rank0保存
实际finite FLA输入/输出的选定行原张量，留在远端用于原算子数值重放，避免
再次全模型前向；不改kernel/规则/精度/学习信号。未运行任何新精度判据。
CPU原输入身份与CUDA未初始化检查已完成；尚未GPU启动、尚未信用修复。
Text仍hold，App正式不重启；memory候选的有界容量验证状态不变。


## 2026-10-08 当前极值首次反号已缩小到 GDN mixer 内部

2ce604be、PID1645795/birth1791399078.23完成退出；本次仅到decoder30，
没有全程signed/QVA。两rank所有分支收缩完全相同，32/31/30边界逐值等于
上一完整旁观结果；31/30的五类事实端点也逐值相同。FA31：MLP后30.6475、
post RMSNorm后29.9594、attention mixer后5.27249、input RMSNorm后4.78742。
GDN30：MLP后5.25186、post RMSNorm后5.39004、GDN mixer后-5.59863、
input RMSNorm后-6.81460。因此MLP/外层RMSNorm没有制造首次反号；首次发生
在gdn_finite_pullback内部，FA31 mixer是此前最大缩减。整体joint端点收缩
32/31/30分别700.203/700.346/700.230；不能把整体守恒当作该token精确。

原两partial DT约50.74/17.73秒，CPU边界快照最终0；无optimizer/rollout/恢复。
本次启动至最后worker完成141.27秒，physical终态4/5各859MiB。结果见
results_current_extreme_subops_20261008.json及subops-results原SHA回执。
尚不把定位称作GDN内核数值错误或信用修复；需继续分清原norm/gate与FLA。
Text同birth hold，release不存在；App正式不重启，memory候选尚未正式部署。


## 2026-10-08 decoder31/30 原分支旁观诊断：已准备，尚未启动

对当前 AppWorld row3/packed2883/ID198，同一原B4、memory候选与producer不变。
只在原 decoder_finite_pullback 的 MLP、两次 norm_residual 和 mixer callback
外记录实际 single 状态差的收缩，所有 owner 参数/返回值不变。原native hooks
只保存31/30两层所需端点；两次诊断均在decoder30后受控结束，原release_owner_params
清理参数。不输出全程signed/QVA、不做PPO更新、不部署新的传播规则。
CPU输入selected/target/offset与原件逐值一致且CUDA未初始化；真实producer的
__call__/release_owner_params/attribute_episodes已按实际导入SHA2aa5f552核对。
下一步只运行此有界分支定位，并核对31/30边界是否仍等于上一全程旁观结果。
未声称精度通过、信用修好或训练恢复。Text仍hold，App正式不重启。


## 2026-10-08 02:29 当前极值首次反号在decoder30；不是旧样本layer27

26b43fa2、PID1470525/birth1791397411.29完成退出；GPU4/5各859MiB。实际
row3/f0f85f5c/traj64/response125/packed2883/ID198、原B4 SHA3e902bc0。
两rank整份joint signed逐值等于此前无逐层旁观的同B4 content1结果，差0；
33边界的candidate事实状态逐值相同。原single DT d=+22.500395，当前cached
native端点差=+21.783119；两者残差+0.717276。此前独立uncached native
单删除=+23.089031，与当前cached相差1.305913，分别保留，不冒称同一端点。
joint系数×真实single状态差：输出边界+27.714940，经FA decoder31到+4.787422，
经GDN decoder30到-6.814600，输入-4.415554，等于原joint token d。实际模型
日志layer_types确认31 full_attention、30 linear_attention。定位只缩小范围，
不能称GDN内核数值错误；下一步需分清31/30层的MLP、RMSNorm及mixer贡献。

每rank原两DT约95.75/71.8秒，无observer-mode/rollout/反向/optimizer/恢复。
5.326GB single CPU快照最终全释放；phase记录PSS最大14.115/13.555GB。
这不是物理VRAM峰值，原root.__call__未被forward_root hooks覆盖，也不把缺失
native/cache记录当零差。结果results_current_extreme_layer_20261008.json、原始
回执/源/配置/SHA/图已保存。没有新增容差、纠偏倍率或算法改动。Text仍同birth
hold、两个release不存在；App正式未重启，信用误差未修。显存候选单独通过有界
回归的状态不变，尚未正式部署。不以旁观逐值相同取代FA/FLA官方容差。


## 2026-10-08 02:24 当前极值token原逐层诊断已实际启动

26b43fa2，PID1470525/birth1791397411.29实际核对存活、GPU4/5，当前原actor
初始化，尚无DT完成结果。诊断66f77d08继承原inspect_layer_effect，只增加
case参数并将旧固定row0切片改为case.row；当前row3、f0f85f5c/traj64/
response125/packed2883/ID198。原load_request CPU身份/映射检查通过。
case SHA9827ca42、source58209daa、原B4 SHA3e902bc0、runner7d6f57f6。
两次原producer调用仅测single/joint边界标量，不改变返回值、算子或公式；
计划没有rollout/反向/optimizer/恢复。Text同birth hold且两个release不存在，
App正式未重启。此处不引用旧样本的layer27反号来替代当前样本的定位。


## 2026-10-08 当前最大负优势样本的逐层诊断仅CPU准备完成

原inspect_layer_effect诊断新增可选case绑定；旧row0默认不变，旁观切片和
单EOS位置改为读取既有case.row，因此可保持原B4顺序测当前row3。未复制
DT计算，没有改变有限算子/FA/FLA/头/信用/PPO；仅现有边界系数的scalar
收缩及native状态比较。当前case f0f85f5c、traj64、response125、packed2883、
ID198，native SHA3e902bc0、source58209daa。CPU原load_request核对UID、
位置、prior/target映射，原readout核对selected/target IDs和offset，CUDA未初始化。
使用另有容量回执的runner7d6f57f6；正式入口未修改、未启动诊断或释放更新。


## 2026-10-08 02:09 作者原累计删除/RISE/MAS对照已完成；不能靠修极值切换规则

a7505556，PID1322365/birth1791396004.81完成退出。两个rank各42次原模型
前向，driver到最后worker完成556.30秒；原函数583f4b7d、k20未改。原
signed排序RISE=.393269，已有content0=.425759；正值评估视图MAS=.631998
对.690447。原source/positions、事实与全EOS端点一致，另外三行分数恒定，
paired twin最大差1.14e-13，仅作控制观察，不称官方数值容差验收。
同一实际极值轨迹2693个prior source位置；原signed排序最后改变134个
负DT token，signed和=-28.6496，但保留该组使原目标logp增加89.9530。
这是该累计背景下的组交互不匹配，不把组效应当单token真值，也不证明全部
归因排序差。两规则都有该现象；content0修了换行点却没有改善该轨迹的原
整体指标，不部署。归因误差仍未修，QVA/PPO/白化/任务/FA-FLA容差未改。
结果results_existing_PV_author_curves_20261008.json、原始数组/CSV/图及运输
SHA已保存。物理峰值56523/56524MiB、worker PSS峰值7.564/8.488GB；结束
后GPU4/5各859MiB。Text2833207同birth、两个release不存在、零正式更新。
App正式仍终止。显存修复单独通过原失败B4与两次32k容量回归，尚未正式部署；
不把曲线结束、显存候选或启动等同于信用修复或训练健康。


## 2026-10-08 02:03 作者原累计删除/RISE/MAS双规则对照运行中

诊断a7505556、PID1322365/birth1791396004.81、GPU4/5，两个worker已分别
完成signed_RISE的12/21原模型前向点；每次约10.81秒。原曲线诊断55f88c76
未改，原作者ft_ifr_improve 583f4b7d的排序、分组、k20、归一化和指标未抄写。
薄调用层只为rank0/1分别绑定此前保存的content1/content0 signed向量；实际
AppWorld B4 IDs、target和O未变。每rank共42次原前向，无DT/反向/optimizer。
原FA/FLA、QVA、PPO、正式profile和rank8/alpha16/B4不变。positive_MAS的
正值视图仅评估，不修改训练信用。Text仍同birth hold、零正式更新；App未重启。
运行回执curve-launch.json和curve-observation-1791396237.json已本地保存。
尚无最终质量结论，不能把部分曲线或启动等同于信用修复。


## 2026-10-08 原已有attention PV两种顺序对照完成；尚未更换正式规则

诊断9567ba35、PID1117764/birth1791394038.38完成退出。实际AppWorld B4
rank1/batch16 SHA3e902bc0，原source58209daa；两个原owner选项共享完全相同
事实/删除target logp数组，各自signed和QVA跨rank逐值一致。极值换行当前
content1 d=-4.415554/A=-61.295715，content0 d=+3.562343/A=+0.728721；
之前独立单删除d=+23.089031/A=+0.75。旧正式content1 d=-4.29009与此次
fresh重放并非逐值一致，不将独立运行漂移冒称修复影响。
预选12点的反号4→5，不能为修好极值就部署content0；正式profile未动。
物理显存两卡各峰值54196MiB，进程phase最大PSS约8.34GB，QVA有限。
两调用约107.5/67.9秒，后者热调用，不能据此宣称规则本身提速。原诊断main
未输出effective-config.yaml，冻结source和实际main/runner SHA另有绑定，
不把文件缺失当训练故障，也不冒充完整运行配置抓取。结果见
results_existing_PV_rule_20261008.json；未更改FA/FLA容差、QVA、PPO或白化。
作者累计删除/RISE/MAS原函数比较只已CPU准备，尚未启动或得出质量结论。
Text原同birth hold、两个release不存在、零正式optimizer更新；App正式终止。


## 2026-10-08 01:28 原已有attention PV顺序诊断已启动；未换正式profile

诊断9567ba35，PID1117764/birth1791394038.38已实际核对存活，GPU4/5。
原App source58209daa，实际极值B4 rank1/batch16 SHA3e902bc0；四行12512–12603。
CPU通过原readout核对selected/target IDs和offset逐值一致，未初始化CUDA。
原qwen35_decoder_finite 1c58c33c的attention_finite_pullback明确支持content1/content0：
后者仅交换原有限FA端点及V参照，不是新的FA实现或新QVA。此处对同一B4调用
原producer两次，只比较已有选项；24个GDN symmetric/head/原奖励/QVA不变。
继承单独有界验证的显存候选runner7d6f57f6/env cd28a6e2，不把其旧preparation
状态字段当当前验收。未改正式source或profile，未恢复/采样/反向/optimizer。
Text原driver同birth仍hold，两个release不存在。该诊断尚无结果或质量接受结论，
也不以规则差异测试代替FA/FLA数值容差或作者累计删除/RISE/MAS。


## 2026-10-08 01:12 原首个PPO minibatch的极值token梯度已量出；正式更新保持hold

results_update_gradient_20261008.json绑定诊断5fba8de4、Text source2796233e、原actor
3a65e173/core fc2f992b导入路径SHA、原两rank pre-update SHA和完整配置。使用原第一
global64/local32、每卡实际B4×8，203153有效token；未在此minibatch重新白化。
原VERL update_policy运行两次PG诊断，保留原前后向/累积/clip，optimizer/scheduler
写入关闭；496个可训练参数张量/rank逐值不变。没有DT、rollout、恢复或正式重启。
原observer历史字段grpo_pg仅承载已存在的Format单token系数，不是GRPO算法对照。

全PG范数.04516558，Format单项.00977713，范数比21.6473%；这不是可相加的占比。
其在全梯度方向的投影约4.0002%；两份实测梯度相减，方向变化12.4954度。
不能据此声称单项主导，更不能把首次minibatch诊断当旧发散训练的完整因果证明。
各pass/rank原8个microbatch的ppo_kl和上下clip fraction为0；未保存逐token ratio，
不将这些scalar观察冒称逐token ratio逐值验收。没有发明梯度或数值容差。
两pass每rank约129.0/115.2秒，诊断PID952923/birth1791392443.42已完成退出，
4/5回到各859MiB。原Ray指标exporter不可用告警已保留，不影响已完成的梯度回执。
Text2833207仍同birth，两个release不存在、零正式更新；App正式终止、未重启。
信用误差未修复；显存候选仍为单独有界验证，不混称正式部署或训练健康。


## 2026-10-08 00:49 真实非零FA端点两种接口逐值一致；信用仍未修复

results_nonzero_fa_layout_20261008.json记录原App source58209daa、decoder27真实
operands ed804b5c、owner3e1d6103/库4f42c391；4行K长度9567/9644/9655/9773，
Q为各自原后缀，Q/K两端点均非零变化。原scalar与row入口均与原B4保存输出
dq/dk/dv/tau/center逐值一致，40个张量比较，8次原调用各0.33–0.34秒。
这验证位置/padding表示未改值，不是非零有限归因的FA精度接受标准。
没有改dtype/数学/PPO/QVA，无模型、DT、反向、优化器或恢复，未部署正式。
PID880541/birth1791391756.04完成退出，后续4/5均859MiB；Torch峰值0.654GiB
allocated，进程PSS约7.13GiB，不把它冒称物理峰值。Text仍hold，App未重启。
正在准备复用原VERL完整64条first-minibatch的任务/Format贡献梯度观察，
只关闭optimizer/scheduler写入；准备状态不冒充已完成梯度结果或训练修复。


## 2026-10-08 00:33 实际 actor 位置和原白化核对完成；未运行更新

results_actor_credit_mapping_20261008.json 绑定正式Text source2796233e、真实两rank
pre-update输入SHA及原VERL masked_whiten导入文件SHA。只读CPU诊断，CUDA未初始化，
进程RSS约9.00GiB；没有新的GPU诊断、参数更新、恢复或正式重启。
先前native端点对照的12个source位置全部在原保留映射及有效actor mask内。
Format的14606是token ID，原回复位置351；原A=-22.816927，实际actor系数=-140.340958。
整批713539个有效token的actor系数与原masked_whiten一次计算逐值相同；
response_mask与原loss_mask回复切片逐值相同。不是按卡或B4另行白化。
这排除了这些位置的截断/映射/白化实现错误，不证明DT估计已修复。
尚未测参数梯度，不能把系数平方占比当梯度占比；12点仍是偏向高影响的诊断样本。
Text两rank继续hold/release不存在，App正式仍终止，显存候选验收范围不变。


## 2026-10-08 00:20 显存候选完成连续32k及原失败B4回归；信用仍未修复

诊断6cfa4a9d，PID595731/birth1791388988.67完成退出，4/5各860MiB。
results_memory_capacity_20261008.json绑定实际runner导入7d6f57f6、环境cd28a6e2、
原App source58209daa、补丁79922486、完整配置来源、脚本SHA、运输SHA及phase曲线。
Text2833207同出生仍hold，两rank未release，零optimizer；App正式保持终止。

每卡原B4、LoRA8/16、原VERL async actor及原vLLM sleep，三次DT/rank：
精确32768重复两次，各四行，317.108/317.150秒及279.267/279.234秒；
两次QVA逐值相同。物理峰值60.803/62.627GiB，均低于64，第二卡余量约1.37GiB。
容量输入只延长原observation段，原policy/target IDs及顺序保持；明确是容量压力，
不是实际官方32k任务、信用质量或PPO更新验证。每次结束live allocation增70144B，
不能称严格零增长；未见大张量跨调用累积。最后finite层后FA DynamicLayer存储0，
GDN LinearAttentionLayer保留204MiB本地状态，整个调用结束回到近起点。
每rank采样PSS约8.11/8.30GiB，未见异常主机内存膨胀。

随后原真实失败B4完成185.449/185.100秒，最长full context27334（causal27324），
QVA与先前同候选、同原async-vLLM完成输出逐值相同；QVA全部有限。
这些对照验证存储调度没有改值，不冒称整个DT通过FA/FLA有限归因精度标准。
没有clip、倍率、dtype/公式/PPO/白化、任务参数或batch改动，没有恢复/正式重启。
候选已通过此有界回归，仍未替换正式冻结source；不宣称完整训练健康。

信用诊断补充：实际profile是gdn-symmetric-v1，不是误选clean-v1；原LOOP
extractor不要求Code:或其后的换行，不能把该source移进target消除负值。
Text Format实际单EOS支持负方向（A=-8.1575），joint给-22.8169是幅度夸大；
App换行实际单EOS为正（A=.75），joint给-53.9798是方向错误。公式中的样本
重要性权重可使|A|超过观测r，不能只按数值大判错；这两个位置按真实端点
区分方向和幅度。未改现有Q/V/A，未用单点替代累计删除/RISE/MAS。


## 2026-10-08 00:03 实际归因profile核对；32k连续调用显存诊断已提交

诊断代码6cfa4a9d；PID595731/birth1791388988.67，原App source58209daa，
GPU4/5，原VERL AsyncActorRolloutRefWorker与原vLLM wake/sleep。候选79922486
runner7d6f57f6及环境cd28a6e2不变。CPU通过原DirectActionTargetReadout准备8行，
每行实际DT causal input精确32768；仅延长原observation段，policy/target IDs
顺序及数量保持。它是单独容量压力输入，不冒称真实32k任务或归因准确性。
计划两次相同32k调用检查phase/重复后显存回落，再复测原失败B4，与先前同候选
同async-vLLM生命周期完成输出比较。不请求采样/优化器/恢复/正式重启。
Text2833207仍hold、两rank未release；App正式保持终止。当前只是已提交，
尚无GPU32k成功、容差或长期无泄漏结论。

远端正式factory/profile核对结果已保存direct-target-rule-audit-20261008/v1：
Text/App都用gdn-symmetric-v1，24个GDN norm-gate symmetric并平均两种memory
endpoint顺序；attention保留正式content1。未发现调用旧clean-v1的配置错误。
未擅自修改正式profile、Q/V/A、PPO、白化或加入信用纠偏。


## 2026-10-07 23:49 保存信用全量统计及高影响B4单删除完成；未放行更新

诊断代码34d4c390，results_credit_sample_20261007.json绑定原source、实际owner
导入路径/SHA、原TaskRunner scheduler标量、有效配置、PID出生、运输SHA及原始结果。
App诊断431317/birth1791387446.35、Text诊断457448/birth1791387673.50均完成退出；
4/5最终各859MiB。Text原2833207同出生时间仍hold，两rank无release、零更新；
App正式任务保持终止，不恢复检查点，不重启正式训练。

CPU统计调用原counterfactual owner SHA0d3412b8，未初始化CUDA，无DT/更新。
Text44个保存native批次176行170UID，含6重复行；App54批216行216UID，
不包含未完成native批次。统计按保存请求位置计数，不称独立轨迹总体频率。
Text prior A<=-5有6位置，平方和占prior总14.98%、all-policy总4.51%；App有2位置，
分别34.26%、2.50%。都是原始系数平方占比，不是参数梯度或白化后占比。

新发现App rank1/batch16、UIDf0f85f5c、response125/packed2883，token198为
Code:之后换行。r=.75，joint d=-4.29009095，预计A=-53.97983551；单个位置占
保存prior平方和29.58%、all-policy2.16%。原HF单EOS d=+23.08903143、同公式A=.75。
主导紧邻target71093为```，p由事实.998526126降至单EOS7.3486e-10；该token有
强影响潜力，但原joint负方向与事实single删除不符，不能用格式解释掩盖方向。
Text Format旧极值原joint A=-22.81693；native=-8.15750，方向有据而幅度夸大。

各任务选负prior系数平方和最大一个真实B4，检查每行最负/最正/负中位位置，
12位置各任务，共4次native paired B8前向/rank；未改原模型/scorer/ids/targets。
两个rank所有target向量逐值相同，恒等对照0、所有更早target差0、事实端点跨模式
逐值相同。Text6/12、App4/12方向差异只描述刻意选择的样本，不是总体错误率。
所选各行最负点覆盖保存负prior平方和44.12%/77.41%；并非任意抽例。
新native与保存formal事实端点的漂移完整保留；不以rank一致/恒等控制冒充官方
FA/FLA容差或个体反事实精度验收。作者累计删除/RISE/MAS证据仍保留、未被替代。

无新参考token、标签、clip/倍率/公式/PPO/白化改动。原OOM候选79922486保持
故障B4+原async-vLLM共存验证，物理55.716/55.290GiB、signed/QVA逐值一致；
未正式部署，未扩称精确32768/整轮28批通过。信用方向/幅度问题仍未修复，更新继续hold。

一次Text提交检查误用了App runner SHA，启动前即拒绝；改为核对每任务自身原source
绑定后启动，没有换Text的runner。保存失败命令/错误，不混成模型/环境失败。
函数inspect.getsourcefile读到no_grad装饰器Torch contextlib的元数据原样保留；
另用同原环境import模块及inspect.unwrap绑定真实counterfactual0d3412b8，未改计算。


## 2026-10-07 23:16 实际输出端/decoder27拆查完成；FA原断言通过，未放行训练

诊断代码aeaa1352，results_extreme_operator_20261007.json绑定原source58209daa、
真实B4 aaa03be7、脚本SHA、导入owner、配置、PID出生和原始张量运输SHA。
PID200641/birth1791385210.51已完成退出，4/5最终各859MiB；Text PID2833207同
出生时间仍hold，两rank无release；App保持终止，不恢复检查点、不推进optimizer。
两个原DT调用无observer，无新增模型前向/梯度或公式改动；四项完整signed向量
与此前原调用逐值一致，maxabs0。大张量约5.45GB留远端并存SHA，未搬本机/Git。

真实换行单EOS native cached效应+20.71933239，而原joint head seed作用于同一
single-EOS logit变化为+7.98411942（FP32）/+7.98411319（FP64）。原joint seed
在其自身两端点的FP64收缩100.45795617519036，与LP差100.4579561751856接近；
这说明其联合端点恒等式与逐token单删除效应是不同检查，不能把前者当后者。
主导紧邻code-fence target原生single效应+21.33316231，joint seed方向分配仅
+6.65016737。事实logits与normalized-hidden端点跨两调用逐值相同。
原BF16 head转置后single方向收缩+8.51523615，final norm后+7.61414271，分别保留。

完整decoder27输出+.99628511，MLP系数对single post-norm输出差-.50737119，
post-norm/residual输入+.86740143，attention后+.51147562；input-norm/residual
合并后-1.58704043，其中residual branch为-2.22394793。两个norm原FP32 eager
与实际compiled系数最大差7.45e-9；FP64单方向仍-1.58704043。原finite与Torch
functional.rms_norm的连续FP64 joint端点恒等式残差均1.42e-14。此为描述性核算，
没有新容差/守恒倍率，不认定norm或FA公式因整层反号而错误。

复用未改verify_saved_fa_dtypes SHA7ff11d9d和FA v2.6.3原源码SHAa290e11c的
AST断言：真实row0 Q6815/KV9567、BF16 QKV/FP32保存upstream、coefficient_start
2762，原生out/dq/dk/dv和coincident finite dq/dk/dv共7项通过；out2x、grad3x
门槛未改。因未保存v1，明确检查原reference q0/k0/v0而非冒称factual；统一实际
row query_start2752只做既有scalar接口表示转换，无缩短有效上下文。
该结果不提供非零有限归因的误差界、不覆盖旧其他操作数的BF16 dk失败。
attention参考自身调用原算子autograd，但没有模型训练反向或更新。

可选CPU HF norm导入触发device runtime初始化，原no-CUDA-init guard拒绝该次
辅助结果；原脚本/stderr保留。完成的核算改为保存张量、原DT及Torch RMSNorm，
CUDA未初始化，不将失败尝试写成成功，也未采用自行抄写的norm前向。

证据当前指向实际joint-reference分解与事实single-token删除效应的大幅差异，
发生在reward指数变换/白化前；不能提高核算精度或守恒纠偏后宣称已解决。
PLAN精确假设及Q/V/A/PPO均不变，不撤回理论前提，也不以近似误差代替接口正确性。
未称credit已修好或允许更新。原OOM修复79922486仍有界验证：原失败B4、原async
vLLM共存、最长27334、物理55.716/55.290GiB，signed/QVA逐值一致；未正式部署，
未扩大为精确32768/整轮28批验证。此次诊断采样PSS10.46/8.72GiB，最终容器141.65GiB，
不是物理峰值或内存泄漏结论。代码/证据/部署状态逐项分开记录。


## 2026-10-07 22:49 极端换行的原DT逐层诊断完成；未改信用

诊断代码d07de9e8，results_token_layer_20261007.json绑定真实导入路径/SHA、配置、
运输SHA和PID出生时间。PID69498/birth1791383985.22已完成退出；两rank各两次
原DT，零采样/反向/optimizer/检查点恢复，GPU4/5最终各859MiB。
Text原PID2833207同出生时间仍hold，无release；App正式任务保持终止。

同一实际App B4，single-EOS及原joint-EOS完整signed向量分别与此前无观察调用
逐值相等（四项maxabs0）。未传observer，原有限函数及返回对象保持不变。
被动记录原joint系数与真实single-EOS隐藏状态差的33个边界收缩；两rank逐值
相同，所有事实端点逐值相同。输出head+final norm之后为+7.61414271，
decoder28输入+.99628511，完整decoder27输入-1.58704043，最终输入-3.12821981
恰好等于原joint该token分量。cached single原生root为+20.71933239。

该收缩仅用于定位：输出端已发生较大差异，第27层还包含MLP/norm/residual/FA，
不能把整层变化认定为FA错误，不能冒称新的信用或容差验收。原完整single-DT
+20.98582570、原joint-DT-3.12821981分别保留；累计删除RISE/MAS证据未替代。
额外native-root/cache hook因原root实际走model.__call__而未取得数据，明确记录
capture_available=False，不称identity-row漂移已定位，不为这项可选缺口重复长跑。

CPU隐藏快照每rank峰值3796058112字节，已全部释放；采样PSS最大11.06/11.23GiB。
这些数字不是物理VRAM峰值。原OOM候选79922486仍只在原失败B4、原async-vLLM
共存下有界验证，未正式部署、未扩称精确32768/整轮容量或信用已修复。
当前继续只读拆查已定位的输出端与decoder27真实操作数；保留原FA/FLA门槛，
不加守恒倍率、裁剪或更改Q/V/A/PPO/白化及训练参数。


## 2026-10-07 22:26 实际动作target累计删除完成；极端信用和显存结论分别保留

诊断代码d0f2ebf4，汇总results_action_curve_20261007.json。PID4098367/
birth1791382409.93已正常退出，物理4/5各859MiB；Text原PID2833207同出生时间仍
在首次更新前hold，两rank无release；App原正式作业终止，未恢复检查点或重启。
正式source2796233e/58209daa、LoRA8/16、实际每卡B4、信用/PPO/白化均不变。

复用作者原faithfulness_test_skip_tokens，SHA583f4b7d，默认k20；由其负责排序、
累计删除、density、归一化和RISE/MAS。薄接口传原IDs、1266个非连续source位置
及原联合动作Y的标量分数，原PackedAnswerTargets/NativeTargetLogitRows/selected
target logp负责评分。明确属于联合Y目标适配，不能称为原单suffix evaluator未改。
target/O/其他三行不动，不生成奖励标签；positive-only仅为MAS评测视图，不进入信用。
两CPU接口测试仅证明原ID重组及source/target分离，不证明模型或数值正确。

两rank各42次paired原生前向，含初始化450.779秒，曲线和作者返回值逐值相同，
twins差0、其他三行分数恒定，factual/all-EOS分数与前次原生端点相同。
signed-input RISE=.31342148，positive-input MAS=.48193712；单条轨迹，不增设
门槛，也不把分数作为FA/FLA验收或整体归因质量结论。
最后signed组实际改变63个token，原DT值全负、总和-7.29023080，但删除后原生
joint logp下降15.72953576；异常换行在此组。此为组上下文中的效应不匹配，
补充前次单token真实上下文反号证据，不把组效应冒称逐token真值。
早期高正值排序删除仍显著降低目标分数，不能从该异常宣称全部归因无效。
原始非单调分数及作者running-minimum/penalty数组分别保存；评测变换未修改信用。

此前实际Text极端“ Format”原生单删除仍有负向概率作用：删除/事实比9.1575，
原DT估计23.8169，故负号有依据而原幅度未获此对照支持；App换行真实单删除
则支持代码生成，与其联合DT负值不符。两例不混同，不按词义或符号先判毒性。
完整批次极端白化系数的平方仅占有效系数平方和2.7603%，不是参数梯度占比。

新增existing-numeric-scope.json仅拆已保存数据：single-source全局root20.57395893、
head+final norm20.89706807、signed20.98582570；前者差.32310914、decoder累计差
.08875763分别保留。聚焦行root20.71933239；三个cached identity-row差
-.13623197/-.28043370/+.27129222不隐藏或纠偏。native replay与保存checkpoint相同
不等于官方容差验收，全局守恒残差也不提供联合向量各分量的误差界。
原signed反号在指数/白化前已经存在，当前未声称信用估计已修好。

OOM存储修复仍为79922486：原失败B4在原async-vLLM共存下完成，实际最长27334、
物理峰值55.716/55.290GiB，完整signed/QVA与已有offload同批逐值相等。
已定位GDN捕获增加约27.14GiB、已消费FA缓存留存、原HF/PEFT多个4.499GiB临时量
叠加；修复复用已有mixer卸载并释放已消费的原HF层cache。未正式部署，未扩大为
精确32768/整轮28批容量或训练健康已验收；本次曲线没有重复DT或推进更新。
原始结果、导入路径/SHA、effective config、运输SHA及物理资源图在
direct-target-action-author-curve-20261007/v1/；每项代码与证据分开绑定。

## 2026-10-07 21:51 极端token参照上下文的真实反号已测量

诊断代码c61c6772，汇总results_reference_context_20261007.json。原Text PID2833207/
birth1791370325.16仍在首次更新前hold，两rank无release；原App PID2786671终止，
正式source SHA仍为2796233e/58209daa。未重启、恢复检查点、推进训练或改变信用公式。
物理4/5诊断PID3902140/birth1791380517.62已完成退出；最终4/5均859MiB。

App原实际极端token198，Code:后的换行，response4498/packed7260：前次原模型
单token EOS端点在真实上下文中给d=+20.82963333。本次复用同一个真实B4、原
VERL fresh actor/LoRA8/16、HF/PEFT/FA/FLA及原目标选择器，将原1266个
prior-source设为EOS，只恢复该token，原生端点给d=-5.32932580。
两种上下文的边际影响相差26.15895913。紧邻代码围栏target52451的贡献分别
为+21.50193501和-3.26080132；确认上下文交互足以造成符号反转。
原联合DT分配为-3.11176191，不能把联合参照分解的负贡献直接解释成真实
上下文的单token删除效应；但也不把这两种边际任一项冒称原联合有限分配值，
或把全部有限估计误差归结为一个原因。这是具体估计问题，未否定PLAN精确假设。

两rank原生结果相同，其他三行差值0，更早68个target差值0；目标和原token
身份保持，LoRA_B本地分片全0。BF16原生logits/FP32目标logp/FP64累加保留。
旧缓存事实root与完整原生前向的-.28363漂移另存，不纠偏；本测量不是FA/FLA
容差验收，亦不替代累计删除/RISE/MAS。CPU8项通过（3项新增真实运输、5项复用
原端点接口），只证明运输/target身份。原始分数、有效配置、导入SHA、PID出生、
逐文件运输SHA和最终状态在direct-target-reference-interaction-20261007/v1/。

此前79922486的显存候选仍只完成原失败B4在原vLLM共存下的有界验证，未正式部署；
其55.716/55.290GiB物理峰值和signed/QVA逐值一致证据保留，不称信用已修复或
精确32768已验收。本次仅补原生端点对照，DT/rollout/反向/optimizer均为0。

## 2026-10-07 极端token已做真实端点对照；失败B4显存修复通过原vLLM共存验证

代码79922486，汇总results_extreme_credit_memory_20261007.json。正式Text PID2833207/
birth1791370325.16、source2796233e不变，首次更新仍Event hold，两rank无release。
App原PID2786671终止，未恢复检查点、未重启正式训练。最新物理4/5回到859MiB；
本次诊断只使用4/5，2/3的Text不推进。当前snapshot另存于
direct-target-native-mlp-memory-20261007/v2/runtime-snapshot-20261007.json。

两条实际极端source均由原VERL初始化fresh LoRA8/16、每卡B4，原HF/PEFT/FA/FLA
前向对照，仅一个真实token换EOS，其余三行identity及更早target差值均为0。
Text的token14606“ Format”、response351/packed666：原d=-3.17039663、rawA=-22.81692696，
单EOS原生端点d=-2.21457285，删除/事实概率比9.1575，对应样本系数-8.1575。
DT原估计比23.8169；负信用有实际概率影响依据，但原幅度未获该对照支持。
主要变化落在官方parser实际执行的首段引用动作内的“ format”target，
其原生logp由-2.250018变为-.029680；不是凭“格式词不重要”下判断。

App的token198为Code:后的换行、response4498/packed7260：原d=-3.11176191，
预期rawA=-21.46058（正式App未完成QVA，不称已消费优势）。原生单EOS端点
d=+20.82963333，紧邻代码围栏target的logp从-4.634799降至-26.136734，
说明该换行支持代码生成。另在同一prefix/正式producer/runner内仅改变该token，
DT signed=+20.98582570，缓存原生root=+20.71933239，残差-.26649331；其他位置signed=0。
同一诊断多source原接法仍给该位置-3.12821981。定位到联合有限分解对单删除效应的
估计差异；非指数/白化造成原始反号。残差如实保留，不加纠偏，不将守恒flag或
单例诊断称作FA/FLA整条验收，亦不替代累计删除/RISE/MAS的整体评价。
完整前向与旧缓存root的事实漂移Text+.06321、App-.28363另存，未强行对齐。

OOM已按实际phase/storage核查：GDN原GPU捕获先增加约27.14GiB，随后原HF MLP
产生多份[8,24572,12288] BF16约4.499GiB张量，最终原PEFT乘scaling分配失败。
此外，逐层有限消费结束后原replay cache仍累计保留FA K/V；末批rank0/1末尾
7.3767/6.3305GB。不是白化阶段，也不是只修改报错行即可解决。
先复用现有offload_replay_mixer=True，再在已完成原生重放和有限消费后替换该层
为原HF空DynamicLayer，释放该本地replay cache的已消费K/V。默认offload关闭路径
不变；不重写MLP、FA、FLA、cache状态机、PPO或信用公式。
候选runner7d6f57f61ecd、environment cd28a6e21401；原runner628006b63751保留。

v6诊断PID3714027/birth1791378722.34已结束：复用原create_colocated_worker_cls/spawn、
AsyncActorRolloutRefWorker及AsyncLLMServerManager初始化、wake/sleep；原vLLM共存。
原最后失败B8按每GPU真实B4完成，实际最长27334，180.795/180.363秒，物理mx-smi
采样峰值57053/56617MiB（55.716/55.290GiB），worker PSS峰值8.19/8.14GiB。
与此前原offload-only完成的同一末批相比，完整signed及Q/V/A均逐值相等、maxabs=0；
短B4的offload关闭/开启也逐值相等。未增设或放宽容差。原生/有限算子数学未改，
这项实际向量对照只证明存储生命周期修改保持本次输出，不证明单token估计准确。
原offload-only actor-only末批181.554/181.147秒；初始化/共存条件不同，不宣称严格提速比。
已消费cache末尾降至213909504B；物理图与逐层storage图在candidate-analysis-v6/。

验证范围是实际失败末批及真实短B4，不冒称精确32768、整批28次或正式训练质量已验收。
候选verification.json明确bounded验证通过但未正式部署；prepared原记录保留不覆盖。
v2/v4/v5仅诊断初始化失败、未进入DT，原因及来源保留；v6使用原trainer组合方式，
未加vLLM executor/name registry替代实现。CPU接口5+8项通过，不作为GPU算法证据。
所有原token向量、源/导入SHA、PID出生、transport及有效配置随汇总绑定；未裁剪信用、
放行Text更新、恢复旧检查点、启动备份或其他任务。

## 2026-10-07 19:41 极端优势取证优先，Text更新前hold，App末批原生OOM终止

Text同一PID2833207/source2796233e已完成首次DT22/22，两rank原raw readout和pre-update
实际落盘，Event hold/actor_saved=true，release均不存在，无本次optimizer step。
完整48文件CPU分析16.444秒、peakRSS1155891200B/CUDA未初始化，源文件前后SHA/mtime/size不变。
原raw A最负及最大绝对为-22.816926956176758；UID6e76f70e-bdeb-4726-8b35-d9e21eff2f68，
rank1-readout row77、response351、packed/input666、token14606，prior source非self。
原native FP64 d=-3.1703966315267027，实际FP32 d=-3.170396566390991，Q=1、V=23.816926956176758。
actor rank0 row5/response351/input863原QVA三项与该readout逐值一致；
实际整批白化A=-140.34095764160156，也是white最负/最大绝对，唯一tie。
原joint factualLP=-3.905878253712217、all-prior-EOS LP=-265.1357741963493，
它们不是该token单删除端点；不能用全EOS端点替代个别token反事实可信性检查。
raw/white均有限不等于极端估计可信，不据此放行更新。white绝对值>=100一个负token、>=50两个负token，
统计二阶量不称参数梯度占比。164个唯一artifact候选rawQVA全相等；6个DP补齐重复候选
均至少一个与actor全相等，保留全部候选，不将另一个补齐副本差值误报scatter错误。
86个无非零trace候选的actor原值保留，不补造其d。

App同一PID2786671/source58209daa最后第28批在原HF/PEFT MLP再次OOM，
19:41:29 driver/两worker均NoSuchProcess，GPU4/5各859MiB，原完整DT/QVA/PPO未完成。
已保存完整原input、前27批native原向量及原栈；没有重启或恢复旧检查点。
实际失败replay_call890/layer6：up输入[8,24572,4096] BF16，up.base_layer和LoRA_B
输出各[8,24572,12288] BF16=4831051776B(4.499GiB)。LoRA_B成功返回后
原PEFT layer.py:807的乘scaling申请额外同尺寸4.50GiB，设备仅余2.90GiB。
同时gate/silu各同尺寸已有storage证据。不是FP32猜测，也不是白化阶段；
前缀接线修复并未解决该原生MLP峰值，不称容量已修复。虚拟100.08GiB不作物理占用。

App前12批96工作行/121077 prior tokens CPU读取3.043秒/peak681254912B，无非有限。
原最大预期|A|候选为rank1batch6/UIDb98de8ee-995e-49e7-acb8-1510b4433a6a，response4498、
packed7260/input7288、token198，原r=1。官方既有tokenizer确认是Code:后、代码围栏前的换行。
原FP64 d=-3.1117619098301255；以冻结原counterfactual函数0d341作CPU诊断得到
预期FP32 A=-21.460580825805664，不冒称失败App作业已消费/已白化的A。
同一token只有单删除原模型端点对照后才能判断这种22.46倍概率影响是否可信。
当前优先Text实际极端及App该候选；单例诊断不能替代作者累计删除/RISE整体归因评价。

证据见results_prefix_runtime_20261007.json、textcraft-credit-cpu-complete-1791373053/
actual-complete-summary.json、native-credit-cpu-first12-1791372102/及
appworld-terminal-native-shapes-1791373289.json。两组原信用/PPO/白化/任务参数未改；
LoRA8/16、每卡B4保持。未release、信用裁剪/纠偏、备份或其他任务提交。

## 2026-10-07 19:21 首次DT逐token取证进行中，未经检查的首次更新未放行

19:21:42同一App两rank均完成12/28个B4，最新length11515/11506、42.811/42.813秒，
各12个native原文件；未见本次新OOM/原native异常，完整readout/pre-update/hold尚未出现。
Text完成25/30交互、当前26/30 active88，尚无本次DT/PPO更新。原更新前Event等待已安装，
没有release。物理2/3=49924/49854MiB、4/5=35610/35730MiB，容器253.244GiB。
本次取证优先，不能用采样进度代替数值检查；不重启或推进未经检查的更新。
原snapshot paired-first-dt-readonly-1791372102.json SHA cb541544940302ee126941147621f94357a12c25b0ac15998d79c9277e18e257。

CPU分析器0dd2ab8f新增读取已经保存的独立native-batch，修正App裁尾到完整artifact的映射；
只消费原数据、不生成FP32 d/QVA或重算白化。实际schema接口8/8通过，独立只读review未见阻断，
真实首批文件零分析错误；这是诊断接口验证，非DT数值/反事实准确性验收。
下文19:15首批统计及对应dtype观测仍仅覆盖首B4，不冒充12批统计或完整actor结果。

同一AppWorld PID2786671/birth1791369896.67/source58209daa，物理4/5；TextCraft
PID2833207/birth1791370325.16/source2796233e，物理2/3。没有重启、恢复检查点或额外模型调用。
App原采样已返回225条轨迹、3377个responses、639079个policy tokens，最长context27334；
原old-logprob之后，两rank19:13:17激活已安装的首次DT观察。各保存103574211字节完整原DataProto，
两rank第1/28个真实B4 trace完成，length8433/8188，原调用40.3047/40.2788秒。
各首native文件4900067/4811107字节，经既有SCP复制并核对远端/本机SHA一致；
不是新DT、额外采样或重建token。完整readout、pre-update与hold在19:15:02仍未出现，未release。

本机CPU只读首native文件：8条真实轨迹共7680个prior source，原signed为FP64，
范围-1.4845290905653894至7.370119502443012，未见非有限值；1156个负值。
保留原UID、response槽、token ID与对应原奖励；负值数量和二阶量不作为梯度占比，
也不单凭极值宣布整体归因差或训练健康。这是native原结果，尚无完整读出消费的FP32 d/QVA及白化actor输入。

被动原MLP记录的首批最后layer0：up LoRA_B实际输出[8,5681,12288]/[8,5436,12288]，
均BF16，逻辑1116930048/1068761088字节；down base及LoRA输出也为BF16。
不能把旧OOM临时量直接猜作FP32；此为已完成较短首批，不能证明最长joint容量或末批OOM已消失。
19:15:02物理2/3=49868/49606MiB，4/5=26834/26114MiB，容器252.061GiB；
原torch虚拟allocator另存，不作物理峰值。Text已完成16/30交互，当前17/30 active99，未到首次DT。

当前冻结App奖励owner7/7 SHA与原cached-reward审计一致：reward_extra_info KeyError
回退同一EpisodeRewardManager的同一rm_scores tensor，不重跑环境或重算奖励。
这是源码与缓存返回契约核对，非实际奖励tensor全值验证；不新增奖励管理器。
运行/运输源见results_prefix_runtime_20261007.json、paired-first-dt-readonly-1791371702.json
及first-native-artifacts/transport-1791371828.json。18:28 prepared-only快照已明确标为历史，
不与现部署混淆。公式、官方容差、任务参数、LoRA8/16与每卡B4未改。

## 2026-10-07 19:03 两组新正式采样，首次DT原调用观测已安装

原submit owner806651已从基础权重启动AppWorld PID2786671/birth1791369896.67/source58209daa，
物理4/5；TextCraft PID2833207/birth1791370325.16/source2796233e，物理2/3。
两组冻结prefix接线实现6469fa432e5eed381139dab86292fc92493e7f6e已实际部署：producer2aa5，
App readoutb602，Text readout814c。各自原DT/VERL/任务owner保留，未将App628/1c58/head1e209
替换到Text5f14/047c/headd473。原CPU导入及配置对照分别核1699/1640绑定；差异只在输出目录、
entry路径及App同源dataset symlink路径，预算/生成/白化/PPO/LoRA8/16/B4每卡不变。
无检查点恢复、备份、SQL/GRPO提交、重装或缓存清空。PLAN未改。

观测实现56e4d778b222b3ab4dd64500e810cfa9812e65fd，Text准备helper40dc2a7a6d1bf289fab1111010406e17a7603acc。
两个controller均已由原WorkerDict RPC恰好提交两次、正常结束，两rank实际armed；原worker e5eb、
actor3a65、LoRA8/16及B4已在原worker核查。v4首次原attribute入口保存完整DataProto，再调用同一次原方法；
v2逐批保存native signed、完整读出保存实际FP32 d/Q/V/A；原更新入口保存scatter和整批白化后的输入并Event等待
显式release文件。没有STOP Ray、额外模型/DT/更新或数值修改。v4的armed.owners同时列出继承这些方法的ref对象，
不是重复提交；激活匹配实际producer，v2只对拥有direct_readout的原owner安装。

独立CPU15/15通过，范围仅接口/原AST/绑定与异常恢复。旧保存B8每行只有当前response，不能代表多轮joint；
新完整输入在首次DT之前保存，19:03尚未产生。两组尚在原采样、无本次完整DT或PPO更新，不称训练健康、
32k joint容量或原HF/PEFT MLP OOM已修复。原MLP hook只记录shape/dtype/storage及allocator元数据，
不保留张量或开启runner observer。Text/App head只差128目标行分块，没有遗留BF16端点倍率纠偏。

19:03:24 App两rank采集80/120、82/120，Text已完成4/30交互、当前active206。
物理GPU2/3=49811/51345MiB、4/5=55290/55226MiB，容器214.772GiB，是采样阶段快照；
各worker PSS单独保存，不累加fork RSS。无本次DT shape/dtype现场，原MLP API只读审查确认：
官方apply_chunking_to_forward存在但当前Qwen/capture没有可直接启用的完整接线；没有擅自套分块或改dtype。
完整源/commit/配置/原RPC回执见results_prefix_runtime_20261007.json及对应current_runtime快照；
极端信用的逐token证据和原native MLP峰值仍待此次原调用，不能用启动或旧汇总值代替验收。

## 2026-10-07 18:28 前缀接线和原调用观测候选，仅本机接口验证

已确认新DirectActionTargetReadout漏接既有prefix工厂/provider，原自动common-prefix仍存在。
候选实现提交6469fa432e5eed381139dab86292fc92493e7f6e：分别基于Text31e2/App7900实际冻结源码补同一薄接线，
不混入另一任务的准备/裁尾行为；cut不越首个真实变更与最早target predictor。
原factory/capture/cache、Q/V/A、mask、B4、白化、PPO、LoRA8/16及容差未改。
本机独立prefix接口12项中8过/4缺真实依赖skip，不能冒充GPU缓存、DT数值、32k容量或OOM修复。

首次DT被动观测v3复用v2，挂原producer惰性创建后首次attribute入口；同一次原调用，
更新前Event等待，不再STOP Ray。独立CPU12过；测试已从旧模板纠正为实际e5eb owner原AST，
保留host-cache finally但CPU该CUDA分支关闭未验。原生MLP被动hooks记录base/LoRA实际shape/dtype/storage；
CPU6过，只是hook/绑定/异常恢复接口。失败29批原shape/dtype/layer仍未知，
未证实接prefix即可消除原HF/PEFT瞬时峰值，不把有限MLP分块算成native覆盖。

全部prepared-only，独立候选不在默认launch/patch路径。本轮无远端调用、模型、GPU、采样、
DT/optimizer、重启、检查点或备份；两组最后终态及17:55:52物理快照不改成新观察。
完整源SHA、两版patch、测试与范围见results_interface_inheritance_20261007.json；PLAN未改。


## 2026-10-07 17:56 数值检查优先；TextCraft错误暂停导致终止，AppWorld新原生重放OOM

用户要求先查极端信用，不继续拿训练进展替代逐token证据。TextCraft原PID110053/
source5013完成7次更新；第8次在ref阶段尚无DT/actor更新。父代理17:31对原进程树
SIGSTOP，17:37仅恢复基础设施/workers，仍暂停driver/TaskRunner。这是错误的调试暂停：
原GCS17:40:04判停止的driver控制连接不可用并销毁TaskRunner，raylet17:40:09
SIGKILL并级联回收两rank。原区间日志已保存；不是DT OOM，不能称仍安全暂停。
观测器安装前检查即因TaskRunner已不存在退出，未在worker导入或改变数值。
17:55仅恢复已失败driver接收原Ray错误/清理；17:55:52全部原进程已无，未重新提交。
旧−23/−110等汇总极值缺少原位置/d，不能恢复配对；第8批未保存现场也已丢失。

新增被动观测v2仅prepared，CPU8/8接口测试通过，未部署/未证明实际Ray心跳或DT精度。
每个原trace成功返回立即保存FP64 signed、case/UID/原IDs/位置/原detail；完整读出再保存
实际FP32 d与原Q/V/A；原actor更新入口保存scatter及整批官方白化后的DataProto。
显式释放文件前Event.wait等待，不再SIGSTOP Ray进程；原公式/返回/exception/参数不变。
v1错误暂停候选及失败安装记录保留，不在默认启动路径。没有模型重放、额外采样、
信用裁剪/纠偏、检查点导出/恢复或新增数值验收标准；真实向量采集尚未完成。

AppWorld原PID2001805/source24b9本次在第29/29批原HF/PEFT MLP重放up_proj LoRA
乘scaling申请4.49GiB时仅余3.03GiB，完整DT/PPO未完成；失败批length/layer/dtype未记。
此前有限MLP分块628006/1c58仍是该有限边界的已对照实现，不覆盖此次原native MLP。
与旧稳定sourceaa8fac/checkpoint28比对：B4、LoRA8/16、FSDP参数/优化器卸载、
activation_offload与vLLM资源配置相同；官方offload及worker文件SHA未变且sleep有原日志。
旧为辅助类别标签EventRatioReadout，新为真实joint action target；旧显式lease工厂/provider
未接入新DirectActionTargetReadout，但runner自动common-prefix仍存在，不能说前缀全部关闭。
该遗漏和新增原生重放峰值分别调查；未证明仅接lease就能修峰值。错误早于优势白化。
官方activation_offload原eval分支直接原forward，不等于能消除DT eval前向的MLP临时量。
不能用旧标签长期运行、旧短target32k或有限MLP对照宣称新joint整链容量已验收。

本次完整来源/SHA、源码行号、prepared状态和终态见results_token_credit_debug_20261007.json
及old-stable-to-joint-memory-path-audit-20261007.json。17:55:52物理所有GPU858MiB/no processes，
是退出后快照非峰值；两组均未重启，SQL/GRPO未启动，方法PLAN未改。


## 2026-10-07 17:14 AppWorld首次正式DT推进，末批尚待实际完成

本goal turn为verified wait：重新核对同一存活PID/birth/source，而非依据旧清单推断。
AppWorld原PID2001805/source24b9两rank8/29个B4完成，最新length10080/10106、
单批36.0134/36.0091秒。原采样保留232条、3748 responses、671515 policy tokens，
max_context27290；231个非零请求不能当作成功率。旧失败在末批，本次尚未到末批，
完整DT/PPO和当前联合目标32k容量仍未验证，不把阶段推进称为已修复。
原reward相关7文件SHA与此前cached-rm_scores官方路径审计相同；原可选metadata
fallback未改变，没有新缺奖证据。当前回报分布未落盘，留缺失，不从DT计数推断。
17:14物理GPU4/5=31060/30858MiB，容器约235.60GiB，均为阶段快照非峰值。
TextCraft原PID110053/source5013仍第8轮采样，17:14完成13/30交互、进入14/30，
active123；最新完整仍7轮。未重复旧metric冒充新更新，未新测试/部署/恢复/改参数。
最新原只读phase及来源SHA绑定在current-formal-phase-observation-1791364455.json、
current_runtime和结果索引。原模型/LoRA8/16/actor与DT每卡B4保持。

## 2026-10-07 17:01 TextCraft第七次完整更新与信用尾部计量

前一goal turn为progress（真实MLP对照、AppWorld正式提交及版本绑定）；本次同PID/source
只读核对产生新完整step7指标，未重新提交或改生产代码/参数。TextCraft实际rollout7为
166个1分/90个0分、256条，训练成功率0.6484375；entropy0.651、grad0.034、PG−0.097。
原gen1839.132/oldprob125.917/ref109.901/DT175.327/actor457.612/完整2708.344秒，已进入第8轮采样。
7点曲线已用原plot_training_progress解析/绘制；只有step7的成功率来自实际保存score1/总数，
其他缺失成功率未补造；平均回报曲线仍为原console值，未混入独立评估。

step7原prior最小−110.061，原框架白化后最小−553.449，均有限。离线合并DP补齐DT
工作行的raw self/prior/other原始二阶量：最大单项占比41.023%，前6次最高3.374%；
这是原始工作行的统计，不是白化后或梯度占比，不从极值判定bug或退化，未做裁剪/纠偏。
目前成功率波动且熵稳定，未证明学习改善或长期稳定。AppWorld17:00原采集198/240条，
transport3637请求/664929生成token/1087.0秒；仍无首轮完整DT/PPO。容器229.134GiB为阶段快照。
来源current-formal-observation-1791363676.json、step7-complete原回执及steps1-7原始二阶量分析；
完整路径/SHA和图表在current_runtime及结果索引。范围仍只有两组DTPO，SQL/GRPO未启动。

## 2026-10-07 16:43 AppWorld有限MLP分块版本已提交，原生采样中

AppWorld新PID2001805/birth1791362313.39/source24b9e671已由原bind/submit入口
在物理GPU4/5从基础权重新开，resume=disable，无检查点导出/恢复。实际实现提交
d5b879d7acd49a5e7ba7550a52cb53a0f993d0d5；诊断提交4a98683445b33879f57bac6cf4872819458c5365、
绑定助手提交d3447bef0f71b2629a0d32512eb90c5728244e2c及后续账本提交各自独立。
本轮只将runner ba639b→628006与decoder047c→1c58替换为有限MLP token分块2048；
GDN448ef、head1e209、readout7900a、actor3a65及VERL/LOOP/entry哈希保持。
原launch仅四个输出路径不同；run-env仅DT_ROOT/PYTHONPATH对应目录映射，
任务配置、采样/训练预算、LoRA rank8/alpha16、每卡actor/DT B4均未改。

同一真实B4/卡的32次原有限MLP对照均通过原torch.testing.assert_close dtype默认值；
rank0逐位相同，rank1保留实际非逐位误差（最大约3.43e-7），没有放宽容差或加纠偏。
实测有多chunk及非整除宽度；这是该MLP同输入回归，不是FA/FLA/PPO整链、
32k联合目标容量、正式速度或训练健康证明。原FA/FLA断言及PPO未改。
旧PID1468126/source942在最后DT B4因buf7 OOM退出，旧源和终态收据完整保留。

16:42:56原LOOP transport已采样63个请求、生成9149token、调用累计61.2秒，
尚无完整rollout、DT或PPO更新；此时间包含生成调用，不称pure decode。
当时物理GPU4/5=55314/55298MiB，容器约221.069GiB，为阶段快照非峰值保证。
TextCraft原PID110053/birth1791344324.6/source5013、GPU2/3保持；最新完整更新仍6，
第7轮原采样结束后的处理在推进，未把旧step6当新metric。

完整active-training/active-source/formal/source/prepared/CPU-import/launch/verification
按原字节保存于direct-target-mlp-token-chunk-20261007/v1/runtime-metadata，
索引runtime-metadata-1791362591.json SHAe14454c4…c9efdf；阶段收据
formal-phase-1791362528.json SHAd4e4f6d5…58e201及
readonly-formal-initial-phase-1791362576.json SHA7ba6c49d…8e825bf。
完整路径/SHA和验证范围见current_runtime及results_direct_target_mlp_token_chunk_20261007.json。
本次账本更新仅一次原collector和小JSON只读采集，未改生产代码/任务参数或操作检查点。


## 2026-10-07 16:12 AppWorld末批DT显存不足，TextCraft第六次完整更新

AppWorld原PID1468126/birth1791357190.77/source942c2d68已退出，原终态收据确认
第29/29个DT B4在有限MLP编译产物buf7申请2.35/2.37GiB时OOM；调用链为
runner497→decoder228→decoder95 `_mlp_input_rule`→Inductor line286，非GDN或PPO。
两rank此前各完成28个B4，最大完成length22268/21672；失败批长度没有打印，
不从前批推测。没有完整DT读出、PPO更新或检查点，不称首轮成功或32k容量通过。
原PyTorch约103GiB allocator文字不作为物理显存；16:11:36物理GPU4/5为
64841/64341MiB，是退出前快照，不代表退出后资源状态。

TextCraft原PID110053/birth1791344324.6/source5013ebc8完成step6，随后第7轮原采样。
原rollouts/6.jsonl为256条、171个1分/85个0分、平均0.66796875，至此才把171
核对为本轮实际成功数；不是仅凭nonzero_requests推断。原entropy0.648、grad0.035、
PG loss -0.056，gen1771.923秒、oldprob121.314、ref105.839、DT156.021、
actor441.282，整轮2596.801秒。两rank原第6份DT报告均88工作行/22次B4，
读出154.039/154.009秒并行；self全1，prior范围[-3.06927,1]/[-10.20562,0.99999988]。
原step6完整metric为TaskRunner113691.out第333行，完整报告及逐行SHA均保留。
这些是当前训练批次的观察，不证明学习改善、token精确归因或32k容量。

16:11:36 TextCraft物理GPU2/3=50204/50140MiB，容器236.290GiB；
主进程/TaskRunner/两worker PSS约5.140/19.230/8.432/8.323GiB，均为阶段快照。
此条及图表仅在本地复用原收据，未远端调用、改PLAN/训练参数、恢复检查点或重启。
当前部署实现仍d7af4eaed5c16aacb1730b81946fe89d282e40d5；账本/绘图提交不替代运行版本。
来源：readonly-DT-final-batches-1791360485.json SHA3f59488a…49fe0e9、
textcraft-step6-original-1791360668.json SHA5036ed57…ab772fe、
formal-phase-1791360696.json SHA288580f5…abc8fb8，完整路径/SHA见current_runtime和结果索引。



## 2026-10-07 16:00 两组正式DT推进，未重启或改参

前一goal turn为progress：原AppWorld完成前6个真实B4，原信用工作量与版本记录已提交。
本次为verified wait：按原PID创建时间/source SHA连续确认两组活跃，未把观察窗口
到点当成失败，也未重新提交任务或扩大测试。
AppWorld15:59两rank各20/29B4，length14360/14554、60.8726/60.8847秒；
16:00同源原日志已见21/29（最新rank1 length15234、65.7209秒）。
新完成13–20批原长度约11.7k–14.6k、45.1–60.9秒，未见新fatal/OOM；
15:59物理GPU4/5=43128/44388MiB，cgroup观测236.66–236.85GiB，
worker PSS约8.05/8.21GiB。不是连续峰值或最后长批次容量保证。

TextCraft第6轮30次原采样结束并经过原old log-prob；16:00两rank各完成DT4/22，
原工作量256轨迹、171个非零请求、19476个target token、658969个source token。
仍无step6完整metric，不把step5当新更新或把非零请求数直接当成功率。
同源16:00物理GPU2/3=17347/17027MiB，容器236.666GiB，均为阶段快照。
原缓存奖励metadata提示后已继续进入DT，未新增重评分、奖励替代或容差。

15:53原文件系统剩余353658470400字节（约329.37GiB），两组均尚无完成保存标记；
现有启动配置max_actor_ckpt_to_keep=2，TextCraft save_freq25/AppWorld save_freq1未变。
未恢复任何检查点，也未启动备份或旧SQL/GRPO。
来源：readonly-DT-batches-watch-1791359542.json、
textcraft-step6-observation-window-1791360034.json、formal-phase-1791360034.json、
formal-disk-and-checkpoints-1791359589.json；完整SHA保存在current_runtime及结果索引。

## 2026-10-07 15:48 AppWorld前六个真实B4完成

同一AppWorld PID/birth/source942已在两rank各完成6/29个联合目标B4。
rank1473507第5/6批compute length9684/9942、33.9729/35.4048秒；
rank1475058对应9749/9989、33.9677/35.3912秒，均来自原完成行。
未见本次新OOM；这不是最后长批次、完整DT/PPO或精确32k容量通过。
采样227条真实轨迹对应227个联合请求，不按3794个response展开；
分布式补齐后的B4工作行与原环境轨迹、actor批量仍分别记录。
同窗口TextCraft第6轮采样已完成19次交互、正在20/30；完整更新仍5。
容器236.691GiB为阶段读数，来源formal-phase-1791359322.json
SHAf85eb468…23591f3；未改训练参数、源码、方法或重启任何任务。

## 2026-10-07 15:45 AppWorld首轮进入真实DT

AppWorld当前PID1468126/birth1791357190.77/source942保持，原采样已经结束，
原loop_trajectory保留227条轨迹/3794个response、716788个policy token，
最长实际context28665。15:44两个worker均执行原actor_rollout_compute_log_prob；
15:45已转为原actor_rollout_compute_dt_token_advantages。DT工作量原日志为
227条轨迹/227个非零回报请求、193460个target token、523328个source token。
非零回报包含原trainer处理后的奖励，不能把请求数当成功数。
目前尚无本次B4完成行，不猜批次长度/耗时，不据此称完整DT、PPO或32k容量通过。
15:45原物理GPU4/5=27520/27420MiB，容器236.59GiB，均为阶段快照。
已有reward_extra_info提示仍走未修改官方cached reward metadata fallback，
当前source audit已核对其owner字节；未为该提示增加重评分或替代奖励。
来源formal-phase-1791359067.json SHAa42a0b77…4aeb1f及
readonly-first-DT-watch-1791358683.json SHAd48b95bc…e32e554。
TextCraft原第6轮采样已完成13次交互、正在第14/30次，完整更新仍5；任务/信用/PPO/参数未变。

## 2026-10-07 15:36正式进程复核及第五轮信用离线观察

前一goal turn为progress：实际绘制本次训练曲线、绑定第五次原更新/部署源码并提交推送。
本次继续当前用户接受的两组DTPO，不恢复旧goal文字中的已取消SQL/GRPO。
15:36同一TextCraft110053/source5013、AppWorld1468126/source942均存活且出生时间匹配；
TextCraft第6轮原采样推进至6/30，AppWorld原collection107+102=209/240，
transport740034 token/1196.8秒；容器229.606GiB，为阶段读数而非峰值。
来源formal-phase-1791358591.json SHA9d8c10e5…f10cef。

原TextCraft step5离线观察：137个唯一非零回报traj_uid在原DP补齐后形成144工作行，
7个重复UID；两rank各72行、18次B4，DT阶段142.329秒，两rank读出139.659/139.745秒
并行，不能相加或把144工作行叫actor训练批量。self原信用全1，prior范围两rank为
[-7.2645869,1]和[-23.0169945,1]，已有other组全0。已有2002个数值字段未见非有限值；
报告没有逐token向量，不能推断分布、错位或精确反事实准确性。白化后系数仍为原日志
[-164.145,6.917]，原PPO梯度0.034。工作量和公式入口已生效不等于学习改善。
来源textcraft-step5-credit-workload-offline.json SHA67709a57…4f560；仅离线读取已存
收据，没有模型、环境、优化器调用或参数/容差变化。

## 2026-10-07 15:27 TextCraft第五次完整更新，AppWorld新源码绑定核对

TextCraft原PID110053/birth1791344324.6/source5013ebc8不变，完整step5已返回，
随后进入第6轮原采样。原rollouts/5.jsonl为256条、137个1分/119个0分，平均
0.53515625；原console reward0.535、entropy0.665、grad_norm0.034。
整轮2726.901秒，gen1888.977、oldprob125.779、ref109.726、优势计算142.329、
actor459.605。白化优势[-164.145,6.917]、raw prior最低-23.01699如实保留；
没有裁剪或补倍率，也不能由五个训练批次宣称学习效果改善。
完整原日志行SHA e5399786bc2ef758af9986995420acf90f45b379710a20ec1d46f86b32466038，
textcraft-step5-original-1791358084.json SHA9a1cb076…e79fa。

AppWorld仍为PID1468126/birth1791357190.77/source942c2d68，原LOOP采样中，
尚无完整DT/PPO结果。两个实际WorkerDict的DT/VERL/entry/LOOP路径全部匹配；
45项当前文件SHA、15项已保存CPU导入记录对应当前冻结文件。仅runner/GDN
生命周期修复，FA/FLA、PPO、head/readout未变；B4/卡、rank8/alpha16、32768、
resume disable保持。审计readonly-source-audit-1791358028.json SHAac2c14ac…51a7c5。
源码绑定不扩大已保存数值验证和新joint target32k容量的范围。

15:27原物理显存2/3=50246/50234MiB、4/5=55312/55290MiB，容器226.81GiB；
均为阶段快照，不是峰值。原图脚本修复了旧任务名硬编码和双卡显示，新增原
actor/entropy_loss曲线；两项原日志解析测试通过，实际当前两组快照已渲染。
原日志未提供episode/success_rate时保留缺失，AppWorld完整指标也不补零。
图表/CSV/快照位于direct-target-gpu-lifetime-20261007/v1/plots/textcraft-1791344324；
仅观察和显示改动，未改训练，也未恢复旧检查点或SQL/GRPO。

15:34 AppWorld同一PID/birth/source继续首轮采样，原两rank collection90/120、85/120，
合计175/240、取消0。原transport为656007生成token/1048.3秒/3494 requests，
包含调度、prefill、decode和RPC，非纯decode或完整采样耗时。未进入DT，尚无
完整更新；本次日志尾未见新OOM/RuntimeError。原只读阶段回执
readonly-phase-final-1791358443.json SHA4f9b20ad…77bbd54。

## 2026-10-07 15:13 AppWorld生命周期修复正式提交，TextCraft继续

AppWorld由原submit_prepared_direct_targets从base新开，PID1468126/birth1791357190.77，
物理4/5，source942c2d686a701317e8441f5b299dd86dfb55deb7356c7427e62bc4cbdcd36488。
DT root为candidates/direct-target-gpu-lifetime-20261007-v1/deltatrace；冻结源码commit
d7af4eaed5c16aacb1730b81946fe89d282e40d5，后续提交不覆盖这个部署身份。
仅runner ba639b28/GDN448ef32c将已有最后消费者释放与CPU搬运开关分开；head1e209、
readout7900、原VERL/LOOP/任务/PPO、每卡B4、rank8/alpha16及所有计算参数保持。
CPU实际导入和原配置比较1677项通过，仅四个输出路径变化；正式原提交owner再核验1681项。
不恢复检查点，未改变TextCraft原PID110053，也未恢复SQL/GRPO。

修复前后在同一份真实原生捕获上，两rank各24次GDN输出bitwise一致，原PyTorch
assert_close默认dtype断言也通过，FA/FLA算子和既有官方断言未变。真实B8原上下文
11678-13153，实际GDN重放width1632/1125；这不是新joint target的32k容量证书。
对照约48.7秒，包含重复baseline调用/比较，不能当正式速度或物理峰值。
verification.json SHA2475457e866f812fea17b0915bde17eafa179735cd74e065d3187589bdf4ba65。
15:14原正式worker加载权重，尚未完成新rollout/DT/PPO，仍需观察正式长批次和保存。
当前不称OOM已全范围消除或训练效果已改善。来源在direct-target-gpu-lifetime-20261007/v1。

TextCraft第5轮采样已完成30次交互，原完整更新仍step4；四批奖励为
0.664/0.629/0.695/0.609、熵0.677/0.644/0.675/0.658。step4白化优势
[-115.301,6.544]、raw prior最低-17.0131仍如实保留，不仅凭极值判数值错误。
15:14全容器184.60GiB为阶段快照，非峰值；启动不等于完整更新健康性。
15:18同一PID/source组合已推进：TextCraft第5轮DT两rank15/18，AppWorld原
LOOP native_async已生成44382 token/113.1秒并有真实环境轨迹返回；该计数包含
调度、prefill、decode及RPC，不是纯decode或完整rollout。cgroup222.49GiB。
原source-bound快照formal-phase-1791357508.json SHA1b10108f…e600；新App完整更新仍未返回。

## 2026-10-07 14:52 AppWorld第29个DT批次GDN OOM；未重启

AppWorld PID996278/birth1791352696.13/sourceed3fdf7b已NoSuchProcess，物理4/5均释放至859MiB。
本次未完成首个新DT/PPO更新，rollouts/1.jsonl不存在。最后成功两rank28/29、length19511/
19340、99.735/99.709秒；失败第29批没有完成行，不能把前一批长度当失败批长度。
rank1003748在GDN253→原symmetric反向→finite_fla编译临时FP16 buf77申请174MiB失败；
rank1002185在GDN301→原decoder._linear_transpose72的BF16 cast申请1.18GiB失败。
这是GDN传播峰值，不是head；runner382在进入finite decoder前已删除out/z/seed/mnorm。
原OOM103-104GiB为虚拟allocator数字，不是物理占用；终态cgroup140802150400字节。
原栈/解析symlink/关键SHA见appworld-terminal-oom-original.json SHA
2ed338b9cbaf7bd088f2a7feaf56d315e5a30851c351f4c080855dd4614ea2ef。

发现当前dt_offload_replay_mixer=false把已有consume_captures也关掉，GDN最后消费者后
释放分支同样仅绑定offload_endpoints，GPU常驻路径把已用完的MLP/GDN捕获留到函数末。
只准备隔离生命周期候选direct-target-gpu-lifetime-20261007/v1/candidate：复用decoder/FA
已有consume接口，GDN消费释放与搬运开关分离；不改算子、公式、目标、B4、LoRA、任务或PPO。
runner ba639b28、GDN448ef32c仅AST语法检查，尚未完成数值/容量对照、未部署或提交任务。
当前正式TextCraft原PID110053继续第五次采样，已有四次更新；SQL/GRPO保持停止。
本节所有原终态与源副本回执位于direct-target-causal-prefix-20261007/v3，候选不属于
当前可启动/已验证版本。没有恢复检查点或为OOM直接更改内存/计算参数。

## 2026-10-07 14:41 TextCraft第四次更新返回，数值来源仍匹配

TextCraft原PID110053/birth1791344324.6/source5013ebc8完整step4已返回并进入第五次原采样。
整轮2808.834秒，gen1879.989、oldprob135.719、ref118.353、DT180.260、actor494.122、
dump0.383；reward0.609、entropy0.658、grad_norm0.049、白化优势[-115.301,6.544]。
actor实际954558 token、每token0.518ms（step3为0.508ms），不把总耗时增长判为计算异常。
四批采样reward0.664/0.629/0.695/0.609、entropy0.677/0.644/0.675/0.658，没有复现
旧实验的连续熵上升；这仍不是独立评估或学习改进结论。原完整metrics行SHA
7e453a6b3637cd76897109e17a9637a3c9d1bd2bd98934a1e67e5cab96f08ae1，回执
textcraft-step4-original-1791355304.json SHA4283b701d4a2ba2cd31b2cd19967b50277c61fd8502cb83aaa595ecfcc3e0aa6。
同窗口AppWorld原PID996278两rank为DT21/29，未完成首个更新；物理4/5为53812/54132MiB，
全容器235.31GiB，来源original-phase-1791355322.json SHAf49c585f…ef280；非峰值。

14:37:48只读核对两组PID/source/实际CPU导入回执、11个VERL owner文件、7个DT关键文件、
安装补丁和有限.so，当前字节均匹配各自source。runner5f14/vendorFA3e1d/GDNef55未变；
Text head d473与App head1e209明确区分，App readout7900只包含已记录因果前缀修复。
原FA v2.6.3断言仍out2×、dq/dk/dv3×参考基线误差；FLA v0.4.1的o/ht仍0.005，
没有放宽或纠偏。已有v4的56FA/4FLA仅覆盖原记录实际算子；head1e209为真实B4的
PyTorch FP32默认assert_close对照，不称FA/FLA/VERL整网容差；历史失败回执不当通过证书。
核对不是live Python方法反射或整网健康证明。source-bound-tolerance-audit.json SHA
ef67413ab8d1a1874c8aa1dd357d63b39ba66cc1419247aa6ca9ec58ffd8f221，均在
direct-target-causal-prefix-20261007/v3。生产、参数与任务范围没有修改。

## 2026-10-07 14:30 两组正式DT推进，原奖励缓存路径已核对

AppWorld原PID996278/birth1791352696.13/sourceed3fdf7b未变。原采样实际返回231条轨迹、
3390个response、635419个policy token，最大context24794；原完成条件不要求恰好216条。
transport累计688654生成token/1218.7秒包含调度、prefill和decode，不是纯decode。
原old/ref概率阶段后两rank已完成DT 7/29个B4；首批52.374秒，后续2-7批约27-36秒，
尚无完整新更新。没有因第一批冷启动耗时外推整轮，也没有重启、恢复或更改参数。
TextCraft原PID110053/source5013ebc8继续第四轮DT，两rank6/20；原完整更新仍为step3。
最新物理2/3为17329/17047MiB，4/5为37176/36862MiB；全容器cgroup235.54GiB、swap0。
App DT worker PSS8.81/8.12GiB，Text8.43/8.32GiB；这是阶段快照，不是峰值。
本次原日志尾没有新OOM/非有限值；来源bounded-original-pair-1791354644.json，
SHA848554303af2068fbbc150c43cabb8c86f18762a6eb3c03cf7839aa897bf99e9。

原reward_extra_info日志已按实际EpisodeRewardManager而非Naive核对：cached rm_scores
分支缺metadata，原compute_reward的兼容fallback再次调用同一manager并直接返回同一个
rm_scores张量；不进入decode、环境评分或归一化，不改值/重算/丢奖。原reward.py、
Episode和Naive源与固定20bd331官方字节一致，不为此加入包装或修改。
来源cached-reward-owner-audit.json SHA51aa40f76b4ac0d0a1c18b15042b08feb174eb3edd29ec9d15c49db5f15b0f65。
以上回执位于direct-target-causal-prefix-20261007/v3；current_runtime已刷新实际PID、
源码与启动绑定。代码仍为App139273c3/Textf29cc7c0，后续Git提交只保存观察记录。

## 2026-10-07 14:08 TextCraft前三轮原轨迹与DT工作量已核对

原PID110053/birth1791344324.6/source5013ebc8不变。原rollouts1/2/3.jsonl各256行，
成功170/161/178，精确均值0.6640625/0.62890625/0.6953125，与原actor完整metrics一致。
两rank DT unique traj_uid恰为170/161/178；DP分别补6/7/6至176/168/184，未将
response展开成新轨迹或重复奖励。raw self信用全1，prior两rank均值约0.055-0.065，
RMS0.144-0.173；raw prior极小值逐轮-23.0554/-19.0920/-5.6769，other_policy全0。
这些raw统计只覆盖非零奖励DT请求并含DP复制，不冒充原整批raw分布。
官方整批白化后原actor优势范围依次[-141.923,5.905]/[-127.002,6.421]/[-36.775,6.195]。
原熵0.677/0.644/0.675、梯度范数0.037/0.039/0.036；三轮不显示持续下降，仍不足以
判断改进有效。负信用频率、逐token原始/白化配对、原动作validity与终止原因未保存，
不重解析文本补造。DT报告无global_step，按持久worker顺序及unique数量对应前三轮；
不是逐token join。回执direct-target-causal-prefix-20261007/v3/textcraft-observations.json
SHA55341018…d1331。README仅更正旧任务范围/旧head的当前说明，没有新方法规范。


## 2026-10-07 14:02 两组原正式任务在运行

AppWorld PID996278/birth1791352696.13/sourceed3fdf7b已进入原native_async真实采样，
两rank原sample_tokens执行，最新transport为100calls、5494生成token/55.5秒；含调度、
prefill与decode，不是纯decode速度。首个新DT/PPO未完成，不把启动当健康验收。
TextCraft原PID110053继续第四次采样，已有3次完整更新。14:01物理2/3显存
49944/50400MiB，4/5为47730/47714MiB，全容器191.915GiB；非峰值。
来源见results_direct_target_causal_prefix_20261007.json及最新current_runtime快照。
实现commit139273c3，后续提交仅保存来源/状态记录；运行版本不能用新HEAD覆盖。


## 2026-10-07 13:59 AppWorld 因果右尾修复正式提交，TextCraft第三轮完成

AppWorld已由原submit_prepared_direct_targets提交到物理4/5，从base新开，PID996278/
birth1791352696.13/source SHA ed3fdf7b3468116b65c96c4df915a4c549c0d9d9e55f72725f805751b2071cc6。
源码commit139273c36f2546acf37cfab7612a22e2b54d2597，entry为
candidates/direct-target-causal-prefix-20261007-v2/appworld/entry；仅reward_readout.py
SHA7900a369…67b27改变。DT root仍head-memory-v2、answer1e209，VERL/LOOP/任务/PPO
不变。冻结1670项核验通过，远端原生产导入CPU37项通过/2.03秒；配置差异只为
四个输出目录及相同dataset文件的入口路径，没有算法/任务参数改变。rank8/alpha16、
每卡B4、200迭代/原40组×6及90%完成、原minibatch32/epochs2、32000训练/32768生成
上限保留。原完整执行史回填保留，只省去最后真实target之后的DT计算右尾。
当前仅原Ray/VERL初始化，未完成首个新DT/PPO，不称完整容量或训练健康已通过。
来源results_direct_target_causal_prefix_20261007.json；v1为未提交的CPU测试对象初始化
失败，原log保留；v2仅补测试构造状态，不改生产producer。原头数值回执仍原范围。

TextCraft原PID110053/source5013ebc8继续2/3，第三轮已完整返回，进入第四次采样。
step3整轮2589.735秒：gen1765.050、oldprob116.819、ref101.904、DT179.880、
actor425.637；reward0.695/entropy0.675/grad_norm0.036。三批reward0.664/0.629/0.695
均为各批更新前采样，暂不判断质量趋势。原token优势范围-36.775至6.195。
原日志/SHA见direct-target-causal-prefix-20261007/v1/textcraft-latest-original.json。
无checkpoint恢复、新备份、新自动化；SQL/GRPO保持停止。


## 2026-10-07 13:50 AppWorld 新终态与因果右尾打包候选

AppWorld PID670069/birth1791349552.74/source42bb0eb9已退出；此次不是OOM。
原采样216条完整轨迹/3358个response，进入DT接线时完整原记录32835超过32768，
在reward_readout检查处、首次DT模型调用之前报错。原PPO实际仍为216个完整轨迹行，
没有展开为3358行；每rank原32全局minibatch/DP2、epochs2应14次optimizer step，
本次实际0。记录中training截断32000与完整执行史不是同一个量，不能截DT到32000。
官方最后工具观测/终止消息会追加在最后生成之后。准备仅让DT计算到最后真实target，
保留完整原行、target ID/offset、reward、ratio/mask和scatter；尾部policy按既定d=0。
必要前缀仍拒绝超过32768，不提高上限或改变环境。当前仅本机CPU37项通过，尚未
准备/部署；失败真实轨迹未保存，不能称已重放修好或完整joint容量通过。
唯一候选见results_direct_target_causal_prefix_20261007.json。头1e209、FA/FLA、PPO不变。

13:50只读证据：TextCraft PID110053/source5013ebc8未改；第三次原采样完成，
两rank DT各23个B4完成、177.47/177.32秒，最新完整指标仍step2；物理2/3为
40646/35862MiB，4/5各859MiB，全容器129.63GiB。新前缀候选不改运行中的TextCraft。
来源direct-target-causal-prefix-20261007/v1/observed-before-submit.json SHA dc577177…96d38；
AppWorld原终态direct-target-head-memory-20261007/v3/terminal-original-1791351381.json
SHA c3c9b72f…b0b14。SQL/GRPO仍停，无checkpoint恢复或新备份。


## 2026-10-07 13:14 TextCraft 第二轮完成，AppWorld 正式采样推进

两组PID/birth/source仍为下节记录。TextCraft已完成step2并进入第三次采样；原整轮
2624.925秒（43.75分），gen1807.604、old_log_prob117.072、ref102.251、adv169.113、
update_actor428.475秒；原grad_norm0.039、entropy0.644、reward均值0.629。
该reward来自第二批更新前采样；与第一批0.664仅两点，不报告改善或退化趋势。
AppWorld原native_async累计157873生成token/284.9秒，尚未完成首个新DT/PPO更新。
整个容器cgroup237520596992字节。原阶段回执active-head-v2-pair-phase-1791350061.json，
SHA8d51c39b…407d1，完整指标见results_direct_action_target_textcraft_step2_20261007.json。
保留实际输出中的allocator虚拟计数，不将其冒充物理VRAM峰值。未重启两组或增加测试。

## 2026-10-07 13:11 AppWorld 输出层内存修复已对拍并正式提交；TextCraft 第二轮更新中

AppWorld 已由原提交器重新启动，PID670069/birth1791349552.74/source SHA42bb0eb9…，
物理4/5。实现绑定 f921d49a70378488ba09b15e80f21ed5c1ef03ac；仅 DT 输出层文件改为
1e20956a…3541e，entry、VERL、LOOP 沿用 direct-action-target-20261007-v3 原目录。
实际发布为 candidates/direct-target-head-memory-20261007-v2/deltatrace；冻结 source
核验1663项，原配置只改变四个输出/可见目录。LoRA8/16、每卡B4、原任务预算及原
Q/V/A、PPO均未改，resume_mode=disable。旧PID167065的OOM属于已退役运行。
原准备回执的 prepared_only 状态记录的是准备时刻；后续 verification.json、正式
submit 回执和当前PID/source分别证明数值对拍、提交及当前运行，不能混为同一状态。

修复按128个target row调用同一个原self.seed，保持原公式/FP32 dtype/顺序/checks，
合并后一次原scatter；root log-softmax同样分片。对拍使用原真实输入，一卡一个B4，
两rank分别160/196个target row，仍不是完整多轮joint目标的32k容量验证。
两端目标log-prob逐值相等；hidden最大差1.222e-6/1.483e-6，allocated最大差
3.248e-6/1.907e-6。原torch.testing.assert_close无rtol/atol覆盖，实际FP32默认
rtol1.3e-6/atol1e-5，两rank均通过。该判据仅属于PyTorch输出层回归；原FA/FLA
断言及PPO没有改变，不把它宣称为FA/FLA整链或完整训练可靠性证明。
没有裁剪、倍率纠偏或更改信用；原full-vocab logits仍保留，正式完整joint峰值待观察。
v1导入身份失败和v2过强bitwise诊断门槛的原记录保留；前者未进入DT，后者正常FP32
差异通过后续原dtype默认对照，不将其写成原官方容差失效。

TextCraft仍为f29cc7c0/PID110053/birth1791344324.6/source5013ebc8…、物理2/3。
已完成step1；第二次采样30/30，原生成29分56秒，第二次DT两rank均21/21完成，
当前在第二次actor update。第二批161个唯一非零奖励轨迹，原DP补至168；尚无step2
完整指标，不能由启动/单阶段完成报告新的学习效果。

13:11只读阶段回执 active-head-v2-pair-phase-1791349875.json SHA51fe7a68…e1d76：
AppWorld已进入原native_async采样，完成31993生成token/99.6秒；该时间包含RPC、
prefill和decode，不是纯decode。当前未完成首个新DT或更新，仍需观察原失败阶段。
物理2/3显存41662/30924MiB，4/5为55308/55226MiB；容器cgroup235750555648字节
（约219.56GiB，属于整个容器）。这只是当前占用，不是峰值或训练健康证明。
完整来源见 results_direct_target_head_memory_20261007.json 及 current_runtime.json。
SQL/GRPO保持停止，未恢复检查点、启动新备份或增加额外GPU测试。

## 2026-10-07 12:38 真实动作目标：TextCraft首轮完成；AppWorld有限输出层OOM

TextCraft仍为f29cc7c0/PID110053/birth1791344324.6/source5013ebc8…、物理2/3，
已完成step1并开始下一轮原生成。原整轮2804.125秒：gen1825.958、old_log_prob148.132、
ref117.135、adv213.580、update_actor498.869；原grad_norm0.037、entropy0.677，
本批reward均值0.664来自更新前采样，不能当作更新后改善。原save_freq25，step1没有
checkpoint不是保存失败。原指标完整行/来源见results_direct_action_target_textcraft_step1_20261007.json。
完整DT两个rank210.289/210.421秒，各22个B4；170唯一非零奖励轨迹由原DP补至176。
raw prior credit范围−23.0554至1、self credit=1；这些是DT子集/补齐行的未白化统计，
不是原整批actor优势，也不能仅据极值判定归因质量或训练退化。

AppWorld同一v3 PID167065/birth1791344807.34/source70ffdcfd…已退出，未完成新更新。
两rank均完成21/27个B4；第22个在正式FiniteAnswerOps全词表finite_seed内，
logarithmic_mean_with_checks的torch.where申请5.43GiB失败，原physical free1008.18MiB。
原日志没有失败批次的准确N/U/context，不能拿v4短输入几何替代或据此声称32k越界。
12:38原mx-smi确认4/5已释放至各859MiB；2/3仍只有原TextCraft。终态原日志回执
direct-target-semantics-20261007/fresh-v3-terminal-check-1791347887.json SHAee6aad2a…fc970，
逐行OOM审计appworld-finite-seed-oom-1791347887-readonly.json SHA9d89c3ca…03ec0。
未恢复检查点、缩小B4、更改LoRA或提交SQL/GRPO。

仅准备隔离head内存候选：从实际d47333ea…复制，128个target row一片调用原self.seed，
原checks合取并按原顺序拼接，最后一次原scatter；root诊断同样按row调用原log_softmax。
它不重跑模型/32层、不分奖、不改变DT/PPO公式；候选源码1e20956a…3541e，尚未部署。
实际dtype/原head逐值对照与大联合目标容量须分别确认，不以语法检查宣称修复成功。

两组先前reward_extra_info提示已核对固定VERL20bd331原EpisodeRewardManager和compute_reward
的字节及原兼容fallback：原rm_scores张量不变、没有重算环境或丢奖，随后均进入DT。
该提示不是本次OOM原因，未为它增加包装或修复。

## 2026-10-07 12:06 v3 正式采样继续；尚无首个新更新

已重新核对两组当前PID出生时间和冻结source SHA，仍为下节的f29cc7c0版本。
TextCraft已完成22/30次交互，进入23，原中央256条中90条仍活跃；该计数不是每卡256条，
也不把已结束轨迹数称为成功数。AppWorld原collector最近分别完成74/120、92/120；
原transport累计580032 token/933.7秒，约621 token/s，含调度、prefill与decode。
官方停止条件仍是全局至少216且40个任务组各至少4条，之后原取消/收尾；不是每卡108。
两组仍0/330与0/200，完整DT/PPO、梯度及学习效果尚未观察，保持运行，不重复提交。

物理2/3显存50152/49934MiB，4/5为55260/55284MiB；容器cgroup218.93GiB。
采样计数持续增加，当前无具体停滞或新增OOM证据。新阶段回执
`direct-target-semantics-20261007/fresh-v3-pair-phase-1791345961.json`，
SHA c0706fe75bdbea00de254227e908fde5d80684971b24f29d9160f33cdc64cc43；
摘要与已有监测字段见`results_direct_action_target_observation_20261007.json`。
只读阶段采集器原来仅筛旧DT EOS标记，现补上现有direct-target/readout日志标记；
旧采集器8e3f2300…按原字节保留v1，新采集器ecdf981f…，未改训练日志或训练代码。
不能用筛选缺标记断言DT未执行，也不能用此前单response回执证明完整多轮joint容量。
现有原runner detail含head形状/字节/phase计时，但readout最终摘要未转发这些字段；
当前仅记录该观察范围，不因尚无完整轨迹数据新增仪表、测试、优化或容量结论。
SQL和GRPO继续停止；未恢复检查点、改变资源/任务配置、启动额外GPU任务或备份。

## 2026-10-07 真实动作目标 v3：两组正式 fresh 已提交

当前方法代码为 `f29cc7c055fcd4e5a055639c48aaf8d913bac180`（含 a317575e 的
真实动作目标接线）；正式发布目录为远端 `candidates/direct-action-target-20261007-v3`。
TextCraft 物理 GPU2/3：PID110053/birth1791344324.6，source SHA5013ebc8…；
AppWorld 物理 GPU4/5：PID167065/birth1791344807.34，source SHA70ffdcfd…。
两组均从基础权重启动，`resume_mode=disable`，没有恢复旧辅助标签检查点。
11:50:34 观测 TextCraft 已完成5轮交互、进入6/30、0/330；AppWorld已进入
原LOOP实际task/world runner初始化、原vLLM已wake并列出LoRA，尚无完成回复计数、0/200；
尚未观察本次完整更新，不能据启动或算子验证报告学习效果已改善。
该窗口物理2/3为50239/50241MiB、4/5为47730/47714MiB；两组worker PSS
分别8.25/8.14与8.68/7.89GiB，整个容器cgroup204.64GiB（不归因于某一个job）。
新窗口未见Traceback/OOM。来源为fresh-v3-pair-phase-1791345034.json，
SHA c8c07389f5f6da950afc9c83b083a766ef9f507fa46d7872a7ef9ea7d02200ab。
11:52:01 的后续 current_runtime 快照中，TextCraft 已进入7/30、仍0/330；
AppWorld PID/source仍匹配，尚未观察到完成回复或新训练迭代。上述资源数值属于11:50:34。

完整 PID/source/argv、实际入口 SHA、原有效配置、验证来源见
`results_direct_action_target_20261007.json` 与带采集时间的 `current_runtime.json`；
运行事实仍由远端 active-training/active-source/formal-training 清单给出。
注意 source 继承的 `submission_repository_commit` 属于前序版本；本发布的实现
由 `local_patch_commit=f29cc7c0…` 及实际字节绑定。只读采集器已显式区分这两项，
没有原地改写冻结 source。不要凭旧字段、目录名或旧 workspace 入口重发历史版本。

本次接线调用原 parser/execute 保留的实际 Action/代码位置和原 token IDs。
每轨迹一次联合 Y、一次官方终局奖励；self-target用已接受的 Q=r/V=0/A=r，
前序 source 用原 EOS DT signed 值及 expm1 组合。O/padding仍按原mask；
没有额外标签、参考token采样、target长度分奖、source/target单独缩放或第二次GAE。
原 VERL 整批 masked_whiten/PPO、LoRA8/16、每卡actor/DT B4和官方任务预算保持。
TextCraft 32组×8、global PPO mini64、30epochs/330iters；
AppWorld 40组×6、global mini32、PPO epochs2、200iters；原完成比例和评估设置不变。
训练长度仍为各任务官方口径（TextCraft10752/eval14848，AppWorld32000），
32768是运输/DT上限，不把上限冒充实际轨迹长度。

上游固定版本：VERL-agent20bd331bdbc9026a5668e11362178e10ab7400c8，
AgentGym-RL82402a99c62a293735a3f412fb8ac9a600673bc0，
AgentGymd014732d9fe39b975c368c03749bfd50950067f6，
LOOPf14107a976e5793990329d3193df4742076c5a1d。
原producer494b53b0…、PPO actor3a65e173…保持；新readout31e2acfb…。
两个实际12项CPU导入和所有冻结sourcebindings均已核对，旧未接受候选未混入启动。

CPU：组合21passed、官方parser/桥23passed、direct readout6passed、
原VERL运输5passed，分别对应各自范围。真实多目标首次发现原runner输出Float64
写入Float32载体的index_put错误；修复仅在运输边界明确转换载体dtype，
没有更改原signed/detail、公式、FA/FLA或增加纠偏。v2 AppWorld刚启动的
PID27716/birth1791343616.74已精确停止，163个所属进程退出，原source3f34…
保持；无等待/创建/导出/恢复checkpoint。其日志不是当前v3异常。

数值回执为 `direct-target-numerics-20261007/v4/direct-target-numerical-receipt.json`
（SHA85091a1f…63fb188）：8条真实原ID current-response载体、两UID历史，
上下文11678–13153、356个实际target token；每rank一次B4完整读出/QVA回填成功。
原FA 8实际row×7断言=56passed；原cached FLA两实际B8调用的o/ht共4passed，
保留原BF16初态/FP16 qkv及原参考、原阈值。没有把它扩成整网有限传播容差、
完整多轮联合目标32k容量或PPO训练成功。冷调用44.9秒、带操作数导出调用
69.1/65.5秒仅为该输入诊断耗时，不是正式每迭代耗时或纯内核吞吐。
本批raw self A范围0.6–0.8，prior A范围约−0.1192–0.7993；不能据此断言策略改善。
数值进程已全部退出，4/5释放后才提交当前AppWorld正式任务。

TextCraft原服务3313391/birth1791334851.48及36005监听已重新核对后直接复用。
没有重装环境、下载资产、清理持久编译缓存、提交SQL/GRPO或修改备份进程。

## 2026-10-07 10:22 辅助标签两组已停止；目标自身删除边界按用户新澄清分析

TextCraft PID3327460/birth1791334993.84、AppWorld PID2360541/birth1791325655.01
已通过原resume_at_native_checkpoint.py的stop-only生命周期停止，两个原进程树
均无剩余非zombie进程。原环境/采样/算法/DT源码未替换，没有提交替代训练，
也没有创建、导出或恢复检查点。TextCraft停止前最后已读原进度1/330（首轮3342.758秒，
gen1850.655、DT723.082、actor502.406秒），原保存marker尚不存在；AppWorld原
进度2/200、第三轮DT，既有marker2只读保留，未因此宣称该实验语义正确。

原停止助手默认要求等待完成检查点，第一次在TextCraft缺少marker时拒绝且零signal；
随后stop descriptor遗漏source中owner_head_sha256的.deltatrace-patch.lock条目，原source guard
再次拒绝且零signal。两次失败材料保留；最后v2 descriptor合并原两份源码映射、
验证重叠项相等后复用同一助手。仅在该本地生命周期owner增加显式
--stop-only --stop-now：不等待/创建/恢复检查点，原PID-birth、source和所属树保护
保留；无新flag时默认路径不变。原CPU真实父子进程测试20passed（9.31秒），
只证明生命周期，不是训练数值验证。实际助手SHAb5be347a4ba7deb2f02b7c892f375642b91d4992290d64f1d7b7c2659b211f0f。

终态回执direct-target-semantics-20261007/terminal-state.json已与远端active-training、
formal-training、active-source状态对齐；10:22原mx-smi显示GPU0–7均无进程、各858MiB。
原备份独立PID3591046/birth1791337520.95身份一致且仍存活，未重复启动；存活不等于
restic完成或校验通过。原训练输出和source记录保留。

用户本轮进一步明确：真实action target自身的token删除后，该literal结果的反事实
概率按定义为0。该直接信用项与目标之前source的DT归因须区分；本轮尚未修改PLAN、
Q/V代码或发布新信用版本，不能把已停止的标签实验当成新接法验收。

## 2026-10-07 10:06 实际奖励目标审计：当前辅助标签不是官方执行输入

用户本轮明确关注实际 LLM 生成、经原环境执行/验证并产生奖励的 target span。
核对当前 reward_readout.py SHA94a7afbc09da72b62572d31fd32a6534f6e8f3daf656fce1011cdfa68b3c3e2b：
132–145追加 Future cumulative return forecast 询问，209–216按实际累计回报选取
人为类别标签，284–287走类别子集归一化。这是在估计新增预测任务的标签概率，
不是评分官方环境实际执行的生成结果；两种概率尚无等价证明。原DT算子容差、
归因守恒或优势白化均不构成这层目标语义的验收。

原TextCraft调用链是采样assistant→官方Action解析→原库存/配方/目标状态转移→奖励；
原AppWorld调用链是PolicyMessage→官方代码提取→world.execute→world.evaluate→
原测试通过比例。不能用任意final文本代替被验证的执行结果。两个桥已保留完整
response边界；TextCraft原始采样IDs与原handler重编码后的训练IDs是不同的已有
artifact，均有记录，不假定逐token相同。解析后子片段的token offsets及当前完整
原始IDs磁盘dump未确认。原PackedAnswerTargets默认全词表分支支持真实target IDs，
但这只证明评分接口存在，不证明替换target后所有action-token Q/V均已闭合。

具体owner版本、实际源码SHA/行号与已有artifact字段见
results_direct_target_owner_mapping_20261007.json（SHAe5be308a9705d375af746f11414260c330cf8644f3e0ccd4cbd484a5f61fba41）。
本次仅本机只读源码审计与事实记录，零环境/模型/GPU调用，未改PLAN、算法或部署，
未停止/重启TextCraft GPU2/3和AppWorld GPU4/5。两组现有运行仍是辅助标签目标，
不得将其结果报告为“实际生成结果触发验证奖励”的直接target实验。


## 2026-10-07 09:47 TextCraft首轮DT、AppWorld2/200；备份已越过旧断线窗口

TextCraft同一PID3327460/source5faa/物理GPU2/3，原30/30采样用时30:40，官方服务
256reset/3236step/256close；现两rank进入原compute_dt_token_advantages，当前类别组
42/175个B4，约2.5–3.7秒/B4，该组不是整轮进度。没有恢复旧checkpoint；原整批
masked_whiten与LoRA8/16、actor/DT每卡B4不变。原reward_extra_info提示已核实为
VERL可选metadata的官方fallback，保留同一rm_scores，不重跑环境/模型、不补零。
AppWorld同一PID2360541/sourcec83/物理GPU4/5，原进度2/200、两rank第二次actor
phase释放已返回，第三轮原LOOP采样在推进。未对TextCraft首轮更新或学习恢复
下结论。实际phase1791337516与reward原文件/hash回执见TextCraft结果索引。

原attached SSH保活未阻止reset。只在本地backup运输边界使用标准Popen独立会话，
沿用现有launcher lifecycle pattern；不是调用原launcher可复用helper（其无此API）。
原training owner未改，restic cmd/flags及summary→restore/fullSHA/tag/cleanup同原。
help与真实单manifest detached dryrun通过；完整源/范围回执见detached-restic边界。
代码b139ddd/SHAcebe75b9已提交推送；本机PID13676/birth2026-10-07T01:44:56.0990573Z。
远端控制PID3591046/birth1791337520.95与原restic独立于启动SSH，09:47实际持续
103秒、原log累计96秒/83766files/548502552bytes，已越过此前53/70秒失败窗口。
本次仅一个任务，原meta传输仍在进行；不占GPU，没有手工checkpoint导出/恢复，
尚无完成snapshot或restore/fullSHA验收。原3次失败记录均保留，当前身份与日志/
exitfile见results_backup_restart_20261007.json，不把prepared或失败版本混作运行。


## 2026-10-07 09:37 备份native长连接亦失败；训练入口不变

fresh-v3-native本机PID12432已退出；原OpenSSH255，日志明确Connection reset，
末次原restic状态53秒/49363files/469745000bytes、无summary或完成snapshot。
原保活未避免本次通道断开，不能称备份修好。精确远端restic3489870/birth1791336566.77
及子进程存活在查，先不重复提交。正在复用已有远端独立进程/日志/exitfile入口，
只处理原restic长命令生命周期，不改备份/恢复/SHA语义，不影响训练。真实失败
receipt与日志保留，results_backup_restart_20261007.json已更新失败状态。
下方09:30“运行”仅为当时38秒的观测，不代表现在仍在备份。


## 2026-10-07 09:30 异机备份原OpenSSH长命令边界已启动，快照校验未完成

仅原restic长命令改用现有stage_environment_entry.SSH（严格可信key与保活15/3）；
原Paramiko保留SFTP/元数据并启用其公开keepalive15。未增加重试或另一套备份逻辑，
原restic参数与summary→restore/fullSHA/tag/cleanup AST相同。原help和真实单个
manifest dry-run通过，非全备份验收；此前70秒channel关闭根因尚未证实。
修复commit1863fc8、脚本SHAa8461073；精确restic与本机backup均无残留后，仅提交
一次fresh-v3-native：本机PID12432/birth2026-10-07T01:29:00.0633513Z，远端原restic
PID3489870/birth1791336566.77，09:30仍运行，RSS749404160字节，CPU累计9.22秒。
实际来源与运行记录见results_backup_restart_20261007.json。数据从MetaX经4090直写
原A6000存储；不占GPU，没有手工导出或向训练恢复checkpoint。尚无完成snapshot、
restore --verify或完整SHA结果，不称备份成功；原两次失败记录保留。TextCraft2/3、
AppWorld4/5正式入口/PID/LoRA8/16/每卡B4保持，下方训练相位记录仍有效。


## 2026-10-07 09:19 TextCraft原正式环境已执行；AppWorld进入第二次actor更新

TextCraft同一PID3327460/birth1791334993.84/source5faa6e2d，物理GPU2/3；
原官方服务3313391已记录256次正式reset、461次step及HTTP200，排除了已关闭的
readiness env0。observe由原client读取返回缓存；未新增API、模型或GPU探测。
09:18原日志仍首轮generate_sequences、0/330，没有完整训练更新验收。
AppWorld同一PID2360541/sourcec83、物理GPU4/5、参数及入口不变；第二次DT两rank
均已执行原phase末缓存释放，随后进入原update_actor。只读phase-1791335936
及service-formal-log-1791335480来源见results_textcraft_fresh_20261007.json。

备份fresh-v2的原restic实际传输70秒后channel exit=-1，无summary或完成snapshot。
原失败receipt与输出保留；09:15原OpenSSH短命令成功，精确backup-access restic
进程及子进程均为0，没有仍在运行的备份，不把首次TCP超时当成主机停机。
正在定位原SSH长连接边界，未重复提交；backup/restore/SHA/tag语义不变。
此段纠正下方09:06“传输及校验尚待确认”的历史状态，不宣称异机备份已成功。

## 2026-10-07 09:06 TextCraft fresh正式GPU2/3已进入原生成；AppWorld4/5不变

TextCraft PID3327460/birth1791334993.84，提交代码14ba007，实际source SHA5faa6e2d；
entry/VERL为textcraft-fresh-row-prefix-20261007-v2，正式output为同名runs/textcraft-dt。
新worker3332777/3334413已加载原模型，进入原generate_sequences；原trainer显示
0/330、Total training steps330。没有恢复旧检查点，旧无效轨迹未复用。LoRA8/16、
每卡actor/DT B4与原作者负载不变；原validator1passed及CPU准备31afb沿用。
实际source scope已修正为fresh，不继承prepared模板中的历史checkpoint25描述；
原template/hash仍保留作可审计输入。actualsource绑定本次prepared/plan/interface/
service-current四份收据，不把旧服务PID或旧prepared当运行来源。

09:06物理mx-smi：TextCraft GPU2/3 worker49472/49164MiB（生成阶段，不是峰值），
AppWorld原GPU4/5 worker31348/31890MiB，driver2360541/sourcec83不变、第二轮原DT。
TextCraft worker PSS约8.3/8.1GiB；cgroup总299.42GiB，不累加fork RSS。
新TextCraft尚无完整rollout/DT/actor返回，不能称已稳定或质量恢复。
原日志/资源回执phase-1791335163及正式source/job详见results_textcraft_fresh_20261007。

首次异机backup PID10132在SSH exec输入处EOF，零snapshot；实际新manifest1678条
路径使命令305599字节，旧manifest去重18KB结果不适用于新symlink源树。旧失败
receipt保留。仅改用现有restic官方--files-from-raw：原路径NUL列表304758字节
SFTP读回完全相同，实际命令1019字节；原单个真实manifest的native dry-run4.88秒
exit0，未写snapshot。脚本commit1180126/SHA0b44f13d，原restic备份/restore verify/
完整SHA/tag不变；09:07:53以新receipt fresh-v2重启本机PID12800，不占GPU，
传输及校验尚待确认。未手工创建或向训练恢复任何checkpoint。


## 2026-10-07 09:02 用户指定TextCraft GPU2/3：fresh原入口就绪；备份边界修复

AppWorld物理GPU4/5、PID2360541/birth1791325655.01、source c83b96de保持不动。
TextCraft旧任务已停止；最新用户授权新任务使用物理GPU2/3，不恢复旧检查点。
新候选textcraft-fresh-row-prefix-20261007-v2/prepared.json SHA31afb085，
source-template7dfe857d、launch-plan c82f10f1。仅fresh launcher auto→disable，
producer复用3e0c、lease b947、DT runner5f14/库4f42/environment4ff；原TextCraft
同步rollout/任务客户端/奖励、原VERL wholebatch masked_whiten trainer736保持。
actor仅复用已验3a65共享右padding补丁，update_policy AST仍原样；原配置32组×8、
globalmini64、有效PPO epoch1、30 epochs，LoRA8/16与每卡actor/DT B4不变。
v1只链接Python文件而缺原version资源的准备失败已保存；v2恢复原1450文件资源，
14项真实模块CPU导入/factory接口通过，PSS546.7MB/maxRSS981.4MB、CUDA未初始化。
这不是新的模型/数值/32k验收或已完成训练。

原TextCraft服务4049454已退出。仅重启作者agentenv_textcraft.launch原CPU服务36005，
复用已有textcraft-extras-20260930，不安装；新PID3313391/birth1791334851.48，
原client create/reset/observe/close实际返回，service-current SHAd7ebfac1。
原Hydra/native validator单项通过；原entry/textcraft-service.json已更新，旧记录保留。
本段为prepared/live-service状态；正式训练PID以随后的提交收据为准。

异机备份此前未运行。原私有连接器仅端口改31146，严格比对原可信ed25519公钥后
绑定新端口。原backup_metax_to_restic.py修正新manifest output/checkpoints字段，
记录的实际源/外置HF与FA库、PID birth匹配的Ray日志纳入原restic接口；1557路径
去除已被目录覆盖的重复项后SSH路径段18,231字节，实际覆盖不减。原传输、
restore --verify、完整SHA、tag/cleanup循环AST未改，完成CP先于旧47GiB profiler。
最终脚本SHA09ae15e6、真实边界收据4b3032da；尚未启动传输，不能称备份成功。
目标A6000只作存储，实测剩余139427086336字节；原恢复空间检查保留。


## 2026-10-07 08:40 有界有限FA算子trace定位dQ；原shared-A候选更慢，未部署

同一fresh driver2360541/birth1791325655.01、source c83b96de、worker2367855/2369144
继续GPU4/5，正式4f42库与3e0c/5f14入口不变。第二次原LOOP采样已完成：219轨迹、
3234实际responses、600234 policy tokens、最长context27448；08:40进入原DT，当前
一组两rank均76/83个B4。该组进度不是整轮完成率。当前worker PSS57.0/57.7GiB、
cgroup255.5GiB是相位观测，不作峰值。没有旧检查点恢复、手工导出或生产profiler attach。

在空闲GPU2串行使用已保存的真实decoder3第8次FA操作数，两rank各一次原4f42调用，
用原torch profiler采到三个有限kernel与7次转换/contiguous；不是整DT或32kprofile。
rank0/1 kernel累计90.323/77.222ms，其中Phase1 dQ58.968/49.981ms（65.29/64.72%），
Phase0 center26.565/22.710ms（约29.41%），Phase2 dK/dV3.038/2.816ms（约3–4%），
转换1.753/1.714ms。不能把wrapper嵌套self_device或CPU同步再重复相加。
原profiler只用于观察；model/CP/生产变更均0。源/trace绑定profile-analysis SHAa5868fa5。

复用原352415的shared-A gemm_opt调用，只改变当前CUDA Phase1三处条件，P0/P2、
tile、dtype、有限公式和ABI不动。候选CUDA0f4a3add、库a1288cf0，原675d builder
49.45秒编译，PSS峰值0.944GiB；隔离8完整真实B1×7条原FA断言全部通过，实际非零
B4的dq/dk/dv/tau/center五输出与4f42逐位相同。原7ff/e32/a290源及容差未改；仍仅
原coincident derivative-limit范围，不扩大为全DT精度。2warm+5中位数rank0
90.295→97.369ms（慢7.83%）、rank1 77.421→84.002ms（慢8.50%）。原公共
mcFuncGetAttributes每库独立进程确认P1 localSize160→20字节、numRegs仍256；
静态local减少不能代表动态spill流量或加速。明确rejected_performance_not_deployed，
不扫tile/warp参数、不把该候选混入默认入口。全部源和失败收益证据保留在
finite-fa-query-shared-A-candidate-v1，决策回执SHA7b16f208。

原actor update_policy AST仍与VERL20bd331相同，无新增forward/optimizer循环；旧LP
在actor计时之外。作者LOOP同样epochs2/globalmini32，但其4学习+4推理、B1累积、
可选CPU卸载及标量优势过滤，与当前两卡共享、固定B4和原VERL不等同。当前31.1分
更新超过21.5分采样本身不构成错误，也不证明官方吞吐达标；未改actor/采样配置或
导入LOOP训练器。具体原路径/hash见actor-official-workload-scope SHA394c244b。
当前整批白化的完整首轮执行与非零更新证据沿用下段，不宣称学习质量已恢复。
当前逐行root/replay的算子/搬运trace仍缺；旧scalar H2D计量和旧19%root-tape收益
不能代替它。该测量缺口是后续具体工作，不通过重复整轮或无依据参数扫描解决。


## 2026-10-07 08:13 同一fresh首轮完成：原采样/DT/更新计时与整批白化执行

同一driver2360541/birth1791325655.01、source c83b96de、worker2367855/2369144、
GPU4/5已完成原step1并进入下一轮LOOP采样；fresh base、resume_mode=disable，
没有旧检查点恢复或手工导出。原VERL自行保存完成标记1，save52.529秒。
actual runtime仍37b9085逐行prefix修复与3e0c/5f14/4f42绑定，本段仅记录，不把
9850e12后的诊断/记录commit冒充新运行代码。LoRA8/16、actor/DT每卡B4不变。

同一本轮原timing_s：gen1288.023秒(21.47分)、old_log_prob265.116秒、
adv2482.782秒(41.38分)、update_actor1866.647秒(31.11分)、step5956.037秒。
更新/采样1.449，DT/更新1.330，DT占整轮41.68%。原两rank DT外层2476.151/
2475.368秒与driver adv边界不同；不是用DT batch计时替代原trainer完整adv。
218原轨迹经原copy调整到224；原perf/total_num_tokens=2959628含O及6复制行，
两epochs为5919256有效context位置呈现，不是policy tokens/FLOPs/dense padding
槽。原采样policy_tokens667405；没有额外epochs或逐response PPO更新的证据。
56 B4/card、14 optimizer边界是原代码工作量推导，不冒充逐调用计数；不能拿旧
partial-mini容量probe134.131秒直接外推。当前更新比采样长有原负载解释，尚非
与发表的同MetaX官方速度达标。没有新增测试、attach profiler或生产patch。

整批原masked_whiten已在本次actor前真实执行；原d/Q/V/A保留。原actor/grad_norm
0.041、pg_loss-0.002、pg_clipfrac0.001、entropy_loss0.391，均为原console三位小数；
这是完整非零更新的证据，不是孤立任务梯度或后续学习恢复证明。原episode reward
均值0.788不是独立评估成功率；本轮rollout发生在更新前。

原training/rollout_probs_diff mean0.436/max1包含O：trainer d35的1173–1183按
attention_mask选位置，LOOP转换01eca对O的rollout logprob填0，exp后为1；原actor
3a65的432–433仍按loss_mask排除O。该汇总量不能宣称action-only推理误差或
vLLM官方容差失败，不改原指标、损失或补偿。原perf虚拟allocator77.657/84.029GB
不作物理显存峰值；实际mx-smi相位观测约43GiB/卡，无OOM，不扩成全程峰值。

原日志/精确绑定：native-actor-return-1791331896-readonly.json SHAec64455e；
解析native-first-iteration-metrics-readonly.json SHA8fcefb9c；mask源回执5083b0e1。
最新current_runtime-1791331987.json SHA0e5bd19c。全部在row-prefix-owner-fresh-v1。
现有完整轨迹有效长度总量缺失项已由原perf实测补齐；效率主项仍为DT归因批次，
保留已验收的历史重算修复，未部署无端到端收益的预取或变慢FA融合候选。
旧H2D trace2c01的27.315/25.117GB不是当前3e0c/c83搬运量；无shape/stack不能
精确分参数/cache。对应版本与rank算术已固定h2d-owner-scope-readonly.json
SHA2aba1789，不再把混合copy设备累计秒宣称可删除的权重搬运墙钟。

## 2026-10-07 07:51 有界有限FA融合候选通过原断言但变慢，拒绝部署

已在空闲GPU2使用此前保存的真实rank0/1 BF16/FP32 FA操作数，独立ABI融合原
Phase0 center/tau与Phase1 dQ，原Phase2及默认三阶段ABI保留。候选CUDA7a6dfeb3，
library60b9a5df；复用原cucc/header/flags，55秒编译峰值PSS约0.95GiB；未安装包、
加载模型/检查点、清缓存或改正式4/5。原7ff checker对8个完整逻辑B1行的56项原
FA断言全部通过，候选默认ABI对真实非零B4五输出与4f42逐位相同。融合非零dQ
maxAbs差0.000732/0.000610，其余四输出相同；这不是新增有限误差容差或全DT精度。

同输入原wrapper调用2warm+5次中位数：rank0 0.090186→0.127183秒(+41.02%)，
rank1 0.076876→0.109576秒(+42.54%)，包含原Python/分配/转换/同步，不叫纯kernel。
原三组score MMA被少算并不保证此融合快；尚未证明具体寄存器/spill原因，不据
源码猜测下结论，也不继续扫参数。候选明确rejected_performance_not_deployed，
保留在finite-fa-center-query-fusion-candidate-v1及原回执，正式source c83b96de和
4f42库未改变。两rank测试串行约38秒，实测进程树PSS6.17/5.97GiB；无训练更新。
来源verification-and-performance-decision.json；精确源码/SHA、原断言、计时与
执行命令分别保存。这次失败候选不进入默认launch/patch路径。

## 2026-10-07 07:39 当前fresh完整DT及整批白化已执行，原actor更新中

同一driver2360541/birth1791325655.01、worker2367855/2369144、source c83b96de
继续GPU4/5，fresh base、resume_mode=disable，没有旧检查点导入或导出。
phase-1791329982原两worker进程名均为actor_rollout_update_actor；冻结trainer d35
373对整批原response_mask调用原masked_whiten，374写入actor advantages，原1256–
1271的compute_advantage返回后才进入1285 actor RPC。本次白化已经真实执行，
不再仅是已安装/源码存在。原d/Q/V/A保留；actor尚未返回，不称任务梯度强度或
学习质量已恢复。源和执行证据绑定whitening-execution-and-original-actor-workload-
1791329982-readonly.json SHA9543cb0d，原actor expected56 B4/card与14 optimizer
边界仍是源推导，不是完成次数。

当前原DT七组全部返回，每卡121/58/39/71/23/107/20，共439 B4、1756 transport
contrasts。rank0/1外层2476.151/2475.368秒；归因batch2044.021/2043.840秒，
历史capture准备428.593/428.031秒，其他外层3.537/3.496秒。两rank并行，不加总
成墙钟。实际3485源response保留，padding contrasts共3512不是新的轨迹或reward。
回执completed-dt-readout-costs-1791330003-readonly.json SHA8b2d7d4f，绑定原worker
日志前缀SHA/行号。未拿439/56调用比冒充FLOP比或效率达标。

07:39 actor阶段物理GPU35384/40057MiB，worker PSS45.16/46.11GiB，cgroup
223153205248B约207.83GiB；这些是当前时刻，不是峰值。仍等同一原迭代的
VERL gen/update_actor完整计时；不引用旧2495秒判断新actor正常。当前只读记录，
不添加生产hook或数值补偿，不改变LoRA8/16、actor/DT每卡B4及官方任务预算。

## 2026-10-07 07:26 本次DT前五组成本完整；核查两个具体重复计算假设

同一driver2360541/birth1791325655.01、source c83b96de继续GPU4/5，无检查点操作。
原workload一行明确response_rows=unique_rows=retained_sources=nonzero_before=
nonzero_after=3485，skipped_nonzero=0；这是完整未来Gt的源行数，不是非零环境
reward行数或成功轨迹数。当前原DT每卡至少需要ceil(3485/8)=436个B4调用，
不同num_tests组的原padding可能再增加调用数；原PPO预计56个B4/卡，两者计量
单位与执行内容不同，不拿调用数比值直接当耗时/FLOP比或正常性证明。

原前五组每卡计划并完成121/58/39/71/23，共312 B4、1248 transport contrasts；
两rank累计outer1771.738/1771.317秒，batch1448.534/1448.365秒，capture准备
320.747/320.534秒，剩余外层2.457/2.418秒。第六组已计划107 B4、428 contrasts
每卡，尚未完成。回执completed-dt-readout-costs-1791329194-readonly.json SHA
ea772d73，原日志前缀SHA、来源行与collector版本b80a67b3同时保留。不能从已完成
transport contrasts直接减出精确剩余unique行数，不能将此称作整轮DT耗时。

已排除“算完整历史dK/dV再丢弃”的具体假设：当前wrapper3e1d的100行只分配
like(q0)的[B,Hq,Smax,256]系数，112–117走row_cached_suffix；CUDA9ebcef的
60–69读取每行pi与实际Li，303–313按Smax/64启动，Phase2从pi后的key row开始。
runner5f14/384–386传真实pi/Li/Smax，decoder047c/213–216直接消费后缀系数，
没有先生成j<pi的历史K/V系数再切掉。Phase0/1读取历史K/V以计算变化后缀Q的
概率、归一化和finite center，是当前有限传播的输入；未据假设添加新kernel。

官方FSDP2公开参数驻留接口当前也已接入：producer3e0c沿用原2c01的126–145，
在单层replay时set_reshard_after_forward(False,recurse=False)，finally恢复原设置；
147–157通过原unshard/reshard衔接finite。原root101–109显式release decoder
gathers后单独保持head/norm，并未整模型跨批驻留。现有B8 profile的67gathers
没有finite前第三次重复gather证据；H2D25GB/.65秒混有参数与状态，不全算权重。
本机只有当前Torch原文件SHA/行号摘要f84dd2a6，没有完整CPUOffloadPolicy源，
不从另一个Torch版本或混合copy推断转移生命周期，未更改任何卸载策略。
本次核查没有新增GPU测试、profiler或生产代码；当前完整DT/白化/actor仍待原调用。

## 2026-10-07 07:16 本次DT前两组完成，拆清原缓存准备与归因成本

当前driver2360541/birth1791325655.01、source c83b96de、GPU4/5未变，fresh base、
resume_mode=disable，无旧检查点操作。原完整采集字段为218 trajectories、3485
responses、667405 policy_tokens、max_context26218；原adjust_batch默认copy按8补齐
到224，因而原两epochs预计每卡56个B4、14次同步optimizer step，尚非已执行次数。

原第一组121个B4/卡完整返回：readout747.022/747.048秒，batch计时合计609.393/
609.392秒，原shared_native_prefix.capture_and_preparation_seconds为136.761/
136.778秒，占18.31%；外层剩余0.868/0.879秒。第二组58个B4已完整返回，readout
294.011/293.860秒，batch240.928/240.763秒，准备52.584/52.648秒。两rank并行，
不能相加当墙钟时间。第三组39个B4在07:16观察到27个完成；整轮DT、白化、actor
尚未返回，不能宣称本次梯度或学习质量已恢复，也不能拿旧2495秒证明当前正常。

当前第一组与旧v2轨迹不同（121对130个B4、context_sum反而约多6%），不把平均
batch时间或整组墙钟直接当加速比。原readout.compute_tokens仍为完整batch最大
长度，original_shared_prefix_token_slots仍为原共同MIN边界的兼容元数据；不能
把两者相减叫逐行缓存实际后缀/FLOPs。单个保存真实B8的9.75到3.96秒仍只作用于
原同输入对照范围。首组events26但unique_histories63是先按完整轨迹算Gt、再按
response分卡：另一卡37条原非零reward行，worker沿用预计算dt_complete_return，
没有本卡重算Gt或为零Gt构建DT请求的证据。

实际environment4ff已启用dt_dynamic_shapes和compile_gdn_scalar_rules。只读扫描
未找到DT窗口后的新增Inductor文件（深度8范围内完整扫描）；Triton扫描超时，
不能排除其编译，更不能把17.96秒首批定性为重编译。首批保存的实际后缀宽度400，
其他长批没有完整源边界，未据猜测修改任何编译参数。07:12物理GPU31415/31257MiB，
cgroup218068828160B（203.09GiB）均为时刻值，不是峰值。原FA/FLA容差不变。

来源：row-prefix-owner-fresh-v1/completed-dt-readout-costs-1791328506-with-field-semantics-readonly.json
SHA cdb95e8f；actual-dt-batches-1791328581-readonly.json SHA7f23b0da；
compiler-cache-conclusion-readonly.json SHA478be7ea；phase-1791328350.json。
本次只读定位和记录，没有新生产实现、GPU测试、profiler、RPC或训练参数改动。

同次actor源核实：当前两worker实际use_fused_kernels=True、backend=torch、
use_remove_padding=False；原VERL qwen3_vl的forward_with_torch_backend复用原
FusedLinearForPPO，默认chunk_size512，反向重算分块logits/probabilities，并非
保留完整B×S×vocab张量。当前trajectory单token prompt使整数logits_to_keep等于
保留序列长度，不据此认定新bug。O和B4内部尾pad仍进入dense fused head，但没有
其成本占比，不当作已量出的大头，不关闭fused或新造head。来源actor-head-computation-
readonly-20261007.json SHA f8c08c43，含实际flags、原文件SHA和行号。本轮actor仍
未返回，不据静态设置提前确认完整更新时间或白化后的实际梯度。

## 2026-10-07 06:59 本次fresh已完成原旧概率重算，并开始正式DT批处理

同一driver2360541/source c83b96de、原worker2367855/2369144继续GPU4/5。
phase-1791327400已见两rank进入compute_dt_token_advantages；原当前类别组各
484 contrasts/121个B4，不是整轮总数。无完整trainer阶段timer，旧概率重算
仅按06:53:47仍在运行与06:56:40已进入DT的观察边界记录，不补造精确耗时。
原/proc maps只读回执mapped-library-1791327429证明两rank实际加载已验证
libfinite_row_query_starts.so，SHA4f42c391，与冻结源/数值验收绑定相同；加载
本身不等于一次有限计算完成。

phase-1791327553（06:59:13）已记录本次正式DT前三个B4完成：首批17.9598/
17.9442秒，后两批2.6746/2.6717、2.7118/2.7125秒，输入约3152–3188。
不将首批额外时间全部称作编译，不从短输入外推整轮/32k耗时。此时cgroup
234.06GiB，两DT worker PSS57.68/58.18GiB，是阶段时刻值不是峰值。
首批signed归因仍有原守恒诊断false（示例root .4666/sum .3065），只保留
定位信息，不引入新的容差、补偿或扩大已有FA/FLA原断言的验证范围。
本次全部DT、整批白化及actor更新仍待实际原调用完成；没有恢复旧训练检查点。

## 2026-10-07 06:53 本次fresh原采集完成，进入原旧概率重算

同一driver2360541/birth1791325655.01/source c83b96de继续GPU4/5。原LOOP在
06:51:46.682/.697分别完成rank0/1采集：106/112返回、14/8取消，共218条返回、
22条取消；按原完成条件停止，没有本地重新判定或补造轨迹。原transport最后
3837回复/763099生成token/1250.6秒不是trainer完整gen timer，不称纯decode。
phase-1791327227（06:53:47）显示两rank实际进程均进入
ray::WorkerDict.actor_rollout_compute_log_prob，cgroup131.64GiB。原source/配置
未改，未恢复/导出旧检查点，未启动其他任务或新GPU测试。

本次DT/整批白化/actor仍未执行完。实际trainer d35在373–374调用原
masked_whiten，1256–1271的compute_advantage返回后才可进入1285 actor RPC；
届时以同一当前作业的原actor入口证明执行，不将源码存在或DT结束提前称通过。
原生产没有raw_A或actor系数std/norm日志；原critic/advantages mean/max/min
是白化后系数，actor/grad_norm是完整原损失梯度。不要把d极值当raw_A标准差，
不要声称仅凭grad_norm证明任务梯度或学习质量已恢复，也不增加新hook来作门槛。

## 2026-10-07 06:45 当前fresh采样继续；补齐已有实际工作量计量

当前仍为driver2360541/birth1791325655.01/source c83b96de，GPU4/5，入口和
配置与06:27部署相同。06:45:26只读phase-1791326726观察到原采样仍在推进，
原transport累计约54.2万生成token/2818已返回请求、约873秒；该窗口包含
prefill/decode、环境和取消任务，不是纯decode或trainer完整gen timer。
物理GPU55309/55295MiB，cgroup134072762368B（约124.86GiB）；无当前DT或
actor完成记录。整批官方白化源码和旧v2执行证据保留，不能称新fresh已执行。
没有恢复/导出旧训练检查点、改参数、重启本次作业或新增GPU测试。

CPU只读原176请求/44个B4的计量表明，逐行缓存使39批减少历史重放，后缀
槽位166372降到117844（29.17%）。该范围不是本次正式迭代，也不是FLOP或
整轮加速比。剩余padding24826槽占新后缀21.07%；按后缀长度重排的槽位
下界虽可减少18648槽，但dense KV carrier增加28.11%，且改变原capture组。
未据此部署排序或添加新的性能要求。原metadata/query版本差异在回执中列明。

已保存热调用的阶段计量：实际B8约3.96秒中，原root forward及CPU检查点
0.86–0.87秒、32层forward重建0.94–1.05秒、有限传播1.49–1.60秒；精确
32768约8.30秒中有限传播4.16–4.19秒，其中8层FA约2.94秒。旧实际profile
没有finite前第三次重复FSDP gather的证据；这些值不支持再改卸载策略，
也不能把wall与stream差值直接称作搬运。原报告汇总范围/嵌套计时限制保留。

新证据：appworld-efficiency-20261007/saved-b8-phase-costs-readonly-20261007.json
SHA8f049083；individual-prefix-owner-candidate-v1/replay-slot-distribution-20261007/
prefix-replay-slot-distribution.json SHA bd9f5825。原预算、PPO公式、Q/V/A、
LoRA8/16、每卡B4均未改；新完整DT/白化/更新耗时继续以当前作业日志为准。

## 2026-10-07 06:27 AppWorld逐行缓存正式fresh入口已部署，无旧检查点

旧v3 driver1953903/birth1791321793.72/source97e3已按唯一身份停止，receipt
stop-before-row-prefix-1791325457/stopped.json，remaining_non_zombie=[]；没有
保存/导入检查点，没有选择其他任务树。新driver2360541/birth1791325655.01、
GPU4/5，output runs/appworld-fresh-row-prefix-20261007-v1/appworld-dt，source
c83b96debba90d85476e58b56b2c8c7f3900901f26ce41094e8f60ac489c7789。

源版本绑定已提交推送37b9085ab49babc582695cd0158ac1b837339484的验证候选；
CPU prepare a7055ef8、inputs7559b090、prepare4b69366c、starter29221c62。
原launcher1223字节不变，1550原/新绑定及entry95/VERL396/LOOP55/DT105文件
核对完成。hash变化仅原allowlist（producer/lease/runner/artifact/environment，
完整DT inventory另列answer/wrapper）；原sampling/全部非路径options与v3相同，
5个新路径字段在prepared中逐项列出。原预算200×40groups×6、mini32、epochs2、
train32000/max32768、rank8/alpha16、actor和DT每卡B4保持。fresh base，
resume_mode=disable，argv没有--resume-from，不读取任何旧训练检查点。

生产producer3e0c仅将两项已验owner参数透传原factory；runner5f14、answerd473、
artifact37a860、leaseb947、wrapper3e1d/library4f42通过canonical接口使用，不用
诊断class/sys.modules替换。正式VERL/LOOP/vLLM/HF/GDN/finiteFLA与trainer d35
未改。原整批有效action mask的masked_whiten出口保持；旧v2已真实执行过，
不能把本次初始化或源码存在当作本次已执行/梯度恢复/学习质量恢复。

独立实际B8热调用9.77/9.74→3.962秒只证明保存的这一批，exact32768热调用
8.30秒、allocated peak25351057920B只属于DT容量，不是新PPO或vLLM共存验收。
FA/FLA原断言和实际dtype来源、先前NaN checker失败与最小接口修复、跨bank
QVA差异都在下方06:20记录/verification-index，未增纠偏或整网容差。
原只读collector33e1705c已更新current_runtime；本次没有完整迭代计时。
06:30原worker只读RPC实际检查rank0/1 PID2367855/2369144，VERL actor3a65、
worker e5eb、torchfunctional079a与HF59f9均匹配。各卡mini16是原global32分发，
LoRA8/16、micro4、epochs2、entropy.001、dualclip3及32768一致；两owner
开关True和有限库4f42都在实际环境。DT仍lazy未import，没有主动调用DT，
trainer d35/373整批masked_whiten源核对不等于本次已经执行白化。
06:28原配置检查通过，仍初始化/register-center阶段，没有正式DT/actor完成。
06:31:05原LOOP采样服务启动，两rank原sample_tokens已进入；06:31:22物理GPU
4/5为55299/55287MiB、容器118.28GiB，是采样阶段时刻值非峰值；未见OOM。
原Ray metrics exporter连接警告及Qwen3.5未支持MFU计数保留在日志，未将其
误判为训练异常或据此修改运行。06:31:58原collector更新主snapshot，实际
workers2367855/2369144，源哈希核对无差异；DT/完整更新计时与学习恢复仍未完成。

更新时间比采样时间长尚不能判为正常：原actor每行是完整trajectory，DT每请求
是当前response；v2实测3198 responses，DT403个B4/卡；217trajectory原padding
到224，原PPO两epochs预计56个B4/卡、14同步optimizer steps。调用数不是FLOP
或时间比；v2 actor未完成，旧2495秒属于旧padding版，不是本次成对计时。
所有当前路径、birth、配置及来源见row-prefix-owner-fresh-v1/deployment-index.json。


## 2026-10-07 06:20 逐行缓存与存储组合完成有界验收，生产接线仅prepared

正式v3仍1953903/birth1791321793.72、GPU4/5、source97e3cb75，没有导入旧训练
检查点。旧v2已实际执行整批官方masked_whiten；v3继续相同trainer d35ddd26，
06:11仍DT第二组44/46，无本次actor入口/完整gen-update timer。此前2495秒actor
属于旧padding版本，不能充作当前正常性能证明或本次更新时间。

隔离real-b8-v3已完成，非仍在运行：result f637f0e5，runner5f14/answerd473、
未压缩artifact50af/lease54ad，wrapper3e1d/library4f42。原same-input热调用
9.77/9.74秒→逐行3.93/3.94秒。实际FA varlen原断言通过，原7ff保存操作数的
每行7项coincident断言通过；实际非零B4五项输出与原scalar逐行bitwise相同。
GDN旧checker对FP16全零padding直接F.normalize产生NaN，实际native张量全finite。
新d107只接原FLA l2norm_fwd后执行不变reference/o-ht断言，两rank通过；本机
canonical checker c2853a15已同样修接口。旧5c192源/失败保留，没有nan_to_num或
放宽tol。上述均为对应算子范围，不是新整网容差。

组合artifact37a860/leaseb947复用已验证4a/6d存储owner，只存消费的边界行。
实际176请求CPU合同a3b469：原literal IDs/边界/last-write保持、HF Cache 32项
字节检查通过、CUDA未初始化。隔离combined-capacity-b8-v1原verify22bc、
diag9b1d、prepared0189，PID2235930/birth1791324452.14已自行完成并释放GPU2/3。
总体完成result87deb9be；原模型初始化顺序真实B8冷暖→exact32768冷暖，无CP/
optimizer step/profiler。真实热调用3.962秒，32768热调用8.30秒/每卡B4。
32768 torch allocated peak25351057920B是该DT容量范围，不是新PPO更新或vLLM
共存峰值验收。真实行PSS13289958400/13900460032B，32k PSS约14.4GB。

跨bank/layout的A最大差.02030/.03152、1012有效action的方向cosine.989422、
L2比.98455均保留；native端点已有.16量级变化，不能归咎单FA finite或把缓存
dtype转换当作全部原因。组合对未压缩row A/V最大差.00194/.00140，同bank
重复也有非零差；不加整网通过阈值。原Q/V/A/奖励/观察mask/PPO公式未改。

production-wiring-v1 prepared6c65，producer3e0c只给原prefix factory透传已有
individual_prefixes/boundary_row_storage参数，默认不进入；canonical路径直接
指向上述owner，不使用诊断class/sys.modules替换。environment4ff仅改库/sha与
两开关；原factory/profile、VERL/LOOP/trainer不变。禁CUDA真实14模块import
f1f481通过，未构造模型；此时仍仅prepared，不能当部署或正式DT已执行。
来源索引individual-prefix-owner-candidate-v1/verification-index.json；所有源、
实际dtype/原断言、失败和测量范围在对应冻结收据，PLAN没有修改。


## 2026-10-07 05:53 逐行前缀候选进入真实B8；正式owner与整批白化保持

正式v3仍driver1953903/birth1791321793.72、GPU4/5、source97e3cb75，原冻结
3cd90b2d/preparedf5232ac1的源字节不变。整批官方masked_whiten已在旧v2实际执行，
本次fresh继续相同trainer；不能把尺度修复生效称为学习质量恢复。不载入旧训练
检查点。05:46只读LOOP采样完成207/240，运输3695回复/737768 tokens/1168.0s；
尚无本次完整gen/update_actor时间比，旧actor2495秒也不能作为当前版本实测。

逐行切点隔离候选实际176请求/44个B4 CPU接口检查通过：target label/sample/
paired顺序、suffix位置、原action绝对索引与原NativeTargetLogitRows保持。小CPU
Cache接口单独标注，没用它代替真实B8/数值验收。source-review f6b97835完整绑定
runner5f14bb3c、answerd47333ea、artifact50af8daf、lease54ad5ae1；new finite wrapper
3e1d6103/library4f42c391、kernel CUDA9ebcef18只扩展逐行query starts表示，公式不变。
原FA真实saved operands的官方coincident检查与非零layout逐值对照见05:13记录；
这些不能冒称真实异质B8/32k/完整信用已验收。

第一次隔离probe2138789之前的v1/2105248已在DT bank前因导入路径断言退出：
原producer212再次prepend DT_ROOT/clean/qwen35，遮盖out的answer接口。原错误
日志保留；root随后停止请求发现PID已退出，实际上没发signal。v2仅prepared，
不覆盖其源或把它称为执行。v3复用原已存在的链接owner目录接口，只有目标/
artifact指向惰性扩展，bare runner仍原e9c757，原wrapper不遮盖；未改正式目录。

隔离B8 v3 driver2138789/birth1791323526.94、GPU2/3，prepared e30a1f17，原verify
22bc698c字节不变，原88请求/rank的SHA8714abf2/1fdc1eb0不变。diagnose cdfe43a5
只改诊断分发与读取表示：前4次baseline/row冷暖分别测量，第5次组合原GDN0、
varlen FA3与真实finite输入/LSE/upstream被动观察，不算速度。05:53实际两rank
inspect/SHA确认answer/artifact/lease及6个DI身份，进入shared_cold，尚无完成结果。
此时owned子树PSS36.98GiB、容器168.89GiB；GPU2/3约17.9/17.7GiB，资源只作
该时刻观测，不称峰值或容量验收。没有挂live tracer、优化器step或新容差。

新FA varlen离线接口37c2c9fb复用原a290e11c reference/原断言：原packed每行完整
Q/K取出后再还原整批比较，不能把KV carrier中间洞直接喂给原reference。GDN0
仍用原5c1924ce/check_saved_fla与原FLA断言。原fa/LSE/finite非零输出分别报告，
不把native forward断言扩大为完整DT验收。候选仍不在默认launch路径。
收据位于appworld-efficiency-20261007/individual-prefix-owner-candidate-v1/，
结果索引experiments/rl/results_appworld_efficiency_20261007.json；正式事实继续以
v3 job/source/原worker日志为准。PLAN、PPO损失、任务预算、LoRA8/16、每卡B4未改。


## 2026-10-07 05:23 AppWorld原冻结owner fresh v3已提交，采样开始

旧v2 actor中途退出保留在下方。复用原launch_appworld_native.py1223c007与原env
beb9c001，从原基础权重fresh启动；没有--resume-from，原resume_mode=disable。
新driver1953903/birth1791321793.72，GPU4/5；worker1959905/1961414。
运行目录/runs/appworld-fresh-native-conv-canonical-20261007-v3/appworld-dt。
新source97e3cb754f505b79a52d3dc3464b9027ae7bdc38e7074b866f80dc7624be3481；
继承原冻结3cd90b2d与preparedf5232ac1，95entry/396VERL/55LOOP/105DT文件逐项
SHA无差异。新source只变运行身份与来源记录，不是新的数值算法版本。
提交5155c9b3保留原guard，旧一次性v2脚本未原样重跑，不重prepare、不改参数。
本次完整launch对照v2，只有4个output/visibility目录字段变化。

实际原worker RPC已核LoRA8/alpha16、actor micro4、PPO epochs2、entropy.001、
dualclip3、sharedpadding1、canonical HF59f9、actor3a65/fsdp e5eb/torchfunction079a；
原trainer d35ddd26/373行整批masked_whiten保持。v2该白化实际已执行；本次
仍在采样，尚未执行DT/actor更新，不能冒称本次梯度或学习恢复。DT/vLLM lazy
模块没有出现在此时worker sys.modules，未将其冒称已执行的真实DT导入证据。
预算仍200×40groups×6、mini32、epoch2、40turns、32runners/rank、train32000/
上限32768，没改rank/alpha/B4，也没接storage或row-cut候选。

05:27原vLLM已初始化、LOOP任务采样服务开始；物理GPU4/5分别48501/48485MiB。
容器总122789081088B（114.36GiB），anonymous约89.98GiB；未相加fork RSS。
没有完整rollout/DT/actor时间、梯度或新检查点；进程存活不是训练健康证明。
不再live attach mcTracer；候选继续在隔离目录做原操作数、官方断言与表示对照。
本机snapshot更新为原只读collector独立输出e094acb7；collector33e1705c字节未改，
仅内存重定向保存目的地，IO重定向收据可查。来源都在
research/temporary/rl_upstream_alignment_20260929/appworld-efficiency-20261007/
native-conv-canonical-owner-v3/，旧snapshot/失败日志仍保留，不当新运行事实。


## 2026-10-07 05:13 逐行切点finite-FA接口通过实际操作数检查，未接入训练

正式driver1181392/worker仍已退出；本轮没有重启、检查点加载/导出/恢复或参数修改。
隔离candidate=/candidates/appworld-row-cuts-finite-20261007-v1，CUDA9ebcef18、
wrapper3e1d6103，复用既有prepare_remote.py699b3f1b原cucc flags/头文件，49.32秒
编译成功，library4f42c391055afec0a0fee9ee698c0820163ff413e42f1c4909b2961ce81e5157。
编译实际子树PSS采样峰1.017GB，无GPU launch；初次诊断模板替换误中STDOUT、
第二次调用缺失/usr/bin/time，均在compiler启动前失败，已保存stderr并改用已装
psutil PSS采样，不安装工具、不清缓存、不替换正式owner。没有称失败为通过。

实际GPU2原saved-operand检查15.46秒退出0，reuse actual-dt-layer3-boundaries.pt
SHA0ade21d748c0fa37bd46a08ee3c1452d6cb298dac36b0fed6ad7c7be4b9978aa。
单decoder3真实八操作数：Q BF16[4,16,527,256]、K/V BF16[4,4,911,256]、u FP32
同Q、LSE FP32[4,16,527]，原cut384/coefficient start433。不是模型权重或checkpoint。
原verify_saved_fa_dtypes.py7ff11d9d字节未改，只注入隔离library和row-layout表示，
使用原FA2.6.3 source a290e11c的attention_ref/普通低精度参照和原断言：output2倍、
dq/dk/dv3倍。原native四项及coincident finite三项全部通过；这个官方误差断言
只用于相同端点的普通导数极限，绝不作为非零finite的新误差门槛。

非零真实操作数仍保留原完整合法范围433以后：新scalar与新row ABI（四行cut384、
Q stride527）对原5d2af760库的dq/dk/dv/tau/center均逐值相同，max_abs0。
混合cut384/448/512/576只补充核查索引，保留范围同步缩至433/448/512/576后，
五输出也相同。此范围不是原四行训练action范围，不能当原完整credit保持或
真实异质cut/runner验证。尚无真实异质prefix-cache/mask/position/target/lease接入、
新FLA/32k B8/full DT速度验收；候选仍未部署，原冻结3cd90b2d不替换。

单GPU检查采样物理峰2556MiB、子树PSS10.015GB，Torch峰allocated1.136GB/
reserved1.439GB；不同资源口径分开。没有模型forward/训练/额外rollout/optimizer。
实际GPU计算由原FA reference和finite owner完成，未使用随机u补造异质训练样本。
原源、编译、实际dtype、原断言、所有残差与资源由kernel-verification.json索引。

排序owner只读复查也已完成：原consumer按context稳定排序，原DataProto.reorder/
等行数跨卡分区已接；动态microbatch不能直接保B4。已有88行按切点排序没有大
收益（约1.58% suffix槽减少且rank1反增2.91%），不新增影子调度器。逐行cut后
FA phase0/1仍必须读完整事实KV，不能把必要score/LSE计算当可删历史重算。


## 2026-10-07 04:44 原actor期间worker意外退出；正式任务已停止

driver1181392/birth1791314386.58及worker1189140/1190575均已退出，GPU4/5释放。
04:44:14.041 rank0断连，随后原Ray清理peer/TaskRunner；没有完成本轮、没有原
完整actor计时/梯度指标/检查点。04:51:21原只读recorder确认alive=false，来源
仍3cd90b2d；远端manifest旧declared status不作为存活事实。新 dated snapshot
continuation-1791314906/current_runtime-after-exit-1791319881.json保留所有源/哈希。

必须保留诊断操作与时序：本任务04:43:23使用已安装官方mcTracer附加rank0，
首次因绝对odname被拼在cwd下导致输出初始化失败；04:43:43改相对目录重试，
04:44:03按官方Ctrl+T停止，04:44:04退出0且当时target尚存活，约10秒后rank0
断连。没有新Python traceback、dmesg OOM/GPU故障或明确退出信号；cgroup
failcnt0/oom_kill0。不能断言profiler导致正式退出，也不能隐去这项相邻操作。

仅对空闲GPU2的两个同构128x128 BF16 Torch客户端做有界隔离：一次正常附加
后两者正常结束；另一轮附加初始化因port-file缺失失败，被附加客户端睡眠卡在
CUDA matmul，对照正常结束。这证明附加失败存在挂起风险，未复现正式退出或
最初odname错误。精确PID1746517/birth1791319799.2/命令SHA核对后已清理本任务
该诊断子进程，未向训练或其他用户进程发信号。不再对正式worker live attach。
原始worker/raylet/dmesg、两次附加及隔离/cleanup回执全部由结果JSON索引。

本轮DT已完整结束：每卡403次B4，readout2499.85/2498.78秒，bank382.25/
381.55秒，B4 sum2114.36/2113.99秒，双卡不相加。整批原masked_whiten已在
原compute_advantage373实际经过，然后进入原actor1285；原d/Q/V/A/returns不改。
这证明批准的尺度处理生效，不证明学习退化恢复。04:39:40 actor局部变量两卡
epoch0/batch_idx5，至少已完成5/14 optimizer step；不能把未完成更新称为完整
计时或把旧2495秒充作本轮结果。原actor仍每卡56 B4/14 step的既定两epoch负载。

实际GPU trace为20.464秒有界、instrumented窗口：GEMM union9.809秒，FA1.453秒，
collective0.387秒，copy union1.049秒；总GPU active20.383秒。CPU nonzero/stream
synchronize等待可能是在等前序GPU计算，不能另算一笔耗时。该证据不支持把
offload搬运或mask nonzero直接当主要浪费改掉，也不是完整actor吞吐验收。
原trace65926059字节，SHA688b055c09f2dbc585faba00eaa30153f2532bb4bdd699b53a19a61616ec28dc。

七组完整suffix token槽已核算：历史尾部/对齐/padding占39.58%/39.03%；这不是
等额FLOP/墙钟承诺。当前B4统一最早切点确会多算其他行的事实历史段；原HF
位置/mask/varlen接口可组合，但现有lease/runner/finite-FA ABI还不能完整表达
逐行切点。隔离owner候选只加per-row起点与独立Q stride，三个旧scalar ABI
函数体/签名不变，CPU5项和独立只读索引审查通过；没有CUDA编译/GPU数值或
官方容差/完整B4接入验收，不部署。旧88行metadata的suffix槽减少31.01%/27.43%
仅是表示上界，不能报本轮整体提速。存储行优化仍verified/prepared未部署。

没有加载、导出或恢复任何检查点，没有重启正式任务，没有改rank8/alpha16/
每卡B4/原PPO与任务配置/FA/FLA容差。当前受影响的正式更新停止；不把诊断或
候选当修复完成。出处results_appworld_efficiency_20261007.json最新退出与操作索引。



## 2026-10-07 04:29 本轮DT结束，已实际经过白化进入原actor

同一driver1181392/birth1791314386.58保持。04:29:02两worker均进入原
actor_rollout_update_actor；04:29:29非阻塞原TaskRunner栈为ray_trainer.py:1285。
实际trainer仍d35ddd26，fit1256调用compute_advantage后才到1285；DT分支373
唯一调用原masked_whiten(raw_A,response_mask)，374仅交给actor advantages，
375保留原returns。因此本轮已经过所批准的整批白化出口，不能仍报告为
“仅启用、尚未执行”。这不等于已拿到本轮梯度范数或证明学习退化恢复；
完整原actor计时/指标尚未返回，不用旧2495秒代替。

七组完整原DT日志：每rank1612 contrasts、403 B4，readout分别2499.8495/
2498.7801秒（约41.7分钟，双卡并行不相加）；bank382.2528/381.5483秒，
原B4 wall合计2114.3560/2113.9928秒，未解释差额仅3.2407/3.2389秒。
这定位了本轮时间大头在403次B4调用，而非外层RPC或几十秒以下的复制。
3198个原response对应两卡3224运输槽，原padding26槽，不是平方请求展开。
actor仍按217条整轨迹pad224、每卡56次B4前后向/14 optimizer step运行；
两种调用单位和token范围不同，次数比7.20不是FLOP或速度比。

原始完成日志前缀SHA、逐组计时、实际trainer片段及TaskRunner栈见
continuation-1791314906/completed-DT-and-actor-entry-1791318569-readonly.json。
实际阶段与资源见official-phase-1791318542.json：container344.43GB，
failcnt0、oom_kill0；该瞬间资源不是32k峰值验收。没有加载/导出检查点，
没有重启或改训练参数、PPO/FA/FLA、容差及存储owner部署状态。

## 2026-10-07 04:25 当前DT计算与官方actor重算审查

31146同一driver1181392/birth1791314386.58、source3cd90b2d、GPU4/5仍从base运行，
resume disable；本次不加载/导出检查点、不重启、不替换冻结owner。
原第6组两卡71个B4完成，第7组各136 contrasts/34 B4开始；尚未进入actor，
不能拿旧2495秒判定本次actor>sampling，不能声称本次白化更新已经完成。
实际trainer d35ddd26仍在整批action mask上唯一调用原masked_whiten，
随后原DP/mini/B4/update_policy；原d/Q/V/A/returns和LoRA8/16/B4不变。

当前前5个完整组各1192 contrasts/298 B4的原字段已核算：单端点suffix槽数
996996/978580，其中action230753/234665、query+target372184/372178，
历史尾部+64对齐+右padding394059/371737，占39.52%/37.99%。右padding本身
119620/113500，其余历史尾部与对齐尚不能分开。这是token工作量，不能称为
可删墙钟/FLOPs比例；旧轮44–45%不代填本轮。原字段两种query计数逐组相等。
回执current-native-slot-accounting-1791318168-readonly.json保留原日志前缀SHA。

实际owner审查排除两项具体重复：fsdp_workers e5eb4afc:347–350启用原FSDP2
CPUOffloadPolicy后关闭手工参数/optimizer offload；activation_offload82904250:
538–544先禁用HF checkpoint，再装原activation-offload checkpoint。实际
enable_activation_offloading与update_policy AST均与固定VERL20bd331相同；
actor3a65e173已使用共同右padding裁切和原fused head。此为源码行为审查，
不是新PPO数值容差或完整更新计时。来源current-actor-offload-owner-review-readonly.json。

runner e9c75764的root保存32层CPU输入，replay逐层恢复finite实际操作数；
其计算换内存的用途与checkpoint重算对应，不能整段当可删浪费。既有GPU
root capture容量尚未覆盖最大真实后缀；CPU版本实测回退38%，继续拒绝。
不同response的EOS端点/query不同；PackedAnswerTargets多offset合为一个
dense seed，没有独立query归因轴，不能把它当成合并独立response的接口。
本次未发现可直接部署的大计算提速，不把“未发现”写成没有改进空间。
出处current-root-replay-reuse-audit-readonly.json；两项报告及来源SHA由
results_appworld_efficiency_20261007.json索引。

状态行存储修复仍verified/prepared，不做未经验证的canonical模块热交换：
官方Worker RPC只提供串行调用边界，不能单独证明完整artifact/lease交换正确。
当前任务不中断；不会为无明显热耗时收益而丢弃这批已计算轨迹，也不会为
应用存储选项重新引入用户已禁止的旧checkpoint恢复。

## 2026-10-07 04:13 新冻结存储入口准备完成，未重启当前正式训练

新的entry=/candidates/appworld-native-boundary-row-storage-20261007-v1/entry，
prepared fd233d09（完整路径仍在MetaX固定ROOT下）；复用通过实际B8的DT linked tree，
只替artifact4a461f21、lease6d2aecb0及producer的原factory绑定partial(True)。
新producer SHA 900f7ae41b6840ed0834d9744e275adc6f12b51d78d04069b636cb94c809d0bd。
原VERL/HF/LOOP/environment.json、白化、rank8/alpha16/B4、budget与原launcher不变。
CPU实际import身份通过，原AppWorld workload/native validator 1项通过（11.04秒），
无model/GPU/checkpoint加载，无job启动或停止。准备脚本e12c54d6；
完整source库存按prior source的原键重哈希，只有明确3文件不同。
prepared-only不冒充deployed：active-training仍driver1181392/source3cd90b2d。
当前已采集/计算的正式轨迹保留，不为纯存储节省重启本轮；待当前更新确认后再切换。

## 2026-10-07 04:07 状态行存储修复通过真实B8运输检查，正式进程未切换

实际driver1181392/出生1791314386.58、GPU4/5、source3cd90b2d保持，
从base启动、resume disable，无检查点加载。VERL trainer d35ddd26仅在完整
action mask上调用一次原masked_whiten(373行)，原fit1256后才update_actor1285；
原Q/V/A/returns不改。04:05:59尚未到actor，不能把启用等同于已完成本轮白化更新
或学习退化恢复。当前前三个DT组已完成，每卡736contrast/184B4/1125.36秒，
其中bank193.69秒、B4sum930.30秒，第4组380contrast/95B4刚开始。
3198个有效response用于DT，actor输入217个完整trajectory；原adjust_batch补至224，
DP每卡112、epoch2、mini每卡16、micro4，待执行每卡56次B4 fwd/backward和14次
optimizer step，oldlogprob每卡28次B4。不能拿旧2495秒替代本次actor结果。

有界诊断1442344/出生1791316826.03、GPU2/3，同当前base/owner、offset84每卡B4，
没有checkpoint/backward/optimizer/额外rollout。候选artifact4a461f21、lease6d2aecb0
默认None/False不改原路径。实际相同capture所有被消费BF16卷积/FP32递推状态逐字节
相同：rank0每类1896行、rank1每类1872行；全88请求的消费身份/边界/row映射检查通过。
原FA KV、完整B4递推、runner e9c75764/GDN ef55ce08/canonicalHF59f9c339不变。
此为精确表示运输检查，不新增整网FP容差，不修改官方FA/FLA断言。

每卡完整bank去重存储16.8815→10.3934GB、17.3219→10.3667GB，
减少6.4881/6.9552GB（38.43%/40.15%），未消费GDN行归零。
热wall9.7783/9.7888→9.7675/9.7748秒，未证明有意义的热计算提速。
OFF先捕获42.45秒、ON后捕获24.86秒混有首次/顺序差异及1.42秒字节观测，
不能称为独立提速对照。PSS23.785/24.270→17.997/18.251GB，仍含原allocator驻留。
不同capture/重复warm Q/V/A最大残差0.00189224只记录，不裁剪、不纠偏、
不自设通过阈值。原CPU接口7项通过；整轮训练、32k更新和学习效果不由此新增背书。
代码进入主分支的默认惰性接口；本次正式冻结06f/d5入口未覆盖，尚未部署该存储选项。
原始及汇总：native-prefix-boundary-rows-20261007-v1/accepted-storage-transport.json
与runtime-v1/runtime-summary.json。

现有实际热trace每卡67次FSDP allgather，顺序root/embed/32层正向/root/32层逆向；
每层正好2次，finite前32个prepare范围collective为0。producer2c01已用原
set_reshard_after_forward(False)保留replay参数至finite后，PyTorch原unshard
对已展开参数直接返回。未发现第三次收集；3.6368秒含另一rank等待，不能当可删成本。
详见current-base-hot-profile-v1/fsdp-parameter-owner-review.json。

## 2026-10-07 03:42 当前真实B8 profile完成，白化入口保持

当前正式AppWorld仍driver1181392/出生1791314386.58、GPU4/5，从base启动，
resume disable、无检查点加载。source3cd90b2d、preparedf5232ac1、
canonical HF59f9c339、DT runner e9c75764/GDN ef55ce08、
VERL trainer d35ddd26/actor3a65e173/worker e5eb4afc、helper079a6d20不变。
白化只在原trainer完整action mask上调用一次原masked_whiten，然后原DP/mini/B4；
原d/Q/V/A/returns不改，无GAE/value model。当前仍采样，未观测本次白化更新；
不能以入口正确、历史梯度增大或进程存活宣称学习退化已经修复。

GPU2/3原有界诊断1276181/出生1791315255.77已完成退出，物理显存回到860MiB。
当前正式相同owner/base/JSON conv=true、实际offset84:88、每卡B4，
长度rank0 12007/12423/12907/13482，rank1 12214/12633/12822/13230；
原CPU接口7项通过。无采样/actor backward/optimizer/checkpoint调用，未改生产路径。
非instrumented热wall9.76066/9.73879秒；cold67.80545/67.76047秒含首次开销，
profiler wall28.52419/27.56367秒含instrumentation，不能当成热调用退化。
原CUDA-event范围：root2.12729/2.12106秒、逐层原forward重放2.07505/3.30085秒、
有限FA decoder1.98616/1.48796秒、有限GDN decoder2.43970/1.90637秒、
原FA LSE .24429/.18201秒。device sum可重叠，不是可删除墙钟。

原trace定位FA有限核三个阶段各8次：rank0 .48504/1.09360/.12862秒，
rank1 .37032/.82508/.07598秒；不是重复两次endpoint。精确copy账目显示
root检查点各上传两次，但其中一半只约.092/.068秒device time；
不是本次主要瓶颈，不为它引入生产改造。保留未知owner内核分类，不凭名称猜用途。
rawtrace仍在远端；local仅small原prepared/job、CPUxml和只读分析，
current-base-hot-profile-v1/read-only-current-base-profile-analysis.json SHA772a7028，
原结果/实际source SHA包含在该回执。诊断stager144b4e3b/diagnose115cd56d；
analyzer不导入torch或调用模型。无新数值断言、纠偏或容差。

03:40:14正式原LOOP collected108/89，共197/240；运输3270回复、635022生成token、
1033.2秒，含prefill/decode/交互，非纯decode。尚无DT/actor完成或新检查点。
container127.09GB/966.37GB、failcnt0，物理4/5为55309/55243MiB，
仅采样phase观测，不是整轮峰值。原phase来源continuation-1791314906/
official-phase-1791315609.json；只读current_runtime已于1791315689.64刷新。

“actor>采样”不能单独判为异常：官方actor两epoch、完整轨迹前后向、checkpoint
重算与原offload。但旧step9的2495.232秒含已证明的共同右侧padding浪费，不能
称正常或当新版本耗时。对应原compute_log_prob同权重实际输入热14.6427→6.1881秒
仅证明该forward修复；当前actor3a65已启用，仍等本次原阶段计时，不外推整轮加速。

03:42:42原LOOP达到原完成条件：116/101、共217/240轨迹、取消4/19；
保留3198个response、607470policy token，最长context25047；运输累计3495回复、
693496token/1178.7秒。03:44:48原两worker实际在原compute_log_prob/HF FA调用，
不是DT卡住；nonblocking py-spy出处formal-stack-1791315884.json。
原TaskRunner打印reward_extra_info缺字段是原compute_reward对EpisodeRewardManager
rm_scores字典的兼容回退，返回同一个reward张量/空metadata，无环境或模型重算；
reward37686e6a/episode bb7e5f85与source及官方AST一致，不为它改奖励或停作业。
当前FA adapterf5ea2f67/CUDA d408cce5/库5d2af760与原release manifest闭合，
原FA断言回执绑定同库；三阶段各8次及compact GQA/cached query均已接，
没有发现漏接的已验更快路径或可删阶段，故本轮未据此改FA数值/内核。
出处current-base-hot-profile-v1/finite-FA-current-owner-review.json SHA c275c0f5。

03:47:59本次DT source原日志：3198 unique/retained、nonzero3198、skipped0；
首alphabet两rank各520 contrasts/130 B4，events37/38是reward事件数，不能
当全轮组数、完成batch或canonical UID数。source当前前缀factory与06f/d5对齐。
03:50:32 container365.15GB，匿名340.49GB、file24.40GB；worker PSS123.42/122.79GB，
物理4/5为25911/25963MiB，OOM/failcnt0。非阻塞栈已在runner418有限循环后的
输入贡献收缩，不再在bank准备；不能由日志尚无B4完成判卡死。
只读memory-stack-1791316232.json SHA85354e38，当前尚不能分开live bank与allocator。

进一步源证据：capture不保留root hidden checkpoint；已证浪费是B4 capture所有UID
needed边界的union，每boundary却保存整B4 GDN状态。原真实88 inventory中
rank0/1未消费GDN6.488/6.955GB，不能把这份比例外推本次整轮内存。
已有10/05暂停候选可只在现有导出边界保存实际需要row，原B4 forward/FA KV/
FLA FP32递推不变；本次复用该候选准备隔离验证，默认及当前formal owner均未改，
未把候选称已接受/部署。整体证据current-producer-workload-review.json SHA33542524。
原88请求的CPU分组核算：按source_start反增3.76%，按(start//64,context)仅减1.58%
且另一rank更差；不为这项小/负收益改分组，不把88比例外推473-B4旧正式。


## 2026-10-07 03:19 最新：移除任务服务继承的额外导入

下方1119928不是健康训练：它在原LOOP构造64个runner时全部readiness失败，
192次server launch/128次restart/0ready，尚未采样、DT或更新；已退出、八卡无进程。
原Popen不传env，新的namespace sitecustomize因而进入每个AppWorld解释器启动。
同实际CLI shebang解释器、纯CPU的OFF/ON：.04018/4.03122秒、19,308/78,872KiB；
ON导入Transformers/Qwen包，未导入torch或Qwen modeling。不能写成torch急导入。
约3.991秒额外开销占原5×1秒sleep预算大部；requests无timeout，故5秒非绝对墙钟。
这不是修改LOOP wait/retry的理由，原参数与服务行为保留。

已将相同验过的最小HF owner patch f7e1a804→59f9c339原子写入本任务private VENV
原canonical modeling_qwen3_5.py，原before文件及SHA保存于native-conv-canonical-owner-v2。
不是重装包/替代模型/新loader。生产PYTHONPATH删除isolation父目录，移除两个
namespace loader变量；DT linked tree/runner e9c75764/GDN ef55ce08及producer2c01c47e
保持不变。新CPU实际import从原installed canonical路径得到59f9c339，CUDA未初始化。
原VERL、LOOP、LoRA8/16、每卡B4、epoch2/全部采样预算和DT Q/V/A及白化不变。

新base driver1181392、出生1791314386.58、GPU4/5，run
appworld-fresh-native-conv-canonical-20261007-v2/appworld-dt；resume disable，不加载检查点。
source SHA3cd90b2db649cd477bc21398e7677dc8ea5534d5230296fc5f63c04037cd3427；
prepared SHAf5232ac1dc7cc13a6142902488fdb28243f86dd7de91774bb91c3279fe5c0ec3。
原isolated_owners回执仍作为DT tree和历史诊断身份，不能把其中旧HF/sitecustomize路径
当当前导入；当前路径以canonical_HF_owner和CPU实际import回执为准。
03:20只读RPC在两个actor尚未注册时退出，未进行模型调用，不把诊断未就绪当训练失败。
原AppWorldInterface真实单服务已ready：constructor3.2181秒、原dummy GET200、close_server.464秒，PID1198262退出0/_server清空；原默认等待和重试未改，未执行任务/模型。单服务不是64服务池验收。03:22:51新job仍引擎初始化，物理4/5卡47729/47713MiB，container88.64GB/966.37GB，OOM/failcnt0；非DT峰值。03:23:47原池已64/64ready、64次launch、0restart/连接错误，实际loop_transport开始采样（latest两rank16/19项回复、2044/2495生成token；含交互，非纯decode）。03:24原worker1189140/1190575实际导入installed canonical HF59f9、actor3a65/worker e5eb/helper079a及两个vLLM已验SHA7e55/37db，namespace flag为None；LoRA8/16/B4/epoch2/entropy.001/dualclip3同原。完整白化更新尚未发生，不称学习退化已修复。
本次实际运行源以这些哈希为准；ae35ecc只记录前一次source，不能冒充新source。


## 2026-10-07 03:10 最新：官方卷积 initial_states 接口已部署，仍无检查点恢复

覆盖下方722849/v2的当前状态。旧base作业722849在1791313817.799完整停止，
remaining_non_zombie为空；未保存/加载检查点。新driver1119928、出生1791313820.3，
GPU4/5，run appworld-fresh-native-conv-20261007-v1/appworld-dt，从基础Qwen3.5-9B启动。
SQL/TextCraft未重启，其他六卡空闲。正式entry为appworld-native-conv-initial-states-
20261007-v1/entry；VERL与LOOP仍原已验路径，DT为该候选isolated-owners/deltatrace。
source SHA b8ae7e7da7cf79aff067f8524027140a091ca2c1374ff904ab17c83249e95901，
prepared e9dacc24168d1849ec9b8da9b1024bed7e0b349ce308b70aba66ab02c52b1644，
submit f214e078b7eb021d0baac5af91532d1c659db2fb636b9cbd2227363b404f342f。
Git提交前继承c1a079f，实际运行版本由上述源码/脚本SHA单独固定，不能用新提交号
反推这些作业已导入新源码。两个DT owner实际e9c75764/ef55ce08、私有canonical HF
59f9c339；原installed HF文件未改。105项DT源103项保持、仅两owner不同。
VERL白化trainer d35ddd26、actor3a65e173、worker e5eb4afc、原PPO/core及LOOP均未变。
LoRA8/16、每卡B4/双卡B8、原采样/预算/epoch2/entropy.001/dualclip3不变。
本机两owner已保存相同验过字节；producer仅加默认不传的可选keyword。本机producer
与实际冻结producer原有其他差异仍保留，不能用本机源SHA冒充实际2c01c47e。

此修复使用原causal_conv1d_fn(initial_states=...)，将原缓存末3项在原inplace
更新前保留，避免每层大张量torch.cat；原缓存生命周期与卷积/finite公式不改。
真实12k–13.5k同B8、同bank：critical总11.32949→9.79171秒，省13.57%；
root与逐层原forward重算各省约.67秒，finite省.18–.20秒。原bank只建一次，
80阶段计数同。primitive原v1.5.0 BF16 output/dx断言各rank9项通过，
新旧输出逐值同、dx最大差9.54e-7/1.91e-6；BF16原rtol=.01/atol=.05，
不扩展为整DT的FA/FLA容差。原FSDP预取无总收益，未部署。

原capacity fixture实际32768、每卡4条/action512的三模式都结束、Q/V/A全finite、
无OOM；同bankOFFwarm→ON两rank省4.36%/4.62%。运行中mx-smi采样44520MiB/卡，
非连续峰值；MetaX torch allocator raw计数不作为物理占用。原CPU测试15项无失败跳过。
这是DT容量，不称PPO新更新或真实任务效果。原actor B8/32768更新/LoRA同步回执仍
单独绑定3a65e173，不因本次容量重新发明整网两次更新门槛。

原readout .02守恒诊断在OFFcold/OFFwarm/ON均有旧残差：rank0各2/4、rank1各4/4，
不是FA/FLA官方断言，不是训练gate；原真实长case也存在。OFFwarm残差最大
.17655/.11852，ON .17342/.11970；原cold/warm变化与OFF/ON变化单独保留。
没有加归因倍率、剪裁、改阈值或把该诊断称作通过；不能据此归罪新conv接口。
完整来源在results_appworld_efficiency_20261007.json、runtime-capacity-20261007-v1/
read-only-capacity-analysis.json与native-conv-production-deployed-v1只读review。
03:13原worker RPC核实1128035/1130737实际导入HF59f9、actor3a65、worker e5eb、原helper079a；loader/JSON/LoRA8/16/B4均同prepared。DT producer尚未lazy初始化，未将CPU源检查称实际DT执行。第一次RPC仅因诊断输出目录缺失而写回失败，零模型调用；原失败保存，mkdir后原RPC成功，无训练源码修改。当前新作业仍初始化；实际白化后的更新及学习效果尚未观察，不把源码启用当质量修复。


## 2026-10-07 最新：31146端口、基础模型新训练，不恢复旧检查点

本节覆盖下面所有历史恢复说明。用户明确禁止恢复学习目标改变前的检查点。
原恢复job634197已停止，旧PID在新端口不存在。新AppWorld只用基础Qwen3.5-9B；
launch_appworld_native.py默认resume_mode=disable，不传resume-from。
当前正式driver722849、出生1791310127.33、GPU4/5；输出runs/
appworld-fresh-official-padding-20261007-v2/appworld-dt。SQL/TextCraft旧PID均已退出，
未擅自重启；GPU2/3用于隔离长输入诊断，0/1与6/7不用。

entry和VERL仍为candidates/appworld-fresh-official-padding-20261007-v1/entry、verl，
run-v2不是另一份算法。DT仍为appworld-native-prefix-resume-20261005-v2/deltatrace；
LOOP仍为appworld-loop-readiness-20261006-v1/loop-owner。实际source
SHA dbb3dbf887293680757a44085fdc1c31d310c7b8731bea21c7e321149f2342d5；
prepared SHA6779adfebd0b8d74626970f2bf7567770ea91c2b680b6b4efad89462c083839a。
submit-v2 SHA28c81b66cd3f435bc851415e84ff3c34953287cb9504b602239a52bd28d22a7b。
基础Git c1a079f，以上SHA固定本次未提交脚本与实际冻结源，不用目录名代替验证。

弱梯度修复不是重新写白化：trainer d35ddd26通过原compute_advantage出口对
完整collected action-mask调用一次官方masked_whiten，之后原分发/mini/B4/PPO。
raw d/Q/V/A/returns不改。实际两rank729077/730834的原worker RPC核实actor
3a65e173、worker e5eb4afc、原helper079a6d20；VERL_TRIM_SHARED_PADDING=1，
LoRA8/16、每卡B4、ppo_epochs2、entropy=.001、dual-clip3。无新模型调用或override。
已接受actor共同空白裁剪是原VERL padding断言对照，B8/32768原更新和LoRA同步
回执e65662bf；其单次旧logprob14.6427→6.1881s不是整个新迭代加速比。
同一历史TextCraft样本PG梯度白化增大386.229倍、方向cos=.999751，不能外推
AppWorld学习效果已恢复。当前新正式作业刚开始采样，完整新更新仍待真实记录。

新端口检查发现cumem和gpu_worker分别退回已知before SHA c5581b1e、9f9d0dbc。
首个fresh job532620在1791309877完全停止；未从它保存/恢复任何检查点。
用原c9cd147的patch_source和原before备份精确重建已验证7e557a8f、37db3830，
不重装/升级包、不写新allocator。原vLLM hybrid/sleep断言回执a0b4a63f独立保留。
submit-v2恢复原installed SHA检查；新两worker实际导入也核实上述after SHA。
完整停止、恢复、source、job、launch、worker imports与dated snapshot保留于
research/temporary/rl_upstream_alignment_20260929/appworld-efficiency-20261007/
fresh-official-padding-v2/。本机正式launcher原LOOP配置对照1 passed，
19.531s、峰RSS254.37MiB、CUDA=-1/4GiB限制，resume_mode disable。

DT耗时沿原操作拆开：native FA K/V和GDN历史缓存已复用。root计算改变后的suffix，
逆向每层重跑本层原forward取finite操作数，不是重新执行环境或生成新轨迹。
原长case阶段是root3.126s、layer replay3.102/4.363s、finite4.646/3.568s；
rank间并行，all-gather等待不能当可删搬运。原step9实收216条，每卡108条，
2个epoch=54次B4前反向、14次optimizer.step；非224条/56次。
原矩形输入padding53.16%是有证浪费，update>采样并不能单凭比例判正常或异常。

真实长cached convolution原张量/stride已采集，B8两卡、0checkpoint/optimizer，
观察hook返回None且原函数身份已恢复。只对同一实际张量做public API对照：
原native state+投影transpose拼接及kernel25.475/22.611ms；initial_states
1.083/.814ms。finite materialized-cat kernel+VJP1.776/1.740ms，而initial_states
7.328/5.359ms；两列不能漏掉拼接成本，也不能把native单项倍数说成完整DT倍数。
9项原causal-conv1d v1.5.0 BF16 output/dx断言每rank通过，rtol=.01/atol=.05
从原test AST执行；实际BF16 weight保留，区别于原FP32-weight fixture。
没有增加FA/FLA/PPO容差或输出纠偏。此conv候选尚未正式部署。
原tensor pt仅保留远端及SHA，Git只记录原json/operator回执。

官方FSDP set_modules_to_forward_prefetch的同bank current-base长输入对照已结束。
双卡critical总11.330→11.513s，无端到端收益，不部署；原attribute省.184s/1.63%，
诊断记录抵消收益。Q逐值相等，A非零差在root阶段已出现、OFF冷热也有差；
不归罪预取、不新造整DT容差、不称FA/FLA或32k验收。原设置恢复、pending为空。
原回执completed-prefetch-analysis SHA65030fe9，保存于native-reverse-prefetch-current-
base-20261007/v1。缓存卷积候选仅在GPU2/3隔离：PID889091出生1791311666.24，
canonical HF namespace path与linked DT_ROOT，原installed与正式源码SHA不改。
首次诊断889091在CPU请求校验失败：新增source loop覆盖原expected字典。
已仅改局部变量、执行原verify_prepare闭包CPU复现；三owner字节不改。
失败回执保留，20261007新诊断937178/出生1791312107.98继续相同bank比较。
同一immutable bank做OFF冷/热和initial_states热调用，不载检查点、不跑optimizer；
尚未取得完整候选接口/速度结果，不以单算子收益扩大整DT结论。
本次真实长B8卷积OFF/ON已结束：critical 11.3295→9.7917s，约省13.6%；
root与本层重算各省约.67s，finite省.18/.20s。原bank复用、80阶段计数相同，
Q逐值同；A/V原残差保存，不新造整DT容差。单算子新旧输出同，dx两rank
最大9.54e−7/1.91e−6，原causal-conv断言通过。未部署；原exact32k fixture
只用于新增容量支路，不能替代真实长case或称PPO新对拍。
02:42:28正式722849两rank已实际compute_dt_token_advantages，当前组各7/101；
3573unique/retained、3562nonzero前后相同，无skipped。最新B4约3.1–4.8s非全组
平均；采样232完成/8取消、745220生成token/1365.2s含交互。白化还未执行到更新。
02:38:08 cgroup133GB/966GB、OOM/failcnt0是pre-DT时间，不能冒充当前DT内存；
02:42:28实际actor PSS100.16/99.08GB独立保存。
本次索引：experiments/rl/results_appworld_efficiency_20261007.json。



## 2026-10-07 AppWorld检查点29切换：原worker退出后提交完成

原checkpoint29完整（data.pt及两rank的model/optim/extra_state）。原observer在
1791302651因两actor尚未退出而保留remaining_non_zombie、未调用submitter；
失败stdout原件保留。1791303464只读复核原process tree已全部退出、GPU2/3
物理各859MiB，无其它该卡进程。没有修改或抹去旧停止失败回执。

随后直接调用原submit_prepared_appworld_resume.py（64c10036），不新建停止/
加载器、不改prepare/source。实际新driver634197、出生1791303503.79，
恢复原checkpoint29；entry为appworld-retained-credit-sources-20261006-v2，
VERL/DT/LOOP仍原冻结目录。实际source SHA2d49d687，prepared ff54c20d；
entry71项、owner及DT文件全部与prepared逐SHA一致。原LoRA8/16、每卡B4、
预算及公式不变。1791303573是原Ray初始化，尚未新采样/DT/更新，不能称
提速已验证或健康。Ray记录metrics exporter不可达，仅指标导出警告，
尚未看到训练退出；不新建推理/训练/事件导出服务来掩盖它。

本次部署仅e42f source过滤及CPU计数：完整未来reward先算，再跳过训练
序列没有任何action token的source。新的root重放/conv/prefetch优化未部署，
全量GPU/CPU tape仍拒绝；上述473B4/73.7min属于旧source，不能作为新提速结果。
最新dated snapshot：appworld-efficiency-20261006/1791303464/current_runtime.json；
提交、旧退出复核、完整checkpoint及实际source原件保留在同目录。TextCraft不动，
SQL仍停止，0/1与6/7不用。


## 2026-10-06 AppWorld DT大头：实际473个B4与原hot profile定位

1791300732原两rank均7组/473个B4返回，类别组累计4423.464/4422.388秒，
bank459.625/458.714秒（10.39%/10.37%）。两rank并行，不能相加；这不是
完整迭代wall或gen计时。每批median7.91/7.90秒，max43.39/43.27秒。

同一采样原loop_transport末次回调累计2141.4秒、4298次回复、1439131
生成token，包含交互，不是纯decode或完整trainer gen计时；原件保留在
1791300731/original-sampling-boundary.json，不用driver启动时间代替采样。

6348 runner的root只保存层输入，之后32层同suffix再次原forward取finite
操作数，这是有证重算。原checkpoint20/offset84热B4总12.00/12.01秒：
root3.126/3.125、replay3.102/4.363、finite4.646/3.568秒。它已复用
11520/11776的历史prefix，实际后缀1948/1440，不是每次重算全13k。
原FA suffix梯度及FLA有限传播已接；不存在新forward_tangent重算，
num_tests互斥轨迹组也不重复同traj bank。67个all-gather是该DT trace，
但rank1的4.871秒含peer等待，不是可删墙钟；prepare.unshard通常no-op。

原全量GPU/CPU root tapes拒绝保持。部分保留不是现成跳算开关且有显存
成本，不凭短case19%外推长32k。新的具体接缝是GDN cached conv拼接；
已装官方causal_conv1d_fn支持initial_states，需与原capture/finite左窗口
接口同时对齐，尚未改正式代码，也未把API存在当数值或提速验收。

2026-10-07只读卷积回执d1c4e8ec：已装MetaX接口7286f939与原v1.5.0
Python接口字节一致；原BF16输出/dx/dinitial_state断言rtol=.01、atol=.05，
没有执行这些数值测试。既有467MB实际GDN0文件只含FLA操作数，不含
conv输入/权重/梯度seed，不能据此反构造或造输入验收。下一原调用的
被动采集接缝已记录，未live修改或占GPU，不把数据采集项说成方法阻塞。

1791300912真实App已update_actor：物理GPU2/3各54272MiB，actor PSS
124.04/122.85GiB，cgroup515282579456/966367641600B。原源码没有
readout返回后继续持有bank的路径；缓存释放已执行，不能仅凭更新期PSS
判为泄漏。本次CPU计量与源码排查未修改PPO/DT算法、FA/FLA、参数或占卡。
原checkpoint29切换observer仍等待完整原checkpoint，不称已部署。
原始报告、计算源、hash与声明范围：appworld-efficiency-20261006/
1791300731/major-cost-accounting.json；canonical results_efficiency_work_items_20261006.json。
最新dated snapshot：appworld-efficiency-20261006/1791300731/current_runtime.json。


## 2026-10-06 SQL原588 B4同步负载：官方CPU分区计量

复用原4704个请求长度和原VERL get_seqlen_balanced_partitions(96980f6)，
同步最大context宽度总和2491524→2034648（−18.337%），两卡宽度差
924608→2283（−99.753%）；packed slots32467520→32536104（+0.211%）。
这是减少卡间等待的计算几何证据，不是删总DT工作或墙钟提速测量。
attention/context官方分区indices逐值相同；缺全部prefix切点不称suffix FLOPs。
CPU峰值RSS539812KiB，CUDA未初始化，0模型/DT/训练。首轮原序断言失败
单独保留：原actual_row_lengths是运输序，traces是readout稳定排序后，
不能混为一个序列。成功回执SHA1a7c8f44…及原source/log/owner字节来源
见sql-paired-real-b4-20261006/v1/sql-original-588-b4-partition-workload.json。

SQL host-cache-v2仍有原balance/scope/hostcache，但trainer8816与readout228
缺已批准白化/两句clock，不作为最新正式版本直接部署。新隔离v4已完成CPU准备，
SQL仍停止，未启动模型/训练、未占GPU、未修改App/Text或active清单。

v4沿原SQL复制，仅trainer调用已有patch_dt_advantage_preprocessing，整批
collected action-mask白化一次，再进原actor/B4；raw Q/V/A/returns保留。
entry只有3项改变：common95042a7/3f4a5f70用于保留source与CPU工作计数，
SQL query-only6f2a4e32只移植afe59dd两句。其query_ids方法AST等于已接受94a，
其余SQL readout AST不变；未新增App prefix lease，仍只有pair内prefix。
原entry总71个文件，68个不变；VERL其余文件、DT c9、whole-rollout scope、
host-cache、预算/采样/LoRA8/16/B4/PPO及既有kernel断言保持。

实际CPU transport11项（0错误/失败/跳过）+原白化接口6项均通过，共17项。
总准备76.532秒；CPU PID255733/birth1791299960.09，parent RSS峰值
790130688B、child峰值956944384B，Torch2.8.0+metax3.5.3.9，CUDA未初始化，
CUDA/MACA=-1、OMP/MKL=1。这些是原DataProto运输及原helper/生成trainer
接口测试，不是FA/FLA/DT数值、提速或学习验收，未新增数值门槛。

v4完整prepared SHA：
c0335f7bacbfcaa223f36db8c471f6e4f621522a139837d9d5422414c57a36d9；
执行preparer源SHA：
a2fc07e910720841694896613cfa4ef2bb0aeb332dc6e1062b7fe0cb1fa5a17a。
trainer SHA7366557b482e604d66f47bdc4841ea147ffaac7c80538fa000544bb4e92eb619，
SQL readout SHA6f2a4e326d1200994112a798fd6799c6842d7f7807262197ee864da956c5f5df。
原件在research/temporary/rl_upstream_alignment_20260929/sql-efficiency-20261006/
current-credit-v4；变化、完整源SHA、17项CPU日志/资源绑定到
experiments/rl/results_efficiency_work_items_20261006.json的SQL subsection。

先前v3在tests前被query_ids之外AST差异拒绝：whole94a还带此前App prefix
lease接口，不能整体覆盖SQL228。失败source/stdout与partial candidate保留；
执行源1ebccd227ccf3e452030ba394e1ced68aa51ae13926ae87b0e1f1e1a9e4f205a，
0测试/模型/GPU/部署。v4仅修该source移植范围，没有改变算法或默认运行。
上述588B4全负载分区的−18.337%仍只是同步最大宽度总和变化，不是速度；
总packed slots反而+0.211%，没有声称删除总DT工作或SQL已提速。


## 2026-10-06 AppWorld保留action修复：原checkpoint29自动切换已提交

22bcfdb的retained-action-v2复用e42f596两函数，只过滤无训练token的
source；完整未来奖励先计算。新增CPU计数记录nonzero before/after及skip，
删除计数AST后与已接受cdef计算相同。实际CPU11通过/0失败/0跳过，
不是新DT/FA/FLA数值验收。71个entry及VERL/DT/LOOP/source hash保持。
fullprepared ff54c20d绑定当前3592468/birth1791291997.14，原helper5e1cbffd
已提交（本机12844、远端observer105148），等待原完整checkpoint29后
调用原submitter/launcher/VERL loader；此时仍旧入口，不能称已经部署。

原日志1791298949已有5个完整类别组，每rank341个B4、约3146秒，
加当前组100个已返回B4共441次，median7.8秒/max31.1秒。bank仅约344秒；
rank并行不可相加，未完成DT不能冒称完整迭代时间或提速。
1791297973资源：cgroup508118433792/966367641600B，RSS380.247GiB、
cache92.892GiB，App DT actor PSS112.20/119.57GiB；物理GPU2/3
34940/34242MiB，Text生成GPU4/5 50332/50064MiB。当前阶段可容纳，
不把此观测当32k峰值或无内存浪费证明。0/1、6/7不用，SQL仍停。

对应官方FA输出×2、梯度×3；FLA o/ht .005、qkv梯度.008、g/beta .02
原reference/assertion和实际dtype收据保留，不加纠偏或整网容差。
回执：experiments/rl/results_efficiency_work_items_20261006.json。
最新dated snapshot：research/temporary/rl_upstream_alignment_20260929/
appworld-efficiency-20261006/retained-action-transition-v2/current_runtime.json。


## 2026-10-06 效率修复：AppWorld仅归因保留action（prepared-only）

当前正式App入口仍为2d3a93/da9b8a，完整response表做DT后只scatter保留slices。
已复用原e42f596的95042a7/cdefd4b两函数修复到独立candidate
appworld-retained-credit-sources-20261006-v1；完整未来reward先算，之后仅选择
官方训练保留的source identities。71个其他entry文件SHA保持，VERL/DT/LOOP
及采样/PPO/白化/LoRA8/16/B4/32k未改。实际CPU原VERL transport测试11通过，
不是DT数值或提速验收；未部署、未改正式进程，待原完成checkpoint边界接入。
本次原日志首组122个B4：1135.56秒，bank144.84秒，runner989.77秒；
次组30个B4：287.7秒、bank28.4秒。两rank并行，不相加；不是完整迭代时间。
SQL仍停；latest host-cache-v2已含均衡分发和whole-rollout作用域，尚未运行。
旧SQL step10的DT6418.249/总11995.090秒不能冒充该候选性能。
具体原件/SHA/准备状态：experiments/rl/results_efficiency_work_items_20261006.json。

## 2026-10-06 AppWorld原采样完成并进入DT

1791294467原LOOP明确采样完成：rank0为112+8取消，rank1为105+15取消，
共217/240、3891responses、1258007policy tokens。原全局0.9与每场景0.75
规则保留（原round(6*.75)=4），不改取消或舍入行为。
1791295087原old_log_prob/ref已返回，TaskRunnerfit1244及两actor实际栈
进入既有DT/readout/capture。当前plan每rank122个B4，最初3个已返回，
warm后两批2.89/2.95秒、日志d在−0.178至0.334内；这些只是当前组开头，
不能称完整DT速度/精度/全部有限性验收，尚无完整step29或checkpoint29。

TextCraft下一轮原采样30/30已返回33:01，原旧概率计算进行中；最后完整step26。
本次DT capture阶段cgroup484922011648B、limit966367641600B、
host available493831966720B；App两个DT actor PSS104.14/106.95GiB。
物理GPU2/3为30300/30299MiB，4/5为17058/17058MiB，0/1和6/7各861MiB。
按实际phase区分CPU bank/物理GPU和allocator，不把进程存活当训练健康。
训练源、原算法/预算/LoRA8/16/每卡B4/DT B4未改，未新增模型测试、容差或补丁。
最新dated snapshot：research/temporary/rl_upstream_alignment_20260929/whitening-formal-followup-20261006/1791295087/current_runtime.json。

## 2026-10-06 白化后首个TextCraft正式更新已返回

1791293395原PID3269087/birth1791289021.47已完成正式step26并进入下一轮
原生采样（本次snapshot为round4/30）。原step26共3781.905秒：gen2167.631、
old_log_prob176.273、ref139.455、DT/adv701.016、update_actor597.061。
原日志grad_norm0.065、PG−0.001、entropy0.753、reward0.492均为打印精度；
entropy及reward来自本轮更新前采样，不是白化后的效果提升。原save_freq25，
故没有checkpoint26正常；已有native64保存恢复证据单独保留。

原reward_extra_info缺键已核：原EpisodeRewardManager的rm_scores字典快路径
缺可选extra键，原compute_reward退回同一个rm_scores Tensor；没有丢/重复奖励，
不为此加补丁。该source/phase核查不是新容差或全部梯度有限性证明。

AppWorld同一3592468/birth1791291997.14仍原LOOP采样，两rank原累计50/120、
47/120；latest episode ret0.750/0.500不是完整迭代/独立评测成功率。
本次物理GPU2/3为55316/55300MiB，4/5为50148/49990MiB，
cgroup256780931072B、limit966367641600B、host available727530545152B。
原allocator max_reserved64.398GB是另一口径，不能当物理峰值。
LoRA8/16、每卡B4、DT B4、原预算/代码/PPO保持；0/1与6/7不用，SQL仍停。
未新增训练/RPC/模型测试/容差或改动，学习恢复仍待后续正式趋势。
最新dated snapshot：research/temporary/rl_upstream_alignment_20260929/whitening-formal-followup-20261006/1791293395/current_runtime.json。

## 2026-10-06 AppWorld官方白化与原LOOP服务恢复已提交

source c775ddd，原提交器64c10036…实际Popen原launcher，PID3592468/birth1791291997.14，
1791291997.659开始，GPU2/3，从原完整checkpoint28恢复。原预算40组×6、
completion0.9/0.75、global minibatch32/ppo_epochs2、200iterations/40交互步、
train32000/effective transport32768、LoRA8/16/每卡B4/DT B4、sampling1/1500均保留。
原async/prefix/eval routing与已验FA/FLA/PPO/vLLM源保留；0/1与6/7不用。

原App trainer b174→d35ddd26（仅原masked_whiten helper seam）；readout8acf→94a
只有已验query clock两句；LOOP原78d→17adc301仅末等待/poll位置修复，原5checks/
5waits/2restarts/HTTP100秒保留，旧源未覆盖。App实际6项CPU合同测试/实际导入
与配置通过；LOOP 7项原owner测试+真实init/restart/close通过、两服务和端口清理。
不能称这些是App模型训练/学习验收，旧100秒execute为何卡住仍未查明。

实际提交边界env/argv AST与CPU verified launch-plan逐项相同；继承原
VERL_RELEASE_UNUSED_HOST_CACHE=1。原提交器只补已录资源env与新author source
记录，未改算法/恢复/生成/优化器。source2827c5ae…、准备83393ed0…、计划08cbdfe1…
及actual source/SHA/CPU receipts/PID/配置见results_appworld_official_whitening_20261006.json。
1791292393原两rank checkpoint28 model/optim/extra加载后，TaskRunner step28
进入原LOOP采样，rank1首条episode返回ret0.250（3 policy messages）。
argv/options/10项env及12项源SHA符合提交记录；本次未直接采到LOOP接口live frame，
既有CPU原导入与service回执仍单独绑定。尚无新DT/PPO更新或学习恢复证据。
物理GPU2/3为55316/55300MiB，cgroup251088728064B，host available699792977920B。
当前dated snapshot：research/temporary/rl_upstream_alignment_20260929/appworld-official-whitening-20261006/v2/deployment/current_runtime.json。

TextCraft同一3269087/4,5已完成本轮30次采样交互（35:55），1791291491原调用栈
确认进入旧策略compute_log_prob；尚无新完整26。原64一次白化真实Adam/保存
恢复已有独立完成回执；整轮效果恢复仍待正式结果，不把放大PG当作质量修复。

## 2026-10-06 本轮原生采样进度及LOOP候选修复

1791290776：TextCraft PID3269087/birth1791289021.47仍为原job，已返回18轮
交互、当前19/30（原标签active138），尚无新DT/PPO完整更新。GPU4/5物理
50587/50075MiB，cgroup147514339328B，host available904659709952B。
这些是阶段观测，不是学习恢复、32k峰值或整轮速度验收。最新带时间的snapshot：
research/temporary/rl_upstream_alignment_20260929/textcraft-degradation-20261005/official-whitening-formal-20261006/v2/deployment/runtime-1791290776/current_runtime.json。

AppWorld已退出的原1856052完整checkpoint28保留；原LOOP最后sleep期间服务
已启动，原循环未再次poll即raise的真实日志已保存。ac8bbf9仅移动原最后等待
至最后GET之前，保留5checks/5waits和原restart。该项仍候选，远端CPU真实
owner与服务检查待完成；未覆盖旧验证baseline或启动App训练。

## 2026-10-06 官方白化与既有query时点修复：TextCraft已提交

通过冻结原launcher提交正式恢复：代码faf0d89，PID3269087/birth1791289021.47，
1791289021.992开始，GPU4/5。完整原checkpoint25由原VERL resume loader读取；
正式预算仍30epochs/330iterations、每轮32组×8轨迹、原global minibatch64/4次更新、
30交互步、训练10752/eval14848、transport32768、LoRA8/16、每卡B4、lr1e-6。
原entropy0.001/dual-clip3和save25保留。1791289304实际核对：两rank原model/optim/extra加载25，TaskRunner原step25，
原AgentGym rollout→adapter→原vLLM生成，当前round1/30；
不能将单次手动updated-actor或进程存活当正式26轮/效果恢复。

物理4/5约49.0/49.8GiB、cgroup133.3GiB、host available846.6GiB；本次仅启动观测，
无新完整迭代或峰值/32k复测结论。两actor py-spy读取失败属于观测器结果，
原样保留、不当训练异常；成功TaskRunner栈与实际源路径独立记录。

正式entry textcraft-official-whitening-formal-20261006-v2；原launcher3dcecb5d…、
readout94a7afbc…（afe59dd已有query clock修复），VERL trainer7366557b…
（8f52官方masked_whiten）、原actor1f862e8b…、worker e5eb4afc…，DT c9/c7fc969f…。
保留原host-cache释放env、官方AgentGym服务4049454/36005与持久编译缓存。
启动只标准Popen原argv，不复制训练、恢复、生成、优化器或重试行为。

远端active-training/active-source与新formal-training已登记实际PID/source/旧job历史；
其他task配置未改，旧保存行不代表存活。完整源SHA、有效配置、CPU实际进口、
提交Gitcommit及dated current_runtime.json见
research/temporary/rl_upstream_alignment_20260929/textcraft-degradation-20261005/official-whitening-formal-20261006/v2/deployment/。
本机experiments/rl/current_runtime.json仍是明确日期的10-05历史快照，不能当最新。

1791289725更新：原前3交互轮已返回，当前round4/30、active233，首轮172.33秒，
随后原进度显示累计5:16与7:24；这些是采样/交互轮计时，不是整轮训练耗时。
本机HTTPS连接失败；按GitHub官方公开Ed25519指纹校验后，既有SSH443鉴权
成功，91aaf89→44dbe17已推送同一分支。没有改全局SSH/Git配置或绕过host检查。

## 2026-10-06 官方白化后的原生真实更新与正式准备

原生单次更新已完成：PID3068397/birth1791287168.04，提交源码f441671，
GPU4/5原checkpoint25与真实64条，1791287674.311完成（505.750秒）。
两rank各一次原Adam/scheduler，496项step100→101，全部496个可训练张量
发生变化；参数/梯度/更新/Adam状态非有限值均0。原save/load后每rank496参数、
1488优化器张量的原dtype/layout/SHA相同，scheduler与原配置相同；同job
[64,10240]原LP/H全部逐值相同。原每卡B4/全局64/LoRA8/16/lr1e-6保持。

复用原74b66观测器代理真实step，没有no-op，没有重算DT或采样。手动save
默认global_step0，隔离updated-actor不是正式step26；原正式256条每轮4次
optimizer更新，本项不代表整轮或学习效果恢复。Actor-only未调用vLLM生成。
本次短批次结果不替代先前32k容量、FA/FLA或原PPO容差验收。

只读CPU归约使用原Adam分析器；首attempt缺已有MACA环境标志，随后复用
已哈希绑定的完整环境成功（1.802秒），未重装或重跑GPU。两attempt日志保留。
原件、实际源/配置和执行身份见results_textcraft_learning_degradation_20261006.json
的official_whitened_actual_update及whitened-update-20261006/v1。

正式v2仅已准备，尚未启动：冻结原launcher3dcecb5d…/runtime205c…，
VERL trainer7366557b…/官方helper079a6d20…，DT c9 runnerc7fc969f…，原
checkpoint25。原sampling与训练options比较仅输出路径和resume发生变化。
使用已存在且CPU29检查/真实64查询诊断完成的afe59dd query clock修复94a7afbc…，
将预测起点对齐已生成动作随即执行；不称其修复弱PG。71 entry文件仅readout变；
完整94a源还包含既有可选prefix_lease_factory=None接口，原冻结producer0ad37a17…
未传入，默认路径不变。不得把文件差异说成只有两行。v1准备保留而不启动。

当前旧SQL/AppWorld/TextCraft训练PID均不存在；TextCraft原官方服务
4049454/birth1790706407.49、36005仍在。AppWorld原1856052因执行HTTP超时后
原LOOP重启ready失败退出，完整checkpoint28保留；未据此改DT/PPO或启动重试。
0/1与6/7保持不用，不从旧active清单推当前存活，不重装服务或清缓存。

## 2026-10-06 用户批准的官方 DT 优势白化：隔离对照

本轮用户明确要求实施并测试官方白化。唯一PLAN只增加actor优势预处理，raw
d/Q/V/A及Q−V定义保留。`patch_dt_advantage_preprocessing`复用20bd331的
`verl_F.masked_whiten`，整批action mask计算一次，mask外再置零；没有第二次
GAE、额外奖励分配或自行实现的normalizer。原actor/core/worker和损失参数未改。

隔离VERL副本为`candidates/textcraft-official-whitening-20261006-v1/verl`；
1450文件逐一SHA比较，仅trainer从8816ea4e…变为7366557b…，helper仍079a6d20…。
source patch SHA84155a17…。实际Torch源码/接口测试6通过，原collector/trainer
测试8通过；真实64行CPU检查与直接官方helper逐值一致，245386有效action
tokens，19非advantages tensor、7非tensor字段和metadata保留。

GPU4/5隔离诊断PID2687746/birth1791283608.36观察同一checkpoint25原批次的
raw/white PG/H/KL与cross-PG，原每卡B4/全局64和LoRA8/16保持。实际
optimizer/scheduler、DT重算、rollout均禁止执行；开始记录不代表GPU测试完成。
原parent1856052已退出，环境由已配置entry脚本和此前已存运行字段/完成资源
override恢复，明确不声称拿到旧进程完整继承环境，不重装或清缓存。

隔离诊断已于1791284730.6006439完成并退出，墙钟1121.722秒；每rank原生
反向6pass/48个B4，真实optimizer/scheduler/DT重算/rollout均0。两rank的原owner
统计一致：raw PG=0.000132319951383，white PG=0.0511058214626（386.229倍），
cross-PG cos=0.999751065178。weighted-H为0.000638034939186→0.000638036725375，
H/PG从4.82191降至0.0124846；KL为0.0000221442670258→0.0000221467383538。
这些支持本批尺度失衡已纠正、PG方向基本保留，不证明DT估计或学习效果修复。
原clipping实际执行，clipfrac并非全0：raw最大0.000313152，white最大0.000521921，
dual-clip lower frac均0；没有继承旧样本的“全部未clip”结论。

原PG首调用178.067秒，white热调用146.045秒；两组H/KL均约144.74秒，
不能把首/热差称为白化加速。观测中两卡物理各27069MiB、容器usage约116.1GB，
无OOM；这是此实际批次的观测，不是新的32k容量或显存峰值验收。
source patch属于8f52b95658d37fc7a0cb69647de801188f70be02（SHA84155a17…），
实际trainer为7366557b…；dated current_runtime.json绑定路径、SHA、配置及回执。
旧PLAN来源改为9a9b532的历史Git blob，当前唯一PLAN仍为用户批准后的文件。

当前为CPU/interface/native-gradient verified，未部署正式训练、TextCraft未恢复。
候选源、CPU回执、原件和实际反向结果统一保存在
`research/temporary/rl_upstream_alignment_20260929/textcraft-degradation-20261005/official-whitening-20261006/v1/`。
不能仅凭PG幅度增加宣称学习退化、DT估计质量或正式训练已修复。

## 2026-10-06 成功末动作信用支持与原actor载体闭合

仅CPU重新归约冻结的原64/186成功response与21成功末动作，按原16个B4各自action-mask分母、每rank÷8及两rank均值保留。末动作2402 tokens，此前29851；A-L1分别3.58776707和97.21372312，token |A|均值0.001493658与0.003256632。末动作占G1 token 7.447%，A-L1 3.559%，原B4权重后3.889%；这些是系数份额，不是任务梯度贡献比例。

已存完整response A-L1合100.80149018821629与原actor有效mask内质量相同，partial slice=0、mask外非零A=0，原saved DT与actor A逐值一致。本项只闭合已存response信用接到actor的过程，不恢复未保存的提取前全input归因，也不证明逐token反事实准确。未测得每B4梯度增量，不把系数正负抵消说成梯度相消主因，不因此额外开GPU测试。

两份新增只读分析源与回执绑定terminal_native_credit_support_CPU_observation；root独立重算阶段及原B4权重并重新执行stdlib分析，六份来源身份一致。0新模型/DT/环境/反向/更新，PLAN、Q/V/A、上游PPO、正式版本和原status不变；TextCraft未恢复，pending官方优势白化仍未实施/测试。


## 2026-10-06 原成功终止动作的已存奖励读出

仅CPU读取原DT日志/head端点与原64 metadata，192运输slot映射到186唯一response；21条为导致官方done成功终止的最后已执行动作，165条为此前动作。没有伪造逐轮done字段，没有从Σd/守恒反推概率。原编码terminal事实成功概率均值0.921191、EOS参考0.842047，完整response目标LP差均值+0.0920603，21/21为正；交换编码另报，未选择为修复。

这项证据否定将首轮随机未来的AUC描述扩大为全部终局读出失败；只含G1的子集不证明总体预测、token反事实或历史退化原因。原件/独立复核绑定既有结果字段terminal_reward_readout_CPU_observation。0新模型/DT/环境/反向/更新，PLAN、Q/V/A、上游PPO、正式版本和原status不变；TextCraft保持停止。


## 2026-10-06 奖励读出的条件交互与预算来源审计

仅本机CPU复用已有64条配对概率和冻结owner源码，0模型/DT/反向/更新/远端调用。原事实成功概率的103个同prompt成功/失败pair AUC=0.514563，交换编码=0.475728；每个prompt去其fact+EOS共同log-odds编码偏移后，128点残余RMS=0.649023。这些描述排除仅一个常数标签偏移，不识别全部弱PG来源，不选择交换编码为修复；29/64反号属于完整response端点根差，不是token优势错误率。独立原件复算一致。

冻结AgentGym rollout原while以max_rounds/done推进，10752/10240在collect结束后由原truncate_output_ids裁训练载体。未在forecast写这两个数字不能直接称漏了环境终止预算；原transport的32256 prompt超限走官方truncation=error，异常边界及既有数据的动态证据范围在return-forecast-budget-source-audit-20261006.json单独记录，未改停止策略。

source与独立复核绑定既有results_textcraft_learning_degradation_20261006.json的reward_readout_conditioning_CPU_diagnostics。生产版本/status/PLAN、Q/V/A、原PPO、LoRA8/16与每卡B4均保持，TextCraft未恢复，未运行待答归一化候选。


## 2026-10-06 等价标签编码的原任务梯度诊断

隔离诊断PID552348/birth1791263586.6在GPU[4, 5]已完成并退出，墙钟902.172s（原job started到completed）。同一保存64条和完整checkpoint25，只加载一次原模型/checkpoint。先保存原A的PG/H，再临时将原RewardAlphabet.labels从01换成10，原query_ids 228保持，标签/目标编码同步；原return语义/数值、observed index、原Q/V/A公式、trajectory_credit和compute_advantage保持并恢复临时方法。新信用仅通过原DataProto pop/union接回，不重建轨迹、mask或奖励。

rank0原PG norm 0.00013231116091309，换标签PG 7.0287321615131e-05；旧/新PG原互积cos -0.219494522043，weighted-H 0.00063805396851262→0.00063805092585502。实际反向pass 4，原B4反向 32，optimizer no-op 4，scheduler no-op 2；原DT报告finite_trace_calls=24，event_contrasts=96。

rank1原PG norm 0.00013231116091309，换标签PG 7.0287321615131e-05；旧/新PG原互积cos -0.219494522043，weighted-H 0.00063805396851262→0.00063805092585502。实际反向pass 4，原B4反向 32，optimizer no-op 4，scheduler no-op 2；原DT报告finite_trace_calls=24，event_contrasts=96。

这些是当前条件事件编码对实际PG幅度/方向的描述；不把换标签称为修好训练，不从梯度比值推未来成功率，也不替代原作者累计删除RISE/MAS。原完整action mask、每卡B4/8累积、上游PPO和H/KL配置保持；原clip执行，真实optimizer/scheduler更新和rollout均0。v1在CPU准备阶段因TensorDict键迭代接口失败，0模型/DT/backward；失败源和日志保留，v2仅修.keys()检查。source/import/config/carrier SHA、原报告、CPU分析、独立review及完成资源绑定结果字段`equivalent_label_native_task_gradient`。正式版本、原status和PLAN不变，未运行pending归一化。


## 2026-10-06 弱任务梯度：长度、事件读出及原优势预处理

仅CPU复用原checkpoint25/iteration26的真实64条、8个原prompt组各8条及186成功response原件。长度分解未证明固定1/L是完整人口的主因，expm1组合未再次显著缩幅；原1/211仍是preclip任务梯度范数比，不是Adam更新或归因质量比。原同任务内奖励预测与等价标签交换的响应已绑定新原件，不能把读出缺陷与DT整体归因评价混同。

原VERL源码的GAE接口在central batch/action-token mask上调用masked_whiten，GRPO按原组标准化；本次DT仍直接使用raw A，未执行GAE。masked_whiten默认同时中心化和缩放，不能称为只乘一个系数。原loss mask继续排除O/padding。用户尚未批准改变PLAN的禁止归一化条款，因此未实现或测试该候选；不改熵、Q/V/A、LoRA8/16、每卡B4或部署路径。实际同策略载体来源、mask语义、已存数值分解见results_textcraft_learning_degradation_20261006.json的weak_task_signal_cpu_followup。本次模型/GPU/DT/反向/更新/正式启动均0；先前作者累计删除/RISE/MAS结果保持。


## 2026-10-06 原作者累计删除与奖励事件曲线

GPU[4, 5]隔离PID112109/birth1791259467.52已完成并退出，从启动到完成596.598秒。checkpoint25原成功首response 21 UID，原VERL padding后22运输槽，保留副本差与首个UID主记录。原作者k=20累计删除两视图，每rank原生评分调用[462, 462]；使用保存的d_from_A，不重算DT。原模型及完整checkpoint加载一次，评分B1与固定actor训练B4分开，LoRA8/16保持；DT/反向/optimizer/scheduler/rollout均0。

实际runner c7fc969f9f521993f2449ea5f364adcb3e0fdac5b01c38c103963639551516c1，原作者metric 583f4b7d0426407eb9a517f173365762860a1f4382f472dffb5c07de7d3e94a1，完整导入路径和其余源SHA见绑定原件。signed_RISE: RISE=0.013519047666008659, MAS=0.6549049945432908, 原同集合ΔLP与Σd MAE=0.2088063485939311; positive_MAS: RISE=0.013519047666008659, MAS=0.12702971744703956, 原同集合ΔLP与Σd MAE=0.17183671696707087。原RISE/MAS及曲线由作者函数原样返回；两视图原归一化全零UID数量为{'signed_RISE': 13, 'positive_MAS': 13}，负full-EOS端点使初值clip到0并维持running-min；RISE很小不能自动解释为中间删除无效或归因优秀。同集合累计效应与单token条件删除不是同一统计对象，原非有限指标保留为缺失描述，未加纠偏或容差。完整raw、分析、两图、CPU来源合同、独立复核及物理资源回执绑定到既有degradation报告。

另绑定已有14 B4同端点root/seed/32层/final账本和原VERL GAE/GRPO标准化与DT raw优势的来源审计。这些只解释现有信号的口径，不修改PLAN、Q/V/A、原PPO或参数，不称作者容差通过、世界反事实准确或弱任务梯度已修复；正式TextCraft仍保持停止。


## 2026-10-06 查询语义修复的原任务梯度实测

GPU4/5隔离PID3996191/birth1791256647.02已完成并退出。同一原64条/checkpoint25/LoRA8/16/每卡B4，原模型及完整checkpoint各加载一次。先用保存的原信用测PG/H，再调用原trajectory_credit→worker→producer→runner重算，临时绑定afe59dd的query_ids(94a7afbc)，结束后恢复原方法；原pop/union/compute_advantage接回新信用。原reward、IDs、mask、old/ref LP及原Q/V/A组合保持，CPU逐字段回执另绑定。每rank两个信用组各两个local32反向（8B4/8），原clip执行；optimizer no-op4/scheduler no-op2，真实更新/rollout均0。

原PG norm 0.00013232293184523，新PG 9.6505998641621e-05，新/旧 0.7293217985；weighted-H 0.00063804894845689→0.00063803891811982。这些是同job原preclip Gram的描述，不新增容差；没有新旧PG互积，不声称方向或成功率改善。查询修复对齐环境立即处理已发出的回复，但本次未增强任务梯度，不能当学习问题已修。正式TextCraft保持停止，查询候选仍prepared-only；不做信用缩放、归一化、熵调参或公式改动。

原DT每rank24B4/96contrasts，实际查询225/226 token、最大读出5226/5297，保留原密集ABI、缓存与dtype；内部conservation标志不是FA/FLA官方容差断言。原作者累计删除/RISE/MAS仍是整体归因评价，不用单token或小梯度代替。失败v1在CPU检查TensorDict.keys接口前终止，无模型成本；v2只修诊断键迭代。所有source/import/config/input SHA、原件、阶段日志、完成资源、CPU对照和独立review绑定现有degradation结果。


## 2026-10-06 原任务梯度的成功/失败支持对照

GPU4/5隔离PID3747980/birth1791254337.14已完成并退出，原64条/checkpoint25/LoRA8/16/每卡B4不变。复用原VERL observer，每rank两个支持组各两个完整local32反向（8B4/8），原clip执行；optimizer no-op4/scheduler no-op2，真实更新、rollout、DT重算均0。只把原GRPO系数在指定Q支持之外置零，完整DT A和原loss_mask/分母保留；未改loss。原助手PYTHONPATH导致v1在observer前失败，v2仅修诊断搜索路径并在fresh subprocess核实际导入，保留两版本原件。

成功支持同组原SUM Gram：DT norm .00013230308934、GRPO成功norm .03067111740664，cos +.01494206473（范数约1/231.82）；失败支持DT norm .00013225878060、GRPO失败norm .02704332881123，cos -.00593101386。两rank原Gram相同；重复DT norm变化-.03349%仅描述，不新增容差。新run tiny clipfrac非0，不沿用旧zero-clip结论；没有成功/失败互积和raw gradients，不拼精确总梯度、不混旧run。失败惩罚缺失不能单独解释成功组的方向差；也不把GRPO当oracle。

实际耗时从job start约722.60秒；结束后4/5各860MiB、无诊断GPU进程，主机可用452327534592字节。source、import、配置、输入、失败原件、完成marker、四rank raw和两份独立review绑定现有degradation结果。这是弱任务信号调查，不是生产修复；afe59dd查询语义修复仍prepared-only，TextCraft保持停止，PLAN/c9cd147/原VERL未变。整体归因仍按原论文累计删除/RISE/MAS评价。


## 2026-10-06 弱梯度分项与已交付响应查询修复

本机接口源码提交afe59dd仅修reward_readout的两句查询：当前完整回复已经交付，环境立即执行，不再要求补完同一回复。源码SHA94a7afbc、test ae16a359；64原native查询与先前clock候选编码逐项一致，原prefix/target/source/EOS范围不变，查询增加14token。29项原adapter单测通过，CPU约28.9秒、RSS约.969GiB、无模型/DT/GPU/反向/更新。独立review89134a3f。仅prepared，不部署、不称弱PG修复；既有paired预测测量未改善，不因修正错误条件而扩大质量结论。

另复用原Adam三支LP/H和完整carrier，仅CPU按原16B4分母拆正/负/零A：DT相对当前PG置零对照，正A加权LP额外变化-4.34111e-05，负A+8.87006e-06，冻结DT PG额外+6.95970e-08。这是完整分母贡献，不是组内条件均值，不是参数/梯度分账。原task proxy与实际位移的一阶投影仍下降，有限eval变化、训练/eval路径和原生精度未由此隔离，不能叫Adam反号。CPU PID3612726/birth1791253077.8，3.654秒、RSS893MB，原core PG与旧结果逐项一致。raw6f1dae20、独立reviewf3295891。

当前1/211已用终局漏奖修复v2，21成功终局均未越出载体；旧mask不能解释本批。小d、成功条件幅度和零回报支持分别记录；不乘211、不改熵、不把GRPO当oracle。PLAN、c9cd147 DT、原VERL、LoRA8/16及每卡B4不变；TextCraft保持停止。整体归因评价仍保留原论文累计删除/RISE/MAS。


## 2026-10-06 真实更新方向：弱任务信号不等于Adam反向

复用六份原Adam更新的FP32本地LoRA参数/梯度shards，仅CPU描述DT总梯度与当前PG置零对照总梯度的差。原clip均未触发，496参数初值/布局/支持相同；独立反向的舍入及RNG边界保留，差分不冒充精确单项梯度。实际DT参数位移与该task proxy内积为-3.65394843e-08/-3.79671514e-08，两rank一阶方向均下降。先前完整minibatch Gram的gDT·gTotal=+1.94059959e-08，亦未被H/KL反向。因此仅凭熵范数4.4倍或有限冻结PG升+1.708e-07，不能声称Adam把当前任务方向推反；训练退化与弱信用仍需奖励相关性证据，不把这一更新代理等同真实成功率。

CPU PID3481127/birth1791251847.82，分析3.197秒、RSS约.548GiB、PSS约.261GiB，CUDA/distributed未初始化；0模型/预测/DT/反向/更新。输出bd1d1651、分析source9bcb5861和独立review1d48c06a绑定既有结果。另对64原native B4首轮读出按完全相同prompt/query分组：跨原response长度EOS参考成功概率最大差3.9463pp，相同prompt+长度513的最大差2.72e-06；描述长度依赖，不将其当成211倍原因或新数值门槛。

版本角色：仅保存的原始张量分析，无生产修复或新训练。PLAN、c9cd147 DT、原VERL、LoRA8/16和每卡B4不变；TextCraft保持停止。单token仍仅局部诊断，整体归因评价保留原论文累计删除/RISE/MAS。


## 2026-10-06 当前响应时钟候选与真实更新的冻结目标变化

隔离query候选：原环境直接执行已交付response，冻结228afbc7查询却要求补完同一response。仅在诊断tokenizer入口替换两句，原prefix/source IDs、EOS参照、labels、G、sampling、步数和Q/V/A不变；新增query IDs仅对应上述两句变动。v2 PID3208088/birth1791249307.77完成，GPU4/5，149.368秒；64真实首轮病例，每rank64次native B2事实/EOS配对，无重复行/补padding；actor/DT配置仍每卡B4。原/candidate pooled Brier .38691061/.39847212，成功根绝对均值 .15918764/.12382817。成功/失败评分和相关变化混合，不能称其修复了小token信用或1/211 PG。原B2与旧B4同编码LP差保存为描述，未扩大FA/FLA容差或加纠偏。v1仅CPU prepared；review发现分桶会造成rank42/44次FSDP调用及class来源观察错误，未提交GPU。v2固定同步调用次数并改为method来源，保留v1原始源码和CPU回执。

复用已保存的三分支真实Adam before/after LP/H，仅CPU调用原VERL core损失和归约：DT branch ΔPG=+1.7083357307e-07，Δ(-.001H)=-5.7544275478e-07，Δ(.001KL)=+1.3775558472e-08，Δtotal=-3.9083533920e-07。本次总目标下降但冻结DT任务目标上升；同批目标不是成功率，旧Adam历史不是可线性分账的当前熵向量。保留原old/ref、mask、rank32→B4×8→/8；无模型/预测/DT/反向/更新，约0.535秒CPU统计，RSS约.803GiB。

版本角色：上述是已完成的隔离诊断；不是被接受/部署的生产query修复。原c9cd147 DT、原VERL actor/core、PLAN、LoRA8/16、每卡B4和正式任务配置均未改变。TextCraft未恢复，未从single probe判整个DT差，未放大/归一/裁剪信用或修改entropy系数。


## 2026-10-06 弱信用来源：类别读出配对与历史支持集

隔离v2 PID3037369/birth1791247724.44完成，GPU4/5，231.851秒。原checkpoint25、64条真实首轮输入、原VERL worker/config、LoRA8/16与每卡B4均保留。只由原RewardAlphabet.query_ids交换两个类别label ID，values/meanings顺序、回报及轨迹ID不变；按实际query等长配对。64 native B4前向，DT/反向/optimizer/scheduler均0。

原/交换标签事实读出Brier=0.38641347/0.35645212。根端点变化不是token优势，类别敏感性不自动证明整个弱梯度原因或历史因果。复用原native精度、head与卸载；旧同编码LP差只作描述，不新增官方容差。原论文累计删除/RISE/MAS仍是整体归因评价接口；single probes不替代。

v1因新Ray worker未继承旧reader helper搜索路径，在模型初始化前停止并保存日志。v2只把既有helper目录加入诊断PYTHONPATH，未复制owner或修改生产。新审计绑定二类logprob饱和与正式43/93奖励支持集；历史分项梯度未保存，不将1/211扩展为历史参数更新比例，不从缺失报告造零。生产信用/熵系数/采样/任务配置不变，TextCraft未恢复。


## 2026-10-06 弱任务信号：原 Adam 单步对照与幅度来源

隔离PID2778451/birth1791245326.02已完成，GPU4/5，830.290秒。原64条完整carrier、checkpoint25、LoRA8/16、每卡B4、原VERL actor/core/import SHA与损失默认保持不变；三支分别完整恢复model/optimizer/extra，复用原AdamW动量/RNG/scheduler。每支每rank实际1次optimizer与1次scheduler，496个state均100→101。无rollout、DT重算或正式重启。诊断只保留原返回值，原step/clip/offload未替换。

按原B4分母，DT优势绝对均值为GRPO约1/1446.69，原任务梯度约1/211.33；幅度小已在DT的d里，未发现额外概率、长度除数或expm1衰减。GRPO含官方组内标准化和失败轨迹负系数，两者不要求梯度范数相等，不能将211变成信用倍率。零回报DT项为零是固定估计性质，不标为丢mask。

实际原Adam DT与当前PG置零对照的本地分片更新cos=0.999581/0.999579，差向量约DT更新范数3%；对照仍含共同历史动量与weight decay，不线性分账为纯熵。原B4等权H变化DT+0.000575458、GRPO+0.000070508、对照+0.000238523。fresh before与saved trainer LP存在差异，保留原old/ref未覆写，不声称ratio恒1或新增数值容差通过。三支初始参数与before读数相同，配对变化单独记录。

CPU分析只复用原DataProto/chunk与agg_loss，CUDA/distributed均false；实际本地dtype为FP32 LoRA分片，FP64副本只作描述、无gather。模型与CPU分析source/配置/资源/输入SHA、raw与独立复核全部绑定现有结果。单token对照仅补充，不取代原论文累计删除曲线及RISE/MAS。优先继续查弱信号来源；没有生产修复、信用放大、熵系数改动或TextCraft重启，正式版本保持不动。


## 2026-10-06 读出目标与官方 carrier：真实 ID 只读核查

64条真实首轮prompt/response/query由原tokenizer仅CPU解码；具体Goal、标签0/1语义、30步/512输出/temp1预算均保留，186个正式输入保留对应首轮Goal原ID前缀。无模型、前向、额外采样、backward、更新或目标改写。decode PID2547941，CUDA/distributed均false，maxRSS约1.04 GB；回执记录精确资源与SHA。

已记录连续两个im_end、部分先前think未闭和查询自身的原控制序列；未据此定义新的格式门槛或宣称退化原因。实际AgentGym schema与冻结原件字节相同，原training carrier及Qwen generation模板的system/reasoning/think行为差异由原源码负责。采样prompt IDs未保存，不能推定动态逐行差值。没有改官方schema、历史窗口或查询。源清单与审计绑定到既有degradation结果。


## 2026-10-06 相同完整端点的 FLA 观察：区分算术与单删除估计

隔离PID2564897/birth1791243341.36已完成，GPU4/5，203.829秒。原7 B4/rank、checkpoint25、LoRA8/16与冻结runner/producer保持不变；14 full finite、28 single root，actor backward/optimizer/scheduler均0。读取既有原生FP16 FLA输出与其原BF16 norm边界，不增加模型/FLA调用；原finite返回对象和公式未改。

每层26运输样本按traj_uid/source_step合并为21 UID，四个head组按原输入逐项合并。L6/L8同端点F−Y_public平均绝对差0.0000606994/0.0000905424，均0/21反号；公共FP16输出到BF16 norm边界同do差0.000278545/0.000310393。208个选定incoming h配对差均0；28 public/28 norm/112原finite head返回齐全，组结束保留输出均0，观察器峰值CPU输出63,602,688字节。

原full系数投影single的42 probes仍是另一测量人口，不能混入21 UID分母。目前未支持将较大条件估计偏差归因于同端点算术错误；本观察未执行新的官方容差门槛，也不代表整网或单token信用准确。已绑定原导入SHA、配置、PID/birth、raw回执及独立审计到results_textcraft_learning_degradation_20261006.json。生产未修复，TextCraft保持停止，无信用倍率、纠偏或重启。


## 2026-10-06 原 B4 任务信号：真实分母与既有熵项

仅CPU复用原native-optimizer-minibatch.pkl与原DataProto.load/chunk、原core agg_loss/compute_policy_loss；global64→rank32→连续B4→/8，不重建采样或logits。PID2450806/birth1791242284.97，14.892秒，maxRSS916328448字节；CUDA/distributed未初始化，模型/forward/DT/backward/更新均0。

原B4均值口径零Q分母82.07085%，不同于global pooled的86.85622%；16个B4中4个任务PG为0，原熵项仍在。保留原分母的DT |A|均值0.000523759、GRPO 0.757718；成功内部信用也小，不只分母稀释。优势mass不等于梯度norm或方向；原32个PG标量差只描述，未新设/扩展任何官方容差。

实际输入/官方import SHA、运行回执和独立逐B4复核绑定到results_textcraft_learning_degradation_20261006.json。没有缩放优势、改熵系数或恢复TextCraft；历史完整Adam因果尚未重放。


## 2026-10-06 原 FLA 双顺序：现成系数与相同单删操作数

隔离PID2324520/birth1791241126.75已完成，GPU4/5，204.834秒。原7 B4/rank、checkpoint25、LoRA8/16及冻结runner/producer均不变；14 full finite、28 single root，actor backward/optimizer/scheduler均0。原avg仍执行一次，其两次原FLA调用和返回对象保留。

只在原avg的PY_RETURN读取forward/reverse现成系数。单删阶段复用原观察器已复制的paired CPU操作数，用原_token_effect做两组标量收缩，不重复native复制、监测或模型运算。原52 transport按身份明确合并为42 probe/层；两order对同一Y的差、符号与抵消只是描述，不替换平均规则或放宽官方容差。

两层平均后的F−Y平均绝对差均小于各单order，未支持取消平均。这里F来自full-response EOS/fact，Y来自single-token EOS/fact，区间不同；偏差量化条件估计质量，不是同端点应归零的恒等式，也不能据此指认FLA kernel超差。

原raw、dtype/head/cut/payload、独立复核及64条读出分组统计已关联到results_textcraft_learning_degradation_20261006.json。生产仍未修复，TextCraft保持停止。


## 2026-10-06 FLA 剩余链路：原生系数、操作数与 dtype 边界

隔离PID2154124/birth1791239530.75已完成，GPU4/5，207.520秒。沿用原7个B4/rank、checkpoint25、LoRA8/16、原DT/FLA接口；14 full finite、28 single root，actor backward/optimizer/scheduler均0。原始104 transport-layer保留，按同一UID/source位置明确合并为42 probe×2层；不改变分母。

只观察原外层对称FLA回调最终系数及真实do；原LocalCaptureEvents读取stage q/k/v/beta、public raw-g和FP16输出，避免把stage累计g当raw-g。实际capture_start+cut/head8与原GDN一致。输入链I-Z-F、FLA投影F-Y、dtype边界Y-O闭合到同次原remaining；闭合不是官方数值验收，也不是信用已修复。

原raw、实际dtype/payload、CPU绑定与独立复核见results_textcraft_learning_degradation_20261006.json的conditional_fla。生产代码、Q/V/A、PPO和参数不变，TextCraft继续停止。


## 2026-10-06 GDN 残差定位：原生三段观测

隔离PID1926348/birth1791237432.3已完成，GPU4/5，338.435秒。仍为原7个B4/rank、checkpoint25、LoRA8/16、原runner/producer/目标/切点；14 full finite、28 single root，backward/optimizer/scheduler均0。层6/8共104 transport-layer，按原身份保留重复后84 unique probe-layer，即42 probe×2层。

只读取原norm-gate有限系数及paired8原生输入/输出。compact capture_start来自原gdn_finite_pullback实际frame；只内侧o/z/gated按同一suffix裁切，外侧mixer保持全长。不启用public observer/diagnostics，不保存额外递推状态，不增加模型调用或修正倍率。两层三项中remaining_input_FLA的平均绝对残差最大，但它仍包含投影、卷积、FLA与存储，不能冒称FLA kernel超官方容差。

来源、实际dtype、CPU绑定、完整原始ranks、重复probe统计及独立复核见results_textcraft_learning_degradation_20261006.json的conditional_gdn；这是定位进展，不是已修复学习信号，TextCraft不恢复。


## 2026-10-06 下层真实子块：有限传播与原生存储分项

隔离诊断PID1721442/birth1791235562.77已完成，GPU4/5，243.217秒。复用原7个B4/rank、checkpoint25、LoRA8/16、原runner/producer/目标/切点；14 full finite、28 single root，backward/optimizer/scheduler均0。层6/8(GDN)、11(FA)共156 transport投影，按原映射保留后为126 unique probe-layer，即42 probe×3层。

只观察原decoder有限系数、原NativeDecoderCapture与mixer模块输出；未启用public observer/diagnostics，不改prefix/replay路径。六项闭合仅验证观测算术；norm用原HF/finite CPU调用分开联合条件、dtype存储与GPU/CPU差。MLP/mixer尚未拆开，不能声明核超差。

CPU准备的源名映射、漏传既有helper和序列化来源字段问题均在GPU前修复，失败stdout保留。本次原始ranks、source SHA、独立复核及数值汇总见results_textcraft_learning_degradation_20261006.json的conditional_primitives；不是已修复学习信号，TextCraft不恢复。


## 2026-10-06 真实条件边界：低层传播削弱并错向token信号

隔离诊断v2 PID1472838/birth1791233251.16已完成退出，GPU4/5、185.171秒；原7个
B4/rank、原checkpoint25、LoRA8/16、原c7fc runner/0ad producer及原目标/布局/切点。
实际14次完整joint finite、28次single原生root，backward/optimizer/scheduler均0。
每组仅短暂留所选slot的实际FP32系数CPU副本（每rank最高约1.05GB），用原_token_effect
收缩原已有BF16 CPU roots并只保存标量；不是CPU模拟有限传播或官方GPU容差验收。

52 transport保留后聚合42既定probe/21成功首次response：C0与实际导出DT信用最大差
4.34e-19；C32与原生single d相关0.98627。C15仍42/42同号、平均|C|0.015733；
C0平均|C|0.001818、22/42反号、相关0.04956。重复槽位完整保留，其中一身份跨零。
误差沿低层传播扩大；不能由最大增量就指认单层bug，不能据此宣布FA/FLA核超差。
下一步针对已定位操作数区分原生dtype存储效应和联合有限分解的条件近似。

v1隔离诊断仅执行42原生root后被计数断言截住（finite0），失败资料及提交源冻结。
根因是public module dict与实际执行recipe globals不同；v2沿官方Ray method/VERL
func闭包解析真实字典，CPU roundtrip/实际caller双身份已核对。中间两次CPU失败日志
保留，无GPU提交。修的是诊断接线，PPO/core/QVA/PLAN/训练参数/正式代码均未改。
原始ranks、独立逐槽复核及来源汇总到results_textcraft_learning_degradation_20261006.json。
TextCraft仍停止，未声称已修复质量，不新增第四组或恢复训练。


## 2026-10-06 输出层拆分：反号来源在head之前

仅重用已保存的matched v1原FP32类别logits，正式c9 seed_with_checks及依赖三模块SHA
与原Git对象/实际import一致。CPU PID1147956，2.722秒、maxRSS655851520字节，CUDA/
distributed未初始化，模型/decoder有限传播/backward/optimizer/scheduler均0。
首个CPU导入因未继承MetaX环境变量失败；已复用原记录PID1856052/birth1791197944.7的
完整环境后完成，失败日志另存，没有重装、清缓存或更改训练环境。

52 transport观察先逐项计算再显式聚合42唯一probe（21成功首次response）：joint
categorical seed乘实际single Δlogits与原生单删42/42同号，相关0.985639；最终DT仍21/42
反号。head条件失配MAE0.004105/RMS0.007005，余下传播/分解/投影/存储残差
MAE0.015630/RMS0.023300；交叉项保留，不能按MSE做因果百分比分账。CPU原log-prob
与已存原生差最大1.043e-7，仅描述，未设官方之外的容差或纠偏。

新增证据排除仅靠换输出seed修反号的方向；未部署equal_endpoint或缩放信用，仍须在
原pre-head边界定位残差。独立review逐值复核原raw/map及52 dot/42聚合，来源在
results_textcraft_learning_degradation_20261006.json。PPO/core/optimizer、Q/V/A、
PLAN、LoRA8/16和B4不变，TextCraft正式保持停止，未称已修好质量。


## 2026-10-06 同原生布局核对：token估计差异仍显著

隔离诊断PID930901/birth1791228196.09已完成退出；GPU4/5、checkpoint25、原7个B4/rank
及原目标/补齐/缓存切点，42次paired root、42次原prefix前向，finite/backward/optimizer均0。
原c7fc runner、0ad producer与原VERL加载路径不变；每卡B4、LoRA8/16、32k上限不变。
仅包装原selected_target_log_probs，原分数算一次后停止；single variant通过现有
synchronize_prefix_start委托原MIN同步保持实测旧cut，没有新缓存或新打分实现。

21个首次response的完整端点factual MAE7.59e-7、joint root MAE1.16e-6；同布局单删差异仍在。
21个成功首次response、42个既定随机probe：DT平均|d|0.001816，原生单删0.018964
（10.44倍），相关0.050997；21/42反号，占原生单删绝对效应48.46%。全部52 transport
probe槽及重复范围另存，唯一42统计明确使用replica均值。未设整网容差或纠偏。
该结果定位实际联合EOS归因向量对单token删除量的估计质量，不等于世界反事实oracle，
也不否定固定精确token价值前提。原Q/V/A组合、观察mask、PPO/optimizer仍保留。

结合原完整minibatch DT任务梯度弱211倍、熵项占其4.4倍，当前最强机制证据是任务信号
弱/方向对应差，令原熵项主导小幅持续漂移；原clipping不能校正优势估计或硬约束独立H/KL。
仍未从该局部原始梯度证明全部历史Adam因果。没有降低熵、缩放信用、改容差或重启TextCraft。
原事件读出区分弱是另一个已测质量问题。原owner README27明确单token删除是不同干预；
原runner39–45的逐token收缩使用共同联合端点传播系数，answer86–98守恒该联合log-prob差，
不构造每个token的事实背景单删量。已确认实际估计语义/质量缺口，未给FA/FLA数值算子定罪。
independent-review原raw+pack独立复核84输入SHA/52位置换算/42predictor映射，未见错位；
官方source捕获与原c7fc runner/9819 head SHA对应。没有为修复另造信用分配或替代计划。

本次总墙钟317.66秒，原初始化218.29秒、root观察约54秒/rank；初始化慢点未采到栈，
不归因于编译或扩展效率工作。来源与范围见
[退化学习信号回执](results_textcraft_learning_degradation_20261006.json)及其原始source/PID/SHA。



## 2026-10-06 TextCraft事件读出与随机token删除：实测质量问题

固定checkpoint25与同一原生64轨迹完成只读观察：GPU4/5，v2 PID675223、
birth1791225807.69，约119.7秒完成并退出。原VERL actor加载、原DT
read_outcomes共66次前向；有限DT、backward、optimizer/scheduler均0次。
LoRA8/16、每卡B4、32k上限未变，实际首次response读出764–1371token。
基座BF16、原GDN FP16执行开关、类别log-prob FP32；不新增数值容差。

64条（8个作者prompt组）实际成功21条=32.81%，原事件读出成功概率均值
71.79%；成功/失败组均值74.24%/70.59%，与结果相关0.0986、Brier0.3864。
支持本批实际读出偏高、区分弱；不以此否定精确token价值的既定假设。
每条首次response随机两位置；21个G1共42个依赖token样本，保存DT平均|d|
0.001816，原模型单EOS删除0.019175（10.56倍），相关0.1222，18/42反号，
这些位置占原模型单删绝对效应质量38.90%。没有把G0缺失d补成0。
新旧native前向布局亦有差异：factual log-prob MAE0.00853、joint root
MAE0.00843；故不能把单删差异全部归因于DT分解或称为世界信用错误。
两个同输入重复差异4.29e-6/7.87e-6仅作描述，不成为整网官方容差。

原192请求精确映射186独立response；无token/reward/scatter错位证据。
完整32253个G1 action tokens的length与mean|A|相关仅-0.176，不支持简单
按1/L摊薄作为主要解释。结合上节完整minibatch任务梯度弱、原熵项占优，
当前证据聚焦实际事件概率估计和逐token删除估计；不再凭少数极值归罪，
不任意修改熵、信用尺度、归一化或Q/V/A来掩盖。PPO核心和优化器未改。

v1观察器在Ray重建类的源码身份记录处失败，尚无读出；v2改用原Python方法
身份并先在CPU核验序列化。CPU映射多加sampling字段/错误切mask也只是诊断
重建错误，失败回执保留，未冒充原训练bug。实际源/PID/哈希/范围见
[原读出回执](results_textcraft_native_readout_20261006.json)与
[完整response描述](results_textcraft_readout_signal_scale_20261006.json)。
TextCraft正式保持停止、AppWorld原作业未改；没有声称质量已经修复。

## 2026-10-06 TextCraft完整minibatch：原熵项相对任务梯度主导

checkpoint25实际采集8个原prompt×作者n8=64轨迹，原global64→local32→8个B4，
四次原update_policy backward观察，optimizer和scheduler均0次更新。
v4诊断PID4189526、birth1791219550.81，于1791221830.87完成并退出；4/5只用于观察，
旧TextCraft正式仍停止。实际actor/core/c9runner SHA与旧验证基线相同；
采集入口使用已接受的终局截断漏奖修复v2，不能称历史坏更新原样重放。

两rank完整Gram逐值相同：DT-PG norm0.000145296、加权熵0.000639293、
加权KL0.0000234072；同批原生GRPO-PG0.0307058。加权熵为DT-PG的4.40倍，
为GRPO-PG的2.08%；H+KL为DT-PG的4.28倍。DT-PG与GRPO-PG cosine−0.03467，
近正交，不能称为已证明普遍反号。四分项Gram的代数组合显示DT总loss负梯度
局部提高熵，GRPO总loss负梯度降低熵；不是Adam实际参数更新方向或新数值通过门槛。
独立复算原Gram确认DT总loss与熵项cosine+0.97428，GRPO为−0.03298；
来源见native-minibatch-v4/native-minibatch-entropy-direction-interpretation.json。
43/64零回报轨迹占86.86% action tokens，DT-A全零；原H/KL仍参与。
本批两个think标签ID净credit为正，旧minimum格式负值不能推广。

427个冻结BF16基座tensor/rank在checkpoint25/100逐值相同；496个LoRA tensor
均变化，原Adam计数100→398且超参数相同。只排除这两个保存点的基座误改，
不当作学习方向正确证明。原PPO、LoRA8/16、microbatch4、DT公式均未改。

来源与范围见[完整minibatch回执](results_textcraft_native_minibatch_20261006.json)，
其中记录了v1 worker绑定失败、v2错误诊断范围主动停止、v3仅CPU绑定检查和
v4实际成功观察，均不得混作旧正式训练退化原因。观察口只在隔离research入口，
未接入默认launch。后续聚焦DT任务信号衰减来自读出还是逐token估计；
不以任意缩放、信用裁剪、熵调参或效率工作掩盖，不称训练已修复。

## 2026-10-05 持续退化：原更新端与信用信号分开核查

旧TextCraft仍停止。本次只读核对原actor update_policy/optimizer AST与VERL20bd331一致，
核心PPO文件SHA一致；没有发现新增信用反号变换、whitening或第二次优势累加。
作者TextCraft GRPO原n8组内mean/std与当前DT直接插入的优势不同；G0任务优势为0，
原熵/KL仍作用于全部action。成功轨迹39/93/117为131/8/约1条（每批256），
熵.777/5.498/9.830；尚未把这条趋势当作熵梯度主导的证明。
checkpoint25已测成功B4中合成正则梯度约为PG的4.09%，不能代替完整minibatch。
少数反号按16slots/11token身份分别计量，不能扩成总体或主因。
[信号对照](official_grpo_vs_dt_signal_audit.json)、
[原更新端](results_textcraft_actor_credit_path_audit_20261005.json)及
[退化回执](results_textcraft_degradation_20261005.json)固定源码/SHA与结论范围。
原历史整批uid/IDs/mask/advantages未保留，未从文本重建。仅原输出观察口在已有远端
Torch/原DataProto完成CPU5项测试，CUDA为空，无模型/DT/optimizer；观察口未接入生产。
本次没有更新训练源、超参数或DT方法，也没有确认退化主因/恢复。

## 2026-10-05 TextCraft退化优先调查


用户将TextCraft退化原因置于效率之前。旧PID3218909已停止；截断漏奖修复仍prepared，
没有把接口修复当作收益恢复，也未选择旧checkpoint100继续或从原模型重训。
实际原日志SHA54f4aef6…绑定92轮：优先查35–40更新窗口，39/40实际成功131/256→117/256。
83/85/109原grad_norm为nan，两rank共6 WARN；原actor跳过对应optimizer.step，
发生晚于首轮退化，不能作为已定位的首因。实际PG/entropy/KL分项梯度仍未保存。
原min-B4记录中成功样本的负信用集中于empty-think格式前缀是待核实线索，
不是整批统计。详细来源、范围及下一项原owner读出观察见
[退化回执](results_textcraft_degradation_20261005.json)。

未接受的GDN边界行存储候选及syntax-only测试完整保存在
research/temporary/rl_upstream_alignment_20260929/paused-boundary-row-storage-20261005，
默认两生产源已恢复HEAD基线，未部署/启动该效率候选，避免调查时混入候选版本。
AppWorld正式2/3仍用原冻结入口；4/5可做必要有界诊断，0/1及6/7不用。


本文件记录运行与修复事实，不定义信用方法；信用规范仍是 [PLAN.md](PLAN.md)。
最近三组只读源码快照与原阶段观察分别记录，见下方定向回执。
[current_runtime.json](current_runtime.json)保留精确采集时间，
不能把文档更新时间当作三组阶段都已重新采集。

固定编号对应：DT发布`c9cd147`、DT数值参考`fc2e6c2`、VERL官方提交`20bd331`、
输出头/B4修复`dc4e4d7`。完整40位提交号和文件SHA保存在快照；记录文档的Git提交
不是新的算法版本，也不表示远端执行了该提交的全部文件。

截至2026-10-05 19:28，AppWorld从原完整checkpoint20恢复，当前PID1856052、
GPU2/3；SQL PID552842已因整机global OOM退出，最后完成step11且无完整检查点；
TextCraft旧PID3218909因真实奖励接线缺陷及晚期训练退化已停止，原完整checkpoint100
和日志保留，修复v2准备完成但未提交。当前只有AppWorld正式运行，4/5可用于必要测试。
较早PID、完成步数和覆盖回执属于下方历史记录，不能代替当前作业。

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
SSH端口30821。当前清单仅三组DTPO，SQL仍退出，AppWorld已从20恢复，TextCraft旧作业已停止；
物理GPU0/1空置，仅2–5用于当前实验，GPU6/7不提交GRPO。具体状态以带时间回执为准。

## 当前代码组合

### 2026-10-05 20:25：正式阶段账目与一次真实B8诊断并行

上述诊断20:27已完成并退出，原checkpoint20/实际进口SHA与source对应，
不是新optimizer更新。[原ledger和bank回执](results_native_dt_bank_ledger_20261005.json)
记录rank0/1完整88请求bank为16.8815/17.3219GB，其中从不消费的GDN状态及window
分别6.4881/6.9552GB（6.0425/6.4775GiB）。这是已证明的多存边界行，
不是将全部匿名内存解释为泄漏，也不据此提前承诺整轮加速。
两rank32层记录的root/replay L2与effect差均0；root到seed差−0.00367/−0.00715，
总残差0.19113/0.10159精确分解到head/finalnorm与有限decoder传播，
compiled seed与root的事件logprob差相同，没有旧head倍率放大。
只准备GDN boundary-row存储候选，原FLA计算、Cache更新、dtype及正式源码不改；
同一真实B8的旧/候选消费状态、Q/V/A逐值及热耗时对照尚待执行，不新增容差。
4/5原诊断已退出，后续候选测试必须用独立job和SHA，不能把d1b67a2读账当修复通过。

20:06原AppWorld PID1856052两rank完成三个回报类别组、各199个B4；合计组墙钟
1515.39/1515.13秒，其中归因body1293.55/1293.69秒、prefix准备220.33/219.91秒，
剩余外层约1.5秒。第四组当时64/101；这些是已完成类别组，不是完整step21。
[有界阶段回执](results_dt_phase_bounded_20261005.json)绑定原日志、PID出生时间和
source SHAaa8fac16…；没有因等待整轮结束而延迟定位。

20:09容器使用451.21GB，匿名384.30GB、文件缓存66.61GB；两个DT worker
PSS142.54/145.59GB。原e5eb4afc…worker及实际环境均启用阶段末原host-cache
释放，四次组返回日志的reserved均回到32858176416字节/rank。不能将全部PSS
称为live tensor或泄漏；普通CPU prefix artifacts也不属于该pinned allocator。
已记录的MetaX allocated计数缺陷使该counter不能解释成live字节，reserved及原
释放日志另用。现有13k热B8 trace中的27.25/25.06GB HtoD仅0.659/0.667秒，
不是整个模型重复搬三遍或可立即删除的27GB工作量。

首组内部残差标记236/243个、各448个contrast，实际规则为
abs(root-signed_sum)<=0.02*max(1,abs(root))；不是FA/FLA官方容差，也没有纠偏。
FP32事件head与实际FP16 FLA/BF16 FA来源不变。现有真实13k报告仅保存总残差，
未保存原runner已经返回的layers ledger，因此在空闲4/5做一次原真实B8补充观察。
来源`d1b67a2`，PID2411666/出生1791203111.34，原checkpoint20、原88请求bank、
context排序offset84、每卡4。只保存已有逐层标量与去重storage/消费计数，不增加
forward、optimizer、dtype变体、cache清理或容差；原CPU接口检查返回0。
输出在`receipts/owner-b8-dispatch-20260930/native-prefix-reuse-ledger-offset84-20261005-d1b67a2-current-global_step_20`。
提交`d1b67a27cc368a5a81a2dde596c8702a417621db`已推送，它是诊断版本，不是
正式训练或新的DT数值发布。AppWorld2/3、训练参数、TextCraft待训练起点状态均不变。

### 2026-10-05 19:43：实际漏奖范围与当前DT组进度

TextCraft原停止作业92个完成迭代的原metrics及178个DT报告已只读提取，原train.log
SHA54f4aef6…；8批原rollout score另按原jsonl逐行计数和SHA绑定。
[实际范围回执](results_textcraft_actual_reward_loss_scope_20261005.json)确认step28为
135/256成功、DT非零reward事件73+61=134；step109为3/256成功、事件1+1=2。
原padding只可能增加该计数，故两批各至少遗漏一项真实正奖励。没有原完整source rows、
native slices与padding inverse，不能声称已确定全部漏奖身份或精确频率。

24/92批原response cap-hit ratio为0；其中step55/70/80/91/93真实成功依次
102/50/35/21/8，每批256条，DT事件计数与成功数相符，但entropy依次
0.958/1.304/2.148/5.026/5.498。实际metric_utils SHA80e18175…与固定VERL20bd331
原文件相同，其指标是attention-derived长度等于stored response width的比例。
这些批次不能用当批“终局action被完全截断、丢reward”解释；仍不能排除此前更新的
累积影响。不将三位小数pg_loss=0推成零梯度，也未改熵、学习率、信用倍率或同步。
已修接口和训练起点待答状态保持不变。

AppWorld本次原采样已结束：217条实际轨迹、3460个response、840146个policy token，
最长context24436；原运输日志末值945097 generated token/1513.7秒，包含已取消
轨迹的生成，不冒充只计保留token或纯decode。19:43原两个worker均处于DT，当前
回报类别组77/112个B4，最近批次约4–9秒；112不是整轮总批数，不据此外推整轮。
原native-phase-bounded.json保存采样结束及原old_log_prob阶段；未等待完整迭代。

### 2026-10-05 19:28：当前采样、物理资源与已推送版本

只读回执`appworld-eval-client-routing-20261005/deployment/phase-followup.json`
再次绑定PID1856052/创建1791197944.7。AppWorld原rank0/rank1分别完成107/120、
108/120条实际训练轨迹，尚未写入本次新检查点，不把部分轨迹或GPU忙称为完整更新。
物理GPU2/3分别55315/55297MiB；4/5各860MiB且无模型进程，主机可用
845908254720字节。0/1与6/7仍空置。SQL与TextCraft旧错误尾日志均来自已停止
的原PID，不归入AppWorld当前异常。

真实B8及正式step20账目已交叉复核：430次B4/rank的实际归因body占DT约86.2%，
bank约13.6%，外部未解释时间仅5.52秒；没有证据将整段replay或all-gather等待
当作可立即删除的浪费。全CPU root保留已测更慢且未部署，未因此添加新的优化。
接口修复、测试与版本记录提交`67f0e16`已推送，远端分支完整提交为
`67f0e1620f37ce7440339523b12a0b0d7409f79e`。该Git提交号不是远端数值核心版本。

### 2026-10-05 19:02：AppWorld恢复20；TextCraft旧作业停止、v2修复待训练起点

19:16实际原TaskRunner调用栈指向新entry的loop_owner_worker、loop_async_transport
和loop_owner_rollout；已进入原训练采样。原rank1已返回54/120轨迹，观察到两条
官方训练ret0.857/0.800。它们不是整批成功率、评估分数或新完成更新。
另已核查checkpoint100实际每rank496个LoRA tensor全FP32/Shard(0)，不同于基座BF16；
原state_dict prehook/full_tensor/all-gather物化独立完整payload。vLLM packed scaling
状态与缓存激活链路不支持“回写actor或重复缩放”假设，未添加clone、dtype或同步
改动；11项实际源码SHA绑定的只读审计见TextCraft修复回执。训练起点仍待用户选择。


AppWorld新PID1856052/创建1791197944.7，GPU2/3；原submitter从完整checkpoint20
恢复，当前初始化。实际entry为appworld-eval-client-routing-resume-20261005-v1，
含本机CPU v2已修兼容性的worker SHA1bf093a2…、rollout01eca1bf…、transport8ecbf018…。
三个文件之外entry bytes、原VERL与DT目录均不变；配置composer同输出/恢复点
逐值比较，仅custom dataset定位随entry迁移，官方参数逐值相同。新source.json及launch/job在
appworld-eval-client-routing-20261005/deployment。部署来源6d92fe3；本次启动不计
新完整迭代，不提前承诺评估提速或健康性。数值核心仍c9＋8e7dd71 prefix provider。

TextCraft原PID3218909已停止、完整检查点100保留，晚期完成到118的旧日志仍在，
这些旧训练包含已确认的奖励mask错误，不作为修复后的有效实验。4/5已空闲。
准备完成textcraft-truncated-terminal-resume-20261005-v2，来源e42f596；只包含
已测的真实reward/native slice接线，另将原6781bdd已应用的阶段末host-cache释放
与原Ray环境变量折入恢复入口。原卸载策略、LoRA8/16、每卡B4、PPO、官方
训练及采样参数不变，dt_root仍c9cd147。真实17/1 CPU回执与基座精确值检查已绑定。
候选尚未提交；从原模型重训或checkpoint100继续的选择已询问用户。

未接受的textcraft-truncated-terminal-resume-20261005-v1保留：准备助手误将
AppWorld的data.custom_cls.path假设套给TextCraft，配置比较KeyError后退出，
没有提交模型或训练。e42f596仅移除该无来源假设，比较实际TextCraft owner配置；
v2完整比较通过。SQL仍停止，仅已有prepared版本；0/1与6/7未占用。
19:02物理GPU2/3各19952MiB，4/5各860MiB，主机可用930229796864字节。
当前current_runtime.json采集19:01:49，无entry或数值source hash不匹配。


### 2026-10-05 18:49：修复真实终局奖励丢失，评估服务路由候选补齐默认接口

[TextCraft回执](results_textcraft_truncated_terminal_20261005.json)绑定实际旧入口
SHA92c724a5…、PID3218909/创建1790895651.16。官方AgentGym在整条轨迹结束后
将训练response截到10240；旧本地接线把截后无slice行标为inactive，进而将该行
真实终局奖励排除。22轮×511token的原truncate/source-row AST复现该缺陷：末行
reward1存在、没有训练slice，却使原episode_returns的输入全部变0。

候选仅15行接线变化：真实已执行事件active=True；原complete return先包含全部
未来真实奖励；原dt_response_slices确定需要DT的源行。奖励不迁移，未保留动作
不新增DT请求，部分保留动作仍使用原完整native response端点。Q/V/A、事件读出、
DT有限传播、原PPO、任务parser/采样/奖励/截断参数均未变。真实Torch/VERL CPU
接口检查17通过、1项可选原archive AST检查跳过；本机源契约5通过。这不是新的
模型数值容差验收，不替代原FA/FLA/VERL/vLLM适用回执。

TextCraft晚期step114–117每256条仅1/0/0/1成功，出现无效动作并跑满30交互。
checkpoint100两rank各427个冻结基座tensor与原BF16权重精确相同；不能据此
宣称LoRA同步或训练质量已恢复，也尚未证明mask缺陷解释全部退化。
通过原stop-only助手停止旧树，保留原完整checkpoint100与全部旧日志。
修复候选尚未部署，训练起点已请求用户裁定，不能把旧损坏训练视为正确结果。

[AppWorld评估IPC回执](results_appworld_eval_client_routing_cpu_20261005.json)的
v2保留原LOOP轮转选中的客户端编号：原eval world1只返回一个客户端，导致
双卡配置只向server0发送请求；该轮原评估3460.799秒。候选仅传原选择的索引，
生成/取消按服务编号，回复按原环境rank。审查发现v1在默认单服务eval产生五字段
而原同步collector只收四字段，v1保留为未接受候选；v2只在多服务eval附加索引。
单服务eval/训练默认四字段不变，原同步collector实跑、双服务路由/取消共10项
CPU通过。未新增轮转器、任务执行器、采样或推理内核；真实速度收益尚未测量。


### 2026-10-05 17:41：真实长B8有界profile及原生反向参照完成

[本次回执](results_current_owner_hot_b8_20261005.json)绑定诊断入口`067e641`、
原检查点20和已冻结的AppWorld source SHA `5f190f46…`。两rank取原88请求中的
最后一个B4，事实输入分别B4×13468/B4×13216，DT配对端点分别B8×13468/
B8×13216。这是实际请求耗时检查，不是新增32768容量或整网数值验收。

当前实际runner SHA为`6348ebef…`，对应`8e7dd71`原生prefix provider扩展；
不是目录内保留的nested c9原runner `c7fc969f…`。对Git原字节的再次核对证实
`replay_cache.reorder_cache`及其后的有限传播语句不变。准确版本口径为
“c9数值核心＋8e7dd71 prefix provider”，不能用c9旧runner SHA替代实际进口。

两rank首次bank准备48.029/48.075秒，随后未开profile的热DT为12.003/12.011秒；
原生target autograd热forward为5.839/5.810秒、热backward为15.878/15.918秒，
每rank496个梯度tensor均有限。两者复用同一原actor/checkpoint和原卸载配置；
DT原事件head为FP32，原actor logits为BF16。没有新增head或反向实现，
没有optimizer更新，不是PPO梯度/容差对拍。首次准备成本单列，不能把热调用
说成整个DT流程仅12秒；profile导出的31/30秒也不是正常DT耗时。

实际热phase：root两rank约3.125秒，32层replay合计3.102/4.363秒，finite decoder
4.646/3.568秒。完整replay不等于可删除的MLP工作量，不据此部署root tape、
排序或低秩缓存。原trace保留在远端receipt目录，CPU分析不再占用模型显卡。
外部物理采样2/3峰值36857/36099MiB，测试进程树PSS峰值67713381376字节，
整机最低可用756318842880字节；历史oom_kill=1未增加、failcnt=0。
本次无vLLM共驻，不能把这些峰值扩展成完整训练容量结论。

2/3有界测试已结束，AppWorld完成标记20保留；4/5 TextCraft正式训练继续。
评估单环境rank与推理服务索引的接线仍在单独核查，不把未部署修复称成已提速。


### 2026-10-05 17:20：原step20完整保存，2/3释放给有界实际输入测试

[本次恢复回执](results_appworld_native_prefix_resume_20261005.json)绑定原TaskRunner
step20行5909/SHA c716c2fb…；原计时采样1445.833秒、旧概率384.645秒、
DT3427.063秒、actor2785.245秒、评估3460.799秒、保存49.306秒、整轮11553.830秒。
训练reward均值0.796、原评估test_score0.333、梯度范数0.001；这些来自完整原日志，
不把评估混成训练成功率。新目录完成标记20和data/model/optim/extra_state七个
原文件均存在，两个rank model各8997781971字节。

按用户最新授权，既有停止助手c134d93增加stop-only，不修改训练代码、保存或恢复
实现。真实PID3658900/创建1791175714.44、当前冻结entry/VERL/DT源SHA匹配后，
在原完成标记20保留完毕的条件下停止该进程树；第21轮未完成部分不作为训练完成。
原停止回执及17:20事后检查保存在appworld-prefix-resume-20261005/
completed-step20-and-stop.json，SHA ce148902…：没有该树非zombie残留，
TextCraft PID3218909/创建时间未变。2/3物理各859MiB，4/5正式训练继续；
主机可用846618984448字节。当前快照17:20:38再次确认没有源码漂移；
AppWorld已停供测试，不把停止前状态或旧PID称作仍运行。

[原88请求CPU去重核查](results_native_prefix_duplicate_work_20261005.json)证实
同UID已只取最长canonical：每rank88请求、18个canonical、5次原B4 capture。
跨UID短初始前缀去重41920槽位，占bank约21.8%；正式bank464秒占DT3409秒
13.6%，按冻结88槽位线性且零固定开销乐观折算也只约整DT3%，并非已实测收益。
没有新跨UID缓存/调度器、全量root tape、排序或低秩候选进入默认训练路径。
长B4 root/replay仍沿保存的真实请求与原owner做一次有界profile，不再整轮采集。


### 2026-10-05 15:16：旧SQL资源缺项已准备补齐，AppWorld原更新已返回

[SQL准备回执](results_sql_native_host_cache_20261005.json)绑定新隔离候选
sql-native-host-cache-20261005-v2，prepared SHA630ed1f6…，准备提交2b12898。
复用6781bdd原补丁，worker807e518…→e5eb4afc…；SQL原launcher仅插入
原Ray runtime_env.env_vars的VERL_RELEASE_UNUSED_HOST_CACHE=1。
另两项配置差异仅为同SHA Dataset/template的冻结路径。整树比对只改变上述
worker和launcher；actor1f862e8b…、DT c9、官方任务配置、LoRA8/16、每卡B4与
32768未变。原16项CPU接口检查25.66秒通过，原3项资源边界检查通过；
它们不代替GPU容量、DT/PPO/vLLM数值容差或SQL训练健康验收。
旧目录、active清单与作业不变；新候选prepared-only，没有分配GPU或启动训练。

首次准备bf0dfa8因错误按LF匹配冻结launcher的真实CRLF行而失败。原源SHA51a854…
已实际取回，修正插入后SHA2f4588…、其余字节不变；失败候选及回执保留，不加入
默认launch路径。该问题属于准备器插入错误，没有改动正式算法或环境。

15:15:58原AppWorld两rank已打印update_actor阶段末释放，reserved各从
66252495789降至32858176416字节，耗时2.219/2.242秒；原TaskRunner进入
本次官方评估，实际已返回534个请求、84542生成token，运输elapsed328.1秒。
这包含环境/生成RPC，不是纯decode。检查点尚未保存，符合原fit评估后保存顺序。
物理GPU2/3为56286/48000MiB，主机可用382603558912字节，原oom_kill仍1。
原step18/19的更新耗时3003.368/2673.444秒已从原日志取回，不能与不同本轮轨迹
直接宣称同负载提速。源码快照14:58:11 SHAaff5e942…无漂移，TextCraft已完成115。

用户随后允许2/3做必要有界测试、4/5保留正式训练（或反过来）。当前正在评估的
AppWorld新更新先保留到原完成保存点；同步分析现有实际profile和请求，避免
只等待整轮。SQL尚未启动，GPU0/1与6/7继续空置，GRPO不再提交。

### 2026-10-05 14:52：只核对旧问题的修复范围，没有部署新变体

原PID3658900/创建1791175714.44再次匹配；两rank仍在本次原actor更新，
没有新的完整step或检查点。非阻塞原栈确认实际调用为冻结VERL的
dp_actor.update_policy → torch.backward → 原FSDP2 foreach_reduce。
这只证明采样时的调用路径，不把一次栈当成更新吞吐或无阻塞证明。
原5秒物理采样GPU2/3各51696MiB、App PSS398037095424字节，
主机可用346894004224字节，oom_kill仍1、failcnt0。
对应只读phase-1791183084.json及owner-update-stack-1791183154.json保存在
research/temporary/rl_upstream_alignment_20260929/appworld-prefix-resume-20261005。

修复边界不扩大：已上线前缀复用的同88请求196.66→158.72秒；本轮DT已返回，
整轮更新、保存及后续生成尚未验收。root/replay剩余计算并未宣告消除；
GPU/CPU全量root保留、重排和低秩候选均未进入正式路径。
当前固定VERL原fit在step20的顺序是actor返回→评估→保存→完整step日志，
不能在actor或评估未返回时把尚无完成标记认定为保存错误。

本次仅只读复核SQL旧prepared ecf31cc9…：原worker807e518…仍缺少
AppWorld/TextCraft已验证的e5eb4afc…主机闲置缓存阶段释放补丁，准备配置也没有
VERL_RELEASE_UNUSED_HOST_CACHE=1。它仍是prepared-only，不计入已修好的SQL部署。
SQL没有完整检查点；旧恢复入口固定0/1且校验退役PID，不能照旧启动。
本轮未修改或启动SQL、未改算法/配置/数值容差，未使用0/1或6/7。

### 2026-10-05 14:29：本次完整DT已返回，原actor更新开始

[当前恢复回执](results_appworld_native_prefix_resume_20261005.json)绑定原readout日志行SHA：
七个类别RPC依次122/49/15/87/9/141/7次B4，两rank各430次、1720个response请求。
rank0原RPC耗时合计3409.589秒，rank1为3408.454秒，双卡并行不能相加为113分钟；
约56.8分钟是本轮全部DT请求合计，不是一次runner或一次训练迭代。
含bank准备的平均B4耗时约7.93秒；bank准备约464/463秒，占各rank总量约13.6%。
本轮max_readout为29521/30389。原worker实际已进入actor_rollout_update_actor，
尚无新完整step或检查点，不能把DT返回称为本轮训练已经验收。

14:29:01原5秒物理采样GPU2/3各51636MiB，App进程树PSS398053366784字节，
主机可用347000586240字节，原oom_kill仍1、failcnt0。未使用0/1或6/7。
14:19:28源码快照SHA118a5e95…再次确认原PID创建时间、entry和数值源没有漂移；
actor仍1f862e8b…、worker仍e5eb4afc…，新检查点尚空，TextCraft已完成step114。
截至14:14:25两rank原.err没有recompile/graphbreak/非有限/OOM匹配告警；仅有
原初始化/版本/指标服务及vLLM短序列形状提醒。无告警不能证明绝无编译成本，
也不能把较长B4直接归因于编译。本轮没有新的算法、配置、模型测试或部署。

### 2026-10-05 14:11：只读取回原残差明细，未据诊断计数改数值

[当前恢复回执](results_appworld_native_prefix_resume_20261005.json)新增两份有原路径、
原日志行SHA的只读残差记录，没有模型调用或新容差。正式首组各488条：rank0
绝对残差中位0.024341、观测p95 0.092918、最大0.217002；rank1为0.022916、
0.078759、0.239073。source signed d总体范围-0.362804至0.435319，未见旧倍率
纠偏的数倍信用量级；这不是每个token反事实准确性或整条DT通过官方容差的证明。

已读取原88请求的远端rank JSON（初始化actor，同输入原/复用路径）：original_warm
诊断false为58/48，shared_warm为60/44；最大绝对残差分别0.173678/0.183956和
0.198724/0.183792。原路径本已存在该残差，不能将当前全部标记认定为新增前缀错误；
正式checkpoint19的轨迹不同，也不能逐条归为旧误差。FA/FLA已有实际操作数原
对照的范围保持不变；未加归因倍率、裁剪或整网门槛。

14:02:33原日志按路径去重读取：AppWorld前四个原RPC已返回122/49/15/87个B4，
各约945.34/405.59/133.50/629.45秒，第五组7/9；尚无新完整迭代或检查点。
物理GPU2/3为29358/29218MiB，App PSS311681751040字节，主机可用
476464369664字节，原oom_kill仍1、failcnt0。TextCraft原指标已完成step114，
gen3859.781秒、DT10.949秒、actor595.581秒、整轮4770.307秒，原检查点标记100。
SQL仍退出、0/1空置。没有部署被拒绝的root保留或新算法/训练配置。

### 2026-10-05 13:51：同一正式DT已返回三组，原剩余大头未宣告解决

[当前恢复回执](results_appworld_native_prefix_resume_20261005.json)绑定原PID3658900/
创建1791175714.44，读取原worker完整readout行及行SHA。两rank依次完成122、49、
15次B4，原RPC各耗约945.34、405.59、133.50秒；已进入下一组87次B4的准备。
这些是三个类别RPC，不是三次训练迭代，也不是整条DT阶段或整轮更新耗时。
首组每rank72个独立历史、18次原capture，bank准备约146.6秒；没有新增计时器。
正式轨迹与旧轨迹不同，不能用这些时间宣称同负载整体提速或原效率问题全部修好。

最新5秒资源采样：物理GPU2/3为29328/29188MiB，App进程树PSS
292154768384字节，主机可用518665748480字节；原oom_kill仍1、failcnt0。
当前无新完整检查点。共享前缀修复已部署，但root/replay剩余计算尚未解决；
因容量或38%耗时回退而拒绝的GPU/CPU全量root保留没有部署，排序与低秩候选
也没有进入正式入口。不把候选副作用当作新增正式故障，更不改变Q/V/PPO、
LoRA8/16、actor/DT每卡B4与32768上界。

首组conservation_failures为260/253，来源是原c9整网归因总和诊断，不是官方
FA/FLA算子超差计数；完整RPC返回亦不能代替算子对照。保留原字段并只读核查
残差来源，不加纠偏或新容差。此前FA/FLA原参考和断言回执的适用范围不扩大。

### 2026-10-05 13:34：正式前缀路径实际进入DT，首组66/122批

[实际DT回执](results_appworld_native_prefix_resume_20261005.json)绑定
first-dt-1791178447.json：原采样末已交付886853 token，运输elapsed1394.3秒；
随后两rank原compute_dt_token_advantages各提交488请求/122个B4，现已完成66。
两卡单批中位4.305/4.295秒，范围2.776–26.389秒，最长项未分解为编译或计算。
这是当前回报类别RPC的部分批次，不是整轮DT，更不是完整PPO更新。
前缀bank准备总量仍待原readout结束报告，不能拿plan到首批的间隙全部称bank。

现有py-spy非阻塞栈实际指向v2冻结runner/credit/readout/producer与原VERL worker；
五个当前源SHA与prepared逐项匹配，没有候选或配置漂移。
54次原5秒资源采样中App进程树PSS最高320052067328字节，GPU2/3物理最高
29337/29195MiB，主机可用最低444804636672字节，原oom_kill1→1、failcnt0。
PSS从19批到66批约320GB，没有继续按批线性增长；不是整阶段峰值或内存归因。
已打印d范围约-0.322至0.435，未见本次异常退出；完整DT、PPO、检查点与后续
生成仍未验收，不以这些部分批次宣告训练健康或小时级效率问题已解决。

### 2026-10-05 13:20：继续原重复计算问题，排除低收益排序

[真实批几何核算](results_native_prefix_batch_geometry_assessment_20261005.json)仅使用
原保存的88请求/rank、既定B4和本地prefix边界，没有模型/GPU调用或新调度器。
当前两卡166372个suffix槽位包含88018个action/query/target、33162个右padding、
45192个共同前缀之后的历史；旧跨卡MIN已排除，不能把它当作当前新增问题。
按source_start排序反而增加3.76%；按64对齐start及context排序仅减少1.58%，
且rank1变差。因此没有改正式排列，也没有将槽位差写成时间或FLOPs收益。
原捕获字段已为必要集合；此前真实trace中的dense LoRA应用0.131/0.157秒、
物化约0.013秒也不足解释主要耗时，不继续用这些小项启动候选或GPU测试。
原root/replay大头尚未解决，拒绝的GPU/CPU全量保留仍未部署。

[同一AppWorld恢复的有界观察](results_appworld_native_prefix_resume_20261005.json)新增
13:17:53记录：PID3658900/创建1791175714.44匹配，原rollout已交付3685次生成、
859337 token、elapsed1328秒，尚无本次DT批次或新完整迭代。
物理GPU2/3为55316/55226MiB，App进程树PSS110028387328字节，主机可用
704689729536字节；原oom_kill仍1、failcnt0。SQL仍停、0/1空置，TextCraft继续。
这是原作业的特定阶段观察，不是整轮健康或前缀复用正式提速验收。

### 2026-10-05 12:57：AppWorld原检查点加载返回，正式rollout推进

[本次恢复回执](results_appworld_native_prefix_resume_20261005.json)新增1791176244.48
原阶段记录：TaskRunner此前已设置global_step19，两个原load_checkpoint RPC返回后
进入execute_method和原LOOP rollout。运输累计calls240、已交付requests177、
生成25221 token、elapsed100.5秒，completed_ranks仍0。它不是纯decode速度或
新完整迭代；尚未取得本次DT/PPO批次的正式效率。新PID与创建时间仍匹配。
物理GPU2/3占用55314/55226MiB，主机可用712651153408字节，App进程树PSS
102077540352字节；0/1仍空置，TextCraft4/5继续。保留原CPU准备失败与后续修正
记录，不将启动、加载或首批生成称为三个任务全部修好。

### 2026-10-05 12:51：AppWorld已提交前缀复用恢复，初始化中

[本次恢复回执](results_appworld_native_prefix_resume_20261005.json)绑定修正8a23818、
实际准备1ae223cc…、提交助手6f6e3be…、新PID3658900/创建1791175714.44，
GPU2/3。由原VERL加载器指定原完整global_step_19，没有改变任务/采样、PPO、
LoRA8/16、actor/DT每卡B4或32768上界。原retry worker6ad5…、actor1f862…、
host-worker e5eb…和release=1均继承。未启用拒绝的GPU/CPU root保留。

首次d3b8bcf源码准备因误认为credit在entry而失败，没有prepared.json或GPU作业。
只读查询确认原credit实际来自c9/experiments/rl，修正落在同一owner路径；
提交助手仅将已展开的两个DT PYTHONPATH项指向选定版本，保留entry优先和其余
依赖顺序。旧失败目录保留，v2原配置composer及实际模块路径核对通过。
13项现有Cache/provider/factory接口测试通过（14.78秒），Git源比较另有原AST
对照，不把接口夹具当作新的数值证据。原FA/FLA及本次真实ht回执仍单独索引。

12:51新worker3665902/3667445基础权重已读，仍在原模型/引擎初始化；检查点
指定19不等于此刻已经完成加载，更不等于本次完成新迭代。当前快照绑定新清单
runs/appworld-native-prefix-resume-20261005-v2/formal-training.json。原被动资源
观察PID3673733/创建1791175812.75，源b5ce0035…，仅采物理GPU/PSS/主机/cgroup。
SQL仍停止无检查点，GPU0/1保持空闲；TextCraft原PID3218909继续4/5，已完成113。
正式DT效率仍待新实际批次，不能把88请求约19%的对照扩大成整轮提速。

### 2026-10-05 12:26：原前缀续算检查完成，仍未宣告整体修复

[真实分段状态回执](results_native_prefix_segments_20261005.json)绑定2a9456b诊断、
PID3485128/创建1791174077.38，原完成标记已写且进程退出。使用已保存真实B4
操作数，按原lease factory的实际边界传递FLA FP32状态；最后6720–6976段与
原recurrent参考对照，两rank原ht误差指标0.000284138/0.000282656，均通过
原assert_close('ht', ..., 0.005)，没有FLA CI宽免、归因纠偏或新容差。
HF原Cache将最终状态存为BF16，与已记录实际状态逐值一致。检查未加载模型、
未生成轨迹；它覆盖这项新增状态续算，不作为整网/PPO认证。

两rank脚本实际47.208/20.816秒，另有启动与导入成本；5秒资源采样中GPU2
物理峰值2216MiB，诊断进程树PSS峰值15102793728字节，主机可用最低
868469366784字节，原oom_kill1→1、failcnt0→0。GPU0/1未使用，原TextCraft
继续4/5；该资源回执不能外推正式240轨迹整轮峰值。

当前没有部署root保留候选或前缀新版本。重复前缀复用仍只在同一88请求对照中
实测约19%收益，不能称原小时级DT已修好。GPU/CPU保留新增显存或耗时是候选
副作用，拒绝部署后仍保留失败来源。继续原重复计算的修复，不将候选副作用当作
新的正式故障来扩展任务。

### 2026-10-05：复用已有官方算子证据，继续处理原前缀重复计算

[前缀官方对照索引](results_native_prefix_official_coverage_20261005.json)将已完成的实际
FA/FLA输入、原参考/断言与712795d前缀源SHA对应起来。root-tape诊断中的
shared_warm本身没有保留root capture，可复用它的非空BF16状态/FP16 GDN后缀和
完整BF16 FA后缀操作数对照；不因此接受已失败的GPU/CPU保留候选。
[默认owner对照](results_native_prefix_default_owner_20261005.json)本次实际执行stdlib
AST投影，确认6348 provider默认路径的完整模块与c9原owner相同，有限数学未变。
这些不是整网容差；不增加整条DT的自定义门槛。尚需核查的具体新增计算仅是缓存
生产时原FLA fwd_h在多个真实边界间传递FP32状态，现有从0直接计算/后缀消费
对照没有直接覆盖它。原真实B4操作数已保存，可直接核查，不重载模型或重跑88请求。

本次原日志/PID/物理资源只读采集于1791173005.9116776，原回执为
../../research/temporary/rl_upstream_alignment_20260929/phase-observation-20261004/prefix-context-1791173005.json。
TextCraft原PID3218909创建时间仍匹配，完成step112，原检查点标记100；该轮生成
3795.342秒、DT65.166秒、actor588.385秒、整轮4752.114秒，奖励均值0.004、梯度
0.003。SQL/AppWorld仍退出，原标记分别无/19；没有启动或部署新候选。
物理GPU0/1/2/3/6/7空置，4/5仅原Text worker；主机可用880038387712字节、
cgroup130640531456字节。新索引或准备文件不算训练提速或恢复完成。

### 2026-10-05：CPU root候选已执行，因性能回退拒绝部署

修复代码aac4121仅准备隔离候选：沿原copy/restore/consume接口扩展，accelerated
retained/local-event类只改导入，所有类/方法AST及有限数学函数不变。真实backend明确
绑定私有transport父类，参数统一defer_host_sync；诊断backend仍同步。目录创建冲突
已修。aac4121首次远端CPU检查因测试源码位置错误59通过/16错误，未提交GPU作业；
原失败记录保留。c79ff3c仅修测试的发布路径与stager标量显示，随后75项通过，
无skip/放宽断言，使用原有Torch与环境。

[真实最长B4执行回执](results_native_root_tape_cpu_20261005.json)绑定c79ff3c，
PID3183518/创建1791171308.26，已完成退出。原排序12–15行，实际suffix2630/3332，
上下文最大5382/6084；原全88请求bank保留，每卡B4、LoRA8/16及原输出1500不变。
原shared热调用15.018/15.012秒，候选关闭15.043/15.037秒；CPU保留候选热调用
20.756/20.729秒，增加38.21%/38.08%。root从3.783/3.780到7.779/7.776秒；
replay从5.313/3.652到0；finite从4.733/6.177到9.465/12.006秒。
首个CPU调用88.026/88.033秒，其中root74.722/74.717秒；尚未将冷成本细分为
allocator、编译或具体复制。候选新增CPU运输抵消重算收益，**拒绝正式部署**。
这不是原训练新增的故障，也不是旧重复计算问题已经修好。

[被动资源采样](results_native_root_tape_cpu_resources_20261005.json)39项、间隔5秒，
物理GPU2/3峰值39976/45384MiB，进程树PSS峰值63.24GiB，主机可用最低384.49GiB，
cgroup最高240.53GiB，原OOM计数1→1、failcnt0→0。它覆盖整个有界诊断，
不声称每阶段瞬时峰值；XTT原共享计数不累加，PSS不能代表未映射的驱动分配。
32层capture均被消费。helper的snapshot_copies_added=0只表示没有额外审计克隆，
不能说CPU运输没有复制。原始向量差只作诊断，不新增整网容差或纠偏。
[拒绝范围与版本对应](results_native_root_tape_cpu_assessment_20261005.json)固定本次
结果：接口已实际执行，效率不接受；未替换任何正式worker或参数，也未以性能失败
候选启动额外容差长测。继续只处理原root/replay及重复前缀问题，保留原验证基线。

### 2026-10-05：候选接口核查，不将结构测试当作接通

此前重复前缀/root replay问题尚未完整修好。全量GPU保留新增的显存压力属于候选副作用，
不是原正式训练新增的大头；不能将提速候选当成已完成修复。正式版本未部署该候选。
本次只读核查确认CPU候选传`defer_sync`，私有transport接口名为`defer_host_sync`；
实际accelerated code-local backend仍继承原retained/clean类，runner fallback导入修改
并未接通实际构造路径。11项结构测试的Capture记录器接受任意kwargs，未覆盖这个接口。
上述结论来自源码/签名核查，不声称已执行真实构造失败；候选保持prepared/unaccepted。
本次没有GPU测试、正式部署或新的数值容差。
[逐文件SHA与核查范围](../../research/temporary/rl_upstream_alignment_20260929/phase-observation-20261005/root-capture-candidate-interface-audit.json)。

### 2026-10-05：同一修复的剩余成本；未接受全量 GPU root 保留

本次读取已完成 fd2b25d 的真实 B4 hot trace，不增加模型前向、GPU运行或新的
验收门槛。当前问题仍是此前重复前缀和 root/replay 重算的修复，不另开训练算法。
正式版本未应用这些候选；已测到收益不能写成正式训练已经提速。

| 原问题 | 已取得的实际证据 | 尚未完成的部分 |
| --- | --- | --- |
| 重复前缀计算 | 原88请求196.65→158.74秒，已包括bank准备 | 仅对应保存的实际请求；尚未正式部署 |
| 同一次DT的root/replay重算 | 真实B4热调用3.698/3.702→2.991/2.981秒，32层replay为0 | GPU保留的显存代价尚未满足完整真实输入范围 |
| 候选的容量 | 32768总长原夹具通过，每卡B4、LoRA8/16 | 夹具suffix832；保存的真实B4最大suffix2630/3332，未由该夹具覆盖 |

[原参数阶段拆分](results_native_root_tape_parameter_phases_20261004.json)直接读取
同一trace：原prepare32次CPU inclusive仅0.07519/0.07389秒；相关device HtoD
6,963,708,544字节、0.175/0.179秒，相关all-gather0.274/0.076秒。全次HtoD
19.423/19.501GB、0.571/0.585秒，不能把这些设备时间与CPU同步时间相加。
原分析器默认输出及原事件分区在两份真实trace上保持逐值相同。

[原runtime API归属](results_native_root_tape_runtime_cpu_20261005.json)进一步确认：
最大的mcMemcpyAsync对应最终signed token向量回传，rank0为18,432字节
（4×576×FP64）、rank1为19,520字节（4×610×FP64）。CPU API分别占
1.164304/1.163577秒，device实际复制仅38.912/46.848微秒；该边界等待此前
GPU工作完成，不是搬18KB花1.16秒，不能算成另一个可删除的搬运大头。
这也不授权删掉最终返回、有限计算或原诊断。原prepare/release与有限算子未改。
实际c9cd147 environment.json SHA为2467f6555282c52b4dc646c2242dc700eb5adff46c3c83701d54dfbf1bbf3ad7，
已读原内容并绑定dt_pin_root_host=true，不把已开启pin_memory当作新修复。

[真实请求形状](results_actual_prefix_request_geometry_20261004.json)仅CPU读取原
88行/rank的实际请求和token IDs；最大后缀对应原排序12–15行，完整上下文
分别5382/6084，suffix2630/3332。热计时40–43行只有suffix576/610。
全量GPU保留已测短后缀峰值34.54/36.20GiB，原路径约11.45GiB；这项额外显存
是候选尚未解决的成本，不能隐瞒、不能据832容量把它认定为真实范围已通过。
候选默认False、保持prepared_only_not_deployed，未替换正式worker。
[同布局字段保留量外推](results_native_root_tape_retained_storage_20261005.json)只读实际
inventory并按原storage别名去重：最大真实B4约123.38/156.09GiB，分别包含
decoder56.50/71.59GiB、GDN60.11/76.11GiB及FA6.77/8.40GiB。
短576/610和832容量场景的同一计算为28.55/30.22/46.57GiB，与原保存的字段量
对应。该数值不是实测物理峰值或OOM结果，不包含完整跨层/已有cache别名与分配
生命周期证明，也不能加到独立baseline峰值上。它直接暴露全32层GPU保留布局的
规模问题；不得先把这个候选投入正式任务，随后再用OOM确认。后续先沿既有owner
capture/卸载/释放接口处理这项同一修复的代价，不能再堆新的优化来掩盖。
本次只沿原捕获/运输/释放接口核对该成本，不加入减小B4、缩短输出、改变rank/alpha、
信用裁剪或新卸载器作为替代。

来源对应：hot原诊断提交fd2b25d，PID2521687/创建1791125978.23，已退出。
两个原trace SHA为4df67108ca903dce9b0d970b39c42d16d2b9563916d16b13ef71872132cc427a及
e76e00a28d8b9352f33eded213b1c53777d85ec2895ebf1ed646a81581c011ff。
参数分析器SHA405e730b42ff1aaa4a989d4d0ae8ec3540dd61f4345f308718ce262623a4f16a；
runtime分析器SHA9eb0c1cdffee854e88a58da7625b4e21f0dba757c68b83d7fb7f79398a548c46。
原几何观察脚本执行SHA9b55b18efe8f11c19985b0a882eeb83aa381cdced5d6230c6c026316599377e2；
后续仅修dotted配置键展示的本机源码尚未重跑，不能冒充该原回执的执行版本。


### 2026-10-04：原 root capture 复用，实际操作数与官方容差来源

本次只在隔离诊断 owner 中复用原 root forward 的32层 capture，减少同一次 DT
的逐层前向重放；正式训练尚未部署。原 owner SHA为6348ebef115ff0ea45deeee3df8b65dcfbf0d1ae86619edc67f34d16d92d69aa，
候选SHA为ebc563e5bd07ae290574e4ca494311cc578fcac63a020f5687f4dd95fece0b20，
helper SHA为9cbded61baca35c07c7c7e1096f17a769d45fffddbfd0ab646e42036b07f2fd7。
默认reuse_root_captures=False，原有限算子、FA/FLA、FSDP参数准备/释放、CPU卸载、
LoRA8/16、每卡B4和Q/V/PPO均保留；不能将候选目录当作已部署版本。

[真实B4计时](results_native_root_tape_20261004.json)绑定1f2751c，
PID2131083/创建1791122422.44，已完成退出。固定原88行捕获bank、仅消费真实
第40–43行，长度7410–7586；复用前热调用3.698/3.702秒，复用后2.991/2.981秒，
减少19.12%/19.48%。这是在前缀复用之上的单B4改善，不是完整88行或整轮提速。
每rank32层capture均被消费，重放0次，无新增快照copy；Torch峰值34.54/36.20GiB。
[32k容量](results_native_root_tape_capacity_20261004.json)绑定e326ec9，PID2225427/
创建1791123285.99，已完成退出。仅原容量夹具，实际每行32768、每卡B4、LoRA8/16，
热调用7.733/7.764→6.496/6.478秒，Torch峰值58.63GiB，无OOM；不能代替真实环境、
VERL算法对照、actor更新或与vLLM共存的验收。旧raw的physical_free_bytes来自
cuda.mem_get_info，不作为mx-smi物理峰值。dee3dac的metadata JSON失败未进入DT；
e326ec9仅修标量Tensor日志序列化。原失败和原始结果保留。

[真实GDN0](results_native_root_tape_gdn0_20261004.json)绑定a2dbbc5，PID2307695/
创建1791124027.98，已完成退出。两rank×三variant共六次使用原FLA0.4.1的
recurrent参考和原o/ht assert_close(...,0.005)，CI豁免关闭。原测试SHA为
35f28bf6d01f101f075309133929d1764ab540eb9a892f35eca92227e8768813。
实际Q/K/V/beta为FP16、g为FP32，非空initial_state为BF16，native_ht为FP32；
保留实际状态、scale与一次归一化。
[保存操作数对照](results_native_root_tape_gdn0_operands_20261004.json)显示全部字段
OFF/ON逐值相同。这只证明实际GDN0前向/状态范围，不是整条DT容差。

[真实FA3完整操作数](results_native_root_tape_fa3_20261004.json)绑定48da1c3，
PID2439735/创建1791125232.50，已完成退出。完整Q为B8×576/610×16×256，
完整K/V为B8×7552/7586×4×256，全为实际BF16；没有query/KV截窗。
原FA2.6.3参考与断言源SHA为a290e11cbcb2e65fe7b8399d42eae3bb5c4113bbc12e6190cd7f710ad70abca9。
六次均通过原max_error<=2*普通低精度max_error断言；实际0.03125，原基线
0.09375/0.1015625。同rank三variant的完整操作数快照SHA相同，没有额外模型前向。
插桩复制与参考计算不能当成性能计时；19139d5的远端Python路径引号错误发生在
初始化前，48da1c3仅修日志诊断启动命令。没有数值纠偏或改容差。

用户要求来源：真实检查不能由夹具或接口spy替代；容差使用对应owner的原测试。
VERL固定20bd331：tests/kernels/test_linear_cross_entropy.py原head前向atol/rtol
1e-4/1e-4，反向1e-2/1e-4；适用head输出/梯度，最新原回执为
[results_actor_b8](results_actor_b8.json)。实际dtype补充比较logprob、entropy和hidden
梯度，未声称BF16 weight梯度已覆盖。tests/models/test_transformer.py的1e-2/1e-5
仅masked-mean logprob；test_transformers_ulysses.py的同类梯度阈值仅原SP/non-SP
q_proj测试。不得把这些局部判据、FA或FLA阈值扩为整条PPO统一容差。原core_algos
和update_policy没有被本次候选替换；具体源SHA/适用范围见
[需求与测量索引](results_native_root_tape_assessment_20261004.json)。

最新只读现场绑定phase-observation-20261004/prefix-context-1791125306.json：SQL已退出，
AppWorld已退出并保留完整checkpoint19；TextCraft同PID3218909/创建1790895651.16
在GPU4/5继续，已完成step100。GPU0/1保持空置，2/3仅上述短诊断；不据进程存活
声称训练效果或整体效率合格。既有current_runtime.json保留原采集时间，不覆盖历史。


### 2026-10-04：跨轮复用的实际大头与窄接口修复

[热B4阶段回执](results_native_prefix_warm_phase_20261004.json)绑定78bc272，
原88行bank仅观察最大残差所在原B4第40–43行。rank0原热前缀3.002秒，
复用缓存载入0.100秒；后缀root0.894、原layer重放1.124、有限decoder1.261秒。
这三项仍未缩短，不能把完整88行19.29%的改善称为主要浪费已经处理完。
完整原始结果与token向量留在远端并保留SHA；本机索引去掉重复的完整结果和
minimum_log_ratio_batch token数组，不改变原证据。

[最大残差实际操作数回执](results_native_prefix_peak_components_20261004.json)
绑定ea2d32a、PID819771/创建1791110343.81，已完成退出。真实GDN1两端q/k/v/beta
FP16、g/state FP32，原FLA o/ht断言通过且CI豁免关闭；rank0最早变化实际在FA3，
仍需该层原FA参考对照。它不是整条DT或跨轮复用的数值接受，候选未部署。
此前e881105诊断漏传原cache观察callback的失败回执保留，未改数值核心。

[FA3实际BF16原容差回执](results_native_prefix_peak_fa3_20261004.json)绑定206fece，
PID962286/创建1791111636.14，已完成退出释放2/3。仅执行两次原native forward，
选中已观察到的FA3；实际Q为B4×64×16×256，K/V为B4×6976×4×256。
两rank长/短两端最大输出误差均0.015625，原普通低精度基线误差0.0625或0.140625；
原FA `max_error <= 2*baseline_max_error`断言通过，参考、断言、dtype及原操作数
SHA均保留。不是对整条DT定义额外容差，也未加入缩放或裁剪。

已修的另一薄接口是`_Qwen35CausalOwnerView.synchronize_prefix_start`：保留原
FSDP mesh和MIN collective，只同步是否进入非零前缀分支，保留各rank本地长度。
旧接线同步了长度本身；保存的真实88行使消费者额外处理28672个paired后缀槽位。
13项CPU owner-helper接口对照通过；尚未部署或实际双卡验收，不把槽位计数当提速。
局部B4共同前缀/宽度仍受现有dense finite ABI约束，未另造packed实现或减小B4。

该薄接口随后在[原真实双卡B4热profile](results_native_prefix_hot_profile_20261004.json)
完成，诊断a6b042a、PID1009824/创建1791112064.32，已退出。原冻结owner只替换该
方法行区间，其余bytes保留；实际两rank导入SHA为b603a5a06245…，GPU2/3，LoRA8/16、
每卡4、32768和原Q/V/PPO不变，未部署到正式worker。19项CPU原载体/方法测试通过。
本次仍是原最大残差B4，完整bank捕获5个B4；不是完整88行重复测试。
native原DT调用及有限检查完成，Q逐值相同，shared_warm A/V最大残差分别0.009997/
0.004774；只作定位，不给整条DT自创门槛。插桩包含profiler记录和86MB trace导出，
原25秒/shared44秒总墙钟不作训练速度比较；原runner的热阶段计时另存。

[原设备事件拆分](results_native_hot_device_breakdown_20261004.json)只读两份已保存
trace，按原External id关联原CPU launch/层范围，GPU事件逐条只计一次。shared B4
两rank HtoD2142次，实测20.631/20.781GB，设备耗时0.575/0.589秒；root checkpoint
DtoH约1.25/1.32GB、0.026/0.031秒。all-gather均67次、0.686/0.230秒；
root矩阵投影0.338/0.308、replay矩阵投影0.280/0.304秒。多stream设备时间不能
相加冒充墙钟；outside_native只表示不在层range内，仍包含FSDP prehook和DT阶段。
大部分HtoD匹配原CPUOffloadPolicy参数搬运，未发现另一套整模型offloader。
实际Torch2.8在显式reshard_after_forward=True时会释放root；不能照旧注释声称
跳过手动root.reshard即可消掉第二次gather。原seed需要head/finalnorm物化。
重复恢复checkpoint约占本例HtoD的6%，不能把它单独当成主要提速修复。

后缀root与逐层replay仍重复原投影。PyTorch2.8公开SAC与PEFT0.18.1实际源码已保存
原路径/SHA供接口审查，尚未接入SAC或更改offload策略。只有同次原MLP投影的
重算时间是现有profile支持的可节省上限之一，不能把全部1.12秒replay都算成可省。
批内共同宽度、通信和有限传播的主要成本仍未全部解决。

[同次原root/replay投影输入](results_native_projection_inputs_20261004.json)绑定34ef103、
PID1296311/创建1791114730.04，已完成退出。原实际B4/完整88行bank/官方模型与配置不变，
两rank各96个原MLP base Linear输入逐值相同，shape/stride/dtype/device及权重元信息
也相同。观察hook不替换输出、不添加forward，异常/结束移除hook和CPU快照；本次
拷贝比较的约5.4秒开销不作为DT速度。它证明该样本中重复投影的操作数一致，
不是已接入SAC，也不扩大为所有训练轨迹保证。原SAC存储在GPU且不受参数CPU
卸载自动管理；现有32k夹具实际suffix1024时全32层三项base输出驻留14GiB/卡。
原FSDP前向预取候选随后完成，见
[原预取回执](results_native_reverse_prefetch_20261004.json)：诊断d13e67b、PID1395619/
创建1791115628.55，已完成退出。仅原public forward-prefetch setter在reverse replay
期间生效，两rank各32层原调用、31次next gather已消费，设置和方法恢复、无待清理
handle。实际热B4 attribute为3.69775→3.39947、3.70218→3.46263秒（8.07%/6.47%），
不是完整88行或训练提速。rank0 replay1.14353→0.72697，finite1.25530→1.37334秒，
带宽争用抵消部分收益；原root不变。现有next-owner buffer最大438191488字节，
不是物理显存峰值。两variant仍分别重新捕获prefix bank，root分数在预取启用之前
已变化，因此A/V残差0.002050/0.001189仅作观察，不单独归因为预取、也不设置
整条DT新容差。未部署，不改变原参数卸载、FA/FLA、Q/V/PPO或训练配置。

AppWorld失败重试记录的9行接口修复只在原world.restart成功后清除已被作者丢弃
episode的采样记录；原retry、执行、奖励、消息和token断言不变。原owner AST对照
7项通过，峰RSS51.5MB、GPU0，原Qwen tokenizer入口2项结果复用。
代码SHA6ad5a3e383d032fdc7f0dd2dab1ee027d8f0ed182f9725d07a3faedb46e54ec9；
尚未部署/恢复。现场日志确有原retry，原失败行未保存，不声称逐token现场对拍。

[实际有限线性工作量分解](results_native_linear_work_20261004.json)只读原热trace，
两rank原3953个BMM已闭合：248次base、248次dense LoRA、3456次原对称FLA、
1次首decoder之前的BMM。原对称FLA两方向不是编译重复，设备合计仅0.08938/
0.08417秒。dense LoRA物化0.01322/0.01315、全宽应用0.13104/0.15716秒，
确有rank8仍按full-width应用的冗余，但这些设备时间不是墙钟收益。
原FSDP元信息明确A/B存储FP32、gathered compute BF16；不把初始化dtype当实际算子输入。

隔离低秩owner候选aa008703…仅替换c9 decoder的两个线性map函数，读取原PEFT
B/A/scaling、复用原_mm；未改默认DT或PEFT forward。实际PEFT CPU FP32合同6项
通过，GPU可见置空，峰child RSS900030464字节，测试总40.59秒。回执在
`research/temporary/rl_upstream_alignment_20260929/native-lora-factorization-owner-20261004/`；
不作GPU BF16/FA/FLA通过或真实提速声明。当前候选改变adapter中间舍入位置，
实际dtype对照尚待完成，禁止增加倍率/裁剪来通过。

Qwen3历史root tape有零decoder replay，但Qwen3.5现有加速实现仍重放所有层。
更大复用方向须在原runner每层启闭现有capture、消费其原artifact，不另造算子或
卸载器。当前热B4仅五项decoder captures逻辑payload已12.375/13.105GiB，
尚未加GDN/FA和prefix cache；没有证据声称32k全GPU tape可装下或CPU卸载更快。
这一执行路径未实现/部署，方法与训练参数不变。

[首次root捕获存储回执](results_native_root_capture_inventory_20261004.json)绑定隔离诊断
31f16be、PID1883358/创建1791120151.55，已退出释放2/3；不是正式部署。
沿用真实B4第40–43行、完整88行bank、LoRA8/16、每卡4、32768上限。
复用原capture APIs，每层root进入/退出后读取真实metadata并释放，两rank各32层
完整、calls齐全、metadata错误0、无新增forward、无遗留hook；原GDN cut=0。
实际每层unique storage求和分别28.550/30.222GiB，其中decoder12.375/13.105、
GDN13.136/13.999、FA3.039/3.118GiB。它不是整网同时驻留峰值、新增显存量、
32k容量或提速证明；原checkpoint/cache/参数及有限传播工作集还需考虑。

[测量边界与修正](results_native_root_capture_inventory_assessment_20261004.json)明确保留
已执行缺陷：旧factory在每层退出调用原tensors(cache)，该诊断callback会detach().cpu()，
导致反复CPU快照；root约3.94秒因此不作速度结论，CPU快照也不能分类GPU storage alias。
保存的原GPU字段shape/dtype/存储量仍是真实观察。准备源已移除该可选callback，
外部cache ownership留为未分类，不另造缓存遍历或修改原callback；修正未重跑GPU。
正式源与所有数值公式不变。原47项远端CPU测试通过，含真实Torch decoder hook顺序、
原HF Cache多次消费隔离；最新27项本机源/结构测试通过，均不冒作FA/FLA验收。

同一诊断的shared_factory已只捕获一次原bank，之后复用原immutable lease，
每次仍由原compose接口创建fresh DynamicCache；没有第二份cache实现或正式开关。
后续须用原artifact接入有限消费者，保留FSDP prepare/release与原FA/FLA；
不能把全部replay耗时、诊断CPU快照耗时或这次存储总量当作可实现的提速比。

AppWorld原step19恢复候选已在远端
`candidates/appworld-native-async-retry-recording-resume-20261004`准备，尚未提交。
保留原async桥、actor1f862e8bbdaa…、任务配置与检查点，唯一环境记录改动为上述
成功restart后的9行清理；复用已接受host-cache worker e5eb4afc42f1…及原Ray env
资源开关。新冻结副本7项原retry CPU对照通过；不把准备当恢复，不部署其他候选。

### 2026-10-04：共同右侧padding V4完成；跨轮前缀复用继续隔离验证

[padding完整回执](results_actor_shared_right_padding_20261004.json)绑定诊断提交
2b0a05b、原actor SHA1f862e8bbdaa…和候选SHA3a65e173300b…；两者使用同一份
实际token。原VERL `test_hf_casual_models` 的masked-mean断言（atol=0.01、
rtol=1e-5）通过，未改断言。热前向14.643→6.188秒仅是该padding测例，
不是整轮提速。有效token最大log-prob残差0.120556单独记录，不引入自定整网容差。

双卡实际B8×32768、每卡B4、LoRA8/16、原AppWorld两epoch更新已完成，
占满32768的容量更新134.131秒；原异步vLLM LoRA同步通过。75次物理采样中
诊断GPU2/3最大各51350MiB，作业最大PSS69344674816字节。进程PID52877
（创建1791103305.18）已退出并释放2/3。候选尚未部署，未改变正式actor。
V2/V3初始化失败回执继续保留，不能替代V4完成回执。

本次只核查既有任务；新基准只作检索，不接入或启动。用户要求物理GPU0/1空置，
实验仅2–5；TextCraft仍在4/5，SQL/AppWorld退出状态未因诊断而变化。
保存的真实AppWorld请求每rank88条、18条轨迹，同UID各prompt逐值嵌套。
此前32条测试跨轮复用热耗时比原路径慢3.7%–3.9%，不部署该结果。

后续隔离诊断沿用已完成前缀测例的冻结DT/VERL、实际IDs和官方Cache/FLA，
使用完整88条请求比较缓存准备、搬运及原归因阶段成本；不重新采样任务。
新候选只用原DataProto.reorder按所需前缀长度安排捕获B4，保持归因消费者
顺序、端点、query、目标与Q/V/A不变。运行前CPU接口检查与原始请求逐值核对；
不放宽FA/FLA容差、不做信用纠偏，未完成回执不能称为提速或接受部署。

完整88请求比较已完成，见[实际输入、源码与数值回执](results_native_prefix_reuse_workload_20261004.json)。
提交712795d，隔离PID329336、创建1791105853.62，已退出释放2/3；CPU6项通过。
两rank原热调用196.657/196.648秒，复用热调用158.723/158.747秒，完整成本
减少19.289%/19.274%。捕获5个B4用时23.964/24.002秒，前缀槽位606208→
190208/188672；槽位减少不是提速比。端点、query、target和实际回报逐值核对，
原消费者顺序与每token Q/V组合未变。Q估计逐值相同；优势最大绝对残差
0.071222/0.060951，原冷/热最大残差0.004647/0.003597。未加裁剪、重标定或
自定整网阈值，候选仍未部署；不能把运行完成称为新的整条DT数值接受。

诊断记录器误读`seconds`而非原runner的`stream_elapsed_seconds`，该回执中
阶段汇总的0是缺失计时，不是阶段耗时0；原完整墙钟和调用数有效，原错误记录保留。
仅补测最大残差所在B4（两次原归因、同一完整捕获bank），修正计时字段以定位
阶段与残差，避免再跑88请求四遍。官方数值核心、FA/FLA容差与PPO未改。

补测已完成，[原阶段计时与最大残差位置](results_native_prefix_reuse_phase_20261004.json)
绑定2668489、隔离PID485725及创建1791107307.5，退出后2/3已释放。
最大残差为rank0第42行第0个action token，对应原B4第40–43行；两rank各执行
一次原DT和一次复用DT，保留完整88行捕获bank，没有重复整批比较。
原`stream_elapsed_seconds`字段读取完成；候选该B4缓存载入0.098/0.102秒，
有限decoder合计1.255/1.437秒，原层replay合计1.126/0.933秒。捕获bank另计
26.884/26.882秒。首个原DT是冷调用，不能将其37.70秒与后续3.70秒比作热提速。
该B4优势最大残差0.073683/0.004413，rank0尖峰重现且Q估计相同；并未通过
原FA/FLA容差判据宣布整条信用已验收，也没有加入任何纠偏。后续数值定位保留
这些实际dtype/输入的原来源，不借用无关整网阈值。跨轮复用候选仍隔离，未部署。

### 2026-10-04 16:23：当前作业状态复核；新padding候选未部署

[原日志、PID与物理资源回执](../../research/temporary/rl_upstream_alignment_20260929/phase-observation-20261004/status.json)
采集于1791102216附近。TextCraft原PID3218909及创建时间仍匹配，完成step93/330，
正在下一次原生成；原完成检查点标记75。最新整轮2409.638秒，其中采样1496.832秒、
DT45.989秒、actor570.332秒；训练批奖励均值0.031，不能据进程存活宣称效果健康。
当时仅GPU4/5在用，物理显存50052/50482MiB，其余六卡无任务进程。

AppWorld原PID3232113已退出，最新完整step19及检查点19；第20次采样后的
`loop_owner_rollout.py::to_batch`在原PolicyTokenInfo与response位置比较处断言失败。
原冻结入口SHA c4a46492a665…，未绕过断言、改奖励或提交重启。SQL仍是此前
global OOM退出的PID552842，完成11且无可恢复检查点；没有重启。此前21:29源码
快照保留原采集时间，不能作为本次存活证据。

共同右侧padding候选actor SHA3a65e173300b…仍未部署。CPU98项通过只覆盖载体与
mask接口；真实模型测例V2在旧同步sharding入口失败，V3改为原异步worker后又因
未使用原colocated WorkerDict注册而在服务初始化失败，两次均未进入数值比较。
V3 PID3905745已退出。原failed prepared/result/log与资源观察保留；不标成数值或
容量通过。后续诊断复用原VERL `create_colocated_worker_cls`及`spawn`，不修改
正式算法或异步executor。V3额外CPU工厂检查缺少诊断环境变量的失败也保持记录。

### 2026-10-03 06:30：实际Actor输入工作量与第9层FLA对照完成，未部署复用候选

[完成的输入工作量回执](results_native_actor_workload_completed_20261003.json)绑定原AppWorld
worker3250426/3254727和TextCraft worker3236008/3240133，actor SHA均为1f862e8bbdaa…。
两组各rank记录8次原calculate_entropy=True、每卡B4的forward，随后恢复原绑定。
AppWorld共同右侧padding占输入槽位27.894%/29.372%；TextCraft为39.541%/31.097%。
这是实际mask统计，不是计时或提速比；已有输出头裁剪未去掉骨干网络的共同右侧padding。
记录器未改模型、batch、参数或更新。早先0样本回执保持原采集时间，不再代表当前结果。

[实际前缀逐层诊断](results_native_lease_components_20261003.json)和
[第9层实际dtype对照](results_native_lease_layer9_20261003.json)保留原请求、源码SHA、
准备记录及退出PID。诊断提交b49bf89→80c1054→fbb8f68，只改变观察范围，没有改正式DT。
原B4短前缀2752与实际lease长输入6144/5696使用相同原token；末64位置与完整前缀
的诊断范围分别记录，不能把末64逐值一致扩大成整层逐值一致。扩展观察后最早可见变化
在第8层GDN输出投影，最大BF16差0.001953125；第9层缓存首次出现可见变化。
第9层实际操作数为q/k/v/beta FP16、g FP32，原输出FP16、状态FP32、HF边界BF16。
两端原FLA输出/状态断言均通过；原输出误差比约0.000381、状态约0.000325，
使用固定原参考和断言，未设置整网容差、放宽门槛或加纠偏。
这些是算子范围的证据，不是完整DT信用估计质量或速度接受。8e7dd71复用候选仍未部署；
正式DT核心c9cd147/fc2e6c2、LoRA8/16、每卡B4、32768和任务/PPO配置保持不变。

1790980223再次读取原TaskRunner日志：AppWorld最新完整仍为step13/检查点13，
TextCraft完成step49。TextCraft step49采样30.881分钟、DT7.744分钟、Actor9.648分钟、
旧概率及reference共4.971分钟、整轮53.253分钟；没有把缺失的当前检查点标记补成49。
AppWorld step13仍为采样26.034分钟、DT75.580分钟、Actor45.698分钟、整轮154.398分钟。
采样与完整迭代须分别比较。LOOP原文42小时使用8张采样H100另加8张学习H100，
没有原文单轮采样半小时的阶段记录；同为4卡时不能证明不同机型/模型/任务负载效率对齐。

### 2026-10-03 05:28：原请求记录和有界DT计时完成；复用候选未接受

[原生请求及有界比较回执](results_native_prefix_dt_leases_20261003.json)保留诊断的失败/完成源码、PID、输入哈希和原结果。
9959b70记录的原训练输入已保存并恢复原绑定：每rank88个原始请求、18条轨迹，
IDs/masks/实际回报直接来自原owner，没有另跑环境或decode/encode重建。
旧716-byte失败文件继续保留，未作为比较输入。

有界比较使用其中32个原请求（每rank8次原B4），全成本计时包括原前缀捕获与搬运。
两rank原路径热调用61.465/61.472秒，复用热调用63.846/63.761秒；候选慢约3.7%–3.9%，
不称提速，不接入正式任务。捕获5个B4用时16.374/16.277秒，抵消消费者省下的时间。
完整88请求的前缀重复较多，但输入槽位节省不是实测提速；未按它推算正式时长。

原冷/热路径最大优势残差为0.005644/0.003729，复用热/原热为0.025931/0.027236；
Q估计逐值相同。这里只记录残差，没有借用整网2倍规则作验收，没有加纠偏/裁剪。
需按实际dtype和对应FA/FLA原参考定位变化来源；新候选尚不具备官方数值接受证据。
初次入口错误来自诊断构造器无条件设无效动作惩罚；e62eae3按原VERL worker条件读取，
实际启动前AST对照通过。原训练构造器、DT核心、Q/V/A和PPO未改。

57f58a8通过原worker RPC只记录下一阶段8个原Actor B4的mask/shape，随后恢复原绑定。
[只读观察回执](results_native_actor_workload_20261003.json)截至1790976517仍为0个样本，
不把未采集数据填成padding为0。记录器没有新增模型计算、GPU计时或训练行为。
此时原AppWorld/TextCraft driver创建时间仍匹配，原检查点标记分别13/48；SQL仍已退出。

官方比较口径：LOOP附录D公开42小时是8张采样H100加8张学习H100的完整训练，
未给出单轮采样半小时记录（https://arxiv.org/html/2502.01600v1#A4）。
AppWorld step13的原记录采样26.03分钟、DT75.58分钟、Actor45.70分钟、整轮154.40分钟。
两小时整轮不能作为两小时采样；GPU数量、卡时与不同模型的工作量不能互相代替。
原基线仍是c9cd147/fc2e6c2及已完成原worker覆盖；LoRA8/16、每卡B4、32768未改。

### 2026-10-03 04:54：原训练FP32续算已完成；真实请求记录仅修保存接口

[完成的原训练状态回执](results_native_prefix_streaming_state_completed_20261003.json)
绑定原AppWorld PID3232113、两rank worker3250426/3254727和c9cd147源码SHA。
真实B4×7168前缀按1792-token区间续算，四个FP32边界状态在两rank均与原生
独立读出逐值相同，末边界也与原forward的状态相同。观察已恢复原绑定；未替换
模型或信用计算。前两段含首次shape编译，后两段约0.9毫秒，不冒充完整DT提速。

43f5699的一次性原始行保存发生官方torch.save错误：不同dtype的Tensor共享
storage。旧提交、rank错误记录和716-byte未完成文件保留，不能作为有效输入使用。
9959b70仅用原生torch.utils._pytree.tree_map及detach/cpu/clone保存独立CPU
Tensor，原训练行与结果原封不动返回。其源码SHA为48d154a5aa2b0f586fc9d1e7b3dd40394a50aedb41dfedda227ef73edeef0061。
[原生CPU序列化回执](results_native_request_artifact_serialization_20261003.json)
复现同一官方错误，修复后IDs/mask/response的value、dtype、shape逐值相同，
原对象未改动。它是诊断记录的接口修复，不是DT数值或容量验收。

修复后的观察1790974394已于1790974418提交到原worker RPC边界；截至该提交仅排队，
没有宣称安装或新输入保存完成。有界新lease计时入口尚未启动，候选未部署。
原任务最后定向观察为AppWorld完成13（gen26.03分钟、DT75.58分钟、actor45.70分钟，
整轮154.40分钟），TextCraft完成48；源日志中两小时口径不是采样时间。
LOOP附录D公开42小时是8张采样H100与另8张学习H100的完整实验，未提供采样半小时
的阶段计时。不能仅按GPU卡时或参数量宣称对齐。

### 2026-10-03 04:30：有序请求的原生前缀复用接口已准备，尚未部署

[候选源码映射](prepared_native_prefix_leases_20261003.json)绑定8e7dd71实现、7d669de
CPU检查入口和ecb0c96有界计时入口。原增量artifact类从AUDIT移到Qwen DT owner
模块；本地AUDIT只重新导出同一实现，不保留第二套缓存。移动前后类和函数AST相同。
owner仅新增默认关闭的provider前缀长度上限；不传provider时，完整方法与0936d34
经AST对比一致。新接口未写入默认profile、launch或正式worker。

- 按原排序的B4消费者准备前缀lease，不改变原采样、分发、请求顺序或token信用。
  只捕获原prompt；不捕获当前动作、查询和目标。每条已选真实历史最多一次原模型
  捕获；FLA FP32边界状态按新增区间续算，HF公共update方法构建新Cache。
- 两卡捕获数用原FSDP mesh的原生MAX collective对齐，复制行由原VERL
  pad_dataproto_to_divisor处理；没有rank本地miss-forward、新调度器或额外参考采样。
  每次归因RPC的lease在调用结束释放，不跨优化器更新保留。
- [CPU原回执](results_native_prefix_lease_interfaces_20261003.json)：现有远端Python中
  40项通过、2项本机Git源码比较已单独通过。检查包括原端点/mask/QVA保持、原生
  padding、跨rank捕获次数和精确事实ID拒绝；不是模型、32k容量或速度验收。
- ecb0c96计时入口继续用原VERL双卡worker及固定配置，在原请求排序中各取至多
  前8个B4消费者；比较原冷/热与共享冷/热，完整计算捕获、读出、搬运、模型及日志
  成本。仅准备，尚未启动；没有新增整网数值阈值或改变FA/FLA断言。

43f5699的一次性原_prepare_episode观察已在3250426/3254727安装，保存原行、完整
return和请求张量后恢复原方法。1790973014.10仍未得到下一批请求；AppWorld
PID3232113创建时间1790924078.76及两worker创建时间均已再次核对存活。
原日志已完成step13，gen1562.061、DT4534.808、actor2741.896、整轮9263.866秒；
TextCraft原日志已完成47，当前原DT；SQL仍退出，未重启。阶段原输出保存在
[定向观察](../../research/temporary/rl_upstream_alignment_20260929/phase-observation-20261003/native-prefix-leases-current.txt)。
这不是重新采集完整源码快照；current_runtime.json仍为21:29的原时间。

### 2026-10-03 03:47：原缓存输入接口完成，增量读出仍为待验证候选

[原DT接口回执](results_native_prefix_dt_seam_20261003.json)对应76ba200，独立诊断
目录native-prefix-dt-seam-20261003；driver2196941创建于1790967328.73，已完成并退出。
在原VERL双卡actor上使用四条真实端点，每卡B4、LoRA8/16，总长3800–4666，
共同前缀2752；不是正式检查点恢复、32k容量或整轮训练速度测试。

- 默认不传provider时的完整owner方法经AST对比与0936d34相同；本机1项通过、
  4项因无torch跳过，现有远端环境4项CPU接口检查通过。provider只传原生Cache，
  原HF reorder/deepcopy和全部有限传播保持。正式作业没有部署此provider。
- runner SHA b5a2c36630234a1067a5848136a0a94f81262882c5a3e53f2675374e4bec601b，
  credit接口SHA 8046762ae2fd149b299e29f9a331d8ae1aed665f2de895573e78e61ebc2d8497；
  冻结诊断的artifact仍为2979851e4b55…，不能拿它验证后面的增量版。
- 两rank的64个Cache字段经原HF公共update接口重组，与同次捕获和独立原前缀
  均64/64逐值相同。原热DT11.063/11.066秒；缓存输入的两次调用约9.88–9.92秒。
  一次捕获另耗1.738/1.690秒，不能从排除捕获的单次降幅推算正式整轮提速。
- 原冷/热重复的signed最大差0.003436/0.001478；provider相对热原版最大差
  0.002404/0.002182，均保留原始向量和逐项残差。这不是新的整网FA/FLA容差，
  没有倍率、裁剪或归因纠偏，也不宣称缓存字段相同能证明全部DT数值验收。

[独立增量接口回执](results_native_state_streaming_interface_20261003.json)对应f0f780f，
只用原FLA test_chunk输入夹具、实际FP16 k/w/u及FP32 g、B4×7488。
原FP32状态在1856/3712/5568/7488四个边界均与独立原读出逐值相同，完整边界
也与原forward的final_state逐值相同；增量总共读取7488 token，不是多次从0读。
这不替代正式模型样本或全层Cache验证。首段3.918秒含新形状首次编译，后续段
约1.0毫秒；原前向13.890秒亦含首编译，不能当完整DT或正式热吞吐。
远端目录native-state-interface-1790970639，PID2553911创建于1790970629.91，已结束；
operator SHA e4a81e5f6999…，测试原文件SHA 35f28bf6d01f…，峰值torch3.595GiB，
PSS6.677GiB，未创建或替换模型、内核、训练参数、容差或正式方法。

6ef58eb的增量artifact SHA cb807d533d1404f2f339ef3f9b40d88a322a094da9020362666c9dd2a05364fd
只组合原FLA FP32 initial_state/final_state接口，相邻边界只读新增区间；仍未部署，
且不被上面的旧artifact回执覆盖。[首次提交观察回执](results_native_prefix_streaming_state_20261003.json)
为原正式worker的一次性观察，03:47仍未读出，不能称验证完成。安装在3250426/3254727，
原runner SHA c7fc969f9f52…与Q/V/A未变；观察触发后恢复原绑定。

[实际阶段原输出](../../research/temporary/rl_upstream_alignment_20260929/phase-observation-20261003/native-streaming-worker-phase.txt)
在1790970423.75确认AppWorld两worker、TextCraft两worker均处于原VERL update_actor
的backward；未触发读出与该阶段一致，不等待整轮后猜原因。AppWorld原完整指标
仍step12，TextCraft已完成step46：gen2026.501、DT433.084、actor579.772、整轮3339.239秒。
SQL仍退出，未重启。上述为定向原阶段记录，不替换21:29完整源码快照时间。

### 2026-10-03：实际dtype残差定位与对应官方算子断言

[FLA原始回执](results_native_prefix_components_20261003.json)对应诊断提交5836929；
[FA原始回执](results_native_prefix_attention_20261003.json)对应c2d216d。两次均为
空闲GPU6/7上原VERL双卡actor、每卡B4、LoRA8/16的独立诊断，实际输入4224、
取2752 token前缀；不是32k容量、完整DT、梯度或部署提速验收。未更改正式训练。
冻结目录分别为receipts/owner-b8-dispatch-20260930/native-prefix-components-20261003
和native-prefix-attention-20261003；driver1914366/2005721创建时间分别
1790964712.9/1790965560.08，原rank结果均已完成，后续收集时driver已退出。
每个原JSON的路径、SHA、实际import、配置、dtype及断言保留在回执中。

- 首3层长/短前缀输出逐值相同；首个差异在层3的原BF16 K/V投影、进入FA之前，
  最大绝对差0.0078125。它是原生形状相关低精度差异的定位，不是DT缓存算法错误。
- 首个FLA的长前缀读出/直接短前向、两rank共8个原forward断言均通过：输出误差比
  约0.000346、FP32状态约0.000435，直接执行固定FLA原assert_close阈值0.005；
  FLA_CI_ENV实际为false。实际q/k/v/beta为FP16、g为FP32、返回state为FP32，
  HF原Cache存储BF16。跨dtype描述性残差没有被当作新Cache容差或纠偏依据。
- 首个FA的两个布局、两rank共4个原输出断言均通过：actual BF16 q/k/v，
  相对原FP32参考最大误差0.03125；原低精度基线0.09375，原2倍界限0.1875。
  精确提取并执行固定FA的原attention_ref与原assert，没有定义梯度/整网阈值。
- 原DT c9cd147/参考fc2e6c2、runner SHA c7fc969f9f52…与Q/V/A未改；候选仍
  prepared-only，没有强制守恒、倍率补偿或数值裁剪。通过首个算子不能扩大为
  全模型短缓存或正式DT前缀复用已验收。完整源码快照仍为21:29原采集时间。

[完成阶段计时](results_formal_phase_cost_step12_20261003.json)保存原正式TaskRunner
step12/45的完整行与来源SHA。AppWorld step12：采样1335.606秒（22.26分钟）、
DT4563.260（76.05分钟）、actor2712.531（45.21分钟）、旧概率359.767，
整轮9025.486秒（150.42分钟）；没有本轮独立评估。DT/actor/采样约占51%/30%/15%。
TextCraft step45整轮3132.276秒，gen1794.662、DT460.999、actor577.717。
不同轮次的实际token/奖励/长度不同；这些计时不证明未部署候选提速。
LOOP原论文的42小时是两台各8张H100的完整训练；约28分钟是整轮平均估算，
没有官方“8张H100采样半小时”的分项计时。不能与我们的整轮2.5小时混用，
也不能只按参数量、卡数就宣布与官方效率相当。SQL仍停止，未提交重启。

### 2026-10-03 01:37：原Cache表示对照完成，短前缀容差尚未验收

[原接口回执](results_native_prefix_artifact_api_20261003.json)对应已推送42e2908，
只在空闲GPU6/7运行原VERL双卡actor，每卡B4、LoRA8/16；没有PPO更新、vLLM实例
或正式DT加速部署。冻结目录为receipts/owner-b8-dispatch-20260930/
native-prefix-artifacts-20261003-v2，driver1648447创建于1790962243.69，完成结果
已存在，01:37定向观察PID已退出。entry/VERL沿用AppWorld原路径；诊断VERL冻结副本
fsdp_workers.py SHA e5eb4afc42f1…，actor SHA 1f862e8bbdaa…，数值DT仍c9cd147/fc2e6c2。
候选native_prefix_artifacts_candidate.py SHA 2979851e4b55…、验证入口SHA 8a0a234d2cef…；
完整路径与SHA均在原回执中。配置复制原launch options，actor optim总步数200。

原B4×4224前向的64个Cache字段经HF原update_conv_state / update_recurrent_state /
update重组后，两rank均64/64逐值相同；实际Cache存储BF16，原FLA返回状态FP32，
保留HF原dtype转换。冷捕获18.776/18.702秒包含首次编译、模型前向和CPU拷贝，
不能称热DT耗时；峰值torch allocated约7.64GiB，不是32k完整训练容量证据。

2752 token短前缀与直接原短前向仅6/64字段逐值相同；最早缓存差异在FA层3，
keys最大绝对差0.03125、values为0.0078125。这里没有自定容差，不把跨长度普通
浮点差异直接判为bug或通过；下一步须定位实际dtype与残差来源，使用对应官方
参考和断言。当前候选仅prepared-only，未接入原DT默认路径，不增加倍率、裁剪
或信用纠偏。完整源码快照仍为21:29采集，本项不替换其时间。

第一诊断尝试driver1597493在捕获前因无条件手动搬运参数违反原CPUOffloadPolicy
而退出；原VERL该配置的_is_offload_param=False。42e2908仅为诊断入口恢复原guard，
候选数学源文件未变。首尝试冻结源码、错误日志及回执保留，不能冒充数值失败。


### 2026-10-03 00:52：原FLA状态读出完成，未部署前缀加速

[正式路径回执](results_formal_native_prefix_state_20261003.json)确认c48b8cc诊断已于
00:33在AppWorld原RPC边界挂载，00:34两rank各观察一次原B4、7488 token前缀并
恢复原runner。FLA原`chunk_gated_delta_rule_fwd_h`源码SHA e4a81e5f6999…，完整
前缀FP32返回状态与同次原前向final_state逐值相同。stream读出3.046/3.033毫秒；
首次host等待3.125/3.134秒包含之前异步模型工作，不能把stream时间称正式wall时间。
半前缀原读出2.284/1.974毫秒。原DT runner SHA c7fc969f9f52…、Q/V/A和训练配置未变。

[空闲卡原算子接口回执](results_native_state_interface_20261003.json)对应b6366ab，
复用固定FLA `test_chunk`的输入构造（原文件SHA 35f28bf6d01f…），未加载第二份模型。
B4、7488 token热读出stream3.261毫秒、同步wall3.283毫秒，FP32 final_state逐值相同；
3712 token读出2.133/2.171毫秒，舍入后等于原FP16 chunk边界状态。首次原前向含
编译16.455秒，峰值torch allocated约3.60GiB、进程PSS约6.66GiB，进程已结束。
这是接口身份与成本证据，不能称完整模型/DT精度或速度验收。原回执中的
native_cache_dtype/equal_to_original_cache_state实际指FLA返回final_state，保留原始
回执并另加口径说明；HF Cache的实际写入仍必须走原update_conv_state /
update_recurrent_state，不能绕过原dtype转换直接替换缓存字段。

本项只完成原接口调查；没有部署跨调用Cache、修改递推内核或加入数值纠偏。
00:52物理卡2/3分别42189/42462MiB，4/5各17108MiB，主机MemAvailable约589.7GiB；
AppWorld/TextCraft原进程继续，SQL仍停止。本项不更新21:29完整源码快照时间。

### 2026-10-03：完成阶段最新计时与原统计口径

[原阶段计时](results_formal_phase_cost_20261003.json)来自同一正式PID的原TaskRunner
日志：AppWorld step11采样1312.603秒（21.88分钟）、DT4013.914秒（66.90分钟）、
actor2760.685秒（46.01分钟）、整轮8513.998秒（141.90分钟），此轮没有独立评估。
TextCraft step43采样1923.745、DT503.907、actor579.641、整轮3306.967秒。
各轮原token数和轨迹长度不同，不能把时间下降归于新的未部署加速。LOOP的42小时/
约90次更新只给出整轮平均，不是官方采样计时；不能与我们的两小时整轮混用。

原VERL 20bd331的rollout_probs_diff使用responses区域的attention_mask，包含
工具观测；AppWorld原观测位置rollout log-prob为0。因此该全区域统计不能作为
模型生成action的数值对照结论。保持原统计实现，未添加纠偏或新容差。
原FLA状态接口小测c48b8cc已提交到AppWorld原RPC队列，formal-native-prefix-state-
1790957819；本次采集只见提交回执，不能称实际读出完成。SQL仍停止。

### 2026-10-03：原捕获存储观察完成并恢复

[实际捕获回执](results_dt_native_capture_storage_20261003.json)保存原AppWorld两rank
各一次B4端点配对B8、8232/8237 token调用。共同前缀7488，原完整DT为
7.680/7.702秒；原stream事件前缀3.237/3.238、重放1.422/1.179秒。
32层逐次捕获的存储累计33.665/36.791GiB，全部CUDA，元数据观察各约0.003秒，
两rank均已恢复原方法。该累计值不代表同时存活峰值或真实CPU搬运量，也不能
据此宣称首次前向保留全部中间量会更快。没有部署新Cache或全层保留路径。

安装版FLA的中间h为FP16、返回final_state为FP32；不据此推断HF Cache存储dtype。
继续只核对原chunk_gated_delta_rule_fwd_h的FP32状态读出接口，不复制递推公式，
不改FA/FLA容差或增加归因纠偏。00:09原阶段快照确认AppWorld完成标记11、
第12次DT在进行，TextCraft原日志已完成42；SQL仍停止，没有提交重启。
本项为定向原日志/捕获回执，不替换21:29完整源码快照时间。

### 2026-10-02 22:54：原DT前缀布局与重放存储调查

[原请求布局回执](results_dt_prefix_layout_20261002.json)记录一次正式组的两个rank各52个
请求、6条轨迹；92条相邻历史前缀均逐值相同且继续增长。诊断9598dac仅保存元数据，
约0.96/1.00毫秒，两rank原_prepare_episode均已恢复；DT核心c9cd147/参考fc2e6c2、
原Q/V、每卡4、LoRA8/16不变。没有部署跨调用Cache或新布局。

保持四条轨迹槽位的225种只读组合均使成对后缀槽位至少从340,112增到686,904；
部分前缀少算不能据此保证加速。这里只统计输入槽位，未另设时间/FLOP权重或验收阈值。
原HF混合Cache未提供可直接复用的batch合并接口，GDN状态不能用普通KV裁剪代替。
随后核对原NativeDecoder/FA/GDN捕获接口：首次前向全层保留尚未实现；完整32k成对B8
三份MLP数组的算术上界是576GiB/rank，不能用小后缀的容量替代整条路径。

存储诊断35d4c23已在原AppWorld WorkerDict RPC队列提交，回执目录
formal-dt-capture-storage-1790953114；23:09两rank原RPC均已完成挂载，尚未取得实际捕获或修复。
它只读取一次实际原捕获的shape/dtype/storage bytes，随后恢复，不增加模型调用或张量拷贝。
23:00以前只读阶段回执确认AppWorld第11次DT已结束、原actor更新在进行；完成标记仍10。
TextCraft第41次完整迭代已结束并进入42次采样，41次gen2023.377、DT489.433、actor579.658秒；
这是阶段事实，非本诊断提速。SQL仍停止。本项未替换21:29完整源码快照的采集时间。

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


### 2026-10-05 TextCraft 实际信用反号定位（质量优先）

原停机作业、c9 数值版本、LoRA8/16、每卡actor/DT4及原任务预算均不变。
仅GPU4/5加载原完整checkpoint25，读取step26原IDs；各诊断零optimizer step。
TextCraft未恢复，效率候选仍封存。原成功率首次持续下降窗口仍是35–40；
下述数值证据不能被扩大为已经证明整个退化的唯一原因。

| 诊断 | 身份及实际结果 |
| --- | --- |
| v1 | PID2747303/birth1791206224.41，已结束；原head直接EOS读出及原VERL三项梯度。PG=.00358669，weighted H=.000142814，KL=.0000409251，仅五个独立成功response子集；不代表完整minibatch。 |
| v2 | PID2887764/birth1791207512.79，诊断模块过早import有限目标类导致进入DT前失败；原回执保留，无更新。 |
| v3 | PID2914261/birth1791207731.26，已结束；原B4/paired8、EOS右补702/718、prefix320，三次原trace调用。原joint端点最大差3.24845e-6；16 selected slots中两处joint与single native root反号，一处single finite自身反号。 |
| v4 | PID3002000/birth1791208539.48，已结束；只single-positive一次，原函数及其返回对象不变的被动token_effect观察，98次小标量收缩。rank1 Thought负root-.045309、最终正finite+.004223；decoder3改变+.0282005，GDN2/1/0继续累积。rank0冷/热端点差另列，未代替rank1证据。 |
| v5 | PID3108984/birth1791209508.79，已结束；被动拆查decoder3，原有限算子和参数不变。但rank1 case0 native factual从v4的-.546221变为-.585652，root从-.045309变为-.078381；变化发生在有限包装执行之前，不能把v5层残差混入v4反号路径。v5 signed为-.031702，未证明根因或修复。 |
| v6 | PID3237912/birth1791210683.51，已结束；同进程原/采集norm操作数/原三次同一个single-EOS调用。rank1 case0 native root分别-.07837832/-.07838005/-.07837850；finite分别-.03154174/-.03159654/-.03036623。无观察器仍有.0468–.0480差值；两rank零更新，无采集错误。实际.pt分别93908758/97840918字节，SHA在rankN-norm-operands.json，仅保存第3层case0，不保留全B8张量。 |

所有冻结诊断与原始日志路径/SHA、导入源及effective config在
`research/temporary/rl_upstream_alignment_20260929/textcraft-degradation-20261005/model-probe-v{1,2,3,4}`；
v2/v3/v4/v5的source tar分别保留，当前本机诊断脚本的新版本不替换历史回执。
实际runner仍是原nested c9 `c7fc969f9f521993f2449ea5f364adcb3e0fdac5b01c38c103963639551516c1`，
未调用正式AppWorld的flat prefix候选或封存存储优化。

更正口径：旧实际trainer SHA8816ea4e…通过原轨迹补丁优先loss_mask，
其H统计也屏蔽观测；此前attention-tail解释仅对应未打补丁上游，不对应该作业。
v1 gradient diagnostic用了rawFP64组合再A32，原readout先raw→rowFP32再组合；
CPU调用同一counterfactual owner对八行比较，A最大差1.4901161e-8、V最大差1.1920929e-7，
GPU未初始化。这只限定该诊断精度口径，不设新验收标准或归因退化于该差值。
后续诊断适配已对齐原rowFP32，旧v1源码/结果保留。

完整原readout日志已stream汇总178条report/92条completed metric，logSHA54f4aef6…
匹配原停机记录；缺失不填0、原调整产生的复制槽位不称独立trajectory。
源码全轨迹credit按唯一slice赋值，未见历史动作重复建立PPO loss行；35–40
完整actor数组未保留，不能称真实scatter数值回放完成。当前只继续针对
decoder3原计算拆分MLP/投影、归一化/残差及attention误差；FA/FLA容差、
Q/V/A公式、损失系数及全部训练参数均未改，未称任务质量已恢复。

实际安装HF Qwen3.5 RMSNorm源码SHAf7e1a804…已只读取回：FP32归一化及
乘(1+w)后存回输入dtype，原decoder两次residual相加也存BF16。当前finite norm
是连续FP32解析secant。这只指出需要按实际操作数拆分的误差来源，不构成
FA/FLA超差结论，也不授权守恒倍率或其他纠偏。来源文件及SHA在同目录
`owner-sources/source.json`。同进程原调用/采集norm操作数/原调用诊断v6已完成，
与所有正式训练版本分开；实际.pt留在远端receipt目录，只读CPU拆分已经完成。
两个norm的native/finite输入BF16逐值相同；rank1 post/input连续FP32有限差仅
-6.22e-8/+2.89e-7，FP64约1e-16；保存compiled与原函数CPU eager收缩差
1.41e-8/1.50e-8。主要norm差.003242/.008771来自BF16落盘，GPU/CPU原norm
输出收缩差只4.63e-6/2.10e-5。两次原BF16 residual add在CPU逐值复现，
实际output与两branch和的效应差-.007045/-.003340。对应
`model-probe-v6/rank{0,1}-norm-analysis.json`绑定实际.pt SHA、原函数路径及SHA；
解析norm公式未发现该样本的实现错误，不能为吸收舍入而改正确公式。
原输出conservation_tolerance/verified字段保留作为历史元数据，未使用
它们自建官方门槛或修正信用。TextCraft仍停止。

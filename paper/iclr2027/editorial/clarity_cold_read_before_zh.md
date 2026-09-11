# DeltaTrace 原稿独立冷读：正文理解断点审计

## 审读对象与阅读顺序

本报告只评价本任务开始时取得的论文原稿。主任务将该原稿标识为 `f8a9ddb`；我没有查询 Git 来核验这一标识。原稿 PDF 共 17 页，正文与结论结束于第 8 页，参考文献从第 8 页开始，附录从第 11 页开始。当前原稿满足正文不超过 9 页的要求。

第一遍按摘要、Introduction、Method、Computation、Evaluation、Related Work、Conclusion 的顺序逐句读正文源码及正文表格，实时保留卡点；第二遍按成稿 PDF 第 1–8 页的顺序重新检查段落前提与图表阅读顺序。完成两遍正文后，才读附录及附录表图。第一遍记录没有被后来的理解改写。第一次阅读正文以后出现的图形认识在下文归入第二遍。

读取范围为 `main.tex`、`sections/*.tex`、`figures/*.tex`、`results/*.tex`、`references.bib` 与成稿 PDF。未读取模型实现、绘图程序、数据处理程序、既有编辑记录、既有审稿意见、Git 历史或其他代理的项目解释。为免主任务后续修改影响审计，以上论文文件已冻结至 [原稿快照目录中的 main.tex](/D:/Users/Administrator/Documents/ChatGPT/DeltaTrace/paper/iclr2027/tmp/pdfs/cold-reader/original/main.tex)。

**下文源码行号全部指这个原稿快照，不指主任务修改后的同名文件。** 同时提供 PDF 页码与原稿评审行号，便于定位。P0 表示直接妨碍读者复述问题或理解主要结果；P1 表示妨碍机制理解；P2 表示可以低成本消除的阅读负担。这不是正确性、创新性或文献完整性审稿。

## 主要结论

正文已经能让新读者理解作者想解释“证据内容及其使用方式”，也能理解注意力的两项分解。最明显的缺口不是再少一个定理，而是几个基础连接没有直接说出：输入如何组成、固定响应在两次运行中如何使用、反向传的是系数还是贡献、有限系数如何区别于通常的局部导数、以及实验数字具体测量什么。

修订应先补这些连接，再决定是否保留更多公式。注意力分解及其紧随其后的解释值得保留；A.1 中关于割线系数的两句话比正文中多次泛述“finite changes”更适合进入主文。记忆部分需要补回“用 key 读出旧内容并作差”的动作，而不是把附录整组递推搬进正文。实验定义目前比方法定义更不自足：读完附录仍无法仅凭稿件计算 RISE 或 MAS。

## 第一遍实际卡点记录：未读附录时

以下按真实遇到的顺序记录；“当时不能确定”不等于“全文一定没有”。

| 首读位置 | 当时已经读懂 | 当时卡住的具体问题 |
|---|---|---|
| 摘要 | 要把一次输入变化造成的响应分数变化分成带正负号的 token 分数。 | reference input 是删掉上下文、另一份文本，还是某种 embedding 基准？完整 response 从哪里来？暂时等待方法节。 |
| Introduction 第 1 段 | 姓名提供候选信息，题目中的角色词决定用哪一个名字，例子很直观。 | 作曲家得负分的“相对于什么”还不知道。语义上的干扰项身份本身不能告诉我负分的数值含义。 |
| Introduction 第 2 段 | 同一个固定响应的 log-probability 差是归因对象。 | “fixed response”是否意味着先在原输入上生成一次，再在参考输入上强制评分？还是选定一个外部答案？ |
| Introduction 方法概述 | 两次正向、一次反向；每个局部规则负责一个有限变化。 | 尚不能在纸上演示最简单的一层：局部规则给出的到底是什么量？ |
| §2.1，式 (1) | 对响应 token 的 log-probability 求和；原输入、参考输入、模型参数和响应固定关系基本清楚。 | `T selects the response positions` 是否允许只选最终答案？为什么全文又反复说 complete response？哪些输入位置是 eligible？EOS 是逐位置替换还是结束序列？ |
| §2.2 第 1 段 | 反向从标量系数 1 开始，多个路径最后会相加。 | 接着却说“distributes the contribution arriving at the output”：传递的是系数，还是已经乘过 activation difference 的贡献？残差的 divide 是否按比例分系数？ |
| §2.2 第 2 段 | 最后用 `A_i=<m_i,Δe_i>` 得到一个 token 的分数，并满足总量守恒。 | `m_i` 的维度和计算步骤仍不具体；为何不是普通梯度乘输入差？非线性例子只说“relating”，没有给出这个关系。 |
| §2.3，式 (4) | `P1 ΔV` 与 `ΔP V0` 两项及相互作用分配都能直接理解。这是首读最顺畅的机制段。 | 尚需把这个正向差分恒等式接到上游反向系数；前节的抽象说明没有完全完成这一步。 |
| §2.3，式 (5) | 权重中心化表达同一行内的竞争。 | `z` 第一次出现没有定义；“score coefficient”容易和响应 score 混。logarithmic mean 只给名字，不能看出它为何表示有限变化。 |
| §2.4 正向记忆 | 旧状态先衰减，加一个新写入，再由 query 读取。 | 新写入是 incoming value 减哪一次读取？是当前 key 读取，还是 query 读取？这个动作正是 delta rule 与简单累加记忆的区别。 |
| §2.4 反向记忆 | read、write、retain 都要分别给内容和控制分信用。 | write 列出的去向中为什么没有“被减掉的旧记忆”？还是它被最后的 retention 句隐含了？ |
| §2.4 Shared rules | 两种模型使用同一套规则。 | query–key 和 SwiGLU 用对称分配，前面强调的内容/控制分配却是非对称的；这是不同角色的约定，但理由没有直接说出。 |
| §3 attention tiles | 临时 tile 复用，只保留线性大小的向量与行统计量；算术仍是二次的，解释清楚。 | first sweep 的均值要连接式 (5)，如果那里没有读懂系数含义，这里的 row statistics 就只能记成实现术语。 |
| §3 memory chunks | 边界状态把 chunk 串起来，向后传边界贡献，再处理块内信息。 | “native chunk-state adjoints”“associative scan”为什么能实现前面的有限分配？普通 backward 与有限 backward 的关系仍悬空。 |
| §3 complete attribution | 总时间包括两次执行、replay、反向和传输。 | 此前说“recording internal representations”，这里又要 replay；原来记录了什么、为什么需要重算，没有一个存储直觉句。 |
| §4 tasks and fixed responses | 到这里终于知道主实验用 released complete response plus EOS。 | RULER、NIAH、MQ、MV、VT 和 H2/H4 等只作为任务名出现；不知道输入任务与 gold evidence 分别是什么。 |
| §4 source recovery / faithfulness | Recovery 有预算，RISE/MAS 越低越好，排序视图不同。 | 排序以后测了什么，曲线怎么变成数字？只知道删除次数和排序方式，无法解释 `0.0645` 的意义。 |
| §4 hybrid results | Qwen3.5 使用独立的小开发集。 | `63.92% to 73.35%` 是谁到谁？是两任务平均还是只有检索任务？RISE 的 versus 又把哪种方法放前面？ |
| §4 cost | 主计时范围与成对比较大体明白。 | 首页面板出现时，硬件与度量解释还在第 7 页；FT-mh、K3 等变体需要回查。 |
| §5–6 | 终于直接看到所有响应位置进入一个目标，内容/控制与统一单位的关系更稳固。 | 已在前面困惑过的读者必须逆向修正理解；policy-learning 展望又打开一个还未建立问题设定的新任务。 |

## 第二遍逐段检查：到达该段时是否已有足够前提

第二遍并未把首读困惑全部保留为缺陷。以下区分已经补足、适合后置的细节与仍需前移的前提。

| 正文段落 | 二读判断 | 最小处理 |
|---|---|---|
| 摘要 | 高层目标有，但有限归因、混合模型、内核、多个指标、工程加速及 RL 展望挤在一个摘要里。 | 优先给出一条完整任务链，再保留主要效果；无需摘要解释全部术语。 |
| Introduction 第 1 段 | 例子清楚；不需要在此扩写角色词的语言学定义。 | 只要后面的 baseline 与分数定义及时到位即可。 |
| Introduction 第 2 段 | 参考差分定位明确；fixed y 的执行含义尚未交代。 | 直说两次都评分同一条已选定响应。 |
| Figure 1 | 图中文字确实显示 EOS reference，caption 也说完整 stored response plus EOS。 | 将它作为补充示意；§2.1 仍应自己说明输入和目标。图中 Answer 是摘要而非完整计分目标，caption 已说明，正文可以沿用。 |
| Introduction 挑战段 | 内容与控制共同变化、attention 行竞争、memory 时间依赖的三个前提足够。 | 保留，不需加公式。 |
| Introduction 有限机制段 | 理念和守恒意义清楚，具体有限系数继续悬空。 | 概述保持简短，把可操作的例子安排在 §2.2。 |
| Introduction operator/computation 段 | `chunk-state adjoints` 和 `scans` 首次出现得早于其用途。 | 概述只保留 tile/chunk 复用与线性存储意义，原生算子名称留 §3 或附录。 |
| Introduction 三项贡献 | 三项与方法、内存、计算主线对齐。 | 保留。 |
| Introduction 最后实验与 RL 段 | 结果和未来学习方向属于两条叙事；后者需要新的 action/return/reference 概念。 | 主文最多留一句展望，给问题定义与指标让出空间。 |
| §2.1 例子段 | 与 Introduction 例子重复较多，未新增算法前提。 | 压缩为一句，换入具体问题输入输出。 |
| §2.1 target 段 | 明确了全局对比和正负号；source eligibility、teacher forcing、响应范围仍不足。 | 见下方 P0-1。 |
| §2.2 两段 | 直觉顺序合理，但系数/贡献混用；没有一个具体 local coefficient。 | 见 P0-2。保留末端内积与守恒。 |
| §2.3 起始例子段 | 能理解一个 token 可同时影响 value 与选择。 | 不需把姓名永久等同 value、角色词永久等同 control；稿件的 “can” 表述已较谨慎。 |
| §2.3 PV 分解段 | 两个系数所取的执行端点及相互作用归属均已明说。 | 保留核心式与两句解释。 |
| §2.3 softmax 段 | 竞争直觉成立，但式中对象缺定义；logarithmic mean 的有限含义只在附录。 | 见 P1-1；可少量补定义，或删主文公式而保留直接机制说明。 |
| §2.4 memory 引入段 | 将证据跨位置保留的用途接回例子，足够。 | 保留。 |
| §2.4 state update 段 | `u_t` 的文字定义仍缺读操作主体；“delta” 的修正用途没有落地。 | 见 P1-2。 |
| §2.4 reverse 段 | content/control 的共同原则明说了；写入 correction 的旧状态通路未说。 | 用一个有动作顺序的句子补回，不能仅列名词。 |
| §2.4 shared rules | 对称/非对称都能守恒的事实，要到附录 A.5 才能确认。 | 一句话说明是对不同算子角色固定选择的分配约定；无需证明唯一性。 |
| §3 attention 两段 | 存储复用、row sweep、二次算术三者关系足够。 | 加 `T` 包含 prompt 与固定 response 即可。 |
| §3 memory 段 | 边界状态为什么能压缩跨块历史直觉可懂；复用 adjoint 的作用边界未交代。 | 见 P1-3；保留“跨块传边界，块内分配”，少用原语清单。 |
| §3 complete 段 | replay 为什么出现未解释。 | 一句说明为节省保存的激活，处理某层时重建该层所需值；准确范围由作者确认。 |
| §4 tasks 段 | 数据量、主目标明确；`clean-v1` 和 FP16 eager 不帮助理解实验问题。 | 用任务/gold 的简短定义取代这部分主文细节。 |
| §4 metric 段 | 不足以理解表 1–3。 | 见 P0-3、P0-4。 |
| §4 complete results 段 | 对表格读数的总结清楚；困难来自前置 metric 定义。 | 不需增加胜出数字。 |
| §4 hybrid 段 | 任务范围与比较方法未和数字一一连接。 | 见 P1-5。 |
| §4 cost 段 | 同硬件成对结果本身可以理解；大部分 benchmark execution 细节已在附录。 | 主文只保留方法、相同样本/硬件、完整调用范围、关键结果。 |
| §4 signed examples 段 | 例子路线和曲线对象的说明足够；图位于附录。 | 在 metric 定义完善后，主文无需再铺一段同样的评分说明。 |
| §5 Related Work 三段 | 与本文比较的解释量清楚。 | “All response positions enter one target” 的基本前提前移至 §2.1；不靠 related work 才完成问题定义。 |
| §6 Conclusion | 归纳主线可懂。 | 展望保持一句即可。 |

## 必须优先修复的具体问题与最小解释

### P0-1：给出能照着做的输入、固定响应和输出定义

**位置：** `sections/method.tex:9–14`，PDF 第 3 页评审行 152–161；`method.tex:21–23`，第 4 页行 172–176。

**原文短引：** “The explanation concerns a fixed response $y$.”；“where $\mathcal T$ selects the response positions”；“the reference replaces eligible source tokens with EOS”。

**新人无法自行推断：**

- 一次解释接收的是 prompt/context 与一条事先选定的响应，还是需要算法自己生成响应。两次执行是否自由生成不同文本。
- 原始输入中哪些位置组成 source 集合，哪些位置固定；question 是否也在 source 中。`eligible` 指向发布代码，没有在论文内给出集合的语义。
- EOS 替换是否保持 source 位置/长度；若 EOS 出现在 prompt 内，读者可能误解为“参考运行在此停止”。
- `y` 是全部推理过程加答案，还是最终答案；`T` 的 selector 与 “complete response” 的范围如何一致。
- 最终产物是每个 source token 一个标量，还是每个 response/source token 对一张矩阵。

**最小直接解释：** 将 §2.1 起点写成一条完整过程：给定源输入和选定的完整响应，对原始 source embeddings 与逐位置替换得到的参考 embeddings 分别评分；两个评分都使用同样的 response tokens 及同样的 response prefixes；返回每个可归因 source token 的一个 signed scalar。随后用现有式 (1) 定义求和分数。评估中响应来自 released stored response，并包括 EOS；把这一点前移。eligible 范围必须用自然语言说出，不能用 `released code` 替代定义。

还应加极短的“响应 token 本身保持固定，source 的变化可通过它们的中间表示影响后续评分”。这是解释为何固定 response 可以仍有一条长反向链所需的直觉。没有必要展示全体 token 的张量布局。

**附录核对：** §C 仍将 eligible positions 委托给 release；没有给出完整的 source 范围定义。完整响应范围在正文 §4、Figure 1 caption 和附录 C 多次出现，属于应前移的信息。不能在修订时凭经验自填“只替换 context”或“全部 prompt 都替换”。

### P0-2：反向传的是系数；贡献是系数与这次变化的内积

**位置：** `method.tex:18–31`，PDF 第 4 页行 164–184。

**原文短引：** “starting with coefficient $1$”；“it distributes the contribution arriving at the output”；“residual connections divide them among branches”；“relating its observed output change to its input change across that interval”。

**新人无法自行推断：** 第一段从 coefficient 转到 contribution，再到第二段的 `m_i`，没有说明三者关系。读者可能认为从 `ΔF` 开始分配标量，也可能认为从 1 开始按常规梯度反传；现有文字不能排除任一种理解。`A_i=<m_i,Δe_i>` 给出末端定义，但没解释如何得到 `m_i`。仅说 “finite changes” 不能区分有限割线与局部导数。

**最小直接解释：**

1. 将反向量统一称为 coefficient；到需要解释局部信用时才说该节点贡献等于 coefficient 与该节点 observed difference 的内积。
2. 把附录 A.1 的 scalar nonlinearity 例子前移：普通局部导数对应一个点附近的变化；这里用两次执行之间的割线系数 `(f(a1)-f(a0))/(a1-a0)` 来重现这一次有限输出差。相同输入时使用连续极限。一个行内例子就够，不需要在主文新增完整定理。
3. 接一句“把每个局部规则的系数沿计算图向后复合，并将来自不同后续路径的系数相加”，然后给出已有 token 内积。
4. 显式定义 `Δe_i=e_{i,1}-e_{i,0}`，`m_i` 为同 embedding 维度的反向系数，`i` 是 source token 位置。

**残差必须改清楚：** 原文的 “divide them among branches” 可以解释为分配贡献，却很容易被读成把 coefficient 1 分成几个份额。附录 A.1 明说 “addition sends the coefficient to each summand”。主文应同样直接说：加法将同一上游系数送至各支路；各支路的贡献取决于自己的激活差。无需添加“平均分配”之类不存在的规则。

**附录核对：** `appendix.tex:6–21` 已包含所需的局部恒等式、转置作用、标量割线、共享节点求和与常量输入零贡献。正文的问题是最基础的解释被放在证明前的附录段落里，不是必须再推导一次数学。

### P0-3：RISE 与 MAS 还没有被定义为可理解的测量量

**位置：** `evaluation.tex:7–8`，PDF 第 6 页行 318–321；`results/full_table.tex:3,31`；`appendix.tex:149–150`，PDF 第 13–14 页行 698–706。

**原文短引：** “RISE ranks signed contributions and MAS ranks their positive part; lower is better.”；“Saved signed deletion curves supply RISE”；“MAS always comes from the positive view.”

**新人无法自行推断：** 排序只是产生删除顺序。它没有告诉读者在每步删除之后评分的对象、纵轴含义、归一化、以及怎样从曲线生成最终的两个标量。`max(A_i,0)` 与 descending signed 排序具有相同的正分排序前缀，何以得到不同的 RISE/MAS 数值也需要 metric 定义。只看当前稿件，不能知道 `0.0645` 是曲线面积、删除达到某个水平的位置，还是另一种量。

**最小直接解释：** 正文用两三句走完“按贡献排序 → 分 20 步处理 source → 每步重算同一固定 response 的分数 → 将归一化曲线汇总为 metric”这条链。说明删除操作究竟是移除、替换还是另一个规定动作；正分/带符号的视图是算法分数怎样进入 evaluator 的选择。随后直接定义 RISE、MAS 分别如何汇总，以及为什么更低表示高排名证据能更早削弱模型对该响应的支持。若确有必要，保留一个紧凑公式；完整边界处理留附录。

**严格限制：** 我没有从名称猜测 RISE/MAS 的公式，也没有读取 evaluator。当前论文内部不足以安全写出两个指标的计算式。附录 C 只补出了使用完整 response score、按 full/fully-deleted endpoints 归一化、裁剪至 `[0,1]` 并取 cumulative minimum。作者需要从实际协议核实后给出聚合定义；不能把我上面的“曲线面积”等候选解释当成事实写回论文。

### P0-4：任务、gold evidence、预算分母与目标例外需要一起交代

**位置：** `evaluation.tex:5–8,13`，PDF 第 6–7 页行 309–348；`full_table.tex:59–79`；`appendix.tex:150,155–156`。

**原文短引：** “NIAH recovery uses the top 10% of released eligible tokens.”；“VT reports body-token Recall”；“VT reconstructs the cached answer without its reasoning prefix”。

**新人无法自行推断：** MQ/MV/VT 所考的任务、被找回的 gold token 类型、`eligible` 与 `body-token` 的区别、HotpotQA sentence/fact 单位与 token budget 的对应。表 3 的 10% 是分数排序覆盖的预算，不是 recall 的分母，但正文没直接让这两者分开。NIAH 排正分、HotpotQA 按有符号句子总分，以及 VT 只用去掉推理前缀的答案目标，都到附录才出现。

**最小直接解释：** 正文在任务首次出现时分别用一句介绍检索 key/value、变量追踪、多跳事实与数学题的任务作用，并给表中缩写对应。介绍 recovery 时直接说：预算限制选择多少 eligible/body tokens；Recall 统计这些被选位置覆盖多少 gold evidence tokens，HotpotQA 则统计 gold supporting facts。给一个数值例子即可区分“选 10% 输入”和“找回 66.76% gold evidence”，无需定义很多集合。

VT 的回答目标例外必须在正文与表 3 附近明说。现有主文确实把 complete-response 句限定给 deletion faithfulness，但整篇方法叙事和第一页面板很容易使读者将此设定延伸到全部 recovery。用一句“VT recovery 使用去掉推理前缀的缓存答案；完整响应用于上述 faithfulness”即可给出清楚的范围。这是解释实验究竟验证哪个任务所必需的信息。

### P1-1：softmax 公式要先说明它的输入、输出和有限含义

**位置：** `method.tex:48–55`，PDF 第 4 页行 201–211。

**原文短引：** “its score coefficient is”；“The positive weight $\ell_i$ is the logarithmic mean”。

**具体断点：** `z` 没有定义；`u_i` 虽叫 incoming coefficient，但上一节未澄清 coefficient 的含义；`i` 从 source token 换成 attention-row 元素也未提醒。logarithmic mean 的名字不能让普通熟悉 transformer 的读者自行知道式 (5) 为什么是一条有限变化规则。

**最小直接解释：** 先说“对一个 query 对应的 attention row，`z_i` 是给第 i 个可见 source 的归一化前分数；`u_i` 是从 attention probability 收到的反向系数”。若保留式 (5)，紧接着给 `ell_i` 的行内定义或一句准确的 finite ratio 解释，并说明相同两端概率时它退化为该概率。这样读者才能把“同一行竞争”和“从两个执行得到的系数”联系起来。

若主文页数紧，另一种更自然的写法是移走式 (5)，保留机制句：两次执行的概率共同确定每个 source 的权重，减去同一行的加权平均系数，因而将选择变化归给相对其他候选的变化。附录保留精确公式。不要同时在主文新增完整 `Δp` 矩阵、log-mean 定义、log-softmax 系数和证明。

**附录核对：** A.2 才完整定义 `p_b=softmax(z_b)`、log mean 和 `Δp` 的线性映射。这些很好地支持现有主张，但不能替代主文首次使用 `z` 时的定义。

### P1-2：记忆部分要讲清“先查旧值，再写差额”，并补回其反向去向

**位置：** `method.tex:62–72`，PDF 第 5 页行 220–234；`appendix.tex:97–111`。

**原文短引：** “the difference between the incoming value and the value read from retained memory”；“At a write, they pass to the incoming value, the retrieval key, and the write gate.”

**具体断点：** 两种读取混在一起：更新前用 key 读已有内容，更新后用 query 读输出。主文只在后一种 read 中明确 query，前一种没有说出动作主体。`S_t` 看起来像存了一段证据的向量，但式中的 outer product 表明它在存 key–value 关联；不熟悉 gated delta 的读者无法据此建立一个准确的运行图。

**最小直接解释：** 保留现有 `S_t=alpha_t S_{t-1}+k_t u_t^T`。用连续动作说明：状态存储 key 与 value 的关联；先按 retention gate 保留旧状态，再用当前 key 查出该关联已有的 value，将 incoming value 与查出值之差按 write gate 缩放后写回，最后由当前 query 读出输出。指出写入的是修正量，可直接解释“delta”。所有门都来自同一输入产生的内部计算，不能让读者误认为是手工指定的外部控制。

反向段补一句：由于写入使用“新值减旧读值”，这部分信用也经减法返回被读出的旧状态，并再回到更早写入；key 同时参与查旧值和定位写入。其余 read 与 retention 说明已可保留。这个动作链比主文增加整组六个中间变量公式更有用。

**附录核对：** A.4 的 `r_t=T_t^T k_t`、`c_t=v_t-r_t`、`u_t=beta_t c_t` 恰好澄清上述问题；有限递推也显式包含 `Δv_t−Δr_t`。因此这是省略了关键解释，不是要求新增方法。

### P1-3：把有限传播与 PyTorch VJP / native adjoints 的关系说到恰好够用

**位置：** `method.tex:18,75–76`；`computation.tex:12`，PDF 第 5 页行 257–264；`appendix.tex:136–140`。

**原文短引：** “Like backpropagation”；“native chunk-state adjoints propagate the boundary coefficients”；“their linear transpose actions”。

**具体断点：** 稿件说 finite rules，又说 native adjoints，没说明是怎样兼容的。熟悉 PyTorch 的读者可能将整条路径理解成对原模型做一次普通 VJP，也可能以为所有原始算子 backward 全部被重新实现。两种解读都不能仅凭原稿确认。

**最小直接解释：** 方法节说明所传播的是由两次执行构造的局部有限映射的转置作用。数学上，线性算子的有限映射就是它原有的线性映射，因此可以沿用该线性转置；非线性与交互则按前面的有限分配选择其映射。实现节只需再说明 PyTorch VJP/native adjoint 承担了哪一类已定义的线性作用，以及何处使用有限规则。不要展开 hooks、checkpoint key、API 名单或开发过程。

**证据边界：** 原稿除了环境中的 PyTorch 版本，没有交代实际 PyTorch VJP 调用的作用范围。上述线性关系可由 A.1 与 B 的数学描述推得；实际代码是否和怎样这样调用，不能从稿件确定。我没有读取实现，报告不宣称它用了某个 autograd API。作者必须依实现核对一句准确说明。

### P1-4：效率已经有好直觉，再补两个连接即可

**位置：** `computation.tex:6–16`，PDF 第 5 页行 245–269；`appendix.tex:140,162`。

**原文短引：** “The tile can then be reused.”；“For sequence length $T$”；“Layer replay supplies the activations needed by the current layer”。

**已经清楚的部分：** attention 的 tile 内重建、行统计量跨 tile 保存、辅助空间 `O(Td)`、算术 `O(T^2d)` 都讲得可懂。无需把这部分判作“完全没解释效率”。memory 的边界状态足以说明为何不必保存所有历史交互。

**仍需补的两点：**

- `T` 是源输入和固定响应合在一起的序列长度，附录 D 才说明。对于标题中的 reasoning response，这是理解长度成本的必要范围。
- 前面说两次运行记录 internal representations，§3 又引入 replay；需要一句说明保存检查点和重建当前层激活如何减少同时保存的内容。无需给缓存文件、调度标识或多种保留策略。

再加一句把“所有 response token 的 log-probability 先相加成一个标量”接到“一次反向复合即可累积各位置的 source 信用”。这样单次反向的收益更容易理解，也不会被误读为每个 response token 都需另跑一次。稿件已有这些事实，只是分散在多个节中。

### P1-5：混合模型数字必须逐个标出比较对象

**位置：** `evaluation.tex:15–16`，PDF 第 7 页行 350–355；`development_table.tex:14–17`，PDF 第 14 页 Table 5。

**原文短引：** “recovery rises from 63.92% to 73.35%”；“RISE is 0.1115 versus 0.0942”。

**具体断点：** 第一组数据按 FT → DT 书写，后一组按 DT versus FT 书写；两处均未在同句命名比较对象。前文又提八个 MQ-Q2 与八个 MoreHopQA，读者可能把 recovery 理解为两任务平均。

**最小直接解释：** 明说 Qwen3.5 的八个 MQ-Q2 样例上，DT recovery 为 73.35%，FT 为 63.92%；MoreHopQA 没有这一 recovery 值。所有后续 versus 保持相同顺序，直接概括“MQ-Q2 RISE 较高、MoreHopQA RISE 较低”。正文或紧邻表注展开 FT K3 是三次 recursive hops；不用让读者翻到开发表才解码。

这是读数归属问题，不需要新增实验、显著性检验或更大开发集才能修复。

### P2：保留机制解释，压缩无法帮助首次理解的主文信息

**可压缩位置：** `main.tex` 摘要尾部；`introduction.tex:25,34`；`method.tex:6`；`evaluation.tex:5,19–20`。

**原文短引：** “The frozen clean-v1 method”；“FLA chunk-state adjoints, batched products, and scans”；“a 35.3% throughput increase from real batching and GPU checkpoints”。

这些信息中，完整调用范围与同硬件比较条件对理解成本结果有用，应保留。snapshot 名称、实现入口、多次工程调优的具体命名、在摘要中报一次 batching 调整的收益，对理解方法并无同等价值。可以移至现有实现附录，给上述问题定义与指标解释换位置。

第一页面板不是尺寸排版错误，但解释路径太长：读者在第 1 页遇到多个未定义任务和方法缩写，直到第 7 页才得到各雷达轴范围与硬件条件的说明。最小改法是在面板紧邻位置有一行可读的任务/metric 指向与硬件说明，并在第 1 页贡献叙事中明确面板想回答哪个问题；也可将完整面板移到结果定义之后。两张完整数值表浮在第 6 页 §4 标题之前，也使指标出场晚于数字；如调整浮动位置，优先让 metric 段先于完整表。

## 读附录以后才清楚，以及读完仍不清楚的内容

| 问题 | 正文二读状态 | 附录提供了什么 | 应如何处理 |
|---|---|---|---|
| source 最终分数为何是 coefficient × difference | 有末端公式，缺局部过程 | A.1 的转置作用与局部内积恒等式 | 将一个具体例子和术语关系前移，证明留附录。 |
| 非线性为何不是常规局部梯度 | 只有 finite changes 的概述 | A.1 的 divided difference 与重合极限 | 一句加行内式即可。 |
| residual 的 divide 怎么实现 | 容易读成分掉系数 | A.1 明说系数送至每个 summand | 主文术语改准。 |
| softmax 的 `z` 和 log mean | 有中心化直觉，定义不足 | A.2 给全部定义、恒等式与证明 | 符号定义必须前置；证明不搬。 |
| memory delta 如何查旧值 | 主体与动作先后不明确 | A.4 给 key 读取、value−read、write gate、query output | 用动作顺序解释，现有主文状态公式保留。 |
| 两种 product allocation 的关系 | 原则名称看似不统一 | A.5 证明对称和定向两种恒等式都成立 | 一句点明按算子角色选固定分配；不宣称守恒唯一确定分配。 |
| PyTorch VJP/native adjoint 的作用边界 | 未交代 | B 只说线性 transpose 与 native adjoints | 仍缺；作者核对实际实现后补一句，不能由审读者猜。 |
| eligible source 具体范围 | 委托 released code | C 继续委托 release | 仍缺；在问题设置与协议分别给语义范围。 |
| RISE/MAS 的标量定义 | 只有排序和 lower better | C 给曲线归一化、clip、cumulative minimum | 仍缺两个 metric 聚合定义。 |
| VT recovery 的目标范围 | 容易继承 full-response 理解 | C 明说去掉 reasoning prefix 的 cached answer | 前移至主文 recovery 说明。 |
| Hybrid recovery 的任务、比较对象 | 文字不明确 | Table 5 显示仅 MQ-Q2，FT 与 DT 两行 | 在主文同句标明。 |
| `T` 包含哪些位置 | 未定义范围 | D 说明 input 与 response 都计入 | 前移至首次复杂度声明。 |

## 符号首次出现检查

不是要求把所有维度都塞进正文，而是把直接决定读法的对象说清楚。

| 符号或用语 | 原稿首次关键使用 | 缺失或易混部分 | 最小补充 |
|---|---|---|---|
| `x0,x1` | Figure 1 caption；§2.1 | 在图中为 reference/original；具体可变范围不明 | 0/1 对应两端，以及哪些 token embeddings 变化。 |
| `y` | 摘要/引言 fixed response；§2.1 | 来源、完整范围、两次执行中的使用方式 | 事先选定完整响应，评分时同一 prefix；实验含 EOS。 |
| `T` 花体集合 | 式 (1) | “selects”像可任选子集 | 完整响应的哪些位置进入求和；保留可选接口时说明本文选择。 |
| `theta` | 式 (1) | 未命名，但 `model parameters held fixed` 很快给了含义 | 可行内称固定模型参数，不必独立解释。 |
| `i,A_i` | 引言守恒；式 (2) | source 粒度未直接定义 | i 是可归因输入 token 位置，A_i 为一个标量。 |
| `m_i,Δe_i` | Figure 1；式 (2) | coefficient 维度、embedding difference 方向 | 与 embedding 同维度；original minus reference。 |
| `P1,V0,ΔP,ΔV` | 式 (4) | 端点约定可由 x0/x1 推回 | 统一声明下标 0/1 表示两次执行，即可，不必重复。 |
| `z_i` | 式 (5) | 完全没有在主文定义 | attention normalization 前的 score/logit。 |
| `u_i` | 式 (5) | 反向系数与后文 write vector `u_t` 同字母 | 可以保留局部符号，但每节明确对象；若改为 `m_{p,i}` 更少跳转。 |
| `ell_i` | 式 (5) | 只命名 logarithmic mean | 一个行内比值或移走该主文公式；不要只增加另一个术语。 |
| `S_t,k_t,u_t,alpha_t` | 式 (6) | state 类型、key 的读写双角色 | 用“存 key–value 关联、key 查旧值、write 为修正”解释。 |
| `q_t,beta_t` | 主文只叫 query/write gate，附录正式定义 | 主文不用这些符号并不是问题 | 讲清其动作即可；不必为它们强加一整组公式。 |
| `T,d` 复杂度 | §3 | T 是否含 response、d 是 per-head | d 已定义；补 T 为合并序列长度。 |
| RISE/MAS/NIAH/MQ/MV/VT/K3 | 摘要、首图、§4、表注 | task 缩写与 metric 含义未完整定义 | 任务一句定义，metric 一条计算链，K3 一次展开。 |

## 九页以内的修改配额

原稿正文到第 8 页结束，尚有一定空间，但不建议用新增长推导填满。建议保留现有六个主文展示公式中的目标、token 内积、PV 分解、memory 状态更新；守恒可行内出现，softmax 是否展示取决于能否随式完整定义对象。标量割线用行内式即可。

应新增的是约 3–5 个短段落的信息量：问题输入输出、有限系数与贡献、memory correction、实验 metric/task 定义。可从以下内容等量换出：§2.1 重复的 playwright 叙述、引言中原生内核操作名单、主文 `clean-v1` 与执行入口类标签、摘要中的独立 batching 调优结果、重复的 policy-learning 展望。这样不需要缩小字体、压缩行距或扩大附录对基本问题定义的承担。

机制的数学细节已经在 A.1–A.5 中。正文修订应让读者知道这些公式为什么存在、各自传什么，而不是把公式组全部提前。

## 修改后应让新读者能直接回答的八个问题

1. 算法给定哪些东西，改动哪一部分，最后返回什么？
2. 在 reference input 上为什么仍能评分同一条 response；完整 reasoning/answer/EOS 各有哪些位置计分？
3. `m_i` 与 `A_i` 是什么关系；为什么从 1 反向开始？
4. 一层非线性怎样把两个执行间的变化传回去，与普通局部导数有什么具体关系？
5. attention 的 content/control 两项分别使用哪一端的量；softmax 如何让一个 source 的选择变化联系到同行其他 source？
6. gated memory 的新 write 为什么要减旧 read；这条旧状态路径怎样回到更早 source？
7. 辅助存储为何随总序列长度线性增加，一次 response-level 反向与 tile/chunk 复用分别节省什么？
8. RISE/MAS 和 Recovery 的一个数值各是什么意思，目标响应、source 范围、预算、gold 单位分别是什么？

本报告没有编辑任何 manuscript 文件，也没有用实现细节补足稿内缺失前提。读完附录才获得的理解已单独列明，不能算作原稿首次阅读时已经清楚。

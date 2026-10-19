# v9 执行 v2：两次自然迭代的效率证据

采集：2026-10-19T12:50:37.471292+08:00。TextCraft DTPO已完成2/200，其他五组尚未开始。

第2轮的比较仍显示资源开销，不能把首轮差额全部归为首次编译：

| 阶段 | 当前v9第2轮秒 | 历史八卡v8第2轮秒 | 当前变化 |
|---|---:|---:|---:|
| 生成与环境 | 439.771 | 438.009 | +0.4% |
| old 前向 | 38.835 | 39.453 | -1.6% |
| ref 前向 | 33.858 | 34.277 | -1.2% |
| DT/优势构造 | 218.311 | 177.047 | +23.3% |
| actor 更新 | 163.708 | 142.760 | +14.7% |
| 整轮 | 894.964 | 831.986 | +7.6% |

当前输入938251 token，历史961003 token，当前少2.37%；actor按输入token折算仍慢17.45%。当前DT实际118条唯一轨迹，补齐128行，八rank各4个B4，原readout报告的padding计算长度槽位合计438960。DT的筛选和padding负载与actor整批不同，表中DT比例仅是阶段时长比。

原生actor RPC末尾缓存释放，第1轮各rank15.274–17.386秒、第2轮14.923–16.386秒。第2轮DT的B4末尾缓存释放各rank累计32.031–42.552秒。这些值已包含在对应阶段内，各rank相互重叠，不能相加成额外wall time，也不能据此拼出一个未测的“纯内核耗时”。

代码核对发现，当前`DT_NATIVE_HOST_CACHE_REPLAY_LAYER=0`同时使已消费root/kwargs保留到B4结束。独立候选只在原runner中将消费后的张量释放与`torch._C._host_emptyCache`调用分开；保持原Torch缓存owner、原数值计算、flag=1和observer行为。候选SHA为`9f13c898c9faec72f57d77a81966315f25950c251e4381598b8f5442e287c5cf`，**仅准备、未原生验证、未部署**，不能宣称已加速或容量已验收。正式训练仍使用runner `6a112224`及执行提交`2875c40d`。

完整原生日志路径/行号、两轮八rank原readout、缓存释放记录、历史对应迭代和候选身份见 [JSON证据](D:/Users/Administrator/Documents/ChatGPT/DeltaTrace/experiments/rl/results_dtpo_v9_two_iteration_efficiency_20261019_v2.json)。本次读取和候选准备没有新增模型、DT、optimizer、生成或GPU profiling调用。

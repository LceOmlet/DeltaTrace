# v9 执行 v2 首个完整迭代的效率核对

采集：2026-10-19T12:36:12.228604+08:00。当前 TextCraft DTPO 完成第1轮；该比较只核对效率，不比较训练质量。

| 阶段 | 当前 v9 秒 | 历史八卡 v8 首轮秒 | 当前变化 |
|---|---:|---:|---:|
| 生成与环境 | 453.903 | 471.101 | -3.7% |
| old 前向 | 52.393 | 52.398 | -0.0% |
| ref 前向 | 31.663 | 32.630 | -3.0% |
| DT/优势构造 | 289.211 | 156.175 | +85.2% |
| actor 更新 | 175.328 | 149.933 | +16.9% |
| 整轮 | 1002.954 | 862.651 | +16.3% |

当前原生输入921230 token，历史922204 token，相差-0.106%；32组×8、B4/rank、global64、epochs1及每轮4次actor更新配置一致。阶段之和与整轮相差0.008秒，未发现计时之外的大段未解释等待。按原生输入token折算，actor更新仍慢17.1%，不能仅用轨迹长度解释。第1轮未触发save5，检查点时间缺失不填0。

DT实际处理149个唯一非零信用轨迹，原DP补齐为160行，八rank各5个B4，共40个B4。原readout报告的padding后计算长度槽位合计515792；它不是整批actor输入921230 token，不能用后者宣称DT单位工作量加速或减速。

自然首轮完成后的原只读metadata RPC已实读八rank：各496个Adam状态，全部step=4、scheduler=1、DP8及批准v9模式。已自然导入的readout、counterfactual、DT runner及actor/core源码均与准备绑定一致；dt_training_batch未在actor worker导入，不将其冒称为该worker已导入模块。此检查不调用模型或optimizer。

原生B4缓存释放计时各rank累计35.714–45.946秒，单次最长约27.4秒；历史保留日志同位置多为毫秒至0.1秒。该项在DT总时长内，不能再加到整轮上。当前资源配置使用原cudaHostAlloc及RPC内缓存，以避开已复现的MCR mcHostRegister限制；此首轮证明训练能通过该位置，同时暴露资源路径的实际耗时。尚不能将DT全部差额归因于这两个配置，也不能把未单独测到的编译时间扣除。

历史v6首轮868.221秒、DT150.903秒，亦作为独立参考保存在JSON。其方法及实际DT筛选负载与v9不同，未拼接历史曲线。原native MFU不支持Qwen3.5而输出0，不将其解释为零算力；没有现成连续物理利用率/显存带宽记录，不补估。

所有数值、配置、八rank原readout、缓存释放原行号及历史原日志路径见 [JSON记录](D:/Users/Administrator/Documents/ChatGPT/DeltaTrace/experiments/rl/results_dtpo_v9_first_iteration_efficiency_20261019_v2.json)。本次只读已有训练状态，新增模型、DT、optimizer、环境和profiling调用均为0。

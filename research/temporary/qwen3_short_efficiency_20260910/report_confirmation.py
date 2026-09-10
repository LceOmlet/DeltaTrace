"""Render the measured conclusion and its limits from verified summaries."""
import argparse,json
from pathlib import Path


def render(study):
    summary=json.loads((study/'confirmation_v4_summary.json').read_bytes())
    author=json.loads((study/'author_graph_verification.json').read_bytes())
    inventory=json.loads((study/'study_inventory.json').read_bytes())
    assert summary['acceptance_passed'] and author['all_full_vectors_and_math_exact']
    sizes=[128,256,512,1024];dt='deltatrace_graphed';both='ifr_multi_hop_both';plain='ifr_multi_hop'
    cell=lambda method,n:next(x for x in summary['pooled_cells'] if x['method']==method and x['target_input_tokens']==n)
    comparison=lambda method,n:next(x for x in summary['comparisons'] if x['FT_method']==method and x['target_input_tokens']==n)
    latency=['| 目标输入 | 实际完整序列 | DT 图入口 / ms | FT Both / ms | 原 FT multi-hop / ms | 比 Both 降低 | 比原 FT 降低 |',
        '|---:|---:|---:|---:|---:|---:|---:|']
    memory=['| 目标输入 | DT allocated / GB | FT 最大 allocated / GB | DT reserved / GB | FT 最大 reserved / GB |',
        '|---:|---:|---:|---:|---:|']
    intervals=['| 目标输入 | DT / Both，95% 区间 | DT / 原 FT，95% 区间 | 两轮轮均值均通过 |','|---:|---:|---:|:---:|']
    for n in sizes:
        a,b,c=cell(dt,n),cell(both,n),cell(plain,n);cb,cp=comparison(both,n),comparison(plain,n)
        latency.append(f"| {n} | {a['actual_total_tokens']} | {1000*a['mean_seconds']:.3f} | {1000*b['mean_seconds']:.3f} | {1000*c['mean_seconds']:.3f} | {cb['latency_reduction_percent']:.2f}% | {cp['latency_reduction_percent']:.2f}% |")
        memory.append(f"| {n} | {a['peak_allocated_gb']:.3f} | {max(b['peak_allocated_gb'],c['peak_allocated_gb']):.3f} | {a['peak_reserved_gb']:.3f} | {max(b['peak_reserved_gb'],c['peak_reserved_gb']):.3f} |")
        ib,ip=cb['bootstrap_ratio_95_percentile_interval'],cp['bootstrap_ratio_95_percentile_interval']
        intervals.append(f'| {n} | [{ib[0]:.4f}, {ib[1]:.4f}] | [{ip[0]:.4f}, {ip[1]:.4f}] | 是 |')
    cold=['| DT 进程/轮次 | 目标输入 | 第一完整预热调用 / s | 第二完整预热调用 / s |','|---:|---:|---:|---:|']
    for c in summary['cells']:
        if c['method']==dt:cold.append(f"| {c['round']} | {c['target_input_tokens']} | {c['warm_seconds'][0]:.3f} | {c['warm_seconds'][1]:.3f} |")
    text='''# Qwen3 ≤1k、B1：DT 完整热调用快于两版固定 FT

2026-09-10，预先冻结的四档速度验收已通过。新增显式入口为 [`make_graphed_qwen3`](../../../..//deltatrace/accelerated/graphed_qwen3.py)，[使用说明](../../../../deltatrace/accelerated/GRAPHED_QWEN3.md)包含构造、完整返回、审计开关和释放图缓存的方法。

## 结果及适用范围

使用固定作者 exp1 的输入与输出构造器、原完整同步计时器和原 FT 入口。在 MetaX C550 64 GB、Qwen3-8B FP16、B1 下，四档实际完整序列为 158/265/478/905 token。输出固定为32 token加EOS。原默认数据文件不存在，因此四档沿用作者代码的 `RULER fallback text.`，没有新生成。下表是两轮独立进程共六次测量的均值。

'''+ '\n'.join(latency)+'''

![完整热调用延迟和预先冻结的比值置信区间](warm_latency_and_ratio.png)

两轮各自的均值以及合并均值均须快于 **FT Both 和原 FT multi-hop 两者**；另要求合并 DT/FT 均值比值的95% percentile bootstrap区间上界低于1。两方法独立重采样20,000次，种子20260910；每方法每档六条重复全部纳入。此门槛在测量前已写入[最终协议](confirmation_v4_protocol.json)，没有剔除慢重复。

'''+ '\n'.join(intervals)+'''

最长档第二轮相对 FT Both 仅快 **0.60%**，优势很小；合并后快4.89%，比值95%区间为[0.9205, 0.9848]。因此这里的“通过”指该环境与冻结协议的验收，不是所有机器、文本、形状切换频率的普遍保证。11条原始短样本用于另行验证兼容性，没有把它们称为独立速度复测。

完整[数值汇总](confirmation_v4_summary.json)、[均值CSV](confirmation_v4_summary.csv)保存两轮逐条延迟、预热、峰值和额外审计成本。最终六个进程共120条计时记录：48条预热、72条测量，零排除；计时行合计100.899秒，另行完整向量/旧入口审计合计23.720秒，模型加载与工厂初始化单列。

**固定 FT 输出的已知限制：**FT Both 的完整返回矩阵每档分别含1400/2470/4600/8870个非有限元素，两个验收版本中均保留该原始返回。没有修改、清洗或过滤固定基线，也没有据此作 FT 输出质量结论。此处比较原入口的完整执行耗时；原 FT multi-hop 和 DT 的返回向量有限。

## 实现与一致性

每次归因都执行一次真实原模型 B2根前向，捕获其本来已经计算出的中间张量，省去36次decoder重放。局部 CPython 事件仅被动观察原公开 FA 返回。有限传播沿用原生半精度 GEMM、原编译有限规则、36次额外公开 FA 和原有限 FA 库。

同一有限 GPU 程序由公开 `torch.cuda.CUDAGraph` 记录并重放。每次调用将全部883个逻辑输入视图更新到静态图缓冲；按真实存储关系合并为658次拷贝，并检查原形状、步长、偏移和共享关系。图缓存只保留一种形状，不缓存归因结果。种子校验由原先的同步布尔读取移至返回前统一核验；原829个布尔谓词、36个逐层统计及最终有符号向量有限性检查均执行。

所有有限数学表达式保持；没有改写模型、FT、原生注意力/反向或有限 FA 内核。额外图缓冲及保留的根张量随层数、序列长度线性增长，没有引入全局序列平方矩阵。实现以固定 eval 模型和同一形状的连续热调用为使用场景。

- 最终四档×两轮共8个对照：新入口和同进程未改动 retained 入口的完整有符号向量、目标差、总和、未分配残差、36层最大乘子诊断全部精确相同。
- 所有原始16例中完整序列≤1024的11例均纳入：MH0–7、NI0/3/6。保留原ID、生成轨迹与完整目标；没有裁剪、按结果挑样或新增评分。22次完整调用全部通过向量与数学诊断精确一致检查。
- V3/V4图试验还更换了同长度输入的一个实际可归因token：复用同一图而输出改变，并与原 retained 完整向量精确相同。此控制检验了每次输入更新与结果重新计算。
- 每个新入口审计额外比较720个实际根张量快照和全部883个图输入视图。8种小型真实GPU图重放控制覆盖合法输入、种子失败、相等性失败、Inf、非正值、两处RoPE错误及最终NaN；合法输入通过，七种故障均在返回前被拒绝。
- 捕获异常后的hook/monitoring清理、共享存储视图刷新、错误共享关系拒绝和显式 `close()` 均通过。源码推导核验覆盖13个运行时文件和27个clean源文件；111项远端源码、库及检查点身份核验通过，最后GPU无进程。

证据：[源码推导核验](graph_source_verification.json)、[原始作者样本核验](author_graph_verification.json)、[作者样本冻结选择](author_graph_plan.json)、[结束身份核验计划](postflight_plan.json)。

## 显存和冷启动成本

峰值从完整API入口重置，包含根前向、持久图输入和图内存池。表内数值取两轮所有热测量的最大值；GB按10⁹字节计算。

'''+ '\n'.join(memory)+'''

首个完全空 Inductor/Triton 缓存的完整调用为31.794秒。每种新形状首次调用均包含两次原有限程序预热、一次图记录和一次真实重放；这些额外程序执行未挪到计时外。第二进程复用磁盘编译缓存，仍重新构造自己的图。下表显示全部DT预热调用；构图子阶段、模型加载及初始化另存于JSON。

'''+ '\n'.join(cold)+'''

形状频繁变化时，完整调用需承担构图成本；此时不能使用热调用表估计总用时。40.019GB的最长档显存也是实际代价。

## 保留的失败与中间候选

| 尝试 | 结果与处理 |
|---|---|
| 根张量保留、集中检查、局部捕获、FA输入融合、RoPE检查合并 | 完整向量一致；早期 `confirmation_v2` 的短档不能稳定快于两版FT，未宣告完成。保留构成新入口的必要部分。 |
| 所有FP16→FP32边界合并 | Q/K归一化与三路残差归一化出现约10⁻⁶的局部变化，完整向量相对L2差约0.001888；未采用。随后逐边界诊断保留。 |
| 仅精确的安全cast合并、仅检查GPU图 | 均未同时达到两个短档≥3%筛选门槛，未接入最终图入口。 |
| 整段有限图V1/V2 | 捕获失败；V2完整异常栈定位到原seed包装器的GPU布尔同步读取。V1后续不完整图错误也保留。V3移动同一检查的返回时机，未删除检查。 |
| 整段有限图V3 | 完整向量/数学/换输入控制均通过；128档更快、256档变慢，未采用。剖析发现重复输入拷贝开销。 |
| 按真实共享存储拷贝的V4 | 两短档相对上一入口分别快22.87%/11.64%，所有精确检查通过后才进入四档两版FT验收。 |
| 最初 `confirmation` | 跨编译缓存的历史向量精确门槛失败；后续独立诊断中新入口与未改动旧入口同进程精确一致，二者同时偏离旧缓存向量。未将此归因于已证明的算子原因。首次异常向量仅保留了当时写出的哈希，断言前未保存NPZ；后续向量不冒充该缺失向量。 |
| `confirmation_v3` | 工厂要求先选择原生FA，驱动却稍后选择；在初始化处退出，归因计时行数为0。V4仅修正设置顺序，运行时和FT未改。 |

全阶段共有'''+str(inventory['report_count'])+'份结果报告和'+str(inventory['all_recorded_timing_rows'])+'''条作者计时器记录（170条预热、344条测量），其中5条调用失败保留为空延迟。509条成功计时行合计450.334秒；单独审计、剖析、兼容性和图构建子程序成本保存在各自原始报告，不能把这个合计误当成所有研究总成本。见[逐行CSV](all_timing_rows.csv)、[全部尝试索引](study_inventory.json)及[继承元数据说明](protocol_metadata_corrections.md)。

## 原始材料与CPU复核

两份原始ZIP合计约4.3MB，包含完整NPZ向量、原始JSON/JSONL、失败日志、冻结驱动/协议、剖析及结束状态。每个成员均有字节数与SHA-256清单，ZIP本身也已核验。

- [全部试验和验收原始材料](raw/evidence_pre_author.zip)（231个文件），[逐文件清单](raw/pre_author_manifest.json)。
- [原始作者样本与结束核验](raw/evidence_author_postflight.zip)（11个文件），[逐文件清单](raw/author_postflight_manifest.json)。

在本目录执行下列命令即可复核归档、最终统计及兼容性，不调用模型或GPU。`/path/to/extracted` 为本地解压目标，已有同名文件必须逐字节相同才继续。统计环境使用NumPy2.3.5；绘图依赖单列在[图环境记录](figure_environment.json)。

```bash
python verify_evidence_archives.py --extract /path/to/extracted
python summarize_confirmation.py --raw /path/to/extracted --run confirmation_v4 --protocol confirmation_v4_protocol.json --output-prefix confirmation_v4_summary
python verify_author_graph.py --raw /path/to/extracted --output author_graph_verification.json
python verify_graph_sources.py --repo /path/to/DeltaTrace
```

冻结的[验收驱动](benchmark_confirmation_v4.py)、[串行队列](run_confirmation_v4.py)和[运行时清单](../../../../deltatrace/accelerated/graphed_qwen3_sources.json)保留具体环境与路径。原生库、固定模型及作者FT代码需与清单相符；不能直接将另一环境的结果当作本次复现。
'''
    return text.replace('../../../..//','../../../').replace('../../../../','../../../')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,default=Path(__file__).resolve().parent);a=p.parse_args()
    (a.study/'README.md').write_text(render(a.study),encoding='utf8',newline='\n')
    print(a.study/'README.md')

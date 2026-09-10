"""Write result documents only from independently verified completed evidence."""
from pathlib import Path
import json
S=Path(__file__).resolve().parent;R=S.parents[2];E=R/'experiments/efficiency/qwen3_rollout_20260911'
read=lambda p:json.loads(p.read_bytes())

def main():
    verified=read(S/'memory_v4_confirmation_verification.json');summary=read(S/'memory_v4_confirmation_local_summary.json')
    author=read(S/'memory_v4_author_verification.json');post=read(S/'memory_v4_postflight_verification.json');inventory=read(S/'study_inventory.json')
    curve=read(E/'curve_data.json');figure=read(E/'verification.json')
    assert verified['acceptance_passed'] and summary['acceptance_passed'] and author['all_full_vectors_and_math_exact'] and post['status']=='verified'
    assert figure['all_raw_sources_and_vectors_verified'] and figure['local_cells']==21 and (E/'figures/deltatrace-rollout-scaling.png').exists()
    methods=['deltatrace_streamed','ifr_multi_hop_both','ifr_multi_hop'];labels=['DeltaTrace','FT Both','FT multi-hop']
    table=['|目标输入|实际总长|DT / ms|FT Both / ms|FT multi-hop / ms|相对 Both 降低|相对 multi-hop 降低|','|---:|---:|---:|---:|---:|---:|---:|']
    mem=['|目标输入|DT allocated / GB|Both allocated / GB|multi-hop allocated / GB|DT reserved / GB|Both reserved / GB|multi-hop reserved / GB|','|---:|---:|---:|---:|---:|---:|---:|']
    cis=['|目标输入|DT/Both 95%区间|DT/multi-hop 95%区间|','|---:|---:|---:|']
    for n in [128,256,512,1024]:
        cells=[next(c for c in summary['pooled_cells'] if c['method']==m and c['target_input_tokens']==n) for m in methods]
        comps=[next(c for c in summary['comparisons'] if c['FT_method']==m and c['target_input_tokens']==n) for m in methods[1:]]
        table.append(f"|{n}|{cells[0]['actual_total_tokens']}|"+'|'.join(f"{c['mean_seconds']*1000:.3f}" for c in cells)+'|'+'|'.join(f"{c['latency_reduction_percent']:.2f}%" for c in comps)+'|')
        costs=[[c for c in summary['cost_cells'] if c['method']==m and c['target_input_tokens']==n] for m in methods]
        mem.append(f'|{n}|'+ '|'.join(f"{max(c['all_cost_peak_'+k+'_gb'] for c in rows):.3f}" for k in ['allocated','reserved'] for rows in costs)+'|')
        cis.append(f'|{n}|'+ '|'.join('['+', '.join(f'{x:.4f}' for x in c['bootstrap_ratio_95_percentile_interval'])+']' for c in comps)+'|')
    dtcost=[c for c in summary['cost_cells'] if c['method']==methods[0]]
    cold=['|轮次|目标输入|首个完整调用 / s|第二预热 / s|模型加载 / s|控制器初始化 / s|主机 RSS 高水位 / GB|','|---:|---:|---:|---:|---:|---:|---:|']
    for c in sorted(dtcost,key=lambda c:(c['round'],c['target_input_tokens'])):
        cold.append('|'+ '|'.join([str(c['round']),str(c['target_input_tokens'])]+[f'{c[k]:.3f}' for k in ['first_geometry_seconds','second_warm_seconds','model_load_seconds','controller_setup_seconds','host_process_lifetime_RSS_highwater_gb']])+'|')
    text='''# Qwen3 B1：完整热调用与固定几何 GPU 容量验收

V4 已通过冻结的四档速度和显存门槛。当前显式入口是 [make_memory_efficient_qwen3](../../../deltatrace/accelerated/memory_efficient_qwen3.py)，[用法与运行条件](../../../deltatrace/accelerated/MEMORY_EFFICIENT_QWEN3.md)。最终只维护[一张 rollout 曲线](../../../experiments/efficiency/qwen3_rollout_20260911/README.md)，图的 PNG、SVG、PDF 是同一内容。

## 四档同输入对照

Qwen3-8B FP16、MetaX C550 64GiB、样本 B1／端点 B2。使用固定作者 exp1 的输入构造与同步完整 API 计时器；目标输出32加EOS，作者数据文件缺失时原样沿用 `RULER fallback text.`。两轮反向长度顺序，每种方法、每个长度、每轮各自启动全新进程，每格2次预热、3次正式测量；方法顺序交替，磁盘编译缓存按方法独立。下表合并全部六次正式测量。

'''+ '\n'.join(table)+ '\n\n'+ '\n'.join(cis)+'''

门槛要求 DT 对两种固定 FT 的两轮轮均值、合并均值均更低，且独立重采样20,000次的95% percentile bootstrap比值区间上界小于1（固定种子20260910）。显存门槛同时比较 allocated 和 reserved，覆盖模型加载、初始化以及两轮所有冷／热调用峰值。没有按测量结果删行或放宽门槛。完整[独立复算结果](memory_v4_confirmation_verification.json)、[统计和成本](memory_v4_confirmation_local_summary.json)及[逐条成本索引](all_study_timing_rows.csv)可复核。

'''+ '\n'.join(mem)+'''

GB 为十进制。热调用 allocated 不含图池内部临时分配，因此不能单独用于容量结论。上述范围是固定输出32的四档独立几何容量，长 rollout 与形状切换成本另存；不推断其他设备、任意文本或任意长度的保证。

## 冷启动与完整开销

每次热调用仍包含当前输入刷新、原模型根前向、有限传播与原生 decoder 重算、全部状态与数值检查、完整向量和诊断返回 CPU、返回容器处理。第一调用额外包含两次完整程序预热、一次图记录及一次重放，所有开销均在外层原同步计时器内。

'''+ '\n'.join(cold)+f'''

正式验收共 {summary['all_timed_rows_count']} 条计时（{summary['warm_rows_count']} 预热、{summary['measured_rows_count']} 测量）、24次输出审计和8次原 retained 对照，共152次完整 API 调用。计时行合计 {summary['timed_seconds']:.3f}s，额外对照／审计 {summary['audit_seconds']:.3f}s；模型加载、初始化、进程总耗时与 host 子阶段另存。各阶段可能与排队 GPU 工作重叠，不重复相加。`cold_amortization` 是基于观测热耗时的条件估计，不是测量保证。

DT 与两种 FT 均使用同一作者 CPU 加载器后调用未修改的 `model.to('cuda:0')`；全部400个参数／buffer的完整字节、形状、步长、精度、设备和 tokenizer 特殊 ID 已与原 GPU 加载核对。加载临时 GPU 峰值降低，但主机 RSS 高水位约28GB，记录在每个进程中；没有主机内存更低的结论。见[加载身份核验](cpu_staged_loading_verification.json)。

## 数学与原生实现

保留原模型完整词表 head、有限 seed 完整形状、原生 GEMM 和 FP32 归一化边界。按生命周期释放中间引用，用未改动的公开 SAC 复用同一次原根前向中的原生投影输出。只缓存当前几何的程序和存储，跨归因不缓存激活、分数或结果。原829个严格谓词和36层统计保留；启用 SAC 时另执行36个同输入谓词。返回模板只含静态基础类型元数据，每次创建新的嵌套容器，并读取当前向量与144条动态标量路径。

有限 FA 使用固定厂商 FA2.5.3 的原生 MMA、原布局和原 K 循环，所属输入在 MMA 寄存器中复用，已失效的 A 共享缓冲供 B 使用；phase0 32×32/2warp、phase1/2 64×32/4warp。没有更改有限规则、原模型安装的 FA2.6.3、原生反向、FT 或原质量评测。编译与依赖、全部库哈希见[版本清单](../../../deltatrace/accelerated/memory_efficient_qwen3_sources.json)和[源码推导核验](memory_v4_source_verification.json)。

8个正式 DT 单元均与同进程原 retained 的完整向量和全部数学诊断精确相同。全部11条符合原≤1024总长条件的作者样本（MH0–7、NI0/3/6）完成冷、热、retained逐向量核对；更换同长度真实输入后同一张图输出改变并与原入口精确一致。8种实际GPU图故障控制、4种标量控制、3种RoPE控制、6种模型契约控制和恢复检查通过。见[作者兼容性核验](memory_v4_author_verification.json)。未重跑质量指标，不将数值兼容性称为新的质量提升。

原 FT Both 的部分完整矩阵返回含非有限元素；每格实际计数保存在 `cells.nonfinite_return_elements`。保留作者原计算和完整返回，不清洗分数，也不据此作其数值质量结论。

## 保留的失败与候选

|阶段|处理与证据|
|---|---|
|最初流式／紧凑图、V1混合几何容量测试|保留全向量、构图及形状切换成本；未同时通过全部速度与容量门槛。|
|合并cast、缩减目标seed|出现完整向量或边界变化，停止采纳；恢复原FP32边界、完整seed和GEMM。|
|V2公开版|速度门槛失败；128/256档包含模型加载的显存与FT相等，也未通过显存门槛。|
|V3公开版|最长档第一轮慢于原FT，合并95%比值区间跨1；共同加载峰值仍造成128/256档显存相等。全部24格保存。|
|owner64/2warp及拆分扫描|原线程归约树变化／不满足精确性或收益要求，未采纳。|
|三阶段寄存器预载、phase0 64/4warp|各100次算子控制五个输出逐位相等；后者未提供额外收益，没有升级phase0几何。|
|首次MMA寄存器编译|主机端不同CUTE静态整数类型导致 `std::max` 编译失败，0次算子调用；仅两处显式 `std::max<int>` 修正后独立重建。|
|最终MMA寄存器路径|100次五输出逐位相等，另15次整模型预试验全向量／诊断相同，之后才冻结并运行本次公开V4验收。|

历史程序、未采用候选、异常、原始输出和协议保留在[完整归档索引](study_inventory.json)；[文字元数据更正](protocol_metadata_corrections.md)单列，不改写已冻结源文件。

共核验 {inventory['archive_count']} 个归档、{inventory['unique_archived_files']} 个去重文件、{inventory['report_count']} 份报告与 {inventory['timing_rows_count']} 条原计时行，零排除；算子专用记录合计 {inventory['operator_only_calls']} 次。该计时行数量／耗时不等于全部研究GPU执行或总算力，额外审计、编译与加载保留在各自报告中。结束时 {post['identity_checks']} 项检查点、原模型、原FA／FT／SAC、当前源码／库身份检查通过，GPU无进程，见[结束身份核验](memory_v4_postflight_verification.json)。

## 本地复核

在安装 NumPy 的 Python 环境中运行 `python build_study_inventory.py --extract /path/to/extracted`，以逐字节校验并展开已归档材料。再使用 `summarize_memory_v4_confirmation.py --raw /path/to/extracted --protocol memory_v4_confirmation_protocol.json --output-prefix memory_v4_confirmation_local_summary`、`verify_memory_v4_confirmation.py`、`verify_memory_v4_author.py --raw /path/to/extracted` 与 `verify_memory_v4_postflight.py`。绘图入口为 `experiments/efficiency/qwen3_rollout_20260911/build_curve.py`。上述复核与绘图不会运行模型。
'''
    (S/'README.md').write_text(text,newline='\n')
    curve_readme='''# Attribution cost versus rollout length

![Updated DeltaTrace and FlashTrace with released reference methods.](figures/deltatrace-rollout-scaling.png)

The single figure replaces DT and both FT curves with measurements from the current public v4 runtime and the unchanged author exp1 FT entry points. Qwen3-8B FP16, one C550, nominal 10-token input; rollout targets are 10, 100, 500, 1,000, 2,000, 5,000 and 10,000 tokens. Every cell has its own process and identical model input IDs across all three methods. Lines show the mean of three complete synchronized API calls after two separately retained warm calls. Error bars show the observed range. Model loading, construction, cold calls, GPU allocated/reserved peaks and host memory remain in the data.

IG, IG × Attention, Perturbation, REAGENT, IFR, CLP and AttnLRP retain 35 successful points from the [fixed FlashTrace exp1 release](https://github.com/wbopan/flashtrace/tree/075e7e44ae4d5acd2ed76e0d2aced57107d02736/exp/exp1). These historical results use six/eight devices and are labeled `[ref]`; the overlay does not establish cross-hardware speedup ratios. Missing or failed cells have no inferred latency and split lines. The original FT rows remain in the source files but are replaced in the plotted series.

The measured DT/FT lines cross at 1,000 and 2,000 output tokens. The separately passed short-input speed and GPU-memory gate fixes output length at 32 tokens; it does not establish that DT is faster or uses less memory at every rollout length.

[Plotted JSON](curve_data.json), [CSV](curve_data.csv), [verification](verification.json), and [speed/memory scope and complete evidence](../../../research/temporary/qwen3_stream_memory_20260910/README.md) preserve the experiment details. Rebuild with `python experiments/efficiency/qwen3_rollout_20260911/build_curve.py` (NumPy and Matplotlib). PNG, SVG and PDF are three formats of this one figure; the manuscript is unchanged.
'''
    (E/'README.md').write_text(curve_readme,newline='\n');print({'status':'written','study_readme':str(S/'README.md'),'curve_readme':str(E/'README.md')})

if __name__=='__main__':main()

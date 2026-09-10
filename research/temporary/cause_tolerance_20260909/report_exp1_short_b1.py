"""Write a reviewable report from the author-exp1 raw-data summary."""
import argparse
import json
from pathlib import Path


LABELS={'deltatrace_retained':'DT retained','ifr_multi_hop_both':'FT Both','ifr_multi_hop':'FT multi-hop',
 'ifr_all_positions':'IFR all positions','attnlrp':'AttnLRP','IG':'IG (20 steps)',
 'attention_I_G':'Attention × IG','perturbation_all':'Perturbation (log loss)',
 'perturbation_CLP':'CLP (KL)','perturbation_REAGENT':'REAGENT (MLM)'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--summary',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();d=json.loads(a.summary.read_bytes())
    assert len(d['warm_comparisons'])==16
    lookup={(c['family'],c['method'],c['target_input_tokens'],c['phase']):c for c in d['cells']}
    lines=['# 作者 exp1：单样本、短输入、多方法效率测试','',
      '优化前的 retained DT 在这次两模型、四档短输入测试中均慢于 FT。相同输入下，DT 的峰值已分配显存较低；速度与显存结论分别报告。本报告只记录加速前基线；后续实现和验收见 [监听开销优化](CODE_LOCAL_CAPTURE.md)。没有进行归因质量、Qwen自由回答生成或 RISE/MAS 测试。','',
      '用户随后要求专心做效率优化，剩余慢基线已立即停止：Qwen3.5 CLP 保留中途记录，两模型 REAGENT 的依赖恢复后运行未启动。DT/FT 核心对比已经完整完成，下面的其他方法表明确保留未完成单元。','',
      '## 预热后的完整调用','',
      '每格为额外三次预热调用的均值，单位 ms；使用作者同步 CUDA 的 `measure`，计入完整归因计算及返回对象销毁，模型加载和归因器构造在计时外。原三次调用与这三次额外调用均执行作者的显存清理。','',
      '| 模型 | 目标输入档 | 实际总序列 | DT | FT Both | FT multi-hop | DT 相对 FT Both |',
      '|---|---:|---:|---:|---:|---:|---:|']
    for family in ['qwen3','qwen35']:
        for n in [128,256,512,1024]:
            cells=[lookup[family,m,n,'warm3'] for m in ['deltatrace_retained','ifr_multi_hop_both','ifr_multi_hop']]
            t,b,f=cells;ratio=t['seconds_mean']/b['seconds_mean']
            lines.append(f"| {'Qwen3-8B' if family=='qwen3' else 'Qwen3.5-9B'} | {n} | {t['actual_total_tokens']} | {1000*t['seconds_mean']:.1f} | {1000*b['seconds_mean']:.1f} | {1000*f['seconds_mean']:.1f} | {(ratio-1)*100:+.1f}% |")
    lines += ['', '![预热调用耗时](exp1_short_b1_warm.png)','',
      '## 作者原三次重复：已完成与中断记录','',
      '下表保留作者原始三次重复的均值 ± 总体标准差，单位秒。包含每个方法进程的首次调用，因此最短档可能包含编译/延迟初始化成本，不能拿这一表的首次 DT 均值当成稳定延迟。`OOM` 表示按作者默认参数显存不足；失败格不填伪造耗时，也不临时改变步数或内部批量。','']
    for family,title in [('qwen3','Qwen3-8B / FP16'),('qwen35','Qwen3.5-9B / BF16')]:
        lines += [f'### {title}','','| 方法 | 128 | 256 | 512 | 1024 |','|---|---:|---:|---:|---:|']
        for method,label in LABELS.items():
            values=[]
            for n in [128,256,512,1024]:
                c=lookup.get((family,method,n,'original3'))
                if c is None:values.append('未完成');continue
                if c['valid_timing']:values.append(f"{c['seconds_mean']:.3f} ± {c['seconds_std']:.3f}")
                else:
                    status='/'.join(k.upper() if k=='oom' else k for k in c['statuses'])
                    values.append(status if c['n_ok']==0 else f"{c['n_ok']}/{c['n_runs']} 成功；输出核验={c['audit_finite']}")
            lines.append('| '+label+' | '+' | '.join(values)+' |')
        lines.append('')
    lines += ['![全部方法原三次重复](exp1_short_b1_all_methods.png)','',
      '## 显存与首次调用','',
      '显存单位为作者使用的十进制 GB（10⁹ bytes），峰值包含模型权重。下表是预热三次调用中的最大已分配显存；保留显存另存 CSV。','',
      '| 模型 | 目标输入档 | DT GB | FT Both GB | DT 减少 GB |','|---|---:|---:|---:|---:|']
    for family in ['qwen3','qwen35']:
        for n in [128,256,512,1024]:
            t=lookup[family,'deltatrace_retained',n,'warm3'];f=lookup[family,'ifr_multi_hop_both',n,'warm3']
            lines.append(f"| {family} | {n} | {t['peak_allocated_gb']:.3f} | {f['peak_allocated_gb']:.3f} | {f['peak_allocated_gb']-t['peak_allocated_gb']:.3f} |")
    lines += ['', '![峰值已分配显存](exp1_short_b1_memory.png)','',
      '| 模型 | 方法 | 模型加载秒 | 单独原生初始化秒 | 进程内首次调用秒 |','|---|---|---:|---:|---:|']
    for r in d['first_calls_and_loading']:
        if r['method'] not in ['deltatrace_retained','ifr_multi_hop_both','ifr_multi_hop']:continue
        init=sum(x['seconds'] for x in r['separate_initialization'] or [])
        lines.append(f"| {r['family']} | {LABELS[r['method']]} | {r['model_load_seconds']:.3f} | {init:.3f} | {r['first_call_seconds']:.3f} |")
    lines += ['',
      '这里的“首次”指该方法进程内首次调用。同模型各方法共享磁盘上的原生/Inductor 编译缓存，且 Qwen3.5 DT 有单独记账的原生 eager 延迟初始化；这不是各方法各自清空磁盘缓存的冷启动竞赛。每次归因器构造耗时在原始行的 `runner_init_seconds` 中完整保留。','',
      '## 协议、实际输入与边界','',
      '- 复用作者提交 `075e7e44ae4d5acd2ed76e0d2aced57107d02736` 的 `exp/exp1/run_time_curve.py`，SHA256 为 `e073abadeb20df0fedcfd55acd0061489d0f51813bc8c1486e28ca42c2b28951`。输入/目标构造、计时、清理、IG 20 步和 FT chunk 128 / sink 32 来自原入口；IFR all positions 使用原 sink chunk 1。',
      '- 作者默认八种方法，加上同一原入口已经支持的 FT Both 和当前 DT retained。每种方法单独加载模型、串行使用同一 MetaX C550；这是本次显式扩展。Qwen3 使用原入口；Qwen3.5 使用固定官方扩展 `e81b3be50a48dcfc652fbf1b530069b552736e66` 的对应类。',
      '- 样本批量为 1；DT 的两个端点组成内部 B2。IG 沿积分路径的内部批量仍按作者公式计算并由 IG 上限 20 截断，不能等同于 20 个独立样本。',
      '- 目标输入档为 128、256、512、1024，输出固定为作者 32-token 构造再加 EOS。原构造会 decode/re-tokenize，档名不是实际模型 token 数。实际完整序列均不超过 1k，见下表。',
      '- 默认 RULER 文件缺失，使用作者原代码内置的 `RULER fallback text. `。目标也使用作者固定合成文本。两模型和全部方法保存完整 token IDs；同模型各方法的输入/目标哈希相同。该合成短输入实验不代表真实任务上的归因质量或其他输出长度的速度。','',
      '| 模型 | 目标输入档 | 实际用户输入 | 格式化 prompt | 目标含 EOS | 实际完整序列 |',
      '|---|---:|---:|---:|---:|---:|']
    for r in d['actual_lengths']:
        lines.append(f"| {r['family']} | {r['target_input_tokens']} | {r['user_prompt_tokens']} | {r['formatted_prompt_tokens']} | {r['generation_tokens']} | {r['total_tokens']} |")
    lines += ['',
      '每个成功格另有一次单独记账的输出/原生根输入观察调用；计时调用不挂这些观察器。DT、两种FT及IFR all positions的原生根输入与预期token IDs核对；其他方法保存能观察到的input_ids调用。IG可经inputs_embeds执行，现有观察器不捕获该输入，不能把构造器token一致性说成验证了每次插值/扰动的原生输入。DT 要求原始向量有限；FT、IG、LRP 原矩阵的 NaN 包含作者刻意保留的可视化占位，核验使用作者公开归一化的 NaN→0、非负截断及逐行归一化语义，原 NPZ 完整保留。Infinity 不作为合法占位。','',
      '## 环境恢复与失败记录','',
      '本轮服务器重建后恢复既有字节固定的模型环境、有限核和官方源文件。最初两次 Qwen3.5 FT 运行缺少先前已使用的 FLA 原生兼容文件，后续补齐并在模型加载前增加来源哈希门槛。三个扰动方法最初缺少其原始 Longformer 依赖；随后从官方 `allenai/longformer-base-4096` 固定 revision `301e6a42cb0d9976a6d6a26a079fef81c18aa895` 下载、验证并上传真实权重，再运行原入口。原始失败记录仍保留，只有这八次环境不完整尝试从方法对比中显式排除。','']
    for r in d['excluded_attempts']:
        lines.append(f"- `{r['path']}`：{r['failed_rows']}/{r['rows']} 失败；{r['reason']}")
    lines += ['',
      '恢复后仍出现的 OOM/执行错误是本次原参数可用性结果，不从数据里删去，也没有修补官方算法来制造成功。Qwen3.5 CLP 的中断属于用户转向优化的范围变更，不记作算法失败；REAGENT 未补跑。完整报错见各方法 `results.json` 与日志。','',
      '复算入口为 `summarize_exp1_short_b1.py`、`verify_exp1_short_b1.py`、`report_exp1_short_b1.py`、`plot_exp1_short_b1.py`。原始报告、输入、逐次 CSV/JSONL、输出向量、源文件及执行队列位于 [exp1_short_b1_raw](exp1_short_b1_raw/)；[机器可读汇总](exp1_short_b1_summary.json)、[全部单元 CSV](exp1_short_b1_cells.csv)和[核验结果](exp1_short_b1_verification.json)可独立检查。',
      '']
    cost=d['recorded_selected_execution_cost']
    lines += ['## 本轮记录的计算成本','',
      f"选定尝试包含 {cost['successful_timed_calls']} 次成功计时调用、{cost['failed_timed_calls']} 次失败调用，另有 {cost['separate_audit_calls']} 次输出观察调用。成功计时调用总计 {cost['successful_timed_call_seconds']:.3f} 秒，额外观察 {cost['separate_audit_call_seconds']:.3f} 秒，模型加载 {cost['model_load_seconds']:.3f} 秒，单独原生初始化 {cost['separate_native_initialization_seconds']:.3f} 秒，计时调用之前的归因器构造 {cost['timed_runner_construction_seconds']:.3f} 秒。",'',
      '以上各项单列，不称为整个实验的总墙钟时间：失败调用耗时、导入、GC及缓存清理不在这些和数中；被排除的环境尝试也另记。完整队列的进程墙钟时长保留在原始 queue 文件中。','']
    a.output.write_text('\n'.join(lines),encoding='utf8')


if __name__=='__main__':main()

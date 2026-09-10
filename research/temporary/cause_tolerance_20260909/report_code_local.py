"""Render the measured optimization outcome without mixing pilot and release costs."""
import argparse,json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--summary',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();d=json.loads(a.summary.read_bytes());assert d['status']=='verified'
    lines=['# Qwen3.5：减少短输入 B1 的被动监听开销','',
      '已实现显式入口 `make_code_local_qwen35`。在同一进程、同一输入、同一完整调用范围下，前三个短输入档比 retained 版减少约 20%–37% 延迟；最长档增加 0.47%，没有验证出收益。四档预热峰值显存相同。四个合成输入及原开发集中全部 11 个总长度不超过 1k 的样本，完整归因向量和原有逐层诊断均完全一致。','',
      '用户要求专心效率优化后，剩余慢基线已经停止，未恢复 REAGENT 或追加归因质量评测。[此前的作者 exp1 对比](EXP1_SHORT_B1.md)单独保存；本报告只评估这次实际代码改动。','',
      '## 正式入口的完整调用延迟','',
      '使用固定作者 exp1 输入/目标构造与原 `measure`：每档每版先完整预热一次，再按 retained/local/local/retained/retained/local 交错执行，每版三次，取均值。计入归因计算、完整诊断和返回对象销毁；模型加载、归因器构造单列。没有移除异常重复或混入 profiler 调用。','',
      '| 目标输入档 | 实际总 tokens | retained ms | 新入口 ms | 延迟变化 | 两版预热峰值 GB |',
      '|---|---:|---:|---:|---:|---:|']
    for x in d['production_comparisons']:
        lines.append(f"| {x['target_input_tokens']} | {x['actual_total_tokens']} | {1000*x['retained_seconds']:.1f} | {1000*x['local_seconds']:.1f} | {x['latency_change_percent']:+.2f}% | {x['local_peak_allocated_gb']:.3f} |")
    lines += ['', '![完整调用对比](code_local_latency.png)','',
      '单样本 B1 的两个端点在内部组成 B2。输出是作者固定 32-token 构造加 EOS，RULER 数据缺失时沿用作者内置回退文本。横轴实际总长度包含格式化 prompt、完整目标及 EOS。三次样本不足以声称接近零的差异有统计意义；最长档应视为没有稳定收益。','',
      '本轮新入口仍没有建立快于 FT 的结果。此前同一设备、输入构造下 FT Both 的四档预热均值约为 228/286/341/551 ms；这是另一个进程的基线记录，不能把它与本轮的配对加速倍率合并成新的 FT 胜负实验。Qwen3 的监听试验均值未改善，未采用该候选。','',
      '## 改动及其依据','',
      '原 GDN 捕获器使用 `sys.setprofile` 接收整个线程的 Python/C 事件，并在确认事件属于所需代码之前读取 `frame.f_locals`。短序列中，原生 FLA/Triton 执行伴随大量无关回调。最短输入的已有控制器诊断中，原生 replay 约占 0.23 秒；GPU 算子耗时不能解释同量级的 CPU 开销。','',
      '新捕获器使用 CPython 公开的代码局部 monitoring 接口，仅向原捕获回调传递指定原生函数的 PY_START/PY_RETURN 事件。原函数仍由模型正常调用，模型/FA/FLA/反向代码没有替换；张量捕获、数学规则、检查和诊断保持原实现。控制器只替换一条捕获器 import，派生核验要求其余字节一致。官方接口说明：[Python sys.monitoring](https://docs.python.org/3.12/library/sys.monitoring.html)、[PEP 669](https://peps.python.org/pep-0669/)。','',
      '最初候选在 128 档试验里约减少 42% 延迟，但扩展长度后发现 context→bound callback→capture→context 的循环引用，临时张量延迟释放；904-token 档显存由 20.503 增至 22.703 GB，速度收益消失。正式实现退出时解除回调引用；CPU weakref 检查确认无需等待 GC 就能释放捕获器，GPU 四档复验确认额外显存已消除。早期候选、循环引用问题和全部结果均保留，没有用最好的早期百分比替代正式入口结果。','',
      '长序列下原生 GPU 工作占比上升，去除主机监听开销未必降低完整调用延迟。本轮没有按输入长度切换规则、删去模型层或另造计算内核。','',
      '## 兼容性、入口与复现','',
      '- `deltatrace/accelerated/code_local_qwen35.py::make_code_local_qwen35(root, model, finite_fa, finite_fla)` 返回与 retained 入口相同接口的 runner 和来源记录。调用方显式选择；既有干净、retained 和固定正式评测入口的行为保持。',
      '- 要求 CPython 3.12 或更新版本及已有固定 Qwen3.5 运行环境。捕获上下文单次使用，只处理创建线程事件；已有 `sys.setprofile` 观察器或无空闲 monitoring tool ID 时明确报错。',
      '- `code_local_qwen35_sources.json` 链接并核验 retained/deferred/clean 来源；新控制器派生、生产运行实际文件及发布文件哈希一一核对。',
      '- 四个合成格各有单独记账的完整输出核验，实际根输入同时检查 EOS 基线和真实端点；完整向量、逐层捕获次数及数值诊断均相同。',
      '- 原始 NI0–7/MH0–7 中全部符合总长度 ≤1024 的 11 例都经原 `attribute_batch` B1 检查，完整冻结目标不截断。它们是兼容性调用，含新形状编译，不能作额外预热速度证据；5 个较长 NI 例因当前长度范围排除。没有重做生成或 RISE/MAS。','',
      '| 原始样本 | 实际总 tokens | 完整向量 |','|---|---:|---|']
    for c in d['author_short_cases']:lines.append(f"| {c['key']} | {c['total_tokens']} | 完全相同 |")
    lines += ['',
      '测试输入上的逐元素一致不等于对全部输入的数学证明，也不构成归因质量改进。Qwen3 同一思路的初始试验 retained 294.55 ms、新监听 300.08 ms；完整向量相同但均值慢 1.88%，没有采用或挑选其中最快重复。','',
      '复算与核验：`summarize_code_local.py --raw code_local_raw --repo <repository> --output <output>`；报告生成 `report_code_local.py`，绘图 `plot_code_local.py`。相对路径以运行位置为准。完整实验驱动、协议、首次调用、逐次计时、捕获诊断、失败导入与 NPZ 均在 [code_local_raw](code_local_raw/)；机器可读结果见 [code_local_summary.json](code_local_summary.json) 和 [code_local_cells.csv](code_local_cells.csv)。','',
      '## 本次优化研究成本','']
    c=d['recorded_cost']
    lines += [f"合计 {c['timed_full_calls']} 次完整计时/剖析调用（{c['timed_full_call_seconds']:.3f} 秒），{c['separate_audit_calls']} 次单独输出核验（{c['separate_audit_seconds']:.3f} 秒），{c['author_compatibility_calls']} 次真实作者短样本兼容性调用（{c['author_compatibility_seconds']:.3f} 秒）。模型加载 {c['model_load_seconds']:.3f} 秒、原生 eager 初始化 {c['eager_initialization_seconds']:.3f} 秒另列。首次两次 profile 导入失败未进行模型调用，日志保留。",'',
      '这些成本含试验、首次编译和剖析，不是生产延迟；未计入 Python 导入、GC、下载及文件校验等整个任务墙钟开销。优化阶段未运行 FT、自由生成或质量指标。最终 68 项权重/既有发布文件/新源文件检查通过，GPU 已无运行任务。','']
    a.output.write_text('\n'.join(lines),encoding='utf8')


if __name__=='__main__':main()

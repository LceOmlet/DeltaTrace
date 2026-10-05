"""Bind actual primitive observations to the existing degradation record."""
import hashlib
import json
from pathlib import Path
import time

from analyze_textcraft_native_readout import source
from stage_textcraft_conditional_primitives import LOCAL


REPO = Path(__file__).resolve().parents[3]


if __name__ == '__main__':
    report_path = REPO/'experiments/rl/results_textcraft_learning_degradation_20261006.json'
    report = json.loads(report_path.read_bytes())
    analysis_path = LOCAL/'conditional-primitive-analysis.json'
    analysis = json.loads(analysis_path.read_bytes())
    execution = json.loads((LOCAL/'completed.json').read_bytes())
    job = json.loads((LOCAL/'job.json').read_bytes())
    assert analysis['coverage']['unique_probes'] == 42
    assert execution['finite_trace_calls'] == 14
    assert execution['backward_calls'] == execution['optimizer_steps'] == execution['scheduler_steps'] == 0
    paths = list(LOCAL.glob('*.json')) + list(LOCAL.glob('*.stdout.txt')) + [
        LOCAL/'effective-config.yaml', LOCAL/'diagnostic.log', LOCAL/'physical-before-start.txt']
    paths.extend((LOCAL/'submitted-cpu-source').glob('*'))
    paths.extend(Path(__file__).with_name(name) for name in (
        'observe_textcraft_conditional_primitives.py', 'verify_textcraft_conditional_primitives.py',
        'stage_textcraft_conditional_primitives.py', 'observe_textcraft_conditional_primitive_job.py',
        'analyze_textcraft_conditional_primitives.py', Path(__file__).name))
    recorded = {item['path']: item for item in report['sources']}
    for path in paths:
        item = source(path)
        item['path'] = path.relative_to(REPO).as_posix()
        if item['path'] in recorded:
            recorded[item['path']].update(item)
        else:
            report['sources'].append(item)
            recorded[item['path']] = item
    section = dict(scope=analysis['scope'], analysis=source(analysis_path),
        coverage=analysis['coverage'], layer_summaries=analysis['layer_summaries'],
        execution=dict(pid=job['pid'], pid_birth=job['pid_birth'], devices=job['devices'],
            started_unix=job['started_unix'], completed_unix=execution['completed_unix'],
            wall_seconds=execution['completed_unix']-job['started_unix'],
            full_finite_calls=14, single_root_calls=28, backward_calls=0,
            optimizer_steps=0, scheduler_steps=0),
        provenance=dict(prepared=source(LOCAL/'prepared.json'), owner=source(LOCAL/'native-owner-inspection.json'),
            static_review=source(LOCAL/'independent-source-review.json'),
            independent_raw_review=source(LOCAL/'independent-primitive-review.json')),
        limitations=analysis['limitations'],
        production_state='No production code, PPO, Q/V/A, training configuration or checkpoint has been changed. TextCraft remains stopped.')
    report['conditional_primitives'] = section
    compact = json.loads((LOCAL/'conditional-primitive-compact-summary.json').read_bytes())
    section['compact_summary'] = source(LOCAL/'conditional-primitive-compact-summary.json')
    section['findings'] = [
        'For each selected layer, mixer is the largest individual six-term mean-absolute conditional residual; MLP is second. These are selected-layer diagnostic magnitudes, not percentages of total training degradation.',
        'Norm CPU/GPU output differences are much smaller than its separately measured storage and joint-direction terms. The current observations do not justify attributing the main lower-layer loss solely to norm rounding.',
        'All312 actual residual-add BF16 outputs match CPU same-dtype addition exactly in the independent raw review; their storage contribution is directly observed rather than presumed.',
        'Norm single-pair FP64 identities agree to raw floating arithmetic residuals. This is descriptive original-owner recomputation, not an invented acceptance tolerance or proof that joint coefficients predict every single-token intervention.',
        'MLP and mixer residuals remain mixed conditional approximation/numerical effects. No FA/FLA kernel failure has been established, and no correction has been added.']
    section['compact_layer_data'] = compact['layers']
    report['updated_unix'] = time.time()
    report['status'] = 'actual_lower_decoder_primitive_residuals_measured_no_production_repair'
    report['interpretation']['next_owner_check'] = (
        'Follow the measured native mixer/MLP residuals only where their actual size warrants it. '
        'Keep norm storage/backend and joint conditional terms separate. No credit correction, '
        'new tolerance, entropy tuning or formal TextCraft restart is justified by arithmetic closure.')
    for item in report['sources']:
        assert hashlib.sha256((REPO/item['path']).read_bytes()).hexdigest() == item['sha256'], item['path']
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    runtime = REPO/'experiments/rl/RUNTIME_RECORD.md'
    raw = runtime.read_bytes()
    heading = '## 2026-10-06 下层真实子块：有限传播与原生存储分项'
    entry = (heading+'\n\n'+
        f"隔离诊断PID{job['pid']}/birth{job['pid_birth']}已完成，GPU4/5，"
        f"{section['execution']['wall_seconds']:.3f}秒。复用原7个B4/rank、checkpoint25、"
        'LoRA8/16、原runner/producer/目标/切点；14 full finite、28 single root，'
        'backward/optimizer/scheduler均0。层6/8(GDN)、11(FA)共156 transport投影，'
        '按原映射保留后为126 unique probe-layer，即42 probe×3层。\n\n'+
        '只观察原decoder有限系数、原NativeDecoderCapture与mixer模块输出；未启用public '
        'observer/diagnostics，不改prefix/replay路径。六项闭合仅验证观测算术；norm用原HF/finite '
        'CPU调用分开联合条件、dtype存储与GPU/CPU差。MLP/mixer尚未拆开，不能声明核超差。\n\n'+
        'CPU准备的源名映射、漏传既有helper和序列化来源字段问题均在GPU前修复，失败stdout保留。'
        '本次原始ranks、source SHA、独立复核及数值汇总见'
        'results_textcraft_learning_degradation_20261006.json的conditional_primitives；'
        '不是已修复学习信号，TextCraft不恢复。\n\n').encode('utf-8')
    if heading.encode('utf-8') not in raw:
        title, tail = raw.split(b'\n', 1)
        runtime.write_bytes(title+b'\n\n'+entry+tail)
    print(json.dumps(dict(report=source(report_path), runtime_record=source(runtime),
        recorded_sources=len(report['sources']), status=report['status']), ensure_ascii=False))

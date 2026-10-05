"""Bind the actual GDN observation to the existing degradation/version record."""
import hashlib
import json
from pathlib import Path
import time

from analyze_textcraft_native_readout import source
from stage_textcraft_conditional_gdn import LOCAL


REPO = Path(__file__).resolve().parents[3]


if __name__ == '__main__':
    report_path = REPO / 'experiments/rl/results_textcraft_learning_degradation_20261006.json'
    report = json.loads(report_path.read_bytes())
    analysis_path = LOCAL / 'conditional-gdn-analysis.json'
    analysis = json.loads(analysis_path.read_bytes())
    completed = json.loads((LOCAL / 'completed.json').read_bytes())
    job = json.loads((LOCAL / 'job.json').read_bytes())
    review = json.loads((LOCAL / 'independent-gdn-review.json').read_bytes())
    assert analysis['coverage']['unique_probes'] == 42
    assert analysis['coverage']['transport_probe_layer_records'] == 104
    assert completed['finite_trace_calls'] == 14
    assert completed['backward_calls'] == completed['optimizer_steps'] == completed['scheduler_steps'] == 0
    paths = list(LOCAL.glob('*.json')) + list(LOCAL.glob('*.stdout.txt')) + [
        LOCAL / 'effective-config.yaml', LOCAL / 'diagnostic.log', LOCAL / 'physical-before-start.txt',
        LOCAL / 'independent_raw_gdn_review.py']
    paths.extend(Path(__file__).with_name(name) for name in (
        'observe_textcraft_conditional_gdn.py', 'verify_textcraft_conditional_gdn.py',
        'stage_textcraft_conditional_gdn.py', 'observe_textcraft_conditional_gdn_job.py',
        'analyze_textcraft_conditional_gdn.py', Path(__file__).name))
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
        capture_start_groups=analysis['capture_start_groups'],
        all_full_trace_capture_starts=[dict(rank=row['rank'],
            original_owner_batch_index=row['original_owner_batch_index'],
            starts={index:layer['capture_start'] for index,layer in row['metadata']['layers'].items()})
            for row in analysis['full_finite_GDN_metadata']],
        actual_dtypes=analysis['observed_tensor_dtypes'],
        execution=dict(pid=job['pid'], pid_birth=job['pid_birth'], devices=job['devices'],
            started_unix=job['started_unix'], completed_unix=completed['completed_unix'],
            wall_seconds=completed['completed_unix']-job['started_unix'],
            full_finite_calls=14, single_root_calls=28, backward_calls=0,
            optimizer_steps=0, scheduler_steps=0),
        provenance=dict(prepared=source(LOCAL / 'prepared.json'),
            owner=source(LOCAL / 'native-owner-inspection.json'),
            independent_raw_review=source(LOCAL / 'independent-gdn-review.json')),
        limitations=analysis['limitations'],
        production_state='Passive actual-original observation only. No production code, model weights, PPO, Q/V/A or training configuration changed. TextCraft remains stopped.')
    section['findings'] = [
        'At both measured GDN layers, remaining_input_FLA has the largest individual mean-absolute residual among the three observed terms. It contains input projections, convolution, scalar rules, FLA and storage; it is not an isolated FLA kernel failure.',
        'Output projection is smaller than both norm_gate and the remaining chain on this selected 42-probe sample. The actual compact capture_start is read from the original pullback frame and native norm operands are aligned to the same suffix.',
        'All14 full traces contain capture_start values0/64/192/256. The52 projected transport probes contain0/64 only because the other two groups have no selected first-response slots; no cutoff or denominator has been changed.',
        'Three terms close to the same-run primitive mixer residual using only original coefficients and native root operands. This validates observation composition, not credit accuracy, official tolerance or a learning repair.']
    report['conditional_gdn'] = section
    report['updated_unix'] = time.time()
    report['status'] = 'actual_GDN_remaining_input_chain_localized_no_production_repair'
    report['interpretation']['next_owner_check'] = (
        'Follow the measured remaining GDN input/FLA chain; separate native operand/cast/storage differences '
        'from joint conditional allocation using the original owner seams. No credit correction, normalization, '
        'entropy change, tolerance change or formal TextCraft restart follows from arithmetic closure alone.')
    for item in report['sources']:
        assert hashlib.sha256((REPO / item['path']).read_bytes()).hexdigest() == item['sha256'], item['path']
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    runtime = REPO / 'experiments/rl/RUNTIME_RECORD.md'
    raw = runtime.read_bytes()
    heading = '## 2026-10-06 GDN 残差定位：原生三段观测'
    entry = (heading+'\n\n'+
        f"隔离PID{job['pid']}/birth{job['pid_birth']}已完成，GPU4/5，"
        f"{section['execution']['wall_seconds']:.3f}秒。仍为原7个B4/rank、checkpoint25、"
        'LoRA8/16、原runner/producer/目标/切点；14 full finite、28 single root，'
        'backward/optimizer/scheduler均0。层6/8共104 transport-layer，按原身份保留重复后'
        '84 unique probe-layer，即42 probe×2层。\n\n'+
        '只读取原norm-gate有限系数及paired8原生输入/输出。compact capture_start来自原'
        'gdn_finite_pullback实际frame；只内侧o/z/gated按同一suffix裁切，外侧mixer保持全长。'
        '不启用public observer/diagnostics，不保存额外递推状态，不增加模型调用或修正倍率。'
        '两层三项中remaining_input_FLA的平均绝对残差最大，但它仍包含投影、卷积、'
        'FLA与存储，不能冒称FLA kernel超官方容差。\n\n'+
        '来源、实际dtype、CPU绑定、完整原始ranks、重复probe统计及独立复核见'
        'results_textcraft_learning_degradation_20261006.json的conditional_gdn；'
        '这是定位进展，不是已修复学习信号，TextCraft不恢复。\n\n').encode('utf-8')
    if heading.encode('utf-8') not in raw:
        title, tail = raw.split(b'\n', 1)
        runtime.write_bytes(title+b'\n\n'+entry+tail)
    print(json.dumps(dict(report=source(report_path), runtime_record=source(runtime),
        recorded_sources=len(report['sources']), status=report['status']), ensure_ascii=False))

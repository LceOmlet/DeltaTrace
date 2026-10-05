"""Record passive actual-FLA observations without changing the fixed method."""
import hashlib
import json
from pathlib import Path
import time

from analyze_textcraft_native_readout import source
from stage_textcraft_conditional_fla import LOCAL


REPO = Path(__file__).resolve().parents[3]
OWNER = Path(__file__).parent / 'textcraft-degradation-20261005/native-layout-20261006/v1/owner-contract'


if __name__ == '__main__':
    path = REPO / 'experiments/rl/results_textcraft_learning_degradation_20261006.json'
    report = json.loads(path.read_bytes())
    analysis_path = LOCAL / 'conditional-fla-analysis.json'
    analysis = json.loads(analysis_path.read_bytes())
    completed = json.loads((LOCAL / 'completed.json').read_bytes())
    job = json.loads((LOCAL / 'job.json').read_bytes())
    assert analysis['coverage']['unique_probes'] == 42
    assert analysis['coverage']['transport_probe_layer_records'] == 104
    assert completed['finite_trace_calls'] == 14
    assert completed['backward_calls'] == completed['optimizer_steps'] == completed['scheduler_steps'] == 0
    paths = list(LOCAL.glob('*.json')) + list(LOCAL.glob('*.stdout.txt')) + [
        LOCAL / 'effective-config.yaml', LOCAL / 'diagnostic.log', LOCAL / 'physical-before-start.txt',
        LOCAL / 'independent_raw_fla_review.py']
    paths.extend(Path(__file__).with_name(name) for name in (
        'observe_textcraft_conditional_fla.py', 'verify_textcraft_conditional_fla.py',
        'stage_textcraft_conditional_fla.py', 'observe_textcraft_conditional_fla_job.py',
        'analyze_textcraft_conditional_fla.py', Path(__file__).name))
    paths.extend(OWNER / name for name in (
        'finite_fla_gpu.py', 'native_capture_events.py', 'qwen35_code_local_capture.py',
        'remaining-gdn-source-contract.json', 'native_fla_chunk.py', 'native-fla-chunk-source.json'))
    recorded = {item['path']: item for item in report['sources']}
    for item_path in paths:
        item = source(item_path)
        item['path'] = item_path.relative_to(REPO).as_posix()
        if item['path'] in recorded:
            recorded[item['path']].update(item)
        else:
            report['sources'].append(item)
            recorded[item['path']] = item
    section = dict(scope=analysis['scope'], analysis=source(analysis_path),
        coverage=analysis['coverage'], layer_summaries=analysis['layer_summaries'],
        capture_start_groups=analysis['capture_start_groups'],
        actual_dtypes=analysis['observed_tensor_dtypes'],
        payload=analysis['CPU_coefficient_bank_payload_statistics'],
        execution=dict(pid=job['pid'], pid_birth=job['pid_birth'], devices=job['devices'],
            started_unix=job['started_unix'], completed_unix=completed['completed_unix'],
            wall_seconds=completed['completed_unix']-job['started_unix'],
            full_finite_calls=14, single_root_calls=28, actor_backward_calls=0,
            optimizer_steps=0, scheduler_steps=0),
        provenance=dict(prepared=source(LOCAL / 'prepared.json'),
            owner=source(LOCAL / 'native-owner-inspection.json'),
            independent_raw_review=source(LOCAL / 'independent-raw-fla-review.json')),
        limitations=analysis['limitations'],
        production_state='Passive original-interface observation only. No production code, weights, PPO, Q/V/A or training parameters changed. TextCraft remains stopped.')
    section['largest_mean_absolute_term_by_layer'] = [dict(decoder_index=row['decoder_index'],
        term=max(('input_chain', 'FLA_projection', 'native_cast_bridge'),
            key=lambda name: row['unique_probe_statistics'][name]['abs_mean']))
        for row in analysis['layer_summaries']]
    report['conditional_fla'] = section
    report['updated_unix'] = time.time()
    report['status'] = 'actual_FLA_remaining_residual_split_no_production_repair'
    report['interpretation']['next_owner_check'] = (
        'Use the actual three-part residual split to select the demonstrated largest DT propagation seam. '
        'Keep conditional approximation, native dtype/storage boundaries and official kernel tolerance distinct; '
        'do not repair by rescaling, clipping, normalization, changing entropy or restarting formal training.')
    for item in report['sources']:
        assert hashlib.sha256((REPO / item['path']).read_bytes()).hexdigest() == item['sha256'], item['path']
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    runtime = REPO / 'experiments/rl/RUNTIME_RECORD.md'
    raw = runtime.read_bytes()
    heading = '## 2026-10-06 FLA 剩余链路：原生系数、操作数与 dtype 边界'
    entry = (heading+'\n\n'+
        f"隔离PID{job['pid']}/birth{job['pid_birth']}已完成，GPU4/5，"
        f"{section['execution']['wall_seconds']:.3f}秒。沿用原7个B4/rank、checkpoint25、"
        'LoRA8/16、原DT/FLA接口；14 full finite、28 single root，actor backward/'
        'optimizer/scheduler均0。原始104 transport-layer保留，按同一UID/source位置明确'
        '合并为42 probe×2层；不改变分母。\n\n'+
        '只观察原外层对称FLA回调最终系数及真实do；原LocalCaptureEvents读取stage '
        'q/k/v/beta、public raw-g和FP16输出，避免把stage累计g当raw-g。'
        '实际capture_start+cut/head8与原GDN一致。输入链I-Z-F、FLA投影F-Y、'
        'dtype边界Y-O闭合到同次原remaining；闭合不是官方数值验收，也不是信用已修复。\n\n'+
        '原raw、实际dtype/payload、CPU绑定与独立复核见'
        'results_textcraft_learning_degradation_20261006.json的conditional_fla。'
        '生产代码、Q/V/A、PPO和参数不变，TextCraft继续停止。\n\n').encode('utf-8')
    if heading.encode('utf-8') not in raw:
        title, tail = raw.split(b'\n', 1)
        runtime.write_bytes(title+b'\n\n'+entry+tail)
    print(json.dumps(dict(report=source(path), runtime_record=source(runtime),
        recorded_sources=len(report['sources']), status=report['status']), ensure_ascii=False))

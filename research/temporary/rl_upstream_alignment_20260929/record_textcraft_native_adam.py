"""Bind the original Adam observation and weak-signal accounting to existing results."""
import hashlib
import json
from pathlib import Path
import time

from stage_environment_entry import AUDIT
from stage_textcraft_native_adam import LOCAL

REPO = Path(__file__).resolve().parents[3]


def source(path):
    path = Path(path)
    with path.open('rb') as stream:
        sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(path=path.relative_to(REPO).as_posix(), sha256=sha, bytes=path.stat().st_size)


if __name__ == '__main__':
    report_path = REPO / 'experiments/rl/results_textcraft_learning_degradation_20261006.json'
    report = json.loads(report_path.read_bytes())
    analysis_path = LOCAL / 'native-adam-analysis.json'
    analysis = json.loads(analysis_path.read_bytes())
    job = json.loads((LOCAL / 'job.json').read_bytes())
    completed = json.loads((LOCAL / 'completed.json').read_bytes())
    cpu_run = json.loads((LOCAL / 'analysis-cpu-run.json').read_bytes())
    assert source(analysis_path)['sha256'] == cpu_run['output']['sha256']
    magnitude_path = AUDIT / 'textcraft-degradation-20261005/native-minibatch-v4/task-signal-magnitude-source-audit.json'
    magnitude = json.loads(magnitude_path.read_bytes())
    assert completed['DT_calls'] == completed['sampling_calls'] == 0
    assert completed['global_rows'] == 64
    for rank in analysis['actual_native_updates'].values():
        for branch in rank.values():
            assert branch['optimizer_step_calls'] == branch['scheduler_step_calls'] == 1
            assert branch['optimizer_before']['step_counters'] == {'100.0': 496}
            assert branch['optimizer_after']['step_counters'] == {'101.0': 496}
    paths = list(LOCAL.glob('*.json')) + list(LOCAL.glob('*.txt'))
    paths.extend(LOCAL / name for name in ('diagnostic.log', 'effective-config.yaml', 'independent_native_adam_review.py'))
    paths.extend(AUDIT / name for name in (
        'verify_textcraft_native_adam.py', 'observe_native_adam_update.py',
        'stage_textcraft_native_adam.py', 'observe_textcraft_native_adam_job.py',
        'analyze_textcraft_native_adam.py', 'run_textcraft_native_adam_analysis.py',
        Path(__file__).name))
    paths.append(magnitude_path)
    recorded = {item['path']: item for item in report['sources']}
    for path in paths:
        item = source(path)
        if item['path'] in recorded:
            if path.resolve() == Path(__file__).resolve():
                recorded[item['path']].update(item)
            else:
                assert item['sha256'] == recorded[item['path']]['sha256'], item['path']
        else:
            report['sources'].append(item)
            recorded[item['path']] = item
    geometry = [{key: value for key, value in rank.items() if key != 'parameters'}
                for rank in analysis['local_parameter_geometry']]
    report['task_signal_magnitude_source_account'] = dict(
        source=source(magnitude_path), scope=magnitude['scope'],
        coefficients=magnitude['coefficient_source_account'],
        gradient=magnitude['gradient_account'], mapping=magnitude['mapping_scale_check'],
        established_and_unknown=magnitude['established_and_unknown'])
    report['native_actual_Adam_observation'] = dict(
        source=source(analysis_path),
        independent_review=source(LOCAL / 'independent-native-adam-review.json'),
        scope=analysis['scope'], execution=dict(pid=job['pid'], pid_birth=job['pid_birth'],
            started_unix=job['started_unix'], completed_unix=completed['completed_unix'],
            wall_seconds=completed['completed_unix']-job['started_unix'], devices=job['devices'],
            global_rows=64, actor_microbatch_per_gpu=4, actual_optimizer_steps_per_rank=3,
            actual_scheduler_steps_per_rank=3, DT_calls=0, sampling_calls=0),
        original_sources=completed['sources'], owner_reduction=analysis['owner_reduction'],
        before_readout_comparisons=analysis['initial_readout_comparisons'],
        H_LP_observations=analysis['H_LP_observations'],
        actual_native_updates=analysis['actual_native_updates'], local_parameter_geometry=geometry,
        CPU_analysis_runtime=analysis['runtime'], limitations=analysis['interpretation_limits'],
        interpretation=(
            'With identical complete checkpoint25 initial parameter shards and before observations, '
            'the DT actual native Adam deltas have cosine approximately0.99958 to the zero-current-PG '
            'control in both local ranks, with matched difference norms approximately3% of DT delta '
            'norms. This supports a weak incremental current task effect on this update, without '
            'linearly apportioning Adam momentum/weight decay to entropy or proving the entire '
            'historical learning trajectory. Fresh before log probabilities differ from the saved '
            'original trainer probabilities and those saved values were not overwritten; no ratio1 '
            'or numerical tolerance claim is made.'))
    report['status'] = 'weak_signal_source_account_and_native_Adam_control_observed_no_production_repair'
    report['updated_unix'] = time.time()
    report['interpretation']['priority'] = 'Weak effective task signal: actual return readout, its DT token coefficients, original actor/regularizer balance and real native Adam effects.'
    report['interpretation']['next_owner_check'] = (
        'Investigate the measured return-reader optimism and weak outcome discrimination and the '
        'origin of small response-token d. No mechanical extra probability/length divisor was found. '
        'Do not infer whole-DT quality from single-token supplemental probes: whole attribution '
        'evaluation requires the original cumulative deletion curves and RISE/MAS. Keep token '
        'Q/V/A and upstream PPO unchanged; do not scale by211 or retune entropy to conceal the cause.')
    new_limits = [
        'The new three-branch Adam observation is a single restored checkpoint25 update, not historical optimizer replay. Zero-current-PG control includes restored optimizer history and weight decay.',
        'The native64 1/211 task-gradient ratio compares raw sampled DT credit to officially standardized GRPO coefficients; it is not an expected equality or a DT quality metric.',
        'Single-token model deletions remain supplemental diagnostics. They do not replace cumulative deletion curves or RISE/MAS and do not establish overall DT attribution failure.'
    ]
    for limit in new_limits:
        if limit not in report['interpretation']['limits']:
            report['interpretation']['limits'].append(limit)
    report['findings'][4] = report['findings'][4].replace('whose quality fails this random-probe comparison.', 'whose discrepancy is measured on these supplemental probes; this is not an overall DT quality verdict.')
    report['findings'][7] = 'The sampled native64 task/regularizer imbalance and matched real Adam control support weak effective current task updates. They do not uniquely identify historical degradation or establish whole-DT attribution failure.'
    report['findings'][11] = 'Priority is the measured weakness of effective task credit before and after original Adam. Single-token finite-boundary localization is supplemental and does not supersede original aggregate attribution evaluation.'
    for item in report['sources']:
        assert hashlib.sha256((REPO / item['path']).read_bytes()).hexdigest() == item['sha256'], item['path']
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    runtime = REPO / 'experiments/rl/RUNTIME_RECORD.md'
    heading = '## 2026-10-06 弱任务信号：原 Adam 单步对照与幅度来源'
    raw = runtime.read_bytes()
    if heading.encode('utf-8') not in raw:
        entry = (heading+'\n\n'+
            f"隔离PID{job['pid']}/birth{job['pid_birth']}已完成，GPU4/5，"
            f"{completed['completed_unix']-job['started_unix']:.3f}秒。原64条完整carrier、checkpoint25、"
            'LoRA8/16、每卡B4、原VERL actor/core/import SHA与损失默认保持不变；三支分别完整恢复model/optimizer/extra，'
            '复用原AdamW动量/RNG/scheduler。每支每rank实际1次optimizer与1次scheduler，496个state均100→101。'
            '无rollout、DT重算或正式重启。诊断只保留原返回值，原step/clip/offload未替换。\n\n'+
            '按原B4分母，DT优势绝对均值为GRPO约1/1446.69，原任务梯度约1/211.33；'
            '幅度小已在DT的d里，未发现额外概率、长度除数或expm1衰减。GRPO含官方组内标准化和失败轨迹负系数，'
            '两者不要求梯度范数相等，不能将211变成信用倍率。零回报DT项为零是固定估计性质，不标为丢mask。\n\n'+
            '实际原Adam DT与当前PG置零对照的本地分片更新cos=0.999581/0.999579，差向量约DT更新范数3%；'
            '对照仍含共同历史动量与weight decay，不线性分账为纯熵。原B4等权H变化DT+0.000575458、'
            'GRPO+0.000070508、对照+0.000238523。fresh before与saved trainer LP存在差异，保留原old/ref未覆写，'
            '不声称ratio恒1或新增数值容差通过。三支初始参数与before读数相同，配对变化单独记录。\n\n'+
            'CPU分析只复用原DataProto/chunk与agg_loss，CUDA/distributed均false；实际本地dtype为FP32 LoRA分片，'
            'FP64副本只作描述、无gather。模型与CPU分析source/配置/资源/输入SHA、raw与独立复核全部绑定现有结果。'
            '单token对照仅补充，不取代原论文累计删除曲线及RISE/MAS。优先继续查弱信号来源；没有生产修复、'
            '信用放大、熵系数改动或TextCraft重启，正式版本保持不动。\n\n').encode('utf-8')
        title, tail = raw.split(b'\n', 1)
        runtime.write_bytes(title+b'\n\n'+entry+tail)
    print(json.dumps(dict(report=source(report_path), runtime=source(runtime), sources=len(report['sources']))))

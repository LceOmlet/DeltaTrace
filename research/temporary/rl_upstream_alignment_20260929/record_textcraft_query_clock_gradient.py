"""Bind the completed native query-only PG/H comparison without deployment."""
import hashlib
import json
from pathlib import Path
import time

from stage_environment_entry import AUDIT, REPO
from record_textcraft_native_adam import source


if __name__ == '__main__':
    directory = AUDIT / 'textcraft-degradation-20261005/query-clock-gradient-20261006'
    current = directory / 'v2'
    completed = json.loads((current / 'completed.json').read_bytes())
    job = json.loads((current / 'job.json').read_bytes())
    resource = json.loads((current / 'completion-resource-status.json').read_bytes())
    assert completed['input_sha256'] == '45ae51e3e18f206e238a1f1e934eaf9a4074c9361dabdab83bb3e00a2b4b084d'
    assert all(completed[key] == 0 for key in ('sampling_calls', 'optimizer_steps', 'scheduler_steps'))
    assert resource['completed_unix'] == completed['completed_unix'] and not resource['pid_exists']
    assert resource['pid'] == job['pid'] and resource['pid_birth'] == job['pid_birth']
    assert resource['completed_source']['sha256'] == hashlib.sha256((current / 'completed.json').read_bytes()).hexdigest()
    groups = {}
    for group in ('saved_original', 'clock_recomputed'):
        raw = [json.loads((current / f'rank{rank}-{group}-gradients.json').read_bytes()) for rank in (0, 1)]
        assert raw[0]['gradient_statistics'] == raw[1]['gradient_statistics']
        stats = raw[0]['gradient_statistics']
        norms, dots = stats['norms'], stats['inner_products']
        for item in raw:
            assert item['actual_backward_passes'] == 2
            assert item['scheduler_step_no_op_calls'] == 1
            assert not item['optimizer_step_executed'] and not item['scheduler_step_executed']
            assert all(len(value['microbatch_losses']) == 8 for value in item['passes'].values())
        groups[group] = dict(gradient_statistics=stats,
            weighted_H_over_task_PG_norm=norms['weighted_entropy'] / norms['dt_pg'],
            cosine_task_PG_with_weighted_H=dots['dt_pg:weighted_entropy'] / (norms['dt_pg'] * norms['weighted_entropy']),
            rank_receipts=[source(current / f'rank{rank}-{group}-gradients.json') for rank in (0, 1)],
            pass_wall_seconds_by_rank={str(rank): {name: value['elapsed_seconds'] for name, value in item['passes'].items()}
                                      for rank, item in enumerate(raw)})
    dt = [json.loads((current / f'rank{rank}-clock-native-dt-report.json').read_bytes()) for rank in (0, 1)]
    assert all(item['query_method_restored'] for item in dt)
    assert all(item['candidate_query_source']['sha256'] == '94a7afbc09da72b62572d31fd32a6534f6e8f3daf656fce1011cdfa68b3c3e2b' for item in dt)
    path = REPO / 'experiments/rl/results_textcraft_learning_degradation_20261006.json'
    report = json.loads(path.read_bytes())
    paths = [AUDIT / name for name in ('observe_textcraft_query_clock_gradients.py',
        'verify_textcraft_query_clock_gradients.py', 'stage_textcraft_query_clock_gradients.py', Path(__file__).name)]
    for version in ('v1', 'v2'):
        root = directory / version
        paths += [item for item in root.iterdir() if item.is_file() and item.suffix != '.pkl']
        paths += [item for item in (root / 'submitted-source').iterdir() if item.is_file()]
    known = {item['path']: item for item in report['sources']}
    for item_path in paths:
        item = source(item_path)
        if item['path'] in known:
            assert known[item['path']]['sha256'] == item['sha256'], item['path']
        else:
            known[item['path']] = item
            report['sources'].append(item)
    for item in report['sources']:
        assert hashlib.sha256((REPO / item['path']).read_bytes()).hexdigest() == item['sha256'], item['path']
    before, after = (groups[name]['gradient_statistics']['norms'] for name in ('saved_original', 'clock_recomputed'))
    report['native_query_clock_gradient_observation'] = dict(
        role='Completed query-only saved-native diagnostic; not a formal learning repair or a new tolerance gate.',
        execution=dict(pid=job['pid'], pid_birth=job['pid_birth'], devices=[4, 5],
            elapsed_from_job_start_seconds=completed['completed_unix'] - job['started_unix'],
            completed=source(current / 'completed.json'), final_resource=resource),
        groups=groups,
        task_PG_norm_new_over_old=after['dt_pg'] / before['dt_pg'],
        weighted_H_norm_new_over_old=after['weighted_entropy'] / before['weighted_entropy'],
        original_owner_DT_reports=[source(current / f'rank{rank}-clock-native-dt-report.json') for rank in (0, 1)],
        original_owner_DT_statistics={str(rank): item['report'] for rank, item in enumerate(dt)},
        native_input_comparison=source(current / 'native-query-clock-credit-summary.json'),
        independent_review=source(current / 'independent-query-clock-gradient-review.json'),
        operations=dict(model_initializations=1, original_complete_checkpoint_loads=1,
            native_backward_passes_per_rank=4, native_B4_backward_calls_per_rank=32,
            native_optimizer_step_no_op_calls_per_rank=4, native_scheduler_step_no_op_calls_per_rank=2,
            optimizer_updates=0, scheduler_updates=0, rollouts=0),
        original_contract='Saved64/checkpoint25, original trajectory_credit/producer/runner/compute_advantage; only prepared query_ids is temporarily bound and restored. Old credit fields are replaced through original DataProto.pop/union; all other original fields retained. Original loss and B4/8 denominator unchanged.',
        limits=['A norm change alone is not gradient-direction agreement or restored task learning.',
            'No cross-group task gradient dot or new complete GRPO gradient is available; do not combine previous-run Grams.',
            'Entropy repeated around original DT is descriptive state evidence, without a new equality/tolerance gate.',
            'No formal update, resampling, advantage normalization, credit scaling or entropy/configuration change.',
            'No single-token whole-attribution quality verdict; author cumulative deletion and RISE/MAS remain separate.'])
    report['native_query_clock_gradient_observation']['finding'] = (
        'The prepared query fixes a demonstrated current-response conditioning mismatch, '
        'but the observed task PG norm decreases while weighted entropy repeats at almost the same norm. '
        'It therefore does not demonstrate a repair of weak task learning; no production deployment follows.')
    report['status'] = 'weak_task_signal_measured_query_semantic_repair_gradient_test_completed_no_learning_repair_deployed'
    report['updated_unix'] = time.time()
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    runtime = REPO / 'experiments/rl/RUNTIME_RECORD.md'
    heading = '## 2026-10-06 查询语义修复的原任务梯度实测'
    raw = runtime.read_bytes()
    if heading.encode() not in raw:
        entry = (heading + '\n\n'
            f"GPU4/5隔离PID{job['pid']}/birth{job['pid_birth']}已完成并退出。"
            '同一原64条/checkpoint25/LoRA8/16/每卡B4，原模型及完整checkpoint各加载一次。'
            '先用保存的原信用测PG/H，再调用原trajectory_credit→worker→producer→runner重算，'
            '临时绑定afe59dd的query_ids(94a7afbc)，结束后恢复原方法；原pop/union/compute_advantage接回新信用。'
            '原reward、IDs、mask、old/ref LP及原Q/V/A组合保持，CPU逐字段回执另绑定。'
            '每rank两个信用组各两个local32反向（8B4/8），原clip执行；'
            'optimizer no-op4/scheduler no-op2，真实更新/rollout均0。\n\n'
            f"原PG norm {before['dt_pg']:.14g}，新PG {after['dt_pg']:.14g}，"
            f"新/旧 {after['dt_pg'] / before['dt_pg']:.10g}；"
            f"weighted-H {before['weighted_entropy']:.14g}→{after['weighted_entropy']:.14g}。"
            '这些是同job原preclip Gram的描述，不新增容差；没有新旧PG互积，不声称方向或成功率改善。'
            '查询修复对齐环境立即处理已发出的回复，但本次未增强任务梯度，不能当学习问题已修。'
            '正式TextCraft保持停止，查询候选仍prepared-only；不做信用缩放、归一化、熵调参或公式改动。\n\n'
            '原DT每rank24B4/96contrasts，实际查询225/226 token、最大读出5226/5297，'
            '保留原密集ABI、缓存与dtype；内部conservation标志不是FA/FLA官方容差断言。'
            '原作者累计删除/RISE/MAS仍是整体归因评价，不用单token或小梯度代替。'
            '失败v1在CPU检查TensorDict.keys接口前终止，无模型成本；v2只修诊断键迭代。'
            '所有source/import/config/input SHA、原件、阶段日志、完成资源、CPU对照和独立review绑定现有degradation结果。\n\n').encode()
        title, tail = raw.split(b'\n', 1)
        runtime.write_bytes(title + b'\n\n' + entry + tail)
    print(json.dumps(dict(report=source(path), runtime=source(runtime), sources=len(report['sources']),
        task_PG_norm_new_over_old=after['dt_pg'] / before['dt_pg'],
        weighted_H_norm_new_over_old=after['weighted_entropy'] / before['weighted_entropy'])))

"""Bind a completed labels-only native task-gradient observation to existing records.

This writer calls no model, credit, loss, gradient or normalization code.  It
requires the completed raw reports, CPU analysis and independent review before
changing the existing result record; diagnostic v1 remains a failed CPU prepare.
"""
import argparse
import json
from pathlib import Path, PurePosixPath
import time

from stage_environment_entry import AUDIT, REPO
from record_textcraft_native_adam import source


DIRECTORY = AUDIT / 'textcraft-degradation-20261005/equivalent-label-gradient-20261006'
GROUPS = ('saved_original', 'swapped_labels')
LABELS = ('dt_pg', 'weighted_entropy')
FIELD = 'equivalent_label_native_task_gradient'


def read(path):
    return json.loads(Path(path).read_bytes())


def operations(raw):
    """Describe the actual original-observer passes, without inventing a reducer."""
    return dict(
        backward_passes=raw['actual_backward_passes'],
        native_B4_backward_calls=sum(len(p['microbatch_losses']) for p in raw['passes'].values()),
        optimizer_step_no_op_calls=sum(p['optimizer_boundary']['optimizer_step_no_op_calls']
                                       for p in raw['passes'].values()),
        scheduler_step_no_op_calls=raw['scheduler_step_no_op_calls'],
        optimizer_step_executed=raw['optimizer_step_executed'],
        scheduler_step_executed=raw['scheduler_step_executed'],
        native_gradient_clipping_executed=raw['native_gradient_clipping_executed'],
        pass_wall_seconds={name: p['elapsed_seconds'] for name, p in raw['passes'].items()})


def record(review_path):
    current = DIRECTORY / 'v2'
    analysis_path = current / 'label-gradient-analysis.json'
    resource_path = current / 'completion-resource-status.json'
    required = [current / name for name in (
        'completed.json', 'job.json', 'prepared.json', 'native-owner-inspection.json',
        'runtime-driver-inspection.json',
        'worker-callsite-import-inspection.json', 'independent-source-binding-review.json',
        'physical-before-start.txt')]
    required += [analysis_path, resource_path, Path(review_path)]
    for rank in (0, 1):
        required += [current / f'rank{rank}-{group}-gradients.json' for group in GROUPS]
        required += [current / f'rank{rank}-cross-group-gradients.json',
                     current / f'rank{rank}-swapped-label-native-dt-report.json']
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError('No records changed; completed artifacts are missing: ' + ', '.join(missing))

    completed, job, prepared = (read(current / name) for name in ('completed.json', 'job.json', 'prepared.json'))
    inspection = read(current / 'native-owner-inspection.json')
    source_review = read(current / 'independent-source-binding-review.json')
    analysis, resource = read(analysis_path), read(resource_path)
    read(review_path)
    # These are source/identity checks, never numeric tolerances or learning gates.
    assert completed['input_sha256'] == job['input']['sha256'] == inspection['native_reader_inputs_sha256']
    assert analysis['input_sha256'] == completed['input_sha256']
    assert prepared['checkpoint'] == job['checkpoint']
    assert completed['checkpoint'] == inspection['checkpoint'] == analysis['checkpoint']
    assert completed['checkpoint'] == str(PurePosixPath(job['checkpoint']) / 'actor')
    assert completed['groups'] == analysis['groups'] == list(GROUPS)
    assert completed['labels'] == analysis['labels'] == list(LABELS)
    assert all(completed[key] == 0 for key in ('sampling_calls', 'optimizer_steps', 'scheduler_steps'))
    assert resource['pid'] == job['pid'] and resource['pid_birth'] == job['pid_birth']
    assert resource['completed_unix'] == completed['completed_unix'] and not resource['pid_exists']
    assert resource['completed_source']['sha256'] == source(current / 'completed.json')['sha256']
    for name in ('diagnostic_source', 'diagnostic_dependencies', 'sources', 'reused_diagnostic_sources', 'input'):
        assert prepared[name] == job[name], name
    driver = AUDIT / 'verify_textcraft_label_gradients.py'
    adapter = AUDIT / 'observe_textcraft_label_gradients.py'
    analyzer = AUDIT / 'analyze_textcraft_label_gradients.py'
    original_recipe = AUDIT / 'verify_textcraft_native_adam.py'
    recipe_remote_path = next(path for path in job['reused_diagnostic_sources']
                              if PurePosixPath(path).name == original_recipe.name)
    assert source(original_recipe)['sha256'] == job['reused_diagnostic_sources'][recipe_remote_path]
    assert source(driver)['sha256'] == job['diagnostic_source']['sha256']
    assert source(adapter)['sha256'] == job['diagnostic_dependencies'][adapter.name]['sha256']
    assert source(analyzer)['sha256'] == analysis['analysis_source']['sha256']
    for name, item in analysis['source_files'].items():
        assert source(current / Path(item['path']).name)['sha256'] == item['sha256'], name

    per_rank = []
    for measured in analysis['per_rank']:
        rank = measured['rank']
        assert rank in (0, 1)
        raw = {group: read(current / f'rank{rank}-{group}-gradients.json') for group in GROUPS}
        cross = read(current / f'rank{rank}-cross-group-gradients.json')
        dt = read(current / f'rank{rank}-swapped-label-native-dt-report.json')
        assert measured['original_native_statistics'] == raw[GROUPS[0]]['gradient_statistics']
        assert measured['new_native_statistics'] == raw[GROUPS[1]]['gradient_statistics']
        assert measured['cross_native_statistics'] == cross['gradient_statistics']
        actual_operations = {group: operations(item) for group, item in raw.items()}
        assert all(not item['optimizer_step_executed'] and not item['scheduler_step_executed']
                   for item in actual_operations.values())
        assert dt['original_label_method_restored'] and dt['query_method_unchanged']
        per_rank.append(dict(rank=rank,
            original_PG_H=measured['original_PG_H'], new_PG_H=measured['new_PG_H'],
            old_new_PG=measured['old_new_PG'], old_PG_new_H=measured['old_PG_new_H'],
            native_original_statistics=measured['original_native_statistics'],
            native_new_statistics=measured['new_native_statistics'],
            native_cross_statistics=measured['cross_native_statistics'],
            effective_config=measured['effective_config'],
            config_identity=measured['group_config_exact'], input_shape_identity=measured['input_shapes_exact'],
            rng_observations=measured['rng_observations'], operations=actual_operations,
            native_DT_seconds=dt['seconds'], native_DT_report=dt['report'],
            native_DT_source=source(current / f'rank{rank}-swapped-label-native-dt-report.json')))
    assert sorted(item['rank'] for item in per_rank) == [0, 1]

    result_path = REPO / 'experiments/rl/results_textcraft_learning_degradation_20261006.json'
    report = read(result_path)
    original_status = report['status']
    paths = [AUDIT / name for name in (
        'verify_textcraft_label_gradients.py', 'observe_textcraft_label_gradients.py',
        'stage_textcraft_label_gradients.py', 'analyze_textcraft_label_gradients.py',
        'run_textcraft_label_gradient_analysis.py', Path(__file__).name)]
    # Large native DataProto files remain external artifacts, not text audit sources.
    paths += [path for path in DIRECTORY.rglob('*') if path.is_file() and path.suffix not in ('.pkl', '.pt')]
    paths.append(Path(review_path))
    paths = sorted(set(path.resolve() for path in paths))
    known = {item['path']: item for item in report['sources']}
    for path in paths:
        item = source(path)
        if item['path'] in known:
            assert known[item['path']]['sha256'] == item['sha256'], item['path']
        else:
            known[item['path']] = item
            report['sources'].append(item)
    for item in report['sources']:
        assert source(REPO / item['path'])['sha256'] == item['sha256'], item['path']

    report[FIELD] = dict(
        role='Completed isolated equivalent-label conditional-event encoding diagnostic; no production or method change.',
        analysis=source(analysis_path), independent_review=source(review_path),
        CPU_source_binding_review=source(current / 'independent-source-binding-review.json'),
        v1_failure=dict(source_bound_detail=source_review['v1_failure'],
            frozen_sources=[source(path) for path in sorted((DIRECTORY / 'v1/source').iterdir()) if path.is_file()],
            raw_failure=source(DIRECTORY / 'v1/native-owner-inspection.stdout.txt'),
            scope='Failed CPU TensorDict key inspection before model/DT/backward submission; retained unchanged.'),
        execution=dict(pid=job['pid'], pid_birth=job['pid_birth'], devices=job['devices'],
            started_unix=job['started_unix'], completed_unix=completed['completed_unix'],
            elapsed_from_job_start_seconds=completed['completed_unix'] - job['started_unix'],
            completed=source(current / 'completed.json'), final_resource=resource),
        provenance=dict(job=source(current / 'job.json'), prepared=source(current / 'prepared.json'),
            actual_driver_inspection=source(current / 'runtime-driver-inspection.json'),
            CPU_inspection=source(current / 'native-owner-inspection.json'),
            fresh_worker_import=source(current / 'worker-callsite-import-inspection.json'),
            effective_configuration=inspection['config_source'], original_sources=completed['sources'],
            input=job['input'], checkpoint_root=job['checkpoint'], actor_checkpoint=completed['checkpoint'],
            checkpoint_resolution=dict(source=source(original_recipe), remote_path=recipe_remote_path,
                line=26, expression="CHECKPOINT = Path(os.environ['DT_TEXTCRAFT_CHECKPOINT']) / 'actor'")),
        per_rank=per_rank, carrier_comparison=analysis['carriers'],
        recorded_completed_operations=analysis['operations'],
        contract='Only RewardAlphabet.labels is temporarily reversed to 10 and restored; original query_ids 228, return meanings/values, observed class index, original trajectory_credit/compute_advantage and token Q/V/A composition remain. Original complete loss_mask, local32 to B4 times8 and 1/8 accumulation, PPO loss, weighted entropy, LoRA and checkpoint are retained. Cross-PG uses the same original ownership-aware statistics helper on actual captured preclip gradients.',
        finding='This paired diagnostic measures sensitivity of the current conditional-event encoding in actual native task-PG magnitude and direction. It does not establish a learning repair or determine whole-attribution ranking quality.',
        limits=[
            'Per-rank reports already contain the original mesh-reduced statistics; they are not added into a new global reducer.',
            'Gradient norm/angle changes do not establish improved parameter updates or future task success.',
            'Saved original A and recomputed labels-only A use the same saved carrier; the original A is not claimed to be a fresh rerun.',
            'RNG, config and repeated entropy observations are descriptive source-bound evidence without a numeric tolerance gate.',
            'Original DT conservation diagnostics are not FA/FLA official tolerance assertions.',
            'Author cumulative deletion and normalized RISE/MAS remain separate whole-attribution evaluation.',
            'No whitening, advantage normalization, scaling, loss-coefficient change, rollout, optimizer update or production repair.'])
    assert report['status'] == original_status
    report['updated_unix'] = time.time()

    runtime = REPO / 'experiments/rl/RUNTIME_RECORD.md'
    runtime_raw = runtime.read_bytes()
    heading = '## 2026-10-06 等价标签编码的原任务梯度诊断'
    entry = (heading + '\n\n'
        f"隔离诊断PID{job['pid']}/birth{job['pid_birth']}在GPU{job['devices']}已完成并退出，"
        f"墙钟{completed['completed_unix'] - job['started_unix']:.3f}s（原job started到completed）。"
        '同一保存64条和完整checkpoint25，只加载一次原模型/checkpoint。先保存原A的PG/H，'
        '再临时将原RewardAlphabet.labels从01换成10，原query_ids 228保持，标签/目标编码同步；'
        '原return语义/数值、observed index、原Q/V/A公式、trajectory_credit和compute_advantage保持并恢复临时方法。'
        '新信用仅通过原DataProto pop/union接回，不重建轨迹、mask或奖励。\n\n')
    for rank in per_rank:
        old, new, paired = rank['original_PG_H'], rank['new_PG_H'], rank['old_new_PG']
        ops = rank['operations']
        entry += (f"rank{rank['rank']}原PG norm {old['first_norm']:.14g}，换标签PG {new['first_norm']:.14g}；"
            f"旧/新PG原互积cos {paired['cosine']:.12g}，"
            f"weighted-H {old['second_norm']:.14g}→{new['second_norm']:.14g}。"
            f"实际反向pass {sum(x['backward_passes'] for x in ops.values())}，"
            f"原B4反向 {sum(x['native_B4_backward_calls'] for x in ops.values())}，"
            f"optimizer no-op {sum(x['optimizer_step_no_op_calls'] for x in ops.values())}，"
            f"scheduler no-op {sum(x['scheduler_step_no_op_calls'] for x in ops.values())}；"
            f"原DT报告finite_trace_calls={rank['native_DT_report'].get('finite_trace_calls')}，"
            f"event_contrasts={rank['native_DT_report'].get('event_contrasts')}。\n\n")
    entry += ('这些是当前条件事件编码对实际PG幅度/方向的描述；不把换标签称为修好训练，'
        '不从梯度比值推未来成功率，也不替代原作者累计删除RISE/MAS。'
        '原完整action mask、每卡B4/8累积、上游PPO和H/KL配置保持；原clip执行，真实optimizer/scheduler更新和rollout均0。'
        'v1在CPU准备阶段因TensorDict键迭代接口失败，0模型/DT/backward；失败源和日志保留，v2仅修.keys()检查。'
        'source/import/config/carrier SHA、原报告、CPU分析、独立review及完成资源绑定结果字段'
        f'`{FIELD}`。正式版本、原status和PLAN不变，未运行pending归一化。\n\n')
    new_runtime = runtime_raw
    if heading.encode() not in runtime_raw:
        title, tail = runtime_raw.split(b'\n', 1)
        new_runtime = title + b'\n\n' + entry.encode() + tail
    result_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    if new_runtime != runtime_raw:
        runtime.write_bytes(new_runtime)
    print(json.dumps(dict(result=source(result_path), runtime=source(runtime),
        field=FIELD, status_preserved=report['status'], sources=len(report['sources']))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review', type=Path,
        default=DIRECTORY / 'v2/independent-label-gradient-review.json')
    args = parser.parse_args()
    record(args.review)

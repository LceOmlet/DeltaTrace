"""Record completed author reward curves without changing the learning method.

This writer is intentionally separate from preparation and execution.  Run it
only after the completed raw results, resource observation, analysis, and
independent raw review have been supplied.  It appends diagnostic provenance to
the existing canonical report; it does not launch a model or alter a runtime.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from stage_environment_entry import AUDIT, REPO
from record_textcraft_native_adam import source


LOCAL = AUDIT / 'textcraft-degradation-20261005/author-reward-curve-20261006/v1'
LEDGER_DIRECTORY = AUDIT / 'textcraft-degradation-20261005/author-cumulative-deletion-20261006'
FIELD = 'author_cumulative_reward_curve_observation'
HEADING = '## 2026-10-06 原作者累计删除与奖励事件曲线'


def read(path):
    return json.loads(Path(path).read_bytes())


def completed_observation(directory):
    """Read the completed owner's artifacts; do not compute another metric."""
    directory = Path(directory)
    completed = read(directory / 'completed.json')
    job = read(directory / 'job.json')
    resource = read(directory / 'completion-resource-status.json')
    analysis = read(directory / 'author-reward-curve-analysis.json')
    independent = read(directory / 'independent-author-reward-curve-review.json')
    prepared = read(directory / 'prepared.json')
    ranks = [read(directory / f'rank{rank}-curves.json') for rank in (0, 1)]
    # These are identity/completion contracts, not numerical acceptance gates.
    assert all(rank['phase'] == 'complete_author_reward_curves' for rank in ranks)
    assert all(completed[key] == 0 for key in
               ('optimizer_steps', 'scheduler_steps', 'backward_calls', 'finite_trace_calls'))
    assert resource['pid'] == job['pid'] and resource['pid_birth'] == job['pid_birth']
    assert resource['completed_unix'] == completed['completed_unix']
    assert resource['completed_source']['sha256'] == source(directory / 'completed.json')['sha256']
    assert not resource['pid_exists']
    raw_sources = [source(directory / f'rank{rank}-curves.json') for rank in (0, 1)]
    assert [item['sha256'] for item in analysis['sources']['raw_ranks']] == [
        item['sha256'] for item in raw_sources]
    assert analysis['sources']['completed']['sha256'] == source(directory / 'completed.json')['sha256']
    assert [rank['native_forward_calls'] for rank in ranks] == analysis['coverage']['actual_native_forward_calls']
    assert all(rank['optimizer_steps'] == rank['scheduler_steps'] ==
               rank['backward_calls'] == rank['finite_trace_calls'] == 0 for rank in ranks)
    for actual in completed['ranks']:
        assert actual['sha256'] == source(directory / f"rank{actual['rank']}-curves.json")['sha256']
    observation = dict(
        role='Completed isolated author cumulative deletion and RISE/MAS observation; descriptive, no learning repair or tolerance gate.',
        execution=dict(pid=job['pid'], pid_birth=job['pid_birth'], devices=job['devices'],
            started_unix=job['started_unix'], completed_unix=completed['completed_unix'],
            elapsed_from_job_start_seconds=completed['completed_unix'] - job['started_unix'],
            job=source(directory / 'job.json'), completed=source(directory / 'completed.json'),
            final_resource=source(directory / 'completion-resource-status.json'),
            final_resource_observation=resource),
        analysis=source(directory / 'author-reward-curve-analysis.json'),
        preparation=source(directory / 'prepared.json'),
        independent_preparation_review=source(directory / 'independent-preparation-review.json'),
        independent_raw_review=source(directory / 'independent-author-reward-curve-review.json'),
        raw_rank_receipts=raw_sources,
        actual_original_runtime_sources=[rank['sources'] for rank in ranks],
        actual_author_metric_sources=ranks[0]['cases'][0]['curves']['owner_sources'],
        coverage=analysis['coverage'],
        replica_policy=analysis['replica_policy'],
        original_author_view_summaries=analysis['view_summaries'],
        endpoint_normalization_observation={view: {
            key: independent['views'][view][key] for key in
            ('negative_endpoint_UID_count', 'all_zero_normalized_response_UID_count',
             'all_zero_UIDs_exactly_negative_endpoint_UIDs')}
            for view in ('signed_RISE', 'positive_MAS')},
        replica_evidence_scope='Complete transport rows, primary UID records and per-point replica spreads remain in the referenced analysis/raw; they are not duplicated here.',
        figures=analysis.get('figures', []),
        original_contract=dict(checkpoint=job['checkpoint'], saved_input=job['input'],
            saved_ordering_source='saved_source_d_from_A; inverse from stored FP32 A, not fresh or pre-storage DT',
            author_k=20, views=['positive_MAS', 'signed_RISE'],
            score_batch_per_call=prepared['native_batch_per_call'],
            actor_microbatch_per_gpu=prepared['config_actor_microbatch'],
            lora_rank=8, lora_alpha=16,
            scoring='Original runner.read_outcomes observed categorical target column; unchanged owner metric receives exact saved prefix/target IDs.',
            metric_owner='Original imported ft_ifr_improve.faithfulness_test_skip_tokens; ordering, groups, density, normalization, penalty and metric return are not reimplemented.'),
        operations=dict(actual_native_scoring_calls_by_rank=[rank['native_forward_calls'] for rank in ranks],
            actual_completed_cases_by_rank=[len(rank['cases']) for rank in ranks],
            original_model_initializations=1, original_complete_checkpoint_loads=1,
            DT_calls=0, backward_calls=0, optimizer_updates=0, scheduler_updates=0,
            rollouts=0, production_changes=0),
        limits=analysis['limits'] + [
            'This records original model reward-event scores, not exact external-world counterfactuals.',
            'No new tolerance gate or official numerical-pass claim follows from metric or same-pair conservation statistics.',
            'Original endpoint normalization can make the whole normalized curve zero when the factual-minus-fully-deleted endpoint is negative; a small RISE value is not automatically evidence of absent intermediate deletion effects.',
            'No credit normalization, scaling, clipping, entropy/configuration change or formal training restart.',
        ])
    return observation, job, completed, analysis


def provenance_paths(directory):
    """Bind all exact owned v1 files and the already completed source audits."""
    directory = Path(directory)
    paths = [item for item in directory.iterdir() if item.is_file()]
    paths += [item for item in (directory / 'submitted-source').iterdir() if item.is_file()]
    paths += [AUDIT / name for name in (
        'author_reward_curve_adapter.py', 'verify_textcraft_author_reward_curve.py',
        'stage_textcraft_author_reward_curve.py', 'analyze_textcraft_author_reward_curve.py',
        'analyze_saved_native_joint_signal.py', Path(__file__).name)]
    paths += [LEDGER_DIRECTORY / name for name in
              ('native-joint-signal-ledger.json', 'official-advantage-unit-source-audit.json')]
    return list(dict.fromkeys(path.resolve() for path in paths))


def runtime_entry(observation, analysis):
    """Describe actual completion; retain the prior runtime record verbatim."""
    execution = observation['execution']
    coverage = observation['coverage']
    calls = observation['operations']['actual_native_scoring_calls_by_rank']
    runner = observation['actual_original_runtime_sources'][0]['runner']['sha256']
    author = observation['actual_author_metric_sources']['faithfulness_test_skip_tokens']['sha256']
    zero_counts = {view: values['all_zero_normalized_response_UID_count']
                   for view, values in observation['endpoint_normalization_observation'].items()}
    view_text = '; '.join(
        f"{view}: RISE={rows['metrics_verbatim_UID_summary']['rise']['mean']}, "
        f"MAS={rows['metrics_verbatim_UID_summary']['mas']['mean']}, "
        f"原同集合ΔLP与Σd MAE={rows['cumulative_noninitial_points']['mae']}"
        for view, rows in observation['original_author_view_summaries'].items())
    return (HEADING + '\n\n'
        f"GPU{execution['devices']}隔离PID{execution['pid']}/birth{execution['pid_birth']}已完成并退出，"
        f"从启动到完成{execution['elapsed_from_job_start_seconds']:.3f}秒。"
        f"checkpoint25原成功首response {coverage['unique_UID_source_steps']} UID，"
        f"原VERL padding后{coverage['transport_cases']}运输槽，保留副本差与首个UID主记录。"
        f"原作者k=20累计删除两视图，每rank原生评分调用{calls}；"
        '使用保存的d_from_A，不重算DT。原模型及完整checkpoint加载一次，评分B1与固定actor训练B4分开，'
        'LoRA8/16保持；DT/反向/optimizer/scheduler/rollout均0。\n\n'
        f"实际runner {runner}，原作者metric {author}，完整导入路径和其余源SHA见绑定原件。"
        + view_text + '。原RISE/MAS及曲线由作者函数原样返回；'
        f"两视图原归一化全零UID数量为{zero_counts}，负full-EOS端点使初值clip到0并维持running-min；"
        'RISE很小不能自动解释为中间删除无效或归因优秀。'
        '同集合累计效应与单token条件删除不是同一统计对象，原非有限指标保留为缺失描述，未加纠偏或容差。'
        '完整raw、分析、两图、CPU来源合同、独立复核及物理资源回执绑定到既有degradation报告。\n\n'
        '另绑定已有14 B4同端点root/seed/32层/final账本和原VERL GAE/GRPO标准化与DT raw优势的来源审计。'
        '这些只解释现有信号的口径，不修改PLAN、Q/V/A、原PPO或参数，'
        '不称作者容差通过、世界反事实准确或弱任务梯度已修复；正式TextCraft仍保持停止。\n\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=LOCAL)
    args = parser.parse_args()
    current = args.directory.resolve()
    observation, _, _, analysis = completed_observation(current)
    ledger_path = LEDGER_DIRECTORY / 'native-joint-signal-ledger.json'
    ledger = read(ledger_path)
    unit_path = LEDGER_DIRECTORY / 'official-advantage-unit-source-audit.json'
    unit = read(unit_path)
    observation['existing_same_pair_joint_ledger'] = dict(
        source=source(ledger_path), analysis_source=source(AUDIT / 'analyze_saved_native_joint_signal.py'),
        coverage=ledger['coverage'],
        first_response_unique_UID_summary=ledger['same_pair_first_response_unique_UID_summary'],
        scope='Previously saved full-pair scalar ledger; no new model or individual token-counterfactual validation.')
    observation['official_advantage_unit_source_audit'] = dict(
        source=source(unit_path), scope=unit['scope'],
        shared_loss_interpretation=unit['shared_loss_interpretation'],
        scope_limit='Source/units audit only; it does not add advantage normalization or change the fixed Q/V/A.')
    path = REPO / 'experiments/rl/results_textcraft_learning_degradation_20261006.json'
    report = read(path)
    original_sources = list(report['sources'])
    known = {item['path']: item for item in original_sources}
    added = []
    for item_path in provenance_paths(current):
        item = source(item_path)
        if item['path'] in known:
            assert known[item['path']]['sha256'] == item['sha256'], item['path']
        else:
            added.append(item)
            known[item['path']] = item
    # Append without replacing historical identities or production/method status.
    report['sources'] = original_sources + added
    report[FIELD] = observation
    report['updated_unix'] = time.time()
    encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    runtime = REPO / 'experiments/rl/RUNTIME_RECORD.md'
    prior_runtime = runtime.read_bytes()
    updated_runtime = None
    if HEADING.encode() not in prior_runtime:
        title, tail = prior_runtime.split(b'\n', 1)
        updated_runtime = title + b'\n\n' + runtime_entry(observation, analysis).encode() + tail
    path.write_text(encoded, encoding='utf-8')
    if updated_runtime is not None:
        runtime.write_bytes(updated_runtime)
    print(json.dumps(dict(report=source(path), runtime=source(runtime),
        previous_sources=len(original_sources), appended_sources=len(added),
        sources=len(report['sources']), observation_field=FIELD), ensure_ascii=False))


if __name__ == '__main__':
    main()

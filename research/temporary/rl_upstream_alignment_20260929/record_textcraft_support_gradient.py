"""Bind native success/failure support gradients without changing training."""
import hashlib
import json
import math
from pathlib import Path
import time

from stage_environment_entry import AUDIT, REPO
from record_textcraft_native_adam import source


if __name__ == '__main__':
    directory = AUDIT / 'textcraft-degradation-20261005/grpo-support-gradient-20261006'
    current = directory / 'v2'
    completed = json.loads((current / 'completed.json').read_bytes())
    job = json.loads((current / 'job.json').read_bytes())
    resource = json.loads((current / 'completion-resource-status.json').read_bytes())
    assert completed['input_sha256'] == '45ae51e3e18f206e238a1f1e934eaf9a4074c9361dabdab83bb3e00a2b4b084d'
    assert all(completed[key] == 0 for key in ('sampling_calls', 'DT_calls', 'optimizer_steps', 'scheduler_steps'))
    assert resource['completed'] and not resource['pid_exists']
    assert resource['job_pid'] == job['pid'] and resource['job_pid_birth'] == job['pid_birth']
    groups = {}
    for support in ('q_nonzero', 'q_zero'):
        raw = [json.loads((current / f'rank{rank}-{support}-gradients.json').read_bytes()) for rank in (0, 1)]
        assert raw[0]['gradient_statistics'] == raw[1]['gradient_statistics']
        stat = raw[0]['gradient_statistics']
        dots, norms = stat['inner_products'], stat['norms']
        for rank in (0, 1):
            assert raw[rank]['actual_backward_passes'] == 2
            assert raw[rank]['scheduler_step_no_op_calls'] == 1
            assert not raw[rank]['optimizer_step_executed'] and not raw[rank]['scheduler_step_executed']
            assert all(len(item['microbatch_losses']) == 8 for item in raw[rank]['passes'].values())
        groups[support] = dict(
            original_rank_receipts=[source(current / f'rank{rank}-{support}-gradients.json') for rank in (0, 1)],
            gradient_statistics=stat,
            cosine_DT_with_selected_GRPO=dots['dt_pg:grpo_pg'] / (norms['dt_pg'] * norms['grpo_pg']),
            DT_over_selected_GRPO_norm=norms['dt_pg'] / norms['grpo_pg'],
            selected_GRPO_over_DT_norm=norms['grpo_pg'] / norms['dt_pg'],
            pass_wall_seconds_by_rank={str(rank): {name: item['elapsed_seconds'] for name, item in raw[rank]['passes'].items()} for rank in (0, 1)},
            native_clipfrac_max_by_rank={str(rank): {name: max(item['pg_clipfrac'] for item in value['microbatch_losses']) for name, value in raw[rank]['passes'].items()} for rank in (0, 1)})
    before, after = (groups[s]['gradient_statistics']['norms']['dt_pg'] for s in ('q_nonzero', 'q_zero'))
    path = REPO / 'experiments/rl/results_textcraft_learning_degradation_20261006.json'
    report = json.loads(path.read_bytes())
    paths = [AUDIT / name for name in ('observe_textcraft_grpo_support_gradients.py',
        'verify_textcraft_grpo_support_gradients.py', 'stage_textcraft_grpo_support_gradients.py', Path(__file__).name)]
    for version in ('v1', 'v2'):
        root = directory / version
        paths += [item for item in root.iterdir() if item.is_file()]
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
    report['native_GRPO_support_gradient_observation'] = dict(
        role='Completed isolated diagnostic, not a production repair or a new numerical acceptance test.',
        execution=dict(pid=job['pid'], pid_birth=job['pid_birth'], devices=[4, 5],
            elapsed_from_job_start_seconds=completed['completed_unix'] - job['started_unix'],
            completed=source(current / 'completed.json'), final_resource=resource),
        groups=groups,
        repeated_DT=dict(before_norm=before, after_norm=after,
            relative_norm_change=(after - before) / before,
            scope='Two repetitions in this job; descriptive only, without a tolerance/equality gate.'),
        operations=dict(model_initializations=1, original_complete_checkpoint_loads=1,
            native_backward_passes_per_rank=4, native_B4_backward_calls_per_rank=32,
            native_optimizer_step_no_op_calls_per_rank=4, native_scheduler_step_no_op_calls_per_rank=2,
            optimizer_updates=0, scheduler_updates=0, rollouts=0, DT_calls=0),
        independent_reviews=[source(current / 'independent-support-gradient-review.json'),
            source(current / 'independent-gradient-support-review.json')],
        failed_v1=dict(job=source(directory / 'v1/job.json'),
            failure=source(directory / 'v1/failure-status.json'),
            cause='A fresh Ray worker lacked the existing observer helper PYTHONPATH. Failure preceded original observer/backward. V2 changes only the diagnostic import path and verifies it in a fresh child; no PPO/DT source was replaced.'),
        original_contract='Saved64/checkpoint25; global64->rank32->8B4/8, full original masks and complete DT A retained. Only saved official GRPO coefficients outside selected Q support are zeroed. Original VERL loss, backward, clip and worker lifecycle retained; step/scheduler no-op diagnostic only.',
        interpretation='The DT direction remains nearly orthogonal to the success-only GRPO gradient; the absence of failure coefficients is not the sole explanation of the discrepancy. This does not make GRPO an oracle or establish whole-attribution quality.',
        limits=['No success/failure GRPO cross Gram or raw gradient vector is saved; norms cannot supply an exact sum/cancellation decomposition.',
            'Previous complete-GRPO gradients are a different run; do not combine their dots or norms with this run.',
            'Support counts in raw files include padding/observations; they are not action-token populations or loss denominators.',
            'Tiny measured native PPO clip fractions are nonzero in this run; the old all-zero-clip observation is not inherited.',
            'No credit scaling, entropy/configuration change, official tolerance extension, cumulative-deletion metric or formal TextCraft restart was performed.'])
    report['status'] = 'weak_task_signal_and_support_direction_measured_semantic_query_repair_prepared_no_learning_repair_deployed'
    report['updated_unix'] = time.time()
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    runtime = REPO / 'experiments/rl/RUNTIME_RECORD.md'
    heading = '## 2026-10-06 原任务梯度的成功/失败支持对照'
    raw = runtime.read_bytes()
    if heading.encode() not in raw:
        entry = (heading + '\n\n'
            'GPU4/5隔离PID3747980/birth1791254337.14已完成并退出，原64条/checkpoint25/LoRA8/16/每卡B4不变。'
            '复用原VERL observer，每rank两个支持组各两个完整local32反向（8B4/8），原clip执行；'
            'optimizer no-op4/scheduler no-op2，真实更新、rollout、DT重算均0。只把原GRPO系数在指定Q支持之外置零，'
            '完整DT A和原loss_mask/分母保留；未改loss。原助手PYTHONPATH导致v1在observer前失败，'
            'v2仅修诊断搜索路径并在fresh subprocess核实际导入，保留两版本原件。\n\n'
            '成功支持同组原SUM Gram：DT norm .00013230308934、GRPO成功norm .03067111740664，'
            'cos +.01494206473（范数约1/231.82）；失败支持DT norm .00013225878060、GRPO失败norm .02704332881123，'
            'cos -.00593101386。两rank原Gram相同；重复DT norm变化-.03349%仅描述，不新增容差。'
            '新run tiny clipfrac非0，不沿用旧zero-clip结论；没有成功/失败互积和raw gradients，'
            '不拼精确总梯度、不混旧run。失败惩罚缺失不能单独解释成功组的方向差；也不把GRPO当oracle。\n\n'
            '实际耗时从job start约722.60秒；结束后4/5各860MiB、无诊断GPU进程，主机可用452327534592字节。'
            'source、import、配置、输入、失败原件、完成marker、四rank raw和两份独立review绑定现有degradation结果。'
            '这是弱任务信号调查，不是生产修复；afe59dd查询语义修复仍prepared-only，TextCraft保持停止，'
            'PLAN/c9cd147/原VERL未变。整体归因仍按原论文累计删除/RISE/MAS评价。\n\n').encode()
        title, tail = raw.split(b'\n', 1)
        runtime.write_bytes(title + b'\n\n' + entry + tail)
    print(json.dumps(dict(report=source(path), runtime=source(runtime), sources=len(report['sources']))))

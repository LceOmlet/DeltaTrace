"""Bind completed CPU source/data diagnostics to the existing result record.

No model, DT, reward, advantage, loss or optimizer code is called here.
This records descriptive evidence; it neither deploys a repair nor defines
an alternative credit-assignment method or a numerical acceptance threshold.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import time

from stage_environment_entry import AUDIT, REPO
from record_textcraft_native_adam import source


def read(path):
    return json.loads(Path(path).read_bytes())


def record():
    directory = AUDIT / 'textcraft-degradation-20261005'
    paired = directory / 'readout-quality-20261006/v2/encoding-conditioning-audit.json'
    review = paired.with_name('independent-encoding-conditioning-review.json')
    budget = directory / 'return-forecast-budget-source-audit-20261006.json'
    budget_review = directory / 'independent-return-forecast-budget-review-20261006.json'
    analyzed, reviewed, budget_audit = read(paired), read(review), read(budget)
    read(budget_review)
    # Verify source identity, not a learned-value or tolerance requirement.
    for item in analyzed['source_files'].values():
        assert source(Path(item['path']))['sha256'] == item['sha256']
    assert source(AUDIT / 'analyze_textcraft_encoding_conditioning.py')['sha256'] == analyzed['analysis_source']['sha256']
    assert analyzed['operations'] == dict(model_calls=0, DT_calls=0, backward_calls=0, optimizer_steps=0)
    assert reviewed['operations']['remote_calls'] == 0
    assert reviewed['operations']['production_changes'] == 0
    assert all(value == 0 for value in budget_audit['operations'].values())
    for item in budget_audit['sources']:
        assert source(REPO / item['path'])['sha256'] == item['sha256'], item['path']
    for item in budget_audit['frozen_git_sources'].values():
        blob = subprocess.check_output(['git', 'show', item['git_blob']], cwd=REPO)
        assert hashlib.sha256(blob).hexdigest() == item['sha256']

    result = REPO / 'experiments/rl/results_textcraft_learning_degradation_20261006.json'
    report = read(result)
    protected = {key: report[key] for key in ('status', 'production_status', 'fixed_configuration')}
    plan_before = source(REPO / 'experiments/rl/PLAN.md')
    paths = [paired, review, budget, budget_review,
             AUDIT / 'analyze_textcraft_encoding_conditioning.py', Path(__file__)]
    paths += [REPO / item['path'] for item in budget_audit['sources']]
    known = {item['path']: item for item in report['sources']}
    for path in paths:
        item = source(path)
        if item['path'] in known:
            assert known[item['path']]['sha256'] == item['sha256'], item['path']
        else:
            report['sources'].append(item)
            known[item['path']] = item
    for item in report['sources']:
        assert source(REPO / item['path'])['sha256'] == item['sha256'], item['path']

    report['reward_readout_conditioning_CPU_diagnostics'] = dict(
        role='Completed CPU read-only existing-record and source investigation; no learning repair.',
        equivalent_label_conditioning=source(paired), independent_review=source(review),
        forecast_budget_source_audit=source(budget),
        independent_budget_review=source(budget_review),
        observations=dict(population=analyzed['population'],
            constant_shift=analyzed['descriptive_constant_log_odds_shift'],
            within_prompt_separation=analyzed['within_prompt_separation'],
            source_audit_scope=budget_audit.get('scope')),
        interpretation=[
            'The recorded paired encoding effect is not only a common additive log-odds offset; factual/EOS and within-prompt variation remains.',
            'The 103 comparisons are within8 actual prompt groups, not103 independent tasks; root-sign sensitivity is not a token-credit sign-error rate.',
            'The official rollout applies 10752/10240 training-carrier truncation after collection; these numbers are not the verified environment-loop stopping budget.',
            'Missing those carrier lengths in the forecast does not prove a termination-conditioning defect. Actual transport overflow is a separate exception boundary.',
            'No encoding, constant subtraction, averaging, whitening, scaling or loss-coefficient change is selected or deployed.'],
        operations=dict(new_model_calls=0, new_DT_calls=0, new_backward_calls=0,
            new_optimizer_updates=0, remote_calls=0, production_changes=0),
        preserved_plan=plan_before)
    report['updated_unix'] = time.time()
    assert all(report[key] == value for key, value in protected.items())
    assert source(REPO / 'experiments/rl/PLAN.md') == plan_before

    runtime = REPO / 'experiments/rl/RUNTIME_RECORD.md'
    raw = runtime.read_bytes()
    heading = '## 2026-10-06 奖励读出的条件交互与预算来源审计'
    entry = (heading + '\n\n'
        '仅本机CPU复用已有64条配对概率和冻结owner源码，0模型/DT/反向/更新/远端调用。'
        '原事实成功概率的103个同prompt成功/失败pair AUC=0.514563，交换编码=0.475728；'
        '每个prompt去其fact+EOS共同log-odds编码偏移后，128点残余RMS=0.649023。'
        '这些描述排除仅一个常数标签偏移，不识别全部弱PG来源，不选择交换编码为修复；'
        '29/64反号属于完整response端点根差，不是token优势错误率。独立原件复算一致。\n\n'
        '冻结AgentGym rollout原while以max_rounds/done推进，10752/10240在collect结束后'
        '由原truncate_output_ids裁训练载体。未在forecast写这两个数字不能直接称漏了环境终止预算；'
        '原transport的32256 prompt超限走官方truncation=error，异常边界及既有数据的动态证据范围'
        '在return-forecast-budget-source-audit-20261006.json单独记录，未改停止策略。\n\n'
        'source与独立复核绑定既有results_textcraft_learning_degradation_20261006.json的'
        'reward_readout_conditioning_CPU_diagnostics。生产版本/status/PLAN、Q/V/A、原PPO、'
        'LoRA8/16与每卡B4均保持，TextCraft未恢复，未运行待答归一化候选。\n\n')
    title, tail = raw.split(b'\n', 1)
    new_runtime = raw if heading.encode() in raw else title + b'\n\n' + entry.encode() + tail
    result.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    if new_runtime != raw:
        runtime.write_bytes(new_runtime)
    print(json.dumps(dict(result=source(result), runtime=source(runtime),
        sources=len(report['sources']), preserved=protected['status']), ensure_ascii=False))


if __name__ == '__main__':
    record()

"""Bind CPU-only weak-task-signal analyses to the existing diagnostic record.

This records observations, not another credit method or a production patch.
The original Q/V/A, official actor, and existing author curves stay unchanged.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from stage_environment_entry import AUDIT, REPO
from record_textcraft_native_adam import source


FIELD = 'weak_task_signal_cpu_followup'
HEADING = '## 2026-10-06 弱任务梯度：长度、事件读出及原优势预处理'
BASE = AUDIT / 'textcraft-degradation-20261005'


def read(path):
    return json.loads(Path(path).read_bytes())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--readout', type=Path, required=True)
    parser.add_argument('--owner', type=Path, required=True)
    parser.add_argument('--readout-script', type=Path, required=True)
    parser.add_argument('--review', type=Path, required=True)
    args = parser.parse_args()
    length_path = BASE / 'native-minibatch-v4/response-credit-length-decomposition.json'
    magnitude_path = BASE / 'native-minibatch-v4/task-signal-magnitude-source-audit.json'
    magnitude = read(magnitude_path)
    length = read(length_path)
    readout = read(args.readout)
    report_path = REPO / 'experiments/rl/results_textcraft_learning_degradation_20261006.json'
    report = read(report_path)
    if FIELD in report:
        raise RuntimeError('This dated observation is already recorded; preserve its identity.')
    prior_status = report['status']
    prior_production = dict(report['production_status'])
    paths = [length_path, args.readout, args.owner, args.readout_script, args.review,
             AUDIT / 'analyze_textcraft_response_credit_length.py', Path(__file__)]
    for path in paths:
        if path.suffix == '.json':
            read(path)
    report[FIELD] = dict(
        role='CPU-only analysis of saved actual native64; no method or runtime change.',
        observed_native_task_signal=magnitude['gradient_account'],
        observed_native_coefficients=magnitude['coefficient_source_account'],
        response_length_accounting=source(length_path),
        response_length_observations=dict(
            within_UID_log_length_associations=length['all_G1_responses']['within_UID_log_length_vs'],
            actual_carrier_support=length['whole_native64_carrier_token_lengths'],
            scope='Observed associations, not a DT-only hidden divisor or a causal explanation of the gradient ratio.'),
        grouped_event_readout_accounting=source(args.readout),
        grouped_event_readout_observations=dict(
            actual_groups=readout['group_contract'],
            within_group=readout['within_prompt_aggregate'],
            paired_label_response=readout['paired_label_response'],
            scope='Full-response event readout, not individual-token sign errors or task-gradient reversal; G0 sampled task coefficients remain zero.'),
        official_advantage_preprocessing_and_on_policy_source_audit=source(args.owner),
        independent_CPU_review=source(args.review),
        overall_attribution_evaluation='Existing completed author cumulative deletion and RISE/MAS remain authoritative; no single-token quality verdict.',
        repair_status=dict(
            normalization='Not implemented or tested. Pending human decision because PLAN forbids normalization; owner call would center as well as rescale training coefficients.',
            readout='Measured encoding sensitivity and weak within-task realized-outcome separation; no readout repair is accepted or deployed.',
            query_clock_candidate='Prepared-only semantic correction previously measured; not a demonstrated gradient or learning repair.'),
        operations=dict(model=0, GPU=0, DT=0, backward=0, optimizer=0,
                        production_changes=0, credit_changes=0))
    known = {item['path']: item for item in report['sources']}
    for path in paths:
        item = source(path)
        if item['path'] in known:
            assert item['sha256'] == known[item['path']]['sha256'], item['path']
        else:
            report['sources'].append(item)
            known[item['path']] = item
    report['updated_unix'] = time.time()
    assert report['status'] == prior_status and report['production_status'] == prior_production
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                      allow_nan=False) + '\n', encoding='utf-8')
    runtime = REPO / 'experiments/rl/RUNTIME_RECORD.md'
    prior = runtime.read_bytes()
    if HEADING.encode() not in prior:
        title, tail = prior.split(b'\n', 1)
        entry = f'''\n\n{HEADING}

仅CPU复用原checkpoint25/iteration26的真实64条、8个原prompt组各8条及186成功response原件。长度分解未证明固定1/L是完整人口的主因，expm1组合未再次显著缩幅；原1/211仍是preclip任务梯度范数比，不是Adam更新或归因质量比。原同任务内奖励预测与等价标签交换的响应已绑定新原件，不能把读出缺陷与DT整体归因评价混同。

原VERL源码的GAE接口在central batch/action-token mask上调用masked_whiten，GRPO按原组标准化；本次DT仍直接使用raw A，未执行GAE。masked_whiten默认同时中心化和缩放，不能称为只乘一个系数。原loss mask继续排除O/padding。用户尚未批准改变PLAN的禁止归一化条款，因此未实现或测试该候选；不改熵、Q/V/A、LoRA8/16、每卡B4或部署路径。实际同策略载体来源、mask语义、已存数值分解见results_textcraft_learning_degradation_20261006.json的{FIELD}。本次模型/GPU/DT/反向/更新/正式启动均0；先前作者累计删除/RISE/MAS结果保持。

'''
        runtime.write_bytes(title + entry.encode() + tail)
    print(json.dumps(dict(report=source(report_path), runtime=source(runtime),
                          source_count=len(report['sources']), field=FIELD),
                     ensure_ascii=False))


if __name__ == '__main__':
    main()

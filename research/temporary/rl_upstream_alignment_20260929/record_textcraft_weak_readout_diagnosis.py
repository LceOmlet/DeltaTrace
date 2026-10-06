"""Bind weak-signal source diagnostics to the existing result and runtime ledger."""
import hashlib
import json
from pathlib import Path
import time

from stage_environment_entry import AUDIT
from stage_textcraft_label_encoding import LOCAL
from record_textcraft_native_adam import source, REPO


if __name__ == '__main__':
    report_path = REPO / 'experiments/rl/results_textcraft_learning_degradation_20261006.json'
    report = json.loads(report_path.read_bytes())
    analysis_path = LOCAL / 'label-encoding-analysis.json'
    label = json.loads(analysis_path.read_bytes())
    completed = json.loads((LOCAL / 'completed.json').read_bytes())
    job = json.loads((LOCAL / 'job.json').read_bytes())
    assert label['calls'] == dict(native_forwards=64, finite_attributions=0, backward=0, optimizer=0, scheduler=0)
    assert label['cases'] == 64 and label['task_groups'] == 8
    assert all(value == 0 for key, value in completed.items()
               if key in ('optimizer_steps', 'scheduler_steps', 'backward_calls', 'finite_trace_calls'))
    saturation_path = AUDIT / 'textcraft-degradation-20261005/native-minibatch-v4/binary-readout-saturation-source-audit.json'
    history_path = AUDIT / 'textcraft-degradation-20261005/formal-support-feedback-history-audit-20261006.json'
    saturation = json.loads(saturation_path.read_bytes())
    history = json.loads(history_path.read_bytes())
    failed = LOCAL.parent / 'v1'
    paths = [p for directory in (LOCAL, failed) for p in directory.iterdir()
             if p.is_file() and p.suffix in ('.json', '.yaml', '.log', '.txt', '.py')]
    paths += [AUDIT / name for name in ('verify_textcraft_label_encoding.py',
        'stage_textcraft_label_encoding.py', 'observe_textcraft_label_encoding.py',
        'stop_textcraft_label_import.py', 'analyze_textcraft_label_encoding.py', Path(__file__).name)]
    paths += [saturation_path, history_path]
    recorded = {item['path']: item for item in report['sources']}
    for path in paths:
        item = source(path)
        if item['path'] in recorded:
            assert item['sha256'] == recorded[item['path']]['sha256'], item['path']
        else:
            report['sources'].append(item)
            recorded[item['path']] = item
    for item in report['sources']:
        assert hashlib.sha256((REPO / item['path']).read_bytes()).hexdigest() == item['sha256'], item['path']
    report['binary_readout_saturation_diagnostic'] = dict(source=source(saturation_path),
        scope=saturation['scope'], interpretation='Correct binary log-prob sensitivity, not an extra divisor or numerical repair. No margin substitution or scaling follows from it.')
    report['historical_reward_support_feedback'] = dict(source=source(history_path),
        scope=history['scope'], interpretation='Saved formal success support shrinks while original action entropy rises. Historical component gradients were not saved; checkpoint25 ratio is not extended to history.')
    report['paired_reward_label_encoding'] = dict(source=source(analysis_path),
        scope=label['scope'], execution=dict(pid=job['pid'], pid_birth=job['pid_birth'],
            devices=job['devices'], started_unix=job['started_unix'], completed_unix=completed['completed_unix'],
            wall_seconds=completed['completed_unix']-job['started_unix'], native_calls=64,
            DT_calls=0, backward_calls=0, optimizer_steps=0),
        sources=label['deployed_sources'], pools=label['pools'], task_group_pools=label['task_group_pools'],
        previous_original_encoding=label['previous_native_readout_original_encoding'],
        budget=label['shape_budget'], dtypes=label['runtime_dtypes'], limitations=label['limitations'])
    report['status'] = 'weak_task_signal_sources_and_paired_label_readout_observed_no_production_repair'
    report['updated_unix'] = time.time()
    report['interpretation']['priority'] = 'Weak effective task signal and reward-readout semantics; original actor, optimizer and numerical tolerances remain unchanged.'
    report['interpretation']['next_owner_check'] = (
        'Use this paired reward-readout encoding evidence to choose a demonstrated target-interface repair, '
        'if one is supported. Keep the original token Q/V/A and upstream PPO fixed. Single-token '
        'probes cannot substitute for original cumulative deletion curves/RISE/MAS or establish '
        'overall attribution failure. Small exact credit and misestimated credit remain distinct.')
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    runtime = REPO / 'experiments/rl/RUNTIME_RECORD.md'
    heading = '## 2026-10-06 弱信用来源：类别读出配对与历史支持集'
    raw = runtime.read_bytes()
    if heading.encode() not in raw:
        pool = label['pools']['all']
        original = pool['mappings']['original']['variants']['factual']
        swapped = pool['mappings']['swapped']['variants']['factual']
        entry = (heading+'\n\n'+
            f"隔离v2 PID{job['pid']}/birth{job['pid_birth']}完成，GPU4/5，"
            f"{completed['completed_unix']-job['started_unix']:.3f}秒。原checkpoint25、64条真实首轮输入、"
            '原VERL worker/config、LoRA8/16与每卡B4均保留。只由原RewardAlphabet.query_ids交换'
            '两个类别label ID，values/meanings顺序、回报及轨迹ID不变；按实际query等长配对。'
            '64 native B4前向，DT/反向/optimizer/scheduler均0。\n\n'+
            f"原/交换标签事实读出Brier={original['brier']:.8f}/{swapped['brier']:.8f}。"
            '根端点变化不是token优势，类别敏感性不自动证明整个弱梯度原因或历史因果。'
            '复用原native精度、head与卸载；旧同编码LP差只作描述，不新增官方容差。'
            '原论文累计删除/RISE/MAS仍是整体归因评价接口；single probes不替代。\n\n'+
            'v1因新Ray worker未继承旧reader helper搜索路径，在模型初始化前停止并保存日志。'
            'v2只把既有helper目录加入诊断PYTHONPATH，未复制owner或修改生产。'
            '新审计绑定二类logprob饱和与正式43/93奖励支持集；历史分项梯度未保存，不将'
            '1/211扩展为历史参数更新比例，不从缺失报告造零。生产信用/熵系数/采样/任务配置不变，'
            'TextCraft未恢复。\n\n').encode()
        title, tail = raw.split(b'\n', 1)
        runtime.write_bytes(title+b'\n\n'+entry+tail)
    print(json.dumps(dict(report=source(report_path), runtime=source(runtime), sources=len(report['sources']))))

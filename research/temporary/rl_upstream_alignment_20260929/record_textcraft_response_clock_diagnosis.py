"""Bind query-clock and frozen native objective evidence without method edits."""
import hashlib
import json
from pathlib import Path
import time
from stage_environment_entry import AUDIT
from stage_textcraft_response_clock import LOCAL
from record_textcraft_native_adam import source, REPO


if __name__ == '__main__':
    report_path = REPO / 'experiments/rl/results_textcraft_learning_degradation_20261006.json'
    report = json.loads(report_path.read_bytes())
    analysis = json.loads((LOCAL / 'response-clock-analysis.json').read_bytes())
    complete = json.loads((LOCAL / 'completed.json').read_bytes())
    job = json.loads((LOCAL / 'job.json').read_bytes())
    native = AUDIT / 'textcraft-degradation-20261005/native-adam-20261006/v1'
    objective = json.loads((native / 'native-objective-effect.json').read_bytes())
    contract = AUDIT / 'textcraft-degradation-20261005/readout-quality-20261006/v2/current-action-clock-contract.json'
    assert complete['cases'] == 64 and complete['native_forward_calls'] == 128
    assert all(complete[key] == 0 for key in ('optimizer_steps','scheduler_steps','backward_calls','finite_trace_calls'))
    assert objective['operations'] == dict(model_initializations=0,model_forwards=0,backward=0,optimizer=0,DT=0)
    assert not objective['runtime']['after']['cuda_initialized']
    paths = [p for folder in (LOCAL, LOCAL.parent/'v1') for p in folder.iterdir()
        if p.is_file() and p.suffix in ('.json','.yaml','.txt','.log','.py')]
    paths += [contract, *[p for p in native.iterdir() if p.is_file() and p.name.startswith(('native-objective-effect','independent-native-objective'))]]
    paths += [AUDIT / name for name in ('verify_textcraft_response_clock.py','stage_textcraft_response_clock.py',
        'observe_textcraft_response_clock.py','analyze_textcraft_response_clock.py','analyze_textcraft_native_objective_effect.py',
        'run_textcraft_native_objective_effect.py',Path(__file__).name)]
    recorded = {item['path']:item for item in report['sources']}
    for path in paths:
        item = source(path)
        if item['path'] in recorded:
            assert item['sha256'] == recorded[item['path']]['sha256'], item['path']
        else:
            report['sources'].append(item)
            recorded[item['path']] = item
    for item in report['sources']:
        assert hashlib.sha256((REPO/item['path']).read_bytes()).hexdigest() == item['sha256'], item['path']
    report['paired_current_response_clock'] = dict(contract=source(contract), analysis=source(LOCAL/'response-clock-analysis.json'),
        execution=dict(pid=job['pid'],pid_birth=job['pid_birth'],devices=job['devices'],
            wall_seconds=complete['completed_unix']-job['started_unix'], native_forward_calls=128,
            native_readout_batch=2, actor_config_microbatch=4, DT_calls=0, backward=0, updates=0),
        pools=analysis['pools'], within_group_discrimination=analysis['within_group_discrimination'],
        same_encoding_layout_observation=analysis['previous_label_encoding_unexchanged_original'],
        interpretation='The owner executes the response as emitted; query-clock clarification is a semantics candidate. Paired results are mixed, not a weak-token-PG repair. B2/B4 differences are descriptive, not an official operator tolerance test.',
        production_query_changed=False, full_DT_quality_evaluated=False)
    report['native_frozen_objective_effect'] = dict(source=source(native/'native-objective-effect.json'),
        scope=objective['scope'], original_partition=objective['original_partition'],
        old_policy_source=objective['old_policy_source'], operations=objective['operations'],
        changes={branch:value['changes_after_minus_before'] for branch,value in objective['branches'].items()},
        interpretation='The actual DT branch decreases its total frozen loss while increasing its frozen DT-PG loss. This is a checkpoint-local objective observation, not a task success rate, exact Adam component attribution, or numerical tolerance gate.')
    report['status'] = 'weak_task_signal_native_objective_and_query_clock_observed_no_learning_repair_deployed'
    report['updated_unix'] = time.time()
    report['interpretation']['next_owner_check'] = (
        'Keep the original PPO/QVA/config fixed and investigate the weak task signal and its balance with original regularizers and restored optimizer history. '
        'Do not attribute 1/211 to scalar cancellation or a mandatory 1/L law. The query-clock candidate has not improved the overall probability score or established a token-gradient repair. '
        'Any full attribution-quality claim still needs the author cumulative deletion/RISE/MAS protocol; single probes do not replace it. '
        'Use actual B4 native operands for subsequent numerical claims; this B2 readout diagnosis is not that gate.')
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    runtime = REPO / 'experiments/rl/RUNTIME_RECORD.md'
    heading = '## 2026-10-06 当前响应时钟候选与真实更新的冻结目标变化'
    raw = runtime.read_bytes()
    if heading.encode() not in raw:
        delta = objective['branches']['dt']['changes_after_minus_before']
        entry = (heading+'\n\n'+
            '隔离query候选：原环境直接执行已交付response，冻结228afbc7查询却要求补完同一response。'
            '仅在诊断tokenizer入口替换两句，原IDs、EOS、labels、G、sampling、步数和Q/V/A不变。'
            f"v2 PID{job['pid']}/birth{job['pid_birth']}完成，GPU4/5，{complete['completed_unix']-job['started_unix']:.3f}秒；"
            '64真实首轮病例，每rank64次native B2事实/EOS配对，无重复行/补padding；actor/DT配置仍每卡B4。'
            '原/candidate pooled Brier .38691061/.39847212，成功根绝对均值 .15918764/.12382817。'
            '成功/失败评分和相关变化混合，不能称其修复了小token信用或1/211 PG。'
            '原B2与旧B4同编码LP差保存为描述，未扩大FA/FLA容差或加纠偏。'
            'v1仅CPU prepared；review发现分桶会造成rank42/44次FSDP调用及class来源观察错误，未提交GPU。'
            'v2固定同步调用次数并改为method来源，保留v1原始源码和CPU回执。\n\n'+
            '复用已保存的三分支真实Adam before/after LP/H，仅CPU调用原VERL core损失和归约：'
            f"DT branch ΔPG={delta['dt_pg']:+.10e}，Δ(-.001H)={delta['weighted_entropy']:+.10e}，"
            f"Δ(.001KL)={delta['weighted_kl']:+.10e}，Δtotal={delta['dt_total']:+.10e}。"
            '本次总目标下降但冻结DT任务目标上升；同批目标不是成功率，旧Adam历史不是可线性分账的当前熵向量。'
            '保留原old/ref、mask、rank32→B4×8→/8；无模型/预测/DT/反向/更新，约0.535秒CPU统计，RSS约.803GiB。\n\n'+
            '版本角色：上述是已完成的隔离诊断；不是被接受/部署的生产query修复。'
            '原c9cd147 DT、原VERL actor/core、PLAN、LoRA8/16、每卡B4和正式任务配置均未改变。'
            'TextCraft未恢复，未从single probe判整个DT差，未放大/归一/裁剪信用或修改entropy系数。\n\n').encode()
        title,tail = raw.split(b'\n',1)
        runtime.write_bytes(title+b'\n\n'+entry+tail)
    print(json.dumps(dict(report=source(report_path),runtime=source(runtime),sources=len(report['sources']))))

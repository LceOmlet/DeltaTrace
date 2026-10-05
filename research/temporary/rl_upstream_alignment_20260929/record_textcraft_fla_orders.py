"""Bind actual unchanged-order observations to the existing runtime record."""
import hashlib
import json
from pathlib import Path
import time

from analyze_textcraft_native_readout import source
from stage_textcraft_fla_orders import LOCAL


REPO = Path(__file__).resolve().parents[3]
READOUT = Path(__file__).parent / 'textcraft-degradation-20261005/readout-quality-20261006/v2'


if __name__ == '__main__':
    path = REPO / 'experiments/rl/results_textcraft_learning_degradation_20261006.json'
    report = json.loads(path.read_bytes())
    analysis_path = LOCAL / 'fla-orders-analysis.json'
    analysis = json.loads(analysis_path.read_bytes())
    job = json.loads((LOCAL / 'job.json').read_bytes())
    completed = json.loads((LOCAL / 'completed.json').read_bytes())
    assert analysis['coverage']['unique_probes'] == 42
    assert completed['finite_trace_calls'] == 14
    assert completed['backward_calls'] == completed['optimizer_steps'] == completed['scheduler_steps'] == 0
    paths = list(LOCAL.glob('*.json')) + list(LOCAL.glob('*.stdout.txt')) + [
        LOCAL / 'effective-config.yaml', LOCAL / 'diagnostic.log', LOCAL / 'physical-before-start.txt',
        LOCAL / 'independent_orders_review.py']
    paths.extend(Path(__file__).with_name(name) for name in (
        'observe_textcraft_fla_orders.py', 'verify_textcraft_fla_orders.py',
        'stage_textcraft_fla_orders.py', 'observe_textcraft_fla_orders_job.py',
        'analyze_textcraft_fla_orders.py', Path(__file__).name))
    paths.extend(READOUT / name for name in ('independent_readout_groups.py','independent-readout-groups.json'))
    paths.extend(READOUT.glob('native-b4-denominator*'))
    paths.extend(READOUT / name for name in (
        'analyze_saved_native_b4_denominators.py', 'run_saved_native_b4_denominators.py'))
    b4_path = READOUT / 'native-b4-denominator-analysis.json'
    b4 = json.loads(b4_path.read_bytes())
    assert b4['coverage']['rows'] == 64 and b4['coverage']['microbatches'] == 16
    assert all(value == 0 for value in b4['operations'].values())
    assert not b4['runtime']['cuda_initialized'] and not b4['runtime']['distributed_initialized']
    recorded = {item['path']:item for item in report['sources']}
    for item_path in paths:
        item = source(item_path)
        item['path'] = item_path.relative_to(REPO).as_posix()
        if item['path'] in recorded:
            recorded[item['path']].update(item)
        else:
            report['sources'].append(item)
            recorded[item['path']] = item
    report['endpoint_orders'] = dict(scope=analysis['scope'], analysis=source(analysis_path),
        coverage=analysis['coverage'], layer_summaries=analysis['order_layer_summaries'],
        actual_dtypes=analysis['observed_order_tensor_dtypes'],
        order_payload=analysis['CPU_order_coefficient_bank_payload_statistics'],
        execution=dict(pid=job['pid'],pid_birth=job['pid_birth'],devices=job['devices'],
            started_unix=job['started_unix'],completed_unix=completed['completed_unix'],
            wall_seconds=completed['completed_unix']-job['started_unix'],
            full_finite_calls=14,single_root_calls=28,actor_backward_calls=0,optimizer_steps=0,scheduler_steps=0),
        provenance=dict(prepared=source(LOCAL / 'prepared.json'),
            independent_raw_review=source(LOCAL / 'independent-orders-review.json')),
        limitations=analysis['limitations'],
        production_state='Only passive original-return observations. No averaging rule, DT/PPO formula, dtype, training setting or weights changed; TextCraft remains stopped.')
    report['endpoint_orders']['findings'] = dict(
        conditional_interval=(
            'F0/F1 are computed on the full-response EOS/factual pair; their contractions use '
            'single-token EOS/factual operands. F-Y is an allocation-estimate discrepancy, '
            'not a same-endpoint identity expected to be zero or an official FLA kernel tolerance test.'),
        layer_comparison=[dict(decoder_index=row['decoder_index'], unique_probes=row['unique_n'],
            order0_mean_absolute_distance=row['unique_probe_statistics']['R0']['abs_mean'],
            order1_mean_absolute_distance=row['unique_probe_statistics']['R1']['abs_mean'],
            average_mean_absolute_distance=row['unique_probe_statistics']['R_average']['abs_mean'],
            order_projection_oppositions=row['unique_probe_order_comparisons']['opposite_order_sign_count'],
            average_farther_than_both=row['unique_probe_order_comparisons']['average_farther_than_both'])
            for row in analysis['order_layer_summaries']],
        inference='Both layers have smaller mean distance after averaging; this does not support removing the original average as a repair.',
        independent_raw_review=source(LOCAL / 'independent-orders-review.json'))
    report['independent_readout_groups'] = dict(source=source(READOUT / 'independent-readout-groups.json'),
        scope='Saved actual64 native first-response predictions joined to original prompt UID. No new model prediction, calibration, target mapping or training change.',
        limitation='First-response predictions only; do not generalize their EOS optimism to all formal response requests or declare every negative credit erroneous.')
    report['native_B4_signal_accounting'] = dict(source=source(b4_path),
        independent_review=source(READOUT / 'native-b4-denominator-independent-review.json'),
        coverage=b4['coverage'], native_accumulation=b4['across_ranks'],
        scalar_comparison=b4['same_policy_saved_scalar_comparison'],
        runtime=b4['runtime'], operations=b4['operations'], limitations=b4['limits'],
        interpretation=(
            'The original B4 loss weighting has 82.07085% zero-Q support and four of sixteen '
            'B4s have zero task PG, while original entropy loss remains active. The success '
            'support also has small credit; dilution alone does not explain all weak signal. '
            'These scalar masses do not replace the previously recorded native gradient norms '
            'or prove the complete historical Adam trajectory. No formula or loss setting changed.'))
    report['updated_unix'] = time.time()
    report['status'] = 'actual_original_endpoint_orders_measured_no_production_repair'
    report['interpretation']['next_owner_check'] = (
        'Investigate the measured small success credit through the existing DT/readout owner: '
        'distinguish same-pair finite arithmetic from joint-to-single allocation approximation '
        'and reward-readout quality. The saved original B4 accounting is complete. '
        'Do not infer gradient magnitude from token mass or call cross-interval F-Y a kernel failure. '
        'No credit scaling, rule substitution, entropy tuning or formal TextCraft restart is justified '
        'by cancellation or closure alone.')
    for item in report['sources']:
        assert hashlib.sha256((REPO / item['path']).read_bytes()).hexdigest() == item['sha256'],item['path']
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    runtime = REPO / 'experiments/rl/RUNTIME_RECORD.md'
    raw = runtime.read_bytes()
    heading = '## 2026-10-06 原 FLA 双顺序：现成系数与相同单删操作数'
    entry = (heading+'\n\n'+
        f"隔离PID{job['pid']}/birth{job['pid_birth']}已完成，GPU4/5，"
        f"{completed['completed_unix']-job['started_unix']:.3f}秒。原7 B4/rank、checkpoint25、"
        'LoRA8/16及冻结runner/producer均不变；14 full finite、28 single root，actor '
        'backward/optimizer/scheduler均0。原avg仍执行一次，其两次原FLA调用和返回对象保留。\n\n'+
        '只在原avg的PY_RETURN读取forward/reverse现成系数。单删阶段复用原观察器已'
        '复制的paired CPU操作数，用原_token_effect做两组标量收缩，不重复native复制、'
        '监测或模型运算。原52 transport按身份明确合并为42 probe/层；两order对同一Y'
        '的差、符号与抵消只是描述，不替换平均规则或放宽官方容差。\n\n'+
        '两层平均后的F−Y平均绝对差均小于各单order，未支持取消平均。这里F来自'
        'full-response EOS/fact，Y来自single-token EOS/fact，区间不同；偏差量化条件'
        '估计质量，不是同端点应归零的恒等式，也不能据此指认FLA kernel超差。\n\n'+
        '原raw、dtype/head/cut/payload、独立复核及64条读出分组统计已关联到'
        'results_textcraft_learning_degradation_20261006.json。生产仍未修复，TextCraft保持停止。\n\n').encode('utf-8')
    if heading.encode('utf-8') not in raw:
        title,tail = raw.split(b'\n',1)
        runtime.write_bytes(title+b'\n\n'+entry+tail)
    b4_heading = '## 2026-10-06 原 B4 任务信号：真实分母与既有熵项'
    raw = runtime.read_bytes()
    if b4_heading.encode('utf-8') not in raw:
        b4_entry = (b4_heading+'\n\n'+
            '仅CPU复用原native-optimizer-minibatch.pkl与原DataProto.load/chunk、原core '
            'agg_loss/compute_policy_loss；global64→rank32→连续B4→/8，不重建采样或logits。'
            f"PID{b4['runtime']['pid']}/birth{b4['runtime']['pid_birth']}，"
            f"{b4['runtime']['elapsed_seconds']:.3f}秒，maxRSS{b4['runtime']['maxRSS_bytes']}字节；"
            'CUDA/distributed未初始化，模型/forward/DT/backward/更新均0。\n\n'+
            '原B4均值口径零Q分母82.07085%，不同于global pooled的86.85622%；'
            '16个B4中4个任务PG为0，原熵项仍在。保留原分母的DT |A|均值0.000523759、'
            'GRPO 0.757718；成功内部信用也小，不只分母稀释。优势mass不等于梯度norm或'
            '方向；原32个PG标量差只描述，未新设/扩展任何官方容差。\n\n'+
            '实际输入/官方import SHA、运行回执和独立逐B4复核绑定到'
            'results_textcraft_learning_degradation_20261006.json。没有缩放优势、改熵系数'
            '或恢复TextCraft；历史完整Adam因果尚未重放。\n\n').encode('utf-8')
        title,tail = raw.split(b'\n',1)
        runtime.write_bytes(title+b'\n\n'+b4_entry+tail)
    print(json.dumps(dict(report=source(path),runtime=source(runtime),sources=len(report['sources']),status=report['status'])))

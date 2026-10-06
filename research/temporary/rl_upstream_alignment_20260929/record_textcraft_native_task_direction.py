"""Bind actual saved-update direction evidence; preserve the accepted method."""
import hashlib
import json
from pathlib import Path
import time

from record_textcraft_native_adam import source, REPO
from stage_environment_entry import AUDIT


if __name__ == '__main__':
    native = AUDIT / 'textcraft-degradation-20261005/native-adam-20261006/v1'
    readout = AUDIT / 'textcraft-degradation-20261005/readout-quality-20261006/v2'
    direction = json.loads((native / 'native-task-direction.json').read_bytes())
    review = json.loads((native / 'independent-task-direction-review.json').read_bytes())
    lengths = json.loads((readout / 'eos-reference-length-audit.json').read_bytes())
    assert all(v == 0 for v in direction['operations'].values())
    assert not direction['runtime']['resources_after']['cuda_initialized']
    assert review['inputs']['native-task-direction.json']['sha256'] == source(native / 'native-task-direction.json')['sha256']
    assert hashlib.sha256((REPO / 'experiments/rl/PLAN.md').read_bytes()).hexdigest() == '32f8379f1390bedf8d092cb9bd7765e39dd9f3dd62945d0249e6e3110248d1d0'
    report_path = REPO / 'experiments/rl/results_textcraft_learning_degradation_20261006.json'
    report = json.loads(report_path.read_bytes())
    paths = [AUDIT / name for name in ('analyze_textcraft_native_task_direction.py',
        'run_textcraft_native_task_direction.py', Path(__file__).name)]
    paths += [p for p in native.iterdir() if p.is_file() and p.name.startswith(('native-task-direction', 'independent-task-direction'))]
    paths += [readout / 'independent_eos_reference_length_audit.py', readout / 'eos-reference-length-audit.json']
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
    ranks = []
    for row in direction['ranks']:
        assert row['common_gradient_support']['complete_parameter_support']
        ranks.append(dict(rank=row['rank'], coordinate_checks=row['coordinate_checks'],
            common_gradient_support=row['common_gradient_support'],
            gradient_norms_on_common_support=row['gradient_norms_on_common_support'],
            actual_parameter_delta_norms=row['actual_parameter_delta_norms'],
            gradient_projection_on_actual_displacements=row['gradient_projection_on_actual_displacements']))
    report['native_actual_task_direction'] = dict(artifact=source(native / 'native-task-direction.json'),
        independent_review=source(native / 'independent-task-direction-review.json'),
        operations=direction['operations'], runtime=direction['runtime'], ranks=ranks,
        scope='Per-rank native FP32 local shards, FP64 descriptive copies, original update/config/checkpoint25. Unclipped same-initial-state total-gradient differences are task proxies, not certified exact standalone components.',
        interpretation='Both DT task-proxy dots with actual DT displacement are negative. This does not support a first-order reversal of current task direction by Adam. The observed finite frozen-PG increase alone cannot establish a bad update. No attribution-quality verdict or new tolerance follows.')
    gram = json.loads((AUDIT / 'textcraft-degradation-20261005/native-minibatch-v4/rank0-minibatch-gradients.json').read_bytes())['gradient_statistics']['inner_products']
    report['native_actual_task_direction']['prior_complete_minibatch_Gram'] = dict(
        dt_self=gram['dt_pg:dt_pg'], dt_entropy=gram['dt_pg:weighted_entropy'], dt_kl=gram['dt_pg:weighted_kl'],
        dt_total=gram['dt_pg:dt_pg']+gram['dt_pg:weighted_entropy']+gram['dt_pg:weighted_kl'],
        scope='Earlier complete-minibatch mesh-SUM scalar Gram; not divided by the new per-rank local-shard norms.')
    report['eos_reference_length_observation'] = dict(artifact=source(readout / 'eos-reference-length-audit.json'),
        geometry=lengths['geometry'], pools={k: lengths[k] for k in ('all','fixed_length_513','shorter_than_513','by_actual_G','within_prompt_centered_pearson','same_prompt_same_length')},
        interpretation='Identical original prompt/query with different EOS source extents changes the readout; it does not alone explain 1/211 or prove the accepted EOS idealization false.',
        operations=lengths['operations'])
    report['status'] = 'weak_task_signal_confirmed_current_Adam_direction_not_reversed_no_learning_repair_deployed'
    report['updated_unix'] = time.time()
    report['native_frozen_objective_effect']['interpretation'] += ' Later saved-gradient projections remain downhill for the current task proxy, so the finite PG increase is not by itself evidence of optimizer reversal.'
    report['interpretation']['next_owner_check'] = (
        'Focus on reward-related readout and the finite estimate feeding the task gradient. Do not infer current task reversal from entropy norm alone or finite frozen-loss increase alone. '
        'Preserve Q/V/A, original PPO/config, and use the author cumulative deletion/RISE/MAS for any whole-attribution claim; single-token probes remain supplementary. '
        'No measured clock/label candidate has yet established a learning repair, and no credit multiplier or entropy tuning was deployed.')
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    runtime = REPO / 'experiments/rl/RUNTIME_RECORD.md'
    heading = '## 2026-10-06 真实更新方向：弱任务信号不等于Adam反向'
    raw = runtime.read_bytes()
    if heading.encode() not in raw:
        entry = (heading+'\n\n'+
            '复用六份原Adam更新的FP32本地LoRA参数/梯度shards，仅CPU描述DT总梯度与当前PG置零对照总梯度的差。'
            '原clip均未触发，496参数初值/布局/支持相同；独立反向的舍入及RNG边界保留，差分不冒充精确单项梯度。'
            '实际DT参数位移与该task proxy内积为-3.65394843e-08/-3.79671514e-08，两rank一阶方向均下降。'
            '先前完整minibatch Gram的gDT·gTotal=+1.94059959e-08，亦未被H/KL反向。'
            '因此仅凭熵范数4.4倍或有限冻结PG升+1.708e-07，不能声称Adam把当前任务方向推反；'
            '训练退化与弱信用仍需奖励相关性证据，不把这一更新代理等同真实成功率。\n\n'+
            'CPU PID3481127/birth1791251847.82，分析3.197秒、RSS约.548GiB、PSS约.261GiB，'
            'CUDA/distributed未初始化；0模型/预测/DT/反向/更新。输出bd1d1651、分析source9bcb5861和独立review1d48c06a绑定既有结果。'
            '另对64原native B4首轮读出按完全相同prompt/query分组：跨原response长度EOS参考成功概率最大差3.9463pp，'
            '相同prompt+长度513的最大差2.72e-06；描述长度依赖，不将其当成211倍原因或新数值门槛。\n\n'+
            '版本角色：仅保存的原始张量分析，无生产修复或新训练。PLAN、c9cd147 DT、原VERL、'
            'LoRA8/16和每卡B4不变；TextCraft保持停止。单token仍仅局部诊断，整体归因评价保留原论文累计删除/RISE/MAS。\n\n').encode()
        title, tail = raw.split(b'\n', 1)
        runtime.write_bytes(title+b'\n\n'+entry+tail)
    print(json.dumps(dict(report=source(report_path), runtime=source(runtime), sources=len(report['sources']))))

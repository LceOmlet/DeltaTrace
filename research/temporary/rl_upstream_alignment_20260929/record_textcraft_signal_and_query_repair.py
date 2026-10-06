"""Bind a semantic query repair and saved-update observations, not a PG repair."""
import hashlib
import json
from pathlib import Path
import time

from stage_environment_entry import AUDIT, REPO
from record_textcraft_native_adam import source


if __name__ == '__main__':
    query = AUDIT / 'textcraft-degradation-20261005/query-clock-repair-20261006/v1'
    native = AUDIT / 'textcraft-degradation-20261005/native-adam-20261006/v1'
    checked = json.loads((query/'query-clock-repair.json').read_bytes())
    signs = json.loads((native/'native-action-signs.json').read_bytes())
    assert checked['matched_queries'] == 64 and checked['pytest_exit_code'] == 0
    assert all(value == 0 for value in checked['operations'].values())
    assert all(value == 0 for value in signs['operations'].values())
    assert hashlib.sha256((REPO/'experiments/rl/PLAN.md').read_bytes()).hexdigest() == '32f8379f1390bedf8d092cb9bd7765e39dd9f3dd62945d0249e6e3110248d1d0'
    path = REPO/'experiments/rl/results_textcraft_learning_degradation_20261006.json'
    report = json.loads(path.read_bytes())
    paths = [AUDIT/name for name in ('verify_reward_query_clock_cpu.py', 'run_reward_query_clock_cpu.py',
        'analyze_textcraft_native_action_signs.py', 'run_textcraft_native_action_signs.py', Path(__file__).name)]
    paths += [p for p in query.iterdir() if p.is_file()]
    paths += [p for p in native.iterdir() if p.is_file() and 'action-signs' in p.name]
    recorded = {row['path']: row for row in report['sources']}
    for item_path in paths:
        item = source(item_path)
        if item['path'] in recorded:
            assert item['sha256'] == recorded[item['path']]['sha256'], item['path']
        else:
            report['sources'].append(item)
            recorded[item['path']] = item
    for item in report['sources']:
        assert hashlib.sha256((REPO/item['path']).read_bytes()).hexdigest() == item['sha256'], item['path']
    report['prepared_query_clock_semantic_repair'] = dict(
        source_patch_commit='afe59dd', production_source=source(REPO/'experiments/rl/reward_readout.py'),
        receipt=source(query/'query-clock-repair.json'),
        independent_review=source(query/'independent-query-repair-review.json'),
        matched_native_candidate_queries=64, adapter_unit_tests_passed=29,
        operations=checked['operations'], resources=checked['resources'],
        scope='Only two query clauses now condition on immediate official execution of the emitted complete response, including a generation-limit stop. Original prefix/target/EOS IDs and Q/V/A remain unchanged. The query grows by 14 tokens in these 64 cases.',
        role='Prepared-only, CPU verified; not deployed. Original numerical owner/runtime and formal jobs untouched.',
        learning_claim='Earlier paired native clock forwards had mixed predictive results. This removes a known semantic mismatch, not a demonstrated weak-PG or learning repair.')
    report['native_finite_action_sign_observation'] = dict(
        artifact=source(native/'native-action-signs.json'),
        independent_review=source(native/'independent-native-action-signs-review.json'),
        original_reduction=signs['original_reduction'],
        statistics=signs['mean_of_two_native_rank_reductions'],
        operations=signs['operations'], runtime=signs['runtime'],
        limits=signs['interpretation_limits'],
        interpretation='DT-minus-zero-current-PG control has adverse finite eval LP responses in the signed subsets under the original full B4 denominator. These weighted contributions are not subgroup conditional means. Actual current task-proxy gradient projections remain downhill; no optimizer reversal or history-wide causality is established.')
    report['confirmed_signal_scale_interpretation'] = dict(
        coefficient_ratio_scope='Original B4 token-mean, local32/B4x8 accumulation, mean of the two native ranks; not pooled tokens or parameter-displacement norms.',
        mechanism='Small d already exists before the g*(-expm1(-d)) combination. Successful-token |A| is approximately 1/443 of official standardized GRPO; zero-return support is 82.07% under the original denominator. GRPO also has failure-negative coefficients, yielding the total coefficient ratio about 1/1447. The actual task-gradient ratio about 1/211 is less attenuated; no added averaging or main vector-cancellation cause was found.',
        current_batch_reward_scope='The native64 diagnostic uses the accepted terminal-reward/slice repair; all 21 successful terminal turns lie within the official retained carrier. The historical dropped-reward defect cannot explain this batch ratio.',
        attribution_scope='Single-token probes remain supplemental. Whole-attribution conclusions require author cumulative-deletion curves and RISE/MAS, with raw endpoint scores. No whole-DT quality verdict follows from the norm ratio.',
        repair_scope='Keep Q/V/A, reward, original PPO and all training parameters. Repair demonstrated event-conditioning/interface errors; investigate event-value discrimination and original full-gradient success/failure components before changing learning behavior. No credit rescaling or entropy tuning is accepted by this observation.')
    report['status'] = 'weak_task_signal_confirmed_semantic_query_repair_prepared_no_learning_repair_deployed'
    report['updated_unix'] = time.time()
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    runtime = REPO/'experiments/rl/RUNTIME_RECORD.md'
    heading = '## 2026-10-06 弱梯度分项与已交付响应查询修复'
    raw = runtime.read_bytes()
    if heading.encode() not in raw:
        entry = (heading+'\n\n'
            '本机接口源码提交afe59dd仅修reward_readout的两句查询：当前完整回复已经交付，环境立即执行，'
            '不再要求补完同一回复。源码SHA94a7afbc、test ae16a359；64原native查询与先前clock候选编码逐项一致，'
            '原prefix/target/source/EOS范围不变，查询增加14token。29项原adapter单测通过，CPU约28.9秒、'
            'RSS约.969GiB、无模型/DT/GPU/反向/更新。独立review89134a3f。仅prepared，不部署、不称弱PG修复；'
            '既有paired预测测量未改善，不因修正错误条件而扩大质量结论。\n\n'
            '另复用原Adam三支LP/H和完整carrier，仅CPU按原16B4分母拆正/负/零A：DT相对当前PG置零对照，'
            '正A加权LP额外变化-4.34111e-05，负A+8.87006e-06，冻结DT PG额外+6.95970e-08。'
            '这是完整分母贡献，不是组内条件均值，不是参数/梯度分账。原task proxy与实际位移的一阶投影仍下降，'
            '有限eval变化、训练/eval路径和原生精度未由此隔离，不能叫Adam反号。CPU PID3612726/birth1791253077.8，'
            '3.654秒、RSS893MB，原core PG与旧结果逐项一致。raw6f1dae20、独立reviewf3295891。\n\n'
            '当前1/211已用终局漏奖修复v2，21成功终局均未越出载体；旧mask不能解释本批。'
            '小d、成功条件幅度和零回报支持分别记录；不乘211、不改熵、不把GRPO当oracle。'
            'PLAN、c9cd147 DT、原VERL、LoRA8/16及每卡B4不变；TextCraft保持停止。整体归因评价仍保留原论文累计删除/RISE/MAS。\n\n').encode()
        title, tail = raw.split(b'\n',1)
        runtime.write_bytes(title+b'\n\n'+entry+tail)
    print(json.dumps(dict(report=source(path), runtime=source(runtime), sources=len(report['sources']))))

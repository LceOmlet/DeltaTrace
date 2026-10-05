"""Record same-pair observations without changing the accepted method or runtime."""
import hashlib
import json
from pathlib import Path
import time

from analyze_textcraft_native_readout import source
from stage_textcraft_joint_fla import LOCAL

REPO = Path(__file__).resolve().parents[3]
READOUT = Path(__file__).parent / 'textcraft-degradation-20261005/readout-quality-20261006/v2'


if __name__ == '__main__':
    report_path = REPO / 'experiments/rl/results_textcraft_learning_degradation_20261006.json'
    report = json.loads(report_path.read_bytes())
    analysis_path = LOCAL / 'joint-fla-analysis.json'
    analysis = json.loads(analysis_path.read_bytes())
    job = json.loads((LOCAL / 'job.json').read_bytes())
    completed = json.loads((LOCAL / 'completed.json').read_bytes())
    assert analysis['coverage']['full_transport_samples_per_layer'] == 26
    assert analysis['coverage']['unique_full_response_UIDs_per_layer'] == 21
    assert completed['finite_trace_calls'] == 14
    assert completed['backward_calls'] == completed['optimizer_steps'] == completed['scheduler_steps'] == 0
    for rank in completed['ranks']:
        raw_path = LOCAL / f"rank{rank['rank']}-readout.json"
        assert source(raw_path)['sha256'] == rank['sha256']
    paths = list(LOCAL.glob('*.json')) + list(LOCAL.glob('*.stdout.txt'))
    paths.extend(LOCAL / name for name in (
        'effective-config.yaml', 'diagnostic.log', 'physical-before-start.txt', 'independent_joint_review.py'))
    paths.extend(Path(__file__).with_name(name) for name in (
        'observe_textcraft_joint_fla.py', 'verify_textcraft_joint_fla.py',
        'stage_textcraft_joint_fla.py', 'observe_textcraft_joint_fla_job.py',
        'analyze_textcraft_joint_fla.py', Path(__file__).name))
    readout_manifest_path = READOUT / 'readout-context-source-new-files.json'
    readout_manifest = json.loads(readout_manifest_path.read_bytes())
    readout_audit_path = READOUT / 'readout-context-label-source-audit.json'
    readout_audit = json.loads(readout_audit_path.read_bytes())
    for entry in readout_manifest['files']:
        path = READOUT / entry['name']
        assert source(path)['sha256'] == entry['sha256'], path
        paths.append(path)
    paths.append(readout_manifest_path)
    recorded = {item['path']: item for item in report['sources']}
    for path in paths:
        item = source(path)
        item['path'] = path.relative_to(REPO).as_posix()
        if item['path'] in recorded:
            if path.resolve() == Path(__file__).resolve():
                recorded[item['path']].update(item)
            else:
                assert item['sha256'] == recorded[item['path']]['sha256'], item['path']
        else:
            report['sources'].append(item)
            recorded[item['path']] = item
    report['same_pair_joint_FLA'] = dict(
        scope=analysis['scope'], analysis=source(analysis_path),
        independent_review=source(LOCAL / 'independent-joint-review.json'),
        coverage=analysis['coverage'], layer_summaries=analysis['layer_summaries'],
        dtype=analysis['observed_same_pair_tensor_dtypes'],
        counters=analysis['same_pair_head_observation_counts'],
        CPU_output_payload=analysis['CPU_peak_output_payload_statistics'],
        execution=dict(pid=job['pid'], pid_birth=job['pid_birth'], devices=job['devices'],
            started_unix=job['started_unix'], completed_unix=completed['completed_unix'],
            wall_seconds=completed['completed_unix'] - job['started_unix'],
            full_finite_calls=14, single_root_calls=28,
            backward_calls=0, optimizer_steps=0, scheduler_steps=0),
        limitations=analysis['limitations'],
        interpretation=(
            'On these two layers and 21 full-response UID pairs, joint input contractions '
            'are close to actual public FP16 output differences and have no opposite sign. '
            'This does not support a large same-pair arithmetic failure as the explanation '
            'for the much larger conditional full-to-single allocation discrepancies. '
            'The output FP16-to-BF16 boundary is measured separately with the same actual '
            'incoming adjoint. These are descriptive observations, not a new official '
            'tolerance test or proof of all-layer/token accuracy.'),
        production_state='Read-only observation; no DT/PPO formula, dtype, loss coefficient, averaging rule, LoRA setting or weights changed. TextCraft remains stopped.')
    report['readout_context_source_audit'] = dict(
        source=source(readout_audit_path), manifest=source(readout_manifest_path),
        scope=readout_audit['scope'], operations=readout_audit['operations'],
        first_response_actual_inputs=readout_audit['first_response_actual_inputs'],
        query_chat_boundary=readout_audit['query_chat_boundary'],
        official_carrier_vs_generation=readout_audit['official_carrier_vs_generation'],
        limitations=readout_audit['limits'],
        interpretation=(
            'The saved 64 actual first responses retain concrete Goals, label meanings and '
            'the original policy budget; saved 186 formal requests retain their initial Goal '
            'ID prefixes. Existing control-token and official carrier/generation-template '
            'differences are recorded, not promoted to a DT-specific bug or learning-cause '
            'claim. Sampling prompt IDs were not saved and were not reconstructed. '
            'No prompt candidate, new prediction or production change was made.'))
    report['status'] = 'same_pair_finite_observation_distinguished_from_conditional_credit_error_no_production_repair'
    report['updated_unix'] = time.time()
    report['interpretation']['next_owner_check'] = (
        'The measured large conditional allocation error and weak event-readout discrimination '
        'remain the learning-quality priorities. Same-pair closure is not a reason to cancel '
        'averaging, change official tolerance, amplify credit or restart TextCraft. Identify '
        'an existing owner-supported conditional estimation improvement before any candidate '
        'changes; retain the accepted token Q/V/A and upstream PPO objective.')
    for item in report['sources']:
        assert hashlib.sha256((REPO / item['path']).read_bytes()).hexdigest() == item['sha256'], item['path']
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    runtime = REPO / 'experiments/rl/RUNTIME_RECORD.md'
    heading = '## 2026-10-06 相同完整端点的 FLA 观察：区分算术与单删除估计'
    raw = runtime.read_bytes()
    if heading.encode('utf-8') not in raw:
        entry = (heading + '\n\n' +
            f"隔离PID{job['pid']}/birth{job['pid_birth']}已完成，GPU4/5，"
            f"{completed['completed_unix'] - job['started_unix']:.3f}秒。原7 B4/rank、checkpoint25、"
            'LoRA8/16与冻结runner/producer保持不变；14 full finite、28 single root，'
            'actor backward/optimizer/scheduler均0。读取既有原生FP16 FLA输出与其原BF16 '
            'norm边界，不增加模型/FLA调用；原finite返回对象和公式未改。\n\n' +
            '每层26运输样本按traj_uid/source_step合并为21 UID，四个head组按原输入逐项'
            '合并。L6/L8同端点F−Y_public平均绝对差0.0000606994/0.0000905424，'
            '均0/21反号；公共FP16输出到BF16 norm边界同do差0.000278545/0.000310393。'
            '208个选定incoming h配对差均0；28 public/28 norm/112原finite head返回齐全，'
            '组结束保留输出均0，观察器峰值CPU输出63,602,688字节。\n\n' +
            '原full系数投影single的42 probes仍是另一测量人口，不能混入21 UID分母。'
            '目前未支持将较大条件估计偏差归因于同端点算术错误；本观察未执行新的'
            '官方容差门槛，也不代表整网或单token信用准确。已绑定原导入SHA、配置、'
            'PID/birth、raw回执及独立审计到results_textcraft_learning_degradation_20261006.json。'
            '生产未修复，TextCraft保持停止，无信用倍率、纠偏或重启。\n\n').encode('utf-8')
        title, tail = raw.split(b'\n', 1)
        runtime.write_bytes(title + b'\n\n' + entry + tail)
    raw = runtime.read_bytes()
    heading = '## 2026-10-06 读出目标与官方 carrier：真实 ID 只读核查'
    if heading.encode('utf-8') not in raw:
        entry = (heading + '\n\n' +
            '64条真实首轮prompt/response/query由原tokenizer仅CPU解码；具体Goal、标签0/1'
            '语义、30步/512输出/temp1预算均保留，186个正式输入保留对应首轮Goal原ID前缀。'
            '无模型、前向、额外采样、backward、更新或目标改写。decode PID2547941，'
            'CUDA/distributed均false，maxRSS约1.04 GB；回执记录精确资源与SHA。\n\n' +
            '已记录连续两个im_end、部分先前think未闭和查询自身的原控制序列；未据此'
            '定义新的格式门槛或宣称退化原因。实际AgentGym schema与冻结原件字节相同，'
            '原training carrier及Qwen generation模板的system/reasoning/think行为差异由'
            '原源码负责。采样prompt IDs未保存，不能推定动态逐行差值。没有改官方'
            'schema、历史窗口或查询。源清单与审计绑定到既有degradation结果。\n\n').encode('utf-8')
        title, tail = raw.split(b'\n', 1)
        runtime.write_bytes(title + b'\n\n' + entry + tail)
    print(json.dumps(dict(report=source(report_path), runtime=source(runtime), sources=len(report['sources']), status=report['status'])))

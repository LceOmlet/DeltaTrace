"""Record completed native observations without changing training or its plan."""
import hashlib
import json
import statistics
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / 'experiments/rl'
REMOTE = '/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922'
AUDIT = REMOTE + '/receipts/dtpo-v9-negative-clip2-postgroup-variance-20261019-v4'


def read(name):
    return json.loads((RESULTS / name).read_text(encoding='utf-8'))


def artifact(name):
    path = RESULTS / name
    return dict(path=path.relative_to(ROOT).as_posix(),
                SHA256=hashlib.sha256(path.read_bytes()).hexdigest())


def physical_summary(report):
    groups = []
    for group in report['phases']:
        values = [v['gpu_utilization_percent'] for v in group['devices'].values()]
        count = sum(v['samples'] for v in values)
        groups.append(dict(identity=group['identity'], samples=count,
                           mean_utilization_percent=sum(v['samples'] * v['mean'] for v in values) / count))
    return dict(window=report['window'], groups=groups)


def main():
    stamp = datetime.now(timezone(timedelta(hours=8))).isoformat()
    pair = read('results_dtpo_v9_negative2_root_pageable_pair_v2_20261019_v4.json')
    util = read('results_dtpo_v9_negative2_root_pair_utilization_20261019_v4.json')
    progress = read('results_dtpo_v9_negative2_latest_efficiency_progress_20261019_v4.json')
    earlier = read('results_dtpo_v9_negative2_pageable_candidate_evidence_20261019_v4.json')
    assert pair['successful_same_input_pair_count'] == 8
    rows = pair['same_input_pairs']
    baseline = statistics.median(r['baseline_wall_seconds'] for r in rows)
    candidate = statistics.median(r['pair']['candidate_wall_seconds'] for r in rows)
    names = [
        'results_dtpo_v9_negative2_root_pageable_pair_20261019_v4.json',
        'results_dtpo_v9_negative2_root_pageable_pair_v2_20261019_v4.json',
        'results_dtpo_v9_negative2_root_pair_utilization_20261019_v4.json',
        'results_dtpo_v9_negative2_generation_native_work_20261019_v4.json',
        'results_dtpo_v9_negative2_current_generation_window_util_20261019_v4.json',
        'results_dtpo_v9_negative2_generation_dispatch_20261019_v4.json',
        'results_dtpo_v9_negative2_generation_trace_20261019_v4.json',
        'results_dtpo_v9_negative2_latest_efficiency_progress_20261019_v4.json',
    ]
    result = dict(
        recorded_at=stamp, status='bounded_observations_completed_candidate_not_deployed',
        actual_execution_source='cdd8ea404871d5cf2cf7eb8ff78ba0f1ed4625ea',
        method_source='f26b3b37cb3b88e64978e7a472ae4244670a39f3',
        upstream_VERL='20bd331bdbc9026a5668e11362178e10ab7400c8',
        run=progress['job']['run'], driver={k:progress['job'][k] for k in ('pid','birth')},
        latest_original_log_step=progress['latest']['step'], other_five='queued',
        artifacts=[artifact(name) for name in names],
        root_only_candidate=dict(
            original_flags=dict(pin_root_host=True,pin_replay_host=True),
            candidate_flags=dict(pin_root_host=False,pin_replay_host=True),
            original_runner_SHA256='6a112224b3d4e63d047a576d7df68ff1ee65e4bbacfcb0014f84c17bd82b8520',
            same_input_B4_pairs=8, paired_endpoint_lengths=[r['paired_shape'][1] for r in rows],
            baseline_median_seconds=baseline, candidate_median_seconds=candidate,
            attribution_seconds_reduction_fraction=1-candidate/baseline,
            reciprocal_runtime_factor=baseline/candidate,
            between_pair_original_host_release_seconds=[r['pair']['between_pair_owner_release_seconds'] for r in rows],
            pair_timing_scope='Attribute call only. Between-pair original B4 host release is separately recorded and excluded; one A-then-B pair is not a production whole-iteration speedup.',
            unchanged_DT_vector_ranks=[r['rank'] for r in rows if r['pair']['raw_DT_vector_equal']],
            restored_option_ranks=[r['rank'] for r in rows if r['pair']['original_options_restored']],
            unchanged_RNG_ranks=[r['rank'] for r in rows if r['pair']['rng_unchanged']],
            physical_utilization={side:physical_summary(r) for side,r in util['reports'].items()},
            added_validation_calls=8, added_production_model_calls=0,
            current_production_flags='original True/True restored on all eight ranks',
            acceptance='Promising same-input transport comparison only; candidate exact-32768 capacity and corresponding acceptance remain incomplete. Not deployed.',
            rejected_previous_candidate='Both root and replay pageable: long same-input time approximately doubled; rejected.',
            failed_adapter='Root-only v1 imported native_host_memory without verl.utils prefix before candidate invocation: zero candidate DT calls. v2 imports the actual owner during metadata installation and completed on all eight ranks.'),
        generation=dict(
            scope='Original current-run calls and bounded original torch.profiler; no extra generation or environment call.',
            eight_rank_first_eight_calls=dict(generated_tokens=497331,requests=1808,
                request_max_output_tokens=512,capped_requests=567,
                native_engine_API_coverage_fraction=0.81533,
                cross_rank_finish_lag_fraction=0.04247,
                pure_phase_GPU_Util_mean_percent=50.39732142857143,
                pure_phase_card_samples=224,boundary_or_missing_bracket_samples=280,
                conclusion='Do not label cross-rank waiting the dominant bottleneck from this window; outside API time is not all synchronization.'),
            native_dispatch=dict(ranks=8,observed_steps_per_rank=512,full_graph_decode_steps_per_rank=510,
                                 prefill_without_graph_steps_per_rank=2,restored_ranks=8,errors=0,
                                 async_scheduling=True,cudagraph_mode='FULL_AND_PIECEWISE'),
            bounded_rank0_trace=dict(existing_steps=64,additional_model_calls=0,
                trace= AUDIT+'/native-generation-64step-trace-rank0.json',
                trace_SHA256='5933596c000e0c4d1b1c3f38edf17e200c83ca9cd122ffc6bba79787534ec9f8',
                GPU_event_envelope_us=1371484.928,kernel_interval_union_us=1227283.357,
                kernel_coverage_fraction=1227283.357/1371484.928,
                lora_expand_plus_shrink_us=268664.729,
                lora_kernel_time_over_envelope_fraction=268664.729/1371484.928,
                interpretation='This instrumented single-rank decode interval is not whole-iteration mx-smi GPU-Util or peak-FLOPS utilization. CPU synchronization overlaps GPU work; do not count its full duration as GPU idle.'),
            current_optimization='No generation calculation/scheduler/kernel change deployed; graph and async paths were already active.'),
        official_assertion_source_audit=earlier['original_assertion_source_audit'],
        numerical_acceptance=dict(
            current_change_scope='Only proposed root cache allocation/copy/synchronization via original pin_root_host flag. FA/FLA/VERL arithmetic and training code remain unchanged.',
            same_vector_scope='Diagnostic evidence for sampled host transport; not a new official whole-DT tolerance.',
            FA='Pinned 2.6.3 corresponding attention output and dQ/dK/dV tests with same Q/K/V and FP32 reference plus low-precision baseline. Output maximum error <=2x baseline; dQ/dK/dV <=3x baseline in the cited functions.',
            FLA='Pinned 0.4.1 test_chunk normalized-RMS assertion: o/ht .005, dq/dk/dv/dh0 .008, db/dg .02; absolute <=1e-6 shortcut, no CI warning exemption. Corresponding GDN operator quantities only.',
            VERL='Pinned function-specific references/dtypes/masks. FusedLinearForPPO test uses FP32 reference inputs; its 1e-4 outputs and atol1e-2/rtol1e-4 input/weight gradients are not a BF16 whole-model/PPO threshold.',
            forbidden_substitutions=['checkpoint reload 0/0 as DT/PPO acceptance','multi-update exact logits as whole-method gate','FA/FLA dq thresholds relabeled as whole-model parameter-gradient thresholds'],
            new_official_numerical_suite_completed=False,
            candidate_32768_capacity_completed=False),
        validation_iterations_excluded_from_production_efficiency=[7,8,11],
        iteration10_note='Root v1 adapter failed before any candidate DT invocation; its readout observation is not a successful comparison.',
        training_method_changed=False, training_restarted=False,
        goal_status='active')
    path = RESULTS / 'results_dtpo_v9_negative2_scoped_utilization_findings_20261019_v4.json'
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    current = RESULTS / 'current_runtime.json'
    content = json.loads(current.read_text(encoding='utf-8'))
    field = dict(recorded_at=stamp, report=path.relative_to(ROOT).as_posix(),
                 report_SHA256=hashlib.sha256(path.read_bytes()).hexdigest(),
                 remote_receipt=AUDIT+'/scoped-utilization-findings.json',
                 completed_iterations=dict(TextCraft_DTPO=progress['latest']['step'],TextCraft_PPO=0,TextCraft_GRPO=0,
                                           AppWorld_DTPO=0,AppWorld_PPO=0,AppWorld_GRPO=0),
                 execution_source_commit=result['actual_execution_source'],
                 root_only_candidate_status='Not deployed. Eight same-input long B4 pairs completed, outputs/RNG equal, original flags restored; exact-32768 capacity and corresponding acceptance pending.',
                 root_pair_median_seconds=[baseline,candidate],
                 root_pair_GPU_Util_percent=[39.732421875,46.14402173913044],
                 generation_status='Existing async/full decode graphs confirmed; bounded native trace recorded; no kernel/scheduling change deployed.',
                 validation_iterations_excluded_from_production_efficiency=[7,8,11],goal_status='active')
    content['20261019_DTPO_v9_negative_clip2_six_GPU_v4_current']['latest_scoped_utilization_findings']=field
    current.write_text(json.dumps(content,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    prefix = f'''## {stamp} 生成/DT 原生利用率证据：root-only 传输候选有收益，尚未部署

正式执行仍cdd8ea40，方法f26b3b37，VERL20bd331b；TextCraft DTPO原日志完成{progress['latest']['step']}/200，其余五组queued。v9仍逐事件u=clip(-expm1(-d),-2,1)、乘g后仅方差缩放，原任务/32k/B4/参数/更新不改。本次只调查生成与DT；六组goal保持active。

原生runner的pin_root_host=False、pin_replay_host=True在同一轮第五个自然B4完成八rank配对：输入端点长度7262–13297，中位归因耗时{baseline:.3f}→{candidate:.3f}秒（减少{100*(1-candidate/baseline):.2f}%）；共同配对窗口原mx-smi利用率39.732%→46.144%，原相邻阶段观察严格一致才纳入，候选24个边界样本单列。两次调用之间原B4宿主回收14.9–26.2秒另列，不混入归因耗时；单次先A后B并非整轮提速证明。八rank原DT向量一致、RNG一致、原True/True配置和方法均恢复。此前root+replay同时pageable长输入约慢一倍已拒绝。root-only v1因资源模块包名前缀缺失在候选调用前失败；v2在安装时校验原verl.utils.native_host_memory后完成。原失败证据保留，不把安装/恢复完成称比较通过。

生成前8轮原调用观察含497331生成token/1808请求，567请求达到512上限；native engine API覆盖约81.53%，跨rank完成尾差仅约4.25%。相同窗口严格阶段样本GPU-Util约50.40%，缺失/边界单列。八rank各512个原调度步中510个decode已走FULL graph、2个prefill为NONE，async已启用。rank0原生64步torch.profiler中kernel区间并集覆盖GPU事件窗89.49%，LoRA expand/shrink合计约占该窗19.59%；单rank短时kernel覆盖率不等于整轮GPU-Util，CPU同步与GPU执行重叠不能全算空闲。观察器均恢复，未改生成计算/调度；原trace保存在共享盘，Git只保留摘要。

容差继续绑定对应计算与固定官方测试：FA2.6.3对应输出2倍低精度参考误差、dQ/dK/dV3倍；FLA0.4.1原test_chunk分别核对o/ht、dq/dk/dv/dh0、db/dg。VERL按实际函数的输入dtype/mask/状态和参考核验；FusedLinearForPPO官方FP32参考的阈值不外推成BF16整网/PPO阈值。禁止checkpoint或多步更新0/0替代。当前root-only同向量是传输诊断，不是新官方全DT容差证书；候选32768容量和对应验收未完成，原正式配置未替换。包含额外比较/trace的第7、8、11轮排除生产效率比较；第10轮root v1错误前没有执行候选DT。

完整源/输入/原日志行、各rank结果、官方断言审计及原生Util/trace见results_dtpo_v9_negative2_scoped_utilization_findings_20261019_v4.json及共享receipts/dtpo-v9-negative-clip2-postgroup-variance-20261019-v4/scoped-utilization-findings.json。32k容量、数值正确、GPU活动率及正式整轮吞吐分别记录，不将未完成项写为通过。

'''
    record = RESULTS / 'RUNTIME_RECORD.md'
    record.write_text(prefix+record.read_text(encoding='utf-8'),encoding='utf-8')
    (Path(__file__).parent/'scoped-utilization-runtime-prefix.txt').write_text(prefix,encoding='utf-8')
    print(json.dumps(dict(report=str(path),recorded_at=stamp,step=progress['latest']['step'],candidate_deployed=False)))


if __name__=='__main__':
    main()

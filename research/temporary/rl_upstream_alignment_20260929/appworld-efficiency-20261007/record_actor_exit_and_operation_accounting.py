"""Index saved observations only; no remote operation or training change."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
OBS = HERE / 'continuation-1791314906'


def receipt(path):
    path = Path(path)
    return dict(path=path.relative_to(REPO).as_posix(),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                bytes=path.stat().st_size)


def read(name):
    return json.loads((OBS / name).read_text(encoding='utf8'))


def main():
    result_path = REPO / 'experiments/rl/results_appworld_efficiency_20261007.json'
    result = json.loads(result_path.read_text(encoding='utf8'))
    runtime_path = REPO / 'experiments/rl/current_runtime.json'
    runtime = json.loads(runtime_path.read_text(encoding='utf8'))
    dated = OBS / 'current_runtime-after-exit-1791319881.json'
    if not dated.exists():
        dated.write_bytes(runtime_path.read_bytes())
    assert json.loads(dated.read_text(encoding='utf8'))['observed_utc'] == '2026-10-06T20:51:21.545319+00:00'
    operations = read('actor-original-GPU-operations-readonly.json')
    lifecycle_ok = read('mcTracer-lifecycle-1791319683.json')
    lifecycle_bad = read('mcTracer-lifecycle-1791319799.json')
    cleanup = read('failed-tracer-client-state-and-cleanup.json')
    completed = read('completed-DT-and-actor-entry-1791318569-readonly.json')
    result['status'] = 'formal_worker_exited_during_original_actor_update_no_completed_iteration'
    result['latest_observed_runtime'] = receipt(dated)
    result['latest_post_exit_resources'] = receipt(OBS/'post-exit-resource-identity-readonly.json')
    result['current_formal_exit_20261007'] = dict(
        driver=dict(pid=1181392, birth=1791314386.58, alive=False),
        workers=[dict(rank=0, pid=1189140, birth=1791314428.2, alive=False),
                 dict(rank=1, pid=1190575, birth=1791314441.82, alive=False)],
        frozen_source_sha256='3cd90b2db649cd477bc21398e7677dc8ea5534d5230296fc5f63c04037cd3427',
        rank0_disconnected_local_time='2026-10-07 04:44:14.041 +08:00',
        cause='Unknown: abrupt Ray worker EOF after official mcTracer live attach. Timing correlation is not proof of the exact exit cause.',
        profiler_actions=dict(
            owner='/opt/maca/bin/mcTracer',
            first_attach='04:43:23; failed output initialization because absolute odname was prefixed with cwd; profiler exit 0 is not success',
            second_attach='04:43:43; relative odname; Ctrl+T 04:44:03; profiler exit 04:44:04; target alive at stop observation',
            performed_by='This Codex task for bounded actual GPU operation accounting',
            policy='Do not attach mcTracer to a formal worker again; use an isolated bounded owner diagnostic if GPU tracing is required.'),
        cgroup=dict(failcnt=0, oom_kill=0),
        limitation='No explicit Python traceback, new dmesg OOM/GPU fault or exit signal was recovered. No complete iteration, actor wall timer, gradient metric or checkpoint was returned.',
        operations_after_exit=dict(formal_restarted=False, checkpoint_load=False,
                                  checkpoint_export=False, checkpoint_restore=False,
                                  training_configuration_changed=False),
        evidence={name:receipt(OBS/name) for name in (
            'actor-mcTracer-1791319403.json', 'actor-mcTracer-1791319423.json',
            'worker-exit-evidence-1791319599.json', 'tracer-exit-provenance-1791319622.json')})
    result['profiler_lifecycle_isolation'] = dict(
        scope='Two identical tiny original Torch BF16 CUDA clients on free GPU2; no model, RL, DT, optimizer or checkpoint. Not a tolerance or throughput benchmark.',
        successful_attach=dict(clients=lifecycle_ok['clients'],
                               meaning='Both completed normally; normal detach did not reproduce the formal exit.'),
        failed_initialization=dict(clients=lifecycle_bad['clients'],
            meaning='Both attempts failed to find the attach port file. The attached client remained sleeping in its CUDA matmul while control completed. This reproduces an attach failure/hang, not the exact formal worker exit or the earlier absolute-output error.'),
        cleanup=cleanup['cleanup'],
        evidence={name:receipt(OBS/name) for name in (
            'mcTracer-lifecycle-1791319683.json', 'mcTracer-lifecycle-1791319799.json',
            'failed-tracer-client-state-and-cleanup.json')})
    result['original_actor_gpu_observation'] = dict(
        scope=operations['scope'], source=operations['source'],
        gpu_window_seconds=operations['gpu_window_seconds'],
        gpu_active_union_seconds=operations['gpu_active']['union_seconds'],
        bf16_GEMM_union_seconds=operations['gemm']['union_seconds'],
        FA_union_seconds=operations['FA_kernels']['union_seconds'],
        collective_union_seconds=operations['collectives']['union_seconds'],
        copy_union_seconds=operations['copies']['union_seconds'],
        interpretation='BF16 GEMM is the largest identified GPU category in this bounded instrumented window. Copies do not dominate it. CPU nonzero/stream synchronization waits overlap preceding GPU work and must not be added as independent operation cost. This is not a full-update speed claim.',
        evidence=receipt(OBS/'actor-original-GPU-operations-readonly.json'))
    result['whitening_execution_status_20261007'] = dict(
        status='Approved whole-collected-batch official masked_whiten executed before original actor update',
        trainer_sha256=completed['trainer']['sha256'],
        proof='Original TaskRunner fit reached update_actor at line1285 after compute_advantage1256; the active DT branch373 uniquely calls original masked_whiten before DP/minibatch/B4. Raw d/Q/V/A and returns remain unchanged.',
        limits='No completed fresh actor gradient metric or learning-recovery conclusion because the worker exited.',
        evidence=receipt(OBS/'completed-DT-and-actor-entry-1791318569-readonly.json'))
    result['complete_DT_suffix_workload'] = dict(
        evidence=receipt(OBS/'complete-seven-group-slot-accounting-1791319176-readonly.json'),
        interpretation='All seven groups accounted. Single-endpoint suffix history-tail/alignment/padding is 39.58%/39.03% of token slots by rank; slots are not FLOPs or removable wall time. Different independent outcome queries cannot be collapsed into one attribution seed.')
    candidate_dir = HERE/'individual-prefix-owner-candidate-v1'
    result['individual_prefix_candidate'] = dict(
        status='unaccepted_prepared_only_not_deployed',
        observable_need='Avoid recomputing the interval between the earliest factual cut in B4 and each other row own factual cut.',
        owner='Existing finite FA CUDA owner and Python ABI; original three exported scalar function bodies/signatures preserved.',
        isolated_change='Optional per-row query starts and independent Q storage stride; no new finite formula, task, PPO, whitening, scheduler, value model or reference sampling.',
        measured_bound='Old saved 88-row metadata only: dense suffix slots 80720->55688 and 85652->62156. Not a current whole-DT speedup measurement.',
        checks='Five CPU interface/index checks and independent read-only source review only; no CUDA compilation, GPU numeric/official FA/FLA assertion, real B4 integration or performance test.',
        deployment='No runner/lease/target/canonical/production file was modified. Candidate is outside all default launch/patch paths.',
        evidence=dict(prepared=receipt(candidate_dir/'prepared-candidate.json'),
                      source_review=receipt(candidate_dir/'read-only-indexing-review-v1.json'),
                      workload_bound=receipt(OBS/'individual-prefix-cut-workload-bound-readonly.json'),
                      api_audit=receipt(OBS/'individual-prefix-owner-api-audit-readonly.json'),
                      minimal_design=receipt(OBS/'individual-prefix-minimal-owner-extension-design-readonly.json')))
    result_path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')

    ledger_path = REPO / 'experiments/rl/RUNTIME_RECORD.md'
    ledger = ledger_path.read_bytes().decode('utf8')
    title = '## 2026-10-07 04:44 原actor期间worker意外退出；正式任务已停止'
    section = '''
## 2026-10-07 04:44 原actor期间worker意外退出；正式任务已停止

driver1181392/birth1791314386.58及worker1189140/1190575均已退出，GPU4/5释放。
04:44:14.041 rank0断连，随后原Ray清理peer/TaskRunner；没有完成本轮、没有原
完整actor计时/梯度指标/检查点。04:51:21原只读recorder确认alive=false，来源
仍3cd90b2d；远端manifest旧declared status不作为存活事实。新 dated snapshot
continuation-1791314906/current_runtime-after-exit-1791319881.json保留所有源/哈希。

必须保留诊断操作与时序：本任务04:43:23使用已安装官方mcTracer附加rank0，
首次因绝对odname被拼在cwd下导致输出初始化失败；04:43:43改相对目录重试，
04:44:03按官方Ctrl+T停止，04:44:04退出0且当时target尚存活，约10秒后rank0
断连。没有新Python traceback、dmesg OOM/GPU故障或明确退出信号；cgroup
failcnt0/oom_kill0。不能断言profiler导致正式退出，也不能隐去这项相邻操作。

仅对空闲GPU2的两个同构128x128 BF16 Torch客户端做有界隔离：一次正常附加
后两者正常结束；另一轮附加初始化因port-file缺失失败，被附加客户端睡眠卡在
CUDA matmul，对照正常结束。这证明附加失败存在挂起风险，未复现正式退出或
最初odname错误。精确PID1746517/birth1791319799.2/命令SHA核对后已清理本任务
该诊断子进程，未向训练或其他用户进程发信号。不再对正式worker live attach。
原始worker/raylet/dmesg、两次附加及隔离/cleanup回执全部由结果JSON索引。

本轮DT已完整结束：每卡403次B4，readout2499.85/2498.78秒，bank382.25/
381.55秒，B4 sum2114.36/2113.99秒，双卡不相加。整批原masked_whiten已在
原compute_advantage373实际经过，然后进入原actor1285；原d/Q/V/A/returns不改。
这证明批准的尺度处理生效，不证明学习退化恢复。04:39:40 actor局部变量两卡
epoch0/batch_idx5，至少已完成5/14 optimizer step；不能把未完成更新称为完整
计时或把旧2495秒充作本轮结果。原actor仍每卡56 B4/14 step的既定两epoch负载。

实际GPU trace为20.464秒有界、instrumented窗口：GEMM union9.809秒，FA1.453秒，
collective0.387秒，copy union1.049秒；总GPU active20.383秒。CPU nonzero/stream
synchronize等待可能是在等前序GPU计算，不能另算一笔耗时。该证据不支持把
offload搬运或mask nonzero直接当主要浪费改掉，也不是完整actor吞吐验收。
原trace65926059字节，SHA688b055c09f2dbc585faba00eaa30153f2532bb4bdd699b53a19a61616ec28dc。

七组完整suffix token槽已核算：历史尾部/对齐/padding占39.58%/39.03%；这不是
等额FLOP/墙钟承诺。当前B4统一最早切点确会多算其他行的事实历史段；原HF
位置/mask/varlen接口可组合，但现有lease/runner/finite-FA ABI还不能完整表达
逐行切点。隔离owner候选只加per-row起点与独立Q stride，三个旧scalar ABI
函数体/签名不变，CPU5项和独立只读索引审查通过；没有CUDA编译/GPU数值或
官方容差/完整B4接入验收，不部署。旧88行metadata的suffix槽减少31.01%/27.43%
仅是表示上界，不能报本轮整体提速。存储行优化仍verified/prepared未部署。

没有加载、导出或恢复任何检查点，没有重启正式任务，没有改rank8/alpha16/
每卡B4/原PPO与任务配置/FA/FLA容差。当前受影响的正式更新停止；不把诊断或
候选当修复完成。出处results_appworld_efficiency_20261007.json最新退出与操作索引。

'''
    if title not in ledger:
        header_end = ledger.index('\n') + 1
        assert ledger[:header_end].rstrip('\r\n') == '# 当前运行版本与修复记录'
        ledger_path.write_bytes((ledger[:header_end]+section+ledger[header_end:]).encode('utf8'))
    print(json.dumps(dict(result=receipt(result_path), ledger=receipt(ledger_path),
                          runtime=receipt(dated)),ensure_ascii=False))


if __name__ == '__main__':
    main()

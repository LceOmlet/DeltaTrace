"""Record actual native storage selection and its bounded verification scope."""
import hashlib
import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / 'rl_dtpo_v9_clip_u_20261019_v1'))
from transport import run_python

CODE = r'''
from collections import Counter
import hashlib,json,time,psutil,statistics
from pathlib import Path
R=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
A=R/'receipts/dtpo-v9-negative-clip2-postgroup-variance-20261019-v4'
run=R/'runs/textcraft-dtpo-v9-negative-clip2-8gpu-200-20261019-v4'
raw=run/'runtime-overrides/actual-worker-imports-small-GPU-resident-resource-accepted-v1.json'
binding=json.loads(raw.read_bytes());rows=binding['results']
assert sorted(r['rank'] for r in rows)==list(range(8))
for row in rows:
    v=row['small_GPU_resident_resource_override']
    assert v['status']=='applied_original_storage_option_bounded_small_inputs'
    assert v['resource_extent_limit']==2048 and v['restore_after_each_call']
    assert v['actor_scheduler_epoch']==21
    assert not v['arithmetic_changed'] and v['collectives_for_dispatch']==0
    assert v['default_flags']==dict(pin_root_host=False,pin_replay_host=True,offload_replay_mixer=True)
    assert psutil.Process(row['pid']).create_time()==row['birth']
pair_path=A/'existing-DT-gpu-resident-pair-results.json'
pair=json.loads(pair_path.read_bytes())
assert pair['status']=='completed' and pair['restored_rank_scopes']==8 and not pair['errors']
comparisons=[]
for row in pair['same_input_pairs']:
    p=row['pair'];assert p['raw_DT_vector_equal'] and p['rng_unchanged'] and p['original_options_restored']
    b=next(r['value'] for r in pair['native_timing_records']
           if r['value']['rank']==row['rank'] and r['value']['minibatch']==row['minibatch'])
    c=p['candidate_owner_info'];bc=Counter(r['kind'] for r in b['calls']);cc=Counter(r['kind'] for r in c['calls'])
    assert not (cc-bc) and all(k.startswith('restore_mixer_captures_') for k in bc-cc)
    comparisons.append(dict(rank=row['rank'],shape=row['paired_shape'],
        baseline_seconds=row['baseline_wall_seconds'],candidate_seconds=p['candidate_wall_seconds'],
        baseline_peak_GiB=b['peak_allocated']/2**30,candidate_peak_GiB=c['peak_allocated']/2**30,
        raw_DT_vector_equal=True,max_abs_difference=p['raw_DT_max_abs_difference'],RNG_unchanged=True,
        additional_native_compute_stage_calls=dict(cc-bc),removed_capture_restore_stages=dict(bc-cc)))
assert len(comparisons)==8
source=json.loads((run/'source.json').read_bytes())
snapshot=R/'receipts/textcraft-eight-gpu-v9-negative-clip2-20261019-v4/dt-latest-source-snapshot.json'
job=json.loads(snapshot.read_bytes())['jobs'][0]
handles=[]
for pid,birth in [(2503202,1792394518.8),(2503287,1792394519.29)]:
    assert abs(psutil.Process(pid).create_time()-birth)<.02
    handles.append(dict(pid=pid,birth=birth,alive=True))
target=run/'runtime-overrides/small-GPU-resident-resource-override-completed.json'
record=dict(status='completed_runtime_bounded_original_storage_selection',recorded=time.time(),
    actual_training_execution_source=source['local_patch_commit'],adapter_commit='bfdd79e8',
    method_commit='f26b3b37cb3b88e64978e7a472ae4244670a39f3',upstream_VERL='20bd331bdbc9026a5668e11362178e10ab7400c8',
    run=str(run),handles=handles,raw_binding=dict(path=str(raw),SHA256=hashlib.sha256(raw.read_bytes()).hexdigest()),
    actual_workers=[dict(rank=r['rank'],pid=r['pid'],birth=r['birth'],override=r['small_GPU_resident_resource_override'],modules=r['modules']) for r in rows],
    pair=dict(path=str(pair_path),SHA256=hashlib.sha256(pair_path.read_bytes()).hexdigest(),comparisons=comparisons),
    paired_median_seconds=[statistics.median(r['baseline_seconds'] for r in comparisons),statistics.median(r['candidate_seconds'] for r in comparisons)],
    capacity_scope='Eight natural B4 pairs of paired extent2217..2401 peaked below52.221GiB. Resident production route is conservatively limited to extent<=2048 and <=8 paired rows; original long/32768 route unchanged.',
    production_dispatch=dict(extra_model_calls=0,extra_DT_calls=0,extra_CUDA_calls=0,extra_collectives=0,extra_optimizer_updates=0,return_object_preserved=True),
    validation_calls=dict(DT=8,optimizer=0,rollout=0,iteration_excluded_from_uninstrumented_efficiency=21),
    arithmetic_owners_changed=False,numerical_scope='Existing original FA/FLA/VERL arithmetic and corresponding accepted operator scopes unchanged. Full sampled DT vectors equal; this resource test is not a new official whole-model tolerance certificate.',
    first_entire_production_iteration=22,production_efficiency_verified=False,
    candidate_GPU_Util_claimed=False,candidate_GPU_Util_scope='Exact absolute candidate boundaries were not recorded by this comparison; do not reconstruct or claim a paired native Util measurement. Measure the next natural production phase with the existing strict reducer.',
    latest_metric=job['metrics'][-1],phase=job.get('phase_label'),source_snapshot=str(snapshot),
    checkpoint_or_training_restarted=False,queued_AppWorld_resident_policy_applied=False,
    persistence='Loaded TextCraft runners only; original source/environment preserved; use this completion receipt with raw source bindings.',
    goal_status='active')
if target.exists():
    saved=json.loads(target.read_bytes());assert saved['raw_binding']==record['raw_binding'];record=saved
else:target.write_text(json.dumps(record,indent=2)+'\n')
record['remote_receipt']=str(target)
print(json.dumps(record))
'''


def main():
    record = json.loads(run_python(CODE, timeout=45))
    result = REPO / 'experiments/rl/results_dtpo_v9_negative2_small_GPU_resource_override_20261019_v4.json'
    result.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    stamp = datetime.fromtimestamp(record['recorded'], timezone(timedelta(hours=8))).isoformat()
    path = REPO / 'experiments/rl/current_runtime.json'
    snapshot = json.loads(path.read_text(encoding='utf-8-sig'))
    current = snapshot['20261019_DTPO_v9_negative_clip2_six_GPU_v4_current']
    current.update(recorded_at=stamp, status='native_21_iterations_complete_small_GPU_storage_applied',
                   completed_iterations=record['latest_metric']['step'], phase=record['phase'])
    trace = current['native_replay_trace_observation']
    trace.update(status='completed_native_replay_trace_reduction', results_available=True,
                 result='experiments/rl/results_dtpo_v9_negative2_native_replay_trace_20261019_v4.json',
                 trace_SHA256='a485947f87b4b7aea2715b4b41ed3febff577f1fe71e560d16e3e546de827c1a',
                 validation_iteration_excluded_from_uninstrumented_efficiency=19)
    current['small_GPU_resident_resource_runtime_override'] = dict(
        status=record['status'], recorded_at=stamp, report=result.relative_to(REPO).as_posix(),
        remote_receipt=record['remote_receipt'], adapter_commit=record['adapter_commit'],
        resource_extent_limit=2048, first_entire_production_iteration=22,
        paired_median_seconds=record['paired_median_seconds'], production_efficiency_verified=False,
        candidate_GPU_Util_claimed=False, queued_AppWorld_resident_policy_applied=False)
    current['latest_native_progress_since_plot'] = dict(
        collected_at_beijing=stamp, completed_iterations=record['latest_metric']['step'],
        phase=record['phase'], latest_native_metric=record['latest_metric'], checkpoint_step=20,
        checkpoint_proof='experiments/rl/results_dtpo_v9_negative2_checkpoint20_and_production_20261019_v4.json')
    plots = Path('F:/DeltaTraceBackups/plots')
    directories = {}
    for label, task in [('TextCraft_plot_directory','textcraft-eight-gpu-v9-negative-clip2-20261019-v4'),
                        ('AppWorld_plot_directory','appworld-eight-gpu-v9-negative-clip2-20261019-v4'),
                        ('historical_control_plot_directory','textcraft-v9-negative-clip2-eight-gpu-historical-PPO-GRPO-global-norm-20261019-v4')]:
        directories[label] = max((d for d in (plots/task).iterdir() if (d/'plot-receipt.json').exists()),key=lambda d:d.name).as_posix()
    progress = current['latest_progress']
    progress.update(collected_at_beijing=stamp, completed_iterations=dict(TextCraft_DTPO=21,TextCraft_PPO=0,TextCraft_GRPO=0,AppWorld_DTPO=0,AppWorld_PPO=0,AppWorld_GRPO=0),
        current_phase=record['phase'],native_checkpoint_step=20,latest_native_metric=record['latest_metric'],
        prior_plot_snapshot='20261019-205648',new_iterations_since_prior_plot=dict(TextCraft_DTPO=1,TextCraft_PPO=0,TextCraft_GRPO=0,AppWorld_DTPO=0,AppWorld_PPO=0,AppWorld_GRPO=0),
        efficiency_claim='Iteration21 includes eight optional validation DT calls and is excluded from production efficiency comparisons; method and training return values unchanged.',**directories)
    source = Path(directories['TextCraft_plot_directory'])/'source-snapshot.json'
    progress.update(source_snapshot=str(source),source_snapshot_SHA256=hashlib.sha256(source.read_bytes()).hexdigest())
    path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2)+'\n',encoding='utf-8')
    prefix = f'''## {stamp} 第21轮完成；原GPU replay资源开关的小批次配对通过并已实际应用

原controller2503202/1792394518.8及driver2503287/1792394519.29仍匹配存活，TextCraft DTPO完成21/200，进入22轮生成，后五组queued。共享checkpoint20的data.pt及八rank model/optim/extra_state共25非空文件已经原owner标准核对。执行cdd8ea40、方法f26b3b37、VERL20bd331与clip[-2,1]/乘g后原masked_var保持；未重启。最新三套图在21:11刷新至21轮，历史PPO105/GRPO191只是独立对照，不算新八卡进度。

第19轮的单rank原生两层trace已完成、原方法恢复，0额外model/DT调用；15.760秒包络中记录device事件并集6.740秒，CPU empty_strided区间内无记录device事件8.689秒。首次allGather3.041秒、下一次1.446ms，含启动/到达不齐；不由此宣称每层通信瓶颈或把CPU操作名直接等同分配根因。原始trace SHA a485947f、9464285字节及CPU归约保留；第19轮排除无观测效率比较，未部署prefetch。

第21轮第三自然B4的原offload_replay_mixer True/False配对完成八rank：paired extent2217–2401，调用中位16.905→10.092秒，完整原DT向量逐值相同、RNG不变、flags及八rank观察作用域全部恢复。candidate峰值52.127–52.221GiB；原计算stage无新增，仅删除8个FA capture恢复stage。此轮额外验证DT共8次、0 optimizer/rollout，整轮和高Util不得当生产收益。该比较未保存候选精确绝对起止时间，不能补造配对GPU Util。

资源adapter bfdd79e8已推送，原metadata RPC在所有八worker scheduler21时实际安装完成：仅paired rows<=8且extent<=2048时调用原GPU resident分支，其余/32768仍原卸载分支；每次finally恢复原flag。dispatch无CUDA/collective/模型/DT增量，不改FA/FLA/VERL计算、Q/V/A、mask或参数更新。仅当前TextCraft loaded runner，AppWorld尚未应用该新策略；原root=False/replay=True的AppWorld准备仍保持。第22轮起完整生产效率待读原GPU Util，不以配对40.3%的调用缩短宣称整轮收益。

容差继续按固定FA/FLA/VERL对应计算、dtype、mask和原参考；本次资源输出一致不冒称新增整网官方数值验收，不使用checkpoint或多步更新0/0替代。证据results_dtpo_v9_negative2_native_replay_trace_20261019_v4.json、results_dtpo_v9_negative2_gpu_resident_pair_20261019_v4.json、results_dtpo_v9_negative2_small_GPU_resource_override_20261019_v4.json。六组goal保持active。

'''
    (HERE/'small-GPU-resource-runtime-prefix.txt').write_text(prefix.rstrip()+'\n',encoding='utf-8')
    runtime = REPO/'experiments/rl/RUNTIME_RECORD.md'
    text=runtime.read_text(encoding='utf-8')
    if '原GPU replay资源开关的小批次配对通过并已实际应用' not in text[:1000]:
        runtime.write_text(prefix+text,encoding='utf-8')
    print(json.dumps(dict(status=record['status'],local_result=str(result),latest=record['latest_metric']['step'],first_production=22,production_efficiency_verified=False),ensure_ascii=False))


if __name__ == '__main__':
    main()

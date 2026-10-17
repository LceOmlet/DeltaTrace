"""Record completed acceptance and current native formal identities, without training calls."""
from datetime import datetime
import json
import os
from pathlib import Path
import uuid
from zoneinfo import ZoneInfo

now = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
root = '/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922'
here = Path(__file__).resolve().parent
snapshot = json.loads((here / 'receipts/latest-native-snapshot.json').read_bytes())
manifest = json.loads((here / 'receipts/formal-training.json').read_bytes())
acceptance_dir = Path('research/temporary/rl_PPO_native_final_numerical_20261017_v1/receipts')
acceptance = json.loads((acceptance_dir / 'formal-acceptance.json').read_bytes())
raw_numeric = json.loads((acceptance_dir / 'status.json').read_bytes())
boundary = json.loads(Path('research/temporary/rl_PPO_optimizer_boundary_20261017_v2/receipts/status.json').read_bytes())
assert acceptance['status'] == 'accepted_for_formal_AppWorld_PPO'


def read(path):
    return json.loads(Path(path).read_bytes())


def exit_for(version):
    for path in sorted((here / 'receipts').glob('AppWorld*-exit.json')):
        record = read(path)
        if record.get('job', {}).get('version') == version:
            return path, record
    return None, None


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary, path)


result_path = Path('experiments/rl/results_appworld_formal_training_20261017_v1.json')
result = read(result_path)
for method, key in [('dt', 'DTPO'), ('ppo', 'PPO')]:
    job, = [j for j in manifest['jobs'] if j.get('method') == method]
    observed, = [j for j in snapshot['jobs'] if j['method'] == method]
    source_dir = Path('research/temporary/' + ('rl_appworld_dtpo_formal_20261017_v' if method == 'dt'
                                           else 'rl_appworld_PPO_formal_20261017_v') + job['version'].rsplit('-v',1)[1] + '/receipts')
    source = read(source_dir / 'source.json')
    current = dict(version=job['version'], pid=job['pid'], birth=job['birth'],
                   physical_gpus=job['devices'], run=str(Path(job['output']).parent).replace('\\', '/'),
                   output=job['output'], local_patch_commit=source['local_patch_commit'],
                   method_source_commit=source['method_source_commit'], upstream_commits=source['upstream_commits'],
                   source_bindings=source['source_bindings'], source_receipt=job['source_receipt'],
                   checkpoints=job['checkpoints'], budget=job['budget'],
                   status='native_formal_job_exited' if not observed['alive'] else
                          'native_initialization_in_progress_no_completed_iteration' if observed['completed_iterations']==0
                          and (not observed['workers'] or any('init_model' in w['phase'] or 'load_checkpoint' in w['phase'] for w in observed['workers'])) else
                          'native_first_iteration_in_progress_no_completed_iteration' if observed['completed_iterations'] == 0
                          else 'native_formal_training_in_progress',
                   alive=observed['alive'], workers=observed['workers'],
                   current_worker_phases=[w['phase'] for w in observed['workers']],
                   completed_iterations=observed['completed_iterations'],
                   latest_checkpointed_iteration=observed['latest_checkpointed_iteration'],
                   collected=snapshot['collected'], snapshot=str(here / 'receipts/latest-native-snapshot.json'),
                   environment_owner_patch=source['environment_owner_patch'],
                   training_methods_or_numeric_hyperparameters_changed=False)
    bindings = source_dir / 'actual-worker-imports-formal-v1.json'
    current['actual_worker_bindings'] = read(bindings) if bindings.exists() else dict(status='read_only_rpc_pending')
    after_DT = source_dir / 'actual-worker-imports-after-first-DT-v1.json'
    if after_DT.exists():current['actual_worker_bindings_after_DT']=read(after_DT)
    after_update = source_dir / 'actual-worker-imports-after-first-update-v1.json'
    if after_update.exists():current['actual_worker_bindings_after_update']=read(after_update)
    current['actual_environment_owner_bindings'] = [read(p) for p in sorted(source_dir.glob('loop-owner-*.json'))]
    if source.get('sampler_checkpoint_artifact'):
        current['sampler_checkpoint'] = dict(
            artifact=source['sampler_checkpoint_artifact'],
            extension_version=source['sampler_checkpoint_extension']['version'],
            previous_execution_without_sampler_state=source.get('previous_execution_gap_run'),
            gap_receipt=source.get('checkpoint_gap_receipt'),
            configured_for_native_save_before_completion_marker=True,
            actual_completed_checkpoint_verification=None)
        step = observed['latest_checkpointed_iteration']
        completed_receipt = here / 'receipts' / f'{job["version"]}-checkpoint-{step}-verified.json'
        if step is not None and completed_receipt.exists():
            verified = read(completed_receipt)
            assert verified['version'] == job['version'] and verified['step'] == step
            current['sampler_checkpoint']['actual_completed_checkpoint_verification'] = str(completed_receipt)
            current['sampler_checkpoint']['verification_status'] = verified['status']
        else:
            current['sampler_checkpoint']['verification_status'] = 'pending_actual_native_completed_checkpoint_verification'
    if not observed['alive']:
        exit_path, exit_record = exit_for(current['version'])
        if exit_record is not None:
            current.update(status=exit_record['status'], exit_receipt=str(exit_path))
    if method == 'dt':
        current['previous_failed_run'] = source.get('previous_failed_current_run', source.get('previous_failed_run'))
    else:
        current['acceptance'] = dict(receipt=root + '/receipts/PPO-native-final-numerical-20261017-v1/formal-acceptance.json',
                                    local_receipt=str(acceptance_dir / 'formal-acceptance.json'), status=acceptance['status'],
                                    warm_update_output_max_abs=0, compared_values=65528,
                                    exact_32k_B4_colocated_capacity_passed=True,
                                    universal_first_use_or_future_trajectory_bitwise_identity_claimed=False)
    if result.get(key,{}).get('version') not in (None,current['version']):
        previous=result[key]
        exit_path, exit_record = exit_for(previous['version'])
        if exit_record is not None:
            previous.update(status=exit_record['status'],alive=False,completed_iterations=exit_record['completed_iterations'],
                            latest_checkpointed_iteration=None,exit_receipt=str(exit_path))
        result.setdefault(key+'_history',[]).append(previous)
    result[key] = current
recovery_path=here/'receipts/TextCraft-PPO-native-global-OOM-recovery-v1-job.json'
if recovery_path.exists():
    recovery=read(recovery_path)
    canonical_copy=here/'receipts/TextCraft-PPO-current-native-job.json'
    if canonical_copy.exists():
        recovery=read(canonical_copy)
    execution_suffix=recovery.get('execution_version','').rsplit('-v',1)[-1]
    restored_path=here/f'receipts/native-PPO-restore-global-OOM-20261017-v{execution_suffix}.json'
    restored=read(restored_path) if restored_path.exists() else None
    recovery_status=recovery['status']
    result['protected_TextCraft_jobs'][1]=dict(pid=recovery['pid'],birth=recovery['birth'],physical_gpus=[4,5],
        status=recovery_status,execution_version=recovery['execution_version'],
        execution_patch_commit=recovery['execution_patch_commit'],resumed_completed_iteration=100,
        prior_kernel_OOM=dict(pid=3946004,birth=1792041120.19,completed_log_iteration=103,complete_checkpoint_iteration=100),
        unsaved_iterations_to_rerun=[101,102,103],receipt=str(recovery_path),
        current_canonical_source=str(canonical_copy) if canonical_copy.exists() else None,
        actual_native_restore_receipt=str(restored_path) if restored else None,
        active=recovery.get('active',True),kernel_exit_receipt=recovery.get('kernel_exit_receipt'))
    result['inactive_diagnostic_checkpoint_storage_relocation']=read(here/'receipts/inactive-diagnostic-checkpoint-storage-relocation.json')
    result['AppWorld_PPO_restart_boundary_owner_candidate']=read('research/temporary/rl_appworld_environment_restart_boundary_20261017_v3/receipts/owner-patch.json')
    result['AppWorld_PPO_restart_boundary_CPU_verification']=read('research/temporary/rl_appworld_environment_restart_boundary_20261017_v3/receipts/CPU-recovery-verification.json')
result.update(updated_at=now, status='native_AppWorld_job_exit_recovery_in_progress' if not all(result[k]['alive'] for k in ('DTPO','PPO'))
                                    else 'DTPO_and_native_PPO_formal_training_in_progress', goal_status='active',
              goal_completion='Both original 200-iteration budgets and final native shared complete checkpoints',
              host_MemAvailable_GiB=snapshot['host_MemAvailable_GiB'])
write(result_path, result)

ppo_path = Path('experiments/rl/results_PPO_32k_efficiency_20261016_v4.json')
ppo = read(ppo_path)
ppo.update(updated_at=now, status='accepted_scoped_owner_warm_output_capacity_and_efficiency_deployed_AppWorld_PPO',
           final_native_acceptance=acceptance,
           final_native_acceptance_local_receipt=str(acceptance_dir / 'formal-acceptance.json'),
           final_native_first_use_comparisons_preserved=raw_numeric['comparisons'],
           formal_AppWorld_PPO=dict(version=result['PPO']['version'],pid=result['PPO']['pid'],birth=result['PPO']['birth'],
                                   source_commit=result['PPO']['local_patch_commit'],
                                   checkpoints=result['PPO']['checkpoints'], budget=result['PPO']['budget']))
ppo['native_optimizer_boundary_diagnostic'].update(status=boundary['status'], completed=boundary['finished'],
                                                   comparisons=boundary['comparisons'],
                                                   numerical_guarantee=False,
                                                   acceptance_limit='Instrumented update evidence only; final accepted output comparison has no update hooks')
ppo['owner_test_scope_clarification'].update(
    current_numerical_guarantee=False,
    current_scoped_warm_owner_output_comparison_passed=True,
    applicability='Measured original critic post-update values match exactly under pinned owner warm-update sequence. '
                  'Not a Qwen3.5/Maca universal first-use or future trajectory guarantee.',
    final_receipt=str(acceptance_dir / 'warm-native-output-comparison.json'))
write(ppo_path, ppo)

runtime_path = Path('experiments/rl/current_runtime.json')
runtime = read(runtime_path)
runtime['latest_appworld_formal_training_20261017_v1'] = dict(
    updated_at=now, status=result['status'], goal_status='active', result=str(result_path),
    DTPO={k:result['DTPO'][k] for k in ('version','pid','birth','physical_gpus','completed_iterations','checkpoints','budget','status')},
    PPO={k:result['PPO'][k] for k in ('version','pid','birth','physical_gpus','completed_iterations','checkpoints','budget','status')})
runtime['latest_PPO_32k_efficiency_20261016_v4'] = ppo
runtime['latest_PPO_optimizer_boundary_20261017_v2'].update(status=boundary['status'], completed=boundary['finished'],
    receipt=str(Path('research/temporary/rl_PPO_optimizer_boundary_20261017_v2/receipts/status.json')),
    numerical_acceptance_scope='Instrumented only; superseded by no-update-hook warm output comparison')
runtime['latest_PPO_native_final_numerical_20261017_v1'] = dict(
    updated_at=now, status=acceptance['status'], result=str(acceptance_dir / 'formal-acceptance.json'),
    source_commit=raw_numeric['source_commit'], production_FA_deterministic=False,
    warm_output_max_abs=0, total_values=65528, context=8192, response_length=8191, microbatch_per_rank=4,
    first_use_native_comparisons_retained=raw_numeric['comparisons'], universal_bitwise_guarantee=False)
deployment_path = Path('research/temporary/rl_appworld_native_sampler_checkpoint_20261017_v5/receipts/formal-deployment.json')
if deployment_path.exists():
    deployment = read(deployment_path)
    result['actual_sampler_checkpoint_extension_deployment'] = dict(
        receipt=str(deployment_path), status=deployment['status'],
        source_commit=deployment['source_commit'],
        actual_native_checkpoint_verification={k:result[k].get('sampler_checkpoint') for k in ('DTPO','PPO')})
    runtime['latest_AppWorld_native_sampler_checkpoint_20261017_v5'] = dict(
        updated_at=now, **result['actual_sampler_checkpoint_extension_deployment'])
    write(result_path, result)
if recovery_path.exists():
    runtime[f'latest_TextCraft_PPO_global_OOM_recovery_20261017_v{execution_suffix}']=dict(updated_at=now,
        status=recovery_status,pid=recovery['pid'],birth=recovery['birth'],physical_gpus=[4,5],
        checkpoint=recovery['resume_from_path'],execution_patch_commit=recovery['execution_patch_commit'],
        native_resume_receipt=str(recovery_path),prior_completed_log_iteration=103,native_checkpoint_iteration=100,
        unsaved_iterations_to_rerun=[101,102,103],storage_relocation_receipt=str(here/'receipts/inactive-diagnostic-checkpoint-storage-relocation.json'),
        active=recovery.get('active',True),kernel_exit_receipt=recovery.get('kernel_exit_receipt'))
write(runtime_path, runtime)

record_path = Path('experiments/rl/RUNTIME_RECORD.md')
record = record_path.read_text(encoding='utf-8')
dt, ppo = result['DTPO'], result['PPO']
event = f"""## {now} AppWorld原生日志状态；两项200轮及最终共享完整检查点目标仍active

DTPO执行版本{dt['version']}，driver{dt['pid']}/{dt['birth']}，完成{dt['completed_iterations']}/200，状态{dt['status']}，最新原生完整检查点{dt['latest_checkpointed_iteration']}；实际退出回执{dt.get('exit_receipt')}。信用方法仍为已接受v5，不拼接旧执行日志。

PPO执行版本{ppo['version']}，driver{ppo['pid']}/{ppo['birth']}，完成{ppo['completed_iterations']}/200，状态{ppo['status']}，原worker phase={ppo['current_worker_phases']}，最新原生完整检查点{ppo['latest_checkpointed_iteration']}。既有对应VERL暖态数值/32k B4共置容量和效率验收保持，不声称未来bitwise轨迹一致。

原作者40组×6请求/原取消/200轮/B4/32768/global32/PPO epochs2/eval5/save2/keep2保持；每2迭代原生完整checkpoint直写用户共享独立run目录。当前状态来自{snapshot['collected']}原read_job_logs/parse_logs快照与实际PID出生。TextCraft PPO实际状态={recovery_status if recovery_path.exists() else '未回读'}，最新原生完整100仍保留；不将历史恢复验收当当前进程存活。goal未完成，不恢复SQL、GRPO或旧梯度研究。

"""
record_path.write_text(event + record, encoding='utf-8')
print(json.dumps(dict(updated_at=now, DTPO=[dt['pid'],dt['completed_iterations'],dt['status']],
                      PPO=[ppo['pid'],ppo['completed_iterations'],ppo['status']], goal_status='active')))

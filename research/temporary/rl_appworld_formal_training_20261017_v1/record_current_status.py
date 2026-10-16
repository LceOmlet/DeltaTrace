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
    current['actual_environment_owner_bindings'] = [read(p) for p in sorted(source_dir.glob('loop-owner-*.json'))]
    if method == 'dt':
        current['previous_failed_run'] = source.get('previous_failed_run')
        current['log_warning_classification'] = dict(count=2,
            type='Inductor compiled graph cache FileNotFoundError during concurrent graph capture',
            subsequent_graph_capture_completed=True, fatal=False, cache_changes=0)
    else:
        current['acceptance'] = dict(receipt=root + '/receipts/PPO-native-final-numerical-20261017-v1/formal-acceptance.json',
                                    local_receipt=str(acceptance_dir / 'formal-acceptance.json'), status=acceptance['status'],
                                    warm_update_output_max_abs=0, compared_values=65528,
                                    exact_32k_B4_colocated_capacity_passed=True,
                                    universal_first_use_or_future_trajectory_bitwise_identity_claimed=False)
    if result.get(key,{}).get('version') not in (None,current['version']):
        previous=result[key]
        exit_name='AppWorld-DTPO-v2-restart-readiness-exit.json' if key=='DTPO' else 'AppWorld-PPO-v1-restart-readiness-exit.json'
        if (here/'receipts'/exit_name).exists():
            exit_record=read(here/'receipts'/exit_name)
            previous.update(status=exit_record['status'],alive=False,completed_iterations=exit_record['completed_iterations'],
                            latest_checkpointed_iteration=None,exit_receipt=str(here/'receipts'/exit_name))
        result.setdefault(key+'_history',[]).append(previous)
    result[key] = current
recovery_path=here/'receipts/TextCraft-PPO-native-global-OOM-recovery-v1-job.json'
if recovery_path.exists():
    recovery=read(recovery_path)
    restored_path=here/'receipts/native-PPO-restore-global-OOM-20261017-v1.json'
    restored=read(restored_path) if restored_path.exists() else None
    recovery_status=restored['status'] if restored else recovery['status']
    result['protected_TextCraft_jobs'][1]=dict(pid=recovery['pid'],birth=recovery['birth'],physical_gpus=[4,5],
        status=recovery_status,execution_version=recovery['execution_version'],
        execution_patch_commit=recovery['execution_patch_commit'],resumed_completed_iteration=100,
        prior_kernel_OOM=dict(pid=3946004,birth=1792041120.19,completed_log_iteration=103,complete_checkpoint_iteration=100),
        unsaved_iterations_to_rerun=[101,102,103],receipt=str(recovery_path),
        actual_native_restore_receipt=str(restored_path) if restored else None)
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
if recovery_path.exists():
    runtime['latest_TextCraft_PPO_global_OOM_recovery_20261017_v1']=dict(updated_at=now,
        status=recovery_status,pid=recovery['pid'],birth=recovery['birth'],physical_gpus=[4,5],
        checkpoint=recovery['resume_from_path'],execution_patch_commit=recovery['execution_patch_commit'],
        native_resume_receipt=str(recovery_path),prior_completed_log_iteration=103,native_checkpoint_iteration=100,
        unsaved_iterations_to_rerun=[101,102,103],storage_relocation_receipt=str(here/'receipts/inactive-diagnostic-checkpoint-storage-relocation.json'))
write(runtime_path, runtime)

record_path = Path('experiments/rl/RUNTIME_RECORD.md')
record = record_path.read_text(encoding='utf-8')
event = f'''## {now} AppWorld两项正式训练已提交并实际采样；PPO对应暖态输出、32k容量和效率验收完成，goal仍active

DTPO正式执行v1因原AppWorld initialize的100秒ReadTimeout被包装为AppWorldInitializeTaskError、未被原5次重试捕获而退出，完成0轮/0正式optimizer/0检查点，无survivor。只在新固定LOOP owner原except中加入该初始化异常及原requests.Timeout，保持原5次预算、timeout、环境奖励、取消策略与训练方法；8项CPU恢复/原默认行为检查通过。新owner appworld_evals SHA c1a71a8f7860dd0cb192293229429c4024405e1b0f6070d34bc183b1edd05676；DTPO v2两路环境进程已实际回读，原失败回执保留。

DTPO v2提交b519a2bb138dea7519857e3477ecd865adf1232d、driver1395665/出生1792177213.42、physical6/7，复用v5方法4545c558及原固定VERL20bd331b；native PPO v1提交b4ef1664c4fcfbac7a54d665d294753b0522a498、driver1577251/出生1792178946.21、physical2/3，实际双rank回读worker5bb0211a/critic0de81db3/activation242f7919及原actor/core、B4/LoRA8/16、默认图和生产FA0一致。PPO两路环境进程也实际加载同一c1a71a8f owner。两项当前均在原生成及环境交互，完成0/200，无正式检查点；不将初始化或存活称已完成更新。

PPO采用同源状态/输入、当前进程已有原生更新后的native_repeat与native_async原生更新输出对比，65528个value逐值相同，按固定VERL卸载测试对应的atol0/rtol0检查通过，更新期间无梯度/optimizer观测hook。首次native更新与两路后续对照仍保留最大差2.328125，不宣称已查明首次使用差异或所有未来轨迹bitwise相同；正式接受范围见receipts/PPO-native-final-numerical-20261017-v1/formal-acceptance.json。既有完整共置B8/每rank B4、input32768/response32767的oldlogprob/value/GAE/critic/actor容量通过；生产FA0短输入3对中位比0.993725、32k对官方CPU保存参考比0.964055，有限样本不当作正式整轮加速；GPU-only 32k参考原OOM保持。

两项保持原作者40任务组×6条轨迹请求、原LOOP取消、200轮、minibatch32/epochs2/eval5/keep2；原VERL save_freq=2直接写共享/mnt/si0021787ci2/default/lzq/deltatrace_backups/appworld-20261017/<各独立版本>/checkpoints，PPO actor和critic均由原owner保存。原生完整标记尚未产生。TextCraft0/1与4/5身份保持，未重跑GRPO或SQL。只读观测复用原read_job_logs/parse_logs，不建训练parser/logger。goal必须持续至两项200轮及最终共享完整checkpoint核验，当前仍active。详情results_appworld_formal_training_20261017_v1.json与results_PPO_32k_efficiency_20261016_v4.json。

'''
if 'AppWorld两项正式训练已提交并实际采样；PPO对应暖态输出' not in record[:500]:
    record_path.write_text(event + record, encoding='utf-8')
print(json.dumps(dict(updated_at=now, DTPO=[result['DTPO']['pid'],result['DTPO']['completed_iterations']],
                      PPO=[result['PPO']['pid'],result['PPO']['completed_iterations']], goal_status='active')))

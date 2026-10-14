"""Record the actual native PPO control separately from the unresolved DT repair."""
from datetime import datetime
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def read(name):
    value = json.loads((HERE / name).read_bytes())
    return value.get('report', value)


job = read('native-PPO-control-v8-job.json')
prepared = read('prepare_native_PPO_control_v8.json')
bindings = read('verify_native_PPO_bindings_v8.json')
observation = read('read_native_PPO_control_v8.json')
contrast = read('native-DT-GRPO-recorded-contrast-v8.json')
assert observation['job']['pid'] == job['pid'] and observation['job']['birth'] == job['birth']
assert len(bindings['workers']) == 2
for worker in bindings['workers']:
    assert worker['core']['SHA256'] == 'fc2f992b16fb7fb23aebc683ad5f00136cf61fc9013cd426f5046983babe7299'
    for role in worker['roles']:
        if 'actor' in role:
            assert role['actor']['SHA256'] == '3a65e173300be82a7a9e056a96227c4f746eabc3be778ef41c8d50138d52ce6c'
        if 'critic' in role:
            assert role['critic']['SHA256'] == '0de81db36a77e180fb0f6722193c61d74c238ffb81201f4305e697c631ecd3a8'
        assert role['worker']['SHA256'] == '9dfd6ce795384b98faa962269e1eac98ad05ec65e42af2aac434c41db41641d3'

now = datetime.now().astimezone().isoformat()
record = {'observed_local': now, 'version': 'native-PPO-control-v8',
          'status': 'launched_actual_native_actor_critic_bindings_verified_first_iteration_in_progress',
          'user_authorization': '也可以对比PPO，可以用DTPO空出的显卡运行PPO对照',
          'job': job, 'launcher_commit': '920464e4d77a3752e507589dfb6ab51f4fd79e8e',
          'original_upstream_commit': '20bd331b',
          'original_verified_owner_patch_commit': '0b12b2020f2ece724b17e4e3179aa664df2bd737',
          'source_bindings': prepared['source_bindings'],
          'configuration': prepared['fixed_actor_configuration'],
          'critic_native_defaults': prepared['critic_native_defaults'],
          'actual_worker_readback': bindings,
          'latest_observation': {'unix': observation['unix'], 'alive': observation['alive'],
                                 'protected_GRPO_same_identity': observation['protected_GRPO_same_identity'],
                                 'metrics': observation['native_metrics'],
                                 'ray_sessions': observation['ray_sessions']},
          'algorithm_implementation': 'Original fixed verl.trainer.main_ppo, compute_advantage/GAE, CriticWorker/DataParallelPPOCritic, DataParallelPPOActor and native checkpoint manager.',
          'new_training_function_bodies': 0, 'shadow_algorithm_implementations': 0,
          'initial_readback_scope': 'Both original actor and critic optimizers have zero state entries and scheduler0 at the recorded RPC execution time; forward generation has already started.',
          'not_a_DTPO_repair_or_acceptance': True, 'rootcause_resolved': False,
          'accepted_DTPO_repair': None, 'deployed_DTPO_repair': None,
          'old_DTPO_remains_stopped': True, 'GRPO_reruns': 0,
          'hourly_automation': 'PAUSED', 'goal_status': 'active',
          'evidence': {name: {'path': str((HERE / name).resolve()),
                              'SHA256': hashlib.sha256((HERE / name).read_bytes()).hexdigest()}
                       for name in ['native-PPO-control-v8-job.json', 'prepare_native_PPO_control_v8.json',
                                    'verify_native_PPO_bindings_v8.json', 'read_native_PPO_control_v8.json',
                                    'native-DT-GRPO-recorded-contrast-v8.json']}}
dest = HERE / 'native-PPO-control-v8-record.json'
dest.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
runtime_path = ROOT / 'experiments/rl/current_runtime.json'
runtime = json.loads(runtime_path.read_bytes())
runtime = {'20261014_native_PPO_control_v8': record,
           '20261014_native_DT_GRPO_contrast_v8': contrast, **runtime}
runtime_path.write_text(json.dumps(runtime, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
result_path = ROOT / 'experiments/rl/results_textcraft_gradient_rootcause_20261013_v1.json'
result = json.loads(result_path.read_bytes())
result.update(observed_local=now, goal_active=True,
              status='native_PPO_control_running_original_VERL_DTPO_rootcause_unresolved',
              native_PPO_control_v8=record, native_DT_GRPO_contrast_v8=contrast)
result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
texts = {
    'RUNTIME_RECORD.md': f'''## {now} 用户授权原生PPO对照已启动，固定VERL实际导入回读通过

独立PPO run {job['run']}，physical2/3，PID{job['pid']}/出生{job['birth']}；launcher commit920464e4，固定上游20bd331b/已验收owner patch0b12b202。直接调用原verl.trainer.main_ppo和原GAE/CriticWorker/DataParallelPPOCritic，不复制PPO/GAE/critic/optimizer/检查点函数。原actor3a65e173、corefc2f992b、worker9dfd6ce7、checkpointmanagerf65b6538；native critic0de81db3。两实际WorkerDict 2446582/1791947487.82与2449509/1791947516.57均回读上述class/源路径/fullSHA，actor/critic的optimizer state均0、scheduler均0，无instance方法覆盖。既有FSDP事件等待hook78eb717d亦有实际worker回执；hook回执与训练源导入分别记录。

保持actor32k、LoRA8/16、每卡B4、global64/4次更新、entropy.001、dualclip3、PPOepochs1、30数据epochs/330迭代、save5/keep2和原生vLLM图。独立PPO仅使用原GAE/gamma1/lambda1及原critic默认LR1e-5/全量critic，使用原FSDP2/offload能力共存；不是DT算法替换或根因修复验收。已完成模型初始化与首批30交互，仍0/330；尚无完整actor/critic更新可宣称训练通过。观察超时不重交进程。GRPO3174448/1791793629.73保持，不重做；旧DTPO/分母候选仍停止，小时自动检查PAUSED，修复goal active。

既有DT/GRPO记录重新验算：前10轮梯度中位数0.035/0.0295，DT初期是幅度略大且波动更宽。相同旧iteration81输入/权重/原owner的已完成分项测量中，DT PG0.9583064、原GRPO PG0.3295920，比例2.90755、cosine-0.1456073；16个microbatch的entropy/KL/ppoKL前向核对均一致。该配对只说明优势改变方向及大小，不是85轮根因或从头训练反事实；不同前向点的2.208与0.329不配对，旧0.958/0.162分母结论仍撤回。具体证据与范围见{dest}及native-DT-GRPO-recorded-contrast-v8.json。

''',
    'REMOTE_ENVIRONMENT.md': f'''## {now} 当前PPO对照占用已确认

用户明确授权独立原生PPO对照后，已使用释放的physical2/3启动{job['pid']}/出生{job['birth']}，run {job['run']}。实际两rank原actor/core/critic/worker class、路径/fullSHA及optimizer0/scheduler0已回读，直接固定VERL main_ppo/GAE/critic，不重装或复制算法。首次生成已结束、首更新待确认；不能把初始化或存活当完整训练通过。GRPO4/5保留，6/7另有占用不要触碰；后续任何选卡仍需检查mx-smi。旧DTPO和候选保持停止，小时检查PAUSED，goal active。准确配置、版本和实际覆盖见RUNTIME_RECORD顶部及native-PPO-control-v8-record.json。

'''
}
for name, text in texts.items():
    p = ROOT / 'experiments/rl' / name
    p.write_text(text + p.read_text(encoding='utf-8'), encoding='utf-8')
print(json.dumps({'receipt': str(dest.resolve()), 'PPO_pid': job['pid'], 'goal_status': 'active',
                  'rootcause_resolved': False}))

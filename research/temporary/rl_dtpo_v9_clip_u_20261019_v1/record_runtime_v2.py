"""Reuse the current native evidence recorder for the isolated runtime v2."""
from pathlib import Path

owner=Path(__file__).with_name('record_local_status.py')
code=owner.read_text(encoding='utf-8').replace('dtpo-v9-clip-u-postgroup-variance-20261019-v1',
    'dtpo-v9-clip-u-postgroup-variance-20261019-v2')
code=code.replace("== {'a88d479c199f9b3171a7d29e8a5840dc4759a6ea'}", "== {data['controller']['source_commit']}")
code=code.replace('20261019_DTPO_v9_clip_u_six_eight_GPU_current','20261019_DTPO_v9_clip_u_six_eight_GPU_runtime_v2_current')
code=code.replace('DTPO v9：v5指数信用逐事件裁剪，六组八卡正式队列','DTPO v9执行v2：原生hostalloc与RPC内缓存，六组八卡正式队列')
code=code.replace('原host-register8分配配置、','DT使用原Torch cudaHostAlloc、RPC内缓存复用/原批次释放，PPO/GRPO保留host-register8；')
code=code.replace('本次v9双rank32k验收分别记录','本次v9双rank32k验收分别记录；执行v1首DT也因同一MCR限制exit1/0轮，保留原回执。v2仅DT启用pinned_use_cuda_host_register=False及DT_NATIVE_HOST_CACHE_REPLAY_LAYER=0，原批次/最终RPC释放保持。实际capture B4/32768 BF16复制及步幅在两模式均逐值一致；首次分配较慢、原生缓存复用后接近。新执行源和allocation-owner-verification.json分别记录')
code=code.replace("方法与执行源{first['source_commit']}已提交推送", "方法源a88d479c199f9b3171a7d29e8a5840dc4759a6ea、当前执行源{first['source_commit']}已提交推送")
code=code.replace('<对应20261019-v1 run>', '<对应20261019-v2 run>')
# Add only an evidence reference when the natural first-iteration audit exists.
efficiency=Path('experiments/rl/results_dtpo_v9_first_iteration_efficiency_20261019_v2.json')
if efficiency.exists():
    code=code.replace("key='20261019_DTPO_v9_clip_u_six_eight_GPU_runtime_v2_current'",
        "summary['first_iteration_efficiency_result']='experiments/rl/results_dtpo_v9_first_iteration_efficiency_20261019_v2.json'\nkey='20261019_DTPO_v9_clip_u_six_eight_GPU_runtime_v2_current'")
    code=code.replace("本次完成{first['completed']}/200，实际phase={first.get('phase')}。",
        "本次完成{first['completed']}/200，实际phase={first.get('phase')}。首轮自然训练1002.954秒，DT289.211秒、更新175.328秒；与原相同八卡负载v8首轮相比整轮慢16.3%、DT慢85.2%、按输入token折算更新慢17.1%，不能将首轮通过称效率达标。八rank原只读metadata已实读Adam step4/scheduler1、各496状态；未新增模型/DT/optimizer测试。原生缓存释放每rank累计35.7–45.9秒，明确为DT内部计时；完整效率及原日志行号见results_dtpo_v9_first_iteration_efficiency_20261019_v2.json。")
warm=Path('experiments/rl/results_dtpo_v9_two_iteration_efficiency_20261019_v2.json')
if warm.exists():
    code=code.replace("key='20261019_DTPO_v9_clip_u_six_eight_GPU_runtime_v2_current'",
        "summary['two_iteration_efficiency_result']='experiments/rl/results_dtpo_v9_two_iteration_efficiency_20261019_v2.json'\nsummary['prepared_resource_candidate']={'status':'prepared_only_not_natively_validated_not_deployed','receipt':'/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/dt-native-consumed-root-lifetime-20261019-v1/prepared.json','candidate_runner_SHA256':'9f13c898c9faec72f57d77a81966315f25950c251e4381598b8f5442e287c5cf'}\nkey='20261019_DTPO_v9_clip_u_six_eight_GPU_runtime_v2_current'")
    code=code.replace('完整效率及原日志行号见results_dtpo_v9_first_iteration_efficiency_20261019_v2.json。',
        '完整效率及原日志行号见results_dtpo_v9_first_iteration_efficiency_20261019_v2.json。第2完整迭代894.964秒、DT218.311秒、actor163.708秒，原八卡v8对应第2轮整轮约832秒；按输入token折算actor仍慢17.5%，不能将差额都归给首轮。原生actor RPC末尾释放每rank14.923–16.386秒，已经包含在update_actor计时，不相加为额外整轮时长。两轮完整来源见results_dtpo_v9_two_iteration_efficiency_20261019_v2.json。发现当前DT_NATIVE_HOST_CACHE_REPLAY_LAYER=0将已消费root/kwargs保留到B4结束；仅一个runner文件已在dt-native-consumed-root-lifetime-20261019-v1候选中分离消费释放与缓存清理，SHA9f13c898，未原生验证、未部署，不作为已提速。当前正式DT_ROOT/runner仍6a112224，未改训练或原生保存策略。')
preflight=Path('experiments/rl/results_dtpo_v9_resource_preflight_20261019_v2.json')
if preflight.exists():
    code=code.replace('additional_model_or_optimizer_test_calls=0)',
        "additional_model_or_optimizer_test_calls=0,native_checkpoint_observer=(json.loads((A/'native-TextCraft-DT-checkpoint5-observer-v1-job.json').read_bytes()) if (A/'native-TextCraft-DT-checkpoint5-observer-v1-job.json').exists() else None))")
    code=code.replace("key='20261019_DTPO_v9_clip_u_six_eight_GPU_runtime_v2_current'",
        "summary['native_checkpoint_observer']=data.get('native_checkpoint_observer')\nsummary['resource_CPU_preflight_result']='experiments/rl/results_dtpo_v9_resource_preflight_20261019_v2.json'\nkey='20261019_DTPO_v9_clip_u_six_eight_GPU_runtime_v2_current'")
    code=code.replace('当前正式DT_ROOT/runner仍6a112224，未改训练或原生保存策略。',
        "当前正式DT_ROOT/runner仍6a112224，未改训练或原生保存策略。资源比较入口已用原v9行为harness通过CPU预检：B4各32768、CUDA未初始化，GPU比较仍未启动；见results_dtpo_v9_resource_preflight_20261019_v2.json。原检查点CPU读取器已适配本六组精确身份和八rank，TextCraft检查data.pt及原模型/optimizer/scheduler/训练与rollout RNG，AppWorld额外复用原LOOP sampler状态读取。原被动观察器source e0dbb650、PID1657396/1792386726.09已提交，当前状态={str((data.get('native_checkpoint_observer') or {}).get('status'))}，无额外保存、恢复、模型或optimizer调用；不能把等待marker或CPU预检称检查点/数值/效率验收。")
exec(compile(code,str(owner),'exec'),globals())

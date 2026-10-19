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
exec(compile(code,str(owner),'exec'),globals())

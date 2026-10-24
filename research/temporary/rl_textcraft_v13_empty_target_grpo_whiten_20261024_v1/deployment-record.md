## 2026-10-24T22:29:55.816173+08:00 — v13八卡实际进入首轮生成，原初始化导入回执核对通过

八worker实际PID出生/mode/VERL_ROOT与新正式source一致，phase为actor_rollout_generate_sequences；原初始化观察回执/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/runs/textcraft-dtpo-v13-empty-target-grpo-whiten-ssm-fp32-8gpu-formal-20261024-v1/runtime-overrides/actual-worker-imports-v13-eight-initial-v1.json已完成，SHA fcdefc5abdbf40ecf461314a451a6b42abceea805c073dd384b586cfea0f8d03，八rank共32项已加载预期模块SHA匹配。正式source30项文件SHA匹配。DT/readout尚未首次懒加载，资源回执尚待第一次DT生命周期完成，均明确pending；没有重复metadata RPC或安装逐token记录器。v13目前0/330完整迭代，没有新六指标和共享checkpoint，不把进入生成称已完成或质量通过。

方法/执行源仍2d78c254cba5549ec4f8178f37f45dad8d9e5554，后续记录提交不是训练部署源；controller375902/1792851775.18、driver375944/1792851775.55。完整状态=D:\Users\Administrator\Documents\ChatGPT\DeltaTrace\research\temporary\rl_textcraft_v13_empty_target_grpo_whiten_20261024_v1\deployment-status-v1.json，既有回执/OS状态来源=F:/DeltaTraceBackups/plots/textcraft-v13-empty-target-grpo-whiten-eight-gpu-vs-historical-PPO-GRPO-20261024-v1/20261024-222546/existing-state-and-receipts.json。每小时两图任务已回读ACTIVE且指向v13；初始同一快照v13=0/PPO105/GRPO191两PNG已目视确认。

## 2026-10-24T22:28:21.930833+08:00 — v13原生八卡已启动初始化，按用户切换保留v12并更新双图任务

方法/执行源2d78c254cba5549ec4f8178f37f45dad8d9e5554，固定VERL20bd331bdbc9026a5668e11362178e10ab7400c8，方法父版本b7216e23、训练逻辑4545c558。最新用户授权v13无target回退GRPO、最终gu整批归一化，并启动、提交和推送。沿v11原始QVA及fallback，复用原masked_whiten(shift_mean=True)，Q/V作同一仿射变换保持Q-V；PPO、调度、模型/DT内核、奖励、生成保持。PLAN首条记录用户新方法，未将v13作为已证明根因的修复。

14项原生CPU接口/回归检查通过，原保存DP2自然第6轮256条：174有target、82无target（74非零g）；原始QVA与v11逐值相同，最终A范围[-4.904262,4.323452]、均值1.10e-8、RMS0.9999993，Q-V误差最大5.36e-7，mask/有限值通过。新增模型/DT/optimizer/环境调用0。原Hydra --cfg job正式配置通过。该结果不作为新八卡质量或稳定性验收。

用户切换时v12日志连续1..142，完整原生共享checkpoint140的25项必要文件保留，data.pt SHA d76ef438f8490da6fa1a6a23e9a526686320bac5075be7b5878c913c311e13c4，非自然exit0、非数值失败。原f71ff4d8 cleanup的12秒等待超时回执保留；后续只读确认两sessions及八卡全空后用原完成器登记，没有重复清理信号。证据在本audit preserved-v12/previous-v12-stop/original-cleanup-timeout-resolved-v1.json。

v13 controller375902/1792851775.18、driver375944/1792851775.55，八worker实际出生/mode/VERL_ROOT与新source匹配。目前actor_rollout_init_model，0/330；首个完整迭代、实际导入及资源完成回执pending，不能称质量通过。部署状态=D:\Users\Administrator\Documents\ChatGPT\DeltaTrace\research\temporary\rl_textcraft_v13_empty_target_grpo_whiten_20261024_v1\deployment-status-v1.json。新run=/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/runs/textcraft-dtpo-v13-empty-target-grpo-whiten-ssm-fp32-8gpu-formal-20261024-v1。原一次每phase观察器负责新生命周期已有接口回执，本任务未重复metadata或安装逐token记录器。旧v12被动保存提案未执行，梯度根因未证实。

textcraft原heartbeat已更新为v13且ACTIVE，保留每小时/同线程/两图要求。首个快照北京时间2026-10-24T22:25:46.751765+08:00：v13=0、PPO105、GRPO191，共同0，不造v13点；固定0–200仅展示冻结历史。两PNG已目视检查、共享同一snapshot SHA 3cda97e618a6b66a00c4093f7bd3f121772e0eb1fdf1c48a75558b5286b7dc35，PNG/SVG/CSV和原来源回执均保存F:/DeltaTraceBackups/plots/textcraft-v13-empty-target-grpo-whiten-eight-gpu-vs-historical-PPO-GRPO-20261024-v1/20261024-222546。后续有完整迭代时保存完整v13前缀；仅自然330/exit0/八rank完整checkpoint330/最终展示全部完成后暂停。初次配置检查包装器因重复CUDA_VISIBLE_DEVICES关键字在准备前失败，修正后原正式配置通过；并非训练或数值失败。


# AppWorld 环境提取与对照（2026-09-29）

用户最新范围：只提取 LOOP 的原生环境与配置，与项目对照，不运行 LOOP 训练。
本文记录环境证据，不定义新的信用方法或训练预算。

## 来源及调用边界

- LOOP：`apple-aiml-research/ml-loop@f14107a976e5793990329d3193df4742076c5a1d`。
  [原 README 的 Model Training 命令](https://github.com/apple-aiml-research/ml-loop/blob/f14107a976e5793990329d3193df4742076c5a1d/README.md)
  由作者 `phi_agents.rl.config.get_config` 组合；`loop_owner_recipe.py` 只提取其中
  环境、agent、任务 sampler 和生成边界的参考字段。不继承其 trainer、优化器、
  loss、迭代预算、GPU分配或模型加载。
- 环境直接调用原 `AppWorldScenarioRunner` → 原 `MinimalReactAgent` → 原
  `AppWorldInterface` → 已安装 AppWorld CLI。代码解析、交互循环、终止、奖励
  和任务评估均调用原实现，未复制这些逻辑。
- 对照项目：固定 VERL-agent `20bd331bdbc9026a5668e11362178e10ab7400c8` 的
  当前候选目录。`AppWorldWorker` 的 AST 与该固定版本原文件完全一致；外层
  批处理保留先前 inactive-mask 补丁，本次未修改。对照参数是最近停用的
  `full-horizon-checks/dt-AppWorld/job.json` 及其启动入口，不是新正式实验。
- 两边复用同一已安装 AppWorld 资产与后端（含之前的 execute 运输补丁），
  使用独立测试输出根目录；没有重新配置模型或依赖栈。

## 已核对的行为

| 项目 | LOOP 原示例环境 | 项目最近停用配置 | 证据 |
|---|---|---|---|
| 训练任务 | `train_difficulty_1_and_train_difficulty_2`，72项 | `train`，90项 | 两边原 load_task_ids 的实际返回 |
| 评估任务 | `dev`，57项 | `test_normal`，168项 | 两边原 load_task_ids 的实际返回 |
| 交互上限 | 40 | 50 | 固定动作真实执行至各自上限 |
| 提前结束 | 原 `task_completed()` | 原 `task_completed()` | 同一任务第3步 complete_task，两边均结束 |
| 训练环境回报 | 终止时通过测试数 / 全部测试数；错误/无代码惩罚系数均0 | 终止成功10，否则0 | 实测同为通过1/2、success=false：0.5 对 0 |
| 评估回报 | `sparse_reward=true`，成功为1 | 成功为10，`won` 为成功标志 | 原配置和原worker；未作模型成功率比较 |
| 动作格式 | Markdown Python/py代码块，合并多个块 | `<code>`提取，`<think>`参与valid检查 | 4个输入直接调用两边原parser |
| unsafe-syntax 参数 | 原配置显式false | 未传，继承已安装AppWorld默认true | 原接口和AppWorld `init_defaults`；本次未改变项目该参数 |
| 历史 | 完整原消息列表，包含原few-shot提示；关闭字符窗口截断 | 最近2步；拼接历史超过10000字符保留尾部 | 原agent/renderer源码；实测LOOP第40次生成输入含102条消息 |
| 观测限制 | 原agent每次执行输出上限3000 token | renderer对拼接历史做字符限制，同时另附当前observation | 原代码；本次短输出未触发token截断 |
| 单次生成参考上限 | 原Qwen2.5客户端1500 token | 最近测试512 token | 原配置/原启动参数；此次不启动生成 |

**长度不能混称。** 作者示例 `learning_max_seq_len=32000`，所以导出的
agent字段 `max_seq_len_tokens=32000`；但固定 `MinimalReactAgent` 只保存此字段，
未用它执行长度检查。原生成worker传给客户端的是 `max_model_len=32768`；
`VLLMQwen25.generate` 按剩余长度缩短本次生成，不能再生成时抛出
`MaxSeqLenExceeded`，原环境循环捕获后结束并评估已有状态。
本次不复制其模型客户端，也不把32000字段冒充已验证的实际截断行为。
现有项目32768预算处理未在此次修改。未来接线必须使用已有官方生成接口，
保留这些边界的来源，不只移植一个整数。

## 真实环境对照结果

[原始回执](../../research/temporary/rl_upstream_alignment_20260929/loop-project-environment-comparison.json)
记录两条固定动作测试，环境阶段共71.84秒，CPU执行，CUDA未初始化。
另有[源码身份回执](../../research/temporary/rl_upstream_alignment_20260929/loop-environment-source-audit.json)：
23份环境/配置/任务拆分相关文件SHA256与固定源文件一致；项目的原
`build_text_obs` AST与固定VERL原版一致。不是对整套仓库或所有运行行为的认证。

1. `82e2fac_1`：两次变量累加并输出，再调用原 `complete_task()`。两边3步
   输出和官方测试结果一致；提交完成不等于成功，原评估均为1/2通过、success=false。
   LOOP原 `run` 返回0.5，项目3步reward均0。
2. `82e2fac_2`：持续执行相同变量累加。共同前40步的输出和测试结果一致。
   LOOP在40步停；项目此时done=false，继续到50步才done=true。

固定动作只在测试中替代 LLM 输出；原环境和奖励不作mock。测试中的token接口
用空数据占位，因此不验证tokenizer、3000/32768边界、模型生成、任务性能、
分布式调度或训练数值。本结果不能称“任务性能不下降”。

## 复用与状态

`experiments/rl/environments/metax_loop.env.sh` 指向原 `third_party/ml-loop-f14107a…`，
不是训练适配候选。执行 `loop_owner_recipe.py --owner-root "$LOOP_ROOT" --output …`
可重新导出同一作者环境配置。原环境使用方直接导入上述owner类；没有另写环境实现。
完整测试入口位于 `research/temporary/rl_upstream_alignment_20260929/run_environment_comparison.sh`。

此前误展开的LOOP模型/vLLM探针已停止；新LOOP训练入口、补丁器和信用适配草稿
移至本地临时归档，未进入运行入口。现有DT Q/V/A、PPO/GRPO、WebShop和预算未改。
环境差异已经列明，尚未将LOOP环境接入现有训练collector；不把对照完成写成训练完成。

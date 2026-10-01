# 本机环境与模型入口单元测试（2026-09-30）

本次目标仅包括：官方训练/评估环境、配置，以及 Qwen3.5-9B 接入环境的入口。
验收入口只执行这些本机单元测试；不执行训练算法、DT、模型权重或远端服务。

## 结果

**83 passed，0 failed，0 skipped。**

| 检查组 | 通过 | 实际覆盖 |
|---|---:|---|
| 官方配置 | 8 | 执行作者原 Bash 的参数展开；启动命令由记录夹具接收，配置由原 Hydra/配置类解析 |
| SkyRL SQL | 26 | 原 `skyrl-gym/tests/test_sql.py`，未修改断言；配置组另用真实小型 SQLite 调用原环境查询、提交和六轮终止 |
| AgentGym 客户端 | 7 | 原 TextCraft/WebArena 客户端、factory、动作解析、reset、奖励与终止；HTTP 响应为夹具 |
| WebShop 环境边界 | 6 | 原训练/验证划分、采样、组内任务一致性、奖励映射；Ray 运输及环境回复为夹具 |
| 模型参数与计算负载 | 11 | 原脚本与 Qwen3.5 路径覆盖后的完整配置对照；原 worker 的 batch 归一化、原 collector 行数、原 SkyRL 预算计算 |
| Qwen3.5 官方入口 | 7 | 真实 tokenizer、chat template、config、Transformers 模型注册；原 VERL/AgentGym tokenizer、原 SkyRL 生成头、原 AgentGym 消息与 mask |
| LOOP 环境与入口 | 18 | 原任务 sampler、agent、40 步 episode 循环、提前结束、预算终止、完整历史、训练/评估奖励；真实 tokenizer 的原 Qwen3 客户端，world/HTTP 为夹具 |

所有环境、parser、奖励器、collector 均来自下表固定官方源码。需要隔离重量级导入时，
单测从原文件 AST 取出原函数或配置赋值直接执行，没有抄写一套同名实现。
边界夹具只提供可控的输入和记录调用，不进入生产入口。
本机单测确认的是这些边界和配置行为；实际服务耗时不由夹具推算。

## 官方来源

| 对象 | 固定版本 | 使用入口 |
|---|---|---|
| WebShop / VERL-agent | `20bd331bdbc9026a5668e11362178e10ab7400c8` | `examples/grpo_trainer/run_webshop.sh`，原环境和 collector |
| AppWorld / LOOP | `f14107a976e5793990329d3193df4742076c5a1d` | README 原训练命令的环境配置，原 `appworld_scenario_runner`、agent、episode loop；仅环境部分 |
| SkyRL-SQL | `7d94ccf0eac3439c1731ce32018bf043dd639806` | `examples/train/text_to_sql/run_skyrl_sql.sh`，原 SQL 环境及测试 |
| AgentGym-RL | `82402a99c62a293735a3f412fb8ac9a600673bc0` | TextCraft/WebArena 原 train/eval 脚本、配置和 RolloutHandler |
| AgentGym 客户端 | `d014732d9fe39b975c368c03749bfd50950067f6` | 上述 AgentGym-RL 的实际 gitlink，原 TextCraft/WebArena 客户端 |
| Qwen3.5-9B | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` | 官方 config、tokenizer.json、tokenizer_config、chat_template；没有权重 |

源码与模型入口资产 SHA256、依赖清单、JUnit 和资源回执位于
[local-environment-final](../../research/temporary/rl_upstream_alignment_20260929/local-environment-final/)。
机器可读结果：[results_local_upstream_tests.json](results_local_upstream_tests.json)。
模型元数据的原来源为 [Qwen 官方仓库](https://huggingface.co/Qwen/Qwen3.5-9B/tree/c202236235762e1c871ad0ccb60c8ee5ba337b9a)。

提交包含测试、补丁和回执，不包含依赖缓存或上游仓库副本。新机器复用已有检出；
缺少时，从以下官方仓库取得上表固定 commit，放到
`research/temporary/rl_upstream_alignment_20260929/recipe-sources/` 下对应目录：

| 官方仓库 | 本地目录名 |
|---|---|
| [langfengQ/verl-agent](https://github.com/langfengQ/verl-agent) | `verl-agent-20bd331` |
| [NovaSky-AI/SkyRL](https://github.com/NovaSky-AI/SkyRL) | `SkyRL-7d94ccf0eac3439c1731ce32018bf043dd639806` |
| [WooooDyy/AgentGym-RL](https://github.com/WooooDyy/AgentGym-RL) | `AgentGym-RL-82402a99c62a293735a3f412fb8ac9a600673bc0` |
| [WooooDyy/AgentGym](https://github.com/WooooDyy/AgentGym) | `AgentGym-d014732d9fe39b975c368c03749bfd50950067f6` |
| [apple-aiml-research/ml-loop](https://github.com/apple-aiml-research/ml-loop) | `ml-loop` |

Qwen 官方固定版本的四个文件 `config.json`、`tokenizer_config.json`、
`tokenizer.json`、`chat_template.jinja` 放入相邻的 `qwen35-entry-assets/`。
检查源码和资产时使用回执中的SHA256；不下载权重。测试套件本身不执行上述获取动作。

## 已复现并修复的入口缺陷

补丁只作用于原文件，单测使用临时副本；保存的官方源码保持原样。

1. **AgentGym 评估工作目录重复。** 原脚本先 `cd AgentGym-RL`，再进入
   `AgentGym-RL/scripts`，路径不存在。改为在当前目录调用原 `scripts/model_merger.py`。
   原评估参数完全相同。[路径与参数补丁](patch_agentgym_eval_paths.py)。
2. **AgentGym 原脚本忽略命令行模型参数。** 四个 train/eval 脚本补上 `"$@"`，
   由原 Hydra 接收 `Qwen/Qwen3.5-9B` 路径。无参数时展开与原脚本完全相同；
   指定模型时只允许模型路径及原 Hydra 自带的关联插值发生变化。
3. **LOOP Qwen3 客户端写死旧 token ID。** 使用现有官方 Qwen3 客户端，
   三个特殊 token ID 改从当前 tokenizer 读取；生成头探针使用 user 消息，
   满足 Qwen3.5 官方模板要求。EOS=248046、pad=248044，原动作 token、logprob、
   observation mask 和 HTTP 参数仍由原客户端处理。[补丁](patch_loop_tokenizer_entry.py)。
4. **Transformers 5 的 chat-template 默认返回类型变化。** LOOP 和 AgentGym 的
   原入口需要 `list[int]`，但默认得到 `BatchEncoding`。明确调用官方
   `return_dict=False`，保留原断言和处理流程。
   [AgentGym 入口补丁](patch_agentgym_model_entry.py)。

Qwen3.5 模型注册检查使用 Transformers **5.13.0**：官方 AutoConfig 能识别
`qwen3_5` 和 `qwen3_5_text`，官方 AutoModelForCausalLM 映射均指向
`Qwen3_5ForCausalLM`。只检查注册和元数据，不实例化模型。
旧的4.51.1仅供其他CPU环境/config导入，不作为Qwen3.5模型支持证据。

LOOP 模型入口单测实际调用原 `generate`，只替换 HTTP 回复；检查生成 token
原样保留、stop IDs、1500 输出预算，以及实际编码超过32768时不再发请求。
其训练72任务和dev57任务来自原 split，检查无重叠和每轮采样不重复。

## 按官方实际行为核算计算负载

**范围说明：本节仅是2026-09-30原任务入口的CPU对照。** 当时接入曾把每次有效
response当成训练行，造成SQL更新次数放大；该缺陷随后由完整轨迹接入修复。
这里不再保留过期的“当前每轮120次更新”“正式规模尚未启动”表述。
实际版本、训练行单位、更新次数与尚未解决的执行开销统一见
[当前版本与修复账本](RUNTIME_RECORD.md)。本节CPU结果不能替代正式运行证据。

以下是原脚本、原配置与原代码所定义的单位；没有运行任何优化器。
模型路径修改不改变任务数量、交互上限或数据展开方式。

| 环境 | 原始 batch / 交互 | 原生训练数据单位与更新负载 | 长度口径 |
|---|---|---|---|
| WebShop | 16组×8=128条，最多15次交互 | 每个 active 交互1行，最多1920行；全局 minibatch=64、1遍，满15步为30次更新 | 每状态 prompt≤4096、response≤512，history=2；不会自行改成累计32k历史 |
| SkyRL-SQL | 256个prompt×5=1280条，最多6轮 | mini=256以prompt计；原预算表达式 `256/256×1=1` 次更新/batch，不能把1280/256误算成5次 | 初始prompt≤6000，累计输入≤29000、当轮生成≤3000 |
| TextCraft | 32组×8=256条，最多30轮 | 原 worker 将mini=8乘8，得到全局64；每batch实际4次更新 | 训练512+10240=10752；每轮生成≤512；评估750+14098=14848 |
| WebArena | 32组×4=128条，最多15轮 | 原 worker 将mini=4乘4，得到全局16；每batch实际8次更新 | 训练/评估750+14098=14848；每轮生成≤512 |
| AppWorld | 仅采用原环境；每轨迹最多40次交互 | 原agent生成一次、执行一次后推进；不引入LOOP训练器、优化预算或额外轨迹展开 | 完整历史；原生成1500、观测文本上限3000、模型上限32768，沿用原预算终止 |

AgentGym 配置写 `ppo_epochs=2`，但此固定版本的原 `update_policy` 没有读取它来
驱动第二遍更新，该字段只进入MFU报告。因此这里按实际原代码记4/8次，保留原行为，
不“补上”第二遍。两任务model cap配置为32768，但原handler的实际训练保留长度是
`min(model_cap, prompt_length+response_length)`；上述10752/14848不能写成32k。

WebShop 原 `adjust_batch` 还会按整除要求复制少量行：当前官方两卡设置补到32的
倍数。原有效交互行数测试不把这些官方对齐行称为新增轨迹。
actor microbatch从8减到4，单rank全局mini份额仍是32，累积次数4→8；
**全局minibatch仍是64，不会把更新次数放大成“行数÷4”。**

SkyRL-SQL/AgentGym 原脚本的多卡数、并行方式、生成采样数、训练epoch和验证频率
直接继承。仅换模型时完整配置相等检查通过；不另拼预算。
原 epoch 为 WebShop150、SQL30、TextCraft30、WebArena25，epoch是数据集遍数，
不是环境步数，也不是单卡microbatch次数。

同一模型、设备和输入长度下，对照的是生成请求数、输入/输出token数、训练样本行、
全局minibatch、遍数及官方padding，而非只比较一个“step”名字。
当前入口没有新增逐token模型调用、重复collector、额外更新循环或服务重建。
这些负载的配置与代码单位已对齐；本机不把该结论换算成未经测量的远端秒数。

## 复跑与资源

在仓库根目录执行：

```powershell
./experiments/rl/test_local_owner_suite.ps1
```

七组串行，最终成功执行总计 **55.45秒**（不含源码获取、导入依赖准备及故障排查）。
采样进程树RSS峰值 **575.45 MiB**，系统可用内存最低 **16.37 GiB**；
每组前后物理显存均为 **0 MiB**。显存数据是前后读数，不冒充连续峰值。

Windows Job Object对本次子进程树施加4GiB提交内存硬上限；每0.25秒采样，
系统可用内存下限6GiB、每组300秒超时、CPU线程2、CUDA/HIP设备隐藏。
保护只作用于本次子进程，HF离线、W&B禁用。

复用两套现有解释器：

- `C:/Users/Administrator/miniconda3/envs/pytorch/python.exe`：Python3.11，普通配置和环境边界检查。
- `C:/Users/Administrator/miniconda3/python.exe`：Python3.12，LOOP原生3.12语法和Qwen3.5入口。

隔离的依赖位于 `research/temporary/rl_local_test_deps`、`rl_local_test_deps312`、
`rl_local_model_entry_deps`。入口已设置各组PYTHONPATH，复跑不安装包、不下载模型或资产、
不启动Ray/vLLM/任务服务。原框架分别使用独立测试进程，避免同名VERL包混用。

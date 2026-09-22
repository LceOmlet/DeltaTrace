# A6000 环境记录与复用入口

整理日期：2026-09-21。**这台机器已经配置过环境；恢复工作应从复用开始。**
本文件记录环境、路径和已知问题。RL 方法只以 [PLAN.md](PLAN.md) 为准；
本文件和历史运行记录不代表该计划已实现或 DT 多轮训练已经验收通过。

## 连接与加载配置

本机 PowerShell：

```powershell
ssh -o BatchMode=yes -J 4090 -p 2501 liangchen@10.70.5.230
```

不使用本机别名的同一入口：

```powershell
ssh -J chen@e1b4acb6c4c142d5a00d326cec07fb11.hn.takin.cc:10179 -p 2501 liangchen@10.70.5.230
```

2026-09-21 用户更新 A6000 内网地址为 `10.70.5.230`。已通过旧地址记录的
SSH 主机密钥验证并成功登录，主机名 `hit`、账号 `liangchen`；新地址的
known_hosts 记录复用该已验证密钥。该变更只适用于 A6000，不改 3090 记录。

远端 Bash：

```bash
cd /data/liangchen/deltatrace_resume_20260917/repo/deltatrace
source experiments/rl/environments/a6000.env.sh
```

该文件只导出配置，不安装依赖、不编译、不下载、不启动训练。无需新建 venv
或反复激活 conda；直接调用 `"$VENV_PYTHON"`。其他机器应覆盖自己的路径，
不能直接套用此主机的编译产物。

## 固定资源

| 资源 | 已有位置 |
|---|---|
| 运行根目录 | `/data/liangchen/deltatrace_resume_20260917` |
| 远端 DT 源码及 RL 脚本 | `/data/liangchen/deltatrace_resume_20260917/repo/deltatrace` |
| Python 入口 | `/data/liangchen/deltatrace_resume_20260917/env/bin/python` |
| 日志中解析到的 Python 包目录 | `/data/liangchen/conda_envs/mem1_train/lib/python3.12/site-packages` |
| 模型 | `/data/liangchen/models/Qwen3.5-9B` |
| 运行时权威配置 | `/data/liangchen/deltatrace_resume_20260917/environment.json` |
| VERL | `/data/liangchen/deltatrace_resume_20260917/third_party/verl-agent2` |
| WebShop | `/data/liangchen/deltatrace_resume_20260917/third_party/webshop` |
| AppWorld | `/data/liangchen/deltatrace_resume_20260917/third_party/appworld` |
| DT official root | `/data/liangchen/deltatrace_resume_20260917/flashtrace_paper` |
| DT finite FA 库 | `/data/liangchen/deltatrace_resume_20260917/build/libdeltatrace_fa_finite_bf16_d256.so` |
| Triton 缓存 | `/data/liangchen/deltatrace_resume_20260917/cache/triton` |
| Inductor 缓存 | `/data/liangchen/deltatrace_resume_20260917/cache/inductor` |
| Java | `/usr/lib/jvm/java-17-openjdk-amd64` |

远端 DT 目录是同步的源码目录，不应假设它有 Git 元数据。它的 `profiles/`
和 `clean/` 位于目录顶层；本地仓库则在 `deltatrace/` 子目录下。

`environment.json` 已记录模型配置/tokenizer 哈希、native-stage 源码哈希、
finite 库哈希、扩展目录、`dt_dynamic_shapes=true` 和运行变量。
[历史快照](environments/a6000.owner-environment.snapshot.json)来自本地已保存的
`research/temporary/qwen35_fleet_20260917/runtime-receipts-v4/environment.json`，
用于离线查阅；**不要用此历史快照覆盖远端当前配置**。后续兼容性修改可能使
部分源码哈希变化，应记录差异，不能靠重装抹掉差异。

## 版本与兼容性

- VERL 固定基线：`732f37acd7684b8c24d14ba3ededfe9fab1ed472`。
- AppWorld 固定基线：`42b5bcf3cd334fee33f0c37c02070a9f5807add5`。
- 上游来源见 [upstream.lock](upstream.lock)；附加 RL 包版本见
  [requirements-upstream.txt](requirements-upstream.txt)。这些是安装规格，不是
  最新实测的完整 `pip freeze`，不能据此宣称当前环境完全一致。
- 2026-09-21 的模型加载日志显示 Transformers `5.13.0`、Python `3.12`；
  其他已安装包应直接通过下方只读命令查询，不重装来“确认版本”。
- VERL/HF 兼容性修改由 [patch_verl_agent2.py](patch_verl_agent2.py)维护。
  这是打过补丁的上游代码；启动脚本会调用该补丁脚本，但不会执行 pip、
  重新下载权重或重新编译 CUDA 扩展。
- `PARAM_OFFLOAD=False`：当前 FSDP2 路径曾进入只接受 FSDP1 的参数卸载
  helper 并触发 assertion。保留此设置；该错误不意味着环境需要重建。
- 提前加载本配置，让 Ray/模型首次 import 时即看到持久缓存路径和
  `FLA_BOUNDED_NORM_TUNING=1`、`FLASHTRACE_HYBRID_ATTENTION_CPU_CACHE=1`。
  保留已有缓存；首次新图编译与每次从空缓存编译是不同问题。

FlashAttention 的已知情况：主机 glibc 为 `2.31`，曾使用的预编译 wheel 要求
`GLIBC_2.32`，不能再次安装那个不兼容 wheel。已构建并安装的产物位于：

```text
/data/liangchen/deltatrace_resume_20260917/downloads/flash-build2/flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl
```

这是基于 FlashAttention `2.8.3` 源码、修改构建范围后得到的 **BF16、head_dim=256、
causal、sm80 前向专用构建**，不是完整通用 wheel；不能据其前向通过推断 actor
反向已受支持。相关工作目录曾为 `/tmp/flash_attn_build2`，不是持久依赖入口。
后续应优先复用上述 wheel 和已有 finite 库，先定位具体缺失能力，再决定是否修复。

## 三个任务的现成资产

- WebShop 已有 1000 商品数据和 Lucene `indexes_1k`；VERL 内通过链接访问。
  这是 small 配置，不是完整商品库。
- Sokoban 使用现有上游环境；当前 text-only 启动设置为 `tiny_rgb_array`、
  `6×6`、一个箱子。
- AppWorld 已有官方数据。训练使用服务时，查
  `"$APPWORLD_ROOT/appworld_ports.ports"` 及 VERL 目录同名文件/链接，并检查
  对应服务是否存活。文件存在不等于服务存活，不要重复启动第二套服务。
- `prepare_task_assets.sh` 在 AppWorld 数据缺失时会运行安装/下载命令；
  **它不是日常恢复工作的入口**，不能把它当成纯只读检查。

## 恢复前的轻量只读检查

```bash
source experiments/rl/environments/a6000.env.sh
test -x "$VENV_PYTHON"
test -f "$MODEL_PATH/config.json"
test -f "$DT_ENVIRONMENT_JSON"
test -d "$VERL_ROOT/verl"
"$VENV_PYTHON" -c 'import sys; print(sys.executable); print(sys.version)'
"$VENV_PYTHON" -m pip list --format=json
readlink -f "$VENV_PYTHON"
ls -ld "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu --format=csv
pgrep -a -u "$USER" -f 'verl.trainer.main_ppo|appworld'
```

这些检查不加载模型、不占用训练显存。按当前占用选择空闲卡；历史 GPU 1/2/4
都不是预留卡。断线后先确认已有训练/服务是否仍在运行，再发起下一次测试。
若 SSH banner 超时，先排查连接，不创建新环境。

训练脚本路径为 `experiments/rl/run_verl_agent.sh`。当前 DT 适配代码尚未按
[PLAN.md](PLAN.md) 完成核对与修正，故此处撤下旧 `METHOD=dt` 启动示例，
避免将旧算法入口当作固定计划的可用实现。环境复用和上述只读检查仍然有效。

`32768` 是要求的总上下文上限，实际轨迹长度必须从日志另行报告。

## 已验证范围与尚未通过的部分

- [已有任务环境记录](results_env_smoke.json)：WebShop search、Sokoban step、
  本地 AppWorld `help()` 曾通过，支持复用现有资产，不代表多轮 RL 已通过。
- 2026-09-21 独立 owner DT 前向/归因曾运行；它与 PPO actor 反向验收分开。
- 2026-09-21 11:47 的 WebShop DT 联调在 finite decoder 的 `bmm` 报
  `mixed torch.Tensor and DTensor`。后续重试曾打印
  `owner FSDP2 lifecycle modules=34 remaining_dtensor=0`，但目前没有取得其
  最终成功退出记录。恢复时先查该次 Ray 日志，不应从安装依赖重新开始。
- 当时 Ray 日志入口：`/backup01/ztl/tmp/bash/ray/session_latest/logs`；
  后续 session 会改变这个链接，定位后应保留具体 session 路径。
- 旧 `results_counterfactual_32k_a6000.json` 使用旧信用方案，不能作为当前
  token DT 方案成功的证据。

## 2026-09-21 新地址恢复检查

- 已通过 `10.70.5.230:2501` 经 4090 登录，并验证它与旧地址的 SSH 主机密钥相同。
- 已同步本地唯一 PLAN、状态 README、本记录和 `a6000.env.sh`；旧文档快照保存在
  运行根目录的 `receipts/docs-before-sync-*` 下，不是活动方法规范。
- 已实际 source `a6000.env.sh` 并调用原有 Python，没有安装、下载或编译步骤。
  Python 解析位置为 `/backup01/liangchen/conda_envs/mem1_train/bin/python3.12`。
- 已查询安装版本：Torch `2.8.0+cu128`、Transformers `5.13.0`、
  VERL `0.3.1.dev0`、AppWorld `0.2.0.dev0`；VERL Git HEAD 与上述固定基线一致。
- 模型 `config.json` 及两个持久缓存目录存在。已有 AppWorld 7000--7004 进程存在；
  本次未重启服务，也没有据进程存在宣称任务评估通过。
- GPU 占用检查发现所有六张卡都有已有进程。GPU 1/2/5 约各占 1.8 GiB，
  GPU 4 约占 40.7 GiB，其他卡也有进程；这些不是本任务的资源预留。
  未终止这些进程，未启动新的训练。
- `*.sh` 同步时使用 LF。PowerShell 直接将 CRLF 脚本管入远端 Bash 会导致
  `true\\r` 一类解析错误；应修复传输换行，不重配 Python/CUDA 环境。

上述是连接与环境复用检查。当前固定 Q/V 方法的三个任务训练仍未验收。

## 2026-09-22 目标读出与运行记录

- 继续复用上述 Python、权重和缓存，没有安装或下载步骤。
- 新增 categorical target 支持位于正式 `clean/qwen35/qwen35_answer_finite.py`
  与 `qwen35_dense_finite_runner.py`；同步时保留远端已有动态 shape 执行兼容修改。
  修改前源码保存在 `receipts/readout-before-20260922T084710`。
- 真实模型探针及 CPU 检查记录在运行根目录的 `receipts/readout-20260922/`。
  当次使用 GPU 3 前确认其空闲；该编号不构成后续预留。
- DT 训练现在保持 actor 的 SDPA。奖励事件读出只需原始前向与 head，不需要
  前向专用 FlashAttention wheel 的反向；完整 finite 对照独立运行。
- `VERL_TRIM_SHARED_PADDING=1` 在 HF rollout 和 actor 内移除共有左 padding
  的无效计算。其改动前源码保存在 `receipts/readout-20260922/before-padding/`。
  `patch_verl_agent2.py` 维护该补丁，不重新安装 VERL 或 Transformers。

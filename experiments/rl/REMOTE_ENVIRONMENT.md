# 远端环境记录与复用入口

**当前执行平台（2026-09-22 用户最新指令）：只用 MetaX，不再使用 A6000。**
下方 A6000 内容仅保留为已有环境与历史验证记录。当前任务没有在 A6000
运行的自有训练/探针作业；后续从本文的 MetaX 复用入口恢复。

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
- 历史生成前后读出使用 actor 的 SDPA 和原始 head；该读出方法已撤下，
  其探针不能证明当前 EOS finite 归因路径通过。
- `VERL_TRIM_SHARED_PADDING=1` 在 HF rollout 和 actor 内移除共有左 padding
  的无效计算。其改动前源码保存在 `receipts/readout-20260922/before-padding/`。
  `patch_verl_agent2.py` 维护该补丁，不重新安装 VERL 或 Transformers。

## 2026-09-22 EOS 事件接入修订

- 当前 producer 调用正式 finite `attribute`，临时通过 HF 公共接口切换到 FA，
  完成或异常后恢复 actor 原后端。FA wheel 仍只用于归因前向/重放，PPO 反向保留 SDPA。
- 显式复用 `environment.json.qwen35.dt_dynamic_shapes` 和 `dt_compiler_options`，
  不重新编译/清空持久缓存作为恢复步骤。对应远端 execution overlay 已存在，
  不用本地基础源码直接覆盖它。
- 本次只同步 RL 适配代码、测试和文档，未安装软件、下载权重、重启 AppWorld 服务。
- 新记录目录：`/data/liangchen/deltatrace_resume_20260917/receipts/eos-events-20260922/`。
  GPU 训练结果以该方法的新日志为准；CPU 验证不能替代。

## 2026-09-22 MetaX 资源检查

- 用户提供入口：`ssh -p 32036 root@ssh.v5000-prod-gw.nhss.zhejianglab.com`。
  不在任何仓库文件或运行配置中存放登录密码。
- 查询时 8 张 MetaX C550，各 65536 MiB。GPU 0–2 有 vLLM 进程；GPU 3–7
  无计算进程、利用率 0%、各约 864 MiB 基础占用。此记录不是预留，启动前重查。
- 权重已存在：`/mnt/si0021787ci2/default/models/Qwen3.5-9B`。
- 已有专用运行根目录：
  `/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_qwen35_20260912`。
  `env/bin/python` 可运行，复用系统 MetaX Torch，不新建第二套依赖环境。
- 该专用环境：Torch `2.8.0+metax3.5.3.9`、Transformers `5.13.0`、
  Triton `3.0.0+metax3.5.3.9`、FA `2.6.3+metax3.5.3.9torch2.8`、FLA `0.4.1`；
  当前 Python 尚未发现 VERL 模块。不能把默认 `/opt/conda/bin/python`
  的 Transformers `4.57.3` 当成这套专用环境。
- 原 `build/libdeltatrace_fa_finite_bf16_d256.so` 的 ELF 依赖是 MetaX 的
  `libruntime_cu.so`、`libmccompiler.so`、`libmcruntime.so` 等，已有平台产物。
- 旧环境 JSON 中路径仍为 `/mnt/geogpt-doc-new/...`，恢复时须核对当前挂载映射；
  路径变化不是重装依据。旧运行目录与其他服务不覆盖、不清缓存。
- 原 runtime 设置含 `MACA_TORCH_COMPILE_CONF=maca.disable_maca_triton_heuristics:1`
  和 `FLA_BOUNDED_NORM_TUNING=1`，动态编译选项也已记录。
- 以上只证明资源与已有环境存在；不证明本轮 RL/DT 已在 MetaX 跑通。

## 2026-09-22 MetaX 复用入口与实测故障

- 当前隔离源码/日志根目录：
  `/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922`。
  源码在 `repo`，运行配置在 `environment.json`，记录在 `receipts`。
  旧 20260912 目录仍提供 Python、有限传播库及持久缓存，没有被覆盖。
- 在新源码目录执行 `source experiments/rl/environments/metax.env.sh`。
  该文件只导出已核验路径和 `MACA_PATH=/opt/maca` 等已有运行变量；不安装。
  原 JSON 的旧挂载路径只在新运行配置中修正，没有修改旧配置。
- 使用前查询 `mx-smi`，再同时设置 `CUDA_VISIBLE_DEVICES` 和
  `MACA_VISIBLE_DEVICES`。本轮探针使用 GPU 3，结束后已释放；没有资源预留。
- 该 Python 尚缺 VERL、hydra-core、tensordict、gym/gym-sokoban、codetiming、
  wandb 和 AppWorld。本轮没有安装这些包。三个官方正奖励轨迹从 A6000 导出，
  在 MetaX 复用原始 token/reward 夹具，只检查模型归因，不冒充当地任务服务或训练。
- 实测 MetaX 编译后的 categorical `FiniteAnswerOps` 触发非法地址；独立小算子
  在 no-grad 下复现。使用同一正式 seed 的 eager 执行可越过该错误。新增显式
  `qwen35.dt_answer_compiled=false`，仅用于此运行配置；owner 默认仍为 true，
  其他有限传播图保持既有动态编译。未复制 seed 算法，也未清编译缓存。
- 修复前的真实 Qwen 探针未通过已有归因守恒阈值。当时记录：原生目标差及 seed
  log-prob 差均为 `0.015408515930175781`，head 输入端有限贡献为
  `0.08401765790792126`，最终 norm 输入端为 `0.08037153781038953`，
  全部 source token 归因和为 `0.05778632130990635`。32 层原生重放 L2 差为 0。
  这将差异定位到 head 有限拉回边界；该项已由下面的舍入边界修复处理，
  后续层的整网误差仍单独记录，不改变 PLAN。
- 该次失败 trace 实际长度 266（保持总上限 32768），含诊断用时约 19.79 秒，
  peak allocated 为 19426050048 字节。这不是 32k 容量、训练吞吐或已通过的
  信用分配结果。首次/后续编译成本尚未完成受控测量。
- `receipts/native-eos.json` 保留失败状态和完整逐层诊断；
  `receipts/metax-runtime.json` 保留版本、路径及源码哈希；
  `receipts/probe_answer_seed.py` 和对应日志保留编译器复现。
  本地结果索引为 `results_eos_events.json`。阈值未放宽，归因未归一化，未启动 PPO 更新。

## 2026-09-22 head 舍入边界修复

- 原生 BF16 head 的 `delta(logits)` 与未舍入线性运算 `W delta(hidden)` 不同。
  在捕获的真实操作数上，单纯提高转置乘法精度仍有约 0.087 的差额，主要来自
  原生输出舍入。正式 head 有限规则现已使用已有 `_secant` 计入该操作，再做
  类别行的 FP32 转置乘法；没有改变模型原生输出或把最终归因重新缩放。
- 所需隐藏行由已有 final norm hook 按正式 target selection 打包，只保留
  目标 predictor 行；不复制完整序列，不增加原生模型前向。默认完整词表
  文本 seed 路径保留；categorical seed 必须提供对应的原生隐藏行。
- MetaX 继续使用 `dt_answer_compiled=false`；A6000 保持默认编译设置。
  没有安装依赖、下载权重、重建缓存或重启已有服务。
- A6000 CPU 回归 54 项通过；MetaX 重跑其中 12 项 head 回归通过。
  A6000 GPU 3 确认空闲后，9 项宽度 4096 的编译/eager 对照通过，最大有限
  边界误差 `1.1920928955078125e-7`。编号仍不代表资源预留。
- MetaX 三任务首个真实 head 检查误差分别为：Sokoban `5.57e-8`、WebShop
  `3.41e-8`、AppWorld `7.52e-8`。A6000 真实 Qwen 首个 head 检查误差约 `1.79e-7`。
  三任务的整网归因仍未通过原有守恒阈值；head 修复通过不等于完整 DT/训练通过。
- MetaX 新记录：运行根目录下 `receipts/head-fix/`，包含捕获的 `head.pt`、
  修改前后诊断、三个任务的独立日志和 `task-summary.json`。
  A6000 新记录：`/data/liangchen/deltatrace_resume_20260917/receipts/head-fix-20260922/`，
  包含 CPU、编译对照及原生整网结果。修改前 owner 源码在其 `before/` 中。
- A6000 仅在已有 owner overlay 上应用相应差分；保留此前去除 head shape hook
  的执行兼容修改。原始隐藏行通过已有 norm hook 获取，避免依赖该旧 hook。

## 2026-09-22 MetaX 训练组件接入（当前入口）

- 用户指定只使用 MetaX；不再在 A6000 启动测试或训练。Goal 保持 active，
  三任务完整 DT→PPO 尚未通过，不能把本节的环境准备记成训练成功。
- 已在原 `deltatrace_qwen35_20260912/env` 补齐缺失组件，没有新建环境。
  固定 VERL 源码：`$DT_RUNTIME_ROOT/third_party/verl-agent2-732f37acd7684b8c24d14ba3ededfe9fab1ed472`，
  editable 安装时使用 `--no-deps`，随后应用现有 `patch_verl_agent2.py`。
  AppWorld 源码同层：`appworld-42b5bcf3cd334fee33f0c37c02070a9f5807add5`。
  路径已写入 `environments/metax.env.sh`，恢复时不要重复下载或安装。
- 新增 hydra-core 1.3.2、tensordict 0.10.0、gym 0.26.2、gym-sokoban 0.0.6、
  codetiming 1.4.0、torchdata 0.11.0、wandb 0.30.0；AppWorld 及 WebShop
  必需包亦已安装。安装时用版本约束保护原 Torch、Transformers、Triton、
  FlashAttention、NumPy、PEFT、Ray、Accelerate 和 torchvision。
- 记录目录 `$DT_RUNTIME_ROOT/receipts/training-setup/` 保存安装前后 freeze、
  保护约束、pip 安装报告、固定源码来源、CPU 回归和 Sokoban 奖励记录。
  42 项既有组合/collector/PPO 回归通过，Sokoban 官方脚本奖励接口通过。
- WebShop 新增 Java 17.0.20；apt 原有索引过期导致 404，更新索引后安装，
  未升级模型依赖。断线后的 dpkg 配置已完成；`java -version` 已通过。
  商品数据下载日志为 `receipts/webshop-assets.log`；首次 Google Drive 下载
  因该主机到 `drive.google.com:443` 的网络不可达而失败。包安装已成功，
  此问题需要取得官方资产，不应重新安装包。之后还需构建官方 1k Lucene
  index。不能运行完整旧 `setup.sh` 重装依赖。
- AppWorld GitHub 源码归档含 Git LFS 指针。必需 bundle 从该固定 commit 的
  GitHub media URL 获取，并核对指针中的 SHA256 和字节数后替换。
  已验证的文件不重复获取；余下安装/数据下载见 `receipts/appworld-remaining-lfs.log`
  及同名 `.pid`，恢复时先查看进程和输出，再决定下一步。任务服务尚未启动。
- 数值诊断及未采用候选位于 `receipts/boundary-audit/`。残差/归一化原生
  舍入候选通过局部检查，但整网系数放大，未合入。当地正式两个 owner 文件
  已恢复 `4df76746`；没有放宽阈值或更改奖励公式。FP32 线性转置仅作为
  诊断比较，未接入训练；完整 DT 数值问题继续处理。
- 用户再次确认 token PPO 目标后，核对了当地实际 `DELTATRACE` 分支直接
  传递 `dt_token_advantages`，actor 使用 `vanilla` token PPO loss。
  新增“调用第二次 GAE 即失败、传入优势逐元素不变”的断言，仍复用原有
  collector→trainer→PPO 梯度测试；该检查 1 项通过，日志为
  `receipts/training-setup/no-second-gae.log`。这不替代尚未通过的整网 DT。

## 2026-09-22 MetaX 三任务资产与官方服务就绪

- AppWorld 官方 `install --repo` 和 `download data --root "$APPWORLD_ROOT"`
  已完成。当前 `data/datasets` 含 train/dev/test_normal/test_challenge；不再下载。
  原本地 worker 执行官方训练答案返回 `10, done=True`。随后直接使用已安装
  CLI 的 `serve environment --num-servers 20 --port -1 --no-show-usage
  --without-setup --root "$APPWORLD_ROOT"` 启动服务，未执行旧 VERL 脚本中的
  全局 kill 或新建 conda 环境。20 个 `/openapi.json` 均通过检查，原远程
  `AppWorldWorker.reset/step` 执行同一官方答案返回 `10, done=True`。
- 服务地址来自官方 CLI 输出 `receipts/appworld-services.log`；端口文件在
  `$APPWORLD_ROOT/appworld_ports.ports`，VERL 同名文件链接到它。服务进程使用
  现有 Python、CPU，不占 GPU；恢复时先检查这份日志中的地址是否仍存活，
  不重复启动。机器/服务重启后端口可能变化，历史端口不构成预留。
- WebShop 原 Google Drive 链接在 MetaX 不可达、本机返回 404。缺失的三个
  JSON 改从公开镜像取得：`HongbangYuan/webshop` 固定 commit
  `0129d4a81dbdb827e76afd20a1e2c38b61098613`，并与
  `sparklabutah/timewarp-env-data/webshop` 逐文件核对 SHA256 相同。
  来源、原 Drive ID、字节数和哈希保存在 `$WEBSHOP_ROOT/data/receipt.json`
  及 `results_eos_events.json`。这是数据镜像核验，不声称重新验证了不可达的
  原 Drive 文件。三个文件已经在 MetaX，不再下载。
- 已使用固定上游 `convert_product_file_format.py` 和官方
  `python -m pyserini.index.lucene` 构建 1k 索引，日志确认 1,000 documents、
  0 errors。`search_engine/indexes` 链接到 `indexes_1k`；原 worker 实际搜索、
  选项点击、购买的奖励为 `[0,0,0,0,0,10]`，最终 done=True。
- 现有 `en_core_web_sm` 已确认可用，没有重复安装。索引启动暴露缺失 `faiss`，
  只补装 `faiss-cpu==1.15.1`，使用已有 protected-constraints，未改变模型栈。
  新增安装报告 `receipts/training-setup/faiss-installed.json`。
- 奖励/服务日志：`receipts/training-setup/webshop-index-reward.log`、
  `appworld-official-reward.log`、`appworld-remote-reward.log/.json`。
  这些使用脚本或训练答案验证环境，不能当作模型任务成功率或 DT 训练结果。
- GRPO 在 MetaX 的独立运行检查保持 minibatch=4、group=4、总上限 32768、
  max_steps=15。首次 Ray 初始化因运行根目录过长超过 107-byte Unix socket
  限制失败；短 IPC 目录改为 `/tmp/dt-rl-mx-20260922`，已写入 env.sh。
  日志为 `receipts/training-setup/grpo-Sokoban-short-ray.log`，尚待训练结果。
- DT 控制检查：同一张空闲卡、同一对输入的三次原生类别 log-prob 完全相同；
  之后仍在第 2 个事件复现 `root=0.10085296630859375`、
  `signed_sum=0.04567181524601055`。`receipts/boundary-audit/repeat-native*`
  及 `controlled*` 保存结果。没有修改有限规则、信用公式或验收阈值。
- MetaX SDPA 在真实 actor 启动时提示没有编译 memory-efficient attention。
  对已安装 FA2 做了模型实际头布局（16 Q heads、4 KV heads、head_dim=256、
  BF16）的前向/反向核验：对 FP32 math SDPA 的前向相对 L2 为 0.001858，
  Q/K/V 梯度相对 L2 为 `[0.002546,0.003051,0.002844]`，全部有限。
  `receipts/training-setup/metax-fa-backward.log/.json` 保存结果。当地 env.sh
  因此默认 `VERL_ATTN_IMPLEMENTATION=flash_attention_2`，复用已安装且具备
  backward 的库；没有新增注意力实现。这不是 32k 训练通过的证据。
- 后续 least-change 局部舍入候选仍未通过整网检查，已撤回并恢复正式 owner。
  候选与诊断保留在 `receipts/boundary-audit/least-change*`；未调整验收阈值。
  `verify_reward_readout.py` 现在也会在整体检查失败时保存已有的单 token EOS
  数值对照；保持原失败状态，额外前向只用于诊断，不作为训练 credit fallback。

## 2026-09-22 原始 DT 数值诊断与训练接口分离

- 以实际部署的 `profiles/qwen35_gdn_symmetric.py` 和 dense runner 为准，
  原生舍入残差是正式接口的诊断输出；`0.02` 阻断此前由 RL 适配层加入。
  按 PLAN 对近似和接口的区分，保留同一阈值及 `conservation_verified=False`，
  原始 signed 向量仍按既定逐事件 expm1 公式接入 Q/V。数值 verifier 仍会
  保存全部结果并以失败退出；这不是修好或放宽了数值验收。
- 三任务当地官方正奖励轨迹已全部完成原生 DT/Q/V 组合。文件为
  `receipts/training-setup/native-{Sokoban,Webshop,AppWorld}-events.json/.log`。
  Sokoban 的 15 次 trace 合计约 30.79 秒：首次 20.43 秒，随后 0.66–0.92 秒；
  WebShop 6 次约 36.00 秒；AppWorld 1 次约 20.20 秒。首次时间含运行初始化
  和编译，不应当逐次相乘估计稳态卡时。实际上下文远短于 32768，上限未改。
- 三任务非零 token 优势数分别为 65、46、341；Q−V、mask 和有限值检查通过。
  原守恒审计分别失败 6/15、3/6、1/1；这些失败保留在结果中，不能作为精度
  或真实模型训练验收通过。相关 38 项 CPU 回归通过，日志
  `receipts/training-setup/audit-contract-tests.log`。
- 原 HF rollout 的 micro_batch_size=1 会串行生成 16 条 rollout；当地入口
  使用其已有参数 `ROLLOUT_MICRO_BATCH_SIZE=4`。GRPO 当前运行记录
  `receipts/training-setup/grpo-Sokoban-fa-b4.log`；此前 b1 运行已主动中止，
  不据其未完成状态声称吞吐比较或参数更新成功。
- Ray 使用现有 `+ray_kwargs.ray_init.num_cpus=8` 配置（环境变量
  `DT_RAY_NUM_CPUS`），避免按宿主全部 CPU 预启动大量 worker。
  用 `--cfg job` 验证了此 Hydra 参数；该固定 config 要求新增键的 `+` 语法。
- 同时启动多个任务会同时写固定上游补丁，实际复现了 shared-padding anchor
  检查失败。启动脚本现用系统 `/usr/bin/flock` 对
  `$VERL_ROOT/.deltatrace-patch.lock` 加锁，串行调用原补丁程序；单次串行复核
  通过，日志 `receipts/training-setup/serialized-patch-check.log`。没有复制
  补丁算法或重新安装库。并行任务使用各自的 Ray 短目录和已有数据目录。

## 2026-09-22 生成 token 配置修复与训练采样

- AppWorld、WebShop 模型生成提前结束后，上游 HF rollout 补齐 response 时
  复现 `NoneType * Tensor`。固定 VERL 的 FSDP worker 仅判断
  `GenerationConfig` 对象是否存在，没有判断其 `pad_token_id` 字段是否为空。
  原模型目录没有独立的 generation_config.json。已在该元数据拥有者处修复：
  EOS/PAD 未设置时使用原 tokenizer，明确配置的值（包括 0 和 EOS 列表）保留。
  8 项原上游元数据/补齐接口测试通过（17.80 秒），日志为
  `receipts/training-setup/generation-fix-tests.log`。没有重新安装模型或库。
- 10 秒 py-spy 生成采样中，主线程 874 个样本有 282 个包含 FSDP 栈。
  文件 `receipts/training-setup/grpo-generation-profile.json` 及 summary.json。
  这只是 CPU 栈采样，不能当作 GPU 总耗时比例或实测速比。
  当前通过上游已有 `fsdp_config.reshard_after_forward=False` 测量重复收集
  参数的开销；启动器提供 `FSDP_RESHARD_AFTER_FORWARD`，通用默认值仍为 True。
  显存/速度验收未完成，不提前把这个候选配置写为当地最优设置。
- 原进程使用了缺失 EOS/PAD 的旧元数据；已中止尚未完成的 Sokoban/GRPO
  运行，保存原日志。修复后的三任务运行记录为
  `receipts/training-setup/dt-{Sokoban,Webshop,AppWorld}-tokenids.log`。
  另有 `grpo-capacity-32k.log`，显式使用 31800 个 filler token、max_steps=1，
  只检查长输入资源容量；不是原始任务长度、完整任务训练或成功率实验。

# 未推广的对称 PV19 实验源包

NI0/MH2 的冻结实验 `dt_PV19_symmetric_whole_pilot_20260909_v1` 已运行，原 RISE 两例退步、NI0 MAS 退步、needle 不变，因此没有推广。

真正测试的 `qwen35_decoder_finite.py` 与 `qwen35_dense_finite_runner.py` 保存在同名冻结目录。测试时 SHA256 分别为 `0e673beaa64da667f8c0843fa69043cd0fd3210e4c3c908df6ee9ccf96db3dbc`、`8d3ebe943f852984f459125ad60fce275499669db2d38a9688ca56001741e955`。导出时路径脱敏的公开哈希另见 export_manifest。

当前 `research/runtime` 已只撤回本轮新增的可选 symmetric 分支，恢复到 `397ad56be41ce366cb49aa28f8491c25ff1f9e601541185bebb846828fdf3862`、`42c5407d3ea77802b4dc7d3a8404f42383eae4ae18b6a8c8444e973ddc18c901`。默认 C、官方模型、FA/FLA 与 FT 未被此候选替换。

复现否定结果应使用冻结目录的完整源码和 protocol，而非从当前 runtime 重新组装。归档 builder 是当时的来源记录，其读取当前 runtime 的步骤在恢复后不会重新产生相同冻结版本；不能把这一实验写为当前默认 API 支持。原始大激活和编译库按协议的远端位置及哈希核对，公开 Git 不包含这些二进制资产。

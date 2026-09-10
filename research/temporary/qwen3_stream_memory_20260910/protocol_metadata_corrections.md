# 冻结元数据说明

`memory_v3_confirmation_protocol.json`、`memory_v3_rollout_protocol.json` 和 `memory_v3_author_plan.json` 中，`candidate_finite_library.existing_build_provenance` 的文字误写了 `fa_owner_cached/build_summary.json`。实际构建回执为原始归档 `raw/evidence_fa_owner_cached.zip` 内的 `fa_owner_cached/results.json`，其中含完整编译命令、编译器版本、未改动厂商源码前后哈希和动态库哈希。`fa_owner_cached/study.py` 是实际执行的构建脚本，`fa_owner_cached/protocol.json` 是其冻结计划。

该文字字段不参与驱动执行。三个协议实际使用的动态库路径、动态库 SHA-256、有限内核源文件及其 SHA-256 均正确，并已逐字节核验。冻结协议保持原样；此处明确纠正来源说明，避免把一个不存在的文件当作构建证据。

早期 `postflight_verify.py` 是紧凑图阶段的结束检查，仍指向该阶段的环境目录。V3 的正式门槛失败，故预备的 `postflight_memory_v3.py`、作者兼容性和 rollout 没有执行，不能视作最终证据。V4 使用独立的 `postflight_memory_v4.py` 和 `postflight_memory_v4_plan.json`，执行前要求 V4 作者兼容性及全部 rollout 队列已结束。

V4 汇总脚本继承的 `previous_v1_transition_gate` 说明仍写“retained unchanged for v3”；其数值验收门槛也原样用于 V4。实际 V4 版本、协议、运行目录、驱动和源码哈希均独立固定。V4 对所有方法共同使用原 CPU 加载后原生移至 GPU 的路径；该差异在协议的 `common_loading_*` 字段与全部进程的实际加载记录中明确保存。

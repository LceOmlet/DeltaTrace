# 2026-10-01 部署转换原始证据

`observed.json`是只读采集记录，不是启动器，也不定义算法。
`artifacts[].path/sha256`对应远端原文件及其原始字节SHA；`data`保存解码后的JSON内容，
不是重新序列化后字节与原文件一致的声明。`logs`保留原日志摘录及原进程栈，
文件读取时间与日志采集时间分别记录。

- SQL：旧actor完成第1次原AdamW更新后，两rank在原worker RPC边界应用
  `1f862e8bbdaa…`；原完成回执保留此前的`2b80b938fee4…`及提交脚本SHA。
  后续原TaskRunner已回到采样调用。首轮耗时属于旧actor。
- AppWorld：旧PID2027456完成首轮和原step1检查点后停止；新PID2680224的
  source/job/launch记录冻结文件、提交时HEAD、旧PID和恢复路径。
  原日志确认global_step=1及后续生成，尚不代表新版本完整迭代已完成。
- 两任务首轮优化器观察均是只读原状态，不是额外的优化器更新或数值验收。

当前文件身份及PID创建时间另见`experiments/rl/current_runtime.json`；
版本解释统一见`experiments/rl/RUNTIME_RECORD.md`。历史部署状态不随新部署重写。

# 真实非零 FA 端点：原入口和逐行接口对照

沿用原App source58209daa、真实decoder27 operands ed804b5c、实际Python owner
3e1d6103和现有库4f42c391。4行保留各自全长K/V与原Q后缀，Q/K两端点均不同。
同一原算子分别用scalar query_start及row query_starts入口计算；两种B1表示与
已保存原B4的dq/dk/dv/tau/center全部逐值相同（40个张量比较），8次调用每次约0.33–0.34秒。
它排除了这一真实层/四行上的位置偏移和padding表示改变结果，没有新增精度容差，
不冒称FA官方提供非零有限归因误差标准。原重合端点FA断言是另一份回执。

没有加载模型或执行DT、采样、反向、优化器、恢复；未修改正式路径。
PID880541/birth1791391756.04完成退出；之后GPU4/5均859MiB。
Torch峰值allocated约0.654GiB、reserved约1.096GiB，进程PSS约7.13GiB；
未采集短调用的物理显存峰值，不能把allocator数字叫物理峰值。
TextCraft首次update继续hold，AppWorld仍终止，信用误差未修复。

owner-audit前两次CPU读取失败的stderr保留：首次未加入实际模块路径以解包Layout；
第二次远端无rg。v3加入原DT导入目录并使用局部文件枚举完成，CUDA未初始化。
未重装环境、重建缓存或生成替代操作数。源文件审计JSON保存实际源码和SHA；
没有将审计源码副本接进执行路径。真正GPU比较只运行check_nonzero_layouts.py，
它调用现有owner类/库，不实现attention、参考公式、loss或新的接受阈值。

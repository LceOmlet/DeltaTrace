# padding 候选：原框架比较通过，尚未部署

**未部署到三组正式训练。** 本目录只保存源码和证据，不是训练入口，不定义新容差。
最初的待定位记录保留；后续原框架比较结果见`owner-padding-assertion.json`及
[汇总回执](../../../../experiments/rl/results_actor_response_padding.json)。

- 上游：VERL-agent `20bd331bdbc9026a5668e11362178e10ab7400c8`。
- 基准：SQL v8冻结actor，SHA256 `2b80b938fee442ea5d9273523b6cce2bcc0511b7729aa6cea90333fef5de7cd3`。
- 候选：SHA256 `1f862e8bbdaad6fa116d0670772ad41269529a3a1e4a5b1eb383352d0372e9bd`。
- 仅修改`_forward_micro_batch`共有左padding裁剪和response列还原，原head/loss/update不改。
- `patch_actor_response_boundary.py`、`test_shared_padding.py`保存当时的完整字节；
  `candidate.patch`保存相对于`bf5f4ab64e8e4ad0d9ac5cd9be60c25ed51a62ac`的改动。
  默认工作区两文件已恢复该提交，归档不代表批准应用。

34项CPU接口测试通过，只证明索引/运输。真实Qwen同权重对照的13,312个有效
policy token最大log-prob绝对差17.750082、平均0.054087。
没有给整网差异设置新容差，也未把FA/FLA算子阈值挪来判断它。
随后复用固定VERL `test_hf_casual_models`的原masked-mean断言：均值差0.005007，
在原`atol=0.01, rtol=1e-5`内。它是原padding比较方法，不是逐token最大差阈值。

`layer-localization-v2`保存两卡真实逐层观察：取样的前512个有效token在前3层
完全相同，第一个完整attention层开始不同。`attention-replay`用原Qwen单层定位；
`value-projection`进一步证明相同输入与归一化输出进入原BF16线性投影时，长度变化
即可产生与回放V逐值一致的小差异，两布局相对FP32参考的误差量级相同。
这些是定位证据，没有改变正式精度、算子、PPO/DT公式或阈值。
单层回放仅保留前512个有效位置、其余hidden置零；它不冒充原整层输入的逐值复现。

同一padding夹具热调用为基准13.973732秒、候选2.060242秒；该计时不能推出
正式任务提速。候选的一次原VERL更新和更新后vLLM同步已完成，不抵消上述未解释差异。
夹具虽然载体宽度32768，但实际长度为512/1024/2048/4096，不代替既有满长B8容量结果。

`candidate.json`是CPU阶段的原始记录；`source.json`固定真实诊断的源码身份；
`result.json`、`difference-localization.json`和`physical-resources.jsonl`是原诊断材料；
`status.json`关联早期材料与后续状态。真实测试输入和逐token输出保留在远端
`receipts/owner-b8-dispatch-20260930/actor-response-padding/real-model/same-input-readouts.pt`。
原诊断driver、worker、资源观察进程已经退出，正式训练未修改。

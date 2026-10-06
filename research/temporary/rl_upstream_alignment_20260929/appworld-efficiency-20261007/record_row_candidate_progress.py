"""Record isolated source/test identities without changing the method or launch."""
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
ROW = HERE/'individual-prefix-owner-candidate-v1'
REP = ROW/'native-representation-candidate'


def ref(path):
    path = Path(path)
    return dict(path=str(path.relative_to(REPO)),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    result_path=REPO/'experiments/rl/results_appworld_efficiency_20261007.json'
    result=json.loads(result_path.read_bytes())
    stage=json.loads((REP/'real-b8-v3-stage-receipt.json').read_bytes())
    job=json.loads((REP/'real-b8-v3-launch.stdout.json').read_bytes())
    assert job['prepared_sha256']==stage['prepared_sha256']
    assert job['pid']==2138789 and job['pid_birth']==1791323526.94
    assert not job['checkpoint_restore'] and not job['optimizer_step'] and not job['profiler']
    result['individual_row_candidate_progress']=dict(
        status='isolated_real_B8_running_not_deployed_or_numerically_accepted',job=job,
        actual_sources=stage['individual_row_candidate'],
        kernel_check=ref(ROW/'kernel-verification.json'),
        CPU_interfaces=ref(REP/'cpu-validation-index.json'),
        source_review=ref(REP/'source-review-readonly.json'),
        geometry=ref(ROW/'physical-padding-geometry.json'),
        prepare=ref(REP/'real-b8-v3-stage-receipt.json'),
        phase=ref(REP/'real-b8-v3-phase-1791323617.json'),
        failed_v1=ref(REP/'real-b8-v1-exit-readonly.json'),
        prepared_only_v2=ref(REP/'real-b8-v2-stage-receipt.json'),
        current_formal_unchanged=True,
        scope='Original actor setup, actual saved B8 and original Q/V/A/PPO; first four calls isolate cold/warm cost. Fifth call passively records actual native FA/FLA and finite operands for official tests after release. No checkpoint, optimizer step, new numerical threshold or deployment.')
    result['fresh_v3_latest_phase']=ref(HERE/'native-conv-canonical-owner-v3/phase-readonly-latest-1791323172.json')
    result_path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    ledger_path=REPO/'experiments/rl/RUNTIME_RECORD.md'
    ledger=ledger_path.read_bytes().decode('utf8')
    title='## 2026-10-07 05:53 逐行前缀候选进入真实B8；正式owner与整批白化保持'
    text='''
## 2026-10-07 05:53 逐行前缀候选进入真实B8；正式owner与整批白化保持

正式v3仍driver1953903/birth1791321793.72、GPU4/5、source97e3cb75，原冻结
3cd90b2d/preparedf5232ac1的源字节不变。整批官方masked_whiten已在旧v2实际执行，
本次fresh继续相同trainer；不能把尺度修复生效称为学习质量恢复。不载入旧训练
检查点。05:46只读LOOP采样完成207/240，运输3695回复/737768 tokens/1168.0s；
尚无本次完整gen/update_actor时间比，旧actor2495秒也不能作为当前版本实测。

逐行切点隔离候选实际176请求/44个B4 CPU接口检查通过：target label/sample/
paired顺序、suffix位置、原action绝对索引与原NativeTargetLogitRows保持。小CPU
Cache接口单独标注，没用它代替真实B8/数值验收。source-review f6b97835完整绑定
runner5f14bb3c、answerd47333ea、artifact50af8daf、lease54ad5ae1；new finite wrapper
3e1d6103/library4f42c391、kernel CUDA9ebcef18只扩展逐行query starts表示，公式不变。
原FA真实saved operands的官方coincident检查与非零layout逐值对照见05:13记录；
这些不能冒称真实异质B8/32k/完整信用已验收。

第一次隔离probe2138789之前的v1/2105248已在DT bank前因导入路径断言退出：
原producer212再次prepend DT_ROOT/clean/qwen35，遮盖out的answer接口。原错误
日志保留；root随后停止请求发现PID已退出，实际上没发signal。v2仅prepared，
不覆盖其源或把它称为执行。v3复用原已存在的链接owner目录接口，只有目标/
artifact指向惰性扩展，bare runner仍原e9c757，原wrapper不遮盖；未改正式目录。

隔离B8 v3 driver2138789/birth1791323526.94、GPU2/3，prepared e30a1f17，原verify
22bc698c字节不变，原88请求/rank的SHA8714abf2/1fdc1eb0不变。diagnose cdfe43a5
只改诊断分发与读取表示：前4次baseline/row冷暖分别测量，第5次组合原GDN0、
varlen FA3与真实finite输入/LSE/upstream被动观察，不算速度。05:53实际两rank
inspect/SHA确认answer/artifact/lease及6个DI身份，进入shared_cold，尚无完成结果。
此时owned子树PSS36.98GiB、容器168.89GiB；GPU2/3约17.9/17.7GiB，资源只作
该时刻观测，不称峰值或容量验收。没有挂live tracer、优化器step或新容差。

新FA varlen离线接口37c2c9fb复用原a290e11c reference/原断言：原packed每行完整
Q/K取出后再还原整批比较，不能把KV carrier中间洞直接喂给原reference。GDN0
仍用原5c1924ce/check_saved_fla与原FLA断言。原fa/LSE/finite非零输出分别报告，
不把native forward断言扩大为完整DT验收。候选仍不在默认launch路径。
收据位于appworld-efficiency-20261007/individual-prefix-owner-candidate-v1/，
结果索引experiments/rl/results_appworld_efficiency_20261007.json；正式事实继续以
v3 job/source/原worker日志为准。PLAN、PPO损失、任务预算、LoRA8/16、每卡B4未改。

'''
    if title not in ledger:
        end=ledger.index('\n')+1
        ledger_path.write_bytes((ledger[:end]+text+ledger[end:]).encode('utf8'))
    print(json.dumps(dict(result=ref(result_path),formal_source='97e3cb75',candidate_pid=job['pid'])))


if __name__=='__main__':
    main()

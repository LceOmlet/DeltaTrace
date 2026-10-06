"""Record saved kernel evidence; no remote, runtime or method mutation."""
import hashlib
import json
from pathlib import Path
import re

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[4]


def ref(path):
    return dict(path=path.relative_to(REPO).as_posix(),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    execution=json.loads((HERE/'real-operands-execution.json').read_text(encoding='utf8'))
    checks=json.loads((HERE/'original-official-saved-operand-check.json').read_text(encoding='utf8'))
    finite=json.loads((HERE/'real-layout-result.json').read_text(encoding='utf8'))
    physical=[]
    for sample in execution['physical_samples']:
        lines=sample['physical_mx_smi'].splitlines()
        for i,line in enumerate(lines):
            if re.match(r'\|\s*2\s+MetaX',line):
                physical.append(int(re.search(r'(\d+)/(\d+) MiB',lines[i+1])[1]))
    report=dict(status='kernel_layout_verified_with_stated_scope_not_deployed',
        original_assertions=checks['assertions'],actual_checks=checks['checks'],
        zero_endpoint_scope='Same real saved input; original FA FP32 reference/reordered BF16 baseline and unchanged assertions. Output factor2; dq/dk/dv factor3. Only the coincident-endpoint ordinary-gradient limit is covered.',
        nonzero_finite=finite['same_actual_complete_range'],
        nonzero_scope='Actual B4 eight operands, uniform row cut384/Q stride527, original coefficient range433 onward: new scalar and new row ABI both exactly equal to deployed old scalar owner in all five outputs. No new finite-error threshold.',
        mixed_transport=finite['mixed_index_transport'],
        missing='Actual heterogeneous factual cuts, native cache/mask/position target/lease/runner integration, FLA check, 32k B8 update capacity, complete DT runtime speed or learning recovery.',
        measured_resources=dict(seconds=execution['finished_unix']-execution['started_unix'],
            sampled_peak_tree_PSS_bytes=execution['sampled_peak_tree_PSS_bytes'],
            sampled_physical_GPU2_peak_MiB=max(physical),
            torch=finite['resource']),
        production_changed=False,model_or_checkpoint_loaded=False,
        sources={name:ref(HERE/name) for name in ['compiled-owner.json','official-owner-test-audit-readonly.json',
            'saved-real-finite-fa-schema-readonly.json','original-official-saved-operand-check.json',
            'real-layout-result.json','real-operands-execution.json',
            'compile_isolated_owner.py','verify_row_layout_real_operands.py','run_isolated_real_operand_check.py']})
    path=HERE/'kernel-verification.json'
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    result_path=REPO/'experiments/rl/results_appworld_efficiency_20261007.json'
    result=json.loads(result_path.read_text(encoding='utf8'))
    result['individual_prefix_candidate']['status']='finite_FA_kernel_layout_verified_native_B4_integration_unaccepted_not_deployed'
    result['individual_prefix_candidate']['checks']='Isolated compiler succeeds; actual B4 uniform-cut five finite outputs exact against old scalar owner; same-endpoint real ordinary-gradients pass original FA output2/gradient3 assertions. Mixed cuts are a shortened-range supplementary transport only. No FLA or native-model integration acceptance.'
    result['individual_prefix_candidate']['evidence']['kernel_verification']=ref(path)
    result_path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    ledger_path=REPO/'experiments/rl/RUNTIME_RECORD.md'
    ledger=ledger_path.read_bytes().decode('utf8')
    title='## 2026-10-07 05:13 逐行切点finite-FA接口通过实际操作数检查，未接入训练'
    section='''
## 2026-10-07 05:13 逐行切点finite-FA接口通过实际操作数检查，未接入训练

正式driver1181392/worker仍已退出；本轮没有重启、检查点加载/导出/恢复或参数修改。
隔离candidate=/candidates/appworld-row-cuts-finite-20261007-v1，CUDA9ebcef18、
wrapper3e1d6103，复用既有prepare_remote.py699b3f1b原cucc flags/头文件，49.32秒
编译成功，library4f42c391055afec0a0fee9ee698c0820163ff413e42f1c4909b2961ce81e5157。
编译实际子树PSS采样峰1.017GB，无GPU launch；初次诊断模板替换误中STDOUT、
第二次调用缺失/usr/bin/time，均在compiler启动前失败，已保存stderr并改用已装
psutil PSS采样，不安装工具、不清缓存、不替换正式owner。没有称失败为通过。

实际GPU2原saved-operand检查15.46秒退出0，reuse actual-dt-layer3-boundaries.pt
SHA0ade21d748c0fa37bd46a08ee3c1452d6cb298dac36b0fed6ad7c7be4b9978aa。
单decoder3真实八操作数：Q BF16[4,16,527,256]、K/V BF16[4,4,911,256]、u FP32
同Q、LSE FP32[4,16,527]，原cut384/coefficient start433。不是模型权重或checkpoint。
原verify_saved_fa_dtypes.py7ff11d9d字节未改，只注入隔离library和row-layout表示，
使用原FA2.6.3 source a290e11c的attention_ref/普通低精度参照和原断言：output2倍、
dq/dk/dv3倍。原native四项及coincident finite三项全部通过；这个官方误差断言
只用于相同端点的普通导数极限，绝不作为非零finite的新误差门槛。

非零真实操作数仍保留原完整合法范围433以后：新scalar与新row ABI（四行cut384、
Q stride527）对原5d2af760库的dq/dk/dv/tau/center均逐值相同，max_abs0。
混合cut384/448/512/576只补充核查索引，保留范围同步缩至433/448/512/576后，
五输出也相同。此范围不是原四行训练action范围，不能当原完整credit保持或
真实异质cut/runner验证。尚无真实异质prefix-cache/mask/position/target/lease接入、
新FLA/32k B8/full DT速度验收；候选仍未部署，原冻结3cd90b2d不替换。

单GPU检查采样物理峰2556MiB、子树PSS10.015GB，Torch峰allocated1.136GB/
reserved1.439GB；不同资源口径分开。没有模型forward/训练/额外rollout/optimizer。
实际GPU计算由原FA reference和finite owner完成，未使用随机u补造异质训练样本。
原源、编译、实际dtype、原断言、所有残差与资源由kernel-verification.json索引。

排序owner只读复查也已完成：原consumer按context稳定排序，原DataProto.reorder/
等行数跨卡分区已接；动态microbatch不能直接保B4。已有88行按切点排序没有大
收益（约1.58% suffix槽减少且rank1反增2.91%），不新增影子调度器。逐行cut后
FA phase0/1仍必须读完整事实KV，不能把必要score/LSE计算当可删历史重算。

'''
    if title not in ledger:
        end=ledger.index('\n')+1
        ledger_path.write_bytes((ledger[:end]+section+ledger[end:]).encode('utf8'))
    print(json.dumps(dict(kernel_verification=ref(path),result=ref(result_path)),ensure_ascii=False))


if __name__=='__main__':
    main()

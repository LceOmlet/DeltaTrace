"""Record verified preparation or actual deployment without changing the method."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
RECEIPTS=HERE/'receipts'
ROOT='/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922'
VERSION='textcraft-dt-reward-target-exponential-20261016-v5'
AUDIT=ROOT+'/receipts/textcraft-reward-target-exponential-20261016-v5'
RUN=ROOT+'/runs/'+VERSION
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def identity(path):return dict(path=str(path),SHA256=sha(path),bytes=Path(path).stat().st_size)

checks=read(RECEIPTS/'interface-verification.json')
assert all(checks['checks'].values()) and checks['unique_passing_behavior_tests']==22
launch=read(RECEIPTS/'prepared-launch.json')
report=dict(version=VERSION,recorded_unix=time.time(),status='verified_prepared_not_submitted',
    method_specification='experiments/rl/PLAN.md only',
    upstream_commit=launch['upstream_commit'],local_patch_commit=sys.argv[1] if len(sys.argv)>1 else None,
    remote_run=RUN,remote_audit=AUDIT,
    semantics=dict(credit='Official fixed VERL GRPO group multiplier for native TextCraft. Explicit multi-target events accept their already determined signed stable b_k on the same scale.',
        target='Native reward-settlement action payload, score 0 included. Multiple payload pieces of one settlement form one offset union.',
        multi_target='Original per-event EOS DT vector d_ik; Q=sum b_k, V=sum b_k*exp(-d_ik), A=sum b_k*-expm1(-d_ik). Exponent before aggregation.',
        normalization='No b/DeltaF seed, d normalization, source u normalization, log(signed credit), or repetition of rollout g per event.',
        zero_total_credit='Nonzero cancelling components are retained, including B=0.',
        masks='Original explicit per-component future mask; observation/padding excluded, earlier target may be source for a later event.',
        self_target='Per event Q=b_k,V=0,A=b_k replaces that event term, then other future event terms may still contribute.',
        dispatch='Flatten events on driver, original DataProto pad/sequence balance/DP RPC, exact native retained-slot scatter. Each FSDP rank has equal B4 call count.',
        advantage_preprocessing=launch['variance_standardization'],PPO_reduction=launch['PPO_reduction']),
    efficiency=launch['efficiency'],effective_options=launch['options'],budget=launch['budget'],
    actual_source_bindings=launch['source_bindings'],verification=checks,
    native_multi_target=[read(RECEIPTS/f'native-multi-target-rank{rank}.json') for rank in (0,1)],
    native_32768_capacity=[read(RECEIPTS/f'capacity-multi-32768-rank{rank}.json') for rank in (0,1)],
    multi_target_driver_check=read(RECEIPTS/'native-multi-target-driver.json'),
    capacity_driver_check=read(RECEIPTS/'capacity-multi-32768-driver.json'),
    limits=dict(TextCraft='Native production has one terminal settlement; generic two-target interface is tested with real original parser payloads and the user-specified (-.1,1) credits, not fabricated environment rewards.',
        capacity='Artificial exact-32768 inputs, B4/rank and dual B8. No natural-task quality or whole-training speed claim.',
        equivalence='Accepted exact-counterfactual and same-relative-credit idealization; these behavior checks do not establish identical clipped finite-batch updates.',
        gradient_rootcause='Old goal remains paused; no long-term gradient repair claim.'),
    optimizer_updates_in_verification=0,generated_rollouts_in_GPU_verification=0,quality_evaluations=0,
    prior_failed_check_scope='Initial CPU lazy-import assumption, stale prepared-source test, Ray harness import alias, and metadata lookup after successful small DT; original receipts retained. No package/cache/config repair or formal training changes.',
    exact_receipts=[identity(path) for path in sorted(RECEIPTS.iterdir()) if path.is_file()])
path=REPO/'experiments/rl/results_textcraft_reward_target_exponential_20261016_v5.json'
if (RECEIPTS/'deployment-actual-observation.json').is_file():
    actual=read(RECEIPTS/'deployment-actual-observation.json');job=read(RECEIPTS/'DTPO-v5-job.json')
    initial=read(RECEIPTS/'initial-state-and-existing-passive-records-v5.json')
    assert initial['complete'] and initial['initial_optimizer0_scheduler0_verified']
    assert job['version']==VERSION and job['local_patch_commit']==report['local_patch_commit']
    report.update(status=job['status'],job=job,actual_observation=actual,initial_workers=initial,
        user_authorized_fresh_training=True,accepted_for_training=True,fresh_base_model=True,resume_checkpoint=None)
    write(path,report)
    runtime_path=REPO/'experiments/rl/current_runtime.json';runtime=read(runtime_path)
    current=dict(version=VERSION,status=job['status'],accepted_for_training=True,
        local_patch_commit=job['local_patch_commit'],upstream_commit=job['upstream_commit'],
        pid=job['pid'],birth=job['birth'],physical_gpus=job['physical_gpus'],
        completed_formal_iterations=actual['completed_formal_iterations'],workers=actual['workers'],
        verification=identity(RECEIPTS/'interface-verification.json'),
        completed_runtime_override=actual['completed_initial_runtime_override'],
        actual_source_bindings=launch['source_bindings'],PPO_identity_preserved=actual['PPO_identity_preserved'],
        receipt=identity(path),old_gradient_goal='paused',rootcause_repair_verified=False)
    runtime={'20261016_DTPO_v5_verified_deployed':current,**runtime}
    runtime['latest_textcraft_method_deployment']=current;write(runtime_path,runtime)
    stamp=dt.datetime.fromtimestamp(actual['observed_unix'],dt.timezone(dt.timedelta(hours=8))).isoformat()
    heading=f'## {stamp} DTPO v5实际部署：逐target指数信用、原GRPO与高效原生DT/PPO'
    body=f'''正式v5版本{VERSION}，源提交{job['local_patch_commit']}，固定VERL上游{job['upstream_commit']}。原官方组内信用g保留；原reward结算动作target含结算0，较早非结算Action移为source。多target显式事件在driver展平、复用原pad/balance/DP/RPC，原d逐事件指数后汇总Q/V/A，不用v4仿射seed/端点差归一化或source u分母。B=0的抵消分量仍保留；one-event多文本片段只算一次信用。原观察/self/future mask、原整批masked_var仅方差缩放和1/16 sum(S_j/N_j) PPO保持。

22项CPU检查通过；真正双rank DT的两个原parser payload逐事件读取及指数后汇总逐值检查通过，原参数/optimizer/scheduler保持。每rank B4、双B8、精确32768多target容量通过，0 optimizer/生成/质量测试；人工容量输入不称自然任务质量或整轮吞吐。首轮native small的数学/mask/状态已通过，后续metadata lazy模块查询失败原始记录保留，并复用已完成计算，未重做该DT。其他早期fixture/导入错误及修正回执亦保留。

实际继承验收数值26bef6c8、GDN3f51f5f7、gate CPU捕获/有界回传、原prefix cache、默认vLLM图与冻结head更新补丁7a33e61a（bb46ebe9），不启用head-selection或分母候选。原actor3a65e173/corefc2f992b/worker9dfd6ce7/checkpoint f65b6538及32k/LoRA8/16/B4/global64/330/save5/keep2保持。全部完整路径与SHA、配置、行为/容量/实际绑定回执见results_textcraft_reward_target_exponential_20261016_v5.json。

新run从基础权重启动，driver{job['pid']}/出生{job['birth']}、physical0/1；实际worker及phase为{json.dumps(actual['workers'],ensure_ascii=False)}。原RPC已核对初始optimizer0/scheduler0与实际v5模式、路径，并安装数学AST不变被动记录器。当前原日志完成{actual['completed_formal_iterations']}/330；初始化与首轮采样不称已完成健康更新。现有原PPO {actual['PPO_identity_preserved']['pid']}/{actual['PPO_identity_preserved']['birth']}保持，GRPO已停191轮，v3已停45轮，v4撤回0轮，不拼接旧DT。旧梯度goal仍paused；小时绘图继续依最新v5正式身份刷新，不恢复旧硬件/双备份任务。
'''
    for name in ('RUNTIME_RECORD.md','REMOTE_ENVIRONMENT.md'):
        doc=REPO/'experiments/rl'/name;old=doc.read_text(encoding='utf-8')
        assert heading not in old
        doc.write_text(heading+'\n\n'+body+'\n'+old,encoding='utf-8')
else:write(path,report)
print(json.dumps(dict(result=str(path),status=report['status'],version=VERSION),ensure_ascii=False))

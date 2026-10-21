"""Reuse the existing receipt reader; record v10 preparation and deployment separately."""
import ast
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
sys.path.insert(0,str(HERE.parent/'rl_textcraft_v5_credit_integration_20261020_v1'))
from transport import run_python

owner=HERE.parent/'rl_textcraft_v5_two_gpu_reproduction_20261021_v1/record_state.py'
assert hashlib.sha256(owner.read_bytes()).hexdigest()=='65b15dc2318d8053be50f899147bbbdd3e317962667fd577d63c6eae90d239e0'
assignment=next(n for n in ast.parse(owner.read_text(encoding='utf-8')).body
                if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='remote' for t in n.targets))
remote=ast.literal_eval(assignment.value)
remote=remote.replace('textcraft-v5-two-gpu-reproduction-20261021-v1','textcraft-v10-target-logprob-20261021-v1')
remote=remote.replace('textcraft-dtpo-v5-original-2gpu-formal-20261021-v1','textcraft-dtpo-v10-target-logprob-2gpu-formal-20261021-v2')
remote=remote.replace("('campaign-prepared.json',", "('CPU-behavior-verification.json','native-behavior-job.json',"
    "'native-behavior-status.json','native-verification-summary.json','native-source-shadowing-resolution.json',"
    "'native-verifier-dtype-resolution.json','formal-CLI-serialization-resolution.json','configuration-differences.json','campaign-prepared.json',")
observed=json.loads(run_python(remote,timeout=55))
(HERE/'receipts').mkdir(exist_ok=True)
(HERE/'receipts/latest-deployment-observation.json').write_text(json.dumps(observed,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
files=observed['files']
row=files['dt-prepared.json']['data']
state=files.get('campaign-state.json',{}).get('data',{})
job=next((j for j in state.get('jobs',[]) if j['version']==row['version']),{})
native=files['native-behavior-status.json']['data']
record=dict(version=row['version'],status=state.get('status','prepared_only'),recorded_unix=observed['collected'],
    task='TextCraft',method='DTPO v10 target token log-prob exponential',physical_gpus=[2,3],budget=row['budget'],
    source_credit='original v5 unscaled DT and original official g',
    target_credit='d_t=target_logp1[t]-target_logp0[t]; u_t=-expm1(-d_t)',
    target_reference='original v5 prior-source EOS; fixed target labels and factual target prefix',
    self_target_constant_override=False,additional_model_forwards=0,
    effective_configuration=row['options'],source_bindings=row['source_bindings'],
    method_source_commit=row.get('method_source_commit'),training_source_commit=row.get('execution_source_commit'),
    adapter_source_commit=row.get('adapter_source_commit'),
    upstream_commit='20bd331bdbc9026a5668e11362178e10ab7400c8',
    remote_run=row['run'],remote_audit=row['run'].rsplit('/runs/',1)[0]+'/receipts/textcraft-v10-target-logprob-20261021-v1',
    shared_checkpoints=row['checkpoints'],process_checks=observed['process_checks'],
    completed_iterations=job.get('completed_iterations',0),phase=job.get('phase',state.get('status','prepared_only')),
    controller=files.get('controller-job.json',{}).get('data'),
    driver={k:job[k] for k in ('pid','birth') if k in job},
    actual_runtime_overrides={k:v for k,v in files.items() if k.startswith('runtime-overrides/')},
    CPU_checks=files['CPU-behavior-verification.json'],native_integration=files.get('native-verification-summary.json'),
    failed_attempt_receipts={k:v for k,v in files.items() if k in ('native-source-shadowing-resolution.json','native-verifier-dtype-resolution.json','formal-CLI-serialization-resolution.json')},
    native_verification_status=native['status'],capacity_or_quality_claim=False,
    existing_v5_formal_run_continues=True,
    hourly_plot_entry=str(HERE/'plot_hourly.py'),
    hourly_plot_entry_SHA256=hashlib.sha256((HERE/'plot_hourly.py').read_bytes()).hexdigest(),
    orchestration_source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')},
    full_observation=str(HERE/'receipts/latest-deployment-observation.json'))
path=REPO/'experiments/rl/current_runtime.json'
runtime=json.loads(path.read_text(encoding='utf-8'))
runtime['20261021_TextCraft_v10_target_logprob_two_GPU_formal_current']=record
path.write_text(json.dumps(runtime,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(REPO/'experiments/rl/results_textcraft_v10_target_logprob_two_gpu_20261021_v1.json').write_text(
    json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
stamp=dt.datetime.fromtimestamp(observed['collected'],dt.timezone(dt.timedelta(hours=8))).isoformat()
paragraph=f'''## {stamp} TextCraft v10：复用原逐 token endpoint，独立两卡正式训练

用户批准 target 使用同一原事实/EOS参考上下文中逐 token log-prob 差，再用 u=-expm1(-d)。只在原冻结 v5 readout 中接入其同次 DT 已有 target_logp1/0，并取消 self-target 常数覆盖；原 source DT、官方 g、Q/V/A primitive、乘g后整批优势仅方差缩放和 VERL PPO 保持，不增加每批模型前向。生成候选仅改三个 entry owner，实际路径与SHA见 current_runtime 同日期key及 results_textcraft_v10_target_logprob_two_gpu_20261021_v1.json。方法/适配提交{row.get('method_source_commit')}；原训练执行源码仍4545c558，VERL20bd331b，不能把新提交称整套DT/VERL换源。

CPU6项及实际DP2各B4的接入已通过；每rank12个真实 target label按原因果predictor位置独立核对逐token差和原Q/V/A，模型/optimizer校验值未变。该验证不代表容量或质量结论。首次源码存档同名遮蔽导入、第二次独立验证器float64写float32失败均完整保留；修正的是存档名/验证器，候选方法与DT内核未改。失败回执及完整原生张量见远端audit，不能抹去失败或将其写成正式训练失败。

本次run={row['version']}，实际状态{record['status']}，完整迭代{record['completed_iterations']}/330，phase={record['phase']}。controller={record['controller']}; driver={record['driver']}。物理GPU2/3，独立基础权重/0开始，原30epoch/total_training_steps=null、32组×8/global64/B4/epochs1/每轮4更新、32k/LoRA8/16/采样/LR/save5/keep2保持。原v5在0/1继续330预算；新v10不拼接。共享检查点={row['checkpoints']}。原f71ff4d8 process owner负责原生训练、失败保留与最终330完整检查点，使用本audit独立manifest防止覆盖仍在运行的v5身份。初始化不称完整迭代；正式worker尚未实际导入的懒加载模块保持待核验。

小时绘图复用原c20559e3 parser、已有两卡/八卡适配及f6ed1ff3 renderer，加入独立v10曲线，路径F:/DeltaTraceBackups/plots/textcraft-v10-target-logprob-two-gpu-vs-v5-original-reproduction-and-eight-20261021-v1。三条DP2分片梯度与历史DP8全局梯度分轴且不换算。原v5第10轮观察、两卡/八卡同状态输入对拍继续保留；已在keep2轮换前另存完整原生checkpoint5及自然第6轮原DataProto输入输出，不将准备文件称数值对拍通过。

'''
path=REPO/'experiments/rl/RUNTIME_RECORD.md'
path.write_text(paragraph+path.read_text(encoding='utf-8'),encoding='utf-8')
print(json.dumps({k:record[k] for k in ['version','status','completed_iterations','phase','driver','native_verification_status']},ensure_ascii=False))

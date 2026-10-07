"""Record the original native factual Jacobian diagnosis, without changing credit."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[4]
def binding(path):
    data=path.read_bytes()
    return dict(path=path.as_posix(),bytes=len(data),sha256=hashlib.sha256(data).hexdigest())

folder=HERE/'native-factual-jacobian-results'
for item in json.loads((folder/'transport.json').read_bytes()):
    assert binding(Path(item['local_path']))['sha256']==item['sha256']
measured=json.loads((folder/'results/result.json').read_bytes())
launch=json.loads((folder/'launch.json').read_bytes())
observed_path=sorted(HERE.glob('native-factual-jacobian-observation-*.json'))[-1]
observed=json.loads(observed_path.read_bytes())
assert measured['phase']=='complete' and len(measured['groups'])==4
assert launch['pid']==measured['pid'] and launch['birth']==measured['birth']
assert not observed['alive_same_birth']
assert observed['textcraft_same_birth'] and not any(observed['textcraft_releases'])
for group in measured['groups']:
    assert group['initial_state_pair_equal'] and all(group['factual_operands_equal'].values())
    assert group['gradients_all_finite'] and not any(group['reference_gradient_maxabs'].values())
previous_path=REPO/'experiments/rl/results_current_extreme_native_fla_20261008.json'
previous=json.loads(previous_path.read_bytes())['measured']
summary=dict(measured['summary'],
    previous_original_finite_on_same_single_endpoints=previous['original_finite_on_actual_single_endpoints'],
    previous_joint_coefficients_times_same_single_delta=previous['joint_coefficients_times_actual_single_delta'])
summary['previous_single_finite_minus_native']=summary['previous_original_finite_on_same_single_endpoints']-summary['native_single_deletion_output_effect']
sources=json.loads((HERE/'joint-finite-sources.json').read_bytes())
result=dict(status='Factual native Jacobian diagnostic complete; no credit repair accepted',
    launch=launch,scope=measured['scope'],summary=summary,groups=measured['groups'],
    original_native_owner=measured['owner'],verified_native_source_sha256=measured['verified_native_source_sha256'],
    original_imported_files=[dict(module=r['module'],path=r['path'],sha256=r['sha256']) for r in sources['files']],
    raw_result=binding(folder/'results/result.json'),transport=binding(folder/'transport.json'),
    terminal_observation=binding(observed_path),previous_native_receipt=binding(previous_path),
    interpretation='The unchanged native factual gradient is a local linearization, not the finite single-deletion effect. It also differs materially from this actual finite perturbation. The original finite callback on the same single endpoints is close, whereas reusing joint-reference coefficients gives a different local allocation. Replacing DT with ordinary gradients is not supported by this evidence.',
    operations=dict(original_operator_forwards=4,original_operator_backwards=4,model_load=0,full_DT=0,rollout=0,optimizer=0,checkpoint_restore=0),
    resources=dict(max_torch_live_allocated_peak_bytes=max(g['peak_live_allocated_bytes'] for g in measured['groups']),
        max_recorded_pss_bytes=max(g['pss_bytes'] for g in measured['groups']),
        continuous_physical_peak_measured=False),
    limitations=['Local contractions use the saved original output cotangent; they are not complete token rewards or advantages.',
        'The actual FP16 q/k/v/beta and FP32 raw_g are retained. No FA/FLA official tolerance assertion is added or changed.',
        'These selected four head groups are a mechanism diagnosis, not a population attribution-quality measurement.',
        'The independently verified consumed-FA-cache lifetime repair remains separate and is not formally deployed.'],
    credit_repaired=False,production_profile_changed=False,formal_restart=False,TextCraft_update_released=False,
    official_tolerance_claim=False,
    unchanged=dict(QVA=True,PPO=True,target=True,reward=True,whitening=True,lora_rank=8,lora_alpha=16,microbatch_per_gpu=4,DT_minibatch_per_gpu=4))
target=REPO/'experiments/rl/results_native_factual_jacobian_20261008.json'
target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
runtime=REPO/'experiments/rl/current_runtime.json'
raw=runtime.read_bytes()
key='latest_native_factual_jacobian_diagnosis'
assert key not in json.loads(raw) and raw.endswith(b'}\r\n')
value=dict(receipt=binding(target),launch=launch,summary=summary,imported_files=result['original_imported_files'],
    native_owner=measured['owner'],native_sources=measured['verified_native_source_sha256'],
    terminal_observation=binding(observed_path),credit_repaired=False,production_profile_changed=False,
    TextCraft_update_released=False,AppWorld='terminal_not_restarted',unchanged=result['unchanged'])
tail='\r\n'.join(json.dumps({key:value},ensure_ascii=False,indent=2).splitlines()[1:-1]).encode('utf8')
runtime.write_bytes(raw[:-3].rstrip(b'\r\n')+b',\r\n'+tail+b'\r\n}\r\n')
json.loads(runtime.read_bytes())
ledger=REPO/'experiments/rl/RUNTIME_RECORD.md'
heading='## 2026-10-08 原生事实梯度诊断结束：普通梯度不等于此有限删除效应'
note='''
3c9c7081/PID2889955/birth1791410961.25已完成退出。只对保存的实际GDN30
单删除操作数调用原FLA，4个八head分组各一次forward/backward，未加载模型或更新。
事实输入逐值相同，incoming state固定原值，FP16 q/k/v/beta及FP32 raw_g保留；
原cotangent只在事实行，参考行梯度均零，所有梯度有限。
原生有限输出效应+15.827504；原生事实梯度乘真实差+20.063134，差+4.235630。
之前同一single端点的原finite为+15.825014；联合参考系数乘该single差+9.671126。
这些是固定原输出cotangent的局部分量，不是完整advantage或官方容差通过。
不能据此改用普通梯度，也不能局部替换或加倍率强行修正。信用仍未修复。
Torch live峰值0.948GiB、PSS约6.27GB；未连续采样物理峰值，首group含编译。
Text原birth仍hold、release均不存在；App正式terminal，无恢复/正式重启。
实际源码SHA、启动commit/PID出生、原算子和回执写入current_runtime末字段。
'''
raw=ledger.read_bytes()
assert heading.encode('utf8') not in raw
end=raw.index(b'\n')+1
ledger.write_bytes(raw[:end]+('\n'+heading+'\n\n'+note.lstrip()+'\n\n').encode('utf8')+raw[end:])
print(json.dumps(dict(receipt=binding(target),summary=summary),ensure_ascii=False))

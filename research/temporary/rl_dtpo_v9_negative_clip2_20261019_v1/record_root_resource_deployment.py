"""Record the completed storage override without altering its raw evidence."""
import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
sys.path.insert(0,str(HERE.parent/'rl_dtpo_v9_clip_u_20261019_v1'))
from transport import run_python

CODE=r'''
import hashlib,json,time,psutil
from pathlib import Path
R=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
A=R/'receipts/dtpo-v9-negative-clip2-postgroup-variance-20261019-v4'
run=R/'runs/textcraft-dtpo-v9-negative-clip2-8gpu-200-20261019-v4'
raw=run/'runtime-overrides/actual-worker-imports-root-pageable-resource-accepted-v1.json'
binding=json.loads(raw.read_bytes());source=json.loads((run/'source.json').read_bytes())
rows=binding['results'];assert sorted(r['rank'] for r in rows)==list(range(8))
for row in rows:
    d=row['root_pageable_resource_override']
    assert d['status']=='applied_existing_root_storage_option'
    assert d['before']==dict(pin_root_host=True,pin_replay_host=True)
    assert d['after']==dict(pin_root_host=False,pin_replay_host=True)
    assert d['arithmetic_changed'] is False
    assert psutil.Process(row['pid']).create_time()==row['birth']
capacity=A/'root-pageable-native-capacity-results.json'
pair=A/'existing-DT-root-pageable-pair-v2-results.json'
assert json.loads(capacity.read_bytes())['status']=='completed'
assert json.loads(pair.read_bytes())['successful_same_input_pair_count']==8
target=run/'runtime-overrides/root-pageable-resource-override-completed.json'
record=dict(status='completed_runtime_root_storage_override',recorded=time.time(),
    resource_version='root-pageable-replay-pinned-20261019-v1',
    actual_training_execution_source=source['local_patch_commit'],
    deployment_adapter_commit='692d3174',method_commit='f26b3b37cb3b88e64978e7a472ae4244670a39f3',
    upstream_VERL='20bd331bdbc9026a5668e11362178e10ab7400c8',run=str(run),
    original_source_receipt=str(run/'source.json'),
    raw_binding=dict(path=str(raw),SHA256=hashlib.sha256(raw.read_bytes()).hexdigest()),
    classification_correction='Raw inherited metadata wrapper status says read_only; its per-rank root_pageable_resource_override records the actual mutation. This completion receipt classifies it as a runtime resource override; raw evidence is preserved.',
    actual_workers=[dict(rank=r['rank'],pid=r['pid'],birth=r['birth'],override=r['root_pageable_resource_override']) for r in rows],
    effective_root_flags=dict(pin_root_host=False,pin_replay_host=True),
    original_environment_receipt_unchanged=True,queued_AppWorld_configuration_changed=False,
    checkpoint_or_training_restarted=False,model_calls_for_deployment=0,DT_calls_for_deployment=0,optimizer_calls_for_deployment=0,
    capacity=dict(path=str(capacity),SHA256=hashlib.sha256(capacity.read_bytes()).hexdigest(),scope='Original B4/exact32768 on ranks0/1; other six participate with B4/586. Not eight simultaneous max-length capacity.'),
    same_input_transport=dict(path=str(pair),SHA256=hashlib.sha256(pair.read_bytes()).hexdigest(),unchanged_raw_DT_vectors=8),
    numerical_scope='Existing storage allocation/copy option only. FA/FLA/VERL arithmetic owners and their accepted scopes unchanged; no new whole-model official tolerance certificate claimed.',
    current_acceptance='Applied to current TextCraft loaded runners after same-input and per-rank capacity checks; next complete production DT phase remains to be measured.',
    validation_iteration_excluded=13,first_entire_production_iteration_with_new_storage=14,
    production_efficiency_verified=False)
if target.exists():
    saved=json.loads(target.read_bytes())
    assert saved['raw_binding']==record['raw_binding']
    record=saved
else:target.write_text(json.dumps(record,indent=2)+'\n')
record['remote_receipt']=str(target)
print(json.dumps(record))
'''

def main():
    record=json.loads(run_python(CODE,timeout=45))
    result=REPO/'experiments/rl/results_dtpo_v9_negative2_root_resource_override_20261019_v4.json'
    result.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    stamp=datetime.fromtimestamp(record['recorded'],timezone(timedelta(hours=8))).isoformat()
    prefix=(f'## {stamp} 当前八卡DT仅切换原root缓存传输，生产完整阶段收益待核实\n\n'
        '执行源仍cdd8ea40、方法f26b3b37、VERL20bd331；原八worker于第13轮DT结束后、scheduler12时将原runner pin_root_host=True改为False，pin_replay_host=True保持。资源adapter692d3174已提交推送，未重启、未改FA/FLA/VERL算术或增加生产模型/DT/optimizer调用。当前run和PID出生保持；AppWorld排队配置尚未改变。\n\n'
        '同输入长B4八rank原DT向量逐值相同，归因调用中位128.593→107.653秒，共同窗口GPU Util39.732%→46.144%；两次调用之间宿主回收另列，不能当整阶段或整轮加速。原B4精确32768在rank0/1完成，六个其余rank用原B4/586参与；八rank Q/V/A有限、RNG不变、验证方法及flags已恢复。这不是八rank同时最长输入容量证明。第13轮含容量验证必须排除效率统计；第14轮起才是完整新配置生产窗口，目前尚无该完整窗口结果。\n\n'
        '实际完成回执为runtime-overrides/root-pageable-resource-override-completed.json；原metadata包装顶层沿用read_only字样，更正回执明确记录资源变更并保留原始证据。当前实际配置以该完成回执及逐rank记录为准。数值容差仍按PLAN对应官方计算，不将同向量观察或有限输出称新的全模型容差验收，不用checkpoint/多次更新0/0冒充。生成端官方异步/图已开启；已存trace的LoRA几何单独保存，未执行或部署LoRA调参。\n\n'
        '证据：results_dtpo_v9_negative2_root_capacity_20261019_v4.json、results_dtpo_v9_negative2_root_resource_override_20261019_v4.json及results_dtpo_v9_negative2_generation_LoRA_geometry_20261019_v4.json。六组goal保持active，后五组queued。\n\n')
    prefix_path=HERE/'root-resource-runtime-prefix.txt'
    prefix_path.write_text(prefix,encoding='utf-8')
    runtime=REPO/'experiments/rl/RUNTIME_RECORD.md'
    current=runtime.read_text(encoding='utf-8')
    if '当前八卡DT仅切换原root缓存传输' not in current[:1200]:
        runtime.write_text(prefix+current,encoding='utf-8')
    snapshot_path=REPO/'experiments/rl/current_runtime.json'
    snapshot=json.loads(snapshot_path.read_text(encoding='utf-8'))
    key='20261019_DTPO_v9_negative_clip2_six_GPU_v4_current'
    snapshot[key]['latest_root_storage_runtime_override']=dict(
        status=record['status'],recorded=stamp,receipt=record['remote_receipt'],
        local_result=result.relative_to(REPO).as_posix(),
        effective_root_flags=record['effective_root_flags'],
        actual_training_execution_source=record['actual_training_execution_source'],
        deployment_adapter_commit=record['deployment_adapter_commit'],
        first_entire_production_iteration=14,production_efficiency_verified=False,
        queued_AppWorld_configuration_changed=False)
    snapshot_path.write_text(json.dumps(snapshot,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(local_result=str(result),remote_receipt=record['remote_receipt'],status=record['status'],rank_count=len(record['actual_workers'])),indent=2))

if __name__=='__main__':main()

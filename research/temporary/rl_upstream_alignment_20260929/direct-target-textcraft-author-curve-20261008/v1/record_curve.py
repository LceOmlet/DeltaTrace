"""Bind the completed actual TextCraft author curve to the runtime ledger."""
import hashlib
import json
from pathlib import Path
import re

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[4]
def binding(path):
    data=path.read_bytes()
    return dict(path=path.as_posix(),bytes=len(data),sha256=hashlib.sha256(data).hexdigest())

receipt_path=REPO/'experiments/rl/results_textcraft_actual_author_curves_20261008.json'
result=json.loads(receipt_path.read_bytes())
terminal_path=sorted(HERE.glob('observation-*.json'))[-1]
terminal=json.loads(terminal_path.read_bytes())
assert terminal['completed'] and not terminal['driver']['same_birth']
assert terminal['textcraft_same_driver'] and not any(terminal['textcraft_release_present'])
physical={}
gpu=None
for line in terminal['physical_mx_smi'].splitlines():
    board=re.match(r'^\|\s*(\d+)\s+MetaX\s',line)
    if board:
        gpu=board[1]
    memory=re.search(r'(\d+)/(\d+) MiB',line)
    if memory:
        physical[gpu]=int(memory[1])
source=json.loads((HERE/'actual-results/results/rank0.json').read_bytes())
last=result['views']['signed_RISE']['groups'][-1]
value=dict(state='actual_TextCraft_author_curve_complete_credit_unrepaired',
    receipt=binding(receipt_path),launch=result['launch'],imported_owners=result['original_imported_owners'],
    helper_scripts=result['launch']['scripts'],original_author=result['author'],
    source_sha256=result['source_sha256'],native_sha256=result['native_sha256'],
    geometry=source['geometry'],
    source_count=result['source_count'],metrics={name:v['author_return'] for name,v in result['views'].items()},
    last_signed_group=last,controls=result['controls'],measurements=result['measurements'],
    terminal_observation=binding(terminal_path),physical_mib_at_terminal_poll=physical,
    original_effective_configuration=binding(HERE/'actual-results/results/effective-config.yaml'),
    original_actor_initialization=binding(HERE/'actual-results/results/actor-initialization.json'),
    unchanged=result['unchanged'],formal=dict(TextCraft_pid=2833207,TextCraft_birth=1791370325.16,
        update_release_present=[False,False],AppWorld='terminal_not_restarted'),
    numerical_repair='Not accepted. Original actual single-deletion and original cumulative deletion evidence remain separate. No clipping, multiplier, ordinary-gradient replacement, profile change or new tolerance.',
    memory_repair=dict(commit='799224868e0a9c0f8031b6012bb71505ab801a35',
        receipt=binding(REPO/'experiments/rl/results_memory_capacity_20261008.json'),
        status='Verified on original failed B4 and two exact32768 DT capacity calls with original vLLM lifecycle; not a formal deployment or whole PPO update claim.'))
runtime=REPO/'experiments/rl/current_runtime.json'
raw=runtime.read_bytes()
key='latest_actual_TextCraft_author_curve'
assert key not in json.loads(raw) and raw.endswith(b'}\r\n')
tail='\r\n'.join(json.dumps({key:value},ensure_ascii=False,indent=2).splitlines()[1:-1]).encode('utf8')
runtime.write_bytes(raw[:-3].rstrip(b'\r\n')+b',\r\n'+tail+b'\r\n}\r\n')
json.loads(runtime.read_bytes())
ledger=REPO/'experiments/rl/RUNTIME_RECORD.md'
heading='## 2026-10-08 TextCraft实际Format轨迹的原作者累计删除完成'
note='''
2b98d7f9/PID2967962/birth1791411696.44，物理4/5诊断已完成退出。
同一正式真实B4、3068个原source位置及原joint动作Y，原作者函数583f4b7d/k20，
每rank42次原native forward，零DT/采样/backward/optimizer/恢复，约244.18秒。
只给既有诊断增加task选择，并透传原TaskRunner已解析的330给原actor初始化；
无新训练预算、metric/scorer/sorting实现。真实ID transport原2项CPU检查通过。
两rank原数组/分数一致、paired twins差0、其他三行分数恒定，LoRA B均零。
原signed RISE=0.408195828；positive-only评估MAS=0.694071027。
Format在signed第20组删除。该组153个实际改变source均负、DT总和-13.990249，
native kept-minus-deleted却+56.012404。这是前19组已删除背景下的联合效应，
不能把56归给Format个人，也不替代它的原single删除A=-8.157497对照。
原始非单调logp、作者归一化/penalty数组、CSV、图和完整回执均保留；
这些是单条实际轨迹质量证据，不新设容差/总体质量结论或以排名证明指数幅度准确。
PSS记录峰值7.612/8.558GB，无OOM；本诊断未连续采样物理峰值。
Text原birth仍hold、release均不存在；App正式terminal，无正式重启/恢复。
信用仍未修复。79922486的consumed-FA-cache生命周期修复单独已验收，未正式部署。
导入路径/SHA、原配置、启动commit/PID创建时间与回执已绑定current_runtime末字段。
'''
raw=ledger.read_bytes()
assert heading.encode('utf8') not in raw
end=raw.index(b'\n')+1
ledger.write_bytes(raw[:end]+('\n'+heading+'\n\n'+note.lstrip()+'\n\n').encode('utf8')+raw[end:])
print(json.dumps(dict(receipt=binding(receipt_path),physical_mib=physical,last_signed_group=last),ensure_ascii=False))

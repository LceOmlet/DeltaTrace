"""Append completed CPU phase accounting; preserve all prior runtime records."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[4]


def binding(path):
    raw=path.read_bytes()
    return dict(path=path.relative_to(REPO).as_posix(),bytes=len(raw),
                sha256=hashlib.sha256(raw).hexdigest())


path=REPO/'experiments/rl/results_gdn_context_composition_20261008.json'
result=json.loads(path.read_bytes())
observation_path=sorted(HERE.glob('composition-state-*.json'))[-1]
observation=json.loads(observation_path.read_bytes())
assert observation['TextCraft']['same_birth']
assert not observation['AppWorld']['exists']
assert observation['TextCraft_update_release_present']==[False,False]
value=dict(receipt=binding(path),case=result['case'],
    owner_self_comparisons=result['owner_self_comparisons'],phase_rows=result['phase_rows'],
    formal_observation=binding(observation_path),operations=result['operations'],
    status='Diagnostic evidence only; no new credit path, tolerance or formal deployment.',
    pending_decision='Existing request about bounded native refinement remains unanswered; no refinement selection policy or replacement coefficient implemented.')
runtime=REPO/'experiments/rl/current_runtime.json'
raw=runtime.read_bytes()
key='latest_current_GDN_context_composition'
assert key not in json.loads(raw) and raw.endswith(b'}\r\n')
tail='\r\n'.join(json.dumps({key:value},ensure_ascii=False,indent=2).splitlines()[1:-1]).encode('utf8')
runtime.write_bytes(raw[:-3].rstrip(b'\r\n')+b',\r\n'+tail+b'\r\n}\r\n')
json.loads(runtime.read_bytes())
ledger=REPO/'experiments/rl/RUNTIME_RECORD.md'
raw=ledger.read_bytes()
heading='## 2026-10-08 当前GDN30反号的同操作数分阶段核算完成，零新增GPU调用'
assert heading.encode('utf8') not in raw
note='''
仅CPU核算已经完成的原结果，以相同四份artifact SHA、当前newline2883/UID、
原output cotangent及共享实际single-prefix state对齐；没有新模型/DT/backward/采样。
原joint FLA自己的有限/原生差为107.287136/107.289270，差-0.002134；
原single FLA自己的有限/原生差为15.825014/15.827504，差-0.002490。
原joint系数乘actual single位移只得9.671126，与原生single差-6.156378。
加同一已保存residual_skip=-16.636698、z=1.341236之后，记录的gate处+0.543606，
记录的joint finite FLA处-5.624335；用已测single finite/native作算术诊断分别
为+0.529553/+0.532043。不是新训练系数、全模型替换或官方有限归因容差。
记录值的分支重构残差0；gate implied o与原native BF16重放差-0.001754，
此普通数值差与跨端点背景的6.156378差分别保留，不新设误差门槛。
这把当前点反号的主要实测差定位到joint分解近似single的环节，不能证明全部
误差均来自该处。原V-only替换已全向量失败，不由此重试；原作者RISE/MAS独立保留。
1791415904.0715535只读核实TextCraft同PID/birth仍在、双rank release不存在，
AppWorld原正式PID不存在。79922486显存补丁保持verified/un-deployed；信用未修复。
既有单删除精化裁定仍待用户回复；未增加查询策略、修改PLAN或放行训练。
'''
end=raw.index(b'\n')+1
ledger.write_bytes(raw[:end]+('\n'+heading+'\n\n'+note.lstrip()+'\n\n').encode('utf8')+raw[end:])
print(json.dumps(dict(receipt=binding(path),formal_observation=binding(observation_path),credit_repaired=False)))

"""Record the completed native context diagnosis without changing a launch path."""
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]


def binding(path):
    raw = path.read_bytes()
    return dict(path=path.as_posix(), bytes=len(raw),
                sha256=hashlib.sha256(raw).hexdigest())


receipt = REPO / 'experiments/rl/results_current_token_reference_context_20261008.json'
result = json.loads(receipt.read_bytes())
terminal_path = sorted(HERE.glob('observation-*.json'))[-1]
terminal = json.loads(terminal_path.read_bytes())
assert terminal['completed'] and not terminal['driver']['same_birth']
assert terminal['textcraft_same_driver'] and not any(terminal['textcraft_release_present'])
physical = {}
gpu = None
for line in terminal['physical_mx_smi'].splitlines():
    board = re.match(r'^\|\s*(\d+)\s+MetaX\s', line)
    if board:
        gpu = board[1]
    match = re.search(r'(\d+)/(\d+) MiB', line)
    if match:
        physical[gpu] = int(match[1])

value = dict(
    state='current_AppWorld_context_sign_reversal_measured_credit_unrepaired',
    receipt=binding(receipt), launch=result['launch'],
    actual_imports=result['original_native_owners'],
    case=result['case'], endpoints=result['endpoints'],
    factual_context_effect=result['factual_context_effect'],
    all_EOS_context_effect=result['all_EOS_context_effect'],
    measured_interaction_difference=result['measured_interaction_difference'],
    controls=result['exact_endpoint_controls'], measurements=result['measurements'],
    original_effective_configuration=binding(HERE / 'actual-results/results/effective-config.yaml'),
    original_actor_initialization='Original reference diagnostic uses the recorded source and effective config; no separate actor-initialization artifact was emitted.',
    terminal_observation=binding(terminal_path), physical_mib_at_terminal_poll=physical,
    source_contract=dict(
        path=binding(REPO / 'README.md'),
        definition='DT decomposes the joint factual-versus-reference response log-probability change. Its per-source allocation is not a measured factual single-source deletion effect.',
        fixed_plan=binding(REPO / 'experiments/rl/PLAN.md'),
        scope='PLAN already treats the joint finite decomposition as an estimate of individual EOS deletion. This diagnosis measures an approximation failure; it does not change the accepted idealization or Q/V/A composition.'),
    interpretation=(
        'The same newline helps the joint target in the factual history (+23.089031) '
        'and hurts it in the all-other-source-EOS history (-9.470783). '
        'The original joint DT allocation (-4.415554 in fresh replay) is neither marginal. '
        'This establishes a substantial background interaction, not that all allocation error '
        'comes from one kernel, a precision tolerance failure, or a corrected learning coefficient.'),
    unchanged=result['unchanged'],
    formal=dict(TextCraft_pid=2833207, TextCraft_birth=1791370325.16,
                update_release_present=[False, False], AppWorld='terminal_not_restarted'),
    memory_repair=dict(
        commit='799224868e0a9c0f8031b6012bb71505ab801a35',
        receipt=binding(REPO / 'experiments/rl/results_memory_capacity_20261008.json'),
        status='Actual failed B4 and two exact32768 DT capacity calls passed with original vLLM lifecycle. Consumed FA cache release only; not formally deployed or claimed as a whole PPO update.'),
    decision='No clipping, multiplication, profile replacement, production per-token query, target change, optimizer release, formal restart or checkpoint restore.')

runtime = REPO / 'experiments/rl/current_runtime.json'
raw = runtime.read_bytes()
key = 'latest_current_AppWorld_reference_context_diagnosis'
assert key not in json.loads(raw) and raw.endswith(b'}\r\n')
tail = '\r\n'.join(json.dumps({key: value}, ensure_ascii=False, indent=2).splitlines()[1:-1]).encode('utf8')
runtime.write_bytes(raw[:-3].rstrip(b'\r\n') + b',\r\n' + tail + b'\r\n}\r\n')
json.loads(runtime.read_bytes())

ledger = REPO / 'experiments/rl/RUNTIME_RECORD.md'
heading = '## 2026-10-08 当前AppWorld极端换行的原生背景交互已量出'
note = '''
3c17af2f/PID3063340/birth1791412620.43已完成退出，物理4/5各回到859MiB。
当前真实B4、row3/packed2883/newline198及原joint Y不变。复用原native scorer、
pair builder和原actor，每rank只增加一次27.82/27.85秒的native paired forward；
零DT/backward/optimizer/rollout/恢复。两rank每target结果一致，其他三行差0、
此前target差0，LoRA B均0。新all-EOS端点与此前实际作者曲线末端逐值相同，
旧事实端点与作者曲线首端逐值相同。
原native四端点F=-230.145626,D=-253.234657,B=-928.803608,C=-938.274391。
完整事实背景的单删除效应F-D=+23.089031；其他source全EOS背景下恢复该token的
效应C-B=-9.470783，背景交互差32.559815。原DT鲜重放d=-4.415554既不是F-D
也不是C-B。因此当前大负优势不能解释为该换行在事实轨迹中有负作用；至少存在
显著背景交互分摊的近似误差。不能据此声称已定位全部误差或官方算子超差。
原DT接口明确分解joint端点变化；PLAN已经将其作为逐token删除效应估计，
此次结果量出该近似在实际极端点的失效，不改变既定精确假设或Q/V/PPO公式。
不把C-B或端点平均塞回训练，不再试无机制依据的kernel变体或数值纠偏。
TextCraft Format原生负作用仍有根据但幅度被高估；其原作者累计删除/RISE/MAS
与原single诊断同时保留，不被当前AppWorld四点对照取代。
PSS记录9.111/8.223GB，未连续采样本次物理峰值。Text原birth继续hold，
release均不存在；App正式terminal。信用仍未修复。79922486缓存生命周期补丁
已另行通过真实失败B4及双次32768容量，未部署到正式作业，也不冒充PPO整网验收。
实际导入、SHA、固定配置、PID创建时间、transport及回执写入current_runtime末字段。
'''
raw = ledger.read_bytes()
assert heading.encode('utf8') not in raw
end = raw.index(b'\n') + 1
ledger.write_bytes(raw[:end] + ('\n' + heading + '\n\n' + note.lstrip() + '\n\n').encode('utf8') + raw[end:])
print(json.dumps(dict(receipt=binding(receipt), physical_mib=physical,
                     credit_repaired=False, formal_restart=False), ensure_ascii=False))

"""Persist completed CPU owner-contract audit and the pending semantic decision."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]


def binding(path):
    raw = path.read_bytes()
    return dict(path=path.as_posix(), bytes=len(raw),
                sha256=hashlib.sha256(raw).hexdigest())


receipt = HERE / 'owner-contract-analysis.json'
audit = json.loads(receipt.read_bytes())
terminal_path = sorted(HERE.glob('observation-*.json'))[-1]
terminal = json.loads(terminal_path.read_bytes())
assert terminal['textcraft_same_driver'] and not any(terminal['textcraft_release_present'])
assert terminal['completed'] and not terminal['driver']['same_birth']
value = dict(
    status='actual_owner_contract_audited_no_new_estimator_or_training',
    receipt=binding(receipt), actual_runner=audit['actual_runner'],
    methods=[m['name'] for m in audit['methods']],
    measured_coefficient_errors=audit['measured_coefficient_errors'],
    actual_two_group_nonadditivity=audit['current_actual_two_group_nonadditivity'],
    pending_user_decision=audit['pending_user_decision'],
    terminal_observation=binding(terminal_path), operations=audit['operations'],
    formal=dict(TextCraft_pid=2833207, TextCraft_birth=1791370325.16,
                update_release_present=[False,False], AppWorld='terminal_not_restarted'),
    memory_status='79922486 lifetime fix separately verified on failed real B4 and two exact32768 DT calls; not formally deployed.')
runtime = REPO / 'experiments/rl/current_runtime.json'
raw = runtime.read_bytes()
key = 'latest_DT_single_deletion_interface_audit'
assert key not in json.loads(raw) and raw.endswith(b'}\r\n')
tail = '\r\n'.join(json.dumps({key:value}, ensure_ascii=False, indent=2).splitlines()[1:-1]).encode('utf8')
runtime.write_bytes(raw[:-3].rstrip(b'\r\n') + b',\r\n' + tail + b'\r\n}\r\n')
json.loads(runtime.read_bytes())
ledger = REPO / 'experiments/rl/RUNTIME_RECORD.md'
heading = '## 2026-10-08 实际DT接口审计：联合分解与单删除误差分别记录'
note = '''
本轮仅CPU AST读取此前已保存的实际导入源码，runner SHA7d6f57f6未改变。
原类仅__init__/forward_prefix/read_outcomes/attribute四个方法；attribute接收
调用者给定的交错端点，末端为同一组finite系数乘各token embedding位移。
原接口支持指定single-EOS pair，已有真实对照已验证；但所审计类没有单独的
一次全向量事实单删除估计入口。联合守恒不能代替逐token删除精度。
当前App真实两组（换行/其他source）的事实删除差之和731.217797，joint差
698.657982，非加性32.559815。这不是全部individual差之和或总体错误率。
Text Format的d误差-0.955824使概率比高估2.600812倍；App新鲜重放的d误差
-27.504585使A由原native约+0.75变-61.295728。仅描述原样本误差放大，不新设容差。
精确假设和Q/V/A组合未撤回；当前问题是实际joint有限估计的近似质量。
已向用户请求裁定是否允许研究一次DT之外的有界native单删除精化；选择规则、
阈值、替换系数和额外查询均未实现，未改PLAN。原作者RISE/MAS与官方算子容差
继续分开；不以单点结论判定整体归因或整个学习退化原因。
1791413421.829只读核实原诊断已退出，Text原PID/birth仍hold、release均不存在，
App正式terminal。零模型/DT/backward/optimizer/采样/恢复/重启。
原79922486显存生命周期修复及失败B4/双次32768回执不变，仍未正式部署。
'''
raw = ledger.read_bytes()
assert heading.encode('utf8') not in raw
end = raw.index(b'\n') + 1
ledger.write_bytes(raw[:end]+('\n'+heading+'\n\n'+note.lstrip()+'\n\n').encode('utf8')+raw[end:])
print(json.dumps(dict(receipt=binding(receipt), source=audit['actual_runner'],
                     production_modified=False), ensure_ascii=False))

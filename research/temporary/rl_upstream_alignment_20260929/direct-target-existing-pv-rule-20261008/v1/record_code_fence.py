"""Bind completed target isolation and the explicitly diagnostic GDN comparison."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]


def binding(path):
    return dict(path=path.as_posix(), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


receipt_path = REPO/'experiments/rl/results_code_fence_target_isolation_20261008.json'
receipt = json.loads(receipt_path.read_bytes())
observed = json.loads(sorted(HERE.glob('code-fence-observation-*.json'))[-1].read_bytes())
assert observed['completed'] and not observed['driver']['same_birth']
assert observed['textcraft_same_birth'] and not any(observed['textcraft_release_present'])
current = dict(
    state='completed_actual_next_code_fence_isolation_no_credit_repair',
    receipt=binding(receipt_path),
    launch=receipt['launch'],
    terminal_observation_unix=observed['unix'],
    controls=receipt['controls'],
    selected_newline=receipt['selected_newline'],
    measurements=receipt['measurements'],
    limitations=receipt['limitations'],
    interpretation=receipt['interpretation'],
    following_diagnostic=json.loads((HERE/'clean-gdn-launch.json').read_bytes()),
    following_diagnostic_state='launched_only_not_validated_or_deployed',
    formal=dict(TextCraft_pid=2833207, TextCraft_birth=1791370325.16,
                update_release_present=[False, False], AppWorld_state='terminal_not_restarted',
                restore_checkpoint=False, production_profile_changed=False),
    credit_repaired=False, official_accuracy_pass_claim=False,
    memory_fix_state='79922486 actual_failed_B4_and_two_32768_DT_capacity_verified_not_formally_deployed',
)
runtime = REPO/'experiments/rl/current_runtime.json'
raw = runtime.read_bytes()
key = 'latest_code_fence_target_isolation'
assert key not in json.loads(raw) and raw.endswith(b'}\r\n')
tail = '\r\n'.join(json.dumps({key: current},ensure_ascii=False,indent=2).splitlines()[1:-1]).encode('utf8')
runtime.write_bytes(raw[:-3].rstrip(b'\r\n') + b',\r\n' + tail + b'\r\n}\r\n')
json.loads(runtime.read_bytes())
ledger = REPO/'experiments/rl/RUNTIME_RECORD.md'
heading = '## 2026-10-08 实际下一代码围栏隔离完成：后续目标加入时出现负信用'
note = '''
5699ac34/PID2401809/birth1791406322.4已完成退出，单次完整DT/rank约105秒，
物理峰值各43420MiB。原实际B4、reference source IDs和所有target输入均保留；
只在原目标接口选择row3下一代码围栏71093的score，原2656个score变为1个。
这不是训练Y修改；诊断不导出完整事件Q/V/A，原producer日志也不能当作训练量。
换行符2883的完整DT d=-4.415554，所选围栏d=+4.272880；两者代数差-8.688434。
先前原生单删除围栏d=+21.029863，其余目标净d=+2.059169，总d=+23.089031。
故完整负值在后续目标加入时出现；不能据此删target、裁剪、加倍率或修改Q/V/PPO。
两rank隔离向量逐值相同，围栏端点logp跨运行逐值相同；其他三行归因max差0.014422，
端点logp max差4.7e-7，head输入形状变化。残差未定位，无全DT官方容差结论，
-8.688434只能称两次运行代数差，不能称精确独立剩余目标归因。

下一诊断b5b7b391/PID2575252/birth1791407996.54只在4/5核对作者保留clean-v1的
完整GDN数值规则（原norm/gate map和memory map均空），保留当前已验证存储生命周期、
内核、分块、真实target、Q/V/PPO。不是恢复历史runtime，也未切换正式profile；
源码SHA/CPU身份/启动实参见clean-profile-owner.json和clean-gdn-launch.json。
Text仍原birth hold，无optimizer release；App正式不重启、不恢复检查点。
当前信用未修复，79922486显存容量修复与此分开验收。
'''
raw = ledger.read_bytes()
assert heading.encode('utf8') not in raw
end = raw.index(b'\n') + 1
ledger.write_bytes(raw[:end]+('\n'+heading+'\n\n'+note.lstrip()+'\n\n').encode('utf8')+raw[end:])
print(json.dumps(dict(runtime=binding(runtime), ledger=binding(ledger), receipt=current['receipt'])))

"""Bind completed existing-probe population coverage to the runtime record."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
receipt = HERE/'population-coverage.json'
payload = receipt.read_bytes()
result = json.loads(payload)
observation_path = sorted(HERE.glob('observation-*.json'))[-1]
observation = json.loads(observation_path.read_bytes())
assert observation['textcraft_same_driver'] and not any(observation['textcraft_release_present'])
value = dict(receipt=dict(path=receipt.as_posix(), bytes=len(payload),
                         sha256=hashlib.sha256(payload).hexdigest()),
    coverage={task:{k:v for k,v in row.items() if k not in ('matched_actual_native_probes','scope')}
              for task,row in result['tasks'].items()},
    formal_observation=dict(path=observation_path.as_posix(),
        sha256=hashlib.sha256(observation_path.read_bytes()).hexdigest(),
        unix=observation['unix'], TextCraft_same_birth=True, update_release_present=[False,False]),
    operations=result['operations'],
    scope='Saved completed request slots only; coefficient square shares are not gradients, actor global batch or a population sign error rate.',
    credit_repaired=False, new_query_policy_implemented=False)
runtime = REPO/'experiments/rl/current_runtime.json'
raw = runtime.read_bytes()
key = 'latest_existing_extreme_probe_population_coverage'
assert key not in json.loads(raw) and raw.endswith(b'}\r\n')
tail = '\r\n'.join(json.dumps({key:value},ensure_ascii=False,indent=2).splitlines()[1:-1]).encode('utf8')
runtime.write_bytes(raw[:-3].rstrip(b'\r\n')+b',\r\n'+tail+b'\r\n}\r\n')
json.loads(runtime.read_bytes())
ledger = REPO/'experiments/rl/RUNTIME_RECORD.md'
raw = ledger.read_bytes()
heading = '## 2026-10-08 极端点原生对照覆盖范围量化完成，无新增GPU调用'
assert heading.encode('utf8') not in raw
note = '''
以原population的actual native file SHA、source SHA、UID、token ID和位置，
对齐已有三个原生单删除样本。没有新增前向、DT、backward、optimizer或采样。
AppWorld已完成54个native capture/216请求行/403866 source slots中的两个A<=-5，
恰好均为已测的newline198，均紧邻下一个target。原A=-53.979836/-21.460581，
原生事实删除d=+23.089031/+20.829633，两个负方向均不受事实单删除对照支持。
两点覆盖89.6333%的负source系数平方，但只占全部已存policy系数平方2.50288%；
不能据此声称占89.6%参数梯度或解释了全部学习退化。App正式DT未完成。
TextCraft已测Format覆盖43.6555%的负source平方/2.57717%的全部policy平方，
方向有据、幅度高估；已有真实global64原生PG对照中其full-gradient投影约4.0002%、
移除该项方向夹角12.4954度。此原生PG证据与平方占比分开保留，不运行GRPO。
TextCraft统计按176已保存请求行加权，含六个重复UID，不冒充原actor全局batch。
-5继承原描述bin，未改成新的查询门限、裁剪或验收标准。原RISE/MAS记录不变。
1791414096.140核实Text原PID/birth仍hold，release不存在；App正式terminal。
信用未修复、未增加精化接法。用户关于精化的裁定仍待回复；未改PLAN或正式入口。
'''
end = raw.index(b'\n')+1
ledger.write_bytes(raw[:end]+('\n'+heading+'\n\n'+note.lstrip()+'\n\n').encode('utf8')+raw[end:])
print(json.dumps(value,ensure_ascii=False))

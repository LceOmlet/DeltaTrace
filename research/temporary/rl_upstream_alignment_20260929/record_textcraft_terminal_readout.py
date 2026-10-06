"""Record completed saved head-probability observations, without training calls."""
import json
from pathlib import Path
import time

from stage_environment_entry import AUDIT, REPO
from record_textcraft_native_adam import source


def record():
    directory = AUDIT / 'textcraft-degradation-20261005'
    observation = directory / 'terminal-event-readout-source-observation-20261006.json'
    review = directory / 'independent-terminal-event-readout-review-20261006.json'
    measured = json.loads(observation.read_bytes())
    json.loads(review.read_bytes())
    paths = [observation, review, AUDIT / 'analyze_textcraft_terminal_event_readout.py', Path(__file__)]
    for item in measured['sources'].values():
        assert source(Path(item['path']))['sha256'] == item['sha256']
        paths.append(Path(item['path']))
    assert all(value == 0 for value in measured['operations'].values())
    assert source(AUDIT / 'analyze_textcraft_terminal_event_readout.py')['sha256'] == measured['analysis_source']['sha256']

    result = REPO / 'experiments/rl/results_textcraft_learning_degradation_20261006.json'
    report = json.loads(result.read_bytes())
    protected = {key: report[key] for key in ('status', 'production_status', 'fixed_configuration')}
    plan = source(REPO / 'experiments/rl/PLAN.md')
    known = {item['path']: item for item in report['sources']}
    for path in paths:
        item = source(path)
        if item['path'] in known:
            assert known[item['path']]['sha256'] == item['sha256']
        else:
            report['sources'].append(item)
            known[item['path']] = item
    for item in report['sources']:
        assert source(REPO / item['path'])['sha256'] == item['sha256'], item['path']
    report['terminal_reward_readout_CPU_observation'] = dict(
        role='Completed CPU observation of saved real head endpoints; not a new model test or learning repair.',
        observation=source(observation), independent_review=source(review),
        population=measured['population'], terminal=measured['terminal21'], earlier=measured['earlier165'],
        absolute_probability_source=measured['absolute_probability_source'],
        terminal_designation=measured['terminal_designation'], limits=measured['limits'],
        operations=measured['operations'],
        interpretation='The original coding has positive whole-response contrasts on all21 observed success-terminal actions. First-response future-outcome separation cannot be generalized to failure on every terminal action. Neither result proves individual token attribution or a training repair.')
    report['updated_unix'] = time.time()
    assert all(report[key] == value for key, value in protected.items())
    assert source(REPO / 'experiments/rl/PLAN.md') == plan

    runtime = REPO / 'experiments/rl/RUNTIME_RECORD.md'
    raw = runtime.read_bytes()
    heading = '## 2026-10-06 原成功终止动作的已存奖励读出'
    entry = (heading + '\n\n'
        '仅CPU读取原DT日志/head端点与原64 metadata，192运输slot映射到186唯一response；'
        '21条为导致官方done成功终止的最后已执行动作，165条为此前动作。没有伪造逐轮done字段，'
        '没有从Σd/守恒反推概率。原编码terminal事实成功概率均值0.921191、EOS参考0.842047，'
        '完整response目标LP差均值+0.0920603，21/21为正；交换编码另报，未选择为修复。\n\n'
        '这项证据否定将首轮随机未来的AUC描述扩大为全部终局读出失败；只含G1的子集'
        '不证明总体预测、token反事实或历史退化原因。原件/独立复核绑定既有结果字段'
        'terminal_reward_readout_CPU_observation。0新模型/DT/环境/反向/更新，'
        'PLAN、Q/V/A、上游PPO、正式版本和原status不变；TextCraft保持停止。\n\n')
    title, tail = raw.split(b'\n', 1)
    result.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    if heading.encode() not in raw:
        runtime.write_bytes(title + b'\n\n' + entry.encode() + tail)
    print(json.dumps(dict(result=source(result), runtime=source(runtime), sources=len(report['sources']))))


if __name__ == '__main__':
    record()

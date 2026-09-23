"""Audit real owner parsing, reset isolation, transitions and terminal behavior.

Consumes completed pilot outputs; controlled environment calls are test fixtures,
never policy samples. No simulator, parser, reward or environment wrapper is
reimplemented. AppWorld uses two recorded services outside the live port range.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    from agent_system.environments.env_package.sokoban import sokoban_projection
    from agent_system.environments.env_package.webshop import webshop_projection
    from agent_system.environments.env_package.appworld import appworld_projection
    from agent_system.environments.env_package.sokoban.envs import SokobanWorker
    from agent_system.environments.env_package.webshop.envs import WebshopWorker
    from agent_system.environments.env_package.appworld.envs import AppWorldWorker, load_available_ports
    from appworld import load_task_ids

    result = dict(scope=__doc__, started=time.time(), status='running', parsing={})
    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2)+'\n')
    save()
    for task, projection in [('Sokoban', sokoban_projection), ('Webshop', webshop_projection),
                             ('AppWorld', appworld_projection)]:
        reports = []
        for source in sorted((args.runtime_root/'runs/native-prefix-d6b7351-pilots'/task/'rollouts').glob('*.jsonl')):
            rows = [json.loads(line) for line in source.read_text().splitlines()]
            actions, valids = projection([r['output'] for r in rows])
            missing_think = sum('<think>' not in r['output'] or '</think>' not in r['output'] for r in rows)
            reports.append(dict(source=str(source), sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                rows=len(rows), owner_format_valid=sum(valids), missing_think_tags=missing_think,
                owner_action_types=sorted({type(a).__name__ for a in actions}),
                empty_actions=sum(a == '' for a in actions),
                interpretation='Owner validity includes reasoning-format checks; it is not execution success.'))
        result['parsing'][task] = reports
        save()

    sa = SokobanWorker('tiny_rgb_array', dict(dim_room=(6, 6), num_boxes=1, max_steps=15, search_depth=30))
    sb = SokobanWorker('tiny_rgb_array', dict(dim_room=(6, 6), num_boxes=1, max_steps=15, search_depth=30))
    try:
        initial, _ = sa.reset(7)
        other, _ = sb.reset(7)
        assert initial == other
        rewards, dones = [], []
        for move in ['still', 'right', 'up', 'right', 'down']:
            action, valid = sokoban_projection([f'<think>Interface fixture.</think><action>{move}</action>'])
            obs, reward, done, info = sa.step(action[0])
            assert sb.env.render('tiny_rgb_array') == other
            rewards.append(reward); dones.append(done)
        assert rewards == [-.1, -.1, -.1, -.1, 10.9] and dones == [False]*4+[True]
        assert info['won']
        assert sa.reset(7)[0] == initial
        result['sokoban'] = dict(status='passed', independent_instances=True, reset_reproducible=True,
            rewards=rewards, dones=dones, terminal_won=bool(info['won']))
    finally:
        sa.env.close(); sb.env.close()
    save()
    print('Sokoban state checks passed', flush=True)

    webshop_root = Path(os.environ['WEBSHOP_ROOT'])
    web_kwargs = dict(observation_mode='text', num_products=None, human_goals=False,
        file_path=str(webshop_root/'data/items_shuffle_1000.json'),
        attr_path=str(webshop_root/'data/items_ins_v2_1000.json'))
    wa = WebshopWorker(7, web_kwargs)
    wb = WebshopWorker(7, web_kwargs)
    try:
        initial, _ = wa.reset(500); other, _ = wb.reset(500)
        assert initial == other
        other_state = copy.deepcopy(wb.env.state)
        # Existing official training-goal fixture; never enters training data.
        goal = wa.get_goals()[500]
        commands = [f"search[{goal['name']}]", f"click[{goal['asin'].lower()}]"]
        commands += [f'click[{v}]' for v in goal['goal_options'].values()] + ['click[buy now]']
        transitions = []
        for command in commands:
            projected, _ = webshop_projection([f'<think>Interface fixture.</think><action>{command}</action>'])
            obs, reward, done, info = wa.step(projected[0])
            assert wb.env.state == other_state
            transitions.append(dict(action=projected[0], reward=reward, done=done, task_score=info['task_score']))
            if done:
                break
        assert done and reward == 10 and info['task_score'] == 1 and info['won']
        assert wa.reset(500)[0] == initial
        result['webshop'] = dict(status='passed', independent_instances=True, reset_reproducible=True,
            oracle_fixture_only=True, transitions=transitions)
    finally:
        wa.close(); wb.close()
    save()
    print('WebShop state checks passed', flush=True)

    manifest = json.loads((args.runtime_root/'formal-training.json').read_text())
    job = next(j for j in manifest['jobs'] if j['task'] == 'AppWorld')
    settings = job['settings']
    ports = load_available_ports(settings['APPWORLD_PORT_FILE'])
    occupied = int(settings['TRAIN_SIZE'])*int(settings['GROUP_SIZE']) + int(settings['VAL_SIZE'])
    assert len(set(ports)) == len(ports) and len(ports)-occupied >= 2
    selected = ports[-2:]
    assert not set(selected) & set(ports[:occupied])
    task = load_task_ids(settings['APPWORLD_TRAIN_DATASET'])[0]
    stamp = int(time.time()*1000)
    aa, ab = [AppWorldWorker(stamp+i, 40, port) for i, port in enumerate(selected)]
    try:
        initial, info_a = aa.reset(task); other, info_b = ab.reset(task)
        assert initial == other and info_a == info_b
        _, r, done, _ = aa.step("dt_interface_marker = 7931\nprint(dt_interface_marker)")
        assert r == 0 and not done
        obs, r, done, _ = ab.step("print('dt_interface_marker' in globals())")
        assert 'False' in obs and r == 0 and not done
        for step in range(2, 41):
            obs, reward, done, info = aa.step('print(1)')
            assert info['step_count'] == step and done == (step == 40) and reward == 0
        initial_again, _ = aa.reset(task)
        obs, reward, done, info = aa.step("print('dt_interface_marker' in globals())")
        assert initial_again == initial and 'False' in obs and not done and info['step_count'] == 1
        # Exercise the actual completion API and its persistent task status.
        # The task has not been solved, so closing it must not invent success.
        obs, reward, done, info = aa.step('apis.supervisor.complete_task()')
        assert done and reward == 0 and not info['won'] and aa.env.task_completed()
        assert not ab.env.task_completed()
        aa.reset(task)
        assert not aa.env.task_completed()
        result['appworld'] = dict(status='passed', independent_remote_repl=True, reset_clears_repl=True,
            independent_completion_state=True, reset_clears_completion=True,
            occupied_port_count=occupied, audit_ports=selected, task_id=task, terminal_step=40,
            terminal_failure_reward=0, audit_worker_ids=[stamp, stamp+1], live_training_ports_untouched=True)
    finally:
        aa.close(); ab.close()
    result.update(status='passed', finished=time.time())
    save()
    print('AppWorld state checks passed', flush=True)


if __name__ == '__main__':
    main()

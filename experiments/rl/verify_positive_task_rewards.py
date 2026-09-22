"""Positive official training-task reward fixtures, not agent performance.

Uses disclosed official training-goal/solution access to exercise success paths.
These traces must not be included in policy training or scored as model success.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path


def webshop_success():
    from agent_system.environments.env_package.webshop.envs import WebshopWorker
    worker = WebshopWorker(7, dict(observation_mode='text', num_products=1000))
    transitions = []
    try:
        obs, _ = worker.reset(500)  # Pinned upstream training split starts at 500.
        goal = worker.get_goals()[500]
        commands = [f"search[{goal['name']}]", f"click[{goal['asin'].lower()}]"]
        commands += [f'click[{value}]' for value in goal['goal_options'].values()]
        commands += ['click[buy now]']
        for action in commands:
            nxt, reward, done, info = worker.step(action)
            transitions.append((obs, action, nxt, reward, done, info))
            obs = nxt
            if done:
                break
        if not transitions[-1][4] or transitions[-1][3] != 10:
            raise AssertionError(f'Official positive Webshop fixture failed: {transitions[-1][3:]}')
        return transitions
    finally:
        worker.close()


def appworld_success():
    from appworld import AppWorld, load_task_ids
    from appworld.ground_truth import GroundTruth
    from agent_system.environments.env_package.appworld.envs import AppWorldWorker
    task_id = load_task_ids('train')[0]
    worker = AppWorldWorker(worker_id=0, max_interactions=15, port=0)
    worker.env = AppWorld(task_id=task_id, experiment_name='dt_positive_fixture_' +
                          datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f'))
    try:
        code = GroundTruth.load(task_id, mode='full').compiled_solution_code_body
        obs = worker.env.task.instruction
        nxt, reward, done, info = worker.step(code)
        if not done or reward != 10:
            raise AssertionError(f'Official AppWorld solution failed: {reward}, {done}, {nxt}')
        return [(obs, code, nxt, reward, done, info)]
    finally:
        worker.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rollout-fixtures', type=Path)
    args = parser.parse_args()
    from transformers import AutoTokenizer
    from verify_task_reward_events import check_events, sokoban_events, task_rows
    tokenizer = AutoTokenizer.from_pretrained(os.environ['MODEL_PATH'], local_files_only=True)
    result = dict(scope='official positive reward fixtures plus synthetic-ratio transport; no model/performance/training claim',
                  oracle_access=True, split='train', context_cap=32768, tasks={})
    fixtures = dict(action_source='scripted/oracle verification fixtures; never policy training', tasks={})
    for task, fn in [('Sokoban', sokoban_events), ('Webshop', webshop_success), ('AppWorld', appworld_success)]:
        transitions = fn()
        result['tasks'][task] = check_events(task, transitions, tokenizer)
        if args.rollout_fixtures:
            rows, lengths = task_rows(task, transitions, tokenizer)
            fixtures['tasks'][task] = dict(lengths=lengths, rows=[{
                key: (value.tolist() if hasattr(value, 'tolist') else value)
                for key, value in row.items() if key != 'dt_env_outcome'
            } for row in rows])
        print(task, json.dumps(result['tasks'][task]), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    if args.rollout_fixtures:
        args.rollout_fixtures.parent.mkdir(parents=True, exist_ok=True)
        args.rollout_fixtures.write_text(json.dumps(fixtures)+'\n')


if __name__ == '__main__':
    main()

"""Original WebShop worker/scenario sampling with explicit transport fixtures.

No search index, browser, Ray service, or environment simulator is started.
"""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / ('research/temporary/rl_upstream_alignment_20260929/recipe-sources/'
                 'verl-agent-20bd331/agent_system/environments/env_package/webshop/envs.py')
spec = importlib.util.spec_from_file_location('official_webshop_boundary', SOURCE)
owner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(owner)


@pytest.mark.parametrize('done,score,expected', [(False, 1., 0), (True, .5, 0), (True, 1., 10), (True, 0., 0)])
def test_original_worker_reward_and_raw_task_score(done, score, expected):
    worker = owner.WebshopWorker.__new__(owner.WebshopWorker)
    native_info = {'fixture': 'environment reply'}
    worker.env = SimpleNamespace(step=Mock(return_value=('page', score, done, native_info)),
                                 get_available_actions=Mock(return_value={'clickables': ['buy']}))
    obs, reward, ended, info = worker.step('click[buy]')
    worker.env.step.assert_called_once_with('click[buy]')
    assert (obs, reward, ended) == ('page', expected, done)
    assert info['task_score'] == score and info['won'] == (expected == 10)
    assert native_info == {'fixture': 'environment reply'}  # Owner copies info.


@pytest.mark.parametrize('training,group', [(True, 3), (False, 1)])
def test_original_train_eval_split_and_group_sampling(monkeypatch, training, group):
    workers = []
    def remote_constructor(seed, env_kwargs):
        worker = SimpleNamespace(
            get_goals=SimpleNamespace(remote=lambda: list(range(2000))),
            reset=SimpleNamespace(remote=Mock(side_effect=lambda idx: (str(idx), {'idx': idx}))),
            close=SimpleNamespace(remote=lambda: None))
        workers.append(worker)
        return worker
    monkeypatch.setattr(owner.ray, 'is_initialized', lambda: True)
    monkeypatch.setattr(owner.ray, 'remote', lambda **kwargs: lambda cls: SimpleNamespace(remote=remote_constructor))
    monkeypatch.setattr(owner.ray, 'get', lambda result: result)
    monkeypatch.setattr(owner.ray, 'kill', lambda worker: None)
    env = owner.build_webshop_envs(seed=19, env_num=2, group_n=group,
                                    resources_per_worker={'num_cpus': .1}, is_train=training)
    try:
        assert env.goal_idxs == (range(500, 2000) if training else range(500))
        _, infos = env.reset()
        assert len(infos) == 2 * group
        assert len({info['idx'] for info in infos}) == 2
        for offset in range(0, len(infos), group):
            assert len({info['idx'] for info in infos[offset:offset + group]}) == 1
        assert all(info['idx'] in env.goal_idxs for info in infos)
    finally:
        env.close()

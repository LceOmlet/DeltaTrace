"""Exercise the pinned owner's configuration boundary, not an environment clone."""
from unittest.mock import patch

from agent_system.environments.env_package.appworld import envs


def test_appworld_factory_preserves_defaults():
    with patch.object(envs, 'AppWorldEnvs') as constructor:
        envs.build_appworld_envs()
    assert constructor.call_args.kwargs == dict(
        dataset_name='train', max_interactions=50, seed=0, env_num=1, group_n=1,
        start_server_id=0, resources_per_worker={'num_cpus': .1},
        port_file='appworld_ports.ports',
    )


def test_appworld_factory_forwards_paper_scale_arguments():
    kwargs = dict(dataset_name='train_difficulty_1_and_train_difficulty_2',
                  max_interactions=40, seed=0, env_num=40, group_n=6,
                  start_server_id=0, resources_per_worker={'num_cpus': .1},
                  port_file='/tmp/explicit-owner-port-list.ports')
    with patch.object(envs, 'AppWorldEnvs') as constructor:
        envs.build_appworld_envs(**kwargs)
    assert constructor.call_args.kwargs == kwargs

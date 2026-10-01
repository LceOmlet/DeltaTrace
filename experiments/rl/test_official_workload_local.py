"""Official entry/config workload accounting; no loss or update is executed."""
import ast
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from omegaconf import OmegaConf

from test_official_recipes_local import VERL, AGENTGYM, SKYRL, capture, module_args, compose_verl


def normalize_native_minibatch(source, config, dp):
    """Execute the owner's exact two configuration assignments, not an optimizer."""
    tree = ast.parse(source)
    statements = [n for n in ast.walk(tree) if isinstance(n, ast.AugAssign)
                  and ast.unparse(n.target) == 'self.config.actor.ppo_mini_batch_size']
    assert len(statements) == 2
    statements.sort(key=lambda n: n.lineno)
    worker = SimpleNamespace(config=OmegaConf.create(config), ulysses_sequence_parallel_size=1,
                             device_mesh=SimpleNamespace(size=lambda: dp, shape=(dp,)))
    exec(compile(ast.Module(body=statements, type_ignores=[]), '<native-config-only>', 'exec'), {'self': worker})
    return worker.config.actor.ppo_mini_batch_size


def project_config(options):
    """Compose the real project launcher with the pinned VERL defaults."""
    from owner_runtime_options import owner_command
    return compose_verl(VERL, owner_command(options)[3:])


def assert_project_trajectory_units(cfg, *, groups, samples, global_mini, epochs):
    # env.rollout.n repeats task prompts. The engine must NOT repeat the
    # already expanded trajectories again during worker normalization.
    assert cfg['data']['train_batch_size'] == groups
    assert cfg['env']['rollout']['n'] == samples
    rollout = cfg['actor_rollout_ref']
    assert rollout['rollout']['n'] == 1
    source = (VERL/'verl/workers/fsdp_workers.py').read_text(encoding='utf8')
    assert normalize_native_minibatch(source, rollout, 2) * 2 == global_mini
    assert rollout['actor']['ppo_epochs'] == epochs
    assert rollout['actor']['ppo_micro_batch_size_per_gpu'] == 4
    assert rollout['model']['lora_rank'] == 8
    assert rollout['model']['lora_alpha'] == 16


def test_formal_sql_launcher_matches_native_prompt_and_trajectory_units(tmp_path, monkeypatch):
    from skyrl.train.config import TrainerConfig, GeneratorConfig
    from launch_sql_native import command
    monkeypatch.setenv('MODEL_PATH', 'Qwen/Qwen3.5-9B')
    calls = capture(SKYRL/'examples/train/text_to_sql/run_skyrl_sql.sh', tmp_path/'owner')
    raw = OmegaConf.from_cli(module_args(calls, 'skyrl.train.entrypoints.main_base'))
    trainer = TrainerConfig.from_dict_config(raw.trainer)
    generator = GeneratorConfig.from_dict_config(raw.generator)
    _, options = command(SimpleNamespace(method='dt', phase='formal', data=tmp_path, output=tmp_path))
    cfg = project_config(options)
    assert_project_trajectory_units(cfg, groups=trainer.train_batch_size,
        samples=generator.n_samples_per_prompt,
        global_mini=trainer.policy_mini_batch_size * generator.n_samples_per_prompt,
        epochs=trainer.update_epochs_per_batch)
    assert cfg['trainer']['total_epochs'] == trainer.epochs
    assert cfg['trainer']['total_training_steps'] is None
    assert cfg['trainer']['test_freq'] == trainer.eval_interval
    assert cfg['trainer']['save_freq'] == trainer.ckpt_interval
    assert cfg['env']['max_steps'] == generator.max_turns


def test_formal_textcraft_launcher_matches_native_effective_workload(tmp_path, monkeypatch):
    from launch_textcraft_native import options_for
    monkeypatch.setenv('MODEL_PATH', 'Qwen/Qwen3.5-9B')
    monkeypatch.setenv('AGENTGYM_RL_ROOT', str(AGENTGYM))
    calls = capture(AGENTGYM/'examples/train/AgentGym-RL/textcraft_train.sh', tmp_path/'owner')
    native = compose_verl(AGENTGYM, module_args(calls, 'verl.agent_trainer.main_ppo'),
                          'AgentGym-RL/verl/agent_trainer/config')
    native_rollout = native['actor_rollout_ref']
    native_source = (AGENTGYM/'AgentGym-RL/verl/workers/agent_fsdp_workers.py').read_text(encoding='utf8')
    global_mini = normalize_native_minibatch(native_source, native_rollout, 8) * 8
    # The real author update_policy has one pass, despite the unused recipe
    # value ppo_epochs=2. Inspect that source rather than repeating this value.
    tree = ast.parse((AGENTGYM/'AgentGym-RL/verl/workers/agent_actor/dp_actor.py').read_text())
    update = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'update_policy')
    assert not any(isinstance(n, ast.Attribute) and n.attr == 'ppo_epochs' for n in ast.walk(update))
    options, _ = options_for(tmp_path, tmp_path)
    cfg = project_config(options)
    assert_project_trajectory_units(cfg, groups=native['data']['train_batch_size'],
        samples=native_rollout['rollout']['n'], global_mini=global_mini, epochs=1)
    for key in ('total_epochs', 'save_freq', 'test_freq'):
        assert cfg['trainer'][key] == native['trainer'][key]
    assert cfg['trainer']['total_training_steps'] is None
    assert cfg['env']['max_steps'] == native['algorithm']['rounds_ctrl']['rounds']


@pytest.mark.parametrize('task,groups,n,mini,dp,effective', [
    ('textcraft', 32, 8, 8, 8, 64), ('webarena', 32, 4, 4, 8, 16),
])
def test_agentgym_native_normalization_does_not_treat_microbatch_as_update(task, groups, n, mini, dp, effective, tmp_path):
    calls = capture(AGENTGYM / f'examples/train/AgentGym-RL/{task}_train.sh', tmp_path)
    cfg = compose_verl(AGENTGYM, module_args(calls, 'verl.agent_trainer.main_ppo'),
                       'AgentGym-RL/verl/agent_trainer/config')
    source = (AGENTGYM / 'AgentGym-RL/verl/workers/agent_fsdp_workers.py').read_text(encoding='utf-8')
    actor_rollout = cfg['actor_rollout_ref']
    per_rank = normalize_native_minibatch(source, actor_rollout, dp)
    assert per_rank * dp == effective
    assert groups * n // effective == (4 if task == 'textcraft' else 8)
    assert actor_rollout['actor']['ppo_micro_batch_size_per_gpu'] == 1
    # The pinned implementation does NOT loop over ppo_epochs in update_policy.
    # The value 2 exists in the recipe and is read only by MFU reporting.
    # Preserve the owner's actual workload; don't silently add a second epoch.
    actor_tree = ast.parse((AGENTGYM / 'AgentGym-RL/verl/workers/agent_actor/dp_actor.py').read_text())
    update = next(n for n in ast.walk(actor_tree) if isinstance(n, ast.FunctionDef) and n.name == 'update_policy')
    assert actor_rollout['actor']['ppo_epochs'] == 2
    assert not any(isinstance(n, ast.Attribute) and n.attr == 'ppo_epochs' for n in ast.walk(update))


def test_webshop_native_global_minibatch_is_preserved_when_microbatch_shrinks(tmp_path):
    calls = capture(VERL / 'examples/grpo_trainer/run_webshop.sh', tmp_path)
    cfg = compose_verl(VERL, module_args(calls, 'verl.trainer.main_ppo'))
    source = (VERL / 'verl/workers/fsdp_workers.py').read_text(encoding='utf-8')
    actor_rollout = cfg['actor_rollout_ref']
    assert cfg['data']['train_batch_size'] * cfg['env']['rollout']['n'] == 128
    per_rank = normalize_native_minibatch(source, actor_rollout, 2)
    actor_rollout['actor']['ppo_micro_batch_size_per_gpu'] = 4
    assert normalize_native_minibatch(source, actor_rollout, 2) == per_rank == 32
    assert per_rank * 2 == 64  # Global optimizer minibatch, not 4.
    assert per_rank // 8 == 4 and per_rank // 4 == 8  # More accumulation, same update batch.


def test_skyrl_native_step_budget_uses_prompt_units_not_sample_units(tmp_path):
    from skyrl.train.config import TrainerConfig, GeneratorConfig
    calls = capture(SKYRL / 'examples/train/text_to_sql/run_skyrl_sql.sh', tmp_path)
    raw = OmegaConf.from_cli(module_args(calls, 'skyrl.train.entrypoints.main_base'))
    cfg = SimpleNamespace(trainer=TrainerConfig.from_dict_config(raw.trainer))
    generator = GeneratorConfig.from_dict_config(raw.generator)
    tree = ast.parse((SKYRL / 'skyrl/train/trainer.py').read_text(encoding='utf-8'))
    statement = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == 'policy_steps_per_train_batch' for t in n.targets))
    namespace = {'cfg': cfg}
    exec(compile(ast.Module(body=[statement], type_ignores=[]), '<native-budget-only>', 'exec'), namespace)
    assert cfg.trainer.train_batch_size * generator.n_samples_per_prompt == 1280
    assert namespace['policy_steps_per_train_batch'] == 1  # 256/256 * 1, not 1280/256.


def test_webshop_owner_gathers_each_active_environment_action_once():
    # Compile the unmodified owner method for an isolated boundary test. Only
    # output collation/DataProto are recording fixtures; its loop is not copied.
    file = VERL / 'agent_system/multi_turn_rollout/rollout_loop.py'
    tree = ast.parse(file.read_text(encoding='utf-8'))
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'gather_rollout_data')
    header = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    ns = {'np': np, 'collate_fn': Mock(side_effect=lambda rows: rows),
          'DataProto': SimpleNamespace(from_single_dict=lambda **kwargs: kwargs['data'])}
    code = ast.fix_missing_locations(ast.Module(body=[header, method], type_ignores=[]))
    exec(compile(code, str(file), 'exec'), ns)
    trajectories = [[{'traj_uid': f't{i}', 'active_masks': step < 1 + i % 15, 'step': step}
                     for step in range(15)] for i in range(128)]
    lengths = np.array([1 + i % 15 for i in range(128)])
    output = ns['gather_rollout_data'](None, trajectories, np.zeros(128), lengths,
                                     {}, np.array([f't{i}' for i in range(128)]), np.zeros(128))
    assert len(output) == int(lengths.sum())
    assert len({(row['traj_uid'], row['step']) for row in output}) == len(output)
    assert len(output) <= 128 * 15
    ns['collate_fn'].assert_called_once()

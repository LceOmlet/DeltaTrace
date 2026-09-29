"""Native model-path entry and config workload equality; no model or training."""
from copy import deepcopy

import pytest
from omegaconf import OmegaConf

from test_official_recipes_local import ROOT, VERL, SKYRL, AGENTGYM, capture, module_args, compose_verl
from patch_agentgym_eval_paths import patch, forward_native_overrides

MODEL = 'Qwen/Qwen3.5-9B'


def test_webshop_qwen35_entry_keeps_the_whole_official_workload(tmp_path):
    owner = capture(VERL / 'examples/grpo_trainer/run_webshop.sh', tmp_path / 'owner')
    project = capture(ROOT / 'experiments/rl/run_official_task.sh', tmp_path / 'project',
        env={'VERL_ROOT': VERL.as_posix(), 'ENV_NAME': 'Webshop', 'METHOD': 'grpo'},
        args=(f'actor_rollout_ref.model.path={MODEL}',))
    # Let native Hydra update its model-path interpolations too (critic/tokenizer).
    expected = compose_verl(VERL, module_args(owner, 'verl.trainer.main_ppo') + [
        f'actor_rollout_ref.model.path={MODEL}', 'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4'])
    actual = compose_verl(VERL, module_args(project, 'verl.trainer.main_ppo'))
    assert actual == expected  # Includes generation, environment, eval and update budgets.
    assert actual['data']['train_batch_size'] * actual['env']['rollout']['n'] == 128
    assert actual['env']['max_steps'] == 15
    assert actual['actor_rollout_ref']['actor']['ppo_mini_batch_size'] == 64


def test_sql_qwen35_native_cli_keeps_every_other_override(tmp_path):
    script = SKYRL / 'examples/train/text_to_sql/run_skyrl_sql.sh'
    original = capture(script, tmp_path / 'owner')
    candidate = capture(script, tmp_path / 'qwen35', args=(f'trainer.policy.model.path={MODEL}',))
    old = OmegaConf.from_cli(module_args(original, 'skyrl.train.entrypoints.main_base'))
    new = OmegaConf.from_cli(module_args(candidate, 'skyrl.train.entrypoints.main_base'))
    expected = deepcopy(old)
    expected.trainer.policy.model.path = MODEL
    assert OmegaConf.to_container(new) == OmegaConf.to_container(expected)
    assert new.trainer.train_batch_size * new.generator.n_samples_per_prompt == 1280
    assert new.generator.max_turns == 6


@pytest.mark.parametrize('task', ['textcraft', 'webarena'])
@pytest.mark.parametrize('phase', ['train', 'eval'])
def test_agentgym_native_cli_default_and_qwen35_workloads(tmp_path, task, phase):
    relative = f'examples/train/AgentGym-RL/{task}_train.sh' if phase == 'train' else f'examples/eval/{task}_eval.sh'
    script = AGENTGYM / relative
    model_key = 'actor_rollout_ref.model.path' if phase == 'train' else 'model.path'
    module = 'verl.agent_trainer.main_ppo' if phase == 'train' else 'verl.agent_trainer.main_generation'
    name = 'ppo_trainer' if phase == 'train' else 'generation'
    config_dir = 'AgentGym-RL/verl/agent_trainer/config'
    original = capture(script, tmp_path / 'original', allow_directory_error=phase == 'eval')
    ignored = capture(script, tmp_path / 'reproduce', args=(f'{model_key}={MODEL}',),
                      allow_directory_error=phase == 'eval')
    assert module_args(ignored, module) == module_args(original, module)  # Reproduce ignored CLI.
    source = script.read_text(encoding='utf-8')
    fixed = forward_native_overrides(patch(source) if phase == 'eval' else source)
    assert forward_native_overrides(fixed) == fixed
    candidate_script = tmp_path / 'candidate.sh'
    candidate_script.write_text(fixed, encoding='utf-8')
    default = capture(candidate_script, tmp_path / 'default')
    assert module_args(default, module) == module_args(original, module)
    targeted = capture(candidate_script, tmp_path / 'qwen35', args=(f'{model_key}={MODEL}',))
    new = compose_verl(AGENTGYM, module_args(targeted, module), config_dir, name)
    expected = compose_verl(AGENTGYM, module_args(original, module) + [f'{model_key}={MODEL}'], config_dir, name)
    assert new == expected  # No rollout count, budget, history, reward or update expansion.

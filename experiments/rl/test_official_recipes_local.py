"""Execute unchanged author scripts into a recording command sink, never a trainer.

Only external process launches/activation are replaced in this test. Bash owns
expansion and the framework owns config composition. These tests are not model,
service, or performance validation.
"""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
SOURCES = ROOT / 'research/temporary/rl_upstream_alignment_20260929/recipe-sources'
VERL = SOURCES / 'verl-agent-20bd331'
SKYRL = SOURCES / 'SkyRL-7d94ccf0eac3439c1731ce32018bf043dd639806'
AGENTGYM = SOURCES / 'AgentGym-RL-82402a99c62a293735a3f412fb8ac9a600673bc0'
BASH = os.environ.get('DT_TEST_BASH', 'C:/Program Files/Git/bin/bash.exe')


def capture(script, tmp_path, *, args=(), env=None, allow_directory_error=False):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / 'AgentGym-RL').mkdir(exist_ok=True)
    capture_file = tmp_path / 'commands.jsonl'
    recorder = tmp_path / 'record.py'
    recorder.write_text(
        'import json,os,sys\n'
        'with open(os.environ["DT_TEST_CAPTURE"],"a",encoding="utf-8") as f:\n'
        ' f.write(json.dumps(sys.argv[1:])+"\\n")\n', encoding='utf-8')
    sink = tmp_path / 'python-sink.sh'
    sink.write_text('#!/usr/bin/env bash\n' + shlex.quote(Path(sys.executable).as_posix())
                    + ' ' + shlex.quote(recorder.as_posix()) + ' "$@"\n', encoding='utf-8')
    # No training command, activation, login or installation is executed.
    # The author script itself, including assignments/branches/expansion, is exact.
    driver = '''
python3() { "$DT_TEST_SINK" "$@"; }
python() { "$DT_TEST_SINK" "$@"; }
uv() { "$DT_TEST_SINK" "$@"; }
conda() { :; }
source() { if [[ "$1" == activate ]]; then :; else builtin source "$@"; fi; }
wandb() { :; }
DT_SCRIPT_TO_SOURCE="$1"
shift
builtin source "$DT_SCRIPT_TO_SOURCE" "$@"
'''
    call_env = dict(os.environ, DT_TEST_CAPTURE=str(capture_file),
                    DT_TEST_SINK=sink.as_posix(), VENV_PYTHON=sink.as_posix(),
                    MSYS2_ARG_CONV_EXCL='*')
    call_env.update(env or {})
    proc = subprocess.run([BASH, '--noprofile', '--norc', '-c', driver, 'capture',
                           script.as_posix(), *args], cwd=tmp_path, env=call_env,
                          text=True, capture_output=True, timeout=30)
    assert proc.returncode == 0, proc.stderr
    (tmp_path / 'shell.stderr').write_text(proc.stderr, encoding='utf-8')
    if not allow_directory_error:
        assert 'cd:' not in proc.stderr, proc.stderr
    return [json.loads(line) for line in capture_file.read_text(encoding='utf-8').splitlines()]


def module_args(commands, module):
    return next(row[row.index(module) + 1:] for row in commands if module in row)


def compose_verl(owner, overrides, directory='verl/trainer/config', name='ppo_trainer'):
    with initialize_config_dir(version_base=None, config_dir=str(owner / directory)):
        return OmegaConf.to_container(compose(config_name=name, overrides=overrides), resolve=True)


@pytest.mark.parametrize('method', ['grpo', 'dt'])
def test_webshop_wrapper_matches_whole_native_recipe(tmp_path, method):
    original = capture(VERL / 'examples/grpo_trainer/run_webshop.sh', tmp_path / 'owner')
    current = capture(ROOT / 'experiments/rl/run_official_task.sh', tmp_path / 'project',
                      env={'VERL_ROOT': VERL.as_posix(), 'ENV_NAME': 'Webshop', 'METHOD': method})
    assert original[0] == current[0]  # Native data preparation is not replaced.
    expected = compose_verl(VERL, module_args(original, 'verl.trainer.main_ppo'))
    actual = compose_verl(VERL, module_args(current, 'verl.trainer.main_ppo'))
    expected['algorithm']['adv_estimator'] = 'deltatrace' if method == 'dt' else 'grpo'
    expected['actor_rollout_ref']['actor']['ppo_micro_batch_size_per_gpu'] = 4
    assert actual == expected  # No other train, validation, reward or history change.
    assert actual['actor_rollout_ref']['actor']['entropy_coeff'] == .001
    assert actual['actor_rollout_ref']['actor']['clip_ratio_c'] == 3.


def test_skyrl_sql_recipe_uses_native_config_components(tmp_path):
    from skyrl.train.config import TrainerConfig, GeneratorConfig, EnvironmentConfig
    calls = capture(SKYRL / 'examples/train/text_to_sql/run_skyrl_sql.sh', tmp_path)
    raw = OmegaConf.from_cli(module_args(calls, 'skyrl.train.entrypoints.main_base'))
    # Compose the owner's environment/model-entry configuration components;
    # this local scope does not initialize training backends.
    trainer = TrainerConfig.from_dict_config(raw.trainer)
    generator = GeneratorConfig.from_dict_config(raw.generator)
    environment = EnvironmentConfig.from_dict_config(raw.environment)
    assert environment.env_class == 'text2sql'
    assert generator.max_turns == 6
    assert trainer.max_prompt_length == 6000
    assert generator.max_input_length == 29000
    assert generator.sampling_params.max_generate_length == 3000
    assert generator.eval_sampling_params.max_generate_length == 3000
    assert generator.sampling_params.stop == generator.eval_sampling_params.stop
    assert generator.n_samples_per_prompt == 5
    assert trainer.epochs == 30
    assert trainer.train_batch_size == trainer.policy_mini_batch_size == 256
    assert trainer.eval_interval == 5 and trainer.eval_before_train


@pytest.mark.parametrize('task,rounds,epochs,n,mini,prompt,response', [
    ('textcraft', 30, 30, 8, 8, 512, 10240),
    ('webarena', 15, 25, 4, 4, 750, 14098),
])
def test_agentgym_original_train_and_eval_configs(tmp_path, task, rounds, epochs, n, mini, prompt, response):
    train_calls = capture(AGENTGYM / f'examples/train/AgentGym-RL/{task}_train.sh', tmp_path / 'train')
    # Original script's cwd defect is explicitly tested below. Here we inspect
    # its parameter expansion only, not successful evaluation execution.
    eval_calls = capture(AGENTGYM / f'examples/eval/{task}_eval.sh', tmp_path / 'eval', allow_directory_error=True)
    config_dir = 'AgentGym-RL/verl/agent_trainer/config'
    train = compose_verl(AGENTGYM, module_args(train_calls, 'verl.agent_trainer.main_ppo'), config_dir)
    evaluation = compose_verl(AGENTGYM, module_args(eval_calls, 'verl.agent_trainer.main_generation'), config_dir, 'generation')
    assert train['algorithm']['adv_estimator'] == 'grpo'
    assert train['algorithm']['rounds_ctrl'] == {'type': 'fixed', 'rounds': rounds, 'steps_scaling_inter': 100}
    assert train['data']['train_batch_size'] == 32
    assert (train['data']['max_prompt_length'], train['data']['max_response_length']) == (prompt, response)
    actor = train['actor_rollout_ref']['actor']
    assert (actor['ppo_mini_batch_size'], actor['ppo_epochs']) == (mini, 2)
    assert train['actor_rollout_ref']['rollout']['n'] == n
    assert train['trainer']['total_epochs'] == epochs
    assert evaluation['agentgym']['max_rounds'] == rounds
    assert (evaluation['data']['max_prompt_length'], evaluation['data']['max_response_length']) == (750, 14098)
    assert evaluation['data']['n_samples'] == 1
    assert evaluation['data']['batch_size'] == 32
    for rollout in [train['actor_rollout_ref']['rollout'], evaluation['rollout']]:
        assert rollout['max_model_len'] == 32768 and rollout['max_tokens'] == 512


@pytest.mark.parametrize('task', ['textcraft', 'webarena'])
def test_agentgym_eval_path_fix_keeps_all_native_parameters(tmp_path, task):
    from patch_agentgym_eval_paths import patch
    original = AGENTGYM / f'examples/eval/{task}_eval.sh'
    with pytest.raises(AssertionError, match='cd:'):
        capture(original, tmp_path / 'reproduce')
    before = capture(original, tmp_path / 'before', allow_directory_error=True)
    fixed = tmp_path / 'fixed.sh'
    source = original.read_text(encoding='utf-8')
    changed = patch(source)
    assert patch(changed) == changed
    fixed.write_text(changed, encoding='utf-8')
    after = capture(fixed, tmp_path / 'after')
    assert before[0][0] == 'model_merger.py' and after[0][0] == 'scripts/model_merger.py'
    assert before[0][1:] == after[0][1:]
    assert before[1:] == after[1:]


def test_sql_native_real_database_and_turn_limit(tmp_path):
    import sqlite3
    import skyrl_gym
    from skyrl_gym.envs.sql.env import Text2SQLEnvConfig
    db = tmp_path / 'spider/database/local/local.sqlite'
    db.parent.mkdir(parents=True)
    with sqlite3.connect(db) as conn:
        conn.execute('create table items (n integer)')
        conn.execute('insert into items values (7)')
    extras = dict(db_id='local', data='spider', max_turns=6,
                  reward_spec={'ground_truth': 'SELECT n FROM items'})
    cfg = Text2SQLEnvConfig(db_path=str(tmp_path))
    env = skyrl_gym.make('text2sql', env_config=cfg, extras=extras)
    query = '<think>Read table.</think><sql>SELECT n FROM items</sql>'
    out = env.step(query)
    assert out['reward'] == 0 and not out['done']
    assert '7' in out['observations'][0]['content']
    out = env.step('<think>Return answer.</think><solution>SELECT n FROM items</solution>')
    assert out['reward'] == 1 and out['done']
    env = skyrl_gym.make('text2sql', env_config=cfg, extras=extras)
    for turn in range(1, 7):
        out = env.step(query)
        assert out['done'] == (turn == 6)
        assert out['reward'] == (-1 if turn == 6 else 0)  # Owner penalizes invalid terminal format.

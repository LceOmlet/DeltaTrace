"""Existing VERL main, official SQL environment recipe, and verified model runtime.

This only assembles config and execs VERL. It implements no trainer, sampler,
rollout loop, loss, gradient accumulation or checkpoint manager. Bounded mode
is a labelled two-iteration interface run, with the full official six-turn and
3000-token limits; formal mode takes task/data budgets from the owner receipt.
"""
import argparse
import json
import os
from pathlib import Path
import sys
from owner_runtime_options import runtime_options, owner_command


def command(args):
    directory = Path(__file__).resolve().parent
    native = json.loads((directory / 'owner_environment_configs.json').read_text())['SkyRL-SQL']
    gen, training = native['generator'], native['trainer']
    method = 'deltatrace' if args.method == 'dt' else 'grpo'
    bounded = args.phase == 'bounded'
    train_batch = 2 if bounded else training['train_batch_size']
    data = Path(args.data)
    options = {**runtime_options(),
        'algorithm.adv_estimator': method,
        'data.train_files': str(data / 'train.parquet'),
        'data.val_files': str(data / 'validation.parquet'),
        'data.train_batch_size': train_batch,
        'data.val_batch_size': 2 if bounded else training['eval_batch_size'],
        'data.max_prompt_length': gen['max_input_length'],
        'data.max_response_length': gen['sampling_params']['max_generate_length'],
        'data.filter_overlong_prompts': True,
        'data.truncation': 'error', 'data.return_raw_chat': True,
        'data.custom_cls.path': str(directory / 'owner_task_dataset.py'),
        'data.custom_cls.name': 'SQLDataset',
        '+data.initial_max_prompt_length': training['max_prompt_length'],
        '+data.sql_chat_template': str(directory / 'qwen3_acc_thinking.jinja2'),
        '+data.dataloader_num_workers': 0,
        # SQL already owns format/outcome rewards; there is no extra parser flag.
        'actor_rollout_ref.actor.use_invalid_action_penalty': False,
        # SkyRL's mini-batch is measured in prompt groups. VERL receives one
        # native trajectory per row and rollout.n stays 1 at the engine API.
        'actor_rollout_ref.actor.ppo_mini_batch_size': (
            64 if bounded else training['policy_mini_batch_size'] * gen['n_samples_per_prompt']),
        'actor_rollout_ref.rollout.multi_turn.enable': True,
        'actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu': 1,
        'actor_rollout_ref.rollout.temperature': gen['sampling_params']['temperature'],
        'actor_rollout_ref.rollout.top_p': gen['sampling_params']['top_p'],
        'actor_rollout_ref.rollout.val_kwargs.temperature': gen['eval_sampling_params']['temperature'],
        'actor_rollout_ref.rollout.val_kwargs.do_sample': True,
        'env.env_name': 'SkyRL-SQL',
        'env.max_steps': gen['max_turns'],
        'env.rollout.n': gen['n_samples_per_prompt'],
        '+env.factory._target_': 'sql_owner_rollout.make_sql_owner_environments',
        '+env.sql': dict(db_path=str(data / 'db_files/data'),
            max_input_length=gen['max_input_length'], sampling=native['sampling'], eval_sampling=native['eval_sampling']),
        'trainer.logger': ['console'],
        'trainer.project_name': 'deltatrace_official_environments',
        'trainer.experiment_name': f'{args.method}-SkyRL-SQL',
        'trainer.default_local_dir': str(Path(args.output) / 'checkpoints'),
        'trainer.rollout_data_dir': str(Path(args.output) / 'rollouts'),
        'trainer.validation_data_dir': str(Path(args.output) / 'validation'),
        'trainer.max_actor_ckpt_to_keep': 2,
        'trainer.save_freq': 1 if bounded else training['ckpt_interval'],
        'trainer.test_freq': -1 if bounded else training['eval_interval'],
        'trainer.val_before_train': not bounded,
        'trainer.total_epochs': training['epochs'],
        'trainer.total_training_steps': 2 if bounded else None,
        'trainer.resume_mode': 'disable' if bounded else 'auto',
    }
    if bounded:
        options['+ray_init.runtime_env.worker_process_setup_hook'] = 'observe_vllm_boundary.install'
        options['+ray_init.runtime_env.env_vars.DT_VLLM_OBSERVE_DIR'] = str(Path(args.output))
    argv = owner_command(options)
    return argv, options


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=['dt', 'grpo'], required=True)
    parser.add_argument('--phase', choices=['bounded', 'formal'], required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--config-only', action='store_true')
    args = parser.parse_args()
    argv, options = command(args)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'launch.json').write_text(json.dumps(dict(argv=argv, options=options,
        phase=args.phase, numerical_runtime='c9cd147/fc2e6c2',
        framework='VERL-agent 20bd331 distributed DT with environment seams',
        workload_note='Native SkyRL complete trajectory per VERL row. Formal 256 prompt groups x 5 trajectories; global mini-batch 1280 trajectories, one native VERL optimizer update per complete batch.'), indent=2)+'\n')
    os.environ.update(DT_TASK='SkyRL-SQL', DT_MAX_STEPS='6', DT_MAX_LENGTH='32768')
    os.environ['DT_SAMPLING_JSON'] = json.dumps(options['+env.sql']['sampling'])
    if args.config_only:
        argv += ['--cfg', 'job', '--resolve']
    os.execv(sys.executable, argv)

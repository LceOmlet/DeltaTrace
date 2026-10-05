"""Run existing VERL with the author's TextCraft workload and original rollout."""
import argparse
import json
import os
from pathlib import Path
import sys
from owner_runtime_options import runtime_options, owner_command


def options_for(data, output, *, resume_from=None):
    root = Path(__file__).resolve().parent
    native = json.loads((root/'owner_environment_configs.json').read_text())['TextCraft']
    train, evaluation = native['train'], native['eval']
    actor, rollout = train['actor_rollout_ref']['actor'], train['actor_rollout_ref']['rollout']
    sampling = {k: rollout[k] for k in ('temperature', 'top_k', 'top_p', 'max_tokens', 'ignore_eos')}
    options = {**runtime_options(),
        'algorithm.adv_estimator': 'deltatrace',
        'data.train_files': str(data/'train.parquet'), 'data.val_files': str(data/'validation.parquet'),
        'data.train_batch_size': train['data']['train_batch_size'],
        'data.val_batch_size': evaluation['data']['batch_size'],
        # Per-response DT transport, not the native trajectory's training cap.
        'data.max_prompt_length': 32768-rollout['max_tokens'],
        'data.max_response_length': rollout['max_tokens'],
        'data.return_raw_chat': True, 'data.truncation': 'error', '+data.dataloader_num_workers': 0,
        'actor_rollout_ref.actor.use_invalid_action_penalty': False,
        'actor_rollout_ref.actor.ppo_mini_batch_size': actor['ppo_mini_batch_size']*rollout['n'],
        # Author's actual update_policy iterates each minibatch once. Its
        # ppo_epochs=2 config is used only in MFU reporting, not that loop.
        'actor_rollout_ref.actor.ppo_epochs': 1,
        'actor_rollout_ref.actor.optim.lr': actor['optim']['lr'],
        'actor_rollout_ref.actor.use_kl_loss': actor['use_kl_loss'],
        'actor_rollout_ref.actor.kl_loss_coef': actor['kl_loss_coef'],
        'actor_rollout_ref.actor.kl_loss_type': actor['kl_loss_type'],
        'actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu': 4,
        'actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu': 4,
        'actor_rollout_ref.rollout.multi_turn.enable': True,
        'actor_rollout_ref.rollout.temperature': rollout['temperature'],
        'actor_rollout_ref.rollout.top_p': rollout['top_p'],
        'actor_rollout_ref.rollout.val_kwargs.do_sample': evaluation['rollout']['do_sample'],
        'actor_rollout_ref.rollout.val_kwargs.temperature': evaluation['rollout']['temperature'],
        'actor_rollout_ref.rollout.val_kwargs.top_p': evaluation['rollout']['top_p'],
        'env.env_name': 'TextCraft', 'env.rollout.n': rollout['n'],
        'env.max_steps': train['algorithm']['rounds_ctrl']['rounds'],
        '+env.factory._target_': 'textcraft_environment_entry.make_textcraft_environments',
        '+env.textcraft': dict(owner_root=os.environ['AGENTGYM_RL_ROOT'],
            client=train['actor_rollout_ref']['agentgym'], eval_client=evaluation['agentgym']),
        'trainer.logger': ['console'], 'trainer.project_name': 'deltatrace_official_environments',
        'trainer.experiment_name': 'dt-TextCraft',
        'trainer.default_local_dir': str(output/'checkpoints'),
        'trainer.rollout_data_dir': str(output/'rollouts'),
        'trainer.validation_data_dir': str(output/'validation'),
        'trainer.max_actor_ckpt_to_keep': 2,
        'trainer.total_epochs': train['trainer']['total_epochs'], 'trainer.total_training_steps': None,
        'trainer.save_freq': train['trainer']['save_freq'], 'trainer.test_freq': train['trainer']['test_freq'],
        # The author's training script does not run its separate eval program.
        'trainer.val_before_train': False, 'trainer.resume_mode': 'auto'}
    if resume_from is not None:
        options.update({'trainer.resume_mode': 'resume_path',
                        'trainer.resume_from_path': str(resume_from)})
    return options, sampling


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config-only', action='store_true')
    parser.add_argument('--resume-from', type=Path,
                        help='Pass a completed global_step directory to the original VERL loader.')
    args = parser.parse_args()
    options, sampling = options_for(args.data, args.output, resume_from=args.resume_from)
    argv = owner_command(options)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'launch.json').write_text(json.dumps(dict(argv=argv, options=options,
        source='AgentGym-RL 82402a9 examples/train/AgentGym-RL/textcraft_train.sh',
        workload='32 groups x 8 trajectories; 64 trajectories per minibatch, 4 updates per full batch',
        numerical_runtime='c9cd147/fc2e6c2'), indent=2)+'\n')
    os.environ.update(DT_TASK='TextCraft', DT_MAX_STEPS=str(options['env.max_steps']),
        DT_MAX_LENGTH='32768', DT_SAMPLING_JSON=json.dumps(sampling))
    if args.config_only:
        argv += ['--cfg', 'job', '--resolve']
    os.execv(sys.executable, argv)

"""LOOP's official environment workload, executed with existing VERL/vLLM."""
import argparse
import json
import os
from pathlib import Path
import site
import sys
from owner_runtime_options import runtime_options, owner_command


def options_for(output, resume_from=None):
    if os.environ.get('LOOP_EXTRAS'):
        site.addsitedir(os.environ['LOOP_EXTRAS'])
    from loop_owner_recipe import compose
    from omegaconf import OmegaConf
    cfg = compose(Path(os.environ['LOOP_ROOT']), [])
    evaluation = OmegaConf.merge(cfg, cfg.rl.eval.overrides)
    options = {**runtime_options(),
        'algorithm.adv_estimator': 'deltatrace',
        'actor_rollout_ref.rollout.mode': 'async',
        'data.train_files': 'loop-training', 'data.val_files': 'loop-evaluation',
        'data.custom_cls.path': str(Path(__file__).with_name('loop_iteration_dataset.py')),
        'data.custom_cls.name': 'LoopIterationDataset',
        '+data.loop_train_groups': cfg.rl.params.scenarios_per_iteration,
        '+data.loop_eval_groups': evaluation.rl.params.scenarios_per_iteration,
        'data.train_batch_size': cfg.rl.params.scenarios_per_iteration,
        'data.val_batch_size': evaluation.rl.params.scenarios_per_iteration,
        'data.max_prompt_length': cfg.llm.vllm_server.max_model_len,
        'data.max_response_length': cfg.llm.vllm_class.max_new_tokens,
        'data.return_raw_chat': True, 'data.truncation': 'error', '+data.dataloader_num_workers': 0,
        'actor_rollout_ref.actor.use_invalid_action_penalty': False,
        'actor_rollout_ref.actor.ppo_mini_batch_size': cfg.rl.params.minibatch_size,
        'actor_rollout_ref.actor.ppo_epochs': cfg.rl.params.epochs_per_iteration,
        # Full-trajectory B4 uses the original VERL chunked vocabulary head.
        'actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu': 4,
        'actor_rollout_ref.rollout.multi_turn.enable': True,
        'actor_rollout_ref.rollout.prompt_length': cfg.llm.vllm_server.max_model_len-cfg.llm.vllm_class.max_new_tokens,
        'actor_rollout_ref.rollout.temperature': cfg.llm.temperature,
        'actor_rollout_ref.rollout.val_kwargs.temperature': evaluation.llm.temperature,
        'actor_rollout_ref.rollout.val_kwargs.do_sample': False,
        'env.env_name': 'AppWorld', 'env.rollout.n': cfg.rl.params.rollouts_per_scenario,
        'env.max_steps': cfg.rl.scenario_runner.appworld_config.env.max_interactions,
        '+env.factory._target_': 'loop_environment_entry.make_loop_environments',
        '+env.loop': dict(owner_root=os.environ['LOOP_ROOT']),
        'trainer.logger': ['console'], 'trainer.project_name': 'deltatrace_official_environments',
        'trainer.experiment_name': 'dt-AppWorld',
        'trainer.default_local_dir': str(output/'checkpoints'),
        'trainer.rollout_data_dir': str(output/'rollouts'),
        'trainer.validation_data_dir': str(output/'validation'),
        'trainer.max_actor_ckpt_to_keep': 2,
        'trainer.total_epochs': cfg.rl.params.total_iterations,
        'trainer.total_training_steps': cfg.rl.params.total_iterations,
        'trainer.save_freq': 1, 'trainer.test_freq': cfg.rl.eval.eval_every_n_iterations,
        'trainer.val_before_train': False, 'trainer.resume_mode': 'auto'}
    options.update({'+ray_init.runtime_env.worker_process_setup_hook': 'observe_worker_visibility.install',
        '+ray_init.runtime_env.env_vars.DT_WORKER_VISIBILITY_DIR': str(output/'worker-visibility')})
    if resume_from is not None:
        options.update({'trainer.resume_mode': 'resume_path',
                        'trainer.resume_from_path': str(resume_from)})
    sampling = dict(temperature=cfg.llm.temperature, max_tokens=cfg.llm.vllm_class.max_new_tokens)
    return options, sampling


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config-only', action='store_true')
    parser.add_argument('--resume-from', type=Path,
                        help='Pass a completed global_step directory to the original VERL checkpoint loader.')
    args = parser.parse_args()
    options, sampling = options_for(args.output, resume_from=args.resume_from)
    argv = owner_command(options)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'launch.json').write_text(json.dumps(dict(argv=argv, options=options,
        source='LOOP f14107a README Model Training; environment workload only',
        numerical_runtime='c9cd147/fc2e6c2',
        scope='Original LOOP sampler/pool/cancellation/env/client/reward; original VERL optimizer/loss/update/checkpoint'),
        indent=2)+'\n')
    os.environ.update(DT_TASK='AppWorld', DT_MAX_STEPS=str(options['env.max_steps']),
        DT_MAX_LENGTH='32768', DT_SAMPLING_JSON=json.dumps(sampling))
    if args.config_only:
        argv += ['--cfg', 'job', '--resolve']
    # MetaX's native library can set MACA_VISIBLE_DEVICES in the C environment
    # while composing LOOP config, without updating Python's os.environ. execv
    # inherits that hidden two-card mask and breaks Ray's one-card workers.
    # Pass an explicit environment, preserving Ray's native CUDA assignment.
    environment = dict(os.environ)
    environment.pop('MACA_VISIBLE_DEVICES', None)
    os.execve(sys.executable, argv, environment)

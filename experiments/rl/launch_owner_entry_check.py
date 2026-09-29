"""Two-iteration native VERL checks of the TextCraft/LOOP environment seams.

Only the number of sampled task groups and training iterations is bounded.
Native interaction, sampling, parser, reward and evaluation settings are kept.
This is not a formal experiment budget or another trainer.
"""
import argparse
import json
import os
from pathlib import Path
import sys
from owner_runtime_options import runtime_options, owner_command


def options_for(task, data, output):
    root=Path(__file__).resolve().parent
    native=json.loads((root/'owner_environment_configs.json').read_text())[task]
    options={**runtime_options(),
        'algorithm.adv_estimator':'deltatrace',
        'data.train_files':str(data/'train.parquet'),
        'data.val_files':str(data/'validation.parquet'),
        'data.train_batch_size':2, 'data.val_batch_size':2,
        'data.return_raw_chat':True,'data.truncation':'error',
        'data.filter_overlong_prompts':True,'+data.dataloader_num_workers':0,
        'actor_rollout_ref.actor.use_invalid_action_penalty':False,
        'env.env_name':task,
        'trainer.logger':['console'], 'trainer.project_name':'deltatrace_official_environments',
        'trainer.experiment_name':'dt-'+task+'-bounded',
        'trainer.default_local_dir':str(output/'checkpoints'),
        'trainer.rollout_data_dir':str(output/'rollouts'),
        'trainer.total_training_steps':2,'trainer.total_epochs':2,
        'trainer.save_freq':1,'trainer.test_freq':-1,'trainer.val_before_train':False,
        'trainer.resume_mode':'disable',
        '+ray_init.runtime_env.worker_process_setup_hook':'observe_vllm_boundary.install',
        '+ray_init.runtime_env.env_vars.DT_VLLM_OBSERVE_DIR':str(output)}
    if task=='TextCraft':
        a=native['train']['actor_rollout_ref']
        sampling={k:a['rollout'][k] for k in ('temperature','top_k','top_p','max_tokens','ignore_eos')}
        ev=native['eval']
        eval_sampling={k:ev['rollout'][k] for k in sampling}
        options.update({
            'data.max_prompt_length':32768-sampling['max_tokens'],
            'data.max_response_length':sampling['max_tokens'],
            'env.max_steps':native['train']['algorithm']['rounds_ctrl']['rounds'],
            'env.rollout.n':a['rollout']['n'],
            '+env.factory._target_':'textcraft_environment_entry.make_textcraft_environments',
            '+env.textcraft':dict(owner_root=os.environ['AGENTGYM_RL_ROOT'],client=a['agentgym'],
                eval_client=ev['agentgym'],sampling=sampling,eval_sampling=eval_sampling)})
    else:
        b=native['generation_boundary_reference']
        sampling=dict(temperature=b['training_temperature'],max_tokens=b['client']['max_new_tokens'])
        options.update({
            # Native LOOP supplies per-request max_tokens against its own cap.
            # The dataset's ID carrier is never used as an LLM prompt.
            'data.max_prompt_length':b['max_model_len'],
            'data.max_response_length':b['client']['max_new_tokens'],
            'actor_rollout_ref.rollout.prompt_length':b['max_model_len']-b['client']['max_new_tokens'],
            'env.max_steps':native['training_environment']['appworld_config']['env']['max_interactions'],
            'env.rollout.n':4,
            '+env.factory._target_':'loop_environment_entry.make_loop_environments',
            '+env.loop':dict(owner_root=os.environ['LOOP_ROOT'])})
    options.update({'actor_rollout_ref.rollout.temperature':sampling['temperature'],
        'actor_rollout_ref.rollout.top_p':sampling.get('top_p',1.0)})
    return options,sampling


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',choices=['TextCraft','AppWorld'],required=True)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--config-only',action='store_true')
    a=p.parse_args()
    options,sampling=options_for(a.task,a.data,a.output)
    argv=owner_command(options)
    a.output.mkdir(exist_ok=True,parents=True)
    (a.output/'launch.json').write_text(json.dumps(dict(argv=argv,options=options,
        scope=__doc__,numerical_runtime='c9cd147/fc2e6c2'),indent=2))
    os.environ.update(DT_TASK=a.task,DT_MAX_STEPS=str(options['env.max_steps']),DT_MAX_LENGTH='32768',
        DT_SAMPLING_JSON=json.dumps(sampling))
    if a.config_only:argv+=['--cfg','job','--resolve']
    os.execv(sys.executable,argv)

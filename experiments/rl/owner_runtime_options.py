"""Resource overrides for the already verified VERL/vLLM model runtime.

This is configuration only. Loss, optimizer, accumulation, scheduling and
checkpointing remain owned by the installed VERL. The long-response capacity
check selects microbatch 1 after a reproduced B4 vocabulary-head OOM; DT stays 4.
"""
import os


def runtime_options():
    return {
        'ray_init.num_cpus': 16,
        'actor_rollout_ref.model.path': os.environ['MODEL_PATH'],
        'actor_rollout_ref.model.trust_remote_code': True,
        'actor_rollout_ref.model.lora_rank': 8,
        'actor_rollout_ref.model.lora_alpha': 16,
        'actor_rollout_ref.model.enable_gradient_checkpointing': True,
        'actor_rollout_ref.model.enable_activation_offload': True,
        'actor_rollout_ref.actor.strategy': 'fsdp2',
        'actor_rollout_ref.actor.ppo_mini_batch_size': 64,
        'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu': 1,
        'actor_rollout_ref.actor.ppo_max_token_len_per_gpu': 32768,
        'actor_rollout_ref.actor.use_torch_compile': False,
        'actor_rollout_ref.actor.fsdp_config.offload_policy': True,
        '+actor_rollout_ref.actor.fsdp_config.model_dtype': 'bfloat16',
        'actor_rollout_ref.actor.fsdp_config.param_offload': True,
        'actor_rollout_ref.actor.fsdp_config.optimizer_offload': True,
        'actor_rollout_ref.rollout.name': 'vllm',
        'actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu': 4,
        'actor_rollout_ref.rollout.tensor_model_parallel_size': 1,
        'actor_rollout_ref.rollout.load_format': 'safetensors',
        'actor_rollout_ref.rollout.gpu_memory_utilization': .75,
        'actor_rollout_ref.rollout.max_model_len': 32768,
        'actor_rollout_ref.rollout.max_num_seqs': 32,
        'actor_rollout_ref.rollout.max_num_batched_tokens': 32768,
        '+actor_rollout_ref.rollout.engine_kwargs.vllm.limit_mm_per_prompt': {'image': 0, 'video': 0},
        'trainer.n_gpus_per_node': 2, 'trainer.nnodes': 1,
    }


def owner_command(options):
    import json
    import sys
    def hydra_value(value):
        if isinstance(value, dict):
            return '{' + ','.join(f'{k}:{hydra_value(v)}' for k,v in value.items()) + '}'
        return json.dumps(value, separators=(',', ':'))
    return [sys.executable, '-m', 'verl.trainer.main_ppo'] + [
        f'{key}={hydra_value(value)}' for key,value in options.items()]

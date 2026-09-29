"""Bounded capacity of the unchanged VERL actor with sleeping native vLLM.

An explicit synthetic 32768-token input and 3000 response tokens exercise the
previously failing vocabulary head/backward. Fixed test advantages are not DT
outputs or task rewards. All log-probability, accumulation and updates are the
actual installed VERL worker. No numerical implementation is replaced.
"""
import json
import os
from pathlib import Path
import time

import numpy as np
import ray
import torch
from transformers import AutoTokenizer
from omegaconf import OmegaConf
from verl import DataProto
from verl.single_controller.base.decorator import Dispatch, register
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.utils.model import compute_position_id_with_mask
from verl.workers.fsdp_workers import ActorRolloutRefWorker


@ray.remote
class CapacityWorker(ActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def sync_and_measure(self):
        with self.rollout_sharding_manager:
            pass
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        return dict(rank=self.rank, physical_used_bytes=total-free)


if __name__ == '__main__':
    from launch_sql_native import command
    from types import SimpleNamespace
    from hydra import initialize_config_dir, compose
    out = Path(os.environ['LONG_RESPONSE_OUTPUT'])
    out.mkdir(exist_ok=True)
    args = SimpleNamespace(method='grpo', phase='bounded', data=os.environ['SQL_DATA'], output=str(out))
    argv, _ = command(args)
    # Same runtime; only the owner's supported accumulation geometry changes.
    with initialize_config_dir(config_dir=str(Path(os.environ['VERL_ROOT'])/'verl/trainer/config'), version_base=None):
        config = compose(config_name='ppo_trainer', overrides=argv[3:])
    c = config.actor_rollout_ref
    c.actor.ppo_mini_batch_size = 8
    c.actor.ppo_micro_batch_size_per_gpu = 1
    c.actor.optim.total_training_steps = 2
    c.rollout.log_prob_micro_batch_size_per_gpu = 1
    result = dict(scope=__doc__, input_tokens=32768, response_tokens=3000,
        global_minibatch=8, per_gpu_microbatch=1, dt_invoked=False, stages=[])
    start = time.monotonic()
    def record(phase, **values):
        result['stages'].append(dict(phase=phase, seconds=time.monotonic()-start, **values))
        (out/'result.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result['stages'][-1]), flush=True)
    ray.init(num_cpus=8, include_dashboard=False)
    try:
        group = RayWorkerGroup(RayResourcePool([2], use_gpu=True, max_colocate_count=1),
            RayClassWithInitArgs(CapacityWorker, c, 'actor_rollout'))
        group.init_model()
        record('model_ready', resources=group.sync_and_measure())
        tok = AutoTokenizer.from_pretrained(os.environ['MODEL_PATH'], local_files_only=True)
        unit = tok.encode(' SELECT name FROM customers WHERE id = 1; ', add_special_tokens=False)
        ids = torch.tensor((unit*(32768//len(unit)+1))[:32768]).repeat(8,1)
        mask = torch.ones_like(ids)
        data = DataProto.from_dict(tensors=dict(input_ids=ids, attention_mask=mask,
            position_ids=compute_position_id_with_mask(mask), prompts=ids[:,:-3000], responses=ids[:,-3000:],
            response_mask=mask[:,-3000:], advantages=torch.linspace(-.1,.1,3000).repeat(8,1)),
            meta_info=dict(temperature=.6, global_token_num=mask.sum(-1).tolist()))
        record('old_logprob_start')
        data.batch['old_log_probs'] = group.compute_log_prob(data).batch['old_log_probs']
        record('old_logprob_done')
        for index in range(2):
            record('update_start', index=index)
            metrics=group.update_actor(data).meta_info['metrics']
            assert all(np.isfinite(x) and x>0 for x in metrics['actor/grad_norm']), metrics
            record('update_done', index=index, metrics=metrics)
        record('native_sync_after_updates', resources=group.sync_and_measure())
        result['status']='passed_capacity_not_numerical_parity'
    except Exception as error:
        result.update(status='failed', error=repr(error))
        raise
    finally:
        record('finished')
        ray.shutdown()

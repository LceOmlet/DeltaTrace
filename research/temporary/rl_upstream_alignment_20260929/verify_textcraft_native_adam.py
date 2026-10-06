"""Observe one real original VERL update from the same complete checkpoint.

The saved original global64 update boundary supplies IDs, masks, rewards,
old/ref log probabilities and DT/official GRPO advantages. Three independent
checkpoint restores compare their advantages and a zero-PG regularizer control.
No rollout, DT recomputation, alternate loss, optimizer or production patch.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import ray
import torch
from omegaconf import OmegaConf
from verl.protocol import DataProto
from verl.single_controller.base.decorator import Dispatch, register
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.workers.fsdp_workers import ActorRolloutRefWorker

OUT = Path(os.environ['DT_TEXTCRAFT_READOUT_ROOT'])
BASE = Path(os.environ['DT_TEXTCRAFT_READOUT_BASE'])
CHECKPOINT = Path(os.environ['DT_TEXTCRAFT_CHECKPOINT']) / 'actor'
INPUT = BASE / 'native-optimizer-minibatch.pkl'
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE.parent / 'textcraft-native-readout-20261006-v2'))
from observe_textcraft_native_batches import _snapshot
from verify_textcraft_native_readout import identity, resources
from observe_native_adam_update import observe_saved_native_adam_update

BRANCHES = ('dt', 'grpo', 'regularizers_only')


@ray.remote
class AdamObservationWorker(ActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def observe_update(self, data):
        return observe_saved_native_adam_update(self, data,
            original_update_actor=super().update_actor,
            output_directory=OUT, branch=data.meta_info['native_adam_branch'])


def save_artifact(name, data):
    path = OUT / name
    _snapshot(data).save_to_disk(str(path))
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size)


def phase(label, **details):
    value = dict(phase=label, pid=os.getpid(), observed_unix=time.time(), **details)
    (OUT / 'driver-phase.json').write_text(json.dumps(value, indent=2) + '\n')
    print('TEXTCRAFT_NATIVE_ADAM ' + json.dumps(value), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inspect-only', action='store_true')
    args = parser.parse_args()
    original_boundary = json.loads((BASE / 'native-minibatch-update-boundary.json').read_bytes())
    input_sha = hashlib.sha256(INPUT.read_bytes()).hexdigest()
    assert input_sha == original_boundary['snapshot']['sha256']
    saved = DataProto.load_from_disk(str(INPUT))
    assert len(saved) == 64
    assert saved.batch['advantages'].shape == saved.batch['diagnostic_grpo_advantages'].shape
    cfg = OmegaConf.load(Path(os.environ['VERL_ROOT']) / 'verl/trainer/config/ppo_trainer.yaml')
    launch = json.loads((BASE / 'launch.json').read_bytes())
    for key, value in launch['options'].items():
        OmegaConf.update(cfg, key.lstrip('+'), value, force_add=True)
    cfg.actor_rollout_ref.actor.optim.total_training_steps = 330
    assert cfg.actor_rollout_ref.model.lora_rank == 8 and cfg.actor_rollout_ref.model.lora_alpha == 16
    assert cfg.actor_rollout_ref.actor.ppo_mini_batch_size == 64
    assert cfg.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu == 4
    assert cfg.actor_rollout_ref.actor.entropy_coeff == .001
    assert cfg.actor_rollout_ref.actor.kl_loss_coef == .001
    from verl.workers.actor.dp_actor import DataParallelPPOActor
    from verl.trainer.ppo.core_algos import compute_policy_loss
    owner_sources = dict(worker=identity(ActorRolloutRefWorker),
        worker_constructor=identity(ActorRolloutRefWorker.__init__),
        worker_model_loader=identity(ActorRolloutRefWorker.init_model),
        worker_checkpoint_loader=identity(ActorRolloutRefWorker.load_checkpoint),
        worker_update=identity(ActorRolloutRefWorker.update_actor),
        worker_log_probs=identity(ActorRolloutRefWorker.compute_log_prob),
        actor=identity(DataParallelPPOActor), core=identity(compute_policy_loss),
        observer=identity(observe_saved_native_adam_update),
        snapshot=identity(_snapshot), harness=identity(phase))
    assert owner_sources['actor']['sha256'] == '1f862e8bbdaad6fa116d0670772ad41269529a3a1e4a5b1eb383352d0372e9bd'
    assert owner_sources['core']['sha256'] == 'fc2f992b16fb7fb23aebc683ad5f00136cf61fc9013cd426f5046983babe7299'
    (OUT / 'effective-config.yaml').write_text(OmegaConf.to_yaml(cfg))
    inspection = dict(scope='Original saved DataProto and owner/config/serialization only; no model created.',
        native_reader_inputs_sha256=input_sha, rows=len(saved), branches=list(BRANCHES),
        cuda_initialized=torch.cuda.is_initialized(), worker_constructor=owner_sources['worker_constructor'],
        sources=owner_sources, checkpoint=str(CHECKPOINT),
        actor_config=OmegaConf.to_container(cfg.actor_rollout_ref, resolve=True),
        fields={k:dict(shape=list(v.shape), dtype=str(v.dtype)) for k,v in saved.batch.items()},
        saved_old_ref_source='Unchanged saved original trainer compute_log_prob/ref output; measurement results never overwrite these fields.')
    import cloudpickle
    method = AdamObservationWorker.__ray_metadata__.modified_class.observe_update
    assert identity(cloudpickle.loads(cloudpickle.dumps(method))) == identity(method)
    (OUT / 'native-owner-inspection.json').write_text(json.dumps(inspection, indent=2) + '\n')
    if args.inspect_only:
        print(json.dumps(dict(status='inspected_no_model', rows=len(saved), cuda_initialized=inspection['cuda_initialized'])))
    else:
        ray.init(num_cpus=8, include_dashboard=False)
        results = []
        try:
            group = RayWorkerGroup(RayResourcePool([2], use_gpu=True, max_colocate_count=1),
                RayClassWithInitArgs(AdamObservationWorker, cfg.actor_rollout_ref, 'actor'))
            phase('original_model_init_begin')
            group.init_model()
            phase('original_model_init_complete')
            for branch in BRANCHES:
                phase('original_complete_checkpoint_restore_begin', branch=branch)
                group.load_checkpoint(local_path=str(CHECKPOINT), del_local_after_load=False)
                phase('original_complete_checkpoint_restore_complete', branch=branch)
                update_data = _snapshot(saved)
                if branch == 'grpo':
                    update_data.batch['advantages'] = saved.batch['diagnostic_grpo_advantages']
                elif branch == 'regularizers_only':
                    update_data.batch['advantages'] = torch.zeros_like(saved.batch['advantages'])
                update_data.meta_info['native_adam_branch'] = branch
                phase('original_before_log_prob_begin', branch=branch)
                before = group.compute_log_prob(_snapshot(update_data))
                before_artifact = save_artifact(f'{branch}-before-logprob.pkl', before)
                del before
                phase('original_before_log_prob_complete', branch=branch, artifact=before_artifact)
                phase('original_update_actor_begin', branch=branch)
                update_output = group.observe_update(update_data)
                update_artifact = save_artifact(f'{branch}-update-output.pkl', update_output)
                del update_output
                phase('original_update_actor_complete', branch=branch, artifact=update_artifact)
                phase('original_after_log_prob_begin', branch=branch)
                after = group.compute_log_prob(_snapshot(update_data))
                after_artifact = save_artifact(f'{branch}-after-logprob.pkl', after)
                del after
                phase('original_after_log_prob_complete', branch=branch, artifact=after_artifact)
                results.append(dict(branch=branch, before=before_artifact, update=update_artifact, after=after_artifact))
                del update_data
            value = dict(completed_unix=time.time(), branches=results, global_rows=64,
                source_minibatch_sha256=input_sha, sources=owner_sources,
                checkpoint=str(CHECKPOINT), sampling_calls=0, DT_calls=0,
                scope='Three isolated single original optimizer updates after three complete original checkpoint restores; not production training or a numerical tolerance gate.')
            (OUT / 'completed.json').write_text(json.dumps(value, indent=2) + '\n')
            phase('complete_native_adam_observation')
        finally:
            ray.shutdown()

"""Prepared isolated native support-gradient observation; no rollout or DT.

Compose the completed native Adam recipe's official imports and saved effective
configuration. Its real-Adam observer is never called. The existing original
minibatch observer owns all scalar hooks, backwards, native clip and no-op step.
"""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time

ADAM = Path(os.environ['DT_TEXTCRAFT_ADAM_RECIPE_ROOT'])
sys.path.insert(0, str(ADAM))
import verify_textcraft_native_adam as recipe
from observe_textcraft_grpo_support_gradients import SUPPORTS, support_snapshot

OUT, BASE, CHECKPOINT, INPUT = recipe.OUT, recipe.BASE, recipe.CHECKPOINT, recipe.INPUT


@recipe.ray.remote
class SupportGradientWorker(recipe.ActorRolloutRefWorker):
    @recipe.register(dispatch_mode=recipe.Dispatch.DP_COMPUTE_PROTO)
    def observe_support(self, data):
        # Import at the actual worker call site; no serialized callback-global patch.
        from observe_textcraft_grpo_support_gradients import observe_saved_grpo_support
        return observe_saved_grpo_support(self, data,
            original_update_actor=super().update_actor, output_directory=OUT)


def inspect_inputs():
    import cloudpickle
    import psutil
    from verl.workers.actor.dp_actor import DataParallelPPOActor
    from verl.trainer.ppo.core_algos import compute_policy_loss

    started = time.monotonic()
    boundary = json.loads((BASE / 'native-minibatch-update-boundary.json').read_bytes())
    input_sha = hashlib.sha256(INPUT.read_bytes()).hexdigest()
    assert input_sha == boundary['snapshot']['sha256']
    saved = recipe.DataProto.load_from_disk(str(INPUT))
    assert len(saved) == 64
    cfg_path = ADAM / 'effective-config.yaml'
    cfg = recipe.OmegaConf.load(cfg_path)
    prior = json.loads((ADAM / 'native-owner-inspection.json').read_bytes())
    assert recipe.OmegaConf.to_container(cfg.actor_rollout_ref, resolve=True) == prior['actor_config']
    assert str(CHECKPOINT) == prior['checkpoint']
    original = importlib.import_module('observe_native_optimizer_minibatch')
    adapter = importlib.import_module('observe_textcraft_grpo_support_gradients')
    sources = dict(worker_constructor=recipe.identity(recipe.ActorRolloutRefWorker.__init__),
        worker_update=recipe.identity(recipe.ActorRolloutRefWorker.update_actor),
        worker_checkpoint_loader=recipe.identity(recipe.ActorRolloutRefWorker.load_checkpoint),
        actor=recipe.identity(DataParallelPPOActor), core=recipe.identity(compute_policy_loss),
        original_observer=recipe.identity(original.observe_native_optimizer_minibatch),
        original_loss_hooks=recipe.identity(original.NativeLossHooks),
        original_gradient_statistics=recipe.identity(original.native_gradient_statistics),
        snapshot=recipe.identity(recipe._snapshot), recipe=recipe.identity(recipe.phase),
        adapter=recipe.identity(adapter.observe_saved_grpo_support), driver=recipe.identity(inspect_inputs))
    for name in ('worker_constructor', 'worker_update', 'worker_checkpoint_loader', 'actor', 'core', 'snapshot'):
        assert sources[name] == prior['sources'][name]
    assert sources['original_observer']['sha256'] == '022466b2bac94617b8e627ea027fb3afdf4490dc4bc2bcaa11944ab444e36214'
    assert original.observe_native_optimizer_minibatch.__globals__ is vars(original)
    support_counts = {}
    for support in SUPPORTS:
        selected = support_snapshot(saved, support)
        for key, tensor in saved.batch.items():
            if key != 'diagnostic_grpo_advantages':
                assert recipe.torch.equal(tensor, selected.batch[key]), key
        keep = saved.batch['dt_q_estimates'] != 0 if support == 'q_nonzero' else saved.batch['dt_q_estimates'] == 0
        assert recipe.torch.equal(selected.batch['diagnostic_grpo_advantages'][keep],
                                  saved.batch['diagnostic_grpo_advantages'][keep])
        assert not recipe.torch.count_nonzero(selected.batch['diagnostic_grpo_advantages'][~keep]).item()
        support_counts[support] = int(keep.sum().item())
    method = SupportGradientWorker.__ray_metadata__.modified_class.observe_support
    assert recipe.identity(cloudpickle.loads(cloudpickle.dumps(method))) == recipe.identity(method)
    value = dict(scope='Prepared original saved DataProto/config/source and serialized method inspection; no model/GPU/backward.',
        native_reader_inputs_sha256=input_sha, rows=len(saved), supports=list(SUPPORTS), labels=['dt_pg', 'grpo_pg'],
        worker_constructor=sources['worker_constructor'], sources=sources, checkpoint=str(CHECKPOINT),
        config_source=dict(path=str(cfg_path), sha256=hashlib.sha256(cfg_path.read_bytes()).hexdigest()),
        actor_config=prior['actor_config'], support_counts_including_padding=support_counts,
        full_DT_advantages_and_other_native_tensors_unchanged=True,
        loss_scope='Original global64 dispatch to local32, B4 x8 and /8; full original loss_mask retained.',
        original_observer_module_globals_identity=True, cuda_initialized=recipe.torch.cuda.is_initialized(),
        distributed_initialized=recipe.torch.distributed.is_initialized(), pid=os.getpid(), pid_birth=psutil.Process().create_time(),
        rss_bytes=psutil.Process().memory_info().rss, elapsed_seconds=time.monotonic()-started,
        sampling_calls=0, DT_calls=0, backward_calls=0, optimizer_steps=0, scheduler_steps=0)
    (OUT / 'native-owner-inspection.json').write_text(json.dumps(value, indent=2) + '\n')
    return saved, cfg, value


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inspect-only', action='store_true')
    args = parser.parse_args()
    saved, cfg, inspection = inspect_inputs()
    if args.inspect_only:
        print(json.dumps(dict(status='inspected_no_model', rows=len(saved), cuda_initialized=inspection['cuda_initialized'])))
    else:
        recipe.ray.init(num_cpus=8, include_dashboard=False)
        try:
            group = recipe.RayWorkerGroup(recipe.RayResourcePool([2], use_gpu=True, max_colocate_count=1),
                recipe.RayClassWithInitArgs(SupportGradientWorker, cfg.actor_rollout_ref, 'actor'))
            recipe.phase('original_model_init_begin')
            group.init_model()
            recipe.phase('original_complete_checkpoint_restore_begin')
            group.load_checkpoint(local_path=str(CHECKPOINT), del_local_after_load=False)
            recipe.phase('original_complete_checkpoint_restore_complete')
            for support in SUPPORTS:
                data = support_snapshot(saved, support)
                recipe.phase('original_support_gradient_begin', support=support)
                output = group.observe_support(data)
                recipe.save_artifact(f'{support}-original-worker-output.pkl', output)
                recipe.phase('original_support_gradient_complete', support=support)
            (OUT / 'completed.json').write_text(json.dumps(dict(sources=inspection['sources'],
                input_sha256=inspection['native_reader_inputs_sha256'], supports=list(SUPPORTS),
                checkpoint=str(CHECKPOINT), sampling_calls=0, DT_calls=0, optimizer_steps=0, scheduler_steps=0,
                native_backward_passes_per_rank=4, completed_unix=time.time(),
                scope='Two original worker calls; each selects complete DT PG and support-specific saved GRPO PG. No production update.'), indent=2) + '\n')
        finally:
            recipe.ray.shutdown()

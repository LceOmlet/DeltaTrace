"""One saved native64, officially whitened, real VERL update and checkpoint roundtrip.

Compose the completed whitening/Adam recipes and original worker APIs. No
rollout, DT, alternative loss, repeated update branch or numerical gate.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import time

import verify_textcraft_official_whitening as whitening

recipe = whitening.recipe
OUT, CHECKPOINT = recipe.OUT, recipe.CHECKPOINT
SAVED_CHECKPOINT = OUT / 'updated-actor'


def native_state(worker, label):
    """Describe finite local trainable/Adam storage; no gather or state mutation."""
    from observe_native_adam_update import _local_cpu_copy, _state_summary
    torch = recipe.torch
    actor = worker.actor
    parameters = dict(actor.actor_module.named_parameters())
    names = {id(parameter): name for name, parameter in parameters.items()}

    def describe(tensor):
        local, layout = _local_cpu_copy(tensor)
        return dict(layout=layout, nonfinite_elements=int((~torch.isfinite(local)).sum()),
            local_storage_sha256=hashlib.sha256(
                local.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest())

    trainable = {name: describe(parameter) for name, parameter in parameters.items()
                 if parameter.requires_grad}
    optimizer_tensors = {}
    for parameter, state in actor.actor_optimizer.state.items():
        name = names.get(id(parameter), '<parameter not in actor named_parameters>')
        optimizer_tensors[name] = {key: describe(value) for key, value in state.items()
                                  if isinstance(value, torch.Tensor)}
    value = dict(label=label, rank=int(worker.rank), observed_unix=time.time(),
        scope='Original dtype local trainable parameter shards and every tensor in original Adam state; frozen base parameters are not copied. No gather, optimizer step, numerical tolerance or state change.',
        local_copy_source=recipe.identity(_local_cpu_copy),
        resources=recipe.resources(), trainable_parameters=trainable,
        optimizer_tensor_states=optimizer_tensors,
        optimizer=_state_summary(actor.actor_optimizer),
        scheduler=worker.actor_lr_scheduler.state_dict())
    path = OUT / f'rank{worker.rank}-{label}-native-state.json'
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


@recipe.ray.remote
class WhitenedUpdateWorker(recipe.ActorRolloutRefWorker):
    @recipe.register(dispatch_mode=recipe.Dispatch.DP_COMPUTE_PROTO)
    def observe_update(self, data):
        from observe_native_adam_update import observe_saved_native_adam_update
        return observe_saved_native_adam_update(self, data,
            original_update_actor=super().update_actor,
            output_directory=OUT, branch='dt')

    @recipe.register(dispatch_mode=recipe.Dispatch.ONE_TO_ALL)
    def observe_native_state(self, label):
        from verify_textcraft_whitened_update import native_state
        return native_state(self, label)


def inspect_inputs():
    import cloudpickle
    raw, white, cfg, report = whitening.inspect_inputs()
    original = recipe.ActorRolloutRefWorker
    signatures = {name: str(inspect.signature(getattr(original, name)))
                  for name in ('save_checkpoint', 'load_checkpoint', 'compute_log_prob')}
    # Bind the actual installed methods before any model or worker submission.
    inspect.signature(original.save_checkpoint).bind(None,
        local_path=str(SAVED_CHECKPOINT), max_ckpt_to_keep=None)
    inspect.signature(original.load_checkpoint).bind(None,
        local_path=str(SAVED_CHECKPOINT), del_local_after_load=False)
    cls = WhitenedUpdateWorker.__ray_metadata__.modified_class
    assert recipe.identity(cls.__init__) == report['worker_constructor']
    serialized = {}
    for name in ('observe_update', 'observe_native_state'):
        method = getattr(cls, name)
        assert recipe.identity(cloudpickle.loads(cloudpickle.dumps(method))) == recipe.identity(method)
        serialized[name] = recipe.identity(method)
    report.update(scope='Same native64/checkpoint25 and one full-batch official whitening; actual update/save/load/logprob API inspection only, no model.',
        sources={**report['sources'], 'real_update_observer': recipe.identity(recipe.observe_saved_native_adam_update),
                 'new_harness': recipe.identity(inspect_inputs),
                 'native_state_observer': recipe.identity(native_state),
                 'original_checkpoint_save': recipe.identity(original.save_checkpoint)},
        original_api_signatures=signatures, serialized_worker_methods=serialized,
        saved_checkpoint=str(SAVED_CHECKPOINT), worker_role='actor',
        checkpoint_scope='Manual original save interface with its default global_step=0; one optimizer step is not a completed formal training iteration.',
        requested_real_optimizer_steps_per_rank=1,
        requested_real_scheduler_steps_per_rank=1,
        real_update_branch='dt with officially whitened actor advantages; raw dt_* retained')
    (OUT / 'native-owner-inspection.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return white, cfg, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inspect-only', action='store_true')
    args = parser.parse_args()
    white, cfg, inspection = inspect_inputs()
    if args.inspect_only:
        print(json.dumps(dict(status='inspected_no_model', rows=len(white),
            cuda_initialized=recipe.torch.cuda.is_initialized(),
            original_api_signatures=inspection['original_api_signatures'])))
        return
    recipe.ray.init(num_cpus=8, include_dashboard=False)
    try:
        group = recipe.RayWorkerGroup(recipe.RayResourcePool([2], use_gpu=True, max_colocate_count=1),
            recipe.RayClassWithInitArgs(WhitenedUpdateWorker, cfg.actor_rollout_ref, 'actor'))
        recipe.phase('original_model_init_begin')
        group.init_model()
        recipe.phase('original_complete_checkpoint_restore_begin')
        group.load_checkpoint(local_path=str(CHECKPOINT), del_local_after_load=False)
        data = recipe._snapshot(white)
        data.meta_info['native_adam_branch'] = 'dt'
        recipe.phase('original_whitened_update_begin')
        update = recipe.save_artifact('white-original-update-output.pkl', group.observe_update(data))
        recipe.phase('original_whitened_update_complete')
        state_after = group.observe_native_state('after-update')
        after = recipe.save_artifact('white-after-update-logprob.pkl',
            group.compute_log_prob(recipe._snapshot(data)))
        recipe.phase('original_checkpoint_save_begin')
        group.save_checkpoint(local_path=str(SAVED_CHECKPOINT), max_ckpt_to_keep=None)
        recipe.phase('original_checkpoint_save_complete')
        group.load_checkpoint(local_path=str(SAVED_CHECKPOINT), del_local_after_load=False)
        recipe.phase('original_saved_checkpoint_restore_complete')
        state_restored = group.observe_native_state('after-restore')
        restored = recipe.save_artifact('white-after-restore-logprob.pkl',
            group.compute_log_prob(recipe._snapshot(data)))
        (OUT / 'completed.json').write_text(json.dumps(dict(
            scope='One true original VERL update of the saved global64 after one full-batch official whitening, original checkpoint save/load and original logprob readbacks. State and readback equality are descriptive; no new tolerance gate. No generation or formal training resume.',
            completed_unix=time.time(), global_rows=len(data),
            original_checkpoint=str(CHECKPOINT), saved_checkpoint=str(SAVED_CHECKPOINT),
            checkpoint_scope=inspection['checkpoint_scope'],
            input_sha256=inspection['native_reader_inputs_sha256'],
            white_carrier=inspection['white_carrier'], sources=inspection['sources'],
            normalization=inspection['normalization'], update=update,
            after_update_logprob=after, after_restore_logprob=restored,
            after_update_states=state_after, after_restore_states=state_restored,
            requested_optimizer_steps_per_rank=1, requested_scheduler_steps_per_rank=1,
            DT_calls=0, sampling_calls=0), indent=2) + '\n', encoding='utf-8')
        recipe.phase('complete_whitened_original_update_roundtrip')
    finally:
        recipe.ray.shutdown()


if __name__ == '__main__':
    main()

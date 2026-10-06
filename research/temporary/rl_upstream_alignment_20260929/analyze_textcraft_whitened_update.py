"""CPU-only description of the completed single native whitened VERL update.

Reuse the saved-native-Adam analyzer's local-shard geometry and original
DataProto/B4 loss reduction. No Adam formula, model call or numerical gate.
Large parameter/DataProto artifacts stay in the existing remote CPU runtime.
"""
import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import time

ADAM_ANALYZER_SHA = '318a994fe35f68ae6400c76c1e735973ed1955771918677213f0bed92be792d1'


def compare_native_states(after, restored):
    """Compare original observation fields, excluding resource/time metadata."""
    result = {}
    for key in ('trainable_parameters', 'optimizer_tensor_states'):
        if key == 'trainable_parameters':
            a, b = after[key], restored[key]
        else:
            a = {(name, field): value for name, fields in after[key].items()
                 for field, value in fields.items()}
            b = {(name, field): value for name, fields in restored[key].items()
                 for field, value in fields.items()}
        shared = sorted(set(a) & set(b))
        changed = [name for name in shared if
                   a[name]['local_storage_sha256'] != b[name]['local_storage_sha256']]
        result[key] = dict(after_update_tensor_count=len(a), after_restore_tensor_count=len(b),
            shared_tensor_count=len(shared), equal_storage_sha_count=len(shared)-len(changed),
            differing_storage_sha_keys=changed,
            missing_after_restore_keys=sorted(set(a)-set(b)),
            added_after_restore_keys=sorted(set(b)-set(a)),
            layout_exact_equal_count=sum(a[name]['layout'] == b[name]['layout'] for name in shared),
            after_update_nonfinite_elements=sum(x['nonfinite_elements'] for x in a.values()),
            after_restore_nonfinite_elements=sum(x['nonfinite_elements'] for x in b.values()))
    result.update(optimizer_counters_and_groups_exact=after['optimizer'] == restored['optimizer'],
                  optimizer_after_update=after['optimizer'], optimizer_after_restore=restored['optimizer'],
                  scheduler_state_exact=after['scheduler'] == restored['scheduler'],
                  scheduler_after_update=after['scheduler'], scheduler_after_restore=restored['scheduler'])
    return result


def build_analysis(root, owner, torch, DataProto, agg_loss):
    root = Path(root)
    started, resource_before = time.time(), owner.resources(torch)
    sources = dict(analyzer=owner.identity(__file__), reused_analyzer=owner.identity(owner.__file__),
        local_parameter_geometry=owner.source(owner.local_parameter_geometry),
        original_observation_reduction=owner.source(owner.owner_observation),
        protocol_load=owner.source(DataProto.load_from_disk),
        protocol_chunk=owner.source(DataProto.chunk), agg_loss=owner.source(agg_loss))
    if sources['reused_analyzer']['sha256'] != ADAM_ANALYZER_SHA:
        raise ValueError('Use the recorded completed native Adam analyzer unchanged.')
    if (sources['protocol_load']['sha256'] != owner.PROTO_SHA
            or sources['protocol_chunk']['sha256'] != owner.PROTO_SHA
            or sources['agg_loss']['sha256'] != owner.CORE_SHA):
        raise ValueError('Use the recorded original protocol and loss-reduction owner.')
    paths = dict(completed=root/'completed.json', inspection=root/'native-owner-inspection.json')
    completed, inspection = [json.loads(paths[name].read_bytes()) for name in ('completed', 'inspection')]
    inputs = {name: owner.identity(path) for name, path in paths.items()}

    def bound_artifact(name, receipt):
        path = Path(receipt['path'])
        identity = owner.identity(path)
        if identity['sha256'] != receipt['sha256']:
            raise ValueError('Completion-bound artifact differs: ' + name)
        inputs[name] = identity
        return path

    saved = DataProto.load_from_disk(str(bound_artifact('white_carrier', completed['white_carrier'])))
    after = DataProto.load_from_disk(str(bound_artifact('after_update_logprob', completed['after_update_logprob'])))
    restored = DataProto.load_from_disk(str(bound_artifact('after_restore_logprob', completed['after_restore_logprob'])))
    if not (len(saved) == len(after) == len(restored) == 64):
        raise ValueError('Use the same saved native global64 and its full readbacks.')
    mode = inspection['actor_config']['actor']['loss_agg_mode']
    readback = owner.owner_observation(saved, after, restored, agg_loss, mode, torch)
    selected = saved.batch['loss_mask'][:, -saved.batch['responses'].shape[-1]:] != 0
    readback_storage = {key: dict(
        after_update=owner.tensor_identity(after.batch[key], torch),
        after_restore=owner.tensor_identity(restored.batch[key], torch),
        whole_tensor_exact=torch.equal(after.batch[key], restored.batch[key]),
        action_mask_selected_exact=torch.equal(after.batch[key][selected], restored.batch[key][selected]))
        for key in ('old_log_probs', 'entropys')}
    per_rank = []
    for rank in (0, 1):
        report_path = root / f'rank{rank}-dt-native-adam.json'
        report = json.loads(report_path.read_bytes())
        inputs[f'rank{rank}_native_update'] = owner.identity(report_path)
        payload_path = bound_artifact(f'rank{rank}_native_shards', report['parameter_shards'])
        payload = torch.load(payload_path, map_location='cpu', weights_only=False)
        # Reuse the existing descriptive geometry for its sole actual branch;
        # restore its module setting immediately, without changing owner source.
        previous = owner.BRANCHES
        try:
            owner.BRANCHES = ('dt',)
            geometry = owner.local_parameter_geometry(rank, {'dt': payload}, {'dt': report}, torch)
        finally:
            owner.BRANCHES = previous
        del payload
        state_receipts = (completed['after_update_states'][rank], completed['after_restore_states'][rank])
        state_paths = [bound_artifact(f'rank{rank}_{when}_state', receipt)
                       for when, receipt in zip(('after_update', 'after_restore'), state_receipts)]
        states = [json.loads(path.read_bytes()) for path in state_paths]
        if any(state['rank'] != rank for state in states) or report['rank'] != rank:
            raise ValueError('The actual native rank order differs from the recorded dispatch.')
        per_rank.append(dict(rank=rank, real_optimizer_step_calls=report['optimizer_step_calls'],
            real_scheduler_step_calls=report['scheduler_step_calls'],
            optimizer_before=report['optimizer_before'], optimizer_after=report['optimizer_after'],
            scheduler_before=report['scheduler_before'], scheduler_after=report['scheduler_after'],
            original_step_records=report['optimizer_steps'], original_scheduler_records=report['scheduler_steps'],
            original_metrics=report['original_metrics'], source_bindings=report['sources'],
            effective_actor_config=report['effective_actor_config'],
            local_parameter_update=geometry, checkpoint_restore=compare_native_states(*states)))
    checkpoint = Path(completed['saved_checkpoint'])
    checkpoint_files = [dict(path=str(path), bytes=path.stat().st_size)
                        for path in sorted(checkpoint.rglob('*')) if path.is_file()]
    result = dict(
        scope='One saved global64, full-collected-batch officially whitened advantages, real original VERL update and official checkpoint roundtrip. CPU saved-storage analysis only; no tolerance gate or formal-training claim.',
        source_files=inputs, analysis_sources=sources, runtime_sources=completed['sources'],
        input_sha256=completed['input_sha256'], actor_config=inspection['actor_config'],
        normalization=completed['normalization'], per_rank=per_rank,
        same_job_after_update_vs_after_restore_readback=readback,
        same_job_readback_storage=readback_storage,
        readback_field_scope={
            'before': 'Current job after its single actual update, before original save/load.',
            'after': 'Same job after original save/load of that updated actor.',
            'before_LP_minus_saved_original': 'Supplement only: current post-update LP minus historical saved old policy LP; includes cross-run arithmetic and the update. Not a same-job pre-update baseline.'},
        saved_checkpoint=dict(path=str(checkpoint), scope=completed['checkpoint_scope'], files=checkpoint_files,
            inspection_scope='File names/size only; model checkpoint tensors are not loaded or separately hashed. Original restore and local state storage observations provide the readback evidence.'),
        operations=dict(analysis_model_calls=0, analysis_DT_calls=0, analysis_backward_calls=0,
            analysis_optimizer_steps=0, generation_calls=completed['sampling_calls'], DT_calls=completed['DT_calls']),
        resources=dict(before=resource_before, after=owner.resources(torch)), elapsed_seconds=time.time()-started,
        limitations=[
            'Parameter geometry is native local-shard storage per rank; no cross-rank gather, global norm or Adam formula.',
            'Finite checks cover all trainable local parameters, stored clipped input gradients and all optimizer tensor states; frozen base model tensors are not copied or scanned.',
            'After-update/after-restore LP/H differences use the original action mask, DataProto chunk(2), local32/B4x8 and original agg_loss. Numerical differences are descriptive, not an invented tolerance.',
            'One true optimizer step from checkpoint25 is not one completed formal rollout/training iteration. Manual save uses original global_step default0.',
            'The actor-only diagnostic does not test generation, vLLM coexistence, learning convergence or overall DT attribution quality.'])
    return owner.json_finite(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--adam-analyzer-dir', type=Path,
        default=os.environ.get('DT_TEXTCRAFT_ADAM_RECIPE_ROOT'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or not args.adam_analyzer_dir:
        raise ValueError('Use CUDA_VISIBLE_DEVICES empty and the existing native Adam analyzer directory.')
    sys.path.insert(0, str(args.adam_analyzer_dir))
    owner = importlib.import_module('analyze_textcraft_native_adam')
    import torch
    from verl.protocol import DataProto
    from verl.trainer.ppo.core_algos import agg_loss
    if torch.cuda.is_initialized() or torch.distributed.is_initialized():
        raise RuntimeError('CPU analysis must not initialize CUDA or distributed workers.')
    result = build_analysis(args.root, owner, torch, DataProto, agg_loss)
    output = args.output or args.root / 'whitened-update-analysis.json'
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(dict(output=owner.identity(output), per_rank=[dict(rank=x['rank'],
        optimizer_step_calls=x['real_optimizer_step_calls'], scheduler_step_calls=x['real_scheduler_step_calls'],
        parameter_update=x['local_parameter_update']['branches']['dt'],
        restore=x['checkpoint_restore']) for x in result['per_rank']])))


if __name__ == '__main__':
    main()

"""CPU-only description of a completed equivalent-label native gradient probe.

Norms and dots are read from the original ownership-aware helper's reports.
Saved/new DataProto files supply exact masks and coefficients; no model, loss,
DT, backward, gradient reducer or optimizer is called by this analysis.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path


GROUPS = ('saved_original', 'swapped_labels')
LABELS = ('dt_pg', 'weighted_entropy')


def source(path):
    path = Path(path)
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                bytes=path.stat().st_size)


def divide(a, b):
    return a / b if b else None


def gradient_pair(statistics, a, b):
    norms, dots = statistics['norms'], statistics['inner_products']
    key = f'{a}:{b}' if f'{a}:{b}' in dots else f'{b}:{a}'
    dot = dots[key]
    return dict(first=a, second=b, first_norm=norms[a], second_norm=norms[b],
        dot=dot, cosine=divide(dot, norms[a] * norms[b]),
        second_to_first_norm_ratio=divide(norms[b], norms[a]))


def coefficient_description(values, torch):
    x = values.detach().cpu().double().reshape(-1)
    finite = x[torch.isfinite(x)]
    return dict(count=x.numel(), finite_count=finite.numel(),
        positive_count=int((finite > 0).sum()), negative_count=int((finite < 0).sum()),
        zero_count=int((finite == 0).sum()),
        sum=float(finite.sum()), absolute_sum=float(finite.abs().sum()),
        mean=float(finite.mean()) if finite.numel() else None,
        mean_absolute=float(finite.abs().mean()) if finite.numel() else None,
        minimum=float(finite.min()) if finite.numel() else None,
        maximum=float(finite.max()) if finite.numel() else None)


def carrier_analysis(original_path, new_path, inspection):
    # Imports are lazy: an analysis of saved tensors requires the already
    # provisioned CPU owner environment, never local Torch installation.
    import numpy as np
    import torch
    from verl.protocol import DataProto
    assert not torch.cuda.is_initialized()
    original = DataProto.load_from_disk(str(original_path))
    swapped = DataProto.load_from_disk(str(new_path))
    assert len(original) == len(swapped) == 64
    assert all(value.device.type == 'cpu' for data in (original, swapped)
               for value in data.batch.values())
    removed = set(inspection['popped_batch_fields'])
    retained = set(original.batch.keys()) - removed
    fields = {key:dict(original_shape=list(original.batch[key].shape),
        new_shape=list(swapped.batch[key].shape), original_dtype=str(original.batch[key].dtype),
        new_dtype=str(swapped.batch[key].dtype), exact=torch.equal(original.batch[key], swapped.batch[key]))
        for key in sorted(retained)}
    non_tensor = {key:bool(np.array_equal(original.non_tensor_batch[key], swapped.non_tensor_batch[key]))
        for key in original.non_tensor_batch if key in swapped.non_tensor_batch}
    meta_keys = set(original.meta_info) | set(swapped.meta_info)
    meta = {key:dict(original=original.meta_info.get(key), new=swapped.meta_info.get(key))
        for key in sorted(meta_keys) if original.meta_info.get(key) != swapped.meta_info.get(key)}
    ranks, pooled_old, pooled_new = [], [], []
    for rank, (old_local, new_local) in enumerate(zip(original.chunk(2), swapped.chunk(2))):
        batches = []
        for index, (old_micro, new_micro) in enumerate(zip(old_local.batch.split(4), new_local.batch.split(4))):
            response_length = old_micro['responses'].shape[-1]
            old_mask = old_micro['loss_mask'][:, -response_length:].bool()
            new_mask = new_micro['loss_mask'][:, -response_length:].bool()
            assert old_mask.shape == old_micro['advantages'].shape
            assert torch.equal(old_mask, new_mask)
            a, b = old_micro['advantages'][old_mask], new_micro['advantages'][new_mask]
            old_q, new_q = old_micro['dt_q_estimates'][old_mask], new_micro['dt_q_estimates'][new_mask]
            pooled_old.append(a)
            pooled_new.append(b)
            batches.append(dict(index=index, rows=old_micro['responses'].shape[0],
                valid_policy_tokens=int(old_mask.sum()), mask_exact=torch.equal(old_mask, new_mask),
                original_A=coefficient_description(a, torch), new_A=coefficient_description(b, torch),
                A_difference=coefficient_description(b.double() - a.double(), torch),
                A_opposite_nonzero_signs=int(((a > 0) & (b < 0) | (a < 0) & (b > 0)).sum()),
                Q_exact=torch.equal(old_q, new_q),
                original_Q_nonzero_tokens=int((old_q != 0).sum()), new_Q_nonzero_tokens=int((new_q != 0).sum())))
        ranks.append(dict(rank=rank, local_rows=len(old_local), microbatches=batches,
            equal_B4_mean_absolute_A=dict(
                original=math.fsum(x['original_A']['mean_absolute'] for x in batches) / len(batches),
                new=math.fsum(x['new_A']['mean_absolute'] for x in batches) / len(batches)),
            mean_scope='Equal average of descriptive masked B4 means; original complete minibatch has eight B4 native backward contributions scaled by 1/8.'))
    assert not torch.cuda.is_initialized()
    return dict(original=source(original_path), new=source(new_path), rows=64,
        input_field_identity=fields, non_tensor_field_identity=non_tensor,
        original_non_tensor_keys=sorted(original.non_tensor_batch),
        new_non_tensor_keys=sorted(swapped.non_tensor_batch), differing_meta_info=meta,
        exact_retained_tensor_fields=all(x['exact'] for x in fields.values()),
        rank_partitions=ranks,
        pooled_policy_tokens=dict(original_A=coefficient_description(torch.cat(pooled_old), torch),
                                  new_A=coefficient_description(torch.cat(pooled_new), torch)),
        mean_scope='Pooled token means use all effective policy-mask positions; this is distinct from equal B4 means and is not a new loss reduction.',
        mask_source='Original loss_mask[:, -responses.shape[-1]:]; observations and padding remain masked.',
        cuda_initialized=torch.cuda.is_initialized(),
        dataproto_source=source(__import__('inspect').getsourcefile(DataProto)))


def build_analysis(root, original_path=None, new_path=None):
    root = Path(root)
    paths = dict(completed=root / 'completed.json', inspection=root / 'native-owner-inspection.json',
                 prepared=root / 'prepared.json', job=root / 'job.json')
    for rank in (0, 1):
        for group in GROUPS:
            paths[f'rank{rank}_{group}'] = root / f'rank{rank}-{group}-gradients.json'
        paths[f'rank{rank}_cross'] = root / f'rank{rank}-cross-group-gradients.json'
        paths[f'rank{rank}_dt'] = root / f'rank{rank}-swapped-label-native-dt-report.json'
    # A completed marker and complete native group reports are a data-presence
    # requirement, not a numerical pass threshold.
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError('Completed diagnostic artifacts are not all present: ' + ', '.join(missing))
    raw = {name:json.loads(path.read_bytes()) for name, path in paths.items()}
    completed, inspection, job = raw['completed'], raw['inspection'], raw['job']
    rank_reports = []
    for rank in (0, 1):
        old, new, cross = (raw[f'rank{rank}_{GROUPS[0]}'], raw[f'rank{rank}_{GROUPS[1]}'], raw[f'rank{rank}_cross'])
        rank_reports.append(dict(rank=rank,
            original_PG_H=gradient_pair(old['gradient_statistics'], 'dt_pg', 'weighted_entropy'),
            new_PG_H=gradient_pair(new['gradient_statistics'], 'dt_pg', 'weighted_entropy'),
            old_new_PG=gradient_pair(cross['gradient_statistics'], 'old_dt_pg', 'swapped_dt_pg'),
            old_PG_new_H=gradient_pair(cross['gradient_statistics'], 'old_dt_pg', 'weighted_entropy'),
            cross_native_statistics=cross['gradient_statistics'],
            original_native_statistics=old['gradient_statistics'], new_native_statistics=new['gradient_statistics'],
            group_config_exact=old['effective_config'] == new['effective_config'],
            effective_config=old['effective_config'], input_shapes_exact=old['input_fields'] == new['input_fields'],
            input_fields=old['input_fields'], rng_observations=dict(original=old['rng_observation'], new=new['rng_observation']),
            group_sources=dict(original=old['sources'], new=new['sources'],
                cross_helper=cross['original_statistics_source'], adapter=cross['adapter_source']),
            operations=dict(update_policy_passes=sum(x['actual_backward_passes'] for x in (old, new)),
                microbatch_backward_calls=sum(len(p['microbatch_losses']) for x in (old, new) for p in x['passes'].values()),
                optimizer_step_executed=[x['optimizer_step_executed'] for x in (old, new)],
                scheduler_step_executed=[x['scheduler_step_executed'] for x in (old, new)],
                scheduler_no_op_calls=[x['scheduler_step_no_op_calls'] for x in (old, new)]),
            swapped_DT=dict(seconds=raw[f'rank{rank}_dt']['seconds'],
                restored_labels=raw[f'rank{rank}_dt']['original_label_method_restored'],
                unchanged_query_method=raw[f'rank{rank}_dt']['query_method_unchanged'],
                native_report={key:raw[f'rank{rank}_dt']['report'].get(key)
                    for key in ('finite_trace_calls', 'event_contrasts', 'policy_tokens', 'seconds')})))
    original_path = Path(original_path) if original_path else Path(job['input']['path'])
    new_path = Path(new_path) if new_path else root / 'labels-recomputed-native-minibatch.pkl'
    carriers = carrier_analysis(original_path, new_path, inspection)
    assert carriers['original']['sha256'] == completed['input_sha256']
    return dict(scope='Equivalent-label encoding sensitivity of this saved native64 reward readout and task PG; not overall DT quality, convergence or a numerical tolerance test.',
        source_files={name:source(path) for name, path in paths.items()},
        analysis_source=source(__file__), sources=completed['sources'],
        checkpoint=completed['checkpoint'], input_sha256=completed['input_sha256'],
        config_source=inspection['config_source'], actor_config=inspection['actor_config'],
        cpu_owner_contract=inspection['cpu_owner_contract'], rows=64, groups=list(GROUPS), labels=list(LABELS),
        per_rank=rank_reports, carriers=carriers,
        operations=dict(native_backward_passes_per_rank=completed['native_backward_passes_per_rank'],
            sampling_calls=completed['sampling_calls'], optimizer_steps=completed['optimizer_steps'],
            scheduler_steps=completed['scheduler_steps'], analysis_model_forward_calls=0,
            analysis_DT_calls=0, analysis_backward_calls=0),
        interpretation_limits=[
            'Each rank report already contains native mesh-reduced sharded statistics; the two reports are not summed into a new reducer.',
            'PG norm ratios are gradient measurements, not predicted parameter-update ratios or success-rate changes.',
            'Equivalent category labels preserve return meanings and Q/V/A composition; sensitivity here does not decide attribution ranking quality.',
            'Entropy is the original coefficient-weighted component; no normalization, rescaling or loss-parameter change is proposed.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=os.environ.get('DT_TEXTCRAFT_LABEL_GRADIENT_ROOT'))
    parser.add_argument('--original-carrier', type=Path)
    parser.add_argument('--new-carrier', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.root is None:
        parser.error('--root must name the completed v2 diagnostic directory')
    result = build_analysis(args.root, args.original_carrier, args.new_carrier)
    output = args.output or args.root / 'label-gradient-analysis.json'
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(dict(path=str(output), sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        per_rank=[dict(rank=x['rank'], old_new_PG=x['old_new_PG'], new_PG_H=x['new_PG_H']) for x in result['per_rank']])))

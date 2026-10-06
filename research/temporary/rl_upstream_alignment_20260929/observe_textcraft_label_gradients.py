"""Compose original native DT/entropy observations for two label encodings.

Only this isolated diagnostic keeps a CPU PG snapshot across its two RPCs.
Original losses, backward, clipping, ownership reduction and output stay owned
by the existing observer/worker. No production entry imports this module.
"""
import importlib
import json
from pathlib import Path
import random
import time
from contextlib import ExitStack


GROUPS = ('saved_original', 'swapped_labels')
LABELS = ('dt_pg', 'weighted_entropy')
_STATE = '_textcraft_label_gradient_diagnostic_state'


def swapped_labels(self):
    """Equivalent encoding; category values/meanings and observed index stay put."""
    return '10'


def observe_label_gradients(worker, data, *, original_update_actor, output_directory):
    try:
        return _observe_label_gradients(worker, data,
            original_update_actor=original_update_actor, output_directory=output_directory)
    except BaseException:
        vars(worker).pop(_STATE, None)
        raise


def _observe_label_gradients(worker, data, *, original_update_actor, output_directory):
    """Observe two native components and return the exact original worker output."""
    import numpy as np
    import torch

    owner = importlib.import_module('observe_native_optimizer_minibatch')
    group = data.meta_info['label_gradient_group']
    if group not in GROUPS:
        vars(worker).pop(_STATE, None)
        raise ValueError(group)
    if group == GROUPS[0]:
        vars(worker).pop(_STATE, None)
    previous = vars(worker).get(_STATE)
    if group == GROUPS[1] and previous is None:
        raise RuntimeError('The original group PG snapshot must precede the swapped group.')
    output_path = Path(output_directory) / f'rank{worker.rank}-{group}-gradients.json'
    cross_path = Path(output_directory) / f'rank{worker.rank}-cross-group-gradients.json'
    original_statistics = owner.native_gradient_statistics
    scheduler_calls = []
    caller_python, caller_numpy = random.getstate(), np.random.get_state()
    device = torch.cuda.current_device()
    caller_cpu = torch.get_rng_state()
    caller_cuda = torch.cuda.get_rng_state(device)
    baseline_rng = (caller_python, caller_numpy, caller_cpu, caller_cuda)
    if previous is not None:
        baseline_rng = previous['rng']
    py, nu, cpu, cuda = baseline_rng
    rng_observation = {
        'Python_caller_state_equals_original_entry': caller_python == py,
        'NumPy_caller_state_equals_original_entry': (
            caller_numpy[0] == nu[0] and np.array_equal(caller_numpy[1], nu[1])
            and caller_numpy[2:] == nu[2:]),
        'Torch_CPU_caller_state_equals_original_entry': torch.equal(caller_cpu, cpu),
        'Torch_rank_CUDA_caller_state_equals_original_entry': torch.equal(caller_cuda, cuda),
        'replayed_original_entry_before_swapped_RPC': previous is not None,
        'scope': 'Public RNG states at diagnostic RPC entry; original observer also replays each pass. '
                 'Caller Python/NumPy states are restored in finally and Torch via public fork_rng.',
    }
    completed = False

    def statistics_tap(trainable, snapshots, groups, statistic_device, labels, *, norm_scope):
        # Return this exact owner result. The second owner call is scalar
        # observation only and uses the same actual DTensor ownership groups.
        result = original_statistics(trainable, snapshots, groups, statistic_device,
                                     labels, norm_scope=norm_scope)
        if group == GROUPS[0]:
            saved_pg = {
                name: (None if value is None else value.detach().cpu().clone(), bucket)
                for name, (value, bucket) in snapshots['dt_pg'].items()
            }
            setattr(worker, _STATE, dict(pg=saved_pg, rng=baseline_rng,
                original_statistics_source=owner._source_identity(original_statistics),
                snapshot_scope=norm_scope))
        else:
            combined = {'old_dt_pg': previous['pg'],
                        'swapped_dt_pg': snapshots['dt_pg'],
                        'weighted_entropy': snapshots['weighted_entropy']}
            combined_labels = ('old_dt_pg', 'swapped_dt_pg', 'weighted_entropy')
            cross = original_statistics(trainable, combined, groups, statistic_device,
                combined_labels, norm_scope='Cross-group original pre-clip local32/B4x8 gradients; '
                'the original helper retains native mesh SUM and local replicated/plain ownership.')
            cross_record = dict(
                scope='Paired original/swapped DT PG and second-group original weighted entropy; '
                      'original gradient-statistics helper, no additional forward/backward/update.',
                rank=worker.rank, labels=list(combined_labels),
                gradient_statistics=cross,
                original_statistics_source=owner._source_identity(original_statistics),
                original_group_statistics_source=previous['original_statistics_source'],
                adapter_source=owner._source_identity(observe_label_gradients),
                gradient_dtypes={label: sorted({str(value.dtype)
                    for value, _ in values.values() if value is not None})
                    for label, values in combined.items()},
                gradient_parameter_counts={label: len(values) for label, values in combined.items()},
                original_group_snapshot_scope=previous['snapshot_scope'],
                rng_observation=rng_observation,
                additional_backward_calls=0, optimizer_steps=0, scheduler_steps=0)
            cross_path.write_text(json.dumps(cross_record, indent=2) + '\n', encoding='utf-8')
        return result

    def observed_update_policy(*, data):
        return owner.observe_native_optimizer_minibatch(
            worker.actor, data, output_path=output_path, rank=worker.rank)

    def no_op_scheduler_step(*args, **kwargs):
        scheduler_calls.append(True)

    try:
        with torch.random.fork_rng(devices=[device]):
            random.setstate(py)
            np.random.set_state(nu)
            torch.set_rng_state(cpu)
            torch.cuda.set_rng_state(cuda, device)
            with ExitStack() as stack:
                stack.enter_context(owner._temporary_attribute(owner, 'LABELS', LABELS))
                stack.enter_context(owner._temporary_attribute(
                    owner, 'native_gradient_statistics', statistics_tap))
                stack.enter_context(owner._temporary_attribute(
                    worker.actor, 'update_policy', observed_update_policy))
                stack.enter_context(owner._temporary_attribute(
                    worker.actor_lr_scheduler, 'step', no_op_scheduler_step))
                output = original_update_actor(data)
        result = json.loads(output_path.read_bytes())
        result['original_observer_scope'] = result['scope']
        result['scope'] = ('One complete original local32/B4x8 /8 minibatch; two native backward '
                           'passes: DT PG and original weighted entropy; no optimizer/scheduler update.')
        result['label_gradient_group'] = group
        result['actual_backward_passes'] = len(result['passes'])
        result['scheduler_step_no_op_calls'] = len(scheduler_calls)
        result['scheduler_step_executed'] = False
        result['sources']['gradient_statistics'] = owner._source_identity(original_statistics)
        result['statistics_tap_source'] = owner._source_identity(statistics_tap)
        result['diagnostic_adapter'] = owner._source_identity(observe_label_gradients)
        result['rng_observation'] = rng_observation
        result['cross_group_statistics_path'] = str(cross_path) if previous is not None else None
        result['unused_GRPO_field'] = ('Original diagnostic_grpo_advantages is retained unchanged '
                                       'for the original observer interface; no GRPO backward is selected.')
        output_path.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
        completed = True
        return output
    finally:
        random.setstate(caller_python)
        np.random.set_state(caller_numpy)
        if group == GROUPS[1] or not completed:
            vars(worker).pop(_STATE, None)


def recompute_labels_credit(worker, data, *, original_compute_credit, output_directory):
    try:
        return _recompute_labels_credit(worker, data,
            original_compute_credit=original_compute_credit, output_directory=output_directory)
    except BaseException:
        vars(worker).pop(_STATE, None)
        raise


def _recompute_labels_credit(worker, data, *, original_compute_credit, output_directory):
    """Reverse only original category labels for one original native credit call."""
    import torch

    owner = importlib.import_module('observe_native_optimizer_minibatch')
    readout = importlib.import_module('reward_readout')
    original_labels = readout.RewardAlphabet.labels
    original_query = readout.RewardAlphabet.query_ids
    started = time.monotonic()
    print(f'label_gradient rank={worker.rank} phase=native_label_credit_begin unix={time.time()}', flush=True)
    try:
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            with owner._temporary_attribute(readout.RewardAlphabet, 'labels', swapped_labels):
                output = original_compute_credit(data)
        producer = worker._deltatrace_producer
        record = dict(
            scope='Original worker/producer/runner credit call with only category labels reversed; '
                  'original query_ids, values, meanings, observed_index, G and Q/V/A composition unchanged.',
            rank=worker.rank, original_label_source=owner._source_identity(original_labels),
            swapped_label_source=owner._source_identity(swapped_labels),
            original_query_source=owner._source_identity(original_query),
            producer_source=owner._source_identity(type(producer)),
            original_label_method_restored=readout.RewardAlphabet.labels is original_labels,
            query_method_unchanged=readout.RewardAlphabet.query_ids is original_query,
            report=producer.readout.last_report,
            seconds=time.monotonic() - started,
            sampling_calls=0, optimizer_steps=0, scheduler_steps=0)
        (Path(output_directory) / f'rank{worker.rank}-swapped-label-native-dt-report.json').write_text(
            json.dumps(record, indent=2) + '\n', encoding='utf-8')
        print(f'label_gradient rank={worker.rank} phase=native_label_credit_complete '
              f'elapsed={record["seconds"]:.3f}s unix={time.time()}', flush=True)
        return output
    except BaseException:
        vars(worker).pop(_STATE, None)
        raise

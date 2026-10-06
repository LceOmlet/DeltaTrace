"""Bind only the prepared query method and compose the existing native observer."""
import importlib
import json
from pathlib import Path

LABELS = ('dt_pg', 'weighted_entropy')
GROUPS = ('saved_original', 'clock_recomputed')


def observe_query_clock_gradients(worker, data, *, original_update_actor, output_directory):
    owner = importlib.import_module('observe_native_optimizer_minibatch')
    group = data.meta_info['query_clock_gradient_group']
    if group not in GROUPS:
        raise ValueError(group)
    path = Path(output_directory) / f'rank{worker.rank}-{group}-gradients.json'
    scheduler_calls = []

    def observed_update_policy(*, data):
        return owner.observe_native_optimizer_minibatch(worker.actor, data,
            output_path=path, rank=worker.rank)

    def no_op_scheduler_step(*args, **kwargs):
        scheduler_calls.append(True)

    with owner._temporary_attribute(owner, 'LABELS', LABELS):
        with owner._temporary_attribute(worker.actor, 'update_policy', observed_update_policy):
            with owner._temporary_attribute(worker.actor_lr_scheduler, 'step', no_op_scheduler_step):
                output = original_update_actor(data)
    result = json.loads(path.read_bytes())
    result['original_observer_scope'] = result['scope']
    result['scope'] = ('One original local32/B4 x8 /8 minibatch; two original backward passes: '
                       'complete DT PG and original weighted entropy. No optimizer/scheduler update.')
    result['query_clock_gradient_group'] = group
    result['actual_backward_passes'] = len(result['passes'])
    result['scheduler_step_no_op_calls'] = len(scheduler_calls)
    result['scheduler_step_executed'] = False
    result['diagnostic_adapter'] = owner._source_identity(observe_query_clock_gradients)
    result['rng_scope'] = ('Original observer restores/replays Python, NumPy and Torch within each group; '
                           'DT call uses public torch.random.fork_rng for CPU and this rank CUDA only. '
                           'No cross-group Python/NumPy equality or numerical threshold is claimed.')
    path.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    return output


def recompute_clock_credit(worker, data, *, original_compute_credit, output_directory):
    import torch
    owner = importlib.import_module('observe_native_optimizer_minibatch')
    readout = importlib.import_module('reward_readout')
    candidate = importlib.import_module('reward_readout_clock_candidate')
    original_method = readout.RewardAlphabet.query_ids
    # Public native RNG context preserves CPU and the actual CUDA rank state.
    # Original producer owns eval/attention restoration and native offload.
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        with owner._temporary_attribute(readout.RewardAlphabet, 'query_ids', candidate.RewardAlphabet.query_ids):
            output = original_compute_credit(data)
    assert readout.RewardAlphabet.query_ids is original_method
    producer = worker._deltatrace_producer
    value = dict(scope='Original worker/producer/readout call; only query_ids method was temporarily bound.',
        rank=worker.rank, old_query_source=owner._source_identity(original_method),
        candidate_query_source=owner._source_identity(candidate.RewardAlphabet.query_ids),
        producer_source=owner._source_identity(type(producer)),
        query_method_restored=True, report=producer.readout.last_report,
        sampling_calls=0, optimizer_steps=0, scheduler_steps=0)
    (Path(output_directory) / f'rank{worker.rank}-clock-native-dt-report.json').write_text(
        json.dumps(value, indent=2) + '\n', encoding='utf-8')
    return output

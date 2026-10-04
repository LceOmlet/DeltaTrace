"""Exact 32768 capacity fixture using the initialized native B4/rank owner.

Calls the existing fixture and EventRatioReadout. It is not an environment
rollout, reward evaluation, actor update, or new numerical acceptance policy.
"""
import hashlib
import importlib.util
import os
from pathlib import Path
import time
import types

import psutil
import torch


def diagnose(runner, producer, out, save, *, cache_tensors=None):
    from native_prefix_leases import prepare_native_prefix_leases
    from reward_readout import EventRatioReadout
    from verify_dt_context_capacity import capacity_fixture

    out = Path(out)
    rank = torch.distributed.get_rank()
    source = out / f'actual-requests-rank{rank}.pt'
    payload = torch.load(source, map_location='cpu', weights_only=False)
    ordered = sorted(payload['requests'], key=lambda request: request['context_tokens'])
    offset = int(os.environ['DT_PREFIX_DIAGNOSTIC_OFFSET'])
    requests = ordered[offset:offset + 4]
    assert len(requests) == 4
    actual_rows = [payload['rows'][request['row_index']] for request in requests]
    readout = EventRatioReadout(runner, producer.readout_tokenizer,
        **producer.readout_options, appworld_num_tests=int(actual_rows[0]['appworld_num_tests']))
    assert readout.minibatch_size == 4 and readout.max_length == 32768
    episodes, returns, details, factual_inputs = [], [], [], []
    for index, (request, original) in enumerate(zip(requests, actual_rows)):
        value = request['observed_return']
        assert value != 0 and bool(original['active_masks'])
        # Explicit synthetic-capacity coefficient, not an official reward
        # claim for the padded trajectory. Complete returns stay unchanged.
        synthetic_source = {**original, 'rewards': value,
            'traj_uid': f'exact32k-capacity-rank{rank}-row{index}'}
        row, factual, detail, _ = capacity_fixture(synthetic_source,
            readout.tokenizer, readout.alphabet, response_tokens=512,
            max_steps=readout.max_steps, sampling=readout.sampling)
        # The original helper accepts tensor-backed recorded masks, so its
        # count metadata may be scalar tensors. Serialize metadata only;
        # leave the original fixture tensors and helper mathematics intact.
        detail={key:value.item() if isinstance(value,torch.Tensor) and value.ndim==0 else value
                for key,value in detail.items()}
        episodes.append([row]); returns.append([value]); details.append(detail)
        factual_inputs.append(factual)
    assert len({episode[0]['traj_uid'] for episode in episodes}) == 4
    native_prepare = readout._prepare_episode
    def verify_prepare(*args, **kwargs):
        vectors, pending = native_prepare(*args, **kwargs)
        assert len(pending) == 1
        request = pending[0]
        index = next(i for i, episode in enumerate(episodes)
                     if episode[0]['traj_uid'] == request['traj_uid'])
        factual = torch.cat([request[key] for key in ('prompt','actions','query','target')])
        assert factual.numel() == 32768 and torch.equal(factual, factual_inputs[index])
        assert request['actions'].numel() == 512
        assert request['observed_return'] == returns[index][0]
        return vectors, pending
    readout._prepare_episode = verify_prepare
    input_path = out / f'exact32k-inputs-rank{rank}.pt'
    torch.save(dict(episodes=episodes, complete_returns=returns,
                    factual_inputs=factual_inputs, details=details), input_path)
    bank = None
    def capacity_bank(*args, **kwargs):
        nonlocal bank
        reused = bank is not None
        if not reused:
            bank = prepare_native_prefix_leases(*args, **kwargs)
        leases, preparation = bank
        assert len(leases) == 1
        return leases, {**preparation, 'diagnostic_bank_reused': reused,
            'synthetic_capacity_bank_separate_from_real_88_requests': True}
    readout.prefix_lease_factory = capacity_bank
    source = out/'root-tape-owner'/'qwen35_dense_finite_runner_root_tape_candidate.py'
    spec = importlib.util.spec_from_file_location('_isolated_exact32k_root_tape', source)
    candidate = importlib.util.module_from_spec(spec); spec.loader.exec_module(candidate)
    previous_class, previous_attribute = runner.__class__, runner.attribute
    had_flag = hasattr(runner, 'reuse_root_captures')
    previous_flag = getattr(runner, 'reuse_root_captures', False)
    reports, vectors = {}, {}
    save('native_root_tape_exact32k_start', fixture_details=details,
        literal_inputs=dict(path=str(input_path),sha256=hashlib.sha256(input_path.read_bytes()).hexdigest()),
        sampling=readout.sampling, max_steps=readout.max_steps,
        diagnostic_scope='Synthetic exact32768 total readout capacity, B4/card and original interleaved B8 root, LoRA8/16; no actor-update or task-performance claim')
    try:
        runner.__class__ = candidate.Qwen35DenseFiniteRunner
        for label in ('capacity_disabled_cold','capacity_disabled_warm',
                      'capacity_root_cold','capacity_root_warm'):
            runner.reuse_root_captures = label.startswith('capacity_root')
            selected_attribute = types.MethodType(candidate.Qwen35DenseFiniteRunner.attribute,runner)
            phases, counts, observations, attribute_walls = {}, {}, [], []
            def observe(*args, **kwargs):
                tick = time.perf_counter(); result = selected_attribute(*args, **kwargs)
                attribute_walls.append(time.perf_counter()-tick)
                for call in result[1]['calls']:
                    kind=call['kind']; counts[kind]=counts.get(kind,0)+1
                    phases[kind]=phases.get(kind,0)+call.get('stream_elapsed_seconds',call.get('seconds',0))
                observations.append(result[1].get('root_capture_reuse'))
                return result
            runner.attribute = observe
            save('native_root_tape_exact32k_variant_start',variant=label)
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();tick=time.perf_counter()
            result = readout.episodes(episodes,complete_returns=returns)
            torch.cuda.synchronize()
            vectors[label]={key:torch.stack([episode[0][key] for episode in result])
                for key in ('dt_token_advantages','dt_q_estimates','dt_v_estimates')}
            reports[label]=dict(total_wall_seconds=time.perf_counter()-tick,
                original_attribute_wall_seconds=attribute_walls,
                original_runner_phase_seconds=phases,original_runner_phase_counts=counts,
                original_readout_report=readout.last_report,
                root_tape_observations=observations,
                peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(),
                physical_free_bytes=torch.cuda.mem_get_info()[0],
                pss_bytes=psutil.Process().memory_full_info().pss)
            assert all(torch.isfinite(vector).all() for vector in vectors[label].values())
            save('native_root_tape_exact32k_variant_complete',variant=label,reports=reports)
    finally:
        runner.__class__=previous_class;runner.attribute=previous_attribute
        if had_flag:runner.reuse_root_captures=previous_flag
        elif hasattr(runner,'reuse_root_captures'):del runner.reuse_root_captures
    tensor_path=out/f'prefix-lease-vectors-rank{rank}.pt';torch.save(vectors,tensor_path)
    baseline=vectors['capacity_disabled_warm']
    comparisons=[dict(variant=label,comparison_reference='capacity_disabled_warm',
        values=[dict(key=key,equal=bool(torch.equal(vector,baseline[key])),
            maximum_absolute_difference=float((vector.double()-baseline[key].double()).abs().max()))
            for key,vector in values.items()]) for label,values in vectors.items()
        if label!='capacity_disabled_warm']
    save('native_root_tape_exact32k_complete',reports=reports,raw_value_observations=comparisons,
        vectors=dict(path=str(tensor_path),sha256=hashlib.sha256(tensor_path.read_bytes()).hexdigest()))

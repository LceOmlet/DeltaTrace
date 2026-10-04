"""Same-workload DT observation; no PPO or new numerical acceptance policy."""
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

    path=Path(out)/f'actual-requests-rank{torch.distributed.get_rank()}.pt'
    payload=torch.load(path,map_location='cpu',weights_only=False)
    limit=int(os.environ['DT_PREFIX_DIAGNOSTIC_ROWS'])
    requests=sorted(payload['requests'],key=lambda request:request['context_tokens'])[:limit]
    if os.environ.get('DT_PREFIX_LEASE_COMPONENT_DIAGNOSTIC') == '1':
        # Reuse the actual adapter's capture/group/padding decisions. Do not
        # reproduce them in a diagnostic implementation. Request 3 is the
        # largest residual recorded in the completed same-input comparison.
        leases, preparation = prepare_native_prefix_leases(
            runner, requests, minibatch_size=4, eos_token_id=producer.readout_tokenizer.eos_token_id)
        source, source_row = leases[0].sources[3]
        prefix = leases[0].prefix_length
        ids = source.input_ids.to(runner.model.execution_device)
        comparison = torch.stack([request['prompt'][:prefix] for request in requests[:4]]).to(ids.device)
        assert torch.equal(ids[source_row, :prefix], comparison[3])
        prepared_cache = leases[0](comparison)
        prepared_fields = cache_tensors(prepared_cache)
        del prepared_cache
        save('actual_lease_component_inputs', original_request_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
             request_index=3, source_step=requests[3]['source_step'],
             source_start=requests[3]['start'], context_length=requests[3]['context_tokens'],
             capture_input_shape=list(ids.shape), native_input_shape=list(comparison.shape),
             prefix=prefix, matched_rows=[[source_row,3]], preparation=preparation,
             scope='Actual prepared lease source and original B4 prefix; original model, FA/FLA reference and assertions; no production deployment')
        del source, leases
        from diagnose_native_prefix_components import diagnose as components
        return components(runner, ids, prefix, save,
                          comparison_ids=comparison, matched_rows=[(source_row,3)],
                          prepared_prefix_fields=prepared_fields, cache_tensors=cache_tensors,
                          gdn_layer_index=9)
    indices=[request['row_index'] for request in requests]
    rows=[payload['rows'][i] for i in indices]
    returns=[payload['complete_returns'][i] for i in indices]
    readout=EventRatioReadout(runner,producer.readout_tokenizer,**producer.readout_options,
                            appworld_num_tests=int(rows[0]['appworld_num_tests']))
    assert readout.minibatch_size==4 and readout.max_length==32768
    assert readout.alphabet.label_ids(readout.tokenizer)==payload['outcome_token_ids']
    native_prepare=readout._prepare_episode
    expected={(r['traj_uid'],r['source_step']):r for r in requests}
    def verify_prepare(*args,**kwargs):
        values,pending=native_prepare(*args,**kwargs)
        assert len(pending)==len(requests)
        for actual in pending:
            saved=expected[(actual['traj_uid'],actual['source_step'])]
            for field in ('prompt','actions','query','target'):
                assert torch.equal(actual[field],saved[field]), field
            assert (actual['start'],actual['end'],actual['observed_return']) == (
                saved['start'],saved['end'],saved['observed_return'])
        return values,pending
    readout._prepare_episode=verify_prepare
    vectors={};reports={}
    native_attribute=runner.attribute
    source=Path(out)/'qwen35_dense_finite_runner_candidate.py'
    spec=importlib.util.spec_from_file_location('_prepared_prefix_lease_runner',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    save('native_prefix_lease_diagnostic_start',actual_rows=limit,
         context_lengths=[r['context_tokens'] for r in requests],
         original_request_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
         diagnostic_scope='Original readout/QVA consumers; raw residuals have no invented full-network tolerance. Loaded actor is not a restored formal checkpoint.')
    for label in ('original_cold','original_warm','shared_cold','shared_warm'):
        if label.startswith('shared'):
            runner.attribute=types.MethodType(module.Qwen35DenseFiniteRunner.attribute,runner)
            readout.prefix_lease_factory=prepare_native_prefix_leases
        else:
            runner.attribute=native_attribute;readout.prefix_lease_factory=None
        selected_attribute=runner.attribute
        phase_totals={}
        phase_counts={}
        def observe_attribute(*args,**kwargs):
            result=selected_attribute(*args,**kwargs)
            for call in result[1].get('calls',[]):
                kind=call['kind']
                phase_totals[kind]=phase_totals.get(kind,0.0)+call.get('seconds',0.0)
                phase_counts[kind]=phase_counts.get(kind,0)+1
            return result
        # Read timings emitted by the existing owner; no extra forward,
        # synchronization, target, or altered finite computation.
        runner.attribute=observe_attribute
        torch.cuda.synchronize();start=time.perf_counter()
        try:
            result=readout.episodes([rows],complete_returns=[returns])[0]
        finally:
            runner.attribute=native_attribute;readout.prefix_lease_factory=None
        torch.cuda.synchronize()
        vectors[label]={key:torch.stack([r[key] for r in result])
                        for key in ('dt_token_advantages','dt_q_estimates','dt_v_estimates')}
        reports[label]=dict(total_wall_seconds=time.perf_counter()-start,
            original_readout_report=readout.last_report,
            original_runner_phase_seconds=phase_totals,
            original_runner_phase_counts=phase_counts,
            peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(),
            physical_free_bytes=torch.cuda.mem_get_info()[0],
            pss_bytes=psutil.Process().memory_full_info().pss)
        save('native_prefix_lease_variant_complete',variant=label,
             total_wall_seconds=reports[label]['total_wall_seconds'],
             shared_preparation=readout.last_report.get('shared_native_prefix'))
    observations=[]
    for label in ('original_cold','shared_cold','shared_warm'):
        observations.append(dict(variant=label,values=[dict(key=key,
            equal=bool(torch.equal(vectors[label][key],vectors['original_warm'][key])),
            maximum_absolute_difference=float((vectors[label][key].double()-vectors['original_warm'][key].double()).abs().max()))
            for key in vectors[label]]))
    tensor_path=Path(out)/f'prefix-lease-vectors-rank{torch.distributed.get_rank()}.pt'
    torch.save(vectors,tensor_path)
    save('native_prefix_lease_diagnostic_complete',reports=reports,raw_value_observations=observations,
         vectors=dict(path=str(tensor_path),sha256=hashlib.sha256(tensor_path.read_bytes()).hexdigest()))

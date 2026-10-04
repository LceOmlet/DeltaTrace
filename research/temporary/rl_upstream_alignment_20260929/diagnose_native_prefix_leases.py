"""Same-workload DT observation; no PPO or new numerical acceptance policy."""
import hashlib
import importlib.util
from contextlib import nullcontext
import os
from pathlib import Path
import time
import types

import psutil
import torch


def select_component_request(all_requests, *, offset, limit, request_index):
    """Select an observed row inside its original complete B4 consumer.

    The absolute index is into the saved, context-length-sorted requests. The
    caller still passes the complete bank to the existing lease producer.
    """
    if offset < 0 or offset % 4 or limit != 4:
        raise ValueError('Component diagnosis must select one original complete B4')
    requests = all_requests[offset:offset+limit]
    if len(requests) != 4 or not offset <= request_index < offset+limit:
        raise ValueError('Observed request index is outside the selected original B4')
    return requests, request_index-offset, offset//4


def diagnose(runner, producer, out, save, *, cache_tensors=None):
    from native_prefix_leases import prepare_native_prefix_leases
    from reward_readout import EventRatioReadout

    path=Path(out)/f'actual-requests-rank{torch.distributed.get_rank()}.pt'
    payload=torch.load(path,map_location='cpu',weights_only=False)
    limit=int(os.environ['DT_PREFIX_DIAGNOSTIC_ROWS'])
    all_requests=sorted(payload['requests'],key=lambda request:request['context_tokens'])
    phase_only=os.environ.get('DT_PREFIX_PHASE_ONLY')=='1'
    offset=int(os.environ.get('DT_PREFIX_DIAGNOSTIC_OFFSET','0'))
    requests=all_requests[offset:offset+limit]
    if os.environ.get('DT_PREFIX_LEASE_COMPONENT_DIAGNOSTIC') == '1':
        # Reuse the actual adapter's capture/group/padding decisions. Do not
        # reconstruct a smaller bank, which would change native B4 geometry.
        # The recorded residual selects an explicit absolute sorted row; no
        # request or GDN layer from a historical comparison is guessed here.
        request_index = int(os.environ['DT_PREFIX_COMPONENT_REQUEST_INDEX'])
        requests, local_row, batch_index = select_component_request(
            all_requests, offset=offset, limit=limit, request_index=request_index)
        leases, preparation = prepare_native_prefix_leases(
            runner, all_requests, minibatch_size=4, eos_token_id=producer.readout_tokenizer.eos_token_id)
        lease = leases[batch_index]
        source, source_row = lease.sources[local_row]
        prefix = lease.prefix_length
        ids = source.input_ids.to(runner.model.execution_device)
        comparison = torch.stack([request['prompt'][:prefix] for request in requests]).to(ids.device)
        assert torch.equal(ids[source_row, :prefix], comparison[local_row])
        prepared_cache = lease(comparison)
        prepared_fields = cache_tensors(prepared_cache)
        del prepared_cache
        save('actual_lease_component_inputs', original_request_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
             request_index=request_index, local_row=local_row, consumer_batch_index=batch_index,
             complete_capture_bank_rows=len(all_requests),
             source_row=source_row, source_step=requests[local_row]['source_step'],
             traj_uid=requests[local_row]['traj_uid'],
             source_start=requests[local_row]['start'], context_length=requests[local_row]['context_tokens'],
             capture_input_shape=list(ids.shape), native_input_shape=list(comparison.shape),
             prefix=prefix, matched_rows=[[source_row,local_row]], preparation=preparation,
             scope='Actual prepared lease source and original B4 prefix; original model, FA/FLA reference and assertions; no production deployment')
        del source, lease, leases
        from diagnose_native_prefix_components import diagnose as components
        return components(runner, ids, prefix, save,
                          comparison_ids=comparison, matched_rows=[(source_row,local_row)],
                          prepared_prefix_fields=prepared_fields, cache_tensors=cache_tensors,
                          gdn_layer_index=None, operand_output_dir=Path(out))
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
    warm_phase=os.environ.get('DT_PREFIX_PHASE_WARM')=='1'
    labels=('original_phase','shared_phase') if phase_only and not warm_phase else (
        'original_cold','original_warm','shared_cold','shared_warm')
    if os.environ.get('DT_PREFIX_REVERSE_PREFETCH')=='1':
        labels=(*labels,'prefetch_warm')
    reference_label='original_phase' if phase_only and not warm_phase else 'original_warm'
    shared_bank=None
    def shared_factory(*args,**kwargs):
        nonlocal shared_bank
        reused=shared_bank is not None
        if not reused:
            if not phase_only:
                shared_bank=prepare_native_prefix_leases(*args,**kwargs)
            else:
                # Preserve the full capture geometry and original B4 selection.
                leases,preparation=prepare_native_prefix_leases(args[0],all_requests,**kwargs)
                selected=leases[offset//4:(offset+limit)//4]
                shared_bank=(selected,{**preparation,'diagnostic_selected_consumer_batches':len(selected)})
        # The owner lease creates a fresh DynamicCache on every consumption.
        # Reuse only its immutable source artifacts during this diagnosis;
        # capture_and_preparation_seconds remains the initial bank cost.
        leases,preparation=shared_bank
        return leases,{**preparation,'diagnostic_bank_reused':reused}
    for label in labels:
        if label.startswith(('shared','prefetch')):
            runner.attribute=types.MethodType(module.Qwen35DenseFiniteRunner.attribute,runner)
            readout.prefix_lease_factory=shared_factory
        else:
            runner.attribute=native_attribute;readout.prefix_lease_factory=None
        selected_attribute=runner.attribute
        phase_totals={}
        phase_counts={}
        attribute_walls=[]
        def observe_attribute(*args,**kwargs):
            profile_this_call=(os.environ.get('DT_PREFIX_HOT_PROFILE')=='1'
                               and label.endswith('_warm'))
            projection_audit=None
            if os.environ.get('DT_PREFIX_PROJECTION_INPUTS')=='1' and label=='shared_warm':
                from diagnose_native_projection_inputs import NativeProjectionInputAudit
                projection_audit=NativeProjectionInputAudit(runner.model.model.language_model.layers)
            projection_context=projection_audit if projection_audit is not None else nullcontext()
            prefetch_candidate=None
            if label=='prefetch_warm':
                from native_reverse_prefetch_candidate import NativeReversePrefetch
                prefetch_candidate=NativeReversePrefetch(runner.model)
            prefetch_context=prefetch_candidate if prefetch_candidate is not None else nullcontext()
            root_inventory = None
            inventory_factory = None
            if os.environ.get('DT_PREFIX_ROOT_CAPTURE_INVENTORY')=='1' and label=='shared_warm':
                from diagnose_qwen35_root_capture_inventory import NativeRootCaptureInventory
                from native_root_capture_inventory_factory import OriginalRootCaptureFactory
                owner_globals = selected_attribute.__func__.__globals__
                inventory_factory = OriginalRootCaptureFactory(runner, owner_globals)
                root_inventory = NativeRootCaptureInventory(
                    runner.model.model.language_model.layers, inventory_factory)
            inventory_context = root_inventory if root_inventory is not None else nullcontext()
            context=(torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
                record_shapes=False,with_stack=False,profile_memory=False)
                if profile_this_call else nullcontext())
            with context as profile, projection_context, prefetch_context, inventory_context:
                handles=[];ranges={};passes={}
                if profile_this_call:
                    # Native layer hooks add profiler ranges only. Prefix,
                    # root, and replay still call the exact original forward.
                    for index,layer in enumerate(runner.model.model.language_model.layers):
                        def enter(_module,_args,index=index):
                            passes[index]=passes.get(index,0)+1
                            value=torch.profiler.record_function(
                                f'DT_native_layer_{index}_pass_{passes[index]}')
                            value.__enter__();ranges[index]=value
                        def leave(_module,_args,_output,index=index):
                            value=ranges.pop(index,None)
                            if value is not None:value.__exit__(None,None,None)
                        handles.append(layer.register_forward_pre_hook(enter))
                        handles.append(layer.register_forward_hook(leave,always_call=True))
                try:
                    attribute_started=time.perf_counter()
                    result=selected_attribute(*args,**kwargs)
                    # The original runner owns its completion synchronization.
                    # Excludes bank preparation and later JSON/trace export.
                    attribute_walls.append(time.perf_counter()-attribute_started)
                finally:
                    for handle in handles:handle.remove()
                    for value in reversed(list(ranges.values())):value.__exit__(None,None,None)
            if projection_audit is not None:
                import json
                observation=projection_audit.report()
                observation.update(
                    variant=label,rank=torch.distributed.get_rank(),
                    scope='Inputs observed during one existing paired root and reverse layer replay; original forward unchanged; CPU copies are diagnostic overhead',
                    source=dict(path=__file__,sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
                path=Path(out)/f'projection-inputs-rank{torch.distributed.get_rank()}.json'
                path.write_text(json.dumps(observation,indent=2)+'\n')
                save('native_projection_input_comparison',
                     receipt=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()),
                     report=observation)
            if prefetch_candidate is not None:
                import json
                observation=prefetch_candidate.report()
                observation.update(variant=label,rank=torch.distributed.get_rank(),
                    scope='One original shared warm paired endpoint and reverse replay; only original FSDP forward prefetch setting changed during replay')
                path=Path(out)/f'reverse-prefetch-rank{torch.distributed.get_rank()}.json'
                path.write_text(json.dumps(observation,indent=2)+'\n')
                save('native_reverse_prefetch_observation',
                     receipt=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()),
                     report=observation)
            if root_inventory is not None:
                import json
                observation = root_inventory.report()
                observation.update(variant=label, rank=torch.distributed.get_rank(),
                    original_runner_gdn_coefficient_start=result[1].get('gdn_fla_coefficient_start'),
                    original_runner_native_prefix_length=result[1].get('native_shared_prefix_length'),
                    original_capture_api=inventory_factory.provenance(),
                    source=dict(path=__file__,sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
                path=Path(out)/f'root-capture-inventory-rank{torch.distributed.get_rank()}.json'
                path.write_text(json.dumps(observation,indent=2)+'\n')
                save('native_root_capture_inventory',
                     receipt=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()),
                     summary={k:v for k,v in observation.items() if k!='rows'})
            if profile_this_call:
                # Original PyTorch profiler observes one already-scheduled
                # warm B4. Preparation is outside attribute; these instrumented
                # timings are not another speed-comparison receipt.
                trace=Path(out)/f'hot-{label}-rank{torch.distributed.get_rank()}.json'
                profile.export_chrome_trace(str(trace))
                records=[dict(name=event.key,count=event.count,
                    self_cpu_microseconds=event.self_cpu_time_total,
                    cpu_microseconds=event.cpu_time_total,
                    self_device_microseconds=getattr(event,'self_device_time_total',0),
                    device_microseconds=getattr(event,'device_time_total',0))
                    for event in profile.key_averages()]
                save('native_hot_b4_profile',variant=label,
                     scope='One actual warm attribute B4; owner computations unchanged; excludes bank preparation; instrumented costs only',
                     trace=dict(path=str(trace),sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),bytes=trace.stat().st_size),
                     events=records)
            for call in result[1].get('calls',[]):
                kind=call['kind']
                seconds=call.get('stream_elapsed_seconds',call.get('seconds'))
                if seconds is not None:
                    phase_totals[kind]=phase_totals.get(kind,0.0)+seconds
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
            original_attribute_wall_seconds=attribute_walls,
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
    for label in labels:
        if label==reference_label:
            continue
        observations.append(dict(variant=label,values=[dict(key=key,
            equal=bool(torch.equal(vectors[label][key],vectors[reference_label][key])),
            maximum_absolute_difference=float((vectors[label][key].double()-vectors[reference_label][key].double()).abs().max()))
            for key in vectors[label]]))
    if 'prefetch_warm' in vectors:
        observations.append(dict(variant='prefetch_warm',comparison_reference='shared_warm',
            scope='Same prefix reuse and original input geometry; changes scheduling only',
            values=[dict(key=key,
                equal=bool(torch.equal(vectors['prefetch_warm'][key],vectors['shared_warm'][key])),
                maximum_absolute_difference=float((vectors['prefetch_warm'][key].double()-vectors['shared_warm'][key].double()).abs().max()))
                for key in vectors['prefetch_warm']]))
    tensor_path=Path(out)/f'prefix-lease-vectors-rank{torch.distributed.get_rank()}.pt'
    torch.save(vectors,tensor_path)
    save('native_prefix_lease_diagnostic_complete',reports=reports,raw_value_observations=observations,
         vectors=dict(path=str(tensor_path),sha256=hashlib.sha256(tensor_path.read_bytes()).hexdigest()))

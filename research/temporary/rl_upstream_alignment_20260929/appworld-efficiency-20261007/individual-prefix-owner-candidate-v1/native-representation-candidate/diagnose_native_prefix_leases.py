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


def native_bank_inventory(leases, requests, *, minibatch_size):
    """Read existing artifact metadata; never copy or consume its tensors."""
    artifacts={}; consumed=set(); source_slots=0; consumed_gdn_row_checks=0
    for batch_index,lease in enumerate(leases):
        batch=requests[batch_index*minibatch_size:(batch_index+1)*minibatch_size]
        if lease is None:
            continue
        assert len(lease.sources)==len(batch)
        for request,(artifact,row) in zip(batch,lease.sources):
            length=int(lease.prefix_length)
            assert 0<=row<artifact.input_ids.shape[0]
            assert torch.equal(artifact.input_ids[row,:length],request['prompt'][:length].detach().cpu())
            row_map=getattr(artifact,'boundary_rows',None)
            packed_row=row if row_map is None else row_map[length][row]
            for layer in artifact.layers:
                if 'boundaries' in layer:
                    conv,state=layer['boundaries'][length]
                    assert 0<=packed_row<conv.shape[0] and packed_row<state.shape[0]
                    consumed_gdn_row_checks+=1
            identity=id(artifact)
            if identity not in artifacts:
                artifacts[identity]=(len(artifacts),artifact)
            consumed.add((identity,int(lease.prefix_length),int(row)))
            source_slots+=1
    storages={}
    def tensor_metadata(tensor):
        storage=tensor.untyped_storage()
        identity=(str(tensor.device),storage.data_ptr())
        storages.setdefault(identity,storage.nbytes())
        return dict(shape=list(tensor.shape),stride=list(tensor.stride()),
            dtype=str(tensor.dtype),device=str(tensor.device),
            pinned_host=tensor.is_pinned() if tensor.device.type=='cpu' else False,
            logical_bytes=tensor.numel()*tensor.element_size(),
            storage_bytes=storage.nbytes())
    totals=dict(input_ids_bytes=0,fa_keys_bytes=0,fa_values_bytes=0,
        gdn_conv_bytes=0,gdn_state_bytes=0,gdn_stored_boundary_rows=0,
        gdn_consumed_boundary_rows=0,gdn_consumed_conv_bytes=0,gdn_consumed_state_bytes=0)
    records=[]
    for identity,(index,artifact) in artifacts.items():
        pairs={(boundary,row) for source,boundary,row in consumed if source==identity}
        ids=tensor_metadata(artifact.input_ids)
        totals['input_ids_bytes']+=ids['logical_bytes']
        layers=[]
        for layer_index,layer in enumerate(artifact.layers):
            if 'boundaries' in layer:
                boundaries=[]
                for boundary,(conv,state) in layer['boundaries'].items():
                    conv_meta=tensor_metadata(conv);state_meta=tensor_metadata(state)
                    rows=sorted(row for length,row in pairs if length==int(boundary))
                    totals['gdn_conv_bytes']+=conv_meta['logical_bytes']
                    totals['gdn_state_bytes']+=state_meta['logical_bytes']
                    totals['gdn_stored_boundary_rows']+=conv.shape[0]
                    totals['gdn_consumed_boundary_rows']+=len(rows)
                    totals['gdn_consumed_conv_bytes']+=conv_meta['logical_bytes']//conv.shape[0]*len(rows)
                    totals['gdn_consumed_state_bytes']+=state_meta['logical_bytes']//state.shape[0]*len(rows)
                    row_map=getattr(artifact,'boundary_rows',None)
                    stored_original_rows=(sorted(row_map[int(boundary)],key=row_map[int(boundary)].get)
                        if row_map is not None else list(range(conv.shape[0])))
                    boundaries.append(dict(prefix_length=int(boundary),consumed_rows=rows,
                        stored_original_rows=stored_original_rows,
                        conv=conv_meta,state=state_meta))
                layers.append(dict(index=layer_index,kind='GDN',boundary_count=len(boundaries),
                    boundaries=boundaries))
            else:
                keys=tensor_metadata(layer['keys']);values=tensor_metadata(layer['values'])
                totals['fa_keys_bytes']+=keys['logical_bytes']
                totals['fa_values_bytes']+=values['logical_bytes']
                layers.append(dict(index=layer_index,kind='FA',keys=keys,values=values))
        records.append(dict(artifact_index=index,input_ids=ids,
            consumed_boundary_rows=[dict(prefix_length=n,row=row) for n,row in sorted(pairs)],layers=layers))
    totals['gdn_unconsumed_boundary_rows']=totals['gdn_stored_boundary_rows']-totals['gdn_consumed_boundary_rows']
    totals['gdn_unconsumed_conv_bytes']=totals['gdn_conv_bytes']-totals['gdn_consumed_conv_bytes']
    totals['gdn_unconsumed_state_bytes']=totals['gdn_state_bytes']-totals['gdn_consumed_state_bytes']
    totals['logical_tensor_bytes']=sum(totals[key] for key in (
        'input_ids_bytes','fa_keys_bytes','fa_values_bytes','gdn_conv_bytes','gdn_state_bytes'))
    totals['distinct_live_storage_bytes']=sum(storages.values())
    return dict(scope='Complete originally constructed bank before bounded B4 selection; consumption counts cover all original requests, not only the diagnostic subset. Unconsumed rows are retained representation, not evidence of a leak.',
        original_requests=len(requests),original_consumer_batches=len(leases),source_slots=source_slots,
        distinct_artifacts=len(artifacts),distinct_consumed_artifact_boundary_rows=len(consumed),
        lease_source_checks=dict(factual_id_prefixes_equal=source_slots,
            gdn_original_to_stored_rows_in_range=consumed_gdn_row_checks,
            scope='Every original lease source and needed GDN boundary checked without materializing a cache or running a model'),
        totals=totals,artifacts=records)


def _prepare_native_conv_capacity_inputs(readout, requests, payload, out, rank):
    """Call the unchanged existing fixture and current readout input owner."""
    from verify_dt_context_capacity import capacity_fixture

    capacity_episodes, capacity_returns, capacity_details = [], [], []
    capacity_factual_inputs, capacity_requests = [], []
    assert len(requests) == 4
    for capacity_index, capacity_request in enumerate(requests):
        capacity_original = payload['rows'][capacity_request['row_index']]
        capacity_coefficient = capacity_request['observed_return']
        assert capacity_coefficient != 0 and bool(capacity_original['active_masks'])
        capacity_source = {**capacity_original, 'rewards': capacity_coefficient,
            'traj_uid': f'exact32k-conv-capacity-rank{rank}-row{capacity_index}'}
        capacity_row, capacity_factual, capacity_detail, _ = capacity_fixture(
            capacity_source, readout.tokenizer, readout.alphabet, response_tokens=512,
            max_steps=readout.max_steps, sampling=readout.sampling)
        capacity_report = dict(nonzero_reward_events=0, policy_tokens=0, actual_row_lengths=[])
        _, capacity_pending = readout._prepare_episode(
            [capacity_row], capacity_report, [capacity_coefficient])
        assert len(capacity_pending) == 1
        capacity_actual = capacity_pending[0]
        capacity_prepared = torch.cat([capacity_actual[key] for key in ('prompt','actions','query','target')])
        assert capacity_actual['context_tokens'] == capacity_prepared.numel() == 32768
        assert torch.equal(capacity_prepared, capacity_factual)
        assert capacity_actual['actions'].numel() == 512
        assert capacity_actual['observed_return'] == capacity_coefficient
        capacity_detail = {key: value.item() if isinstance(value,torch.Tensor) and value.ndim==0 else value
                           for key,value in capacity_detail.items()}
        capacity_detail['original_source'] = dict(row_index=capacity_request['row_index'],
            traj_uid=capacity_request['traj_uid'], source_step=capacity_request['source_step'])
        capacity_episodes.append([capacity_row]); capacity_returns.append([capacity_coefficient])
        capacity_details.append(capacity_detail); capacity_factual_inputs.append(capacity_factual)
        capacity_requests.append(capacity_actual)
    assert len({episode[0]['traj_uid'] for episode in capacity_episodes}) == 4
    capacity_path = Path(out)/f'exact32k-conv-inputs-rank{rank}.pt'
    torch.save(dict(episodes=capacity_episodes, complete_returns=capacity_returns,
                    factual_inputs=capacity_factual_inputs, details=capacity_details), capacity_path)
    capacity_receipt = dict(path=str(capacity_path), sha256=hashlib.sha256(capacity_path.read_bytes()).hexdigest())
    return (capacity_episodes, capacity_returns, capacity_requests, capacity_factual_inputs,
            capacity_details, capacity_receipt)


def diagnose(runner, producer, out, save, *, cache_tensors=None):
    from native_prefix_leases import prepare_native_prefix_leases
    from reward_readout import EventRatioReadout

    path=Path(out)/f'actual-requests-rank{torch.distributed.get_rank()}.pt'
    payload=torch.load(path,map_location='cpu',weights_only=False)
    base_prefetch=os.environ.get('DT_PREFIX_BASE_MODEL_PREFETCH')=='1'
    individual_rows=os.environ.get('DT_PREFIX_INDIVIDUAL_ROW_CANDIDATE')=='1'
    conv_initial_states=os.environ.get('DT_PREFIX_NATIVE_CONV_INITIAL_STATES')=='1'
    conv_capacity=os.environ.get('DT_PREFIX_NATIVE_CONV_CAPACITY')=='1'
    boundary_row_storage=os.environ.get('DT_PREFIX_BOUNDARY_ROW_STORAGE')=='1'
    if boundary_row_storage:
        assert base_prefetch and os.environ.get('DT_PREFIX_CURRENT_FORMAL_OWNER')=='1'
        assert os.environ.get('DT_PREFIX_PHASE_ONLY')=='1' and os.environ.get('DT_PREFIX_PHASE_WARM')=='1'
        assert not os.environ.get('DT_PREFIX_CHECKPOINT')
        assert not any(os.environ.get(key)=='1' for key in (
            'DT_PREFIX_NATIVE_BACKWARD','DT_PREFIX_HOT_PROFILE','DT_PREFIX_REVERSE_PREFETCH',
            'DT_PREFIX_NATIVE_CONV_INITIAL_STATES','DT_PREFIX_NATIVE_CONV_CAPACITY','DT_PREFIX_LEDGER_ONLY',
            'DT_PREFIX_PROJECTION_INPUTS','DT_PREFIX_ROOT_CAPTURE_INVENTORY','DT_PREFIX_ROOT_TAPE',
            'DT_PREFIX_ROOT_TAPE_CPU','DT_PREFIX_ROOT_TAPE_GDN0','DT_PREFIX_ROOT_TAPE_FA3',
            'DT_PREFIX_ROOT_TAPE_HOT','DT_PREFIX_LEASE_COMPONENT_DIAGNOSTIC'))
    if conv_capacity:
        assert conv_initial_states
    if conv_initial_states:
        assert base_prefetch and os.environ.get('DT_PREFIX_CURRENT_FORMAL_OWNER')=='1'
        assert os.environ.get('DT_PREFIX_REVERSE_PREFETCH')!='1'
        assert not os.environ.get('DT_PREFIX_CHECKPOINT')
    current_request_origin=None
    if base_prefetch:
        assert os.environ.get('DT_PREFIX_CURRENT_FORMAL_OWNER')=='1'
        assert (individual_rows or os.environ.get('DT_PREFIX_REVERSE_PREFETCH')=='1' or conv_initial_states or boundary_row_storage
                or (os.environ.get('DT_PREFIX_HOT_PROFILE')=='1'
                    and os.environ.get('DT_PREFIX_PHASE_ONLY')=='1'
                    and os.environ.get('DT_PREFIX_PHASE_WARM')=='1'))
        assert not os.environ.get('DT_PREFIX_CHECKPOINT')
        # The frozen rows predate the accepted response-clock query repair.
        # Reuse the actual current owner to emit its query/case metadata; keep
        # every literal policy prefix, source action and complete return.
        readout=EventRatioReadout(runner,producer.readout_tokenizer,**producer.readout_options,
                                appworld_num_tests=int(payload['rows'][0]['appworld_num_tests']))
        metadata_report=dict(nonzero_reward_events=0,policy_tokens=0,actual_row_lengths=[])
        _,current_requests=readout._prepare_episode(
            payload['rows'],metadata_report,payload['complete_returns'])
        saved_by_row={request['row_index']:request for request in payload['requests']}
        assert len(saved_by_row)==len(current_requests)==len(payload['requests'])
        query_changes=[]
        for request in current_requests:
            saved=saved_by_row[request['row_index']]
            for field in ('prompt','actions','target'):
                assert torch.equal(request[field],saved[field]),field
            assert (request['traj_uid'],request['source_step'],request['start'],request['end'],
                    request['observed_return']) == (saved['traj_uid'],saved['source_step'],
                    saved['start'],saved['end'],saved['observed_return'])
            query_changes.append(dict(row_index=request['row_index'],
                saved_query_tokens=saved['query'].numel(),current_query_tokens=request['query'].numel(),
                saved_query_sha256=hashlib.sha256(saved['query'].numpy().tobytes()).hexdigest(),
                current_query_sha256=hashlib.sha256(request['query'].numpy().tobytes()).hexdigest(),
                query_equal=bool(torch.equal(request['query'],saved['query']))))
        saved_order=[request['row_index'] for request in sorted(
            payload['requests'],key=lambda request:request['context_tokens'])]
        current_order=[request['row_index'] for request in sorted(
            current_requests,key=lambda request:request['context_tokens'])]
        assert current_order==saved_order,'Current query changed the original B4 request grouping'
        current_request_origin=dict(owner_method='EventRatioReadout._prepare_episode',
            policy_ids_and_returns_unchanged=True,request_count=len(current_requests),
            original_B4_request_order_preserved=True,original_context_sorted_row_indices=saved_order,
            query_id_hash_representation='CPU int64 tensor bytes',query_changes=query_changes,
            scope='CPU request metadata preparation through the current frozen owner; not a model call, credit result or historical query parity claim')
    limit=int(os.environ['DT_PREFIX_DIAGNOSTIC_ROWS'])
    all_requests=sorted(current_requests if base_prefetch else payload['requests'],
                        key=lambda request:request['context_tokens'])
    phase_only=os.environ.get('DT_PREFIX_PHASE_ONLY')=='1'
    offset=int(os.environ.get('DT_PREFIX_DIAGNOSTIC_OFFSET','0'))
    requests=all_requests[offset:offset+limit]
    if conv_capacity:
        capacity_source_offset=offset
        (capacity_episodes, capacity_returns, capacity_requests, capacity_factual_inputs,
         capacity_details, capacity_receipt)=_prepare_native_conv_capacity_inputs(
            readout,requests,payload,out,torch.distributed.get_rank())
        all_requests=capacity_requests;requests=capacity_requests;offset=0
        save('native_conv_exact32k_capacity_inputs',source_sorted_offset=capacity_source_offset,
             fixture_details=capacity_details,literal_inputs=capacity_receipt,
             sampling=readout.sampling,max_steps=readout.max_steps,
             diagnostic_scope='Existing synthetic exact32768 fixture, response512, B4/rank; recorded return is a capacity coefficient, not a synthetic task reward or training-effect claim')
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
    if conv_capacity:
        rows=[episode[0] for episode in capacity_episodes]
        returns=[episode[0] for episode in capacity_returns]
    else:
        indices=[request['row_index'] for request in requests]
        rows=[payload['rows'][i] for i in indices]
        returns=[payload['complete_returns'][i] for i in indices]
    if not base_prefetch:
        readout=EventRatioReadout(runner,producer.readout_tokenizer,**producer.readout_options,
                                appworld_num_tests=int(rows[0]['appworld_num_tests']))
    assert readout.minibatch_size==4 and readout.max_length==32768
    assert readout.alphabet.label_ids(readout.tokenizer)==payload['outcome_token_ids']
    native_prepare=readout._prepare_episode
    expected={(r['traj_uid'],r['source_step']):r for r in requests}
    def verify_prepare(*args,**kwargs):
        values,pending=native_prepare(*args,**kwargs)
        if conv_capacity:
            assert len(pending)==1
            capacity_actual=pending[0]
            capacity_index=next(i for i,episode in enumerate(capacity_episodes)
                                if episode[0]['traj_uid']==capacity_actual['traj_uid'])
            capacity_prepared=torch.cat([capacity_actual[key] for key in ('prompt','actions','query','target')])
            assert capacity_actual['context_tokens']==capacity_prepared.numel()==32768
            assert torch.equal(capacity_prepared,capacity_factual_inputs[capacity_index])
            assert capacity_actual['actions'].numel()==512
            assert capacity_actual['observed_return']==capacity_returns[capacity_index][0]
            return values,pending
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
    original_request_path=path
    current_owner=os.environ.get('DT_PREFIX_CURRENT_FORMAL_OWNER')=='1'
    ledger_only=os.environ.get('DT_PREFIX_LEDGER_ONLY')=='1'
    if ledger_only:
        assert current_owner and phase_only and os.environ.get('DT_PREFIX_PHASE_WARM')=='1'
        assert not any(os.environ.get(key)=='1' for key in (
            'DT_PREFIX_HOT_PROFILE','DT_PREFIX_NATIVE_BACKWARD','DT_PREFIX_PROJECTION_INPUTS',
            'DT_PREFIX_REVERSE_PREFETCH','DT_PREFIX_ROOT_CAPTURE_INVENTORY','DT_PREFIX_ROOT_TAPE',
            'DT_PREFIX_ROOT_TAPE_CPU','DT_PREFIX_ROOT_TAPE_GDN0','DT_PREFIX_ROOT_TAPE_FA3',
            'DT_PREFIX_ROOT_TAPE_HOT','DT_PREFIX_LEASE_COMPONENT_DIAGNOSTIC'))
    native_backward_inputs=None
    module=None
    if not current_owner:
        source=Path(out)/'qwen35_dense_finite_runner_candidate.py'
        spec=importlib.util.spec_from_file_location('_prepared_prefix_lease_runner',source)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    save('native_prefix_lease_diagnostic_start',actual_rows=limit,
         context_lengths=[r['context_tokens'] for r in requests],
         original_request_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
         current_request_origin=current_request_origin,
         native_conv_initial_states=getattr(runner,'native_conv_initial_states',False),
         diagnostic_scope=('Existing synthetic exact32768/response512 capacity input with current original readout; no checkpoint, optimizer, task-performance or invented full-network tolerance.'
             if conv_capacity else 'Current frozen formal owner and original checkpoint loader; raw residuals have no invented full-network tolerance.'
             if current_owner and not base_prefetch else 'Current frozen owner on base-model initialization; same saved literal policy IDs and current owner query; no checkpoint load or invented full-network tolerance.'
             if base_prefetch else 'Original readout/QVA consumers; raw residuals have no invented full-network tolerance. Loaded actor is not a restored formal checkpoint.'),
         restored_checkpoint=os.environ.get('DT_PREFIX_CHECKPOINT'))
    warm_phase=os.environ.get('DT_PREFIX_PHASE_WARM')=='1'
    labels=('original_phase','shared_phase') if phase_only and not warm_phase else (
        'original_cold','original_warm','shared_cold','shared_warm')
    if current_owner:
        labels=(('shared_cold','shared_warm')
                if base_prefetch and os.environ.get('DT_PREFIX_HOT_PROFILE')!='1'
                else ('shared_cold','shared_warm','shared_profile_warm'))
    if individual_rows:
        assert current_owner and base_prefetch and phase_only and warm_phase
        assert not os.environ.get('DT_PREFIX_CHECKPOINT')
        assert not any(os.environ.get(key)=='1' for key in (
            'DT_PREFIX_HOT_PROFILE','DT_PREFIX_NATIVE_BACKWARD','DT_PREFIX_REVERSE_PREFETCH',
            'DT_PREFIX_NATIVE_CONV_INITIAL_STATES','DT_PREFIX_NATIVE_CONV_CAPACITY',
            'DT_PREFIX_BOUNDARY_ROW_STORAGE','DT_PREFIX_LEDGER_ONLY','DT_PREFIX_ROOT_TAPE'))
        import inspect,importlib,json,sys
        prepared=json.loads((Path(out)/'prepared.json').read_bytes())
        owner=prepared['individual_row_candidate']
        modules={}
        for name in ('runner','finite_wrapper'):
            item=owner[name];source=Path(item['path'])
            assert hashlib.sha256(source.read_bytes()).hexdigest()==item['sha256']
            spec=importlib.util.spec_from_file_location('_isolated_individual_row_'+name,source)
            value=importlib.util.module_from_spec(spec);sys.modules[spec.name]=value
            spec.loader.exec_module(value);modules[name]=value
        row_module=modules['runner'];row_finite=modules['finite_wrapper']
        row_module.RightPaddedLengths=row_finite.RightPaddedLengths
        row_operation=row_finite.VendorFAFiniteP1BF16D256(
            owner['finite_library']['path'],owner['finite_library']['sha256'])
        for name,value in (('artifact',importlib.import_module('qwen35_native_prefix_artifacts')),
                           ('lease',prepare_native_prefix_leases),
                           ('answer',importlib.import_module('qwen35_answer_finite'))):
            source=Path(inspect.getsourcefile(value));item=owner[name]
            assert source.resolve()==Path(item['path']).resolve()
            assert hashlib.sha256(source.read_bytes()).hexdigest()==item['sha256']
        original_finite=runner.finite_fa
        labels=('shared_cold','shared_warm','shared_individual_rows_cold','shared_individual_rows_warm',
                'shared_individual_rows_observed')
        save('individual_row_candidate_sources',sources=owner,
             scope='Isolated actual B8 dispatch and representation comparison; no checkpoint, optimizer, new full-network tolerance or formal deployment')
    if ledger_only:
        labels=('shared_ledger',)
    if boundary_row_storage:
        labels=('shared_cold','shared_warm','shared_boundary_rows_cold','shared_boundary_rows_warm')
        import inspect,importlib,json
        prepared=json.loads((Path(out)/'prepared.json').read_bytes())
        candidate=prepared['boundary_row_storage_candidate']
        sources={}
        for name,value,expected_sha in (
            ('artifact',importlib.import_module('qwen35_native_prefix_artifacts'),candidate['artifact']['sha256']),
            ('lease',prepare_native_prefix_leases,candidate['lease']['sha256']),
            ('runner',type(runner),'e9c7576486f742c26f895cd4078891a98d94c189e84a6e565fc8042a74aabcab'),
            ('gdn_finite',importlib.import_module('qwen35_gdn_finite'),'ef55ce08dec9374304018b43b9f85510fca36054408c439ab1109dca8ac23ca9'),
            ('model',type(runner.model.model.language_model.layers[0].linear_attn),candidate['canonical_HF_owner']['sha256'])):
            source=Path(inspect.getsourcefile(value))
            digest=hashlib.sha256(source.read_bytes()).hexdigest()
            assert digest==expected_sha,(name,str(source),digest)
            sources[name]=dict(path=str(source),resolved_path=str(source.resolve()),sha256=digest)
        assert getattr(runner,'native_conv_initial_states',False) is True
        save('native_boundary_row_storage_sources',sources=sources,
            native_conv_initial_states=runner.native_conv_initial_states,
            scope='One isolated candidate artifact/lease default OFF then ON; current runner/GDN and installed canonical HF unchanged; no checkpoint, optimizer or formal deployment')
    if os.environ.get('DT_PREFIX_REVERSE_PREFETCH')=='1':
        labels=(*labels,'prefetch_warm')
    root_tape_module=None
    if os.environ.get('DT_PREFIX_ROOT_TAPE')=='1':
        source=Path(out)/'root-tape-owner'/'qwen35_dense_finite_runner_root_tape_candidate.py'
        if os.environ.get('DT_PREFIX_ROOT_TAPE_CPU')=='1':
            import sys
            sys.path.insert(0,str(source.parent))
        spec=importlib.util.spec_from_file_location('_isolated_native_root_tape_runner',source)
        root_tape_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(root_tape_module)
        labels=(*labels,'root_tape_disabled_warm','root_tape_cold','root_tape_warm')
    if conv_initial_states:
        # Same initialized base actor, original immutable bank and real
        # B4 geometry. The original preparation happens only once.
        labels=('shared_cold','shared_warm','shared_conv_initial_states_warm')
        assert runner.native_conv_initial_states is False
        import inspect, importlib
        module=runner.model.model.language_model.layers[0].linear_attn
        owners={}
        for name,value,expected_source_sha256 in (
            ('runner',type(runner),'e9c7576486f742c26f895cd4078891a98d94c189e84a6e565fc8042a74aabcab'),
            ('finite',importlib.import_module('qwen35_gdn_finite'),'ef55ce08dec9374304018b43b9f85510fca36054408c439ab1109dca8ac23ca9'),
            ('model',type(module),'59f9c339e3c01672b67ba09b0e56e2e960ba28c6c3d9975faadf00d15ee0b4c8')):
            path_=Path(inspect.getsourcefile(value))
            digest=hashlib.sha256(path_.read_bytes()).hexdigest()
            assert digest==expected_source_sha256,(name,str(path_),digest)
            owners[name]=dict(path=str(path_),sha256=digest)
        save('native_conv_initial_states_candidate_sources',sources=owners,
             default_enabled=False,prefetch_enabled=False,
             scope='Original public initial_states API; same-bank three-variant observation, no new tolerance')
    reference_label='original_phase' if phase_only and not warm_phase else 'original_warm'
    if current_owner:reference_label='shared_warm'
    if ledger_only:reference_label='shared_ledger'
    shared_bank=None
    boundary_row_checks=[]
    def observe_boundary_rows(kind,layer_index,prefix_length,original_rows,full_tensor,saved_tensor):
        # Observe the same capture export, then release each original row.
        # Copies belong to diagnostic bank preparation, never DT timing.
        started=time.perf_counter()
        assert full_tensor.dtype==saved_tensor.dtype
        assert tuple(full_tensor.shape[1:])==tuple(saved_tensor.shape[1:])
        assert saved_tensor.shape[0]==len(original_rows)
        compared_bytes=0
        for packed,original in enumerate(original_rows):
            factual=full_tensor[original].detach().cpu().contiguous().view(torch.uint8)
            exported=saved_tensor[packed].detach().cpu().contiguous().view(torch.uint8)
            assert torch.equal(factual,exported),(kind,layer_index,prefix_length,original)
            compared_bytes+=factual.numel()
            del factual,exported
        boundary_row_checks.append(dict(kind=kind,layer_index=layer_index,prefix_length=prefix_length,
            original_rows=list(original_rows),dtype=str(saved_tensor.dtype),
            saved_shape=list(saved_tensor.shape),logical_element_bytes_compared=compared_bytes,
            equal=True,observer_seconds=time.perf_counter()-started))
    def shared_factory(*args,**kwargs):
        nonlocal shared_bank
        reused=shared_bank is not None
        if not reused:
            if not phase_only:
                shared_bank=prepare_native_prefix_leases(*args,**kwargs)
            else:
                # Preserve the full capture geometry and original B4 selection.
                if ledger_only or boundary_row_storage:
                    def host_observation():
                        observation=dict(pss_bytes=psutil.Process().memory_full_info().pss,
                            scope='Raw original allocator counters only; MetaX allocated counters have a recorded accounting defect and are not interpreted as live bytes. Artifact CPU storage inventory is separate from the pinned allocator.')
                        try:observation['original_host_memory_stats']=torch.cuda.memory.host_memory_stats()
                        except (AttributeError,RuntimeError) as exc:
                            observation['original_host_memory_stats_unavailable']=repr(exc)
                        return observation
                    host_before=host_observation()
                if boundary_row_storage and label.startswith('shared_boundary_rows'):
                    leases,preparation=prepare_native_prefix_leases(args[0],all_requests,**kwargs,
                        boundary_row_storage=True,observe_boundary_rows=observe_boundary_rows)
                else:
                    leases,preparation=prepare_native_prefix_leases(args[0],all_requests,**kwargs,
                        **({'individual_prefixes':True} if individual_rows and label.startswith('shared_individual_rows') else {}))
                if ledger_only or boundary_row_storage:
                    import json
                    host_after=host_observation()
                    inventory=native_bank_inventory(leases,all_requests,minibatch_size=kwargs['minibatch_size'])
                    inventory.update(rank=torch.distributed.get_rank(),
                        variant=label,boundary_row_storage=label.startswith('shared_boundary_rows'),
                        original_request_path=str(original_request_path),
                        original_request_sha256=hashlib.sha256(original_request_path.read_bytes()).hexdigest(),
                        original_preparation=preparation,host_before_prepare=host_before,
                        host_after_prepare=host_after,diagnostic_selected_offset=offset,
                        diagnostic_selected_rows=limit)
                    if boundary_row_storage:
                        inventory['same_capture_row_checks']=list(boundary_row_checks)
                        checked_rows_by_kind={kind:sum(len(check['original_rows'])
                            for check in boundary_row_checks if check['kind']==kind) for kind in ('conv','state')}
                        if label.startswith('shared_boundary_rows'):
                            assert all(count==inventory['totals']['gdn_consumed_boundary_rows']
                                for count in checked_rows_by_kind.values())
                        inventory['same_capture_row_check_summary']=dict(
                            exported_boundaries=len(boundary_row_checks),
                            exported_rows=sum(len(check['original_rows']) for check in boundary_row_checks),
                            checked_rows_by_kind=checked_rows_by_kind,
                            logical_element_bytes_compared=sum(check['logical_element_bytes_compared'] for check in boundary_row_checks),
                            observer_seconds=sum(check['observer_seconds'] for check in boundary_row_checks),
                            scope='Exact same-capture logical element bytes, dtype and shape of each declared consumed row; contiguous observer copies only. Copy time is included in capture_and_preparation_seconds, not attribute. No new floating tolerance.')
                    inventory_name=(f'prefix-bank-inventory-{label}' if boundary_row_storage else 'original-prefix-bank-inventory')
                    inventory_path=Path(out)/f'{inventory_name}-rank{torch.distributed.get_rank()}.json'
                    inventory_path.write_text(json.dumps(inventory,indent=2)+'\n')
                    save('native_original_prefix_bank_inventory',receipt=dict(path=str(inventory_path),
                        sha256=hashlib.sha256(inventory_path.read_bytes()).hexdigest()),
                        variant=label,totals=inventory['totals'],host_before_prepare=host_before,host_after_prepare=host_after,
                        same_capture_row_check_summary=inventory.get('same_capture_row_check_summary'))
                selected=leases[offset//4:(offset+limit)//4]
                shared_bank=(selected,{**preparation,'diagnostic_selected_consumer_batches':len(selected)})
        # The owner lease creates a fresh DynamicCache on every consumption.
        # Reuse only its immutable source artifacts during this diagnosis;
        # capture_and_preparation_seconds remains the initial bank cost.
        leases,preparation=shared_bank
        return leases,{**preparation,'diagnostic_bank_reused':reused}
    for label in labels:
        if individual_rows and label=='shared_individual_rows_cold':
            shared_bank=None
            import gc
            gc.collect()
        if boundary_row_storage and label=='shared_boundary_rows_cold':
            # Do not retain original/full and compact banks simultaneously.
            # Pinned allocator retention remains visible in the PSS receipt.
            shared_bank=None
            boundary_row_checks.clear()
        save('native_prefix_lease_variant_start',variant=label,
             shared_bank_already_prepared=shared_bank is not None)
        previous_conv_option=getattr(runner,'native_conv_initial_states',False)
        if conv_initial_states:
            runner.native_conv_initial_states=label=='shared_conv_initial_states_warm'
            save('native_conv_initial_states_execution_option',variant=label,
                 native_conv_initial_states=runner.native_conv_initial_states)
        previous_class=runner.__class__
        if individual_rows and label.startswith('shared_individual_rows'):
            runner.__class__=row_module.Qwen35DenseFiniteRunner
            runner.finite_fa=row_operation
        previous_capture_backend=runner.capture_backend
        had_root_flag=hasattr(runner,'reuse_root_captures')
        previous_root_flag=getattr(runner,'reuse_root_captures',False)
        if label.startswith('root_tape'):
            # Same initialized owner/model/compiled finite objects. Only the
            # isolated owner dispatch seam changes, and is restored below.
            runner.__class__=root_tape_module.Qwen35DenseFiniteRunner
            runner.reuse_root_captures=label!='root_tape_disabled_warm'
            if os.environ.get('DT_PREFIX_ROOT_TAPE_CPU')=='1':
                # Import binding only: original retained/local-event bodies,
                # inheriting the same private transport owners as this runner.
                runner.capture_backend=root_tape_module._root_capture_backend
        if label.startswith(('shared','prefetch','root_tape')):
            if current_owner:
                runner.attribute=(types.MethodType(row_module.Qwen35DenseFiniteRunner.attribute,runner)
                    if individual_rows and label.startswith('shared_individual_rows') else native_attribute)
            else:
                owner=(root_tape_module if label.startswith('root_tape') else module)
                runner.attribute=types.MethodType(owner.Qwen35DenseFiniteRunner.attribute,runner)
            readout.prefix_lease_factory=shared_factory
        else:
            runner.attribute=native_attribute;readout.prefix_lease_factory=None
        selected_attribute=runner.attribute
        phase_totals={}
        phase_counts={}
        attribute_walls=[]
        root_tape_reports=[]
        gdn0_observations=[]
        fa3_observations=[]
        ledger_receipts=[]
        def observe_attribute(*args,**kwargs):
            nonlocal native_backward_inputs
            if (current_owner and os.environ.get('DT_PREFIX_NATIVE_BACKWARD')=='1'
                    and native_backward_inputs is None):
                # Preserve only the real factual IDs and target metadata on
                # CPU. No model tensors or prefix bank outlive diagnose().
                pair,selection=args[0],args[2]
                native_backward_inputs=dict(input_ids=pair[1::2].detach().cpu(),
                    positions=selection.positions.detach().cpu(),
                    samples=selection.samples.detach().cpu(), labels=selection.labels.detach().cpu(),
                    outcome_token_ids=selection.outcome_token_ids.detach().cpu(),
                    dt_paired_input_shape=list(pair.shape))
            profile_this_call=(os.environ.get('DT_PREFIX_HOT_PROFILE')=='1'
                               and (label=='shared_profile_warm' if current_owner else label.endswith('_warm'))) or (
                               os.environ.get('DT_PREFIX_ROOT_TAPE_HOT')=='1'
                               and label=='root_tape_warm')
            parameter_context=nullcontext()
            if ((os.environ.get('DT_PREFIX_ROOT_TAPE_HOT')=='1' and label=='root_tape_warm')
                    or (current_owner and base_prefetch and profile_this_call)):
                from native_finite_parameter_ranges import NativeFiniteParameterRanges
                parameter_context=NativeFiniteParameterRanges(runner.model)
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
            gdn0_audit=None
            fa3_records=None
            saved_capture_backend=runner.capture_backend
            if os.environ.get('DT_PREFIX_ROOT_TAPE_GDN0')=='1' and label in (
                    'shared_warm','root_tape_disabled_warm','root_tape_warm'):
                from observe_native_gdn0_operands import NativeGDN0Operands
                globals_=selected_attribute.__func__.__globals__
                def capture_class(name):
                    return globals_[name] if saved_capture_backend is None else getattr(saved_capture_backend,name)
                gdn0_audit=NativeGDN0Operands(runner.model.model.language_model.layers[0].linear_attn,
                    Path(out)/'actual-gdn0-operands',variant=label,rank=torch.distributed.get_rank())
                runner.capture_backend=types.SimpleNamespace(
                    NativeDecoderCapture=capture_class('NativeDecoderCapture'),
                    NativeDenseAttentionCapture=capture_class('NativeDenseAttentionCapture'),
                    NativeGDNCapture=gdn0_audit.capture_type(capture_class('NativeGDNCapture')))
            if os.environ.get('DT_PREFIX_ROOT_TAPE_FA3')=='1' and label in (
                    'shared_warm','root_tape_disabled_warm','root_tape_warm'):
                from observe_native_fa3_operands import make_attention_capture_observer
                globals_=selected_attribute.__func__.__globals__
                def capture_class(name):
                    return globals_[name] if saved_capture_backend is None else getattr(saved_capture_backend,name)
                fa3_records=[]
                runner.capture_backend=types.SimpleNamespace(
                    NativeDecoderCapture=capture_class('NativeDecoderCapture'),
                    NativeDenseAttentionCapture=make_attention_capture_observer(
                        capture_class('NativeDenseAttentionCapture'),
                        target_module=runner.model.model.language_model.layers[3].self_attn,
                        records=fa3_records),
                    NativeGDNCapture=capture_class('NativeGDNCapture'))
            actual_row_finite=None
            row_fa_records=None
            if individual_rows and label=='shared_individual_rows_observed':
                from observe_native_gdn0_operands import NativeGDN0Operands
                from observe_native_fa3_varlen_operands import make_attention_capture_observer
                from observe_actual_row_finite_fa import ActualRowFiniteFA
                globals_=selected_attribute.__func__.__globals__
                def row_capture_class(name):
                    return globals_[name] if saved_capture_backend is None else getattr(saved_capture_backend,name)
                gdn0_audit=NativeGDN0Operands(runner.model.model.language_model.layers[0].linear_attn,
                    Path(out)/'actual-gdn0-operands',variant=label,rank=torch.distributed.get_rank())
                row_fa_records=[]
                runner.capture_backend=types.SimpleNamespace(
                    NativeDecoderCapture=row_capture_class('NativeDecoderCapture'),
                    NativeGDNCapture=gdn0_audit.capture_type(row_capture_class('NativeGDNCapture')),
                    NativeDenseAttentionCapture=make_attention_capture_observer(
                        row_capture_class('NativeDenseAttentionCapture'),
                        target_module=runner.model.model.language_model.layers[3].self_attn,
                        records=row_fa_records))
                actual_row_finite=ActualRowFiniteFA(row_operation,
                    Path(out)/f'actual-row-finite-fa3-rank{torch.distributed.get_rank()}.pt')
                runner.finite_fa=actual_row_finite
            context=(torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
                record_shapes=False,with_stack=False,profile_memory=False)
                if profile_this_call else nullcontext())
            with context as profile, projection_context, prefetch_context, inventory_context, parameter_context:
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
                    runner.capture_backend=saved_capture_backend
                    if actual_row_finite is not None:
                        runner.finite_fa=row_operation
                    for handle in handles:handle.remove()
                    for value in reversed(list(ranges.values())):value.__exit__(None,None,None)
            if individual_rows and label=='shared_individual_rows_observed':
                import json
                assert len(gdn0_audit.records)==len(row_fa_records)==1
                assert actual_row_finite.calls==8 and actual_row_finite.saved is not None
                fa_path=Path(out)/f'actual-row-native-fa3-rank{torch.distributed.get_rank()}.pt'
                with fa_path.open('xb') as stream:torch.save(row_fa_records[0],stream)
                save('actual_individual_row_operands_saved',
                     actual_finite=actual_row_finite.saved,
                     native_gdn0=gdn0_audit.records[0],
                     native_fa3=dict(path=str(fa_path),sha256=hashlib.sha256(fa_path.read_bytes()).hexdigest()),
                     scope='Passive actual operands including original finite LSE/upstream; copy/export overhead belongs to this observed variant only. Official numerical checks run after model release.')
                gdn0_audit=None
            if gdn0_audit is not None:
                from observe_native_gdn0_operands import check_saved_fla
                observation=gdn0_audit.report()
                assert observation['recorded_calls']==1
                # Receipt nesting: .../receipts/owner-b8-dispatch/<probe>.
                official=Path(out).parents[1]/'training-setup/official-kernel-tests/test_gated_delta_v041.py'
                check=check_saved_fla(observation['records'][0]['path'],official)
                observation['official_check']=check
                gdn0_observations.append(observation)
                save('actual_cached_suffix_gdn0_checked',variant=label,actual_gdn0_observations=gdn0_observations)
            if fa3_records is not None:
                from observe_native_fa3_operands import check_saved_operands
                import json
                assert len(fa3_records)==1
                record=fa3_records[0]
                path=Path(out)/f'actual-fa3-operands-{label}-rank{torch.distributed.get_rank()}.pt'
                with path.open('xb') as stream:torch.save(record,stream)
                official=Path(out).parents[1]/'training-setup/official-kernel-tests/test_flash_attn_v263.py'
                check=check_saved_operands(record,official)
                observation={k:v for k,v in record.items() if k!='tensors'}
                observation.update(variant=label,rank=torch.distributed.get_rank(),
                    saved_actual_operands=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()),
                    official_check=check)
                fa3_observations.append(observation)
                save('actual_cached_suffix_fa3_checked',variant=label,actual_fa3_observations=fa3_observations)
                del fa3_records,record
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
            if ledger_only:
                import json
                # The unchanged owner already returns this scalar ledger.
                # Save it after its completion timer; no tensor capture or
                # extra model call, and return the same original result below.
                detail=result[1]
                fields=('root_effect','seed_effect','signed_sum','relative_residual','layers',
                    'compiled_seed_logprob_effect','compiled_seed_logprob_effect_minus_root',
                    'target_logp0','target_logp1','native_shared_prefix_length',
                    'gdn_fla_coefficient_start','controller_diagnostic_scheduling')
                owner_source=Path(selected_attribute.__func__.__code__.co_filename)
                observation=dict(variant=label,rank=torch.distributed.get_rank(),
                    original_request_sha256=hashlib.sha256(original_request_path.read_bytes()).hexdigest(),
                    original_request_path=str(original_request_path),original_sorted_offset=offset,
                    paired_input_shape=list(args[0].shape),
                    restored_checkpoint=os.environ.get('DT_PREFIX_CHECKPOINT'),
                    owner_source=dict(path=str(owner_source),
                        sha256=hashlib.sha256(owner_source.read_bytes()).hexdigest()),
                    scope='Existing original owner scalar details from one actual shared B4; no numerical acceptance criterion or tensor capture.',
                    owner_details={key:detail[key] for key in fields if key in detail})
                ledger_path=Path(out)/f'owner-ledger-{label}-rank{torch.distributed.get_rank()}-call{len(ledger_receipts)}.json'
                ledger_path.write_text(json.dumps(observation,indent=2)+'\n')
                receipt=dict(path=str(ledger_path),sha256=hashlib.sha256(ledger_path.read_bytes()).hexdigest(),
                    variant=label,layer_count=len(detail['layers']))
                ledger_receipts.append(receipt)
                save('native_original_scalar_ledger',receipt=receipt)
            for call in result[1].get('calls',[]):
                kind=call['kind']
                seconds=call.get('stream_elapsed_seconds',call.get('seconds'))
                if seconds is not None:
                    phase_totals[kind]=phase_totals.get(kind,0.0)+seconds
                phase_counts[kind]=phase_counts.get(kind,0)+1
            if label.startswith('root_tape'):
                root_tape_reports.append({k:v for k,v in result[1].items()
                    if 'root_tape' in k or 'root_capture' in k or 'replay_calls' in k})
            return result
        # Read timings emitted by the existing owner; no extra forward,
        # synchronization, target, or altered finite computation.
        runner.attribute=observe_attribute
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
        try:
            if conv_capacity:
                result=[row for episode in readout.episodes(capacity_episodes,complete_returns=capacity_returns)
                        for row in episode]
            else:
                result=readout.episodes([rows],complete_returns=[returns])[0]
        finally:
            runner.attribute=native_attribute;readout.prefix_lease_factory=None
            runner.__class__=previous_class
            if individual_rows:runner.finite_fa=original_finite
            if conv_initial_states:runner.native_conv_initial_states=previous_conv_option
            runner.capture_backend=previous_capture_backend
            if had_root_flag:runner.reuse_root_captures=previous_root_flag
            elif hasattr(runner,'reuse_root_captures'):del runner.reuse_root_captures
        torch.cuda.synchronize()
        vectors[label]={key:torch.stack([r[key] for r in result])
                        for key in ('dt_token_advantages','dt_q_estimates','dt_v_estimates')}
        reports[label]=dict(total_wall_seconds=time.perf_counter()-start,
            original_attribute_wall_seconds=attribute_walls,
            original_readout_report=readout.last_report,
            original_runner_phase_seconds=phase_totals,
            original_runner_phase_counts=phase_counts,
            root_tape_observations=root_tape_reports,
            actual_gdn0_observations=gdn0_observations,
            actual_fa3_observations=fa3_observations,
            peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(),
            physical_free_bytes=torch.cuda.mem_get_info()[0],
            pss_bytes=psutil.Process().memory_full_info().pss)
        if ledger_only:reports[label]['original_scalar_ledger_receipts']=ledger_receipts
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
    for label in ('root_tape_disabled_warm','root_tape_cold','root_tape_warm'):
        if label in vectors:
            observations.append(dict(variant=label,comparison_reference='shared_warm',
                scope='Same initialized model, prefix bank, input geometry and finite operators; isolated root-capture lifetime seam only',
                values=[dict(key=key,
                    equal=bool(torch.equal(vectors[label][key],vectors['shared_warm'][key])),
                    maximum_absolute_difference=float((vectors[label][key].double()-vectors['shared_warm'][key].double()).abs().max()))
                    for key in vectors[label]]))
    tensor_path=Path(out)/f'prefix-lease-vectors-rank{torch.distributed.get_rank()}.pt'
    torch.save(vectors,tensor_path)
    save('native_prefix_lease_diagnostic_complete',reports=reports,raw_value_observations=observations,
         vectors=dict(path=str(tensor_path),sha256=hashlib.sha256(tensor_path.read_bytes()).hexdigest()))
    shared_bank=None
    return native_backward_inputs

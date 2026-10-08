"""Passively localize measured source errors on the frozen development set.

The original readout prepares each B4 and calls DT once. Existing finite
functions return their unchanged objects; only coefficients/factual states
are copied to a one-batch CPU bank. Original native single-EOS forwards and
the original _token_effect supply boundary contractions. No observer-mode
path, rule replacement, credit correction, training or new sample selection.
"""
from contextlib import nullcontext
import inspect
import json
import os
from pathlib import Path
import time

import torch
import inspect_action_curve as initializer
from inspect_extreme_endpoint import check_imports, sha


def make_worker():
    import ray
    from verl.single_controller.base.decorator import Dispatch, register
    from verl.workers.fsdp_workers import ActorRolloutRefWorker

    @ray.remote
    class LayerCollectionWorker(ActorRolloutRefWorker):
        @register(dispatch_mode=Dispatch.ONE_TO_ALL)
        def inspect_action_curve(self, source_path, output, case_name='appworld'):
            import psutil
            import reward_readout
            from deltatrace_rollout import DeltaTraceRolloutProducer
            from native_target_logit_rows import NativeTargetLogitRows
            from qwen35_answer_finite import PackedAnswerTargets, selected_target_log_probs
            from verl.utils.fsdp_utils import load_fsdp_model_to_gpu, offload_fsdp_model_to_cpu
            from verl.utils.torch_functional import pad_2d_list_to_length

            out = Path(output)
            plan_path = out.parent/'layer-collection-inputs.json'
            plan = json.loads(plan_path.read_bytes())
            source = json.loads(Path(source_path).read_bytes())
            task = case_name
            spec = plan['tasks'][task]
            factual_controls = os.environ.get('DT_LAYER_FACTUAL_CONTROL') == '1'
            suboperations = os.environ.get('DT_LAYER_SUBOPERATIONS') == '1'
            attention_branches = os.environ.get('DT_ATTENTION_BRANCHES') == '1'
            assert not attention_branches or suboperations
            subobserver = None
            needed = set(range(33))
            if suboperations:
                assert factual_controls
                protocol_path = out.parent/'suboperation-protocol.json'
                protocol = json.loads(protocol_path.read_bytes())
                assert protocol['unchanged_queries_plan_sha256'] == sha(plan_path)
                needed = set(protocol['required_original_boundaries'])
            record = dict(scope=__doc__, task=task, rank=self.rank, pid=os.getpid(),
                birth=psutil.Process().create_time(), source_sha256=sha(source_path),
                input_plan_sha256=sha(plan_path), script_sha256=sha(__file__),
                owners=check_imports(source), batches=[], diagnostic_wall_budget_seconds=1800,
                factual_controls=factual_controls,
                suboperations=suboperations,
                operations=dict(DT=0, native_forward=0, backward=0, optimizer=0,
                                scheduler=0, rollout=0, checkpoint_restore=0))
            phases = (out/f'rank{self.rank}-phases.jsonl').open('a', buffering=1)
            started = time.perf_counter()
            def save(phase, **extra):
                event = dict(phase=phase, unix=time.time(), elapsed_seconds=time.perf_counter()-started,
                    allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved(),
                    runtime_free_bytes=torch.cuda.mem_get_info()[0],
                    process_pss_bytes=psutil.Process().memory_full_info().pss, **extra)
                record.update(event)
                phases.write(json.dumps(event)+'\n')
                (out/f'rank{self.rank}.json').write_text(json.dumps(record, indent=2)+'\n')
            def budget():
                if time.perf_counter()-started > record['diagnostic_wall_budget_seconds']:
                    raise RuntimeError('Bounded layer collection reached 1800 seconds; retain partial data, do not retry automatically.')

            producer = text = None
            handles = []
            bank = {}
            original_training = self.actor_module_fsdp.training
            previous_attention = None
            globals_ = None
            trace_owner = reward_readout.trace_token_attribution
            try:
                assert self.actor.config.ppo_micro_batch_size_per_gpu == 4
                assert (self.config.model.lora_rank, self.config.model.lora_alpha) == (8, 16)
                assert source['fresh_base_model'] and not source['checkpoint_restore_requested']
                shards = []
                for name, parameter in self.actor_module_fsdp.named_parameters():
                    if '.lora_B.' in name:
                        value = parameter.detach()
                        value = value.to_local() if hasattr(value, 'to_local') else value
                        shards.append(dict(name=name, elements=value.numel(), nonzero=int(torch.count_nonzero(value))))
                assert shards and not any(s['nonzero'] for s in shards)
                record['lora_B_local_shards'] = shards
                if self._is_offload_param:
                    load_fsdp_model_to_gpu(self.actor_module_fsdp)
                self.actor_module_fsdp.eval()
                producer = DeltaTraceRolloutProducer(self.actor_module_fsdp,
                    eos_token_id=self.tokenizer.eos_token_id, pad_token_id=self.tokenizer.pad_token_id)
                runner = producer.runner
                text = runner.model.model.language_model
                previous_attention = text.config._attn_implementation
                globals_ = runner.attribute.__func__.__globals__
                decoder_owner, effect_owner = globals_['decoder_finite_pullback'], globals_['_token_effect']
                actual_runner = inspect.getsourcefile(runner.attribute)
                assert sha(actual_runner) == source['actual_CPU_imports']['qwen35_dense_finite_runner']['sha256']
                record['finite_owner'] = dict(path=actual_runner, sha256=sha(actual_runner),
                    decoder_path=inspect.getsourcefile(decoder_owner), decoder_sha256=sha(inspect.getsourcefile(decoder_owner)),
                    copy_replay_captures=runner.copy_replay_captures,
                    offload_replay_mixer=runner.offload_replay_mixer, reuse_native_prefix=runner.reuse_native_prefix)
                layer_index = {id(layer): i for i, layer in enumerate(text.layers)}
                entries_by_uid = {e['traj_uid']: e for e in spec['entries']}
                for pair_index in range(0, len(spec['batches']), 2):
                    budget()
                    batch_index = pair_index+self.rank
                    batch_spec = spec['batches'][batch_index]
                    entries = [entries_by_uid[u] for u in batch_spec['uids']]
                    entries += [entries[-1]]*(4-len(entries))
                    rows = []
                    for entry in entries:
                        assert sha(entry['native']['path']) == entry['native']['sha256']
                        native = torch.load(entry['native']['path'], map_location='cpu', weights_only=False)
                        rows.append(next(r for r in native['rows'] if str(r['traj_uid']) == entry['traj_uid']))
                        del native
                    assert [r['selected'].numel() for r in rows] == sorted(r['selected'].numel() for r in rows)
                    batch = dict(index=batch_index, uids=batch_spec['uids'], actual_rows=batch_spec['actual_rows'],
                        identity_controls=batch_spec['B4_identity_controls'], points=[], native_phases=[])
                    record['batches'].append(batch)
                    captured = {}
                    active = dict(queries=None, contractions=None)
                    def retain(index, coefficients, endpoints):
                        if index not in needed:
                            return
                        assert index not in bank
                        bank[index] = (coefficients.detach().to('cpu', copy=True),
                                       endpoints[1::2].detach().to('cpu', copy=True))
                    def effect(coefficients, endpoints):
                        if 32 not in bank:
                            retain(32, coefficients, endpoints)
                        return effect_owner(coefficients, endpoints)
                    def decoder(*args, **kwargs):
                        if subobserver is not None:
                            subobserver.enter_decoder(layer_index[id(args[0])])
                        try:
                            value = decoder_owner(*args, **kwargs)
                        finally:
                            if subobserver is not None:
                                subobserver.leave_decoder()
                        retain(layer_index[id(args[0])], value[0], args[1]['input_norm_input'])
                        return value
                    def trace(*args, **kwargs):
                        value = trace_owner(*args, **kwargs)
                        assert not captured
                        captured.update(signed=value[0].detach().cpu(), detail=value[2])
                        return value
                    save('DT_begin', batch=batch_index, original_readout=True, observer_argument=False)
                    if suboperations:
                        from passive_suboperations import PassiveSuboperations
                        observer_type = PassiveSuboperations
                        if attention_branches:
                            from passive_attention_gate import PassiveAttentionGate
                            observer_type = PassiveAttentionGate
                            record['attention_branch_source_sha256'] = sha(inspect.getsourcefile(observer_type))
                        record['suboperation_source_sha256'] = sha(inspect.getsourcefile(PassiveSuboperations))
                        record['suboperation_protocol_sha256'] = sha(protocol_path)
                        subobserver = observer_type(runner, protocol['selected_decoder_layers'],
                            effect_owner, bank, active, None, rows, None)
                        subobserver.__enter__()
                    globals_['_token_effect'], globals_['decoder_finite_pullback'] = effect, decoder
                    reward_readout.trace_token_attribution = trace
                    try:
                        producer.attribute_episodes([[r['row'] for r in rows]], [0.0])
                    finally:
                        globals_['_token_effect'], globals_['decoder_finite_pullback'] = effect_owner, decoder_owner
                        reward_readout.trace_token_attribution = trace_owner
                    assert set(bank) == needed and captured
                    record['operations']['DT'] += 1
                    batch['DT_detail'] = captured['detail']
                    batch['retained_bank_bytes'] = sum(v.numel()*v.element_size() for pair in bank.values() for v in pair)
                    if subobserver is not None:
                        batch['retained_suboperation_bytes'] = subobserver.bytes()
                    save('DT_complete', batch=batch_index, retained_bank_bytes=batch['retained_bank_bytes'])
                    detail = captured['detail']
                    starts = detail.get('native_row_prefix_lengths')
                    if starts is None:
                        starts = [detail.get('native_shared_prefix_length') or 0]*4
                    width = max(r['selected'].numel() for r in rows)
                    selection = PackedAnswerTargets([r['case'] for r in rows],
                        [r['target_offsets'] for r in rows], width, 'cuda')
                    selector = NativeTargetLogitRows(selection)
                    if subobserver is not None:
                        subobserver.mode, subobserver.starts, subobserver.selection = 'native', starts, selection
                    def hook(index, hidden):
                        coefficients, factual = bank[index]
                        for row, query in enumerate(active['queries']):
                            if query is None:
                                continue
                            start = starts[row]
                            length = rows[row]['selected'].numel()-start
                            assert 0 < length <= coefficients.shape[1]
                            m = coefficients[row:row+1, :length].to(hidden.device)
                            pair = hidden[2*row:2*row+2, start:start+length]
                            value = float(effect_owner(m, pair).sum())
                            original_factual = factual[row, :length].to(hidden.device)
                            current_factual = pair[1]
                            active['contractions'][row][str(index)] = dict(value=value,
                                factual_endpoints_equal=torch.equal(original_factual, current_factual),
                                factual_endpoint_maxabs=float((original_factual.float()-current_factual.float()).abs().max()),
                                coefficient_dtype=str(m.dtype), endpoint_dtype=str(pair.dtype),
                                time_start=start, tokens=length)
                            if factual_controls:
                                # A diagnostic contraction through the same
                                # owner, not a subtraction from token credit.
                                control_pair = torch.stack((current_factual, original_factual))
                                control = float(effect_owner(m, control_pair).sum())
                                active['contractions'][row][str(index)].update(
                                    DT_factual_minus_native_factual_effect=control,
                                    DT_factual_minus_native_deleted_effect=value+control)
                    for index, layer in enumerate(text.layers):
                        if index not in needed:
                            continue
                        def before(_module, args, kwargs, index=index):
                            hook(index, args[0] if args else kwargs['hidden_states'])
                        handles.append(layer.register_forward_pre_hook(before, with_kwargs=True))
                    handles.append(text.norm.register_forward_pre_hook(lambda _module, args:hook(32, args[0])))
                    precision = nullcontext()
                    if producer.native_fla_fp16:
                        from accelerated.qwen35.native_fla_precision import native_fla_fp16
                        precision = native_fla_fp16(self.actor_module_fsdp)
                    conv = globals_['_native_conv_initial_states_scope']
                    text.set_attn_implementation('flash_attention_2')
                    # Equal FSDP call counts across ranks; exhausted rows are
                    # explicit identity controls, never counted as queries.
                    rounds = max(spec['batches'][i]['native_paired_forwards'] for i in (pair_index, pair_index+1))
                    with torch.no_grad(), precision, conv(text.layers, runner.native_conv_initial_states):
                        for round_index in range(rounds):
                            budget()
                            queries = [entry['queries'][round_index] if row < batch_spec['actual_rows'] and round_index < len(entry['queries']) else None
                                       for row, entry in enumerate(entries)]
                            ids = []
                            for row, query in zip(rows, queries):
                                deleted = row['selected'].clone()
                                if query is not None:
                                    assert int(deleted[query['packed_slot']]) == query['token_id']
                                    deleted[query['packed_slot']] = self.tokenizer.eos_token_id
                                ids.extend((deleted, row['selected']))
                            packed = pad_2d_list_to_length([v.tolist() for v in ids], self.tokenizer.eos_token_id, max_length=width).to('cuda')
                            active.update(queries=queries, contractions=[{} for _ in range(4)],
                                          suboperations=[{} for _ in range(4)])
                            save('native_forward_begin', batch=batch_index, round=round_index, paired_shape=[8, width])
                            tick = time.perf_counter()
                            with subobserver.native_scope() if attention_branches else nullcontext():
                                result = runner.model.forward_root(input_ids=packed, attention_mask=torch.ones_like(packed),
                                    use_cache=False, logits_to_keep=selector.rows)
                            logits = selector.pack_logits(result.logits)
                            del result, packed
                            if subobserver is not None:
                                subobserver.head_contractions(logits)
                            logp = selected_target_log_probs(logits, selection)
                            del logits
                            deleted = selection.sample_sums(logp[0::2].double()).cpu()
                            factual = selection.sample_sums(logp[1::2].double()).cpu()
                            del logp
                            record['operations']['native_forward'] += 1
                            for row, query in enumerate(queries):
                                if query is None:
                                    continue
                                contractions = active['contractions'][row]
                                assert len(contractions) == len(needed)
                                values = {i:contractions[str(i)]['value'] for i in needed}
                                native_d = float(factual[row]-deleted[row])
                                residuals = [values[i]-values[i+1] if i in values and i+1 in values else None
                                             for i in range(32)]+[values[32]-native_d]
                                point = dict(query, traj_uid=entries[row]['traj_uid'],
                                    initial_state_sha256=entries[row]['initial_state_sha256'],
                                    previously_examined=entries[row]['previously_examined'],
                                    fresh_DT_d=float(captured['signed'][row, query['packed_slot']]),
                                    factual_target_logp=float(factual[row]), deleted_target_logp=float(deleted[row]),
                                    native_single_d=native_d, boundaries=contractions,
                                    residuals=residuals, telescoping_roundoff=(
                                        sum(residuals)-(values[0]-native_d) if not suboperations else None))
                                if factual_controls:
                                    DT_factual = detail['per_sample'][row]['factual_target_logp']
                                    score_drift = DT_factual-float(factual[row])
                                    matched = {i:contractions[str(i)][
                                        'DT_factual_minus_native_deleted_effect'] for i in needed}
                                    matched_residuals = [matched[i]-matched[i+1] if i in matched and i+1 in matched else None
                                                         for i in range(32)]
                                    matched_residuals.append(matched[32]-native_d-score_drift)
                                    point.update(DT_factual_target_logp=DT_factual,
                                        DT_minus_native_factual_score=score_drift,
                                        matched_factual_residuals=matched_residuals,
                                        matched_telescoping_roundoff=(sum(matched_residuals)-(
                                            matched[0]-native_d-score_drift) if not suboperations else None))
                                if subobserver is not None:
                                    # Persist the actual contractions even if
                                    # optional report formatting fails.
                                    point['suboperations'] = active['suboperations'][row]
                                    batch['points'].append(point)
                                    subobserver.finish_point(point, active['suboperations'][row])
                                else:
                                    batch['points'].append(point)
                            batch['native_phases'].append(dict(round=round_index, seconds=time.perf_counter()-tick,
                                measured_queries=sum(q is not None for q in queries)))
                            if attention_branches:
                                subobserver.native_gate.clear()
                            save('native_forward_complete', batch=batch_index, round=round_index)
                    for handle in handles:
                        handle.remove()
                    handles.clear()
                    if subobserver is not None:
                        batch['additional_suboperation_readout_seconds'] = subobserver.readout_seconds
                        if attention_branches:
                            batch['attention_branch_readout'] = dict(
                                extra_projection_calls=subobserver.extra_projection_calls,
                                extra_sigmoid_slice_calls=subobserver.extra_sigmoid_calls,
                                seconds=subobserver.gate_readout_seconds,
                                linear_owner_path=subobserver.linear_readout_owner,
                                linear_owner_sha256=sha(subobserver.linear_readout_owner))
                        subobserver.__exit__(None, None, None)
                        subobserver = None
                    bank.clear()
                    captured.clear()
                    del rows, selection, selector
                    save('batch_complete', batch=batch_index, retained_bank_bytes=0)
                save('complete', completed_points=sum(len(b['points']) for b in record['batches']))
                return dict(rank=self.rank, completed=True, optimizer_steps=0)
            except BaseException:
                import traceback
                save('failed', traceback=traceback.format_exc())
                raise
            finally:
                if subobserver is not None:
                    subobserver.__exit__(None, None, None)
                reward_readout.trace_token_attribution = trace_owner
                if globals_ is not None:
                    globals_['_token_effect'], globals_['decoder_finite_pullback'] = effect_owner, decoder_owner
                for handle in handles:
                    handle.remove()
                bank.clear()
                if text is not None and previous_attention is not None:
                    text.set_attn_implementation(previous_attention)
                self.actor_module_fsdp.train(original_training)
                if producer is not None:
                    producer.runner.model.release_owner_params()
                if self._is_offload_param:
                    offload_fsdp_model_to_cpu(self.actor_module_fsdp)
                phases.close()
    return LayerCollectionWorker


if __name__ == '__main__':
    initializer.make_worker = make_worker
    initializer.main()

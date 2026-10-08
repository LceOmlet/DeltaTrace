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
    class LifecycleCollectionWorker(ActorRolloutRefWorker):
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
            record = dict(scope='Original DT and layer-observer lifecycle isolation on all frozen B4 groups; not a replacement quality sample.', task=task, rank=self.rank, pid=os.getpid(),
                birth=psutil.Process().create_time(), source_sha256=sha(source_path),
                input_plan_sha256=sha(plan_path), script_sha256=sha(__file__),
                owners=check_imports(source), batches=[], diagnostic_wall_budget_seconds=1800,
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
                    def retain(index, coefficients, endpoints):
                        assert index not in bank
                        bank[index] = (coefficients.detach().to('cpu', copy=True),
                                       endpoints[1::2].detach().to('cpu', copy=True))
                    def effect(coefficients, endpoints):
                        if 32 not in bank:
                            retain(32, coefficients, endpoints)
                        return effect_owner(coefficients, endpoints)
                    def decoder(*args, **kwargs):
                        value = decoder_owner(*args, **kwargs)
                        retain(layer_index[id(args[0])], value[0], args[1]['input_norm_input'])
                        return value
                    def trace(*args, **kwargs):
                        value = trace_owner(*args, **kwargs)
                        assert not captured
                        captured.update(signed=value[0].detach().cpu(), detail=value[2])
                        return value
                    def native_probe(name):
                        query_list = [entry['queries'][0] if row < batch_spec['actual_rows'] else None
                                      for row, entry in enumerate(entries)]
                        width_probe = max(row['selected'].numel() for row in rows)
                        selected_probe = PackedAnswerTargets([row['case'] for row in rows],
                            [row['target_offsets'] for row in rows], width_probe, 'cuda')
                        selector_probe = NativeTargetLogitRows(selected_probe)
                        ids_probe = []
                        for row, query in zip(rows, query_list):
                            deleted_probe = row['selected'].clone()
                            if query is not None:
                                assert int(deleted_probe[query['packed_slot']]) == query['token_id']
                                deleted_probe[query['packed_slot']] = self.tokenizer.eos_token_id
                            ids_probe.extend((deleted_probe, row['selected']))
                        packed_probe = pad_2d_list_to_length([v.tolist() for v in ids_probe],
                            self.tokenizer.eos_token_id, max_length=width_probe).to('cuda')
                        def flags():
                            return dict(matmul_precision=torch.get_float32_matmul_precision(),
                                allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                                allow_bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                                actor_training=self.actor_module_fsdp.training,
                                layer_training=[layer.training for layer in text.layers],
                                attention=text.config._attn_implementation,
                                native_fla_fp16=producer.native_fla_fp16,
                                native_conv_initial_states=runner.native_conv_initial_states,
                                native_FLA_bindings=[dict(layer=i, function=m.chunk_gated_delta_rule.__qualname__,
                                    unwrapped=inspect.unwrap(m.chunk_gated_delta_rule).__qualname__)
                                    for i, layer in enumerate(text.layers)
                                    if (m := getattr(layer, 'linear_attn', None)) is not None])
                        observation = dict(name=name, before_flags=flags(),
                            input_sha256=__import__('hashlib').sha256(packed_probe.cpu().numpy().tobytes()).hexdigest(),
                            paired_shape=list(packed_probe.shape))
                        save('native_lifecycle_probe_begin', batch=batch_index, probe=name)
                        tick_probe = time.perf_counter()
                        precision_probe = nullcontext()
                        if producer.native_fla_fp16:
                            from accelerated.qwen35.native_fla_precision import native_fla_fp16
                            precision_probe = native_fla_fp16(self.actor_module_fsdp)
                        text.set_attn_implementation('flash_attention_2')
                        with torch.no_grad(), precision_probe, globals_['_native_conv_initial_states_scope'](text.layers, runner.native_conv_initial_states):
                            output_probe = runner.model.forward_root(input_ids=packed_probe,
                                attention_mask=torch.ones_like(packed_probe), use_cache=False,
                                logits_to_keep=selector_probe.rows)
                            logits_probe = selector_probe.pack_logits(output_probe.logits)
                            values_probe = selected_target_log_probs(logits_probe, selected_probe)
                            factual_probe = selected_probe.sample_sums(values_probe[1::2].double()).cpu()
                            deleted_probe = selected_probe.sample_sums(values_probe[0::2].double()).cpu()
                        observation.update(after_flags=flags(), seconds=time.perf_counter()-tick_probe,
                            factual_logp=factual_probe.tolist(), deleted_logp=deleted_probe.tolist())
                        batch.setdefault('lifecycle_probes', []).append(observation)
                        record['operations']['native_forward'] += 1
                        del packed_probe, output_probe, logits_probe, values_probe
                        save('native_lifecycle_probe_complete', batch=batch_index, probe=name)
                    native_probe('before_DT_repeat_1')
                    native_probe('before_DT_repeat_2')
                    save('DT_begin', batch=batch_index, original_readout=True, observer_argument=False)
                    globals_['_token_effect'], globals_['decoder_finite_pullback'] = effect, decoder
                    reward_readout.trace_token_attribution = trace
                    try:
                        producer.attribute_episodes([[r['row'] for r in rows]], [0.0])
                    finally:
                        globals_['_token_effect'], globals_['decoder_finite_pullback'] = effect_owner, decoder_owner
                        reward_readout.trace_token_attribution = trace_owner
                    native_probe('after_DT_without_observer_hooks')
                    assert set(bank) == set(range(33)) and captured
                    record['operations']['DT'] += 1
                    batch['DT_detail'] = captured['detail']
                    batch['retained_bank_bytes'] = sum(v.numel()*v.element_size() for pair in bank.values() for v in pair)
                    save('DT_complete', batch=batch_index, retained_bank_bytes=batch['retained_bank_bytes'])
                    detail = captured['detail']
                    starts = detail.get('native_row_prefix_lengths')
                    if starts is None:
                        starts = [detail.get('native_shared_prefix_length') or 0]*4
                    width = max(r['selected'].numel() for r in rows)
                    selection = PackedAnswerTargets([r['case'] for r in rows],
                        [r['target_offsets'] for r in rows], width, 'cuda')
                    selector = NativeTargetLogitRows(selection)
                    active = dict(queries=None, contractions=None)
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
                    for index, layer in enumerate(text.layers):
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
                    rounds = 1  # Fixed first query per existing B4; preservation test, not replacement quality sample.
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
                            active.update(queries=queries, contractions=[{} for _ in range(4)])
                            save('native_forward_begin', batch=batch_index, round=round_index, paired_shape=[8, width])
                            tick = time.perf_counter()
                            result = runner.model.forward_root(input_ids=packed, attention_mask=torch.ones_like(packed),
                                use_cache=False, logits_to_keep=selector.rows)
                            logits = selector.pack_logits(result.logits)
                            del result, packed
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
                                assert len(contractions) == 33
                                values = [contractions[str(i)]['value'] for i in range(33)]
                                native_d = float(factual[row]-deleted[row])
                                residuals = [values[i]-values[i+1] for i in range(32)]+[values[32]-native_d]
                                batch['points'].append(dict(query, traj_uid=entries[row]['traj_uid'],
                                    initial_state_sha256=entries[row]['initial_state_sha256'],
                                    previously_examined=entries[row]['previously_examined'],
                                    fresh_DT_d=float(captured['signed'][row, query['packed_slot']]),
                                    factual_target_logp=float(factual[row]), deleted_target_logp=float(deleted[row]),
                                    native_single_d=native_d, boundaries=contractions,
                                    residuals=residuals, telescoping_roundoff=sum(residuals)-(values[0]-native_d)))
                            batch['native_phases'].append(dict(round=round_index, seconds=time.perf_counter()-tick,
                                measured_queries=sum(q is not None for q in queries)))
                            save('native_forward_complete', batch=batch_index, round=round_index)
                    for handle in handles:
                        handle.remove()
                    handles.clear()
                    native_probe('after_original_observer_hooks_removed')
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
    return LifecycleCollectionWorker


if __name__ == '__main__':
    initializer.make_worker = make_worker
    initializer.main()

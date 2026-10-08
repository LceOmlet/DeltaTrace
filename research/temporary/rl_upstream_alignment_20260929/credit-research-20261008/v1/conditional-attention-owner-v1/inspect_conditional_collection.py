"""Pair complete original/candidate DT on the frozen development identities.

Original VERL initializes the unchanged fresh actor. Original readout prepares
the joint target and invokes DT. The existing author metric owns cumulative
deletion, RISE and MAS. Single deletions diagnose extreme attribution only;
no training, credit replacement, rewhitening or effect-based sample selection.
"""
from contextlib import nullcontext
from functools import partial
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import time

import torch
import inspect_action_curve as initializer
from inspect_author_collection import evaluate_author_curves_batched
from inspect_extreme_endpoint import check_imports, sha


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def make_candidate(original, spec, environment):
    """Load explicit patched owner modules; keep the original model and GDN."""
    for item in spec['files']:
        assert sha(item['path']) == item['sha256']
    if spec.get('kind') == 'endpoint_head':
        # Only the new finite seed is local. The actual head, runner, FA/FLA,
        # model, author metric and all native execution remain their owners.
        module = load_module('research_endpoint_seed', spec['seed'])
        return module.make_candidate(original, spec, environment)
    if spec.get('kind') == 'conditional_gdn':
        module = load_module('research_conditional_gdn_builder', spec['builder'])
        return module.make_candidate(original, spec, environment)
    decoder = load_module('research_conditional_decoder', spec['decoder'])
    previous = sys.modules['qwen35_decoder_finite']
    try:
        sys.modules['qwen35_decoder_finite'] = decoder
        module = load_module('research_conditional_runner', spec['runner'])
    finally:
        sys.modules['qwen35_decoder_finite'] = previous
    wrapper = load_module('research_conditional_finite_FA', spec['wrapper'])
    # The isolated wrapper is a distinct module object. Use the actual runner's
    # existing layout type rather than constructing a second owner identity.
    wrapper.RightPaddedLengths = module.RightPaddedLengths
    finite = wrapper.VendorFAFiniteP1BF16D256(spec['library'], spec['library_sha256'])
    execution = dict(dynamic_shapes=environment.get('dt_dynamic_shapes', False),
        compiler_options=environment.get('dt_compiler_options', {}),
        answer_compiled=environment.get('dt_answer_compiled', True))
    options = {}
    for name in inspect.signature(module.Qwen35DenseFiniteRunner).parameters:
        if name in ('model', 'finite_fa', 'finite_fla', 'conditional_attention'):
            continue
        options[name] = execution[name] if name in execution else getattr(original, name)
    candidate = module.Qwen35DenseFiniteRunner(original.model, finite,
        original.finite_fla, conditional_attention=True, **options)
    assert candidate.model is original.model
    assert candidate.capture_backend is original.capture_backend
    assert candidate.finite_fla is original.finite_fla
    assert candidate.finite_fla_by_layer == original.finite_fla_by_layer
    return candidate, dict(options={k: str(v) for k, v in options.items()},
        runner_path=inspect.getsourcefile(candidate.attribute),
        runner_sha256=sha(inspect.getsourcefile(candidate.attribute)),
        decoder_path=inspect.getsourcefile(decoder.attention_finite_pullback),
        decoder_sha256=sha(inspect.getsourcefile(decoder.attention_finite_pullback)),
        same_model_object=True, same_GDN_callbacks=True,
        production_modified=False)


def make_worker():
    import ray
    from verl.single_controller.base.decorator import Dispatch, register
    from verl.workers.fsdp_workers import ActorRolloutRefWorker

    @ray.remote
    class ConditionalCollectionWorker(ActorRolloutRefWorker):
        @register(dispatch_mode=Dispatch.ONE_TO_ALL)
        def inspect_action_curve(self, source_path, output, case_name='appworld'):
            import ft_ifr_improve
            import psutil
            import reward_readout
            from deltatrace_rollout import DeltaTraceRolloutProducer
            from native_target_logit_rows import NativeTargetLogitRows
            from qwen35_answer_finite import PackedAnswerTargets, selected_target_log_probs
            from verl.utils.fsdp_utils import load_fsdp_model_to_gpu, offload_fsdp_model_to_cpu
            from verl.utils.torch_functional import pad_2d_list_to_length

            out = Path(output)
            plan = json.loads((out.parent / 'comparison-inputs.json').read_bytes())
            source = json.loads(Path(source_path).read_bytes())
            task = case_name
            spec = plan['tasks'][task]
            record = dict(scope=__doc__, task=task, rank=self.rank, pid=os.getpid(),
                birth=psutil.Process().create_time(), source_sha256=sha(source_path),
                script_sha256=sha(__file__), plan_sha256=sha(out.parent / 'comparison-inputs.json'),
                owners=check_imports(source), batches=[],
                diagnostic_wall_budget_seconds=spec['wall_budget_seconds'],
                operations=dict(DT=0, native_forward=0, optimizer=0, backward=0,
                    rollout=0, checkpoint_restore=0))
            phases = (out / f'rank{self.rank}-phases.jsonl').open('a', buffering=1)
            started = time.perf_counter()

            def save(phase, **extra):
                event = dict(phase=phase, unix=time.time(), elapsed_seconds=time.perf_counter()-started,
                    allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved(),
                    runtime_free_bytes=torch.cuda.mem_get_info()[0],
                    process_pss_bytes=psutil.Process().memory_full_info().pss, **extra)
                record.update(event)
                phases.write(json.dumps(event) + '\n')
                (out / f'rank{self.rank}.json').write_text(json.dumps(record, indent=2) + '\n')

            def budget():
                if time.perf_counter()-started > spec['wall_budget_seconds']:
                    raise RuntimeError('Bounded complete-candidate diagnostic reached its recorded budget; preserve partial results without automatic retry.')

            producer = text = None
            training = self.actor_module_fsdp.training
            attention = None
            trace_owner = reward_readout.trace_token_attribution
            try:
                assert self.actor.config.ppo_micro_batch_size_per_gpu == 4
                assert (self.config.model.lora_rank, self.config.model.lora_alpha) == (8, 16)
                assert source['fresh_base_model'] and not source['checkpoint_restore_requested']
                nonzero = []
                for name, parameter in self.actor_module_fsdp.named_parameters():
                    if '.lora_B.' in name:
                        value = parameter.detach()
                        value = value.to_local() if hasattr(value, 'to_local') else value
                        nonzero.append(int(torch.count_nonzero(value)))
                assert nonzero and not any(nonzero)
                if self._is_offload_param:
                    load_fsdp_model_to_gpu(self.actor_module_fsdp)
                self.actor_module_fsdp.eval()
                producer = DeltaTraceRolloutProducer(self.actor_module_fsdp,
                    eos_token_id=self.tokenizer.eos_token_id, pad_token_id=self.tokenizer.pad_token_id)
                original = producer.runner
                assert producer.direct_readout is not None
                assert sha(inspect.getsourcefile(original.attribute)) == source['actual_CPU_imports']['qwen35_dense_finite_runner']['sha256']
                environment = json.loads(Path(source['environment']['DT_ENVIRONMENT_JSON']).read_bytes())['qwen35']
                candidate, record['candidate'] = make_candidate(original, spec['candidate'], environment)
                candidate_label = spec.get('candidate_label', 'conditional')
                cached_batches = None
                if 'original_baseline' in spec:
                    baseline = spec['original_baseline'][self.rank]
                    assert sha(baseline['path']) == baseline['sha256']
                    cached = json.loads(Path(baseline['path']).read_bytes())
                    assert cached['phase'] == 'complete' and cached['source_sha256'] == sha(source_path)
                    cached_batches = {b['index']: b for b in cached['batches']}
                    record['original_baseline'] = baseline
                runners = ({candidate_label: candidate} if cached_batches is not None
                           else dict(original=original, **{candidate_label: candidate}))
                text = original.model.model.language_model
                attention = text.config._attn_implementation
                function = ft_ifr_improve.faithfulness_test_skip_tokens
                assert sha(inspect.getsourcefile(function)) == plan['metric_owner']['sha256']
                assert inspect.signature(function).parameters['k'].default == 20
                conv = original.attribute.__func__.__globals__['_native_conv_initial_states_scope']
                precision = nullcontext
                if producer.native_fla_fp16:
                    from accelerated.qwen35.native_fla_precision import native_fla_fp16
                    precision = partial(native_fla_fp16, self.actor_module_fsdp)
                layer_index = {id(layer.self_attn): i for i, layer in enumerate(text.layers)
                    if layer.block_type == 'full_attention'}
                entries_by_uid = {entry['traj_uid']: entry for entry in spec['entries']}
                save('original_actor_and_candidate_ready')
                for pair in range(0, len(spec['batches']), 2):
                    budget()
                    batch_index = pair + self.rank
                    chosen = spec['batches'][batch_index]
                    entries = [entries_by_uid[uid] for uid in chosen['uids']]
                    entries += [entries[-1]] * (4-len(entries))
                    rows = []
                    for entry in entries:
                        assert sha(entry['native']['path']) == entry['native']['sha256']
                        native = torch.load(entry['native']['path'], map_location='cpu', weights_only=False)
                        rows.append(next(row for row in native['rows'] if str(row['traj_uid']) == entry['traj_uid']))
                        del native
                    ordered = sorted(range(4), key=lambda i: rows[i]['selected'].numel())
                    entries, rows = [entries[i] for i in ordered], [rows[i] for i in ordered]
                    positions = [(row['prompt_length'] + row['prior'][row['suffix_positions']].nonzero().flatten()).tolist() for row in rows]
                    width = max(row['selected'].numel() for row in rows)
                    selection = PackedAnswerTargets([row['case'] for row in rows],
                        [row['target_offsets'] for row in rows], width, 'cuda')
                    selector = NativeTargetLogitRows(selection)
                    batch = dict(index=batch_index, primary=chosen['primary'], actual_rows=chosen['actual_rows'],
                        width=width, uids=[entry['traj_uid'] for entry in entries],
                        variants={}, forward_phases=[],
                        trajectories=[dict(traj_uid=entry['traj_uid'], initial_state_sha256=entry['initial_state_sha256'],
                            first_stage=entry['first_stage'], previously_examined=entry['previously_examined'],
                            views={}, single_deletions=[]) for entry in entries])
                    record['batches'].append(batch)
                    if cached_batches is not None:
                        cached_rows = {r['traj_uid']: r for r in cached_batches[batch_index]['trajectories']}
                        for row in batch['trajectories']:
                            old = cached_rows[row['traj_uid']]
                            assert old['initial_state_sha256'] == row['initial_state_sha256']
                            if 'original' in old['views']:
                                row['views']['original'] = old['views']['original']
                            row['single_deletions'] = [dict(q) for q in old['single_deletions']]
                        batch['original_baseline_reused'] = True
                    signed = {}
                    for label, runner in runners.items():
                        budget()
                        captured = {}
                        def trace(*args, **kwargs):
                            value = trace_owner(*args, **kwargs)
                            assert not captured
                            captured.update(signed=value[0].detach().cpu(), detail=value[2])
                            return value
                        globals_ = runner.attribute.__func__.__globals__
                        attention_owner = globals_['attention_finite_pullback']
                        def attention_trace(*args, **kwargs):
                            save('finite_attention_begin', batch=batch_index, variant=label, layer=layer_index[id(args[0])])
                            result = attention_owner(*args, **kwargs)
                            save('finite_attention_end', batch=batch_index, variant=label, layer=layer_index[id(args[0])])
                            return result
                        producer.runner = producer.direct_readout.runner = runner
                        reward_readout.trace_token_attribution = trace
                        globals_['attention_finite_pullback'] = attention_trace
                        save('DT_begin', batch=batch_index, variant=label, B4_shape=[4, width])
                        try:
                            producer.attribute_episodes([[row['row'] for row in rows]], [0.0])
                        finally:
                            reward_readout.trace_token_attribution = trace_owner
                            globals_['attention_finite_pullback'] = attention_owner
                        assert captured
                        signed[label] = [captured['signed'][i, positions[i]].float() for i in range(4)]
                        path = out / f'rank{self.rank}-batch{batch_index}-{label}.pt'
                        torch.save(dict(uids=batch['uids'], positions=positions, selected=[row['selected'] for row in rows],
                            signed=captured['signed'], detail=captured['detail']), path)
                        batch['variants'][label] = dict(artifact=str(path), sha256=sha(path), detail=captured['detail'])
                        record['operations']['DT'] += 1
                        save('DT_complete', batch=batch_index, variant=label)

                    def forward(ids, phase):
                        budget()
                        assert len(ids) == 8
                        for i, value in enumerate(ids):
                            changes = value.ne(rows[i//2]['selected']).nonzero().flatten()
                            assert bool(torch.isin(changes, torch.tensor(positions[i//2])).all())
                            assert bool(value[changes].eq(self.tokenizer.eos_token_id).all())
                        inputs = pad_2d_list_to_length([value.tolist() for value in ids],
                            self.tokenizer.eos_token_id, max_length=width).to('cuda')
                        save('native_forward_begin', batch=batch_index, active_phase=phase)
                        torch.cuda.synchronize()
                        tick = time.perf_counter()
                        value = original.model.forward_root(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                            use_cache=False, logits_to_keep=selector.rows)
                        logits = selector.pack_logits(value.logits)
                        del value
                        logp = selected_target_log_probs(logits, selection)
                        sums = torch.stack((selection.sample_sums(logp[0::2].double()),
                            selection.sample_sums(logp[1::2].double())), dim=1).flatten().cpu()
                        del logits, logp, inputs
                        torch.cuda.synchronize()
                        record['operations']['native_forward'] += 1
                        batch['forward_phases'].append(dict(phase=phase, seconds=time.perf_counter()-tick, scores=sums.tolist()))
                        save('native_forward_complete', batch=batch_index, active_phase=phase)
                        return sums

                    text.set_attn_implementation('flash_attention_2')
                    with torch.no_grad(), precision(), conv(text.layers, original.native_conv_initial_states):
                        if chosen['primary']:
                            for label in runners:
                                curves = evaluate_author_curves_batched(function, rows, self.tokenizer,
                                    signed[label], positions,
                                    lambda ids, phase: forward(ids, label+'_'+phase))
                                for (slot, view), result in curves.items():
                                    batch['trajectories'][slot]['views'].setdefault(label, {})[
                                        'signed_RISE' if view == 0 else 'positive_MAS'] = result
                                    if cached_batches is not None and view == 0:
                                        old = batch['trajectories'][slot]['views']['original']['signed_RISE']
                                        batch['trajectories'][slot]['reused_baseline_factual_difference'] = (
                                            result['score_points'][0]['logp']-old['score_points'][0]['logp'])
                                save('author_curves_complete', batch=batch_index, variant=label)
                        if cached_batches is not None:
                            # Exact saved native queries stay diagnostic data;
                            # do not repeat their model calls or retry old FA.
                            for slot, row in enumerate(rows):
                                for q in batch['trajectories'][slot]['single_deletions']:
                                    index = positions[slot].index(q['packed_slot'])
                                    q['fresh_'+candidate_label+'_d'] = float(signed[candidate_label][slot][index])
                            del selection, selector, signed, rows
                            save('paired_batch_complete', batch=batch_index)
                            continue
                        if not chosen['primary']:
                            batch['native_factual_scores'] = forward(
                                [row['selected'] for row in rows for _ in range(2)],
                                'additional_tail_factual').tolist()
                        rounds = chosen['query_rounds']
                        for offset in range(0, 2*rounds, 2):
                            queries = []
                            inputs = []
                            for row, entry in zip(rows, entries):
                                active = entry['queries'][offset:offset+2]
                                for index in range(2):
                                    query = active[index] if index < len(active) else None
                                    ids = row['selected'].clone()
                                    if query is not None:
                                        assert int(ids[query['packed_slot']]) == query['token_id']
                                        ids[query['packed_slot']] = self.tokenizer.eos_token_id
                                    inputs.append(ids)
                                    queries.append(query)
                            values = forward(inputs, 'frozen_single_'+str(offset))
                            for slot, row in enumerate(rows):
                                if chosen['primary']:
                                    factual = batch['trajectories'][slot]['views']['original']['signed_RISE']['score_points'][0]['logp']
                                else:
                                    factual = batch['native_factual_scores'][2*slot]
                                for j in range(2):
                                    query = queries[2*slot+j]
                                    if query is None:
                                        continue
                                    source_index = positions[slot].index(query['packed_slot'])
                                    batch['trajectories'][slot]['single_deletions'].append(dict(query,
                                        fresh_factual_target_logp=factual, fresh_deleted_target_logp=float(values[2*slot+j]),
                                        fresh_native_single_d=factual-float(values[2*slot+j]),
                                        fresh_original_d=float(signed['original'][slot][source_index]),
                                        **{'fresh_'+candidate_label+'_d':float(signed[candidate_label][slot][source_index])}))
                    del selection, selector, signed, rows
                    save('paired_batch_complete', batch=batch_index)
                save('complete', scope='Complete frozen development comparison; no training, production repair or learning-effect claim.')
                return dict(rank=self.rank, complete=True, operations=record['operations'])
            except BaseException:
                import traceback
                save('failed', traceback=traceback.format_exc())
                raise
            finally:
                reward_readout.trace_token_attribution = trace_owner
                if text is not None and attention is not None:
                    text.set_attn_implementation(attention)
                self.actor_module_fsdp.train(training)
                if producer is not None:
                    producer.runner.model.release_owner_params()
                if self._is_offload_param:
                    offload_fsdp_model_to_cpu(self.actor_module_fsdp)
    return ConditionalCollectionWorker


if __name__ == '__main__':
    initializer.make_worker = make_worker
    initializer.main()

"""Measure output-head rounding on every existing frozen single-EOS query.

The actor, native forward, target packing and log-prob reader remain the
existing owners. A passive head hook projects the very same native hidden
rows and weights at FP32, then also scores that reference rounded to the
native output dtype. No output is replaced. This is a diagnostic reference,
not a new head, DT rule, numerical tolerance or production implementation.
"""
from contextlib import nullcontext
import copy
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
    class NativeHeadCollectionWorker(ActorRolloutRefWorker):
        @register(dispatch_mode=Dispatch.ONE_TO_ALL)
        def inspect_action_curve(self, source_path, output, case_name='appworld'):
            import psutil
            from deltatrace_rollout import DeltaTraceRolloutProducer
            from native_target_logit_rows import NativeTargetLogitRows
            from qwen35_answer_finite import PackedAnswerTargets, selected_target_log_probs
            from verl.utils.fsdp_utils import load_fsdp_model_to_gpu, offload_fsdp_model_to_cpu
            from verl.utils.torch_functional import pad_2d_list_to_length

            out = Path(output)
            plan_path = out.parent/'layer-collection-inputs.json'
            plan = json.loads(plan_path.read_bytes())
            source = json.loads(Path(source_path).read_bytes())
            spec = plan['tasks'][case_name]
            record = dict(scope=__doc__, task=case_name, rank=self.rank,
                pid=os.getpid(), birth=psutil.Process().create_time(),
                source_sha256=sha(source_path), input_plan_sha256=sha(plan_path),
                script_sha256=sha(__file__), owners=check_imports(source), batches=[],
                diagnostic_wall_budget_seconds=1500,
                reference=dict(input='same native head hidden rows and gathered head weight',
                    projection='torch.nn.functional.linear FP32; autocast and TF32 disabled',
                    row_chunk=64, rounding='reference.to(native_hidden.dtype).float()',
                    scoring_owner=inspect.getsourcefile(selected_target_log_probs),
                    scoring_owner_sha256=sha(inspect.getsourcefile(selected_target_log_probs))),
                operations=dict(DT=0, native_forward=0, backward=0, optimizer=0,
                    rollout=0, checkpoint_restore=0, production_patches=0))
            phases = (out/f'rank{self.rank}-phases.jsonl').open('a', buffering=1)
            started = time.perf_counter()

            def save(phase, **extra):
                event = dict(phase=phase, unix=time.time(), elapsed_seconds=time.perf_counter()-started,
                    allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved(),
                    peak_allocated=torch.cuda.max_memory_allocated(),
                    runtime_free_bytes=torch.cuda.mem_get_info()[0],
                    process_pss_bytes=psutil.Process().memory_full_info().pss, **extra)
                record.update(event)
                phases.write(json.dumps(event)+'\n')
                (out/f'rank{self.rank}.json').write_text(json.dumps(record, indent=2)+'\n')

            def budget():
                if time.perf_counter()-started > record['diagnostic_wall_budget_seconds']:
                    raise RuntimeError('Native head diagnostic reached its 1500-second wall budget; retain partial results, no automatic retry.')

            producer = text = handle = None
            previous_attention = None
            training = self.actor_module_fsdp.training
            old_tf32 = torch.backends.cuda.matmul.allow_tf32
            active = {}
            try:
                assert self.actor.config.ppo_micro_batch_size_per_gpu == 4
                assert (self.config.model.lora_rank, self.config.model.lora_alpha) == (8, 16)
                assert source['fresh_base_model'] and not source['checkpoint_restore_requested']
                shards = []
                for name, p in self.actor_module_fsdp.named_parameters():
                    if '.lora_B.' in name:
                        p = p.detach()
                        p = p.to_local() if hasattr(p, 'to_local') else p
                        shards.append(dict(name=name, elements=p.numel(), nonzero=int(torch.count_nonzero(p))))
                assert shards and not any(s['nonzero'] for s in shards)
                record['lora_B_local_shards'] = shards
                if self._is_offload_param:
                    load_fsdp_model_to_gpu(self.actor_module_fsdp)
                self.actor_module_fsdp.eval()
                producer = DeltaTraceRolloutProducer(self.actor_module_fsdp,
                    eos_token_id=self.tokenizer.eos_token_id, pad_token_id=self.tokenizer.pad_token_id)
                runner = producer.runner
                actual_runner = inspect.getsourcefile(runner.attribute)
                assert sha(actual_runner) == source['actual_CPU_imports']['qwen35_dense_finite_runner']['sha256']
                text = runner.model.model.language_model
                previous_attention = text.config._attn_implementation
                globals_ = runner.attribute.__func__.__globals__
                conv = globals_['_native_conv_initial_states_scope']
                entries_by_uid = {e['traj_uid']: e for e in spec['entries']}

                def head_reference(module, args):
                    selection, selector = active['selection'], active['selector']
                    hidden = args[0]
                    assert hidden.shape[:2] == (8, len(selector.rows))
                    assert module.bias is None and not hasattr(module.weight, 'placements')
                    # The FSDP pre-forward owner has already gathered this head.
                    # Pack genuine target rows, not the union's unused row slots.
                    packed = hidden[selection.paired_samples, selector.packed_rows]
                    dtype = hidden.dtype
                    assert dtype == torch.bfloat16
                    tick = time.perf_counter()
                    weight32 = module.weight.detach().float()
                    active['weight32_bytes'] = weight32.numel()*weight32.element_size()
                    reference, rounded = [], []
                    save('head_reference_begin', batch=active['batch'], round=active['round'],
                        packed_rows=len(packed), weight32_bytes=active['weight32_bytes'],
                        native_hidden_dtype=str(dtype))
                    torch.backends.cuda.matmul.allow_tf32 = False
                    try:
                        with torch.autocast(device_type=hidden.device.type, enabled=False):
                            for start in range(0, len(selection.labels), 64):
                                budget()
                                stop = min(start+64, len(selection.labels))
                                chunk_selection = copy.copy(selection)
                                chunk_selection.labels = selection.labels[start:stop]
                                z = torch.nn.functional.linear(packed[2*start:2*stop].float(), weight32)
                                reference.append(selected_target_log_probs(z, chunk_selection).double().cpu())
                                rounded.append(selected_target_log_probs(z.to(dtype), chunk_selection).double().cpu())
                                del z
                    finally:
                        torch.backends.cuda.matmul.allow_tf32 = old_tf32
                    del weight32, packed
                    active['reference'] = torch.cat(reference)
                    active['rounded'] = torch.cat(rounded)
                    active['head_seconds'] = time.perf_counter()-tick
                    save('head_reference_complete', batch=active['batch'], round=active['round'],
                        head_seconds=active['head_seconds'])
                    # Return None: leave the native input and native output untouched.

                handle = runner.model.lm_head.register_forward_pre_hook(head_reference)
                text.set_attn_implementation('flash_attention_2')
                precision = nullcontext()
                if producer.native_fla_fp16:
                    from accelerated.qwen35.native_fla_precision import native_fla_fp16
                    precision = native_fla_fp16(self.actor_module_fsdp)
                with torch.no_grad(), precision, conv(text.layers, runner.native_conv_initial_states):
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
                        width = max(r['selected'].numel() for r in rows)
                        selection = PackedAnswerTargets([r['case'] for r in rows],
                            [r['target_offsets'] for r in rows], width, 'cuda')
                        selector = NativeTargetLogitRows(selection)
                        batch = dict(index=batch_index, uids=batch_spec['uids'],
                            actual_rows=batch_spec['actual_rows'], points=[], native_phases=[])
                        record['batches'].append(batch)
                        rounds = max(spec['batches'][i]['native_paired_forwards'] for i in (pair_index, pair_index+1))
                        for round_index in range(rounds):
                            budget()
                            if 'probability_sample_queries' in batch_spec:
                                assert rounds == 1 and round_index == 0
                                queries = batch_spec['probability_sample_queries']+[None]*(4-batch_spec['actual_rows'])
                                assert len(queries) == 4
                                assert all(q is None or q['traj_uid'] == e['traj_uid'] for q,e in zip(queries,entries))
                            else:
                                queries = [entry['queries'][round_index] if row < batch_spec['actual_rows'] and round_index < len(entry['queries']) else None
                                           for row, entry in enumerate(entries)]
                            ids = []
                            for row, query in zip(rows, queries):
                                deleted = row['selected'].clone()
                                if query is not None:
                                    assert int(deleted[query['packed_slot']]) == query['token_id']
                                    deleted[query['packed_slot']] = self.tokenizer.eos_token_id
                                ids.extend((deleted, row['selected']))
                            packed = pad_2d_list_to_length([v.tolist() for v in ids],
                                self.tokenizer.eos_token_id, max_length=width).to('cuda')
                            active.clear()
                            active.update(selection=selection, selector=selector,
                                batch=batch_index, round=round_index)
                            save('native_forward_begin', batch=batch_index, round=round_index, paired_shape=[8, width])
                            tick = time.perf_counter()
                            result = runner.model.forward_root(input_ids=packed,
                                attention_mask=torch.ones_like(packed), use_cache=False, logits_to_keep=selector.rows)
                            native_logp = selected_target_log_probs(selector.pack_logits(result.logits), selection).double().cpu()
                            del result, packed
                            scores = {}
                            for name, values in [('native', native_logp), ('FP32', active['reference']), ('rounded_FP32', active['rounded'])]:
                                deleted = selection.sample_sums(values[0::2].to('cuda')).cpu()
                                factual = selection.sample_sums(values[1::2].to('cuda')).cpu()
                                scores[name] = dict(deleted=deleted.tolist(), factual=factual.tolist())
                            for row, query in enumerate(queries):
                                if query is None:
                                    continue
                                metrics = {name: dict(factual=v['factual'][row], deleted=v['deleted'][row],
                                    d=v['factual'][row]-v['deleted'][row]) for name, v in scores.items()}
                                batch['points'].append(dict(query, traj_uid=entries[row]['traj_uid'],
                                    initial_state_sha256=entries[row]['initial_state_sha256'],
                                    previously_examined=entries[row]['previously_examined'], scores=metrics))
                            record['operations']['native_forward'] += 1
                            batch['native_phases'].append(dict(round=round_index, seconds=time.perf_counter()-tick,
                                head_reference_seconds=active['head_seconds'], measured_queries=sum(q is not None for q in queries)))
                            save('native_forward_complete', batch=batch_index, round=round_index)
                        active.clear()
                        del rows, selection, selector
                        save('batch_complete', batch=batch_index)
                save('complete', completed_points=sum(len(b['points']) for b in record['batches']))
                return dict(rank=self.rank, completed=True, optimizer_steps=0)
            except BaseException:
                import traceback
                save('failed', traceback=traceback.format_exc())
                raise
            finally:
                if handle is not None:
                    handle.remove()
                active.clear()
                torch.backends.cuda.matmul.allow_tf32 = old_tf32
                if text is not None and previous_attention is not None:
                    text.set_attn_implementation(previous_attention)
                self.actor_module_fsdp.train(training)
                if producer is not None:
                    producer.runner.model.release_owner_params()
                if self._is_offload_param:
                    offload_fsdp_model_to_cpu(self.actor_module_fsdp)
                phases.close()
    return NativeHeadCollectionWorker


if __name__ == '__main__':
    initializer.make_worker = make_worker
    initializer.main()

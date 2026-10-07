"""Existing DT rules on the same real high-impact AppWorld B4.

Calls the unchanged producer/runner and its existing rule-selection options.
This is a diagnostic comparison, never a production profile or credit repair.
Default compares attention PV; --memory compares averaged/forward memory only.
Source, target, reward, norm/gate, head, Q/V/A and PPO parameters persist.
--code-fence-only isolates one actual target score diagnostically, retaining
all original target INPUT tokens and original reference IDs. It never exports
partial-score contributions as complete-event Q/V/A or changes training.
"""
import inspect
import json
import os
from pathlib import Path
import time
from types import MethodType

import torch
import profile_existing_offload as diagnostic_owner
from inspect_extreme_endpoint import check_imports, sha, CASES


def make_worker(with_vllm=False, *, compare_memory=False, code_fence_only=False):
    assert not with_vllm
    assert not (compare_memory and code_fence_only)
    import ray
    from verl.single_controller.base.decorator import Dispatch, register
    from verl.workers.fsdp_workers import ActorRolloutRefWorker

    @ray.remote
    class ExistingPVRuleWorker(ActorRolloutRefWorker):
        @register(dispatch_mode=Dispatch.ONE_TO_ALL)
        def profile_existing(self, source_path, native_path, output, failed_batch):
            import psutil
            import reward_readout
            from deltatrace_rollout import DeltaTraceRolloutProducer
            from verl.utils.fsdp_utils import load_fsdp_model_to_gpu, offload_fsdp_model_to_cpu

            assert failed_batch is None
            source = json.loads(Path(source_path).read_bytes())
            owners = check_imports(source)
            assert sha(source_path) == CASES['appworld']['source_sha256']
            assert sha(native_path) == '3e902bc058ca1c06bec4c742be53523fd3e336b806d74ce19120682af2281a0a'
            native = torch.load(native_path, map_location='cpu', weights_only=False)
            prepared = sorted(native['rows'], key=lambda row: row['batch_row'])
            assert len(prepared) == 4
            rows = [row['row'] for row in prepared]
            assert self.actor.config.ppo_micro_batch_size_per_gpu == 4
            assert (self.config.model.lora_rank, self.config.model.lora_alpha) == (8, 16)
            assert self._is_actor and not self._is_rollout
            root = Path(output)
            record = dict(rank=self.rank, pid=os.getpid(), birth=psutil.Process().create_time(),
                          source_sha256=sha(source_path), native_path=native_path,
                          native_sha256=sha(native_path), owners=owners,
                          operations=dict(DT=1 if code_fence_only else 2, optimizer=0, backward=0, rollout=0, checkpoint_restore=0),
                          modes={})
            log = (root / f'rank{self.rank}-phases.jsonl').open('a', buffering=1)

            def save(phase, **values):
                record.update(phase=phase, unix=time.time(), **values)
                (root / f'rank{self.rank}.json').write_text(json.dumps(record, indent=2) + '\n')

            def emit(phase, **values):
                log.write(json.dumps(dict(phase=phase, unix=time.time(),
                    pss_bytes=psutil.Process().memory_full_info().pss,
                    allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved(),
                    device_free=torch.cuda.mem_get_info()[0], **values)) + '\n')

            producer = None
            trace_owner = reward_readout.trace_token_attribution
            traces = []
            details = []
            original_pv = None
            original_memory = None
            release_owner = None
            mode = ['unset']
            try:
                lora = []
                for name, p in self.actor_module_fsdp.named_parameters():
                    if '.lora_B.' in name:
                        t = p.detach()
                        t = t.to_local() if hasattr(t, 'to_local') else t
                        lora.append(dict(name=name, nonzero=int(torch.count_nonzero(t)), elements=t.numel()))
                assert lora and not any(x['nonzero'] for x in lora)
                record['fresh_lora_B_local_shards'] = lora
                if self._is_offload_param:
                    load_fsdp_model_to_gpu(self.actor_module_fsdp)
                producer = DeltaTraceRolloutProducer(self.actor_module_fsdp,
                    eos_token_id=self.tokenizer.eos_token_id, pad_token_id=self.tokenizer.pad_token_id)
                runner = producer.runner
                path = inspect.getsourcefile(type(runner))
                assert sha(path) == '7d6f57f61ecde7ce3506b8ef58d61268859fc04357c35de99a1892928c4ba7a8'
                assert runner.offload_replay_mixer
                original_pv = dict(runner.attention_pv_rules)
                assert original_pv == {}
                original_memory = dict(runner.finite_fla_by_layer)
                layers = runner.model.model.language_model.layers
                full = [i for i, layer in enumerate(layers) if layer.block_type == 'full_attention']
                assert full == [3, 7, 11, 15, 19, 23, 27, 31]
                if compare_memory:
                    assert sorted(original_memory) == [i for i, layer in enumerate(layers)
                        if layer.block_type == 'linear_attention']
                prepared_inputs = []
                for i, row in enumerate(rows):
                    actual = producer.direct_readout._prepare_row(row, 0)
                    assert torch.equal(actual['selected'], prepared[i]['selected'])
                    assert torch.equal(actual['case']['target_ids'], prepared[i]['case']['target_ids'])
                    assert actual['target_offsets'] == prepared[i]['target_offsets']
                    prepared_inputs.append(actual)

                def trace(*args, **kwargs):
                    if code_fence_only:
                        # Diagnostic target isolation only. The producer has
                        # already constructed the original complete-reference
                        # input. Keep it and every actual target input token;
                        # select just the observed next code-fence score at the
                        # existing PackedAnswerTargets boundary, not a new Y
                        # for training or a new reference-construction rule.
                        trace_runner, reference, factual, cases, offsets, *rest = args
                        assert len(cases) == 4 and not rest
                        chosen = offsets[3][0]
                        assert cases[3]['prompt_length'] + chosen - 1 == 2883
                        assert int(cases[3]['target_ids'][chosen]) == 71093
                        source_slots = (prepared_inputs[3]['prior'][prepared_inputs[3]['suffix_positions']]
                                        .nonzero().flatten() + prepared_inputs[3]['prompt_length'])
                        observed = (reference[3] != factual[3]).nonzero().flatten().cpu()
                        expected = source_slots[factual[3, source_slots.to(factual.device)].cpu()
                                                != self.tokenizer.eos_token_id]
                        assert torch.equal(observed, expected)
                        targets = torch.tensor(offsets[3], device=factual.device) + cases[3]['prompt_length']
                        assert torch.equal(reference[3, targets], factual[3, targets])
                        selected = [list(value) for value in offsets]
                        selected[3] = [chosen]
                        record['diagnostic_target_selection'] = dict(
                            row=3, predictor_position=2883, target_position=2884,
                            target_token_id=71093, original_target_counts=list(map(len, offsets)),
                            score_target_counts=list(map(len, selected)),
                            reference_changed_positions=observed.tolist(),
                            original_reference_sources_exact=True,
                            original_target_input_tokens_retained=True,
                            training_target_changed=False)
                        args = (trace_runner, reference, factual, cases, selected)
                    value = trace_owner(*args, **kwargs)
                    traces.append(value[0].detach().cpu())
                    details.append(value[2])
                    return value

                reward_readout.trace_token_attribution = trace
                release_owner = runner.model.release_finite_layer

                def release(_self, layer):
                    value = release_owner(layer)
                    index = next(i for i, item in enumerate(layers) if item is layer)
                    emit('after_finite_layer', mode=mode[0], layer=index)
                    return value

                runner.model.release_finite_layer = MethodType(release, runner.model)
                record.update(runner=dict(path=path, sha256=sha(path)),
                              existing_offload=True, original_attention_pv_rules=original_pv,
                              original_GDN_rules=runner.norm_gate_rules,
                              comparison_kind='actual_next_code_fence_score_only' if code_fence_only else ('memory_endpoint_order' if compare_memory else 'attention_PV'),
                              original_memory_override_layers=sorted(original_memory),
                              geometry=dict(selected_lengths=[row['selected'].numel() for row in prepared],
                                  uids=[row['traj_uid'] for row in prepared],
                                  target_counts=[len(row['target_offsets']) for row in prepared]))
                save('producer_ready')
                modes = [('original_symmetric_memory', original_memory),
                         ('existing_forward_memory', {})] if compare_memory else [
                         ('original_content1', original_pv),
                         ('existing_content0', {i: 'content0' for i in full})]
                if code_fence_only:
                    modes = [('original_next_code_fence_score_only', original_pv)]
                for name, rules in modes:
                    mode[0] = name
                    if compare_memory:
                        # The owner's empty override map selects the existing
                        # original compiled finite_fla callback at every GDN.
                        # Norm/gate, FA, head and the producer remain unchanged.
                        runner.finite_fla_by_layer = dict(rules)
                        descriptor = dict(memory_override_layers=sorted(rules))
                    else:
                        runner.attention_pv_rules = dict(rules)
                        descriptor = dict(attention_pv_rules=rules)
                    before = len(traces)
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    emit('DT_begin', mode=name, **descriptor)
                    save('DT_begin', active_mode=name)
                    tick = time.perf_counter()
                    values = producer.attribute_episodes([rows], [0.0])[0]
                    torch.cuda.synchronize()
                    assert len(traces) == before + 1
                    artifact = root / f'rank{self.rank}-{name}.pt'
                    saved_values = dict(values=values, report=producer.direct_readout.last_report)
                    if code_fence_only:
                        # Partial-score contributions are not the PLAN's
                        # complete-event Q/V/A and must not be saved as such.
                        saved_values = dict(diagnostic_only=True,
                            target_selection=record['diagnostic_target_selection'])
                    torch.save(dict(signed=traces[-1], detail=details[-1], **saved_values,
                                    **descriptor, source_sha256=sha(source_path),
                                    native_sha256=sha(native_path)), artifact)
                    record['modes'][name] = dict(seconds=time.perf_counter() - tick,
                        artifact=str(artifact), sha256=sha(artifact), **descriptor,
                        signed_all_finite=bool(torch.isfinite(traces[-1]).all()),
                        saved_worst_token=dict(row=3, packed_slot=2883, token_id=198,
                                              signed=float(traces[-1][3, 2883])),
                        peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(),
                        peak_torch_reserved_bytes=torch.cuda.max_memory_reserved())
                    if not code_fence_only:
                        record['modes'][name]['QVA_all_finite']=all(bool(torch.isfinite(t).all()) for item in values for t in item.values())
                    emit('DT_complete', mode=name, **record['modes'][name])
                    save('DT_complete', active_mode=name)
                runner.attention_pv_rules = original_pv
                runner.finite_fla_by_layer = original_memory
                save('complete', scope=__doc__, production_deployment=False,
                     credit_repaired=False, profile_restored=(runner.attention_pv_rules == original_pv
                         and runner.finite_fla_by_layer == original_memory))
                return dict(rank=self.rank, completed=True, optimizer_steps=0)
            except BaseException:
                import traceback
                save('failed', traceback=traceback.format_exc())
                raise
            finally:
                reward_readout.trace_token_attribution = trace_owner
                if producer is not None:
                    if original_pv is not None:
                        producer.runner.attention_pv_rules = original_pv
                    if original_memory is not None:
                        producer.runner.finite_fla_by_layer = original_memory
                    if release_owner is not None:
                        producer.runner.model.release_finite_layer = release_owner
                    producer.runner.model.release_owner_params()
                if self._is_offload_param:
                    offload_fsdp_model_to_cpu(self.actor_module_fsdp)
                log.close()

    return ExistingPVRuleWorker


if __name__ == '__main__':
    import sys
    from functools import partial
    memory = '--memory' in sys.argv
    if memory:sys.argv.remove('--memory')
    code_fence = '--code-fence-only' in sys.argv
    if code_fence:sys.argv.remove('--code-fence-only')
    diagnostic_owner.make_worker = partial(make_worker, compare_memory=memory, code_fence_only=code_fence)
    diagnostic_owner.main()

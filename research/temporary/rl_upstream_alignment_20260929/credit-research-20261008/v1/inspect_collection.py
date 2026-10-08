"""Native factual/single-EOS measurements on the frozen broad development set.

Uses the previous verified diagnostic's main unchanged for original VERL actor
initialization. Only the diagnostic callback differs. It calls the original
native model/target reader; no DT, optimizer, rollout or alternate scorer.
This measures single-deletion errors, not author cumulative-deletion quality.
"""
from contextlib import nullcontext
import hashlib
import inspect
import json
import os
from pathlib import Path
import time

import torch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_worker():
    import ray
    from verl.single_controller.base.decorator import Dispatch, register
    from verl.workers.fsdp_workers import ActorRolloutRefWorker

    @ray.remote
    class CollectionWorker(ActorRolloutRefWorker):
        @register(dispatch_mode=Dispatch.ONE_TO_ALL)
        def inspect_action_curve(self, source_path, output, case_name='appworld'):
            import psutil
            from inspect_extreme_endpoint import check_imports
            from deltatrace_rollout import DeltaTraceRolloutProducer
            from native_target_logit_rows import NativeTargetLogitRows
            from qwen35_answer_finite import PackedAnswerTargets, selected_target_log_probs
            from verl.utils.fsdp_utils import load_fsdp_model_to_gpu, offload_fsdp_model_to_cpu
            from verl.utils.torch_functional import pad_2d_list_to_length

            query_path = Path(os.environ['DT_RESEARCH_QUERIES'])
            query_document = json.loads(query_path.read_bytes())
            queries = [row for row in query_document['rows'] if row['task'] == case_name]
            assert len(queries) == 32 and all(row['status'] == 'prepared_only' for row in queries)
            root = Path(output)
            source = json.loads(Path(source_path).read_bytes())
            assert self._is_actor and not self._is_rollout
            assert self.actor.config.ppo_micro_batch_size_per_gpu == 4
            assert (self.config.model.lora_rank, self.config.model.lora_alpha) == (8, 16)
            assert not source['checkpoint_restore_requested'] and source['fresh_base_model']
            record = {'task': case_name, 'rank': self.rank, 'pid': os.getpid(),
                      'birth': psutil.Process().create_time(), 'owners': check_imports(source),
                      'source_sha256': sha(source_path), 'query_sha256': sha(query_path),
                      'script_sha256': sha(__file__), 'batches': [],
                      'operations': {'DT': 0, 'backward': 0, 'optimizer': 0,
                                     'rollout': 0, 'checkpoint_restore': 0}}
            log = (root/f'rank{self.rank}-phases.jsonl').open('a', buffering=1)
            native_calls = 0

            def phase(name, **values):
                current = {'phase': name, 'unix': time.time(), 'native_forward_calls': native_calls,
                           'allocated': torch.cuda.memory_allocated(),
                           'reserved': torch.cuda.memory_reserved(),
                           'free': torch.cuda.mem_get_info()[0],
                           'pss_bytes': psutil.Process().memory_full_info().pss, **values}
                log.write(json.dumps(current)+'\n')
                record.update(current)
                (root/f'rank{self.rank}.json').write_text(json.dumps(record, indent=2)+'\n')

            producer = text = attention = None
            training = self.actor_module_fsdp.training
            started = time.perf_counter()
            try:
                lora_B = []
                for name, parameter in self.actor_module_fsdp.named_parameters():
                    if '.lora_B.' in name:
                        local = parameter.detach()
                        local = local.to_local() if hasattr(local, 'to_local') else local
                        lora_B.append(int(torch.count_nonzero(local)))
                assert lora_B and not any(lora_B)
                record['lora_B_nonzero_local'] = lora_B
                if self._is_offload_param:
                    load_fsdp_model_to_gpu(self.actor_module_fsdp)
                self.actor_module_fsdp.eval()
                producer = DeltaTraceRolloutProducer(self.actor_module_fsdp,
                    eos_token_id=self.tokenizer.eos_token_id, pad_token_id=self.tokenizer.pad_token_id)
                runner = producer.runner
                text = runner.model.model.language_model
                attention = text.config._attn_implementation
                text.set_attn_implementation('flash_attention_2')
                precision = nullcontext()
                if producer.native_fla_fp16:
                    from accelerated.qwen35.native_fla_precision import native_fla_fp16
                    precision = native_fla_fp16(self.actor_module_fsdp)
                conv = runner.attribute.__func__.__globals__['_native_conv_initial_states_scope']
                record['reader'] = {}
                for name, function in [('targets', PackedAnswerTargets), ('logit_rows', NativeTargetLogitRows),
                                       ('log_probs', selected_target_log_probs)]:
                    path = inspect.getsourcefile(function)
                    record['reader'][name] = {'path': path, 'sha256': sha(path)}
                torch.cuda.reset_peak_memory_stats()
                phase('original_actor_ready')
                with torch.no_grad(), precision, conv(text.layers, runner.native_conv_initial_states):
                    for batch_start in range(0, len(queries), 4):
                        batch_queries = queries[batch_start:batch_start+4]
                        rows = []
                        for query in batch_queries:
                            binding = query['native']
                            assert sha(binding['path']) == binding['sha256']
                            native = torch.load(binding['path'], map_location='cpu', weights_only=False)
                            row = next(row for row in native['rows'] if row['batch_row'] == query['batch_row'])
                            assert row['traj_uid'] == query['traj_uid']
                            rows.append(row)
                        width = max(row['selected'].numel() for row in rows)
                        original = pad_2d_list_to_length([row['selected'].tolist() for row in rows],
                            self.tokenizer.eos_token_id, max_length=width)
                        targets = PackedAnswerTargets([row['case'] for row in rows],
                            [row['target_offsets'] for row in rows], width, 'cuda')
                        selector = NativeTargetLogitRows(targets)
                        result = {'batch': batch_start//4, 'paired_shape': [8, width],
                                  'rows': [{'traj_uid': row['traj_uid'], 'query_results': []} for row in rows]}
                        record['batches'].append(result)
                        for query_index in range(-1, 12):
                            if time.perf_counter()-started > 14000:
                                raise RuntimeError('Bounded research time exhausted; no continuation/retry')
                            reference = original.clone()
                            if query_index >= 0:
                                for slot, query in enumerate(batch_queries):
                                    point = query['queries'][query_index]
                                    assert int(reference[slot, point['packed_slot']]) == point['token_id']
                                    reference[slot, point['packed_slot']] = self.tokenizer.eos_token_id
                            pair = torch.stack((reference, original), dim=1).flatten(0, 1).cuda()
                            tick = time.perf_counter()
                            model_output = runner.model.forward_root(input_ids=pair,
                                attention_mask=torch.ones_like(pair), use_cache=False,
                                logits_to_keep=selector.rows)
                            logits = selector.pack_logits(model_output.logits)
                            del model_output
                            logp = selected_target_log_probs(logits, targets)
                            del logits
                            left = targets.sample_sums(logp[0::2].double()).cpu()
                            right = targets.sample_sums(logp[1::2].double()).cpu()
                            torch.cuda.synchronize()
                            native_calls += 1
                            for slot, query in enumerate(batch_queries):
                                value = {'reference_joint_logp': float(left[slot]),
                                         'factual_joint_logp': float(right[slot]),
                                         'native_d': float(right[slot]-left[slot])}
                                if query_index < 0:
                                    result['rows'][slot]['identity'] = value
                                else:
                                    result['rows'][slot]['query_results'].append({**query['queries'][query_index], **value})
                            phase('native_query_complete', active_batch=batch_start//4,
                                  query_index=query_index, call_seconds=time.perf_counter()-tick,
                                  completed_trajectories=batch_start)
                            del pair, reference, logp
                        phase('batch_complete', completed_trajectories=batch_start+4)
                        del rows, native, original, targets, selector
                phase('complete', completed_trajectories=len(queries), seconds=time.perf_counter()-started,
                      torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      scope='Broad frozen source-deletion diagnosis only; author RISE/MAS comparison not performed here')
                return {'rank': self.rank, 'completed': True, 'trajectories': len(queries), 'optimizer_steps': 0}
            except BaseException:
                import traceback
                phase('failed', traceback=traceback.format_exc())
                raise
            finally:
                if text is not None and attention is not None:
                    text.set_attn_implementation(attention)
                self.actor_module_fsdp.train(training)
                if producer is not None:
                    producer.runner.model.release_owner_params()
                if self._is_offload_param:
                    offload_fsdp_model_to_cpu(self.actor_module_fsdp)
                log.close()
    return CollectionWorker


if __name__ == '__main__':
    raise SystemExit('Withdrawn by user: symptom refinement adds model passes; do not launch this draft.')
    # Reuse the previous diagnostic main for owner config/model/FSDP setup.
    import inspect_action_curve as owner_entry
    assert sha(owner_entry.__file__) == '7277fade4e9b1cb49f825e4cb2fc54453c9ff3ce4859b20350aa2bd67ffaf3a6'
    owner_entry.make_worker = make_worker
    owner_entry.main()

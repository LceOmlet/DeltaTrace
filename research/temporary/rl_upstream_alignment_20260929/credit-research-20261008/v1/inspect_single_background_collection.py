"""Separate joint-reference allocation from single-reference finite error.

Use every already frozen source query, unchanged real B4 rows and the existing
producer/trace interface. Only the reference IDs of a diagnostic call change:
one queried token per row becomes EOS, all other IDs stay factual. The original
joint DT and native single-delete results are reused, never regenerated here.
This is a diagnostic, not a candidate estimator or extra training-time calls.
"""
import inspect
import json
import os
from pathlib import Path
import time

import torch
import inspect_action_curve as initializer
from inspect_extreme_endpoint import check_imports, sha


def single_reference(factual, queries, eos):
    reference = factual.clone()
    for row, query in enumerate(queries):
        if query is not None:
            slot = query['packed_slot']
            assert int(factual[row, slot]) == query['token_id']
            reference[row, slot] = eos
    return reference


def make_worker():
    import ray
    from verl.single_controller.base.decorator import Dispatch, register
    from verl.workers.fsdp_workers import ActorRolloutRefWorker

    @ray.remote
    class SingleBackgroundWorker(ActorRolloutRefWorker):
        @register(dispatch_mode=Dispatch.ONE_TO_ALL)
        def inspect_action_curve(self, source_path, output, case_name='appworld'):
            import psutil
            import reward_readout
            from deltatrace_rollout import DeltaTraceRolloutProducer
            from verl.utils.fsdp_utils import load_fsdp_model_to_gpu, offload_fsdp_model_to_cpu

            out = Path(output)
            plan_path = out.parent/'layer-collection-inputs.json'
            spec = json.loads(plan_path.read_bytes())['tasks'][case_name]
            source = json.loads(Path(source_path).read_bytes())
            record = dict(scope=__doc__, task=case_name, rank=self.rank, pid=os.getpid(),
                birth=psutil.Process().create_time(), script_sha256=sha(__file__),
                source_sha256=sha(source_path), plan_sha256=sha(plan_path), owners=check_imports(source),
                batches=[], diagnostic_wall_budget_seconds=1800,
                operations=dict(DT=0, native_forward=0, optimizer=0, backward=0, rollout=0, checkpoint_restore=0))
            phases = (out/f'rank{self.rank}-phases.jsonl').open('x', buffering=1)
            started = time.perf_counter()

            def save(phase, **values):
                event = dict(phase=phase, unix=time.time(), elapsed_seconds=time.perf_counter()-started,
                    allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved(),
                    runtime_free_bytes=torch.cuda.mem_get_info()[0], process_pss_bytes=psutil.Process().memory_full_info().pss,
                    **values)
                record.update(event)
                phases.write(json.dumps(event)+'\n')
                (out/f'rank{self.rank}.json').write_text(json.dumps(record,indent=2)+'\n')

            producer = None
            previous_training = self.actor_module_fsdp.training
            original_trace = reward_readout.trace_token_attribution
            try:
                assert self.actor.config.ppo_micro_batch_size_per_gpu == 4
                assert (self.config.model.lora_rank,self.config.model.lora_alpha) == (8,16)
                assert source['fresh_base_model'] and not source['checkpoint_restore_requested']
                shards = []
                for name, parameter in self.actor_module_fsdp.named_parameters():
                    if '.lora_B.' in name:
                        value = parameter.detach()
                        value = value.to_local() if hasattr(value,'to_local') else value
                        shards.append(dict(name=name,elements=value.numel(),nonzero=int(torch.count_nonzero(value))))
                assert shards and not any(v['nonzero'] for v in shards)
                record['lora_B_local_shards'] = shards
                if self._is_offload_param:
                    load_fsdp_model_to_gpu(self.actor_module_fsdp)
                self.actor_module_fsdp.eval()
                producer = DeltaTraceRolloutProducer(self.actor_module_fsdp,
                    eos_token_id=self.tokenizer.eos_token_id,pad_token_id=self.tokenizer.pad_token_id)
                runner_path = inspect.getsourcefile(producer.runner.attribute)
                assert sha(runner_path) == source['actual_CPU_imports']['qwen35_dense_finite_runner']['sha256']
                record['finite_runner'] = dict(path=runner_path,sha256=sha(runner_path))
                entries_by_uid = {entry['traj_uid']:entry for entry in spec['entries']}
                for pair_index in range(0,len(spec['batches']),2):
                    batch_index = pair_index+self.rank
                    batch_spec = spec['batches'][batch_index]
                    entries = [entries_by_uid[uid] for uid in batch_spec['uids']]
                    entries += [entries[-1]]*(4-len(entries))
                    rows = []
                    for entry in entries:
                        assert sha(entry['native']['path']) == entry['native']['sha256']
                        native = torch.load(entry['native']['path'],map_location='cpu',weights_only=False)
                        rows.append(next(row for row in native['rows'] if str(row['traj_uid']) == entry['traj_uid']))
                        del native
                    assert [row['selected'].numel() for row in rows] == sorted(row['selected'].numel() for row in rows)
                    batch = dict(index=batch_index,uids=batch_spec['uids'],points=[],rounds=[])
                    record['batches'].append(batch)
                    rounds = max(spec['batches'][i]['native_paired_forwards'] for i in (pair_index,pair_index+1))
                    for round_index in range(rounds):
                        if time.perf_counter()-started > record['diagnostic_wall_budget_seconds']:
                            raise RuntimeError('Existing 1800-second diagnostic budget reached; preserve partial data, no automatic retry.')
                        queries = [entry['queries'][round_index] if row < batch_spec['actual_rows'] and round_index < len(entry['queries']) else None
                                   for row,entry in enumerate(entries)]
                        captured = {}

                        def trace(owner, reference, factual, *args, **kwargs):
                            assert owner is producer.runner and factual.shape[0] == 4 and not captured
                            reference = single_reference(factual,queries,self.tokenizer.eos_token_id)
                            signed, roots, detail = original_trace(owner,reference,factual,*args,**kwargs)
                            captured.update(signed=signed.detach().cpu(),roots=roots.detach().cpu(),detail=detail)
                            return signed,roots,detail

                        save('single_DT_begin',batch=batch_index,round=round_index,queried_rows=sum(q is not None for q in queries))
                        tick = time.perf_counter()
                        reward_readout.trace_token_attribution = trace
                        try:
                            producer.attribute_episodes([[row['row'] for row in rows]],[0.0])
                        finally:
                            reward_readout.trace_token_attribution = original_trace
                        assert captured
                        record['operations']['DT'] += 1
                        artifact = out/f'rank{self.rank}-batch{batch_index}-round{round_index}.pt'
                        torch.save(captured,artifact)
                        for row,query in enumerate(queries):
                            if query is None:
                                continue
                            detail = captured['detail']['per_sample'][row]
                            signed = captured['signed'][row]
                            batch['points'].append(dict(query,traj_uid=entries[row]['traj_uid'],
                                initial_state_sha256=entries[row]['initial_state_sha256'],
                                previously_examined=entries[row]['previously_examined'],
                                single_background_DT_d=float(signed[query['packed_slot']]),
                                single_background_root=float(captured['roots'][row]),
                                single_background_factual_logp=detail['factual_target_logp'],
                                single_background_deleted_logp=detail['reference_target_logp'],
                                signed_off_source_maxabs=float(torch.cat((signed[:query['packed_slot']],signed[query['packed_slot']+1:])).abs().max()),
                                root_minus_self_d=float(captured['roots'][row])-float(signed[query['packed_slot']]),
                                artifact=dict(path=str(artifact),sha256=sha(artifact))))
                        batch['rounds'].append(dict(round=round_index,seconds=time.perf_counter()-tick,queries=sum(q is not None for q in queries)))
                        save('single_DT_complete',batch=batch_index,round=round_index,
                             completed_points=sum(len(b['points']) for b in record['batches']))
                        del captured
                    del rows
                save('complete',completed_points=sum(len(b['points']) for b in record['batches']))
                return dict(rank=self.rank,completed=True,optimizer_steps=0)
            except BaseException:
                import traceback
                save('failed',traceback=traceback.format_exc())
                raise
            finally:
                reward_readout.trace_token_attribution = original_trace
                self.actor_module_fsdp.train(previous_training)
                if producer is not None:
                    producer.runner.model.release_owner_params()
                if self._is_offload_param:
                    offload_fsdp_model_to_cpu(self.actor_module_fsdp)
                phases.close()
    return SingleBackgroundWorker


if __name__ == '__main__':
    initializer.make_worker = make_worker
    initializer.main()

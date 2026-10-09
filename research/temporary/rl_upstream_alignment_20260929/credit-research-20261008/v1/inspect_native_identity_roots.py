"""Observe unchanged native root inputs before finite propagation, then stop.

Original VERL initializes the actor. Original producer, prefix provider,
runner, HF model and target scorer own the entire root call. Each original
frozen B4 is read once, with identical factual/reference IDs. The observer
raises before the finite seed; it never supplies fabricated credit or a
replacement model result. Descriptive differences are not a new tolerance.
"""
import gc
from contextlib import nullcontext
import inspect
import json
import os
from pathlib import Path
import time

import torch
import inspect_action_curve as initializer
from inspect_extreme_endpoint import check_imports, sha


class NativeRootRead(Exception):
    """Intentional diagnostic stop before the first finite operator."""


def pair_difference(value, row, length=None):
    a, b = value[2*row], value[2*row+1]
    if length is not None:
        a, b = a[:length], b[:length]
    result = dict(dtype=str(value.dtype), shape=list(a.shape), equal=torch.equal(a, b))
    if result['equal']:
        result.update(maxabs=0.0, changed_elements=0, first_changed_token=None)
        return result
    maximum, changed, first = 0.0, 0, None
    for start in range(0, a.shape[0], 256):
        delta = a[start:start+256].float()-b[start:start+256].float()
        nz = delta.ne(0)
        changed += int(nz.sum())
        maximum = max(maximum, float(delta.abs().max()))
        if first is None and bool(nz.any()):
            first = start+int(nz.reshape(nz.shape[0], -1).any(-1).nonzero()[0])
    result.update(maxabs=maximum, changed_elements=changed, first_changed_token=first)
    return result


def make_worker(*, operator_observer=None, batch_indices=None):
    import ray
    from verl.single_controller.base.decorator import Dispatch, register
    from verl.workers.fsdp_workers import ActorRolloutRefWorker

    @ray.remote
    class IdentityRootWorker(ActorRolloutRefWorker):
        @register(dispatch_mode=Dispatch.ONE_TO_ALL)
        def inspect_action_curve(self, source_path, output, case_name='appworld'):
            import psutil
            import reward_readout
            from deltatrace_rollout import DeltaTraceRolloutProducer
            from verl.utils.fsdp_utils import load_fsdp_model_to_gpu, offload_fsdp_model_to_cpu

            out = Path(output)
            source = json.loads(Path(source_path).read_bytes())
            spec = json.loads((out.parent/'layer-collection-inputs.json').read_bytes())['tasks'][case_name]
            version = json.loads((out.parent/'protocol.json').read_bytes())
            record = dict(scope=__doc__, rank=self.rank, task=case_name, pid=os.getpid(),
                birth=psutil.Process().create_time(), script_sha256=sha(__file__),
                source_sha256=sha(source_path), owners=check_imports(source), batches=[],
                operations=dict(native_root=0, finite_seed=0, DT=0, optimizer=0,
                    backward=0, rollout=0, checkpoint_restore=0))
            events = (out/f'rank{self.rank}-phases.jsonl').open('x', buffering=1)
            started = time.perf_counter()
            def save(phase, **fields):
                record.update(phase=phase, unix=time.time(), seconds=time.perf_counter()-started,
                    allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved(),
                    runtime_free_bytes=torch.cuda.mem_get_info()[0],
                    process_pss_bytes=psutil.Process().memory_full_info().pss, **fields)
                events.write(json.dumps({k:v for k,v in record.items() if k not in ('batches','owners')})+'\n')
                (out/f'rank{self.rank}.json').write_text(json.dumps(record, indent=2)+'\n')

            old_trace = reward_readout.trace_token_attribution
            training = self.actor_module_fsdp.training
            producer = None
            try:
                assert self.actor.config.ppo_micro_batch_size_per_gpu == 4
                assert (self.config.model.lora_rank, self.config.model.lora_alpha) == (8,16)
                assert source['fresh_base_model'] and not source['checkpoint_restore_requested']
                for name, parameter in self.actor_module_fsdp.named_parameters():
                    if '.lora_B.' in name:
                        p=parameter.detach();p=p.to_local() if hasattr(p,'to_local') else p
                        assert not int(torch.count_nonzero(p))
                if self._is_offload_param:
                    load_fsdp_model_to_gpu(self.actor_module_fsdp)
                self.actor_module_fsdp.eval()
                producer = DeltaTraceRolloutProducer(self.actor_module_fsdp,
                    eos_token_id=self.tokenizer.eos_token_id, pad_token_id=self.tokenizer.pad_token_id)
                runner = producer.runner
                globals_ = runner.attribute.__func__.__globals__
                record['actual_numerical_owners'] = {}
                for name, expected in version['numerical_owners'].items():
                    import importlib
                    path=Path(importlib.import_module(name).__file__)
                    assert sha(path)==expected['sha256'], name
                    record['actual_numerical_owners'][name]=dict(path=str(path),resolved=str(path.resolve()),sha256=sha(path))
                original_answer = runner.answer
                entries = {e['traj_uid']:e for e in spec['entries']}
                active = {}

                def stop_before_seed(*_args, **_kwargs):
                    # The original scorer has completed. Read that runner's
                    # own locals rather than constructing another forward.
                    frame=inspect.currentframe().f_back
                    while frame is not None and frame.f_code is not runner.attribute.__func__.__code__:
                        frame=frame.f_back
                    assert frame is not None
                    f=frame.f_locals
                    original_rows=active['rows']
                    row_starts=f.get('row_starts')
                    starts=(list(row_starts) if row_starts is not None else [f['prefix_start']]*4)
                    selection=f['selection']
                    sums0=selection.sample_sums(f['root_lp0'].to(selection.samples.device).double()).cpu()
                    sums1=selection.sample_sums(f['root_lp1'].to(selection.samples.device).double()).cpu()
                    masks=f['mask'] if isinstance(f['mask'],dict) else {'dense':f['mask']}
                    cache_fields=[]
                    cache=f.get('replay_cache')
                    for i,layer in enumerate(getattr(cache,'layers',[])):
                        for name,value in vars(layer).items():
                            if isinstance(value,torch.Tensor) and value.ndim and value.shape[0]==8:
                                cache_fields.append(dict(layer=i,field=name,shape=list(value.shape),dtype=str(value.dtype),
                                    pair_equal=[torch.equal(value[2*b],value[2*b+1]) for b in range(4)]))
                    rows=[]
                    for row,item in enumerate(original_rows):
                        count=item['selected'].numel()-starts[row]
                        boundary=[]
                        for i in range(33):
                            value=f['root']['final_norm_input' if i==32 else str(i)]
                            boundary.append(dict(boundary=i,**pair_difference(value,row,count)))
                        rows.append(dict(traj_uid=item['traj_uid'],prefix_start=starts[row],
                            valid_suffix_tokens=count,boundaries=boundary,
                            factual_logp=float(sums1[row]),reference_logp=float(sums0[row]),
                            input_pair_equal=torch.equal(f['paired_ids'][2*row],f['paired_ids'][2*row+1]),
                            masks_pair_equal={k:torch.equal(v[2*row],v[2*row+1]) for k,v in masks.items()},
                            positions_pair_equal=(torch.equal(f['root_positions']['position_ids'][2*row],
                                f['root_positions']['position_ids'][2*row+1]) if f.get('root_positions') else None)))
                    active['result']=dict(index=active['index'],rows=rows,
                        original_prefix_preparation=active.get('prefix_report'),
                        original_runner_calls=f['calls'],
                        duplicated_cache_fields=cache_fields,
                        actual_suffix_shape=list(f['paired_ids'].shape),prefix_start=f['prefix_start'],
                        row_starts=starts,first_unequal_boundary_by_row=[
                            next((b['boundary'] for b in r['boundaries'] if not b['equal']),None) for r in rows])
                    del f,frame
                    raise NativeRootRead()

                def trace(owner, reference, factual, *args, **kwargs):
                    assert owner is runner and factual.shape[0]==4
                    frame=inspect.currentframe().f_back
                    active['prefix_report']=frame.f_locals.get('report',{}).get('shared_native_prefix')
                    del frame
                    # Only the diagnostic perturbation changes. Both native
                    # endpoints now carry the same exact original IDs.
                    reference=factual.clone()
                    observation = (operator_observer(runner,out,active)
                                   if operator_observer is not None else nullcontext())
                    with observation:
                        return old_trace(owner,reference,factual,*args,**kwargs)

                runner.answer=stop_before_seed
                reward_readout.trace_token_attribution=trace
                indices = tuple(range(len(spec['batches']))) if batch_indices is None else tuple(batch_indices)
                assert len(indices)%2 == 0 and len(set(indices)) == len(indices)
                assert all(0 <= i < len(spec['batches']) for i in indices)
                record['original_batch_indices'] = indices
                for pair_index in range(0,len(indices),2):
                    if time.perf_counter()-started > 1800:
                        raise RuntimeError('Original bounded diagnostic wall budget reached; no retry')
                    index=indices[pair_index+self.rank]
                    batch=spec['batches'][index]
                    selected=[entries[uid] for uid in batch['uids']]
                    selected += [selected[-1]]*(4-len(selected))
                    rows=[]
                    for e in selected:
                        assert sha(e['native']['path'])==e['native']['sha256']
                        native=torch.load(e['native']['path'],map_location='cpu',weights_only=False)
                        rows.append(next(r for r in native['rows'] if str(r['traj_uid'])==e['traj_uid']))
                        del native
                    active=dict(index=index,rows=rows,rank=self.rank)
                    save('native_identity_root_begin',batch=index)
                    try:
                        producer.attribute_episodes([[r['row'] for r in rows]],[0.0])
                    except NativeRootRead:
                        assert 'result' in active
                    else:
                        raise AssertionError('Unexpected finite propagation: diagnostic stop not reached')
                    record['operations']['native_root']+=1
                    record['batches'].append(active['result'])
                    runner.model.release_owner_params()
                    active.clear();del rows
                    # Early diagnostic exit skips the normal finite consumer
                    # that releases the root checkpoint bank. Collect only
                    # these now-unreachable Python frame/checkpoint cycles.
                    gc.collect()
                    save('native_identity_root_complete',batch=index)
                runner.answer=original_answer
                save('complete',numeric_scope='Description only; no new official tolerance evaluated or changed.')
                return dict(rank=self.rank,completed=True,optimizer_steps=0)
            except BaseException:
                import traceback
                save('failed',traceback=traceback.format_exc())
                raise
            finally:
                reward_readout.trace_token_attribution=old_trace
                self.actor_module_fsdp.train(training)
                if producer is not None:
                    producer.runner.model.release_owner_params()
                if self._is_offload_param:
                    offload_fsdp_model_to_cpu(self.actor_module_fsdp)
                events.close()
    return IdentityRootWorker


if __name__=='__main__':
    initializer.make_worker=make_worker
    initializer.main()

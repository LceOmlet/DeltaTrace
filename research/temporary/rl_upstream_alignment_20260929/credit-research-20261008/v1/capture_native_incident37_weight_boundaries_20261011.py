"""Observe the demonstrated bad weight row across existing native boundaries.

No parameter writes, added model/DT/GEMM calls, altered collectives, or waits
before native replay exits. CPU shard rows are cloned; GPU rows use queued
pinned D2H copies. Timing changes are possible and finite output is no repair.
"""
import functools
import hashlib
import inspect
import json
from pathlib import Path
import time

import capture_native_incident37_async_head_20261011 as async_head

ASYNC_SHA = '782db0e8548f7cfb12bcf9c24271400da895b8bcb9e008eb3813f9014a0e9cc2'
VOCAB_ID = 159048


def make_observed_worker(native_factory):
    import ray
    actor = original_async_factory(native_factory)
    native_class = actor.__ray_metadata__.modified_class.__ray_actor_class__
    native_replay = native_class.replay_incident
    native_old = native_class.compute_log_prob
    native_ref = native_class.compute_ref_log_prob
    native_dt = native_class.compute_dt_token_advantages
    native_update = native_class.update_actor

    class NativeWeightBoundaryObserved(native_class):
        def _observe_weight_row(self, label, tensor, **metadata):
            import torch
            item = dict(label=label, unix_ns=time.time_ns(), **metadata)
            item.update(shape=list(tensor.shape), dtype=str(tensor.dtype),
                        device=str(tensor.device), pointer=tensor.data_ptr(),
                        storage_bytes=tensor.untyped_storage().nbytes())
            if tensor.device.type == 'cpu':
                item['buffer'] = tensor.detach().clone()
            else:
                item['buffer'] = torch.empty_like(tensor, device='cpu', pin_memory=True)
                item['buffer'].copy_(tensor.detach(), non_blocking=True)
                item['event'] = torch.cuda.Event()
                item['event'].record(torch.cuda.current_stream())
                item['stream'] = torch.cuda.current_stream().cuda_stream
            self._weight_observations.append(item)

        def _observe_sharded_weight(self, label):
            names = {id(m): n for n, m in self.actor_module_fsdp.named_modules()}
            for module in self.actor_module_fsdp.modules():
                get_state = getattr(module, '_get_fsdp_state', None)
                if get_state is None:
                    continue
                group = get_state()._fsdp_param_group
                if group is None:
                    continue
                for p in group.fsdp_params:
                    info = p._module_info
                    name = names.get(id(info.module), '')
                    if not name.endswith('lm_head') or info.param_name != 'weight':
                        continue
                    value = p._sharded_local_tensor
                    rank = p.mesh_info.shard_mesh_rank
                    lo = rank * p.padded_sharded_param_size[0]
                    meta = dict(parameter=name + '.weight', shard_rank=rank,
                                global_row=VOCAB_ID, shard_start=lo,
                                shard_shape=list(value.shape), sharded_state=str(p.sharded_state))
                    if lo <= VOCAB_ID < lo + value.shape[0]:
                        self._observe_weight_row(label, value[VOCAB_ID - lo], **meta)
                    else:
                        self._weight_observations.append(dict(label=label,
                            unix_ns=time.time_ns(), not_owned_on_this_rank=True, **meta))

        @functools.wraps(native_replay)
        def replay_incident(self, output):
            import torch
            self._weight_observations = []
            try:
                return native_replay(self, output)
            finally:
                for item in self._weight_observations:
                    event = item.pop('event', None)
                    if event is not None:
                        event.synchronize()
                path = Path(output) / f'rank{self.rank}-weight-boundaries.pt'
                torch.save(self._weight_observations, path)
                metadata = [{k: v for k, v in row.items() if k != 'buffer'}
                            for row in self._weight_observations]
                report = dict(rank=self.rank, source=__file__,
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    artifact=dict(path=str(path), bytes=path.stat().st_size,
                                  sha256=hashlib.sha256(path.read_bytes()).hexdigest()),
                    observations=metadata, extra_model_calls=0, extra_DT_calls=0,
                    pre_replay_exit_explicit_synchronizations=0, production_changes=0)
                (Path(output) / f'rank{self.rank}-weight-boundaries.json').write_text(
                    json.dumps(report, indent=2) + '\n')

        @functools.wraps(native_old)
        def compute_log_prob(self, data):
            self._observe_sharded_weight('before_native_old')
            result = native_old(self, data)
            self._observe_sharded_weight('after_native_old')
            return result

        @functools.wraps(native_ref)
        def compute_ref_log_prob(self, data):
            self._observe_sharded_weight('before_native_ref')
            result = native_ref(self, data)
            self._observe_sharded_weight('after_native_ref')
            return result

        @functools.wraps(native_dt)
        def compute_dt_token_advantages(self, data):
            self._observe_sharded_weight('before_native_DT')
            result = native_dt(self, data)
            self._observe_sharded_weight('after_native_DT')
            return result

        @functools.wraps(native_update)
        def update_actor(self, data):
            import torch
            from verl.utils.experimental import torch_functional as head
            native_head = head.FusedLinearForPPO.forward
            native_head_backward = head.FusedLinearForPPOFunction.backward
            native_backward = torch.Tensor.backward
            signature = inspect.signature(native_head)
            forward_count = backward_count = 0
            self._observe_sharded_weight('before_native_update')

            @functools.wraps(native_head)
            def observed_head(*args, **kwargs):
                nonlocal forward_count
                index = forward_count
                forward_count += 1
                w = signature.bind(*args, **kwargs).arguments['vocab_weights']
                self._observe_sharded_weight(f'head{index}_pre_shard')
                self._observe_weight_row(f'head{index}_pre_full', w[VOCAB_ID])
                result = native_head(*args, **kwargs)
                self._observe_weight_row(f'head{index}_post_full', w[VOCAB_ID])
                return result

            @functools.wraps(native_head_backward)
            def observed_head_backward(ctx, *args, **kwargs):
                w = ctx.saved_tensors[1]
                self._observe_weight_row('native_head_backward_pre_full', w[VOCAB_ID])
                result = native_head_backward(ctx, *args, **kwargs)
                self._observe_weight_row('native_head_backward_post_full', w[VOCAB_ID])
                self._observe_sharded_weight('native_head_backward_post_shard')
                return result

            @functools.wraps(native_backward)
            def observed_backward(tensor, *args, **kwargs):
                nonlocal backward_count
                result = native_backward(tensor, *args, **kwargs)
                self._observe_sharded_weight(f'after_native_backward{backward_count}')
                backward_count += 1
                return result

            head.FusedLinearForPPO.forward = observed_head
            head.FusedLinearForPPOFunction.backward = staticmethod(observed_head_backward)
            torch.Tensor.backward = observed_backward
            try:
                return native_update(self, data)
            finally:
                head.FusedLinearForPPO.forward = native_head
                head.FusedLinearForPPOFunction.backward = staticmethod(native_head_backward)
                torch.Tensor.backward = native_backward

    return ray.remote(NativeWeightBoundaryObserved)


original_async_factory = async_head.make_observed_worker


def main():
    assert hashlib.sha256(Path(async_head.__file__).read_bytes()).hexdigest() == ASYNC_SHA
    async_head.make_observed_worker = make_observed_worker
    try:
        async_head.main()
    finally:
        async_head.make_observed_worker = original_async_factory


if __name__ == '__main__':
    main()

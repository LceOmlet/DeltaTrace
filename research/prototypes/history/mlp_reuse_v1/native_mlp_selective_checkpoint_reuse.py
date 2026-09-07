"""Reuse actual native MLP GEMM outputs with PyTorch's public SAC contexts.

The same native MLP is called during root capture and same-endpoint layer replay.
Only the three GEMMs are cached by unmodified PyTorch; attention/finite rules,
model methods and autograd are not replaced. This is checkpoint recomputation,
not a new model/counterfactual forward. Each layer has its own FIFO context pair.
"""
import inspect
import hashlib
from pathlib import Path


class NativeMLPSelectiveReuse:
    def __init__(self, model, activity):
        import torch
        from torch.utils.checkpoint import create_selective_checkpoint_contexts, CheckpointPolicy
        self.model = model
        self.activity = activity
        self.handles = []
        self.regions = []
        self.active = None
        assert not model.training and all(not v.requires_grad for v in model.parameters())
        assert model.config.hidden_act == 'silu'
        source = Path(inspect.getfile(create_selective_checkpoint_contexts))
        activity.update(public_API='torch.utils.checkpoint.create_selective_checkpoint_contexts',
            installed_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            copied_dispatch_implementation=False, native_model_methods_replaced=False,
            allow_cache_entry_mutation=False, regions=[], saved_mm_outputs=0,
            reused_mm_outputs=0, retained_tensor_bytes=0, maximum_retained_tensor_bytes=0,
            cache_scope='One actual endpoint-pair root and its existing same-endpoint decoder replay only; separate region per native MLP, three mm results. No cache across attribution invocations.')
        for index, layer in enumerate(model.model.layers):
            mlp = layer.mlp
            assert 'forward' not in mlp.__dict__
            weights = [mlp.gate_proj.weight, mlp.up_proj.weight, mlp.down_proj.weight]
            assert all(w.dtype == torch.float16 for w in weights)
            rec = {'layer':index, 'save_calls':0, 'reuse_calls':0, 'input_exact':None,
                   'output_bytes':0, 'input_bytes':0, 'completed':False, 'operations':[]}
            state = {'record':rec, 'input':None, 'contexts':None, 'phase':0,
                     'weights':weights, 'weight_versions':[w._version for w in weights],
                     'native_forward':type(mlp).forward}
            def policy(ctx, op, *args, _state=state, **kwargs):
                if op is not torch.ops.aten.mm.default:
                    return CheckpointPolicy.MUST_RECOMPUTE
                rec = _state['record']
                key = 'reuse_calls' if ctx.is_recompute else 'save_calls'
                step = rec[key]
                assert step < 3 and not kwargs
                x, weight_t = args
                weight = _state['weights'][step]
                assert weight._version == _state['weight_versions'][step]
                assert weight_t.data_ptr() == weight.data_ptr()
                assert weight_t.shape == weight.T.shape and weight_t.stride() == weight.T.stride()
                assert x.ndim == 2 and x.dtype == weight_t.dtype == torch.float16
                descriptor = {'projection':['gate', 'up', 'down'][step],
                    'input_shape':list(x.shape), 'input_stride':list(x.stride()),
                    'weight_shape':list(weight.shape), 'output_bytes':x.shape[0]*weight_t.shape[1]*2}
                if ctx.is_recompute:
                    assert descriptor == rec['operations'][step]
                    self.activity['reused_mm_outputs'] += 1
                else:
                    rec['operations'].append(descriptor)
                    rec['output_bytes'] += descriptor['output_bytes']
                    self.activity['saved_mm_outputs'] += 1
                    self.activity['retained_tensor_bytes'] += descriptor['output_bytes']
                    self.activity['maximum_retained_tensor_bytes'] = max(
                        self.activity['maximum_retained_tensor_bytes'], self.activity['retained_tensor_bytes'])
                rec[key] += 1
                return CheckpointPolicy.MUST_SAVE
            state['contexts'] = create_selective_checkpoint_contexts(policy, allow_cache_entry_mutation=False)
            def enter(module, args, _state=state):
                assert self.active is None and len(args) == 1
                assert type(module).forward is _state['native_forward'] and 'forward' not in module.__dict__
                phase = _state['phase']; assert phase in (0, 1)
                value = args[0]
                rec = _state['record']
                if phase == 0:
                    _state['input'] = value.detach()
                    _state['input_version'] = value._version
                    rec['input_bytes'] = value.numel()*value.element_size()
                    self.activity['retained_tensor_bytes'] += rec['input_bytes']
                    self.activity['maximum_retained_tensor_bytes'] = max(
                        self.activity['maximum_retained_tensor_bytes'], self.activity['retained_tensor_bytes'])
                else:
                    assert _state['input']._version == _state['input_version']
                    assert value.shape == _state['input'].shape and value.stride() == _state['input'].stride()
                    rec['input_exact'] = bool(torch.equal(value, _state['input']))
                    assert rec['input_exact'], 'Refuse to reuse outputs for a different native MLP input.'
                    assert rec['save_calls'] == 3
                assert all(w._version == v for w, v in zip(_state['weights'], _state['weight_versions']))
                context = _state['contexts'][phase]
                context.__enter__()
                self.active = (_state, context)
            def leave(module, args, output, _state=state):
                assert self.active is not None and self.active[0] is _state
                _, context = self.active
                self.active = None
                context.__exit__(None, None, None)
                if output is None: return
                rec = _state['record']
                if _state['phase'] == 1:
                    assert rec['save_calls'] == rec['reuse_calls'] == 3
                    self.activity['retained_tensor_bytes'] -= rec['input_bytes']+rec['output_bytes']
                    _state['input'] = None
                    _state['contexts'] = None
                    rec['completed'] = True
                _state['phase'] += 1
            self.handles += [mlp.register_forward_pre_hook(enter),
                             mlp.register_forward_hook(leave, always_call=True)]
            self.regions.append(state)
            self.activity['regions'].append(rec)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.active is not None:
            _, context = self.active
            self.active = None
            context.__exit__(exc_type, exc, tb)
        for handle in self.handles: handle.remove()
        self.handles.clear()
        if exc_type is None:
            assert all(s['record']['completed'] for s in self.regions)
            assert self.activity['saved_mm_outputs'] == self.activity['reused_mm_outputs'] == 3*len(self.regions)
            assert self.activity['retained_tensor_bytes'] == 0
        for state in self.regions:
            state['input'] = None
            state['contexts'] = None
        self.regions.clear()
        return False

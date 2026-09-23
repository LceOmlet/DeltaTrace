"""Passive capture of an original attention module and its default FA interface.

No replacement callables or monkey patches. Captures are diagnostics/replay
operands and incur CPU transfers; do not include them in unlabelled warm timing.
"""
import sys
import torch


def copy_capture_tensor(value, destination, *, copy=True, preserve_strides=False, pinned_host=False):
    value=value.detach()
    if pinned_host and torch.device(destination).type=='cpu' and value.is_cuda:
        # The capture owner waits once at scope exit before CPU consumption.
        # Reuse Torch's pinned allocator and ordinary stream-ordered copy.
        return torch.empty_strided(value.shape,value.stride(),dtype=value.dtype,
                                   device='cpu',pin_memory=True).copy_(value,non_blocking=True)
    if preserve_strides:
        # Explicit storage layout avoids the MetaX cross-device 4-D copy
        # choosing channels-last and changing downstream reduction kernels.
        return torch.empty_strided(value.shape,value.stride(),dtype=value.dtype,
                                   device=destination).copy_(value,non_blocking=(
                                       value.device.type=='cpu' and value.is_pinned()
                                       and torch.device(destination).type=='cuda'))
    return value.to(destination,copy=copy)


def _metadata_scalar(value):
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError('Expected scalar FA argument metadata.')
        return value.item()
    return value


class NativeAttentionCapture:
    def __init__(self, module, attention_interface, native_varlen_function, destination='cpu', *, copy_tensors=True, retained_names=None, preserve_strides=False, pinned_host=False):
        self.module = module
        self.interface = attention_interface
        self.native_varlen = native_varlen_function
        self.destination = destination
        self.copy_tensors = copy_tensors
        self.retained_names = retained_names
        self.preserve_strides = preserve_strides
        self.pinned_host = pinned_host
        self.values = {}
        self.calls = {'module': 0, 'interface': 0, 'native_varlen': 0}
        self.handles = []

    def retain(self, name, value):
        if self.retained_names is None or name in self.retained_names:
            self.values[name] = copy_capture_tensor(value,self.destination,
                copy=self.copy_tensors,preserve_strides=self.preserve_strides,pinned_host=self.pinned_host)

    def profile(self, frame, event, result):
        if frame.f_code is self.native_varlen.__code__ and event == 'call':
            self.calls['native_varlen'] += 1
            for name in ('q', 'k', 'v', 'cu_seqlens_q', 'cu_seqlens_k'):
                self.retain('packed_' + name, frame.f_locals[name])
            self.packed_arguments = {name: _metadata_scalar(frame.f_locals[name]) for name in
                ('max_seqlen_q', 'max_seqlen_k', 'dropout_p', 'softmax_scale', 'causal', 'return_attn_probs')}
        if frame.f_code is not self.interface.__code__ or frame.f_locals.get('module') is not self.module:
            return
        if event == 'call':
            self.calls['interface'] += 1
            for name in ('query', 'key', 'value', 'attention_mask'):
                value = frame.f_locals[name]
                if value is not None:
                    self.retain(name, value)
            self.interface_arguments = {name: _metadata_scalar(frame.f_locals[name]) for name in
                ('dropout', 'scaling', 'sliding_window', 'softcap', 'is_causal')}
        elif event == 'return' and result is not None:
            self.retain('attention_output', result[0])  # Native FA before sigmoid gate / o_proj.

    def __enter__(self):
        if sys.getprofile() is not None:
            raise RuntimeError('Refuse to overwrite an existing profiler.')
        def before(_module, args, kwargs):
            self.retain('input', args[0] if args else kwargs['hidden_states'])
        def after(_module, _args, output):
            self.calls['module'] += 1
            self.retain('output', output[0])
        self.handles.append(self.module.register_forward_pre_hook(before, with_kwargs=True))
        self.handles.append(self.module.register_forward_hook(after))
        for name in ('q_proj', 'k_proj', 'v_proj', 'q_norm', 'k_norm', 'o_proj'):
            def observe(_module, args, output, name=name):
                self.retain(name + '_input', args[0])
                self.retain(name + '_output', output)
            self.handles.append(getattr(self.module, name).register_forward_hook(observe))
        sys.setprofile(self.profile)
        return self

    def __exit__(self, *_exc):
        sys.setprofile(None)
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        if self.pinned_host:
            torch.cuda.current_stream().synchronize()

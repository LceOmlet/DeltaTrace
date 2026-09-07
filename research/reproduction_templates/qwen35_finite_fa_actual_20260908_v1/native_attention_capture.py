"""Passive capture of an original attention module and its default FA interface.

No replacement callables or monkey patches. Captures are diagnostics/replay
operands and incur CPU transfers; do not include them in unlabelled warm timing.
"""
import sys


class NativeAttentionCapture:
    def __init__(self, module, attention_interface, native_varlen_function, destination='cpu'):
        self.module = module
        self.interface = attention_interface
        self.native_varlen = native_varlen_function
        self.destination = destination
        self.values = {}
        self.calls = {'module': 0, 'interface': 0, 'native_varlen': 0}
        self.handles = []

    def retain(self, name, value):
        self.values[name] = value.detach().to(self.destination).clone()

    def profile(self, frame, event, result):
        if frame.f_code is self.native_varlen.__code__ and event == 'call':
            self.calls['native_varlen'] += 1
            for name in ('q', 'k', 'v', 'cu_seqlens_q', 'cu_seqlens_k'):
                self.retain('packed_' + name, frame.f_locals[name])
            self.packed_arguments = {name: frame.f_locals[name] for name in
                ('max_seqlen_q', 'max_seqlen_k', 'dropout_p', 'softmax_scale', 'causal', 'return_attn_probs')}
        if frame.f_code is not self.interface.__code__ or frame.f_locals.get('module') is not self.module:
            return
        if event == 'call':
            self.calls['interface'] += 1
            for name in ('query', 'key', 'value', 'attention_mask'):
                value = frame.f_locals[name]
                if value is not None:
                    self.retain(name, value)
            self.interface_arguments = {name: frame.f_locals[name] for name in
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

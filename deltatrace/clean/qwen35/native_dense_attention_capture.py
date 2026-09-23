"""Observe the model's standard dense FA call without replacing any callable.

Equal-length, unpadded inputs use flash_attn_func in the actual Transformers
backend. Extend the existing passive observer only; finite propagation and all
model/FA code stay unchanged. No forced padding or alternate model call is used.
"""
from native_attention_capture import NativeAttentionCapture, _metadata_scalar


class NativeDenseAttentionCapture(NativeAttentionCapture):
    def __init__(self, module, attention_interface, native_varlen_function,
                 native_dense_function, destination='cpu', *, copy_tensors=True, retained_names=None, preserve_strides=False, pinned_host=False):
        super().__init__(module, attention_interface, native_varlen_function, destination,
                         copy_tensors=copy_tensors, retained_names=retained_names, preserve_strides=preserve_strides,pinned_host=pinned_host)
        self.native_dense = native_dense_function
        self.calls['native_dense'] = 0
        self.dense_arguments = None
        self.dense_aliases = {}
        self._interface_views = {}

    @staticmethod
    def _view_identity(value):
        return (value.device, value.dtype, value.data_ptr(), tuple(value.shape), value.stride())

    def retain_dense(self, name, value):
        source = {'q': 'query', 'k': 'key', 'v': 'value'}[name]
        key = 'dense_' + name
        if self.retained_names is not None and key not in self.retained_names:
            return
        # The installed native interface transposes these actual tensors. Only
        # reuse a capture when pointer, dtype, shape and stride verify that view;
        # PEFT casts or other native conversions keep their independent capture.
        if (not self.copy_tensors and self.preserve_strides and source in self.values
                and self._interface_views.get(source) == self._view_identity(value)):
            self.values[key] = self.values[source].transpose(1, 2)
            self.dense_aliases[key] = source
        else:
            self.retain(key, value)

    def profile(self, frame, event, result):
        if (frame.f_code is self.interface.__code__ and event == 'call'
                and frame.f_locals.get('module') is self.module):
            self._interface_views = {name: self._view_identity(frame.f_locals[name].transpose(1, 2))
                                     for name in ('query', 'key', 'value')}
        if frame.f_code is self.native_dense.__code__ and event == 'call':
            self.calls['native_dense'] += 1
            for name in ('q', 'k', 'v'):
                self.retain_dense(name, frame.f_locals[name])
            self.dense_arguments = {name: _metadata_scalar(frame.f_locals[name]) for name in
                                    ('dropout_p', 'softmax_scale', 'causal', 'return_attn_probs')}
            # The current Qwen3.5 path has no windows, softcap or positional bias.
            assert frame.f_locals.get('window_size', (-1, -1)) == (-1, -1)
            assert frame.f_locals.get('softcap', 0.0) == 0.0
            assert frame.f_locals.get('alibi_slopes') is None
        super().profile(frame, event, result)

"""Observe the model's standard dense FA call without replacing any callable.

Equal-length, unpadded inputs use flash_attn_func in the actual Transformers
backend. Extend the existing passive observer only; finite propagation and all
model/FA code stay unchanged. No forced padding or alternate model call is used.
"""
from native_attention_capture import NativeAttentionCapture, _metadata_scalar


class NativeDenseAttentionCapture(NativeAttentionCapture):
    def __init__(self, module, attention_interface, native_varlen_function,
                 native_dense_function, destination='cpu'):
        super().__init__(module, attention_interface, native_varlen_function, destination)
        self.native_dense = native_dense_function
        self.calls['native_dense'] = 0
        self.dense_arguments = None

    def profile(self, frame, event, result):
        if frame.f_code is self.native_dense.__code__ and event == 'call':
            self.calls['native_dense'] += 1
            for name in ('q', 'k', 'v'):
                self.retain('dense_' + name, frame.f_locals[name])
            self.dense_arguments = {name: _metadata_scalar(frame.f_locals[name]) for name in
                                    ('dropout_p', 'softmax_scale', 'causal', 'return_attn_probs')}
            # The current Qwen3.5 path has no windows, softcap or positional bias.
            assert frame.f_locals.get('window_size', (-1, -1)) == (-1, -1)
            assert frame.f_locals.get('softcap', 0.0) == 0.0
            assert frame.f_locals.get('alibi_slopes') is None
        super().profile(frame, event, result)

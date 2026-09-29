"""Test-only passive operand capture around unchanged DT owner calls."""
import importlib
from types import SimpleNamespace
import torch


def cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu(item) for item in value)
    return value


class LayerBoundaryCapture:
    def __init__(self, runner, layer_index, output):
        self.runner, self.index, self.output = runner, layer_index, output
        self.layer = runner.model.model.language_model.layers[layer_index]
        self.data = dict(layer=layer_index, native={}, boundaries={})
        self.active = False
        self.handles = []

    def __enter__(self):
        runner = self.runner
        self.module = importlib.import_module(type(runner).__module__)
        self.original_decoder = self.module.decoder_finite_pullback
        capture = self
        def decoder(layer, values, upstream, *args, **kwargs):
            if layer is not capture.layer:
                return capture.original_decoder(layer, values, upstream, *args, **kwargs)
            capture.data['upstream'] = cpu(upstream)
            capture.active = True
            try:
                result = capture.original_decoder(layer, values, upstream, *args, **kwargs)
            finally:
                capture.active = False
            capture.data['input_coefficients'] = cpu(result[0])
            return result
        self.module.decoder_finite_pullback = decoder
        self.original_ops = {}
        for name in ('mlp', 'norm_residual', 'attention_gate', 'attention_input'):
            original = getattr(runner.boundaries, name)
            self.original_ops[name] = original
            def observe(*args, _name=name, _original=original):
                result = _original(*args)
                if capture.active:
                    # Weights/large MLP intermediates remain owned by DT. Only
                    # coefficients and scalar-rule inputs needed below are kept.
                    selected = args[:2]+args[3:5] if _name == 'norm_residual' else ()
                    capture.data['boundaries'].setdefault(_name, []).append(
                        dict(inputs=cpu(selected), output=cpu(result)))
                return result
            setattr(runner.boundaries, name, observe)
        self.original_fa = runner.finite_fa
        def finite_fa(operands, scale, layout, activity=None):
            result = capture.original_fa(operands, scale, layout, activity)
            if capture.active:
                capture.data['fa'] = dict(operands=cpu(operands), coefficients=cpu(result), scale=scale,
                    lengths=layout.lengths, padded_length=layout.padded_length,
                    coefficient_starts=layout.coefficient_starts, query_start=layout.query_start)
            return result
        runner.finite_fa = finite_fa
        self.original_backend = runner.capture_backend
        backend = runner.capture_backend or self.module
        class AttentionCapture(backend.NativeDenseAttentionCapture):
            def __exit__(self, *args):
                result = super().__exit__(*args)
                if self.module is capture.layer.self_attn:
                    capture.data['attention_values'] = cpu(self.values)
                return result
        runner.capture_backend = SimpleNamespace(NativeDecoderCapture=backend.NativeDecoderCapture,
            NativeGDNCapture=backend.NativeGDNCapture, NativeDenseAttentionCapture=AttentionCapture)
        for name, module in [('input_norm', self.layer.input_layernorm),
                             ('post_norm', self.layer.post_attention_layernorm),
                             ('mlp', self.layer.mlp), ('attention', self.layer.self_attn),
                             ('decoder', self.layer)]:
            def hook(module, args, kwargs, output, _name=name):
                value = output[0] if isinstance(output, tuple) else output
                # Cached root and replay overwrite prefix records with their
                # exact latest native outputs. No forward is called here.
                source = args[0] if args else kwargs['hidden_states']
                capture.data['native'][_name] = dict(input=cpu(source), output=cpu(value))
            self.handles.append(module.register_forward_hook(hook, with_kwargs=True))
        return self

    def __exit__(self, *args):
        for handle in self.handles:
            handle.remove()
        self.module.decoder_finite_pullback = self.original_decoder
        for name, original in self.original_ops.items():
            setattr(self.runner.boundaries, name, original)
        self.runner.finite_fa = self.original_fa
        self.runner.capture_backend = self.original_backend
        if args[0] is None:
            torch.save(self.data, self.output)

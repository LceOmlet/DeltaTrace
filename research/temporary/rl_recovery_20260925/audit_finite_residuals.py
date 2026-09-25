"""Passive contraction ledger for actual native/finite decoder boundaries.

Diagnostic only. Captures original module outputs during the existing replay;
does not implement any forward, finite rule, precision change or acceptance gate.
The owner computes every coefficient. All recorded terms share its operands.
"""
from contextlib import contextmanager
import sys


@contextmanager
def boundary_audit(runner, records, selected=(13, 7, 8)):
    module = sys.modules[type(runner).__module__]
    layers = runner.model.model.language_model.layers
    indices = {id(layer): i for i, layer in enumerate(layers)}
    replay = runner.model.replay_finite_layer
    decoder = module.decoder_finite_pullback
    mlp = runner.boundaries.mlp
    norm = runner.boundaries.norm_residual
    state = {}

    def effect(coeff, endpoints):
        return module._token_effect(coeff, endpoints).sum(1).cpu().tolist()

    def combine(*terms):
        return [sum(row) for row in zip(*terms)]

    def minus(a, b):
        return [x-y for x, y in zip(a, b)]

    def replay_logged(layer, fn):
        index = indices[id(layer)]
        if index not in selected:
            return replay(layer, fn)
        state.clear()
        state.update(layer=index, kind=layer.block_type, caps={}, terms={}, norm_calls=0)
        caps, handles = state['caps'], []

        def hook(name, capture_input=False):
            def capture(_module, args, result):
                if name == 'mixer' and layer.block_type == 'full_attention':
                    # Original Qwen decoder consumes the first item of its
                    # (attention output, attention weights) return value.
                    result = result[0]
                caps[name] = result.detach()
                if capture_input:
                    caps[name+'_input'] = args[0].detach()
            return capture

        mixer = layer.self_attn if layer.block_type == 'full_attention' else layer.linear_attn
        for name, component, inp in (
            ('input_norm', layer.input_layernorm, True),
            ('post_norm', layer.post_attention_layernorm, True),
            ('mixer', mixer, False), ('mlp', layer.mlp, False), ('output', layer, False)):
            handles.append(component.register_forward_hook(hook(name, inp)))
        try:
            return replay(layer, fn)
        finally:
            for handle in handles:
                handle.remove()

    def mlp_logged(*args, **kwargs):
        result = mlp(*args, **kwargs)
        if state:
            c, t = state['caps'], state['terms']
            upstream = args[6]
            t['block_output'] = effect(upstream, c['output'])
            t['post_residual'] = effect(upstream, c['post_norm_input'])
            t['native_mlp'] = effect(upstream, c['mlp'])
            t['finite_mlp'] = effect(result, c['post_norm'])
        return result

    def norm_logged(*args, **kwargs):
        result = norm(*args, **kwargs)
        if state:
            c, t = state['caps'], state['terms']
            state['norm_calls'] += 1
            if state['norm_calls'] == 1:
                t['post_norm_input'] = effect(result, c['post_norm_input'])
                t['input_residual'] = effect(result, c['input_norm_input'])
                t['native_mixer'] = effect(result, c['mixer'])
            else:
                t['finite_mixer'] = effect(args[3], c['input_norm'])
                t['block_input'] = effect(result, c['input_norm_input'])
        return result

    def decoder_logged(layer, *args, **kwargs):
        try:
            result = decoder(layer, *args, **kwargs)
            if indices[id(layer)] in selected:
                t = state['terms']
                deltas = dict(
                    mlp_residual_add=minus(combine(t['post_residual'], t['native_mlp']), t['block_output']),
                    mlp_finite=minus(t['finite_mlp'], t['native_mlp']),
                    post_norm_finite=minus(t['post_norm_input'], combine(t['post_residual'], t['finite_mlp'])),
                    mixer_residual_add=minus(combine(t['input_residual'], t['native_mixer']), t['post_norm_input']),
                    mixer_finite=minus(t['finite_mixer'], t['native_mixer']),
                    input_norm_finite=minus(t['block_input'], combine(t['input_residual'], t['finite_mixer'])))
                records.append(dict(layer=state['layer'], kind=state['kind'], terms=t,
                                    deltas=deltas, total=minus(t['block_input'], t['block_output']),
                                    deltas_sum=combine(*deltas.values())))
            return result
        finally:
            state.clear()

    runner.model.replay_finite_layer = replay_logged
    module.decoder_finite_pullback = decoder_logged
    runner.boundaries.mlp = mlp_logged
    runner.boundaries.norm_residual = norm_logged
    try:
        yield
    finally:
        runner.model.replay_finite_layer = replay
        module.decoder_finite_pullback = decoder
        runner.boundaries.mlp = mlp
        runner.boundaries.norm_residual = norm

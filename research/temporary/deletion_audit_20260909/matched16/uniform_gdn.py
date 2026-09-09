"""Explicit experiment on finite allocation; imports the frozen FLA operator.

This is a symmetric endpoint average of the entire FLA finite pullback, not
native backward and not a claim of a Shapley split at every recurrent product.
Both real native endpoint captures are reused. No model, FLA stage or forward
is replaced. The extra finite/native-adjoint work is deliberately measured.
"""


class EndpointSymmetricFLA:
    def __init__(self, finite_pullback):
        self.finite_pullback = finite_pullback
        self.calls = 0

    def __call__(self, endpoints, upstream, scale):
        reverse = {}
        batch = upstream.shape[0]
        for name, value in endpoints.items():
            if value.shape[0] != 2 * batch:
                raise ValueError('Expected actual interleaved paired endpoints: ' + name)
            reverse[name] = value.reshape(batch, 2, *value.shape[1:]).flip(1).reshape_as(value)
        first = self.finite_pullback(endpoints, upstream, scale)
        second = self.finite_pullback(reverse, upstream, scale)
        self.calls += 2
        assert first.keys() == second.keys()
        # No sign change: L(x1,x0)(x0-x1)=y0-y1 also gives
        # L(x1,x0)(x1-x0)=y1-y0. Average finite coefficients, not scores.
        return {name: (first[name] + second[name]) * 0.5 for name in first}


def configure_uniform_gdn(runner, mode):
    """Same choice on every GDN; no layer-, sample-, token- or task-based rule."""
    if mode not in ('clean', 'norm_symmetric', 'fla_symmetric', 'both_symmetric'):
        raise ValueError(mode)
    layers = runner.model.model.language_model.layers
    gdn = [i for i, layer in enumerate(layers) if layer.block_type == 'linear_attention']
    assert len(gdn) == 24
    if mode in ('norm_symmetric', 'both_symmetric'):
        runner.norm_gate_rules = dict.fromkeys(gdn, 'symmetric')
    if mode in ('fla_symmetric', 'both_symmetric'):
        wrapper = EndpointSymmetricFLA(runner.finite_fla)
        runner.finite_fla_by_layer = dict.fromkeys(gdn, wrapper)
    assert not runner.attention_pv_rules and not runner.key_norm_by_layer
    return runner

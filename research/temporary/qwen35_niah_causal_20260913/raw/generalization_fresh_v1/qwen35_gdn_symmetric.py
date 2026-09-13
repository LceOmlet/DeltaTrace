"""Optional uniform symmetric GDN attribution for the Qwen3.5 finite runner.

This changes attribution allocation at every GDN layer. The model's forward,
response target, reference input, attention rules and clean default are retained.
The memory callback runs twice per GDN layer; latency must be measured separately.
"""


def average_memory_endpoint_orders(pullback):
    """Average finite coefficients from the two local endpoint orientations.

    ``endpoints`` contains native tensors in interleaved reference/input rows.
    The incoming coefficient has one row per pair. Reversing every captured
    endpoint tensor reverses both sides of the finite identity, so the resulting
    coefficients are averaged with a plus sign. Rounding residuals remain part
    of the runner's diagnostics.
    """
    if not callable(pullback):
        raise TypeError('The finite memory pullback must be callable.')

    def averaged(endpoints, upstream, scale):
        import torch

        count = endpoints['q'].shape[0]
        if count == 0 or count % 2:
            raise ValueError('Memory endpoints must contain complete reference/input pairs.')
        permutation = torch.arange(count, device=endpoints['q'].device).reshape(-1, 2).flip(1).flatten()
        reversed_endpoints = {}
        for name, value in endpoints.items():
            if not torch.is_tensor(value) or value.ndim == 0 or value.shape[0] != count:
                raise ValueError(f'Captured memory tensor {name!r} does not match the paired batch.')
            reversed_endpoints[name] = value.index_select(0, permutation)
        forward = pullback(endpoints, upstream, scale)
        reverse = pullback(reversed_endpoints, upstream, scale)
        if forward.keys() != reverse.keys():
            raise ValueError('The two finite memory calls returned different coefficient fields.')
        return {name: (forward[name] + reverse[name]) * 0.5 for name in forward}

    return averaged


def make_qwen35_gdn_symmetric_runner(model, finite_fa, finite_fla, **runner_options):
    """Use symmetric output products and memory orientation at every GDN layer.

    Load the established ``deltatrace/clean/qwen35`` runtime on ``sys.path`` before
    calling, as for the clean runner. Optional execution arguments are forwarded
    to the installed runner (for example, dynamic compiler settings in the
    validated execution overlay). Layer-specific rule overrides are disallowed.
    """
    from qwen35_dense_finite_runner import Qwen35DenseFiniteRunner

    rules = {'norm_gate_rules', 'finite_fla_by_layer', 'attention_pv_rules', 'key_norm_by_layer'}
    if rules.intersection(runner_options):
        raise ValueError('This profile applies one fixed rule to every GDN layer.')
    indices = [i for i, layer in enumerate(model.model.language_model.layers)
               if layer.block_type == 'linear_attention']
    if not indices:
        raise ValueError('This profile requires a model containing GDN layers.')
    callback = average_memory_endpoint_orders(finite_fla)
    return Qwen35DenseFiniteRunner(
        model, finite_fa, finite_fla,
        norm_gate_rules={i: 'symmetric' for i in indices},
        finite_fla_by_layer={i: callback for i in indices},
        **runner_options,
    )

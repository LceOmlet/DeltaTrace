"""Versioned official attribution entry points; clean-v1 remains reproducible."""

QWEN35_DEFAULT = 'gdn-symmetric-v1'
QWEN35_PROFILES = (QWEN35_DEFAULT, 'clean-v1')


def make_qwen35_runner(model, finite_fa, finite_fla, *, profile=QWEN35_DEFAULT, **runner_options):
    if profile == QWEN35_DEFAULT:
        from .qwen35_gdn_symmetric import make_qwen35_gdn_symmetric_runner
        runner = make_qwen35_gdn_symmetric_runner(model, finite_fa, finite_fla, **runner_options)
        indices = {i for i, layer in enumerate(model.model.language_model.layers)
                   if layer.block_type == 'linear_attention'}
        assert runner.norm_gate_rules == {i: 'symmetric' for i in indices}
        assert set(runner.finite_fla_by_layer) == indices
        assert all(callable(p) for p in runner.finite_fla_by_layer.values())
    elif profile == 'clean-v1':
        if runner_options:
            raise ValueError('The frozen clean-v1 factory does not accept execution overrides.')
        from qwen35_clean_runner import make_qwen35_clean_runner
        runner = make_qwen35_clean_runner(model, finite_fa, finite_fla)
        assert runner.norm_gate_rules == runner.finite_fla_by_layer == {}
    else:
        raise ValueError(f'Unknown Qwen3.5 profile: {profile}')
    assert runner.attention_pv_rules == runner.key_norm_by_layer == {}
    return runner

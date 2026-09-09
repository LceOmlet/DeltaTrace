"""Uniform key-normalization experiment via the frozen DT callback interface."""
from normalization_geometry import l2_geometry_pullback

def configure_uniform_gdn(runner, mode):
    if mode not in ('clean','key_geometry'):raise ValueError(mode)
    layers=runner.model.model.language_model.layers
    gdn=[i for i,layer in enumerate(layers) if layer.block_type=='linear_attention']
    assert len(gdn)==24
    if mode=='key_geometry':runner.key_norm_by_layer=dict.fromkeys(gdn,l2_geometry_pullback)
    assert not runner.norm_gate_rules and not runner.finite_fla_by_layer and not runner.attention_pv_rules
    return runner

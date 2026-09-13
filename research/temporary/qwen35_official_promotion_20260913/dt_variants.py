"""Uniform research variants; original native model and clean default remain intact."""
def average_memory_endpoint_orders(pullback):
    """Average both local endpoint orders, with the same incoming coefficient.

    Each ordering accounts for the same finite difference after reversing its
    orientation. Averaging coefficients preserves the local finite identity in
    exact arithmetic. All native rounding residuals must still be measured.
    """
    def averaged(endpoints,upstream,scale):
        import torch
        count=endpoints['q'].shape[0]
        assert count%2==0
        permutation=torch.arange(count,device=endpoints['q'].device).reshape(-1,2).flip(1).flatten()
        reversed_endpoints={}
        for name,value in endpoints.items():
            assert torch.is_tensor(value) and value.shape[0]==count,(name,value.shape)
            reversed_endpoints[name]=value.index_select(0,permutation)
        forward=pullback(endpoints,upstream,scale)
        reverse=pullback(reversed_endpoints,upstream,scale)
        assert forward.keys()==reverse.keys()
        return {name:(forward[name]+reverse[name])*0.5 for name in forward}
    return averaged

def build_uniform_runners(model,finite_fa,finite_fla,compiler_options):
    from qwen35_dense_finite_runner import Qwen35DenseFiniteRunner
    indices=[i for i,layer in enumerate(model.model.language_model.layers) if layer.block_type=='linear_attention']
    gate={i:'symmetric' for i in indices}
    memory={i:average_memory_endpoint_orders(finite_fla) for i in indices}
    common=dict(dynamic_shapes=True,compiler_options=compiler_options)
    return {
        'DT_original':Qwen35DenseFiniteRunner(model,finite_fa,finite_fla,**common),
        'DT_gate_symmetric':Qwen35DenseFiniteRunner(model,finite_fa,finite_fla,norm_gate_rules=gate,**common),
        'DT_memory_symmetric':Qwen35DenseFiniteRunner(model,finite_fa,finite_fla,finite_fla_by_layer=memory,**common),
        'DT_gdn_symmetric':Qwen35DenseFiniteRunner(model,finite_fa,finite_fla,norm_gate_rules=gate,finite_fla_by_layer=memory,**common),
    }

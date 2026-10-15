"""Observe the upstream attention that supplies the identified negative route.

Use unchanged production caches and original finite FA. Store actual returned
coefficients and its existing operands/normalizers; do not substitute ordinary
attention gradients or a single-deletion estimand.
"""
from contextlib import contextmanager
import inspect,sys
import torch
from observe_existing_token_transport_v1 import observe_transport as transport_v1


@contextmanager
def observe_transport(runner,focus_row,focus_position):
    rm=sys.modules[type(runner).__module__]
    original_attention=rm.attention_finite_pullback
    details=dict(layer=31,attention_calls=0,positions=[focus_position,focus_position+1])
    attribute_code=inspect.unwrap(type(runner).attribute).__code__
    def loc():
        frame=sys._getframe(1)
        try:
            while frame is not None and frame.f_code is not attribute_code:frame=frame.f_back
            if frame is None:return None
            return {k:frame.f_locals.get(k) for k in ['i','row_starts','prefix_start','m','dc']}
        finally:del frame
    def cpu(x):return x.detach().cpu().clone()
    def native_norm(_module,_args,output):
        f=loc()
        if f is None or f['i']!=31 or f['m'] is None:return
        start=f['row_starts'][focus_row] if f['row_starts'] is not None else (f['prefix_start'] or 0)
        indices=torch.tensor([focus_position-start,focus_position+1-start],device=output.device)
        details['native_input_norm_pair']=cpu(output[2*focus_row:2*focus_row+2].index_select(1,indices))
    handle=runner.model.model.language_model.layers[31].input_layernorm.register_forward_hook(native_norm)

    def attention(module,values,lse,cos,sin,upstream,finite_fa,layout,boundaries,*args,**kwargs):
        f=loc()
        if f is None or f['i']!=31:
            return original_attention(module,values,lse,cos,sin,upstream,finite_fa,layout,boundaries,*args,**kwargs)
        start=f['row_starts'][focus_row] if f['row_starts'] is not None else (f['prefix_start'] or 0)
        indices=torch.tensor([focus_position-start,focus_position+1-start],device=upstream.device)
        key_indices=torch.tensor([focus_position,focus_position+1],device=upstream.device)
        details.update(attention_calls=details['attention_calls']+1,native_row_start=start,
            q_projection_pair=cpu(values['q_proj_output'][2*focus_row:2*focus_row+2].index_select(1,indices)),
            v_pair=cpu(values['value'][2*focus_row:2*focus_row+2].index_select(2,key_indices)),
            q_norm_weight=cpu(module.q_norm.weight),k_norm_weight=cpu(module.k_norm.weight),
            eps=module.q_norm.eps,groups=module.num_key_value_groups,scale=module.scaling,
            pv_rule=kwargs.get('pv_rule','content1'),
            upstream_at_positions=cpu(upstream[focus_row].index_select(0,indices)))
        original_input=boundaries.attention_input
        def input_rule(*a,**kw):
            result=original_input(*a,**kw)
            details['attention_input_arguments']=dict(
                dq=cpu(a[0][focus_row].index_select(1,indices)),
                dk=cpu(a[1][focus_row].index_select(1,indices)),
                dv=cpu(a[2][focus_row].index_select(1,indices)),
                mgate=cpu(a[3][focus_row].index_select(0,indices)),
                q0=cpu(a[4][focus_row].index_select(0,indices)),q1=cpu(a[5][focus_row].index_select(0,indices)),
                k0=cpu(a[6][focus_row].index_select(0,indices)),k1=cpu(a[7][focus_row].index_select(0,indices)),
                cos=cpu(a[10][focus_row].index_select(0,indices)),sin=cpu(a[11][focus_row].index_select(0,indices)),
                result=cpu(result[focus_row].index_select(0,indices)))
            return result
        boundaries.attention_input=input_rule
        def fa(ops,scale,owner_layout,activity=None):
            result=finite_fa(ops,scale,owner_layout,activity)
            if torch.distributed.get_rank()==0:
                details['native_FA_operands']={k:cpu(v[focus_row:focus_row+1]) for k,v in ops.items()}
                details['native_FA_outputs']={k:cpu(v[focus_row:focus_row+1]) for k,v in result.items()}
            details['FA_layout']=dict(type=type(owner_layout).__module__+'.'+type(owner_layout).__name__,
                values={k:cpu(v) if isinstance(v,torch.Tensor) else v for k,v in vars(owner_layout).items()})
            return result
        try:
            result=original_attention(module,values,lse,cos,sin,upstream,fa,layout,boundaries,*args,**kwargs)
            details['attention_return_at_positions']=cpu(result[0][focus_row].index_select(0,indices))
            return result
        finally:boundaries.attention_input=original_input

    rm.attention_finite_pullback=attention
    try:
        with transport_v1(runner,focus_row,focus_position) as (metadata,payload):
            metadata['layer31_payload']=details
            yield metadata,payload
        assert details['attention_calls']==1
    finally:
        rm.attention_finite_pullback=original_attention
        handle.remove()

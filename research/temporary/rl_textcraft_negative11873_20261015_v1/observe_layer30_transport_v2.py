"""Observe the identified layer30 mechanism without replacing an owner rule.

Every callback returns its original result. attribute(observer=None), production
capture release, native dtype, head grouping and coefficient ranges remain.
Only the saved incident trajectory is copied; full FLA operands on rank0 allow
subsequent operator-level checks without repeating the transformer.
"""
from contextlib import contextmanager
import inspect,sys
import torch
from observe_existing_token_transport_v1 import observe_transport as transport_v1


@contextmanager
def observe_transport(runner,focus_row,focus_position):
    rm=sys.modules[type(runner).__module__]
    import qwen35_gdn_finite as gm
    original_decoder=rm.decoder_finite_pullback
    original_gdn=rm.gdn_finite_pullback
    original_linear=gm._linear_transpose
    original_l2=gm._l2_pullback
    details=dict(layer=30,decoder_calls=0,gdn_calls=0,linear_branches=[],fla_groups=[],norm_boundaries=[],l2_boundaries=[])
    active=dict(layer=False,gdn=False,linear_count=0)
    attribute_code=inspect.unwrap(type(runner).attribute).__code__

    def attribute_locals():
        frame=sys._getframe(1)
        try:
            while frame is not None and frame.f_code is not attribute_code:frame=frame.f_back
            return None if frame is None else dict(frame.f_locals)
        finally:del frame

    def cpu(x):return x.detach().cpu().clone()
    def paired_focus(x,index,time_dimension=1):
        selected=x[2*focus_row:2*focus_row+2]
        return cpu(selected.select(time_dimension,index))
    def coefficient_focus(x,index,time_dimension=1):
        return cpu(x[focus_row].select(time_dimension-1,index))

    def native_norm(_module,_args,output):
        local=attribute_locals()
        if local is None or local.get('i')!=30:return
        starts=local.get('row_starts');start=starts[focus_row] if starts is not None else local.get('prefix_start',0)
        details['native_input_norm_pair']=paired_focus(output,focus_position-start)
    norm_handle=runner.model.model.language_model.layers[30].input_layernorm.register_forward_hook(native_norm)

    def decoder(layer,values,upstream,mixer,boundaries,*args,**kwargs):
        local=attribute_locals()
        if local is None or local.get('i')!=30:
            return original_decoder(layer,values,upstream,mixer,boundaries,*args,**kwargs)
        starts=local.get('row_starts');start=starts[focus_row] if starts is not None else local.get('prefix_start',0)
        index=focus_position-start
        details.update(decoder_calls=details['decoder_calls']+1,original_position=focus_position,
            native_row_start=start,native_focus_index=index,selected_length=upstream.shape[1],
            input_pair=paired_focus(values['input_norm_input'],index),
            output_coefficient=coefficient_focus(upstream,index))
        active.update(layer=True,index=index)
        old_norm=boundaries.norm_residual
        def norm(*a,**kw):
            result=old_norm(*a,**kw)
            details['norm_boundaries'].append(dict(
                input0=cpu(a[0][focus_row,index]),input1=cpu(a[1][focus_row,index]),
                upstream=cpu(a[3][focus_row,index]),residual=cpu(a[4][focus_row,index]),
                result=cpu(result[focus_row,index])))
            return result
        boundaries.norm_residual=norm
        try:
            result=original_decoder(layer,values,upstream,mixer,boundaries,*args,**kwargs)
            details['input_coefficient']=coefficient_focus(result[0],index)
            return result
        finally:
            boundaries.norm_residual=old_norm
            active['layer']=False

    def gdn(module,values,endpoints,upstream,scale,fla_pullback,*args,**kwargs):
        if not active['layer']:
            return original_gdn(module,values,endpoints,upstream,scale,fla_pullback,*args,**kwargs)
        capture_start=kwargs.get('capture_start',0)
        index=active['index']-capture_start
        projected_extra=values['projected_qkv'].shape[-1]-(upstream.shape[1]-capture_start)
        details.update(gdn_calls=details['gdn_calls']+1,capture_start=capture_start,
            fla_coefficient_start=kwargs.get('fla_coefficient_start',0),
            native_scale=scale,gdn_focus_index=index,
            norm_gate_rule=kwargs.get('norm_gate_rule','content1'),
            upstream_focus=coefficient_focus(upstream,active['index']),
            captured_a=paired_focus(values['a'],index),captured_b=paired_focus(values['b'],index),
            captured_z=paired_focus(values['z'],index),
            projected_qkv_pair=paired_focus(values['projected_qkv'],index+projected_extra,2),
            projected_left_context=projected_extra,
            conv_width=module.conv1d.weight.shape[-1],num_v_heads=module.num_v_heads,
            num_k_heads=module.num_k_heads,head_k_dim=module.head_k_dim,head_v_dim=module.head_v_dim)
        active.update(gdn=True,linear_count=0)
        original_gate=kwargs.get('norm_gate_pullback') or gm._norm_gate_finite_rule
        def gate(*a,**kw):
            result=original_gate(*a,**kw)
            details['norm_gate']=dict(o_pair=paired_focus(a[0],index),z_pair=paired_focus(a[1],index),
                m=coefficient_focus(a[2],index),weight=cpu(a[3]),eps=a[4],rule=a[5],
                mo_scale=a[6] if len(a)>6 else kw.get('mo_scale',1.0),
                mo=coefficient_focus(result[0],index),mz=coefficient_focus(result[1],index))
            return result
        kwargs['norm_gate_pullback']=gate
        group_head_start=0
        def fla(e,do,actual_scale):
            nonlocal group_head_start
            result=fla_pullback(e,do,actual_scale)
            heads=e['q'].shape[2]
            row_start=details['native_row_start']+capture_start
            cut=(kwargs.get('fla_coefficient_start',0)-capture_start)//64*64
            local_index=focus_position-row_start-cut
            # Only the original focus trajectory; no reshaped/rerun model.
            group=dict(head_start=group_head_start,head_count=heads,scale=actual_scale,
                original_time_start=row_start+cut,focus_index=local_index,
                endpoint_dtypes={k:str(v.dtype) for k,v in e.items()},
                coefficients_focus={k:coefficient_focus(v,local_index) for k,v in result.items()},
                endpoints_focus={k:paired_focus(v,local_index) for k,v in e.items() if k not in ('h','A')})
            caller=sys._getframe(1)
            try:
                exponent=caller.f_locals.get('seed_exponent')
                group['seed_exponent']=None if exponent is None else cpu(exponent[focus_row:focus_row+1,:,group_head_start:group_head_start+heads])
            finally:del caller
            if torch.distributed.get_rank()==0:
                group['native_endpoints']={k:cpu(v[2*focus_row:2*focus_row+2]) for k,v in e.items()}
                group['native_do']=cpu(do[focus_row:focus_row+1])
                group['native_coefficients']={k:cpu(v[focus_row:focus_row+1]) for k,v in result.items()}
            details['fla_groups'].append(group);group_head_start+=heads
            return result
        try:
            result=original_gdn(module,values,endpoints,upstream,scale,fla,*args,**kwargs)
            details['gdn_return_focus']=coefficient_focus(result[0],active['index'])
            return result
        finally:active['gdn']=False

    def linear(upstream,weight):
        result=original_linear(upstream,weight)
        if active['gdn']:
            order=['out_proj','in_proj_qkv','in_proj_z','in_proj_b','in_proj_a']
            number=active['linear_count'];assert number<len(order)
            index=details['gdn_focus_index']
            details['linear_branches'].append(dict(name=order[number],
                upstream=coefficient_focus(upstream,index),result=coefficient_focus(result,index)))
            active['linear_count']+=1
        return result

    def l2(x0,x1,upstream):
        result=original_l2(x0,x1,upstream)
        if active['gdn']:
            index=details['gdn_focus_index'];stop=index+details['conv_width']
            details['l2_boundaries'].append(dict(
                input0=cpu(x0[focus_row,index:stop]),input1=cpu(x1[focus_row,index:stop]),
                upstream=cpu(upstream[focus_row,index:stop]),result=cpu(result[focus_row,index:stop])))
        return result

    rm.decoder_finite_pullback=decoder;rm.gdn_finite_pullback=gdn;gm._linear_transpose=linear;gm._l2_pullback=l2
    try:
        with transport_v1(runner,focus_row,focus_position) as (metadata,payload):
            metadata['layer30_payload']=details
            yield metadata,payload
        assert details['decoder_calls']==1 and details['gdn_calls']==1
        assert [x['name'] for x in details['linear_branches']]==['out_proj','in_proj_qkv','in_proj_z','in_proj_b','in_proj_a']
    finally:
        rm.decoder_finite_pullback=original_decoder
        rm.gdn_finite_pullback=original_gdn
        gm._linear_transpose=original_linear
        gm._l2_pullback=original_l2
        norm_handle.remove()

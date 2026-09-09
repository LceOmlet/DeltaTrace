"""Public default-FA endpoint metadata capture candidate, not a finite FA kernel.

One additional unmodified public FA invocation per replay obtains documented LSE.
No private FA return slots, model/attention forward replacement or custom kernels.
Unsupported public wrappers/metadata allocations fail explicitly.
"""
class NativeLayerReplay:
    def __init__(self,model,checkpoint,reference=None,activity=None,validation=None):
        assert validation is not None
        self.validation=validation
        self.activity=activity if activity is not None else {'auxiliary_attempts':0,'auxiliary_completed':0,'metadata':[]}
        self.model=model;self.checkpoint=checkpoint;self.reference=reference
        self.auxiliary_attention_calls=0;self.public_capture_checks=[]
        self.calls=0;self.checks=[];self.boundary_checks=[];self.last_index=None;self.last_values=None
    def __getitem__(self,index):
        import torch
        if index==self.last_index:
            return self.last_values
        layer=self.model.model.layers[index];values={};handles=[]
        def recorder(in_name=None,out_name=None):
            def record(_module,args,output):
                if in_name is not None:
                    values[in_name]=args[0].detach().clone(memory_format=torch.preserve_format)
                if out_name is not None:
                    values[out_name]=output.detach().clone(memory_format=torch.preserve_format)
            return record
        specs=[(layer.input_layernorm,'x','a'),(layer.self_attn.q_norm,'qpre','qnorm'),
            (layer.self_attn.k_norm,'kpre','knorm'),(layer.self_attn.v_proj,None,'v'),
            (layer.self_attn.o_proj,'concat','attn_out'),(layer.post_attention_layernorm,'mid','mlp_in'),
            (layer.mlp.gate_proj,None,'gate'),(layer.mlp.up_proj,None,'up'),
            (layer.mlp.down_proj,'product','mlp_out'),(layer,None,'out')]
        for module,in_name,out_name in specs:
            handles.append(module.register_forward_hook(recorder(in_name,out_name)))
        def attention_record(_module,_args,_kwargs,output):
            assert output[1] is None
            values['p']=None
        handles.append(layer.self_attn.register_forward_hook(attention_record,with_kwargs=True))
        import inspect,sys
        from flash_attn import flash_attn_func
        # Observe the public Python callable without replacing it or model code.
        # Unsupported wrappers fail explicitly; no private tuple fallback.
        assert hasattr(flash_attn_func,'__code__'), 'Public FA observer requires a Python callable'
        public_code=flash_attn_func.__code__
        signature=inspect.signature(flash_attn_func);observed=[];actual_arguments={}
        assert 'return_attn_probs' in signature.parameters
        def observer(frame,event,result):
            if event=='return' and frame.f_code is public_code:
                assert isinstance(result,torch.Tensor), 'Expected default public FA output tensor'
                local=frame.f_locals
                for name,parameter in signature.parameters.items():
                    assert parameter.kind not in (parameter.VAR_POSITIONAL,parameter.VAR_KEYWORD)
                    assert name in local
                    actual_arguments[name]=local[name]
                assert actual_arguments['dropout_p']==0 and actual_arguments['causal']
                assert actual_arguments['window_size']==(-1,-1)
                for name in ['attn_mask','alibi_slopes','s_aux']:
                    assert actual_arguments.get(name) is None
                assert actual_arguments.get('softcap',0)==0
                assert not actual_arguments['return_attn_probs'] and not actual_arguments.get('return_max_logit',False)
                for name in ['q','k','v']:
                    values['fa_'+name]=local[name].detach().clone(memory_format=torch.preserve_format)
                values['fa_out']=result.detach().clone(memory_format=torch.preserve_format)
                observed.append(True)
        assert sys.getprofile() is None
        try:
            with torch.no_grad():
                self.calls+=1
                sys.setprofile(observer)
                layer(self.checkpoint['layers'][index]['x'],**self.checkpoint['forward_kwargs'])
                assert len(observed)==1
        finally:
            sys.setprofile(None)
            for handle in handles:
                handle.remove()
        # Documented public result: output, log-sum-exp, testing-only S_dmask.
        # Only LSE is used; S_dmask is never decoded as a probability matrix.
        actual_arguments['return_attn_probs']=True
        with torch.no_grad():
            self.auxiliary_attention_calls+=1
            self.activity['auxiliary_attempts']+=1
            auxiliary=flash_attn_func(**actual_arguments)
            self.activity['auxiliary_completed']+=1
        assert isinstance(auxiliary,tuple) and len(auxiliary)==3
        auxiliary_out,lse,testing_matrix=auxiliary
        descriptor={'type':type(testing_matrix).__name__,
            'shape':list(testing_matrix.shape) if isinstance(testing_matrix,torch.Tensor) else None,
            'numel':testing_matrix.numel() if isinstance(testing_matrix,torch.Tensor) else None}
        self.activity['metadata'].append({'layer':index,'testing_return':descriptor})
        assert testing_matrix is None or (isinstance(testing_matrix,torch.Tensor) and testing_matrix.numel()==0), descriptor
        testing_numel=0 if testing_matrix is None else testing_matrix.numel()
        assert lse.shape==(values['fa_q'].shape[0],values['fa_q'].shape[2],values['fa_q'].shape[1])
        assert lse.dtype==torch.float32
        self.validation.finite(lse,f'layer{index}.public_lse')
        output_exact=self.validation.equal(auxiliary_out,values['fa_out'],f'layer{index}.public_output')
        values['fa_lse']=lse.detach()
        self.public_capture_checks.append({'layer':index,'public_output_exact_to_actual_model_FA':output_exact,
            'testing_matrix_numel':testing_numel,'testing_return':descriptor,'auxiliary_batch_size':values['fa_q'].shape[0],
            'lse_shape':list(lse.shape),'lse_dtype':str(lse.dtype),
            'public_function_module':flash_attn_func.__module__,'private_FA_slots_read':False,
            'model_output_replaced':False,'auxiliary_public_FA_calls':1})
        del auxiliary,auxiliary_out,testing_matrix,actual_arguments
        expected_output=(self.checkpoint['layers'][index+1]['x']
            if index+1<len(self.model.model.layers) else self.checkpoint['last'])
        check={'layer':index,'native_input_exact':self.validation.equal(values['x'],self.checkpoint['layers'][index]['x'],f'layer{index}.replay_input'),
            'native_output_exact':self.validation.equal(values['out'],expected_output,f'layer{index}.replay_output')}
        self.boundary_checks.append(check)
        # All predicates are checked before the public DT API returns.
        if self.reference is not None:
            expected=self.reference['layers'][index]
            assert set(values)==set(expected)
            checks={key:{'values_exact':bool(torch.equal(value,expected[key])),
                'strides_exact':value.stride()==expected[key].stride()} for key,value in values.items()}
            self.checks.append({'layer':index,'tensors':checks})
            assert all(x['values_exact'] for x in checks.values()),f'Native layer replay mismatch at layer{index}'
        self.last_index=index;self.last_values=values
        return values
    def clear(self):
        self.last_values=None;self.last_index=None

"""Build a separate public-API capture candidate; preserve frozen P1 sources."""
import ast
from pathlib import Path
A=Path(__file__).resolve().parent
old=(A/'qwen_signed_secant_checkpointed_compiled_difference.py').read_text()
tree=ast.parse(old);node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='NativeLayerReplay')
s=ast.get_source_segment(old,node)
s=s.replace('self.calls=0;self.checks=[];', 'self.auxiliary_attention_calls=0;self.public_capture_checks=[]\n        self.calls=0;self.checks=[];',1)
start=s.index('        import sys\n');end=s.index('        expected_output=',start)
s=s[:start]+'''        import inspect,sys
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
            auxiliary=flash_attn_func(**actual_arguments)
        assert isinstance(auxiliary,tuple) and len(auxiliary)==3
        auxiliary_out,lse,testing_matrix=auxiliary
        assert isinstance(testing_matrix,torch.Tensor) and testing_matrix.numel()==0, 'Public FA unexpectedly allocated the testing matrix'
        assert lse.shape==(values['fa_q'].shape[0],values['fa_q'].shape[2],values['fa_q'].shape[1])
        assert lse.dtype==torch.float32 and torch.isfinite(lse).all()
        output_exact=bool(torch.equal(auxiliary_out,values['fa_out']))
        assert output_exact, 'Public metadata invocation changed endpoint attention output'
        values['fa_lse']=lse.detach()
        self.public_capture_checks.append({'layer':index,'public_output_exact_to_actual_model_FA':output_exact,
            'testing_matrix_numel':testing_matrix.numel(),'auxiliary_batch_size':values['fa_q'].shape[0],
            'lse_shape':list(lse.shape),'lse_dtype':str(lse.dtype),
            'public_function_module':flash_attn_func.__module__,'private_FA_slots_read':False,
            'model_output_replaced':False,'auxiliary_public_FA_calls':1})
        del auxiliary,auxiliary_out,testing_matrix,actual_arguments
''' + s[end:]
doc='"""Public default-FA endpoint metadata capture candidate, not a finite FA kernel.\n\nOne additional unmodified public FA invocation per replay obtains documented LSE.\nNo private FA return slots, model/attention forward replacement or custom kernels.\nUnsupported public wrappers/metadata allocations fail explicitly.\n"""\n'
ast.parse(doc+s)
(A/'qwen_public_fa_layer_replay.py').write_text(doc+s+'\n',encoding='utf-8')
paired=(A/'qwen_signed_secant_native_paired_pv_rules.py').read_text()
paired=paired.replace('from qwen_signed_secant_checkpointed_compiled_difference import NativeLayerReplay','from qwen_public_fa_layer_replay import NativeLayerReplay')
needle="        result['native_layer_replay_calls']=paired.native.calls"
paired=paired.replace(needle,"""        assert paired.native.auxiliary_attention_calls==paired.native.calls
        result['extra_native_fa_attention_calls']=paired.native.auxiliary_attention_calls
        result['extra_native_fa_attention_endpoint_trajectories']=2*paired.native.auxiliary_attention_calls
        result['public_FA_capture_checks']=paired.native.public_capture_checks
        # Correct inherited capture-provenance descriptions, without changing arithmetic.
        for audit in result['native_fa_operand_audits']:
            audit['extra_native_attention_calls']=1
            audit['probability_source']='Explicit FP32 exp(QK*scale minus documented public auxiliary FA LSE), causal masked; attribution operand only. Auxiliary output exactly checked against actual model FA output.'
            audit['extra_call_count_scope']='One shared B2 call per layer; the two endpoint audit rows describe that same call, not two physical calls.'
        result['partial_recomputation']='Unchanged frozen P1 arithmetic on actual endpoint tensors; public LSE metadata requires36 extra unmodified FA calls, charged separately. No model or native backward replacement.'
"""+needle)
paired=paired.replace('Zero auxiliary FA, no surrogate/native-autograd replacement.', '36 additional original public FA batch=2 metadata calls, documented LSE only, testing matrix empty. No surrogate/native-autograd replacement.')
ast.parse(paired)
(A/'qwen_signed_secant_paired_public_fa.py').write_text(paired,encoding='utf-8')
print('Built separate public FA replay candidate; frozen P1 arithmetic untouched.')

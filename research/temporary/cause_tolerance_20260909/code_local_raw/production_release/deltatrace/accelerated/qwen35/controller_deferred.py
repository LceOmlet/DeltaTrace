"""Current DT finite propagation through the real Qwen3.5 dense FA/FLA model.

Native target rows are the default; full output rows remain an explicit option.
Shared controller for full versus native-selected output rows. All finite
mathematics is imported unchanged. Experimental acceleration branch: preserve
native right-padding/varlen FA and optionally retain checkpoints on the GPU.
The frozen clean-v1 tag remains the validated unpadded baseline.
"""
import time
from collections.abc import Mapping
import torch
from flash_attn import flash_attn_func,flash_attn_varlen_func
from flash_attn.bert_padding import pad_input
from transformers.integrations.flash_attention import flash_attention_forward
from qwen35_answer_finite import FiniteAnswerOps
from qwen35_decoder_finite import NativeDecoderCapture,FiniteBoundaryOps,attention_finite_pullback,decoder_finite_pullback
from qwen35_gdn_finite import NativeGDNCapture,gdn_finite_pullback
from native_dense_attention_capture import NativeDenseAttentionCapture
from native_target_logit_rows import NativeTargetLogitRows
from vendor_fa_finite_bf16_d256 import RightPaddedLengths


def _copy(value,device):
    if isinstance(value,torch.Tensor):return value.detach().to(device,copy=True)
    if isinstance(value,tuple):return tuple(_copy(v,device) for v in value)
    if isinstance(value,list):return [_copy(v,device) for v in value]
    if isinstance(value,dict):return {k:_copy(v,device) for k,v in value.items()}
    if value is not None and not isinstance(value,(str,int,float,bool)):raise TypeError(type(value))
    return value


def _effect(m,x):return (m.double()*(x[1::2].double()-x[0::2].double())).sum()


class Qwen35DenseFiniteRunner:
    def __init__(self,model,finite_fa,finite_fla,*,norm_gate_rules=None,finite_fla_by_layer=None,attention_pv_rules=None,key_norm_by_layer=None,checkpoint_device='cpu'):
        """Optional GDN layer-index rules; unspecified layers retain content1.

        The layer0 symmetric candidate is norm_gate_rules={0: 'symmetric'}.
        It replaces that layer's rule in the existing pass, without extra work
        from replaying a second candidate or changing the native forward.
        Optional finite_fla_by_layer explicitly replaces selected GDN callbacks;
        the empty default uses the original finite_fla at every GDN layer.
        Optional attention_pv_rules selects content1/content0 at existing FA
        layers; its empty default retains content1 at every attention layer.
        """
        if checkpoint_device not in ('cpu','cuda'):
            raise ValueError('Checkpoint storage must be cpu or cuda.')
        self.checkpoint_device=checkpoint_device
        rules={} if norm_gate_rules is None else norm_gate_rules
        if not isinstance(rules,Mapping):
            raise TypeError('norm_gate_rules must be a mapping of integer GDN layer indices to rules.')
        self.norm_gate_rules=dict(rules)
        for index,rule in self.norm_gate_rules.items():
            if isinstance(index,bool) or not isinstance(index,int):
                raise ValueError('norm_gate_rules layer indices must be integers, not bools.')
            layers=model.model.language_model.layers
            if not 0<=index<len(layers) or layers[index].block_type!='linear_attention':
                raise ValueError(f'norm_gate_rules requires an existing GDN layer: {index}')
            if not isinstance(rule,str) or rule not in ('content1','symmetric'):
                raise ValueError(f'Unsupported norm_gate_rule at layer {index}: {rule!r}')
        fla_rules={} if finite_fla_by_layer is None else finite_fla_by_layer
        if not isinstance(fla_rules,Mapping):
            raise TypeError('finite_fla_by_layer must map integer GDN layer indices to callable backends.')
        self.finite_fla_by_layer=dict(fla_rules)
        for index,backend in self.finite_fla_by_layer.items():
            layers=model.model.language_model.layers
            if isinstance(index,bool) or not isinstance(index,int) or not 0<=index<len(layers) or layers[index].block_type!='linear_attention':
                raise ValueError(f'finite_fla_by_layer requires an existing GDN layer: {index!r}')
            if not callable(backend):raise TypeError('A finite FLA backend must be callable.')
        pv_rules={} if attention_pv_rules is None else attention_pv_rules
        if not isinstance(pv_rules,Mapping):
            raise TypeError('attention_pv_rules must map integer FA layer indices to rules.')
        self.attention_pv_rules=dict(pv_rules)
        for index,rule in self.attention_pv_rules.items():
            layers=model.model.language_model.layers
            if isinstance(index,bool) or not isinstance(index,int) or not 0<=index<len(layers) or layers[index].block_type!='full_attention':
                raise ValueError(f'attention_pv_rules requires an existing FA layer: {index!r}')
            if not isinstance(rule,str) or rule not in ('content1','content0'):
                raise ValueError(f'Unsupported attention_pv_rule at layer {index}: {rule!r}')
        self.model=model;self.finite_fa=finite_fa;self.finite_fla=finite_fla
        key_rules={} if key_norm_by_layer is None else key_norm_by_layer
        if not isinstance(key_rules,Mapping):
            raise TypeError('key_norm_by_layer must map integer GDN layers to finite normalization callbacks.')
        self.key_norm_by_layer=dict(key_rules)
        for index,callback in self.key_norm_by_layer.items():
            layers=model.model.language_model.layers
            if isinstance(index,bool) or not isinstance(index,int) or not 0<=index<len(layers) or layers[index].block_type!='linear_attention':
                raise ValueError(f'key_norm_by_layer requires an existing GDN layer: {index!r}')
            if not callable(callback):raise TypeError('A finite key normalization callback must be callable.')
        self.boundaries=FiniteBoundaryOps(True);self.answer=FiniteAnswerOps(True)

    def attribute(self,paired_ids,mask,selection,select_output_rows=True,observer=None):
        if paired_ids.shape!=mask.shape or paired_ids.shape!=(2*selection.batch,selection.length):
            raise ValueError('Endpoint input/mask/target dimensions disagree.')
        mask_cpu=mask.detach().cpu()
        if not bool(((mask_cpu==0)|(mask_cpu==1)).all()) or not torch.equal(mask_cpu[0::2],mask_cpu[1::2]):
            raise ValueError('Each endpoint pair requires the same binary mask.')
        lengths=mask_cpu[0::2].sum(-1).to(torch.int64).tolist()
        if not all(n>0 for n in lengths) or not torch.equal(mask_cpu[0::2].bool(),torch.arange(selection.length)[None,:]<torch.tensor(lengths)[:,None]):
            raise ValueError('Only nonempty contiguous right padding is supported.')
        if not bool((selection.positions.cpu()<torch.tensor(lengths)[selection.samples.cpu()]).all()):
            raise ValueError('A selected target predictor is outside its real input.')
        varlen=any(n!=selection.length for n in lengths)
        model=self.model;layers=model.model.language_model.layers;norm=model.model.language_model.norm
        if len(layers)!=32:raise ValueError('Expected the reviewed32-layer Qwen3.5 model.')
        if any(l.self_attn.config._attn_implementation!='flash_attention_2' for l in layers if l.block_type=='full_attention'):
            raise ValueError('Original model must use its default FA implementation.')
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
        root={};kwargs={};handles=[];calls=[];ledger={};head_shapes=[]
        events=[];validity=[]
        def timed(kind,fn):
            begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            begin.record();tick=time.perf_counter();value=fn();end.record()
            calls.append({'kind':kind,'host_enqueue_seconds':time.perf_counter()-tick,'allocated_after':torch.cuda.memory_allocated()})
            events.append((begin,end))
            return value
        for i,layer in enumerate(layers):
            def capture(_module,args,kw,i=i):
                x=args[0] if args else kw['hidden_states'];root[str(i)]=_copy(x,self.checkpoint_device)
                kwargs[str(i)]=_copy({k:v for k,v in kw.items() if k!='hidden_states'},self.checkpoint_device)
            handles.append(layer.register_forward_pre_hook(capture,with_kwargs=True))
        def final_norm(_module,args,output):
            root['final_norm_input']=_copy(args[0],self.checkpoint_device)
            if observer is not None:root['final_norm_output']=_copy(output,self.checkpoint_device)
        handles.append(norm.register_forward_hook(final_norm))
        def head_input(_module,args):head_shapes.append(list(args[0].shape))
        handles.append(model.lm_head.register_forward_pre_hook(head_input))
        selector=NativeTargetLogitRows(selection) if select_output_rows else None
        try:
            with torch.no_grad():out=timed('native_root_with_'+self.checkpoint_device+'_checkpoints',lambda:model(input_ids=paired_ids,attention_mask=mask,use_cache=False,
                **({'logits_to_keep':selector.rows} if selector is not None else {})))
        finally:
            for h in handles:h.remove()
        expected_rows=len(selector.rows) if selector is not None else selection.length
        if head_shapes!=[[2*selection.batch,expected_rows,model.lm_head.in_features]]:
            raise ValueError('Actual native head did not receive the requested rows.')
        if out.logits.shape!=(2*selection.batch,expected_rows,model.lm_head.out_features):raise ValueError('Native vocabulary/output shape changed.')
        output_bytes=out.logits.numel()*out.logits.element_size();root_peak=torch.cuda.max_memory_allocated()
        z=timed('pack_actual_target_logits',lambda:selector.pack_logits(out.logits) if selector is not None else selection.pack_hidden(out.logits));del out
        # Match the established root-G diagnostic independently of the compiled
        # seed's internal log-softmax fusion. Neither result replaces model logits.
        with torch.no_grad():
            root_logp=timed('actual_root_FP32_logprob_diagnostic',lambda:z.float().log_softmax(-1).gather(
                -1,selection.labels.repeat_interleave(2)[:,None]).squeeze(-1))
        root_lp0=root_logp[0::2].detach().cpu();root_lp1=root_logp[1::2].detach().cpu();del root_logp
        # Consume only actual original logits, with the unchanged full-vocabulary seed.
        with torch.no_grad():mnorm,seed=timed('finite_seed',lambda:self.answer(z,model.lm_head,selection))
        if observer is not None:observer.boundary('norm',mnorm.detach(),root['final_norm_output'])
        lp0=seed['logp0'].detach().cpu();lp1=seed['logp1'].detach().cpu()
        effect_G=float((root_lp1.double()-root_lp0.double()).sum())
        compiled_seed_G=float((lp1.double()-lp0.double()).sum())
        x=root['final_norm_input'].to('cuda')
        with torch.no_grad():m=timed('finite_final_norm',lambda:self.boundaries.norm_residual(x[0::2],x[1::2],norm.weight,mnorm,torch.zeros_like(mnorm),norm.eps))
        if observer is not None:observer.boundary('32',m.detach(),root['final_norm_input'])
        seed_effect=_effect(m,x);del x,mnorm,seed,z
        layout=RightPaddedLengths(lengths,selection.length,paired_ids.device)
        for i in reversed(range(32)):
            layer=layers[i];is_fa=layer.block_type=='full_attention';x=root[str(i)].to('cuda');kw=_copy(kwargs[str(i)],'cuda')
            dc=NativeDecoderCapture(layer,destination='cuda')
            mc=(NativeDenseAttentionCapture(layer.self_attn,flash_attention_forward,flash_attn_varlen_func,flash_attn_func,destination='cuda')
                if is_fa else NativeGDNCapture(layer.linear_attn,device='cuda'))
            def replay():
                with torch.no_grad(),dc,mc:return layer(x,**kw)
            y=timed('native_replay_'+str(i),replay)
            if dc.calls!={k:1 for k in ('input_norm','post_norm','gate','up','silu','down','mlp','decoder')}:raise ValueError('Missing actual decoder captures.')
            if mc.calls!=({'module':1,'interface':1,'native_varlen':int(varlen),'native_dense':int(not varlen)} if is_fa else {'module':1,'conv':1,'FLA':1,'stage':1}):raise ValueError('Missing actual native mixer captures.')
            d,c,e=dc.values,mc.values,getattr(mc,'endpoints',{});scale=getattr(mc,'scale',0.0625)
            expected=root['final_norm_input' if i==31 else str(i+1)].to('cuda')
            row={'block_type':layer.block_type,'decoder_calls':dc.calls,'mixer_calls':mc.calls,
                 'root_output_effect':_effect(m,expected),'replay_output_effect':_effect(m,y),
                 'replay_relative_L2':((expected.float()-y.float()).norm()/expected.float().norm().clamp_min(1e-30))}
            ledger[str(i)]=row;del expected,y,x
            lse=None
            if is_fa:
                args={k:v for k,v in (mc.packed_arguments if varlen else mc.dense_arguments).items() if k!='return_attn_probs'}
                if args['dropout_p']!=0 or not args['causal']:raise ValueError('Unsupported native FA settings.')
                if varlen:
                    expected_cu=torch.tensor([0]+list(torch.tensor(lengths).repeat_interleave(2).cumsum(0).tolist()),device=paired_ids.device,dtype=torch.int32)
                    if not torch.equal(c['packed_cu_seqlens_q'],expected_cu) or not torch.equal(c['packed_cu_seqlens_k'],expected_cu):
                        raise ValueError('Actual native packing differs from the endpoint lengths.')
                    with torch.no_grad():aux,lse,unused=timed('public_FA_LSE_'+str(i),lambda:flash_attn_varlen_func(
                        c['packed_q'],c['packed_k'],c['packed_v'],c['packed_cu_seqlens_q'],c['packed_cu_seqlens_k'],return_attn_probs=True,**args))
                    actual_attention=c['attention_output'][mask.bool()]
                    # Installed FA exposes packed LSE. Reuse FA's own padding
                    # helper; this is a layout conversion, not probability
                    # reconstruction or a second attention implementation.
                    if lse.shape!=(layer.self_attn.config.num_attention_heads,int(expected_cu[-1])):
                        raise ValueError('Unexpected public varlen LSE layout.')
                    lse=pad_input(lse.T,mask.flatten().nonzero().flatten(),len(paired_ids),selection.length).transpose(1,2).contiguous()
                else:
                    with torch.no_grad():aux,lse,unused=timed('public_FA_LSE_'+str(i),lambda:flash_attn_func(c['dense_q'],c['dense_k'],c['dense_v'],return_attn_probs=True,**args))
                    actual_attention=c['attention_output']
                if unused is not None and unused.numel():raise ValueError('Unexpected quadratic probability output.')
                row['FA_auxiliary_relative_L2']=((aux.float()-actual_attention.float()).norm()/aux.float().norm().clamp_min(1e-30));del aux,unused,actual_attention
            del dc,mc
            focused=(observer is not None and callable(getattr(observer,'wants_decoder',None))
                     and bool(observer.wants_decoder(i)))
            if focused and not callable(getattr(observer,'decoder',None)):
                raise ValueError('A focused decoder observer requires a decoder callback.')
            def mixer(upstream):
                if is_fa:
                    cos,sin=kw['position_embeddings'];return attention_finite_pullback(layer.self_attn,c,lse,cos,sin,upstream,self.finite_fa,layout,self.boundaries,focused,
                        pv_rule=self.attention_pv_rules.get(i,'content1'))
                finite_fla=self.finite_fla_by_layer.get(i,self.finite_fla)
                if i in self.key_norm_by_layer:
                    return gdn_finite_pullback(layer.linear_attn,c,e,upstream,scale,finite_fla,focused,
                        norm_gate_rule=self.norm_gate_rules.get(i,'content1'),key_norm_pullback=self.key_norm_by_layer[i])
                if i in self.norm_gate_rules:
                    return gdn_finite_pullback(layer.linear_attn,c,e,upstream,scale,finite_fla,focused,
                        norm_gate_rule=self.norm_gate_rules[i])
                return gdn_finite_pullback(layer.linear_attn,c,e,upstream,scale,finite_fla,focused)
            with torch.no_grad():new,terms=timed('finite_decoder_'+str(i),lambda:decoder_finite_pullback(layer,d,m,mixer,self.boundaries,focused))
            validity.append(torch.isfinite(new).all())
            if focused:observer.decoder(i,d,c,e,m,new,terms)
            del terms
            row['input_effect']=_effect(new,d['input_norm_input']);m=new;del new,d,c,e,lse,kw
            if observer is not None:observer.boundary(str(i),m.detach(),root[str(i)])
        x=root['0'].to('cuda');signed=(m.double()*(x[1::2].double()-x[0::2].double())).sum(-1).cpu()
        torch.cuda.synchronize();seconds=time.perf_counter()-started
        info={'select_output_rows':select_output_rows,'complete_attribution_seconds_with_diagnostics':seconds,
              'checkpoint_device':self.checkpoint_device,'valid_lengths':lengths,'sample_batch':selection.batch,'native_varlen_FA':varlen,
              'norm_gate_rules':{str(i):rule for i,rule in sorted(self.norm_gate_rules.items())},
              'finite_fla_by_layer':sorted(self.finite_fla_by_layer),
              'attention_pv_rules':{str(i):rule for i,rule in sorted(self.attention_pv_rules.items())},
              'key_norm_by_layer':sorted(self.key_norm_by_layer),
              'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved(),
              'root_peak_allocated':root_peak,'actual_head_input_shapes':head_shapes,'actual_output_bytes':output_bytes,
              'selected_predictor_rows':selector.rows.detach().cpu().tolist() if selector is not None else None,
              'root_effect':effect_G,'seed_effect':seed_effect,'signed_sum':float(signed.sum()),
              'relative_residual':(effect_G-float(signed.sum()))/effect_G if effect_G else None,
              'calls':calls,'layers':ledger,'target_logp0':root_lp0.tolist(),'target_logp1':root_lp1.tolist(),
              'compiled_seed_logprob_effect':compiled_seed_G,
              'compiled_seed_logprob_effect_minus_root':compiled_seed_G-effect_G}
        flags=torch.stack(validity).cpu().tolist()
        if not all(flags):raise ValueError('Nonfinite DT coefficients; deferred check failed before return.')
        for record,(begin,end) in zip(calls,events):
            record['stream_elapsed_seconds']=begin.elapsed_time(end)/1000
        scalar_tensors=[]
        def collect(value):
            if isinstance(value,torch.Tensor):
                if value.ndim!=0:raise ValueError('Unexpected nonscalar diagnostic')
                scalar_tensors.append(value)
            elif isinstance(value,dict):
                for item in value.values():collect(item)
            elif isinstance(value,(list,tuple)):
                for item in value:collect(item)
        collect(info)
        scalar_values=torch.stack([v.double() for v in scalar_tensors]).cpu().tolist()
        resolved={id(v):float(x) for v,x in zip(scalar_tensors,scalar_values)}
        def convert(value):
            if isinstance(value,torch.Tensor):return resolved[id(value)]
            if isinstance(value,dict):return {k:convert(v) for k,v in value.items()}
            if isinstance(value,list):return [convert(v) for v in value]
            if isinstance(value,tuple):return tuple(convert(v) for v in value)
            return value
        info=convert(info)
        info['controller_diagnostic_scheduling']={'all_32_finite_checks_passed':True,
            'deferred_scalar_count':len(scalar_values),'stage_timing':'CUDA stream elapsed plus host enqueue; no per-stage barrier'}
        info['complete_attribution_seconds_with_diagnostics']=time.perf_counter()-started
        return signed,info

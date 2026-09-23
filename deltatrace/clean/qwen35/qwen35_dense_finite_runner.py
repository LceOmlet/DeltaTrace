"""Current DT finite propagation through the real Qwen3.5 dense FA/FLA model.

Native target rows are the default; full output rows remain an explicit option.
Shared controller for full versus native-selected output rows. All finite
mathematics is imported unchanged. This path currently requires unpadded equal
length endpoints; it does not claim variable-length multi-example acceptance.
"""
import time
from collections.abc import Mapping
import torch
from flash_attn import flash_attn_func,flash_attn_varlen_func
from transformers.integrations.flash_attention import flash_attention_forward
from qwen35_answer_finite import FiniteAnswerOps,outcome_log_probs,selected_target_log_probs
from qwen35_decoder_finite import NativeDecoderCapture,FiniteBoundaryOps,attention_finite_pullback,decoder_finite_pullback
from qwen35_gdn_finite import NativeGDNCapture,gdn_finite_pullback
from native_dense_attention_capture import NativeDenseAttentionCapture
from native_attention_capture import copy_capture_tensor
from native_target_logit_rows import NativeTargetLogitRows
from vendor_fa_finite_bf16_d256 import RightPaddedLengths


def _copy(value,device,*,pinned_host=False):
    if isinstance(value,torch.Tensor):
        # Native position IDs may be expanded zero-stride views. Such views
        # cannot be copy_ destinations; materialize their actual values before
        # using the existing stride-preserving pinned capture interface.
        if pinned_host and any(n>1 and stride==0 for n,stride in zip(value.shape,value.stride())):
            value=value.contiguous()
        return (copy_capture_tensor(value,device,preserve_strides=True,pinned_host=True)
                if pinned_host else value.detach().to(device,copy=True))
    if isinstance(value,tuple):return tuple(_copy(v,device,pinned_host=pinned_host) for v in value)
    if isinstance(value,list):return [_copy(v,device,pinned_host=pinned_host) for v in value]
    if isinstance(value,dict):return {k:_copy(v,device,pinned_host=pinned_host) for k,v in value.items()}
    if value is not None and not isinstance(value,(str,int,float,bool)):raise TypeError(type(value))
    return value


def _token_effect(m,x):
    # The final contraction is independent across tokens. Bound temporary
    # FP64 storage instead of materializing several full B*T*H tensors.
    tokens=max(1,4*1024*1024//(m.shape[0]*m.shape[-1]))
    return torch.cat([(m[:,start:start+tokens].double()*
        (x[1::2,start:start+tokens].double()-x[0::2,start:start+tokens].double())).sum(-1)
        for start in range(0,m.shape[1],tokens)],dim=1)


def _effect(m,x):return float(_token_effect(m,x).sum())


def _relative_l2(actual,reference,*,deferred=False):
    # Diagnostic only: stream the same error/reference squared sums. This
    # statistic must not create the largest live tensors in a training step.
    tokens=max(1,4*1024*1024//(actual.shape[0]*actual[0,0].numel()))
    squares=torch.zeros(2,device=actual.device,dtype=torch.float64)
    for start in range(0,actual.shape[1],tokens):
        a=actual[:,start:start+tokens].float();r=reference[:,start:start+tokens].float()
        squares[0]+=(a-r).square().sum(dtype=torch.float64)
        squares[1]+=r.square().sum(dtype=torch.float64)
    value=squares[0].sqrt()/squares[1].sqrt().clamp_min(1e-30)
    return value if deferred else float(value)


class Qwen35DenseFiniteRunner:
    def __init__(self,model,finite_fa,finite_fla,*,norm_gate_rules=None,finite_fla_by_layer=None,attention_pv_rules=None,key_norm_by_layer=None,dynamic_shapes=False,compiler_options=None,answer_compiled=True,copy_replay_captures=True,offload_replay_mixer=False,gdn_head_batch_size=None,capture_backend=None,defer_diagnostics=False,compile_gdn_scalar_rules=False,pin_replay_host=False,gdn_gpu_capture_names=(),fa_coefficient_suffix=False,gdn_coefficient_suffix=False,compact_gdn_captures=False,pin_root_host=False):
        """Optional GDN layer-index rules; unspecified layers retain content1.

        The layer0 symmetric candidate is norm_gate_rules={0: 'symmetric'}.
        It replaces that layer's rule in the existing pass, without extra work
        from replaying a second candidate or changing the native forward.
        Optional finite_fla_by_layer explicitly replaces selected GDN callbacks;
        the empty default uses the original finite_fla at every GDN layer.
        Optional attention_pv_rules selects content1/content0 at existing FA
        layers; its empty default retains content1 at every attention layer.
        """
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
        self.copy_replay_captures=copy_replay_captures
        self.offload_replay_mixer=offload_replay_mixer
        self.gdn_head_batch_size=gdn_head_batch_size
        self.capture_backend=capture_backend
        self.defer_diagnostics=defer_diagnostics
        self.compile_gdn_scalar_rules=compile_gdn_scalar_rules
        self.pin_replay_host=pin_replay_host
        self.gdn_gpu_capture_names=tuple(gdn_gpu_capture_names)
        self.fa_coefficient_suffix=fa_coefficient_suffix
        self.gdn_coefficient_suffix=gdn_coefficient_suffix
        self.compact_gdn_captures=compact_gdn_captures
        self.pin_root_host=pin_root_host
        key_rules={} if key_norm_by_layer is None else key_norm_by_layer
        if not isinstance(key_rules,Mapping):
            raise TypeError('key_norm_by_layer must map integer GDN layers to finite normalization callbacks.')
        self.key_norm_by_layer=dict(key_rules)
        for index,callback in self.key_norm_by_layer.items():
            layers=model.model.language_model.layers
            if isinstance(index,bool) or not isinstance(index,int) or not 0<=index<len(layers) or layers[index].block_type!='linear_attention':
                raise ValueError(f'key_norm_by_layer requires an existing GDN layer: {index!r}')
            if not callable(callback):raise TypeError('A finite key normalization callback must be callable.')
        self.boundaries=FiniteBoundaryOps(True,dynamic_shapes=dynamic_shapes,compiler_options=compiler_options)
        # Same owner seed, with an explicit execution choice for platforms
        # whose compiler miscompiles the tiny categorical log-softmax.
        self.answer=FiniteAnswerOps(answer_compiled,dynamic_shapes=dynamic_shapes,compiler_options=compiler_options)

    @torch.no_grad()
    def forward_prefix(self,input_ids,*,past_key_values=None):
        """Native cached prefix/branch evaluation, retaining only the last logits.

        The caller owns cache forks; the HF model owns every cache transition.
        No captures, replay, compilation or finite pullback are needed to read
        the root of a single-input-position intervention.
        """
        forward=getattr(self.model,'forward_root',self.model)
        return forward(input_ids=input_ids,past_key_values=past_key_values,
                       use_cache=True,logits_to_keep=1)

    @torch.no_grad()
    def read_outcomes(self,input_ids,outcome_token_ids,*,past_key_values=None):
        out=self.forward_prefix(input_ids,past_key_values=past_key_values)
        return outcome_log_probs(out.logits[:,-1],outcome_token_ids)

    def attribute(self,paired_ids,mask,selection,select_output_rows=True,observer=None):
        if paired_ids.shape!=mask.shape or paired_ids.shape!=(2*selection.batch,selection.length):
            raise ValueError('Endpoint input/mask/target dimensions disagree.')
        if not bool(mask.eq(1).all()):raise ValueError('This validated dense runner requires unpadded equal lengths.')
        model=self.model;layers=model.model.language_model.layers;norm=model.model.language_model.norm
        if len(layers)!=32:raise ValueError('Expected the reviewed32-layer Qwen3.5 model.')
        if any(l.self_attn.config._attn_implementation!='flash_attention_2' for l in layers if l.block_type=='full_attention'):
            raise ValueError('Original model must use its default FA implementation.')
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
        root={};kwargs={};handles=[];calls=[];ledger={};head_shapes=[]
        events=[];validity=[]
        def effect(coefficients,endpoints):
            value=_token_effect(coefficients,endpoints).sum()
            return value if self.defer_diagnostics else float(value)
        def timed(kind,fn):
            if self.defer_diagnostics:
                # Ported from accelerated/qwen35/controller_deferred.py:
                # keep every diagnostic, resolve stream events at API return.
                begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                begin.record();tick=time.perf_counter();value=fn();end.record()
                calls.append({'kind':kind,'host_enqueue_seconds':time.perf_counter()-tick,'allocated_after':torch.cuda.memory_allocated()})
                events.append((begin,end))
                return value
            torch.cuda.synchronize();tick=time.perf_counter();value=fn();torch.cuda.synchronize()
            calls.append({'kind':kind,'seconds':time.perf_counter()-tick,'allocated_after':torch.cuda.memory_allocated()})
            return value
        for i,layer in enumerate(layers):
            def capture(_module,args,kw,i=i):
                x=args[0] if args else kw['hidden_states'];root[str(i)]=_copy(x,'cpu',pinned_host=self.pin_root_host)
                kwargs[str(i)]=_copy({k:v for k,v in kw.items() if k!='hidden_states'},'cpu',pinned_host=self.pin_root_host)
            handles.append(layer.register_forward_pre_hook(capture,with_kwargs=True))
        def final_norm(_module,args,output):
            root['final_norm_input']=_copy(args[0],'cpu',pinned_host=self.pin_root_host)
            if observer is not None:root['final_norm_output']=_copy(output,'cpu',pinned_host=self.pin_root_host)
            if selection.outcome_token_ids is not None:
                # Qwen's native head consumes these final-normalized predictor
                # rows. Keep only packed rows, not another full checkpoint.
                root['packed_head_input']=_copy(selection.pack_hidden(output),'cpu',pinned_host=self.pin_root_host)
        handles.append(norm.register_forward_hook(final_norm))
        def head_input(_module,args):head_shapes.append(list(args[0].shape))
        handles.append(model.lm_head.register_forward_pre_hook(head_input))
        selector=NativeTargetLogitRows(selection) if select_output_rows else None
        try:
            with torch.no_grad():out=timed('native_root_with_CPU_checkpoints',lambda:model(input_ids=paired_ids,attention_mask=mask,use_cache=False,
                **({'logits_to_keep':selector.rows} if selector is not None else {})))
        finally:
            if self.pin_root_host:torch.cuda.current_stream().synchronize()
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
            root_logp=timed('actual_root_FP32_logprob_diagnostic',lambda:selected_target_log_probs(z,selection))
        root_lp0=root_logp[0::2].detach().cpu();root_lp1=root_logp[1::2].detach().cpu();del root_logp
        # Consume only actual original logits, with the unchanged full-vocabulary seed.
        packed_head=root.pop('packed_head_input',None)
        if packed_head is not None:packed_head=packed_head.to(z.device)
        with torch.no_grad():mnorm,seed=timed('finite_seed',lambda:self.answer(z,model.lm_head,selection,
            original_packed_hidden=packed_head))
        del packed_head
        if observer is not None:observer.boundary('norm',mnorm.detach(),root['final_norm_output'])
        lp0=seed['logp0'].detach().cpu();lp1=seed['logp1'].detach().cpu()
        effect_G=float((root_lp1.double()-root_lp0.double()).sum())
        compiled_seed_G=float((lp1.double()-lp0.double()).sum())
        x=root['final_norm_input'].to('cuda',non_blocking=self.pin_root_host)
        with torch.no_grad():m=timed('finite_final_norm',lambda:self.boundaries.norm_residual(x[0::2],x[1::2],norm.weight,mnorm,torch.zeros_like(mnorm),norm.eps))
        if observer is not None:observer.boundary('32',m.detach(),root['final_norm_input'])
        seed_effect=effect(m,x);del x,mnorm,seed,z
        coefficient_starts=None
        if (self.fa_coefficient_suffix or self.gdn_coefficient_suffix) and observer is None:
            # Every operation in this reviewed no-cache decoder is causal.
            # The identical input prefix therefore has zero displacement at
            # every layer. Its finite coefficients cannot contribute to the
            # signed input effect or feed later-position coefficients. Keep
            # full K/V history, omitting only those finite-FA output positions.
            positions=torch.arange(selection.length,device=paired_ids.device)
            changed=paired_ids[0::2]!=paired_ids[1::2]
            coefficient_starts=torch.where(changed,positions,selection.length).amin(-1).cpu().tolist()
        layout=RightPaddedLengths([selection.length]*selection.batch,selection.length,paired_ids.device,
                                  coefficient_starts=coefficient_starts if self.fa_coefficient_suffix else None)
        gdn_cut=(min(min(coefficient_starts)//64,(selection.length-1)//64)*64
                 if self.gdn_coefficient_suffix and coefficient_starts is not None else 0)
        for i in reversed(range(32)):
            layer=layers[i];is_fa=layer.block_type=='full_attention';x=root[str(i)].to('cuda',non_blocking=self.pin_root_host);kw=_copy(kwargs[str(i)],'cuda',pinned_host=self.pin_root_host)
            # The no-cache Qwen path consumes these activations without
            # mutating them. Optional borrowing keeps their native storage,
            # instead of cloning every projection input and its views again.
            # Observers retain the complete copied diagnostic trace.
            copy_captures=self.copy_replay_captures or observer is not None
            offload_mixer=self.offload_replay_mixer and observer is None
            mixer_device='cpu' if offload_mixer else 'cuda'
            needed=None if copy_captures else {'input_norm_input','post_norm_input',
                                               'gate_output','up_output','silu_output'}
            attention_needed=None if copy_captures else {'q_proj_output',
                'attention_output','query','key','value','q_norm_input','k_norm_input',
                'dense_q','dense_k','dense_v'}
            backend=self.capture_backend
            decoder_capture=NativeDecoderCapture if backend is None else backend.NativeDecoderCapture
            attention_capture=NativeDenseAttentionCapture if backend is None else backend.NativeDenseAttentionCapture
            gdn_capture=NativeGDNCapture if backend is None else backend.NativeGDNCapture
            dc=decoder_capture(layer,destination='cuda',copy_tensors=copy_captures,retained_names=needed)
            mc=(attention_capture(layer.self_attn,flash_attention_forward,flash_attn_varlen_func,flash_attn_func,destination=mixer_device,copy_tensors=copy_captures,retained_names=attention_needed,preserve_strides=offload_mixer,pinned_host=offload_mixer and self.pin_replay_host)
                if is_fa else gdn_capture(layer.linear_attn,device=mixer_device,copy_tensors=copy_captures,preserve_strides=offload_mixer,capture_module_outputs=copy_captures,pinned_host=offload_mixer and self.pin_replay_host,capture_input=copy_captures,gpu_capture_names=self.gdn_gpu_capture_names if offload_mixer else (),coefficient_start=gdn_cut if self.compact_gdn_captures else 0))
            def replay():
                with torch.no_grad(),dc,mc:return layer(x,**kw)
            y=timed('native_replay_'+str(i),replay)
            # Optional parameter-lifetime boundary for an existing FSDP actor.
            # The native replay and every finite rule remain unchanged. Plain
            # HF owners expose no callback and retain their original behavior.
            prepare_layer=getattr(model,'prepare_finite_layer',None)
            if callable(prepare_layer):prepare_layer(layer)
            if dc.calls!={k:1 for k in ('input_norm','post_norm','gate','up','silu','down','mlp','decoder')}:raise ValueError('Missing actual decoder captures.')
            if mc.calls!=({'module':1,'interface':1,'native_varlen':0,'native_dense':1} if is_fa else {'module':1,'conv':1,'FLA':1,'stage':1}):raise ValueError('Missing actual native mixer captures.')
            d,c,e=dc.values,mc.values,getattr(mc,'endpoints',{});scale=getattr(mc,'scale',0.0625)
            mixer_input_shape=getattr(mc,'input_shape',None)
            gdn_capture_start=getattr(mc,'coefficient_start',0)
            dense_aliases=getattr(mc,'dense_aliases',{})
            expected=root['final_norm_input' if i==31 else str(i+1)].to('cuda',non_blocking=self.pin_root_host)
            row={'block_type':layer.block_type,'decoder_calls':dc.calls,'mixer_calls':mc.calls,
                 'root_output_effect':effect(m,expected),'replay_output_effect':effect(m,y),
                 'replay_relative_L2':_relative_l2(y,expected,deferred=self.defer_diagnostics)}
            ledger[str(i)]=row;del expected,y,x
            lse=None
            if is_fa:
                args={k:v for k,v in mc.dense_arguments.items() if k!='return_attn_probs'}
                if args['dropout_p']!=0 or not args['causal']:raise ValueError('Unsupported native FA settings.')
                if not offload_mixer:
                    with torch.no_grad():aux,lse,unused=timed('public_FA_LSE_'+str(i),lambda:flash_attn_func(c['dense_q'],c['dense_k'],c['dense_v'],return_attn_probs=True,**args))
                    if unused is not None and unused.numel():raise ValueError('Unexpected quadratic probability output.')
                    row['FA_auxiliary_relative_L2']=_relative_l2(c['attention_output'],aux,deferred=self.defer_diagnostics);del aux,unused
            del dc,mc
            focused=(observer is not None and callable(getattr(observer,'wants_decoder',None))
                     and bool(observer.wants_decoder(i)))
            if focused and not callable(getattr(observer,'decoder',None)):
                raise ValueError('A focused decoder observer requires a decoder callback.')
            def mixer(upstream):
                nonlocal lse
                if offload_mixer and is_fa:
                    # Reuse the captures' existing CPU destination. Restore
                    # only after the MLP finite operator has consumed its
                    # large operands; no model forward or rule is replaced.
                    def restore():
                        for capture in (c,e):
                            for name,value in capture.items():
                                if isinstance(value,torch.Tensor) and name not in dense_aliases:
                                    capture[name]=copy_capture_tensor(value,'cuda',preserve_strides=True)
                        for name,source in dense_aliases.items():
                            c[name]=c[source].transpose(1,2)
                    timed('restore_mixer_captures_'+str(i),restore)
                if is_fa:
                    if offload_mixer:
                        aux,lse,unused=timed('public_FA_LSE_'+str(i),lambda:flash_attn_func(c['dense_q'],c['dense_k'],c['dense_v'],return_attn_probs=True,**args))
                        if unused is not None and unused.numel():raise ValueError('Unexpected quadratic probability output.')
                        row['FA_auxiliary_relative_L2']=_relative_l2(c['attention_output'],aux,deferred=self.defer_diagnostics);del aux,unused
                        for name in ('dense_q','dense_k','dense_v'):
                            del c[name]
                    cos,sin=kw['position_embeddings'];return attention_finite_pullback(layer.self_attn,c,lse,cos,sin,upstream,self.finite_fa,layout,self.boundaries,focused,
                        pv_rule=self.attention_pv_rules.get(i,'content1'),consume_captures=offload_mixer,input_shape=mixer_input_shape)
                finite_fla=self.finite_fla_by_layer.get(i,self.finite_fla)
                if i in self.key_norm_by_layer:
                    return gdn_finite_pullback(layer.linear_attn,c,e,upstream,scale,finite_fla,focused,
                        norm_gate_rule=self.norm_gate_rules.get(i,'content1'),key_norm_pullback=self.key_norm_by_layer[i],offload_endpoints=offload_mixer,fla_head_batch_size=self.gdn_head_batch_size,input_shape=mixer_input_shape,fla_coefficient_start=gdn_cut,capture_start=gdn_capture_start,
                        norm_gate_pullback=self.boundaries.gdn_norm_gate if self.compile_gdn_scalar_rules else None,
                        conv_silu_pullback=self.boundaries.gdn_conv_silu if self.compile_gdn_scalar_rules else None)
                if i in self.norm_gate_rules:
                    return gdn_finite_pullback(layer.linear_attn,c,e,upstream,scale,finite_fla,focused,
                        norm_gate_rule=self.norm_gate_rules[i],offload_endpoints=offload_mixer,fla_head_batch_size=self.gdn_head_batch_size,input_shape=mixer_input_shape,fla_coefficient_start=gdn_cut,capture_start=gdn_capture_start,
                        norm_gate_pullback=self.boundaries.gdn_norm_gate if self.compile_gdn_scalar_rules else None,
                        conv_silu_pullback=self.boundaries.gdn_conv_silu if self.compile_gdn_scalar_rules else None)
                return gdn_finite_pullback(layer.linear_attn,c,e,upstream,scale,finite_fla,focused,offload_endpoints=offload_mixer,fla_head_batch_size=self.gdn_head_batch_size,input_shape=mixer_input_shape,fla_coefficient_start=gdn_cut,capture_start=gdn_capture_start,
                        norm_gate_pullback=self.boundaries.gdn_norm_gate if self.compile_gdn_scalar_rules else None,
                        conv_silu_pullback=self.boundaries.gdn_conv_silu if self.compile_gdn_scalar_rules else None)
            with torch.no_grad():new,terms=timed('finite_decoder_'+str(i),lambda:decoder_finite_pullback(layer,d,m,mixer,self.boundaries,focused,consume_captures=offload_mixer))
            if self.defer_diagnostics:validity.append(torch.isfinite(new).all())
            elif not bool(torch.isfinite(new).all()):raise ValueError('Nonfinite DT coefficients.')
            if focused:observer.decoder(i,d,c,e,m,new,terms)
            del terms
            row['input_effect']=effect(new,d['input_norm_input']);m=new;del new,d,c,e,lse,kw
            release_layer=getattr(model,'release_finite_layer',None)
            if callable(release_layer):release_layer(layer)
            if observer is not None:observer.boundary(str(i),m.detach(),root[str(i)])
        x=root['0'].to('cuda',non_blocking=self.pin_root_host);signed=_token_effect(m,x).cpu()
        torch.cuda.synchronize();seconds=time.perf_counter()-started
        info={'select_output_rows':select_output_rows,'complete_attribution_seconds_with_diagnostics':seconds,
              'fa_coefficient_starts':coefficient_starts if self.fa_coefficient_suffix else None,
              'gdn_fla_coefficient_start':gdn_cut,
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
        if self.defer_diagnostics:
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

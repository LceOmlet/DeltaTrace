"""Same-root native projection reuse through unmodified public PyTorch SAC."""
import hashlib,inspect
from pathlib import Path
import torch
from torch.utils.checkpoint import create_selective_checkpoint_contexts,CheckpointPolicy


class NativeProjectionReuse:
    def __init__(self,model,selected):
        self.handles=[];self.regions=[];self.active=None;self.validation=None
        assert not model.training and all(not p.requires_grad for p in model.parameters())
        names=['q','k','v','o','gate','up','down'];assert set(selected)<=set(names)
        self.activity={'public_API':'torch.utils.checkpoint.create_selective_checkpoint_contexts',
            'installed_source_sha256':hashlib.sha256(Path(inspect.getfile(create_selective_checkpoint_contexts)).read_bytes()).hexdigest(),
            'recording_memory_scope':'Allocator counters in per-layer records are graph-recording snapshots, not per-replay measurements. The outer complete caller measures every current peak.', 'selected_projections':list(selected),'policy_detail':'Omit q caching in final20 native layers to leave allocator headroom; other selections unchanged','allow_cache_entry_mutation':False,
            'native_model_methods_replaced':False,'saved_mm_outputs':0,'reused_mm_outputs':0,
            'logical_cached_bytes':0,'maximum_logical_cached_bytes':0,'regions':[],
            'scope':'One fresh original B2 root and its existing native B2 decoder replay, within one complete graph execution. No inter-call cache.'}
        if not selected:return
        for index,layer in enumerate(model.model.layers):
            weights=[layer.self_attn.q_proj.weight,layer.self_attn.k_proj.weight,layer.self_attn.v_proj.weight,layer.self_attn.o_proj.weight,layer.mlp.gate_proj.weight,layer.mlp.up_proj.weight,layer.mlp.down_proj.weight]
            local_selected=[name for name in selected if not (name=='q' and index>=len(model.model.layers)-20)]
            rec={'selected_projections':local_selected,'layer':index,'save_mm_visits':0,'reuse_mm_visits':0,'input_exact':None,'cached_output_bytes':0,'operations':[],'completed':False}
            state={'selected':local_selected,'rec':rec,'weights':weights,'versions':[w._version for w in weights],'phase':0,'input':None,'contexts':None,'native_forward':type(layer).forward}
            def policy(ctx,op,*args,_s=state,**kwargs):
                if op is not torch.ops.aten.mm.default:return CheckpointPolicy.MUST_RECOMPUTE
                rec=_s['rec'];replaying=ctx.is_recompute;key='reuse_mm_visits' if replaying else 'save_mm_visits';step=rec[key]
                assert step<7 and not kwargs
                x,wt=args;weight=_s['weights'][step]
                assert weight._version==_s['versions'][step] and wt.data_ptr()==weight.data_ptr()
                assert wt.shape==weight.T.shape and wt.stride()==weight.T.stride()
                assert x.ndim==2 and x.dtype==wt.dtype==torch.float16
                desc={'projection':names[step],'input_shape':list(x.shape),'input_stride':list(x.stride()),'weight_shape':list(weight.shape),'output_bytes':x.shape[0]*wt.shape[1]*2}
                save=names[step] in _s['selected']
                if replaying:
                    assert desc==rec['operations'][step]
                    if save:self.activity['reused_mm_outputs']+=1
                else:
                    rec['operations'].append(desc)
                    if save:
                        self.activity['saved_mm_outputs']+=1;rec['cached_output_bytes']+=desc['output_bytes']
                        self.activity['logical_cached_bytes']+=desc['output_bytes']
                        self.activity['maximum_logical_cached_bytes']=max(self.activity['maximum_logical_cached_bytes'],self.activity['logical_cached_bytes'])
                rec[key]+=1
                return CheckpointPolicy.MUST_SAVE if save else CheckpointPolicy.MUST_RECOMPUTE
            state['contexts']=create_selective_checkpoint_contexts(policy,allow_cache_entry_mutation=False)
            def enter(module,args,_s=state):
                assert self.active is None and len(args)==1
                assert type(module).forward is _s['native_forward'] and 'forward' not in module.__dict__
                phase=_s['phase'];assert phase in (0,1);value=args[0]
                _s['rec'][('root' if phase==0 else 'replay')+'_enter_allocated']=torch.cuda.memory_allocated()
                if phase==0:_s['input']=value.detach();_s['input_version']=value._version
                else:
                    assert self.validation is not None and _s['input']._version==_s['input_version']
                    assert value.shape==_s['input'].shape and value.stride()==_s['input'].stride()
                    self.validation.equal(value,_s['input'],f"layer{_s['rec']['layer']}.SAC_native_input")
                    _s['rec']['input_exact']=True
                    assert _s['rec']['save_mm_visits']==7
                assert all(w._version==v for w,v in zip(_s['weights'],_s['versions']))
                context=_s['contexts'][phase];context.__enter__();self.active=(_s,context)
            def leave(module,args,output,_s=state):
                assert self.active is not None and self.active[0] is _s
                _,context=self.active;self.active=None;context.__exit__(None,None,None)
                if output is None:return
                _s['rec'][('root' if _s['phase']==0 else 'replay')+'_exit_allocated']=torch.cuda.memory_allocated()
                if _s['phase']==1:
                    rec=_s['rec'];assert rec['save_mm_visits']==rec['reuse_mm_visits']==7
                    self.activity['logical_cached_bytes']-=rec['cached_output_bytes'];rec['completed']=True
                    _s['input']=None;_s['contexts']=None
                _s['phase']+=1
            self.handles.extend([layer.register_forward_pre_hook(enter),layer.register_forward_hook(leave,always_call=True)])
            self.regions.append(state);self.activity['regions'].append(rec)
    def __enter__(self):return self
    def __exit__(self,kind,error,tb):
        if self.active is not None:
            _,context=self.active;self.active=None;context.__exit__(kind,error,tb)
        for h in self.handles:h.remove()
        self.handles.clear()
        if kind is None:
            assert all(s['rec']['completed'] for s in self.regions)
            assert self.activity['saved_mm_outputs']==self.activity['reused_mm_outputs']==sum(len(s['selected']) for s in self.regions)
            assert self.activity['logical_cached_bytes']==0
        for s in self.regions:s['input']=s['contexts']=None
        self.regions.clear()
        return False

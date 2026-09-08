"""Original decoder0/3 replay/backward and full finite propagation, no root run."""
import os
os.environ.update(MACA_PATH='/opt/maca',HF_HUB_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1')
import ast,hashlib,io,json,sys,time,traceback,zipfile
from collections import Counter
from pathlib import Path
HERE=Path(__file__).resolve().parent;p=json.loads((HERE/'protocol.json').read_bytes())
sha=lambda b:hashlib.sha256(b).hexdigest()
os.environ['TRITON_CACHE_DIR']=p['compiler_cache'];os.environ['TORCHINDUCTOR_CACHE_DIR']=str(HERE/'inductor_cache')
r={'status':'running','protocol':p,'full_model_loads':0,'full_model_forwards':0,'generation_calls':0,'quality_queries':0,
   'whole_model_attributions':0,'meta_model_constructions':0,'decoder_loads':0,'native_forward_attempts':0,
   'native_backward_attempts':0,'auxiliary_FA_attempts':0,'finite_attempts':0,'layers':{},'calls':[],'artifacts':[]}
start=time.perf_counter()
def save():
    f=HERE/'results.partial';f.write_text(json.dumps(r,indent=2));f.replace(HERE/'results.json')
def relative(a,b):
    def array(x):return x.detach().to(device='cpu',dtype=torch.float64).numpy() if isinstance(x,torch.Tensor) else np.asarray(x,dtype=np.float64)
    a=array(a).ravel();b=array(b).ravel();assert a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    na=float(np.linalg.norm(a));nb=float(np.linalg.norm(b))
    return {'relative_L2':float(np.linalg.norm(b-a))/na if na else None,'max_abs':float(np.abs(b-a).max()),
            'cosine':float(a@b)/(na*nb) if na*nb else None,'reference_norm':na}
def timed(label,fn):
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();base=torch.cuda.memory_allocated();t=time.perf_counter()
    out=fn();torch.cuda.synchronize()
    r['calls'].append({'kind':label,'seconds':time.perf_counter()-t,'before_bytes':base,'peak_bytes':torch.cuda.max_memory_allocated()});save();return out
def archive(name,obj):
    f=HERE/(name+'.pt');torch.save(obj,f)
    r['artifacts'].append({'file':f.name,'sha256':sha(f.read_bytes()),'bytes':f.stat().st_size,'download':False});save()
def persist(name,values):
    f=HERE/(name+'.npz');np.savez_compressed(f,**{k:v.detach().float().cpu().numpy() for k,v in values.items()})
    r['artifacts'].append({'file':f.name,'sha256':sha(f.read_bytes()),'bytes':f.stat().st_size,'download':True});save()
def effect(label,upstream,out_pair,coeff,in_pair,record):
    direct=(upstream.double()*(out_pair[1::2].double()-out_pair[0::2].double())).flatten(1).sum(1)
    allocated=(coeff.double()*(in_pair[1::2].double()-in_pair[0::2].double())).flatten(1).sum(1)
    record[label]={'direct':direct.tolist(),'allocated':allocated.tolist(),'residual':(allocated-direct).tolist(),**relative(direct,allocated)}
try:
    for name,digest in p['files_sha256'].items():assert sha((HERE/name).read_bytes())==digest
    parent=Path(p['parent_directory']);raw=(parent/'results.json').read_bytes();assert sha(raw)==p['parent_sha256']
    previous=json.loads(raw)
    fn=next(n for n in ast.parse((parent/'study.py').read_bytes()).body if isinstance(n,ast.FunctionDef) and n.name=='sources')
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'<native-source-guard>','exec'))
    r['sources_before']=sources();assert r['sources_before']['sha256']==p['source_tree_sha256']
    cp=Path(p['checkpoint'])
    for name,digest in p['checkpoint_config_tokenizer_sha256'].items():assert sha((cp/name).read_bytes())==digest
    stats=lambda:{x.name:[x.stat().st_size,x.stat().st_mtime_ns] for x in cp.glob('*.safetensors')}
    r['weight_stats_before']=stats();assert r['weight_stats_before']==previous['weight_stats_before']
    root_file=parent/'native_root_checkpoints.pt';assert sha(root_file.read_bytes())==p['root_checkpoints_sha256']
    import numpy as np
    import torch
    import transformers,flash_attn
    from safetensors import safe_open
    from transformers import AutoConfig
    from transformers.utils import ContextManagers
    from transformers.core_model_loading import _materialize_copy
    from transformers.models.qwen3_5 import modeling_qwen3_5 as native
    from transformers.integrations.flash_attention import flash_attention_forward
    from flash_attn import flash_attn_varlen_func
    from flash_attn.bert_padding import pad_input
    from native_attention_capture import NativeAttentionCapture
    from qwen35_gdn_finite import NativeGDNCapture,gdn_finite_pullback
    from qwen35_decoder_finite import NativeDecoderCapture,FiniteBoundaryOps,attention_finite_pullback,decoder_finite_pullback
    from finite_fla_gpu import make_compiled_finite_pullback,verify_native_sources
    from vendor_fa_finite_bf16_d256 import RightPaddedLengths,VendorFAFiniteP1BF16D256
    assert sha(Path(native.__file__).read_bytes())==p['native_model_sha256'];verify_native_sources(p['native_stage_source_sha256'])
    import flash_attn.flash_attn_interface as fa
    assert sha(Path(fa.__file__).read_bytes())==p['installed_FA_interface_sha256']
    r['versions']={'torch':torch.__version__,'transformers':transformers.__version__,'flash_attn':flash_attn.__version__,'device':torch.cuda.get_device_name()}
    cfg=AutoConfig.from_pretrained(cp,local_files_only=True,trust_remote_code=False);cfg._attn_implementation='flash_attention_2'
    cls=native.Qwen3_5ForConditionalGeneration;r['meta_model_constructions']+=1;save()
    with ContextManagers(cls.get_init_context(torch.bfloat16,False,False,False)):container=cls(cfg)
    assert container._get_dtype_plan(torch.bfloat16)=={}
    tc=cfg.text_config;assert tc.hidden_act=='silu' and tc.intermediate_size==12288
    assert tc.layer_types[0]=='linear_attention' and tc.layer_types[3]=='full_attention'
    rotary_dtype=container.model.language_model.rotary_emb.inv_freq.dtype
    layers={i:container.model.language_model.layers[i] for i in (3,0)};del container
    index=json.loads((cp/'model.safetensors.index.json').read_bytes())['weight_map'];r['weight_tensor_receipts']={}
    for i,layer in layers.items():
        prefix=f'model.language_model.layers.{i}.';desired={k:v.dtype for k,v in layer.state_dict().items()}
        assert set(desired.values())=={torch.bfloat16}
        names={k:v for k,v in index.items() if k.startswith(prefix)};assert {k.removeprefix(prefix) for k in names}==set(desired)
        r['decoder_loads']+=1;save();state={}
        for shard in sorted(set(names.values())):
            with safe_open(cp/shard,framework='pt',device='cpu') as sf:
                for name in names:
                    if names[name]!=shard:continue
                    raw_value=sf.get_tensor(name);local=name.removeprefix(prefix)
                    value=_materialize_copy(sf.get_slice(name),device='cpu',dtype=desired[local]);state[local]=value
                    r['weight_tensor_receipts'][name]={'shape':list(value.shape),'loaded_dtype':str(value.dtype),'checkpoint_dtype':str(raw_value.dtype),
                        'loaded_sha256':sha(value.view(torch.uint8).numpy().tobytes()),'checkpoint_sha256':sha(raw_value.view(torch.uint8).numpy().tobytes())}
        layer.load_state_dict(state,strict=True,assign=True);layer.eval().requires_grad_(False).to('cuda');del state
    root=torch.load(root_file,map_location='cpu',weights_only=True)
    B,T,C=root['3'].shape;assert (B,T,C)==(4,605,4096)
    mask=(torch.arange(T,device='cuda')[None,:]<torch.tensor([605,605,368,368],device='cuda')[:,None]).long()
    pos=torch.arange(T,device='cuda').view(1,1,T).expand(4,B,T)
    rotary=native.Qwen3_5TextRotaryEmbedding(tc,device='cuda');assert rotary.inv_freq.dtype==rotary_dtype
    cos,sin=rotary(root['3'].to('cuda'),pos[1:]);assert cos.shape==(B,T,64)
    assert torch.equal(cos[0::2],cos[1::2]) and torch.equal(sin[0::2],sin[1::2])
    layout=RightPaddedLengths([605,368],T,'cuda')
    finite_fa=VendorFAFiniteP1BF16D256(Path(p['build_directory'])/'libdeltatrace_fa_finite_bf16_d256.so',p['library_sha256'])
    finite_fla=make_compiled_finite_pullback(reuse_scalar_products=False);boundaries=FiniteBoundaryOps(compiled=True)
    benchmark_counts=Counter()
    def observe_benchmarks(frame,event,_arg):
        name=frame.f_code.co_name;file=frame.f_code.co_filename.replace('\\','/')
        if event=='call' and name=='benchmark_gpu' and 'torch/_inductor/' in file:benchmark_counts['inductor_benchmark_gpu']+=1
        if event=='call' and name=='do_bench' and 'triton/' in file:benchmark_counts['triton_do_bench']+=1
    for i,layer in layers.items():
        row={'block_type':layer.block_type,'native_forward_attempts':0,'native_backward_attempts':0,'finite_attempts':0,'effects':{}}
        r['layers'][str(i)]=row;save()
        x=root[str(i)].to('cuda').detach().requires_grad_(True)
        masker=native.create_causal_mask if i==3 else native.create_recurrent_attention_mask
        layer_mask=masker(config=tc,inputs_embeds=x,attention_mask=mask,past_key_values=None,position_ids=pos[0])
        assert torch.equal(layer_mask,mask)
        dc=NativeDecoderCapture(layer)
        mc=(NativeAttentionCapture(layer.self_attn,flash_attention_forward,flash_attn_varlen_func) if i==3 else NativeGDNCapture(layer.linear_attn))
        def forward():
            with dc,mc:return layer(x,position_embeddings=(cos,sin),attention_mask=layer_mask,position_ids=pos[0],past_key_values=None,use_cache=False)
        r['native_forward_attempts']+=1;row['native_forward_attempts']+=1;save()
        y=timed(f'layer{i}_original_forward_with_capture',forward)
        archive(f'layer{i}_native_capture',{'decoder':dc.values,'mixer':mc.values,'endpoints':getattr(mc,'endpoints',{}),
            'scale':getattr(mc,'scale',0.0625),'mask':mask.cpu(),'cos':cos.cpu(),'sin':sin.cpu()})
        row['native_replay_vs_root']=relative(root[str(i+1)],y);row['decoder_capture_calls']=dc.calls;row['mixer_capture_calls']=mc.calls
        assert dc.calls=={k:1 for k in ('input_norm','post_norm','gate','up','silu','down','mlp','decoder')}
        assert mc.calls==({'module':1,'interface':1,'native_varlen':1} if i==3 else {'module':1,'conv':1,'FLA':1,'stage':1})
        seed=y[1::2].detach().float()*mask[1::2,:,None];dy=torch.zeros_like(y);dy[1::2]=seed.to(dy.dtype)
        # Observe real autograd nodes on their execution threads; do not demand
        # that main-thread sys.setprofile see native backward callbacks.
        nodes=[];seen=set();todo=[y.grad_fn];node_types=Counter();events=[];handles=[]
        while todo:
            node=todo.pop()
            if node is None or node in seen:continue
            seen.add(node);name=type(node).__name__;node_types[name]+=1
            if any(part in name for part in ('FlashAttn','GatedDelta','CausalConv')):nodes.append(node)
            todo.extend(child for child,_ in node.next_functions if child is not None)
        for node in nodes:
            def observed(grad_inputs,grad_outputs,name=type(node).__name__):
                events.append({'node':name,'input_gradients_present':[v is not None for v in grad_inputs]})
            handles.append(node.register_hook(observed))
        r['native_backward_attempts']+=1;row['native_backward_attempts']+=1;save()
        try:gradient,=timed(f'layer{i}_original_backward',lambda:torch.autograd.grad(y,x,dy))
        finally:
            for handle in handles:handle.remove()
        row['native_autograd_nodes']=dict(node_types);row['native_backward_node_events']=events
        native_gradient=gradient[1::2].detach();assert torch.isfinite(native_gradient).all()
        persist(f'layer{i}_native_gradient',{'input_gradient':native_gradient})
        d={k:v.to('cuda').contiguous() for k,v in dc.values.items()}
        c={k:None if v is None else v.to('cuda').contiguous() for k,v in mc.values.items()}
        e={k:v.to('cuda').contiguous() for k,v in getattr(mc,'endpoints',{}).items()};scale=getattr(mc,'scale',0.0625)
        row['endpoint_boundaries']={
            'mixer_input':relative(d['input_norm_output'],c['input']),
            'mixer_residual':relative(d['post_norm_input'],(d['input_norm_input']+c['output'])),
            'MLP_product':relative(d['down_input'],d['silu_output']*d['up_output']),
            'MLP_residual':relative(d['output'],d['post_norm_input']+d['mlp_output'])}
        del x,y,gradient,dy,seen,todo,nodes,handles,dc
        lse=None
        if i==3:
            row['actual_public_FA_arguments']=mc.packed_arguments
            args={k:v for k,v in mc.packed_arguments.items() if k!='return_attn_probs'}
            r['auxiliary_FA_attempts']+=1;save()
            with torch.no_grad():aux=timed('layer3_public_FA_auxiliary_LSE',lambda:flash_attn_varlen_func(
                c['packed_q'],c['packed_k'],c['packed_v'],c['packed_cu_seqlens_q'],c['packed_cu_seqlens_k'],return_attn_probs=True,**args))
            out,raw_lse,unused=aux;assert unused is None or unused.numel()==0
            indices=mask.flatten().nonzero().flatten();assert raw_lse.shape==(16,int(mask.sum()))
            lse=pad_input(raw_lse.T,indices,B,T).transpose(1,2).contiguous()
            row['auxiliary_FA_vs_default']=relative(c['attention_output'],pad_input(out,indices,B,T))
            gates=c['q_proj_output'].view(B,T,16,512)[...,256:]
            row['endpoint_boundaries']['attention_gate_product']=relative(c['o_proj_input'],
                (c['attention_output']*gates.sigmoid()).reshape(B,T,4096))
            archive('layer3_public_LSE',{'LSE':lse.cpu()});del out,raw_lse,unused,aux,gates
        del mc
        def call_finite(label,dd,cc,ee,ll,diagnostics):
            assert row['finite_attempts']<3;r['finite_attempts']+=1;row['finite_attempts']+=1;save()
            def mixer(m):
                if i==3:return attention_finite_pullback(layer.self_attn,cc,ll,cos,sin,m,finite_fa,layout,boundaries,diagnostics)
                return gdn_finite_pullback(layer.linear_attn,cc,ee,m,scale,finite_fla,diagnostics)
            def run():
                with torch.no_grad():return decoder_finite_pullback(layer,dd,seed,mixer,boundaries,diagnostics)
            assert sys.getprofile() is None;sys.setprofile(observe_benchmarks)
            try:answer=timed(f'layer{i}_{label}',run)
            finally:sys.setprofile(None)
            assert torch.isfinite(answer[0]).all();return answer
        actual,diagnostic=call_finite('finite_actual_cold',d,c,e,lse,True)
        effect('decoder',seed,d['output'],actual,d['input_norm_input'],row['effects'])
        effect('mixer',diagnostic['m_mixer_output'],c['output'],diagnostic['m_mixer_input'],c['input'],row['effects'])
        effect('MLP',seed,d['mlp_output'],diagnostic['m_mlp_norm_output'],d['post_norm_output'],row['effects'])
        row['actual_padding_max_abs']=float(actual[1,368:].abs().max())
        persist(f'layer{i}_actual',{'finite':actual,'input0':d['input_norm_input'][0::2],
            'input1':d['input_norm_input'][1::2],'output0':d['output'][0::2],'output1':d['output'][1::2],'upstream':seed})
        actual_cpu=actual.cpu();del actual,diagnostic
        warm,_=call_finite('finite_actual_warm',d,c,e,lse,False)
        row['cold_warm_difference']=relative(actual_cpu,warm);del warm,actual_cpu
        def duplicate(v):
            if v is None:return None
            assert v.shape[0]==4;return v[1::2].repeat_interleave(2,dim=0).contiguous()
        dd={k:duplicate(v) for k,v in d.items()}
        cc={k:duplicate(v) for k,v in c.items() if not k.startswith('packed_')}
        ee={k:duplicate(v) for k,v in e.items()};ll=None if lse is None else duplicate(lse)
        equal,_=call_finite('finite_equal_endpoint_limit',dd,cc,ee,ll,False)
        row['equal_endpoint_vs_native']=relative(native_gradient,equal)
        row['equal_padding_max_abs']=float(equal[1,368:].abs().max())
        row['native_padding_max_abs']=float(native_gradient[1,368:].abs().max())
        persist(f'layer{i}_equal',{'finite':equal});row['status']='complete';save()
        del d,c,e,dd,cc,ee,ll,lse,equal,native_gradient,seed
    r['compiler_benchmark_observations']=dict(benchmark_counts)
    r['benchmark_observation_limit']='Main-thread call observations, not exact total tuning repetitions or worker-process GPU calls; all cold wall time is retained.'
    r['compiler_counters']={str(k):dict(v) for k,v in torch._dynamo.utils.counters.items()}
    r['compiler_generated_sources']=[]
    for file in (HERE/'inductor_cache').rglob('*.py'):
        raw=file.read_bytes();r['compiler_generated_sources'].append({'file':str(file.relative_to(HERE)),'sha256':sha(raw),'bytes':len(raw)})
    r['sources_after']=sources();assert r['sources_after']==r['sources_before']
    assert sha(Path(fa.__file__).read_bytes())==p['installed_FA_interface_sha256']
    r['weight_stats_after']=stats();assert r['weight_stats_after']==r['weight_stats_before']
    r['status']='both_original_decoder_families_finite_propagation_executed'
except Exception:
    sys.setprofile(None);r['status']='failed';r['error']=traceback.format_exc()
finally:
    r['job_seconds']=time.perf_counter()-start;save()
    with zipfile.ZipFile(HERE/'review_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
        names=list(p['files_sha256'])+['protocol.json','results.json']+[a['file'] for a in r['artifacts'] if a['download']]
        names += [x['file'] for x in r.get('compiler_generated_sources',[])]
        for name in names:
            if (HERE/name).exists():z.write(HERE/name,name)
    print(json.dumps({'status':r['status'],'finite_attempts':r['finite_attempts'],'error':r.get('error')}),flush=True)

"""One explicit author-answer finite pass over32 original saved root inputs.

Engineering integration only. Original decoder replay and capture are paid;
no shadow model, new root forward, generation or benchmark evaluation.
"""
import os
os.environ.update(MACA_PATH='/opt/maca',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1')
import ast,hashlib,io,json,sys,time,traceback,zipfile
from collections import Counter
from pathlib import Path
HERE=Path(__file__).resolve().parent;p=json.loads((HERE/'protocol.json').read_bytes());sha=lambda b:hashlib.sha256(b).hexdigest()
os.environ['TRITON_CACHE_DIR']=p['compiler_cache'];os.environ['TORCHINDUCTOR_CACHE_DIR']=p['boundary_compiler_cache']
r={'status':'running','protocol':p,'full_model_loads':0,'root_forwards':0,'generation_calls':0,'quality_queries':0,
    'meta_model_constructions':0,'decoder_loads':0,'decoder_replays':0,'decoder_capture_reuses':0,'finite_decoder_calls':0,
    'head_norm_forwards':0,'head_norm_backwards':0,'finite_answer_calls':0,'auxiliary_FA_calls':0,
    'whole_model_attributions':0,'calls':[],'layers':{},'artifacts':[]}
start=time.perf_counter()
def save():
    f=HERE/'results.partial';f.write_text(json.dumps(r,indent=2));f.replace(HERE/'results.json')
def timed(label,fn):
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();base=torch.cuda.memory_allocated();tick=time.perf_counter()
    out=fn();torch.cuda.synchronize()
    r['calls'].append({'kind':label,'seconds':time.perf_counter()-tick,'before_bytes':base,'peak_bytes':torch.cuda.max_memory_allocated()});save();return out
def metric(a,b):
    a=a.detach().to(device='cpu',dtype=torch.float64).numpy().ravel();b=b.detach().to(device='cpu',dtype=torch.float64).numpy().ravel()
    assert a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    na=float(np.linalg.norm(a));nb=float(np.linalg.norm(b))
    return {'relative_L2':float(np.linalg.norm(b-a))/na if na else None,'max_abs':float(np.abs(b-a).max()),'cosine':float(a@b)/(na*nb) if na*nb else None}
def archive(name,value,download=False):
    f=HERE/(name+'.pt');torch.save(value,f)
    r['artifacts'].append({'file':f.name,'bytes':f.stat().st_size,'sha256':sha(f.read_bytes()),'download':download});save()
def persist(name,values):
    f=HERE/(name+'.npz');np.savez_compressed(f,**{k:v.detach().float().cpu().numpy() for k,v in values.items()})
    r['artifacts'].append({'file':f.name,'bytes':f.stat().st_size,'sha256':sha(f.read_bytes()),'download':True});save()
def paired_effect(m,pair):
    return (m.double()*(pair[1::2].double()-pair[0::2].double())).flatten(1).sum(1)
try:
    for name,digest in p['files_sha256'].items():assert sha((HERE/name).read_bytes())==digest
    parent=Path(p['parent_directory']);raw=(parent/'results.json').read_bytes();assert sha(raw)==p['parent_sha256'];previous=json.loads(raw)
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
    from transformers import AutoConfig,AutoTokenizer
    from transformers.utils import ContextManagers
    from transformers.core_model_loading import _materialize_copy
    from transformers.models.qwen3_5 import modeling_qwen3_5 as native
    from transformers.integrations.flash_attention import flash_attention_forward
    from flash_attn import flash_attn_varlen_func
    from flash_attn.bert_padding import pad_input
    from native_attention_capture import NativeAttentionCapture
    from qwen35_gdn_finite import NativeGDNCapture,gdn_finite_pullback
    from qwen35_decoder_finite import NativeDecoderCapture,FiniteBoundaryOps,attention_finite_pullback,decoder_finite_pullback
    from qwen35_answer_finite import PackedAnswerTargets,FiniteAnswerOps
    from finite_fla_gpu import make_compiled_finite_pullback,verify_native_sources
    from vendor_fa_finite_bf16_d256 import RightPaddedLengths,VendorFAFiniteP1BF16D256
    from official_fixed_text_inputs import load_author_preparer,prepare_fixed_text
    import flash_attn.flash_attn_interface as fa
    assert sha(Path(native.__file__).read_bytes())==p['native_model_sha256'];verify_native_sources(p['native_stage_source_sha256'])
    assert sha(Path(fa.__file__).read_bytes())==p['installed_FA_interface_sha256']
    r['versions']={'torch':torch.__version__,'transformers':transformers.__version__,'flash_attn':flash_attn.__version__,'device':torch.cuda.get_device_name()}
    cfg=AutoConfig.from_pretrained(cp,local_files_only=True,trust_remote_code=False);cfg._attn_implementation='flash_attention_2'
    cls=native.Qwen3_5ForConditionalGeneration;r['meta_model_constructions']+=1
    with ContextManagers(cls.get_init_context(torch.bfloat16,False,False,False)):container=cls(cfg)
    assert container._get_dtype_plan(torch.bfloat16)=={} and cfg.tie_word_embeddings is False
    tc=cfg.text_config;layers=container.model.language_model.layers;head=container.lm_head;norm=container.model.language_model.norm
    rotary_dtype=container.model.language_model.rotary_emb.inv_freq.dtype
    assert len(layers)==32 and tc.vocab_size==248320
    index=json.loads((cp/'model.safetensors.index.json').read_bytes())['weight_map'];r['weight_tensor_receipts']={}
    def load_module(module,prefix):
        desired={k:v.dtype for k,v in module.state_dict().items()};assert set(desired.values())=={torch.bfloat16}
        names={k:v for k,v in index.items() if k.startswith(prefix)};assert {k.removeprefix(prefix) for k in names}==set(desired)
        state={}
        for shard in sorted(set(names.values())):
            with safe_open(cp/shard,framework='pt',device='cpu') as sf:
                for name in names:
                    if names[name]!=shard:continue
                    key=name.removeprefix(prefix);value=_materialize_copy(sf.get_slice(name),device='cpu',dtype=desired[key]);state[key]=value
                    r['weight_tensor_receipts'][name]={'loaded_dtype':str(value.dtype),'shape':list(value.shape),
                        'loaded_sha256':sha(value.view(torch.uint8).numpy().tobytes()),'shard':shard}
        module.load_state_dict(state,strict=True,assign=True);module.eval().requires_grad_(False).to('cuda')
    timed('load_original_head',lambda:load_module(head,'lm_head.'))
    timed('load_original_final_norm',lambda:load_module(norm,'model.language_model.norm.'))
    root=torch.load(root_file,map_location='cpu',weights_only=True);B,T,C=root['0'].shape;assert (B,T,C)==(4,605,4096)
    tokenizer=AutoTokenizer.from_pretrained(cp,local_files_only=True,trust_remote_code=False);tokenizer.pad_token=tokenizer.eos_token
    assert tokenizer.eos_token_id==248046
    sraw=(Path(p['spans_directory'])/'results.json').read_bytes();assert sha(sraw)==p['spans_sha256'];spans=json.loads(sraw)
    preparer=load_author_preparer(p['author_root'],p['author_source_sha256']);cases=[];offsets=[];r['target_selection']=[]
    for row in spans['cases']:
        dataset,j=row['dataset'],row['index'];raw=(Path(p['author_root'])/'exp/exp2/data'/f'{dataset}.jsonl').read_bytes()
        assert sha(raw)==p['cache_sha256'][dataset];record=json.loads(raw.decode().splitlines()[j])
        # Same verified CPU text-preparation use as the completed span audit.
        # No model object or model/generation function is fabricated or called.
        engine=preparer.__new__(preparer);engine.tokenizer=tokenizer;engine.device='cpu'
        case=prepare_fixed_text(engine,record['prompt'],record['target']);del engine
        assert sha(case['input_ids'].numpy().tobytes())==row['input_metadata']['input_ids_sha256']
        lo,hi=row['mapping']['new_sink_span'];selected=list(range(lo,hi+1));cases.append(case);offsets.append(selected)
        r['target_selection'].append({'dataset':dataset,'index':j,'sink_closed':[lo,hi],'prompt_length':case['prompt_length'],
            'offsets':selected,'target_ids_sha256':sha(case['target_ids'].numpy().tobytes()),'count':len(selected)})
    selection=PackedAnswerTargets(cases,offsets,T,'cuda');assert selection.counts==[38,21]
    boundaries=FiniteBoundaryOps(True);answer_ops=FiniteAnswerOps(True)
    benchmark_counts=Counter()
    def observe_benchmarks(frame,event,_value):
        if event=='call' and frame.f_code.co_name=='benchmark_gpu' and 'torch/_inductor/' in frame.f_code.co_filename:benchmark_counts['inductor_benchmark_gpu']+=1
    def finite_timed(label,fn):
        assert sys.getprofile() is None;sys.setprofile(observe_benchmarks)
        try:
            with torch.no_grad():return timed(label,fn)
        finally:sys.setprofile(None)
    x=root['final_norm_input'].to('cuda').detach().requires_grad_(True)
    r['head_norm_forwards']+=1
    def native_head_norm():
        y=norm(x);return y,head(selection.pack_hidden(y))
    y,z=timed('original_final_norm_and_packed_answer_head',native_head_norm)
    r['final_norm_replay_vs_root']=metric(root['final_norm_output'],y)
    archive('actual_answer_operands',{'norm_input':x.detach().cpu(),'norm_output':y.detach().cpu(),'packed_logits':z.detach().cpu(),
        'labels':selection.labels.cpu(),'samples':selection.samples.cpu(),'positions':selection.positions.cpu()})
    original_saved=torch.tensor([previous['endpoint_target_logprobs'][2*b+ep][j] for b,offset in enumerate(offsets) for j in offset for ep in (0,1)],device='cuda')
    logp=z.float().log_softmax(-1).gather(1,selection.labels.repeat_interleave(2)[:,None]).squeeze(-1)
    r['packed_head_target_logprobs_vs_original_root']=metric(original_saved,logp)
    r['original_root_answer_logprobs']=original_saved.cpu().tolist();r['replayed_answer_logprobs']=logp.detach().cpu().tolist()
    root_delta=selection.sample_sums(original_saved[1::2].double()-original_saved[0::2].double())
    replay_delta=selection.sample_sums(logp[1::2].double()-logp[0::2].double())
    r['original_root_answer_delta']=root_delta.tolist();r['head_replay_delta']=replay_delta.detach().tolist()
    r['head_norm_backwards']+=1
    gradient,=timed('original_answer_head_norm_backward',lambda:torch.autograd.grad(logp[1::2].sum(),x))
    native_gradient=gradient[1::2].detach();z=z.detach();yn=y.detach();del y,x,logp,gradient
    r['finite_answer_calls']+=1
    mnorm,head_diag=finite_timed('finite_answer_seed_actual',lambda:answer_ops(z,head,selection))
    xnorm=root['final_norm_input'].to('cuda');zero=torch.zeros_like(mnorm)
    m=finite_timed('finite_final_RMS_actual',lambda:boundaries.norm_residual(xnorm[0::2],xnorm[1::2],norm.weight,mnorm,zero,norm.eps))
    r['head_and_norm_finite_effect']=paired_effect(m,xnorm).tolist()
    r['finite_answer_calls']+=1
    eqnorm,eqdiag=finite_timed('finite_answer_seed_equal',lambda:answer_ops(z,head,selection,True))
    eq=finite_timed('finite_final_RMS_equal',lambda:boundaries.norm_residual(xnorm[1::2],xnorm[1::2],norm.weight,eqnorm,zero,norm.eps))
    r['head_norm_equal_vs_native']=metric(native_gradient,eq)
    persist('answer_seed_review',{'finite':m,'input0':xnorm[0::2],'input1':xnorm[1::2],
        'equal':eq,'native_gradient':native_gradient,'logp0':head_diag['logp0'],'logp1':head_diag['logp1'],
        'logit_effect':head_diag['allocated_logit_effect']})
    archive('finite_final_layer_seed',m.cpu())
    del z,yn,mnorm,head_diag,xnorm,zero,eqnorm,eqdiag,eq,native_gradient
    head.to('meta');norm.to('meta')
    mask=(torch.arange(T,device='cuda')[None,:]<torch.tensor([605,605,368,368],device='cuda')[:,None]).long()
    pos=torch.arange(T,device='cuda').view(1,1,T).expand(4,B,T)
    rotary=native.Qwen3_5TextRotaryEmbedding(tc,device='cuda');assert rotary.inv_freq.dtype==rotary_dtype
    cos,sin=rotary(root['0'].to('cuda'),pos[1:]);assert cos.shape==(4,605,64)
    layout=RightPaddedLengths([605,368],T,'cuda')
    finite_fa=VendorFAFiniteP1BF16D256(Path(p['build_directory'])/'libdeltatrace_fa_finite_bf16_d256.so',p['library_sha256'])
    finite_fla=make_compiled_finite_pullback(reuse_scalar_products=False)
    for i in reversed(range(32)):
        layer=layers[i];is_fa=layer.block_type=='full_attention';r['decoder_loads']+=1
        timed(f'load_original_decoder{i}',lambda:load_module(layer,f'model.language_model.layers.{i}.'))
        row={'block_type':layer.block_type};r['layers'][str(i)]=row;save()
        if str(i) in p['saved_decoders']:
            artifact=p['saved_decoders'][str(i)];file=Path(p['decoder_parent_directory'])/artifact['file']
            assert sha(file.read_bytes())==artifact['sha256'];cache=torch.load(file,map_location='cpu',weights_only=True)
            def to_gpu():return tuple({k:None if v is None else v.to('cuda').contiguous() for k,v in cache[key].items()} for key in ('decoder','mixer','endpoints'))
            d,c,e=timed(f'load_saved_decoder{i}_capture',to_gpu);scale=cache['scale'];row['capture_source']='saved_original_decoder';r['decoder_capture_reuses']+=1
            assert torch.equal(cache['mask'],mask.cpu()) and torch.equal(cache['cos'],cos.cpu()) and torch.equal(cache['sin'],sin.cpu())
            del cache
        else:
            x=root[str(i)].to('cuda');masker=native.create_causal_mask if is_fa else native.create_recurrent_attention_mask
            layer_mask=masker(config=tc,inputs_embeds=x,attention_mask=mask,past_key_values=None,position_ids=pos[0]);assert torch.equal(layer_mask,mask)
            dc=NativeDecoderCapture(layer,destination='cuda')
            mc=NativeAttentionCapture(layer.self_attn,flash_attention_forward,flash_attn_varlen_func,destination='cuda') if is_fa else NativeGDNCapture(layer.linear_attn,device='cuda')
            def replay():
                with torch.no_grad(),dc,mc:return layer(x,position_embeddings=(cos,sin),attention_mask=layer_mask,position_ids=pos[0],past_key_values=None,use_cache=False)
            r['decoder_replays']+=1;y=timed(f'original_decoder{i}_replay_with_GPU_capture',replay)
            assert dc.calls=={k:1 for k in ('input_norm','post_norm','gate','up','silu','down','mlp','decoder')}
            assert mc.calls==({'module':1,'interface':1,'native_varlen':1} if is_fa else {'module':1,'conv':1,'FLA':1,'stage':1})
            d=dc.values;c=mc.values;e=getattr(mc,'endpoints',{});scale=getattr(mc,'scale',0.0625)
            row.update(capture_source='new_original_decoder',decoder_calls=dc.calls,mixer_calls=mc.calls)
            args=getattr(mc,'packed_arguments',None);del dc,mc,x,y
        assert torch.equal(d['input_norm_input'].cpu(),root[str(i)])
        expected=root['final_norm_input' if i==31 else str(i+1)].to('cuda')
        row['replay_vs_root']=metric(expected,d['output'])
        row['root_output_effect']=paired_effect(m,expected).tolist();row['replay_output_effect']=paired_effect(m,d['output']).tolist();del expected
        lse=None
        if is_fa:
            if i==3:
                a=p['saved_LSE'];file=Path(p['decoder_parent_directory'])/a['file'];assert sha(file.read_bytes())==a['sha256']
                lse=torch.load(file,map_location='cuda',weights_only=True)['LSE'];row['LSE_source']='saved_public_auxiliary_FA'
            else:
                kw={k:v for k,v in args.items() if k!='return_attn_probs'};r['auxiliary_FA_calls']+=1
                def aux():return flash_attn_varlen_func(c['packed_q'],c['packed_k'],c['packed_v'],c['packed_cu_seqlens_q'],c['packed_cu_seqlens_k'],return_attn_probs=True,**kw)
                with torch.no_grad():out,raw_lse,unused=timed(f'decoder{i}_public_FA_auxiliary_LSE',aux)
                assert unused is None or unused.numel()==0
                indices=mask.flatten().nonzero().flatten();assert raw_lse.shape==(16,int(mask.sum()))
                lse=pad_input(raw_lse.T,indices,B,T).transpose(1,2).contiguous()
                row['public_auxiliary_vs_default']=metric(c['attention_output'],pad_input(out,indices,B,T));row['LSE_source']='new_public_auxiliary_FA'
                del out,raw_lse,unused
        def mixer(upstream):
            if is_fa:return attention_finite_pullback(layer.self_attn,c,lse,cos,sin,upstream,finite_fa,layout,boundaries,False)
            return gdn_finite_pullback(layer.linear_attn,c,e,upstream,scale,finite_fla,False)
        r['finite_decoder_calls']+=1
        new_m,_=finite_timed(f'finite_decoder{i}',lambda:decoder_finite_pullback(layer,d,m,mixer,boundaries,False))
        assert torch.isfinite(new_m).all();row['input_effect']=paired_effect(new_m,d['input_norm_input']).tolist()
        row['padding_max_abs']=float(new_m[1,368:].abs().max());assert row['padding_max_abs']==0
        archive(f'layer{i}_input_coefficients',new_m.cpu())
        m=new_m;del new_m,d,c,e,lse;layer.to('meta');save()
    embed=root['0'].to('cuda');signed=(m.double()*(embed[1::2].double()-embed[0::2].double())).sum(-1)
    r['signed_sums']=signed.sum(1).tolist();r['unassigned_root_effect']=(root_delta-signed.sum(1)).tolist()
    for b,row in enumerate(spans['cases']):
        allowed=set(row['mapping']['eligible_input_positions']);changed=(embed[2*b+1]!=embed[2*b]).any(-1).nonzero().flatten().tolist();assert set(changed)==allowed
        assert all(float(signed[b,j])==0 for j in range(T) if j not in allowed)
    persist('whole_input_review',{'finite':m,'input0':embed[0::2],'input1':embed[1::2],'signed':signed})
    r['whole_model_attributions']=1;r['status']='one_author_answer_32_layer_finite_propagation_executed'
    assert r['decoder_replays']==30 and r['decoder_capture_reuses']==2 and r['finite_decoder_calls']==32 and r['auxiliary_FA_calls']==7
    r['compiler_benchmark_observations']=dict(benchmark_counts);r['compiler_counters']={str(k):dict(v) for k,v in torch._dynamo.utils.counters.items()}
    r['compiler_generated_sources']=[]
    for f in Path(p['boundary_compiler_cache']).rglob('*.py'):
        r['compiler_generated_sources'].append({'file':str(f.relative_to(Path(p['boundary_compiler_cache']))),'sha256':sha(f.read_bytes()),'bytes':f.stat().st_size})
    r['sources_after']=sources();assert r['sources_after']==r['sources_before']
    assert sha(Path(fa.__file__).read_bytes())==p['installed_FA_interface_sha256']
    r['weight_stats_after']=stats();assert r['weight_stats_after']==r['weight_stats_before']
except Exception:
    sys.setprofile(None);r['status']='failed';r['error']=traceback.format_exc()
finally:
    r['job_seconds']=time.perf_counter()-start;save()
    with zipfile.ZipFile(HERE/'review_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in list(p['files_sha256'])+['protocol.json','results.json']+[a['file'] for a in r['artifacts'] if a['download']]:
            if (HERE/name).exists():z.write(HERE/name,name)
    print(json.dumps({'status':r['status'],'finite_decoder_calls':r['finite_decoder_calls'],'error':r.get('error')}),flush=True)

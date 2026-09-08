"""One complete corrected native-content FT0-3 pass; original trajectories B2."""
import os
os.environ.update(MACA_PATH='/opt/maca',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1')
import ast,hashlib,importlib.util,io,json,sys,time,traceback,zipfile,types
from pathlib import Path
HERE=Path(__file__).resolve().parent;p=json.loads((HERE/'protocol.json').read_bytes());sha=lambda b:hashlib.sha256(b).hexdigest()
os.environ['TRITON_CACHE_DIR']=p['compiler_cache']
r={'status':'running','protocol':p,'full_model_loads':0,'root_forwards':0,'generation_calls':0,'quality_queries':0,
    'meta_model_constructions':0,'decoder_loads':0,'decoder_replays':0,'decoder_capture_reuses':0,'cached_parameter_reuses':0,
    'content_calls':0,'aggregate_calls':0,'whole_FT_attributions':0,'calls':[],'layers':{},'hops':{},'artifacts':[]}
start=time.perf_counter()
def save():
    f=HERE/'results.partial';f.write_text(json.dumps(r,indent=2));f.replace(HERE/'results.json')
def timed(label,fn):
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();base=torch.cuda.memory_allocated();tick=time.perf_counter()
    value=fn();torch.cuda.synchronize();r['calls'].append({'kind':label,'seconds':time.perf_counter()-tick,
        'before_bytes':base,'peak_bytes':torch.cuda.max_memory_allocated()});save();return value
def metric(a,b):
    a=a.detach().to(device='cpu',dtype=torch.float64).numpy().ravel();b=b.detach().to(device='cpu',dtype=torch.float64).numpy().ravel()
    assert a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    na=float(np.linalg.norm(a));nb=float(np.linalg.norm(b))
    return {'relative_L2':float(np.linalg.norm(b-a))/na if na else None,'max_abs':float(np.abs(b-a).max()),'cosine':float(a@b)/(na*nb) if na*nb else None}
def archive(name,value):
    f=HERE/(name+'.pt');torch.save(value,f)
    record={'file':f.name,'bytes':f.stat().st_size,'sha256':sha(f.read_bytes()),'download':False};r['artifacts'].append(record);save()
def persist(name,values):
    arrays={k:(v.detach().float().cpu().numpy() if v.dtype==torch.bfloat16 else v.detach().cpu().numpy()) for k,v in values.items()}
    f=HERE/(name+'.npz');np.savez_compressed(f,**arrays)
    r['artifacts'].append({'file':f.name,'bytes':f.stat().st_size,'sha256':sha(f.read_bytes()),'download':True});save()
try:
    for name,digest in p['files_sha256'].items():assert sha((HERE/name).read_bytes())==digest
    parent=Path(p['parent_directory']);raw=(parent/'results.json').read_bytes();assert sha(raw)==p['parent_sha256'];previous=json.loads(raw)
    fn=next(n for n in ast.parse((parent/'study.py').read_bytes()).body if isinstance(n,ast.FunctionDef) and n.name=='sources')
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'<native-source-guard>','exec'))
    r['sources_before']=sources();assert r['sources_before']['sha256']==p['source_tree_sha256']
    import numpy as np
    import torch
    from safetensors import safe_open
    from transformers import AutoConfig
    from transformers.utils import ContextManagers
    from transformers.core_model_loading import _materialize_copy
    from transformers.models.qwen3_5 import modeling_qwen3_5 as native
    from transformers.integrations.flash_attention import flash_attention_forward
    from flash_attn import flash_attn_varlen_func
    import flash_attn.flash_attn_interface as fa
    from native_attention_capture import NativeAttentionCapture
    from qwen35_gdn_finite import NativeGDNCapture
    from qwen35_decoder_finite import NativeDecoderCapture
    from finite_fla_gpu import verify_native_sources
    from qwen35_ft_native_content import RightPaddedBatch,fa_content_components,gdn_content_components,aggregate_native_content_ft
    from qwen35_ft_hops import FTHopBatch
    assert sha(Path(native.__file__).read_bytes())==p['native_model_sha256'];verify_native_sources(p['native_stage_source_sha256'])
    assert sha(Path(fa.__file__).read_bytes())==p['installed_FA_interface_sha256']
    cp=Path(p['checkpoint'])
    for name,digest in p['checkpoint_config_tokenizer_sha256'].items():assert sha((cp/name).read_bytes())==digest
    stats=lambda:{x.name:[x.stat().st_size,x.stat().st_mtime_ns] for x in cp.glob('*.safetensors')}
    r['weight_stats_before']=stats();assert r['weight_stats_before']==previous['weight_stats_before']
    rf=parent/'native_root_checkpoints.pt';assert sha(rf.read_bytes())==p['root_checkpoints_sha256']
    root=torch.load(rf,map_location='cpu',weights_only=True);B,T,C=root['0'].shape;assert (B,T,C)==(4,605,4096)
    sf=Path(p['spans_directory'])/'results.json';assert sha(sf.read_bytes())==p['spans_sha256'];spans=json.loads(sf.read_bytes())
    # Attribution only consumes lengths/spans, never needle labels or deletion curves.
    lengths=[x['input_metadata']['input_length'] for x in spans['cases']]
    absolute=lambda x,key:tuple(j+x['input_metadata']['prompt_length'] for j in x['mapping'][key])
    sinks=[absolute(x,'new_sink_span') for x in spans['cases']];thinking=[absolute(x,'new_thinking_span') for x in spans['cases']]
    assert lengths==[605,368] and sinks==[(566,603),(346,366)] and thinking==[(357,565),(227,345)]
    cfg=AutoConfig.from_pretrained(cp,local_files_only=True,trust_remote_code=False);cfg._attn_implementation='flash_attention_2'
    cls=native.Qwen3_5ForConditionalGeneration;r['meta_model_constructions']+=1
    with ContextManagers(cls.get_init_context(torch.bfloat16,False,False,False)):container=cls(cfg)
    assert container._get_dtype_plan(torch.bfloat16)=={};tc=cfg.text_config;layers=container.model.language_model.layers
    index=json.loads((cp/'model.safetensors.index.json').read_bytes())['weight_map'];r['weight_tensor_receipts']={}
    reference_raw=(Path(p['finite32_directory'])/'results.json').read_bytes();assert sha(reference_raw)==p['finite32_sha256']
    receipts=json.loads(reference_raw)['weight_tensor_receipts']
    def load_module(module,prefix):
        desired={k:v.dtype for k,v in module.state_dict().items()};assert set(desired.values())=={torch.bfloat16}
        names={k:v for k,v in index.items() if k.startswith(prefix)};assert {k.removeprefix(prefix) for k in names}==set(desired);state={}
        for shard in sorted(set(names.values())):
            with safe_open(cp/shard,framework='pt',device='cpu') as sf:
                for name in names:
                    if names[name]!=shard:continue
                    key=name.removeprefix(prefix);value=_materialize_copy(sf.get_slice(name),device='cpu',dtype=desired[key]);state[key]=value
                    record={'loaded_dtype':str(value.dtype),'shape':list(value.shape),'loaded_sha256':sha(value.view(torch.uint8).numpy().tobytes()),'shard':shard}
                    assert record==receipts[name];r['weight_tensor_receipts'][name]=record
        module.load_state_dict(state,strict=True,assign=True);module.eval().requires_grad_(False).to('cuda')
    spec=importlib.util.spec_from_file_location('pinned_official_ft_core',HERE/'official_ft_core.py')
    author=importlib.util.module_from_spec(spec);sys.modules[spec.name]=author;spec.loader.exec_module(author)
    controller=FTHopBatch(lengths,sinks,thinking);layout=RightPaddedBatch(lengths,T,'cuda')
    mask=(torch.arange(T,device='cuda')[None,:]<torch.tensor([605,605,368,368],device='cuda')[:,None]).long()
    pos=torch.arange(T,device='cuda').view(1,1,T).expand(4,B,T)
    rotary=native.Qwen3_5TextRotaryEmbedding(tc,device='cuda')
    assert rotary.inv_freq.dtype==container.model.language_model.rotary_emb.inv_freq.dtype
    cos,sin=rotary(root['0'].to('cuda'),pos[1:]);assert cos.shape==(4,605,64)
    cache_cpu={};all_totals=[];all_observations=[];all_weights=[];all_layer_scores=[];all_ratios=[]
    for hop in range(4):
        weights=controller.weights.to('cuda');total=torch.zeros((2,T),dtype=torch.float32);layer_scores=[];hop_layers={};r['hops'][str(hop)]={'layers':hop_layers};save()
        for i in range(32):
            layer=layers[i];is_fa=layer.block_type=='full_attention';row={'native_node_events':[]};hop_layers[str(i)]=row
            if hop==0:
                r['decoder_loads']+=1;timed(f'load_original_decoder{i}',lambda:load_module(layer,f'model.language_model.layers.{i}.'))
                if str(i) in p['saved_decoders']:
                    artifact=p['saved_decoders'][str(i)];file=Path(p['decoder_parent_directory'])/artifact['file'];assert sha(file.read_bytes())==artifact['sha256']
                    old=torch.load(file,map_location='cpu',weights_only=True);dcv,mcv,ep=old['decoder'],old['mixer'],old['endpoints'];scale=old['scale']
                    assert torch.equal(old['mask'],mask.cpu());r['decoder_capture_reuses']+=1;row['capture']='saved_original_decoder'
                else:
                    x=root[str(i)].to('cuda');masker=native.create_causal_mask if is_fa else native.create_recurrent_attention_mask
                    layer_mask=masker(config=tc,inputs_embeds=x,attention_mask=mask,past_key_values=None,position_ids=pos[0]);assert torch.equal(layer_mask,mask)
                    dc=NativeDecoderCapture(layer,destination='cuda')
                    mc=NativeAttentionCapture(layer.self_attn,flash_attention_forward,flash_attn_varlen_func,destination='cuda') if is_fa else NativeGDNCapture(layer.linear_attn,device='cuda')
                    def replay():
                        with torch.no_grad(),dc,mc:return layer(x,position_embeddings=(cos,sin),attention_mask=layer_mask,position_ids=pos[0],past_key_values=None,use_cache=False)
                    r['decoder_replays']+=1;y=timed(f'original_decoder{i}_replay_with_capture',replay)
                    assert dc.calls=={k:1 for k in ('input_norm','post_norm','gate','up','silu','down','mlp','decoder')}
                    assert mc.calls==({'module':1,'interface':1,'native_varlen':1} if is_fa else {'module':1,'conv':1,'FLA':1,'stage':1})
                    dcv,mcv,ep=dc.values,mc.values,getattr(mc,'endpoints',{});scale=getattr(mc,'scale',0.0625)
                    row.update(capture='new_original_decoder',decoder_calls=dc.calls,mixer_calls=mc.calls);del x,y,dc,mc
                assert torch.equal(dcv['input_norm_input'].cpu(),root[str(i)])
                row['replay_vs_root']=metric(root['final_norm_input' if i==31 else str(i+1)],dcv['output'])
                take=lambda v:v[1::2].detach().to('cpu',copy=True).contiguous()
                d={k:take(dcv[k]) for k in ['input_norm_input','post_norm_input']}
                keys=['query','key','value','q_proj_output','attention_output','o_proj_input','output'] if is_fa else ['projected_qkv','conv_output','norm_output','z','output']
                c={k:take(mcv[k]) for k in keys};e={k:take(ep[k]) for k in ['q','k','v','raw_g','beta','o']} if not is_fa else {}
                module=layer.self_attn if is_fa else layer.linear_attn
                params={'o_proj.weight':module.o_proj.weight.detach().cpu().clone()} if is_fa else {k:module.state_dict()[k].cpu().clone() for k in ['out_proj.weight','norm.weight','conv1d.weight']}
                cached={'decoder':d,'mixer':c,'endpoints':e,'parameters':params,'scale':scale};cache_cpu[i]=cached
                timed(f'archive_layer{i}_FT_actual_inputs',lambda:archive(f'layer{i}_FT_actual_inputs',cached))
                del dcv,mcv,ep
                if str(i) in p['saved_decoders']:del old
            else:
                cached=cache_cpu[i];scale=cached['scale'];module=layer.self_attn if is_fa else layer.linear_attn
                def reload():
                    for path,value in cached['parameters'].items():
                        name,key=path.rsplit('.',1);setattr(module.get_submodule(name),key,torch.nn.Parameter(value.to('cuda'),requires_grad=False))
                r['cached_parameter_reuses']+=1;timed(f'hop{hop}_layer{i}_cached_parameter_to_GPU',reload)
            def to_gpu():return tuple({k:v.to('cuda') for k,v in cached[key].items()} for key in ['decoder','mixer','endpoints'])
            d,c,e=timed(f'hop{hop}_layer{i}_cached_operands_to_GPU',to_gpu)
            local=lambda label,fn:timed(f'hop{hop}_layer{i}_'+label,fn)
            with torch.no_grad():
                r['content_calls']+=1
                if is_fa:
                    components,diagnostic=fa_content_components(module,c,weights,layout,local,row['native_node_events'])
                    head_native=c['o_proj_input'].reshape(2,T,16,256);out_weight=module.o_proj.weight
                else:
                    components,diagnostic=gdn_content_components(module,c,e,weights,scale,local,row['native_node_events'])
                    head_native=c['norm_output'].reshape(2,T,32,128);out_weight=module.out_proj.weight
                assert torch.isfinite(components).all() and float(components[1,368:].abs().max())==0
                head_sum=(head_native.double()*weights[:,:,None,None]).sum(1)
                row['head_reconstruction']=metric(head_sum,components.double().sum(1))
                r['aggregate_calls']+=1
                agg=local('FT_projection_and_proximity_with_diagnostics',lambda:aggregate_native_content_ft(components,out_weight,d,weights,author.proximity,chunk_tokens=32))
                row['projected_reconstruction']=metric((c['output'].double()*weights[:,:,None]).sum(1),agg['projected_head_sums'].sum(1))
                scores=agg['token_scores'].cpu();assert torch.isfinite(scores).all() and torch.count_nonzero(scores[1,368:])==0
                layer_scores.append(scores);total+=scores;row['normalization_sum']=(scores.sum(1)+agg['residual_score'].cpu()).tolist();row['status']='complete'
            del d,c,e,components,diagnostic,agg,head_native,head_sum,out_weight;layer.to('meta');save()
        result=controller.consume(total);all_totals.append(total);all_observations.append(result['observation_sum']);all_weights.append(result['input_weights'])
        all_layer_scores.append(torch.stack(layer_scores));all_ratios.append(result['ratio_after']);r['hops'][str(hop)].update(ratio_before=result['ratio_before'],ratio_after=result['ratio_after'],status='complete')
        persist(f'hop{hop}_FT_scores',{'layer_token_scores':torch.stack(layer_scores),'token_total':total,'weights':result['input_weights'],
            'next_weights':result['next_weights'],'observation_sum':result['observation_sum']});r['whole_FT_attributions']+=1;save()
    # Replay the UNCHANGED author hop controller over already computed per-hop
    # aggregates. This is a CPU bookkeeping reference, never a model/backend or
    # another attribution call. Only its aggregate data provider is injected.
    reference=[]
    for b,n in enumerate(lengths):
        calls=[]
        def supply(**kwargs):
            hop=len(calls);expected=sinks[b] if hop==0 else thinking[b]
            assert (kwargs['sink_start'],kwargs['sink_end'])==expected
            if hop==0:assert kwargs.get('sink_weights') is None
            else:
                w=kwargs['sink_weights'];w=w/(w.sum()+1e-12)
                assert torch.equal(w.float(),all_weights[hop][b,thinking[b][0]:thinking[b][1]+1])
            calls.append(hop);return types.SimpleNamespace(token_importance_total=all_totals[hop][b,:n])
        function=author.compute_multi_hop_ifr.__wrapped__
        scope=dict(function.__globals__);scope['compute_ifr_sentence_aggregate']=supply
        replay=types.FunctionType(function.__code__,scope,function.__name__,function.__defaults__,function.__closure__)
        output=replay(sink_start=sinks[b][0],sink_end=sinks[b][1],thinking_span=thinking[b],n_hops=3,
            cache={},attentions=None,weight_pack=[],params=types.SimpleNamespace(model_dtype=torch.bfloat16))
        assert calls==[0,1,2,3]
        assert torch.equal(output.observation['sum'],all_observations[3][b,:n])
        assert output.thinking_ratios==[x[b] for x in all_ratios]
        reference.append({'sample':b,'aggregate_references':4,'observation_exact':True,'ratios_exact':True,'normalized_weights_exact':True})
    r['author_hop_controller_reference']=reference
    r['source_cache_bytes']=sum(a['bytes'] for a in r['artifacts'] if a['file'].endswith('.pt'))
    r['sources_after']=sources();assert r['sources_after']==r['sources_before']
    assert sha(Path(fa.__file__).read_bytes())==p['installed_FA_interface_sha256']
    r['weight_stats_after']=stats();assert r['weight_stats_after']==r['weight_stats_before']
    assert r['decoder_loads']==32 and r['decoder_replays']==30 and r['decoder_capture_reuses']==2
    assert r['content_calls']==r['aggregate_calls']==128 and r['cached_parameter_reuses']==96
    r['status']='same_checkpoint_corrected_FT32_hops0_to3_executed'
except Exception:
    sys.setprofile(None);r['status']='failed';r['error']=traceback.format_exc()
finally:
    r['job_seconds_before_bundle']=time.perf_counter()-start;save()
    with zipfile.ZipFile(HERE/'review_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in list(p['files_sha256'])+['protocol.json','results.json']+[a['file'] for a in r['artifacts'] if a['download']]:
            if (HERE/name).exists():z.write(HERE/name,name)
    print(json.dumps({'status':r['status'],'error':r.get('error')}),flush=True)

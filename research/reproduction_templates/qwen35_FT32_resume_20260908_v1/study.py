"""Resume FT from immutable native inputs; restore author's causal source bound."""
import os
os.environ.update(MACA_PATH='/opt/maca',HF_HUB_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1')
import ast,hashlib,importlib.util,io,json,sys,time,traceback,zipfile,types
from pathlib import Path
HERE=Path(__file__).resolve().parent;p=json.loads((HERE/'protocol.json').read_bytes());sha=lambda b:hashlib.sha256(b).hexdigest()
os.environ['TRITON_CACHE_DIR']=p['compiler_cache']
r={'status':'running','protocol':p,'full_model_loads':0,'root_forwards':0,'generation_calls':0,'quality_queries':0,'decoder_replays':0,
    'meta_model_constructions':0,'content_calls':0,'aggregate_calls':0,'parameter_cache_loads':0,'completed_hop_outputs':0,'calls':[],'hops':{},'artifacts':[]}
start=time.perf_counter()
def save():
    f=HERE/'results.partial';f.write_text(json.dumps(r,indent=2));f.replace(HERE/'results.json')
def timed(label,fn):
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();base=torch.cuda.memory_allocated();tick=time.perf_counter()
    value=fn();torch.cuda.synchronize();r['calls'].append({'kind':label,'seconds':time.perf_counter()-tick,'before_bytes':base,'peak_bytes':torch.cuda.max_memory_allocated()});save();return value
def metric(a,b):
    a=a.detach().to(device='cpu',dtype=torch.float64).numpy().ravel();b=b.detach().to(device='cpu',dtype=torch.float64).numpy().ravel()
    assert a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    na=float(np.linalg.norm(a));nb=float(np.linalg.norm(b))
    return {'relative_L2':float(np.linalg.norm(b-a))/na if na else None,'max_abs':float(np.abs(b-a).max()),'cosine':float(a@b)/(na*nb) if na*nb else None}
def persist(name,values):
    arrays={k:(v.detach().float().cpu().numpy() if v.dtype==torch.bfloat16 else v.detach().cpu().numpy()) for k,v in values.items()}
    f=HERE/(name+'.npz');np.savez_compressed(f,**arrays)
    r['artifacts'].append({'file':f.name,'bytes':f.stat().st_size,'sha256':sha(f.read_bytes()),'download':True});save()
try:
    for name,digest in p['files_sha256'].items():assert sha((HERE/name).read_bytes())==digest
    parent=Path(p['resume_parent']);raw=(parent/'results.json').read_bytes();assert sha(raw)==p['resume_parent_sha256'];previous=json.loads(raw)
    assert previous['status']=='failed' and previous['decoder_replays']==30 and previous['content_calls']==34
    assert not Path('/proc/'+(parent/'pid').read_text()).exists()
    origin=Path(p['parent_directory']);raw=(origin/'results.json').read_bytes();assert sha(raw)==p['parent_sha256'];root_previous=json.loads(raw)
    fn=next(n for n in ast.parse((origin/'study.py').read_bytes()).body if isinstance(n,ast.FunctionDef) and n.name=='sources')
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'<native-source-guard>','exec'))
    r['sources_before']=sources();assert r['sources_before']['sha256']==p['source_tree_sha256']
    import numpy as np
    import torch
    from transformers import AutoConfig
    from transformers.utils import ContextManagers
    from transformers.models.qwen3_5 import modeling_qwen3_5 as native
    import flash_attn.flash_attn_interface as fa
    from finite_fla_gpu import verify_native_sources
    from qwen35_ft_native_content import RightPaddedBatch,fa_content_components,gdn_content_components,aggregate_native_content_ft
    from qwen35_ft_hops import FTHopBatch
    assert sha(Path(native.__file__).read_bytes())==p['native_model_sha256'];verify_native_sources(p['native_stage_source_sha256'])
    assert sha(Path(fa.__file__).read_bytes())==p['installed_FA_interface_sha256']
    cp=Path(p['checkpoint'])
    for name,digest in p['checkpoint_config_tokenizer_sha256'].items():assert sha((cp/name).read_bytes())==digest
    stats=lambda:{x.name:[x.stat().st_size,x.stat().st_mtime_ns] for x in cp.glob('*.safetensors')}
    r['weight_stats_before']=stats();assert r['weight_stats_before']==root_previous['weight_stats_before']
    cfg=AutoConfig.from_pretrained(cp,local_files_only=True,trust_remote_code=False);cfg._attn_implementation='flash_attention_2'
    cls=native.Qwen3_5ForConditionalGeneration;r['meta_model_constructions']+=1
    with ContextManagers(cls.get_init_context(torch.bfloat16,False,False,False)):container=cls(cfg)
    assert container._get_dtype_plan(torch.bfloat16)=={};layers=container.model.language_model.layers
    spec=importlib.util.spec_from_file_location('pinned_official_ft_core',HERE/'official_ft_core.py');author=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=author;spec.loader.exec_module(author)
    byname={x['file']:x for x in previous['artifacts']};cache={};r['input_cache_receipts']={}
    def load_actual(i):
        name=f'layer{i}_FT_actual_inputs.pt';f=parent/name;assert sha(f.read_bytes())==byname[name]['sha256'];value=torch.load(f,map_location='cpu',weights_only=True)
        for path,tensor in value['parameters'].items():
            full=f'model.language_model.layers.{i}.'+('self_attn.' if i%4==3 else 'linear_attn.')+path
            assert sha(tensor.view(torch.uint8).numpy().tobytes())==previous['weight_tensor_receipts'][full]['loaded_sha256']
        cache[i]=value;r['input_cache_receipts'][str(i)]=byname[name];return value
    oldfile=parent/'hop0_FT_scores.npz';assert sha(oldfile.read_bytes())==byname[oldfile.name]['sha256'];old=dict(np.load(oldfile,allow_pickle=False))
    old_scores=torch.from_numpy(old['layer_token_scores']);assert old_scores.shape==(32,2,605)
    offenders=[i for i in range(32) if any(torch.count_nonzero(old_scores[i,b,end:]) for b,end in enumerate([604,367]))]
    assert offenders==[31];r['FT0_reused_layers']=[i for i in range(32) if i!=31]
    r['old_FT0_forbidden_score']=float(old_scores[31,0,604]);r['old_FT0_sha256']=byname[oldfile.name]['sha256']
    lengths=[605,368];sinks=[(566,603),(346,366)];thinking=[(357,565),(227,345)]
    controller=FTHopBatch(lengths,sinks,thinking);layout=RightPaddedBatch(lengths,605,'cuda');all_totals=[];all_observations=[];all_weights=[];all_ratios=[]
    for hop in range(4):
        weights=controller.weights.to('cuda');limits=[e+1 for _,e in (sinks if hop==0 else thinking)]
        scores_by_layer=old_scores.clone() if hop==0 else torch.zeros((32,2,605),dtype=torch.float32)
        rows={};r['hops'][str(hop)]={'layers':rows};save()
        for i in ([31] if hop==0 else range(32)):
            cached=cache.get(i)
            if cached is None:cached=timed(f'load_saved_actual_layer{i}',lambda:load_actual(i))
            layer=layers[i];is_fa=i%4==3;module=layer.self_attn if is_fa else layer.linear_attn
            row={'native_node_events':[]};rows[str(i)]=row
            def restore():
                for path,value in cached['parameters'].items():
                    name,key=path.rsplit('.',1);setattr(module.get_submodule(name),key,torch.nn.Parameter(value.to('cuda'),requires_grad=False))
                return tuple({k:v.to('cuda') for k,v in cached[name].items()} for name in ['decoder','mixer','endpoints'])
            r['parameter_cache_loads']+=1;d,c,e=timed(f'hop{hop}_layer{i}_cached_inputs_to_GPU',restore)
            local=lambda label,fn:timed(f'hop{hop}_layer{i}_'+label,fn)
            with torch.no_grad():
                r['content_calls']+=1
                if is_fa:
                    components,diagnostic=fa_content_components(module,c,weights,layout,local,row['native_node_events'])
                    head_native=c['o_proj_input'].reshape(2,605,16,256);out_weight=module.o_proj.weight
                else:
                    components,diagnostic=gdn_content_components(module,c,e,weights,cached['scale'],local,row['native_node_events'])
                    head_native=c['norm_output'].reshape(2,605,32,128);out_weight=module.out_proj.weight
                assert torch.isfinite(components).all()
                row['forbidden_components_max_abs']=[float(components[b,end:].abs().max()) for b,end in enumerate(limits)]
                assert row['forbidden_components_max_abs']==[0.0,0.0]
                row['head_reconstruction']=metric((head_native.double()*weights[:,:,None,None]).sum(1),components.double().sum(1))
                diagnostics=(hop,i) in [(0,31),(1,1)];r['aggregate_calls']+=1
                agg=local('FT_projection_and_causal_proximity',lambda:aggregate_native_content_ft(components,out_weight,d,weights,author.proximity,
                    chunk_tokens=32,source_limits=limits,range_diagnostics=diagnostics))
                row['projected_reconstruction']=metric((c['output'].double()*weights[:,:,None]).sum(1),agg['projected_head_sums'].sum(1))
                scores=agg['token_scores'].cpu();assert torch.isfinite(scores).all()
                assert all(torch.count_nonzero(scores[b,end:])==0 for b,end in enumerate(limits))
                if diagnostics:
                    unmasked=agg['unmasked_token_scores'].cpu();assert torch.isfinite(unmasked).all()
                    row['unmasked_forbidden_score_max']=[float(unmasked[b,end:].max()) for b,end in enumerate(limits)]
                    row['unmasked_forbidden_score_mass']=[float(unmasked[b,end:].sum()) for b,end in enumerate(limits)]
                    if hop==0:row['unmasked_vs_saved_FT0']=metric(old_scores[31],unmasked)
                    persist(f'hop{hop}_layer{i}_source_range_review',{'unmasked_scores':unmasked,'bounded_scores':scores,'sink_weights':weights,'limits':torch.tensor(limits)})
                scores_by_layer[i]=scores;row['normalization_sum']=(scores.sum(1)+agg['residual_score'].cpu()).tolist();row['status']='complete'
            del d,c,e,components,diagnostic,agg,head_native,out_weight;layer.to('meta');save()
        total=torch.zeros((2,605),dtype=torch.float32)
        for i in range(32):total+=scores_by_layer[i]
        result=controller.consume(total);all_totals.append(total);all_observations.append(result['observation_sum']);all_weights.append(result['input_weights']);all_ratios.append(result['ratio_after'])
        r['hops'][str(hop)].update(ratio_before=result['ratio_before'],ratio_after=result['ratio_after'],status='complete')
        persist(f'hop{hop}_FT_scores',{'layer_token_scores':scores_by_layer,'token_total':total,'weights':result['input_weights'],
            'next_weights':result['next_weights'],'observation_sum':result['observation_sum']});r['completed_hop_outputs']+=1;save()
    reference=[]
    for b,n in enumerate(lengths):
        calls=[]
        def supply(**kwargs):
            hop=len(calls);expected=sinks[b] if hop==0 else thinking[b];assert (kwargs['sink_start'],kwargs['sink_end'])==expected
            if hop==0:assert kwargs.get('sink_weights') is None
            else:
                w=kwargs['sink_weights'];w=w/(w.sum()+1e-12);assert torch.equal(w.float(),all_weights[hop][b,thinking[b][0]:thinking[b][1]+1])
            calls.append(hop);return types.SimpleNamespace(token_importance_total=all_totals[hop][b,:n])
        function=author.compute_multi_hop_ifr.__wrapped__;scope=dict(function.__globals__);scope['compute_ifr_sentence_aggregate']=supply
        replay=types.FunctionType(function.__code__,scope,function.__name__,function.__defaults__,function.__closure__)
        output=replay(sink_start=sinks[b][0],sink_end=sinks[b][1],thinking_span=thinking[b],n_hops=3,
            cache={},attentions=None,weight_pack=[],params=types.SimpleNamespace(model_dtype=torch.bfloat16))
        assert calls==[0,1,2,3] and torch.equal(output.observation['sum'],all_observations[3][b,:n])
        assert output.thinking_ratios==[x[b] for x in all_ratios]
        reference.append({'sample':b,'aggregate_references':4,'observation_exact':True,'ratios_exact':True,'normalized_weights_exact':True})
    r['author_hop_controller_reference']=reference
    assert r['content_calls']==r['aggregate_calls']==r['parameter_cache_loads']==97 and r['completed_hop_outputs']==4
    r['sources_after']=sources();assert r['sources_after']==r['sources_before']
    assert sha(Path(fa.__file__).read_bytes())==p['installed_FA_interface_sha256']
    r['weight_stats_after']=stats();assert r['weight_stats_after']==r['weight_stats_before']
    r['status']='corrected_FT32_hops0_to3_resumed_with_author_source_bounds'
except Exception:
    sys.setprofile(None);r['status']='failed';r['error']=traceback.format_exc()
finally:
    r['job_seconds_before_bundle']=time.perf_counter()-start;save()
    with zipfile.ZipFile(HERE/'review_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in list(p['files_sha256'])+['protocol.json','results.json']+[a['file'] for a in r['artifacts']]:
            if (HERE/name).exists():z.write(HERE/name,name)
    print(json.dumps({'status':r['status'],'error':r.get('error')}),flush=True)

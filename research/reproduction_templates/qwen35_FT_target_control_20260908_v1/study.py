"""One FT whole-response target diagnostic from saved native inputs, without changing FT helpers."""
import os
os.environ.update(MACA_PATH='/opt/maca',HF_HUB_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1')
import ast,hashlib,importlib.util,io,json,sys,time,traceback,zipfile,types,math
from typing import Sequence,List
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
    lengths=[605,368];starts=[357,227]
    layout=RightPaddedBatch(lengths,605,'cuda')
    for hop in range(1):
        weights=torch.zeros((2,605),dtype=torch.float32,device='cuda')
        for b,(begin_idx,end_idx) in enumerate(zip(starts,lengths)):weights[b,begin_idx:end_idx]=1.0
        limits=lengths
        scores_by_layer=torch.zeros((32,2,605),dtype=torch.float32)
        rows={};r['hops'][str(hop)]={'layers':rows};save()
        for i in range(32):
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
                row['forbidden_components_max_abs']=[(float(components[b,end:].abs().max()) if end<components.shape[1] else 0.0) for b,end in enumerate(limits)]
                assert row['forbidden_components_max_abs']==[0.0,0.0]
                row['head_reconstruction']=metric((head_native.double()*weights[:,:,None,None]).sum(1),components.double().sum(1))
                diagnostics=False;r['aggregate_calls']+=1
                agg=local('FT_projection_and_causal_proximity',lambda:aggregate_native_content_ft(components,out_weight,d,weights,author.proximity,
                    chunk_tokens=32,source_limits=limits,range_diagnostics=diagnostics))
                row['projected_reconstruction']=metric((c['output'].double()*weights[:,:,None]).sum(1),agg['projected_head_sums'].sum(1))
                scores=agg['token_scores'].cpu();assert torch.isfinite(scores).all()
                assert all(torch.count_nonzero(scores[b,end:])==0 for b,end in enumerate(limits))
                scores_by_layer[i]=scores;row['normalization_sum']=(scores.sum(1)+agg['residual_score'].cpu()).tolist();row['status']='complete'
            del d,c,e,components,diagnostic,agg,head_native,out_weight;layer.to('meta');save()
        total=torch.zeros((2,605),dtype=torch.float32)
        for i in range(32):total+=scores_by_layer[i]
        r['hops'][str(hop)]['status']='complete'
        persist('whole_response_FT_diagnostic',{'layer_token_scores':scores_by_layer,'token_total':total,'weights':weights})
        r['completed_hop_outputs']+=1;save()
    assert r['content_calls']==r['aggregate_calls']==r['parameter_cache_loads']==32 and r['completed_hop_outputs']==1
    raw=Path(p['author_metric_source']).read_bytes().replace(b'\r\n',b'\n');assert sha(raw)==p['author_metric_source_sha256']
    helper=next(n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name=='evaluate_attr_recovery_skip_tokens')
    exec(compile(ast.Module(body=[helper],type_ignores=[]),p['author_metric_source'],'exec'))
    raw=Path(p['spans_file']).read_bytes();assert sha(raw)==p['spans_sha256'];case=json.loads(raw)['cases'][0]
    user=case['input_metadata']['author_user_positions'];keep=case['mapping']['keep_local_indices'];gold=case['mapping']['gold_user_token_indices']
    prompt=total[0,user];r['original_recovery_calls']=1
    score=evaluate_attr_recovery_skip_tokens(prompt[None,:],keep_prompt_token_indices=keep,gold_prompt_token_indices=gold,top_fraction=.1)
    values=prompt.clamp(min=0)[keep];top=torch.topk(values,math.ceil(.1*len(keep))).indices.tolist();selected=[keep[j] for j in top]
    eligible=set(gold)&set(keep);hits=sorted(set(selected)&eligible);assert score==len(hits)/len(eligible)
    r['original_NI0_recovery']={'recovery':score,'selected_user_indices':selected,'hit_user_indices':hits,
        'eligible_gold_count':len(eligible),'selected_count':len(selected),'cutoff_ties':int((values==values[top[-1]]).sum()),
        'method':'FT_whole_response_target_diagnostic_only','metric':'Unchanged original helper,top_fraction0.1; not a new official FT baseline.'}
    r['sources_after']=sources();assert r['sources_after']==r['sources_before']
    assert sha(Path(fa.__file__).read_bytes())==p['installed_FA_interface_sha256']
    r['weight_stats_after']=stats();assert r['weight_stats_after']==r['weight_stats_before']
    r['status']='one_FT_whole_response_target_diagnostic_executed'
except Exception:
    sys.setprofile(None);r['status']='failed';r['error']=traceback.format_exc()
finally:
    r['job_seconds_before_bundle']=time.perf_counter()-start;save()
    with zipfile.ZipFile(HERE/'review_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in list(p['files_sha256'])+['protocol.json','results.json']+[a['file'] for a in r['artifacts']]:
            if (HERE/name).exists():z.write(HERE/name,name)
    print(json.dumps({'status':r['status'],'error':r.get('error')}),flush=True)

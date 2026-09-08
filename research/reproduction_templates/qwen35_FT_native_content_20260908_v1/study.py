"""Two original saved mixers, native content adjoints and corrected FT boundaries."""
import os
os.environ.update(MACA_PATH='/opt/maca',HF_HUB_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1')
import ast,hashlib,importlib.util,io,json,sys,time,traceback,zipfile
from pathlib import Path
HERE=Path(__file__).resolve().parent;p=json.loads((HERE/'protocol.json').read_bytes());sha=lambda b:hashlib.sha256(b).hexdigest()
os.environ['TRITON_CACHE_DIR']=p['compiler_cache']
r={'status':'running','protocol':p,'full_model_loads':0,'root_forwards':0,'decoder_replays':0,'generation_calls':0,'quality_queries':0,
    'meta_model_constructions':0,'parameter_loads':0,'calls':[],'layers':{},'artifacts':[],'whole_FT_attributions':0}
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
def persist(name,values):
    arrays={}
    for key,value in values.items():
        value=value.detach().cpu();arrays[key]=(value.float() if value.dtype==torch.bfloat16 else value).numpy()
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
    import flash_attn.flash_attn_interface as fa
    from finite_fla_gpu import verify_native_sources
    from qwen35_ft_native_content import RightPaddedBatch,fa_content_components,gdn_content_components,aggregate_native_content_ft,project_components
    assert sha(Path(native.__file__).read_bytes())==p['native_model_sha256'];verify_native_sources(p['native_stage_source_sha256'])
    assert sha(Path(fa.__file__).read_bytes())==p['installed_FA_interface_sha256']
    cp=Path(p['checkpoint'])
    for name,digest in p['checkpoint_config_tokenizer_sha256'].items():assert sha((cp/name).read_bytes())==digest
    stats=lambda:{x.name:[x.stat().st_size,x.stat().st_mtime_ns] for x in cp.glob('*.safetensors')}
    r['weight_stats_before']=stats();assert r['weight_stats_before']==previous['weight_stats_before']
    cfg=AutoConfig.from_pretrained(cp,local_files_only=True,trust_remote_code=False);cfg._attn_implementation='flash_attention_2'
    cls=native.Qwen3_5ForConditionalGeneration;r['meta_model_constructions']+=1
    with ContextManagers(cls.get_init_context(torch.bfloat16,False,False,False)):container=cls(cfg)
    assert container._get_dtype_plan(torch.bfloat16)=={};layers=container.model.language_model.layers
    index=json.loads((cp/'model.safetensors.index.json').read_bytes())['weight_map'];r['loaded_parameters']={}
    # Materialize only original parameters consumed by this auxiliary study.
    # Other real modules remain meta and are never called as a model surrogate.
    def load_parameter(relative):
        full='model.language_model.layers.'+relative;module_name,key=relative.rsplit('.',1)
        module=layers.get_submodule(module_name);desired=module.state_dict()[key].dtype;assert desired==torch.bfloat16
        with safe_open(cp/index[full],framework='pt',device='cpu') as sf:value=_materialize_copy(sf.get_slice(full),device='cpu',dtype=desired)
        record={'shape':list(value.shape),'dtype':str(desired),'sha256':sha(value.view(torch.uint8).numpy().tobytes())}
        assert record['sha256']==p['decoder_parameter_receipts'][full]['loaded_sha256']
        setattr(module,key,torch.nn.Parameter(value.to('cuda'),requires_grad=False));r['loaded_parameters'][full]=record;r['parameter_loads']+=1;save()
    for name in ['3.self_attn.o_proj.weight','3.input_layernorm.weight','0.linear_attn.out_proj.weight','0.linear_attn.norm.weight','0.linear_attn.conv1d.weight']:
        timed('load_parameter_'+name,lambda name=name:load_parameter(name))
    core_path=HERE/'official_ft_core.py';spec=importlib.util.spec_from_file_location('pinned_official_ft_core',core_path)
    author=importlib.util.module_from_spec(spec);sys.modules[spec.name]=author;spec.loader.exec_module(author)
    layout=RightPaddedBatch([605,368],605,'cuda');mask=layout.mask
    weights=torch.zeros((2,605),device='cuda');weights[0,566:604]=1;weights[1,346:367]=1
    r['target_positions']={'scope':'Original FT sink representations; positions are prompt+generation_offset, not predictor positions.',
        'prompt_lengths':[357,227],'sink_closed':[[209,246],[119,139]],'absolute_closed':[[566,603],[346,366]],'counts':[38,21]}
    r['versions']={'torch':torch.__version__,'device':torch.cuda.get_device_name()}
    for i in (3,0):
        artifact=p['saved_decoders'][str(i)];file=Path(p['decoder_parent_directory'])/artifact['file']
        assert sha(file.read_bytes())==artifact['sha256'];cache=torch.load(file,map_location='cpu',weights_only=True)
        take=lambda value:None if value is None else value[1::2].to('cuda').contiguous()
        d={k:take(cache['decoder'][k]) for k in ('input_norm_input','input_norm_output','post_norm_input')}
        keys=('query','key','value','q_proj_output','attention_output','o_proj_input','output') if i==3 else ('projected_qkv','conv_output','norm_output','z','output')
        c={k:take(cache['mixer'][k]) for k in keys}
        e={k:take(cache['endpoints'][k]) for k in ('q','k','v','raw_g','beta','o')} if i==0 else {}
        scale=cache['scale'];assert torch.equal(cache['mask'][1::2].bool(),mask.cpu());del cache
        row={'native_node_events':[]};r['layers'][str(i)]=row;save()
        local_timed=lambda label,fn:timed(f'layer{i}_'+label,fn)
        with torch.no_grad():
            if i==3:
                layer=layers[3];module=layer.self_attn
                components,diagnostic=fa_content_components(module,c,weights,layout,local_timed,row['native_node_events'])
                head_native=c['o_proj_input'].reshape(2,605,16,256)
                source_reference=c['attention_output']
                old=author.linearize_norm(layer.input_layernorm,d['input_norm_input'])*d['input_norm_input'].float()
                row['official_raw_weight_RMS_vs_native']=metric(d['input_norm_output'],old);del old
                out_weight=module.o_proj.weight
            else:
                module=layers[0].linear_attn
                components,diagnostic=gdn_content_components(module,c,e,weights,scale,local_timed,row['native_node_events'])
                head_native=c['norm_output'].reshape(2,605,32,128);source_reference=e['o'];out_weight=module.out_proj.weight
            row['auxiliary_core_vs_saved_native']=metric(diagnostic['native_core'],diagnostic['auxiliary_core'])
            row['ungated_core_vs_native_gated_output']=metric(head_native,source_reference)
            row['frozen_output_gain_vs_native_gated_output']=metric(head_native,source_reference.float()*diagnostic['gain'])
            head_sum=(head_native.double()*weights[:,:,None,None]).sum(1);component_sum=components.double().sum(1)
            row['head_content_reconstruction']=metric(head_sum,component_sum)
            row['head_content_per_sample']=[metric(head_sum[b],component_sum[b]) for b in range(2)]
            row['component_padding_max_abs']=float(components[1,368:].abs().max());assert row['component_padding_max_abs']==0
            agg=local_timed('FT_projection_and_author_proximity_with_diagnostics',lambda:aggregate_native_content_ft(
                components,out_weight,d,weights,author.proximity,chunk_tokens=32))
            native_sum=(c['output'].double()*weights[:,:,None]).sum(1)
            row['projected_reconstruction']=metric(native_sum,agg['projected_head_sums'].sum(1))
            row['local_token_proximity_summary']={'score_sums':agg['token_scores'].sum(1).tolist(),'residual_scores':agg['residual_score'].tolist(),
                'positive_tokens':(agg['token_scores']>0).sum(1).tolist(),'scope':'Single-layer corrected FT content proximity, not full FT or quality.'}
            assert torch.isfinite(components).all() and torch.isfinite(agg['token_scores']).all()
            assert float(agg['token_scores'][1,368:].abs().max())==0
            h=components.shape[2];dim=components.shape[3];heads=[0,h-1];rows=[7,59,200,300]
            selected=components[:,rows]
            selected_projection=local_timed('extra_fixed_CPU_review_projection',lambda:project_components(selected,out_weight)[:,:,heads])
            arrays={'components':components,'head_native_sum':head_sum,'native_projected_sum':native_sum,'sink_weights':weights,
                'projected_head_sums':agg['projected_head_sums'],'token_scores':agg['token_scores'],'head_scores':agg['head_scores'],
                'residual_score':agg['residual_score'],'mid_sum':agg['mid_sum'],'numerator':agg['numerator'],
                'selected_projection':selected_projection,'selected_out_weights':out_weight.view(4096,h,dim)[:,heads].permute(1,2,0),
                'value_gradient':diagnostic['value_gradient'],'seed_native':diagnostic['seed_native']}
            if i==3:
                a=p['saved_LSE'];f=Path(p['decoder_parent_directory'])/a['file'];assert sha(f.read_bytes())==a['sha256']
                arrays.update(query=c['query'],key=c['key'],value=diagnostic['value'],LSE=take(torch.load(f,map_location='cpu',weights_only=True)['LSE']))
            else:
                arrays.update({k:diagnostic[k] for k in ['q','k','raw_g','beta','pre_value_seed','source_gradient','conv_weights','source']})
                row['frozen_SiLU_vs_native_fused_value']=metric(diagnostic['value'],diagnostic['pre_value'].float()*diagnostic['frozen_silu_gain'])
                row['conv_content_sum_vs_postconv']=metric(diagnostic['post_conv_components'].double().sum(1),components.double().sum(1))
            row['fixed_CPU_reference']={'heads':heads,'source_rows':rows,'FA_scale':0.0625,'GDN_scale':scale}
            persist(f'layer{i}_content_review',arrays);row['status']='complete';save()
            del d,c,e,components,diagnostic,agg,arrays,head_native,head_sum,component_sum,source_reference,selected,selected_projection
    r['sources_after']=sources();assert r['sources_after']==r['sources_before']
    assert sha(Path(fa.__file__).read_bytes())==p['installed_FA_interface_sha256']
    r['weight_stats_after']=stats();assert r['weight_stats_after']==r['weight_stats_before']
    r['status']='both_native_content_adjoint_FT_boundaries_executed'
except Exception:
    r['status']='failed';r['error']=traceback.format_exc()
finally:
    r['job_seconds_before_bundle']=time.perf_counter()-start;save()
    with zipfile.ZipFile(HERE/'review_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in list(p['files_sha256'])+['protocol.json','results.json']+[a['file'] for a in r['artifacts']]:
            if (HERE/name).exists():z.write(HERE/name,name)
    print(json.dumps({'status':r['status'],'error':r.get('error')}),flush=True)

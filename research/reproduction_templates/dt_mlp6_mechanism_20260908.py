"""One explicitly recomputed3GEMM MLP6 diagnostic; no model or candidate run.

The actual saved input/output coefficients remain the ledger endpoints. A
recomputed m_product/mu/mg/mnorm is diagnostic evidence, never relabeled as an
originally captured multiplier. All conditional terms use prediction-actual.
"""
import os,sys,json,time,hashlib,traceback,signal,zipfile,gc,struct
from pathlib import Path
A=Path(__file__).resolve().parent;p=json.loads((A/'protocol.json').read_bytes());sha=lambda b:hashlib.sha256(b).hexdigest()
os.environ.update(MACA_PATH='/opt/maca',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1',
    TRITON_CACHE_DIR=p['compiler_cache'],TORCHINDUCTOR_CACHE_DIR=p['boundary_compiler_cache'])
for overlay in p['dependency_overlays']:sys.path.insert(0,overlay)
sys.path.insert(0,str(A))
r={'status':'starting','protocol':p,'model_loads':0,'DT_calls':0,'FT_calls':0,'native_score_calls':0,'generation_calls':0,
   'finite_FA_calls':0,'finite_FLA_calls':0,'weight_tensors_read':0,'diagnostic_graph_calls_entered':0,
   'diagnostic_graph_calls_returned':0,'semantic_GEMMs_per_returned_graph':3,'calls':[],'points':{}}
started=time.perf_counter();recomputed=None;weight_cpu={}


def save():
    q=A/'results.partial';q.write_text(json.dumps(r,indent=2));q.replace(A/'results.json')


def file_sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):value.update(block)
    return value.hexdigest()


def sources():
    out={name:file_sha(A/name) for name in p['files_sha256']}
    assert out==p['files_sha256']
    for name,want in p['runtime_source_sha256'].items():
        value=file_sha(Path(p['isolated_site'])/name);assert value==want,name;out['native/'+name]=value
    return out


def timed(kind,fn,gpu=False):
    if gpu:torch.cuda.synchronize()
    tick=time.perf_counter();item={'kind':kind,'status':'entered'};r['calls'].append(item)
    try:
        value=fn()
        if gpu:torch.cuda.synchronize()
        item['status']='returned';return value
    except Exception:item['status']='failed';raise
    finally:item['seconds']=time.perf_counter()-tick


def tensor_sha(x):return sha(x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
def project(m,x):
    assert m.shape==x.shape
    return (m.double()*x.double()).sum(-1)


def scalar_secant(g0,g1,s0,s1):
    distance=g1-g0;sigmoid=g0.sigmoid();derivative=sigmoid*(1+g0*(1-sigmoid))
    return torch.where(distance!=0,(s1-s0)/torch.where(distance!=0,distance,torch.ones_like(distance)),derivative)


def summarize(value,groups):
    assert value.device.type=='cpu' and value.ndim==3 and value.shape[:2]==(1,info['total_length'])
    token=value.sum(-1)[0]
    return {'total':float(value.sum()),'positive':float(value.clamp_min(0).sum()),'negative':float(value.clamp_max(0).sum()),
        'absolute_sum_before_cancellation':float(value.abs().sum()),'per_token':token.tolist(),
        'groups':{name:{'total':float(value[:,positions].sum()),'positive':float(value[:,positions].clamp_min(0).sum()),
            'negative':float(value[:,positions].clamp_max(0).sum())} for name,positions in groups.items()}}


def audit_point(label,clean,deleted,eos,groups,expected_error):
    """A telescoping algebra on actual clean-deletion features, not a model replay."""
    gC,uC,sC,pC,xC,yC=[clean[k].double() for k in D_FIELDS]
    gA,uA,sA,pA,xA,yA=[deleted[k].double() for k in D_FIELDS]
    gE,uE,sE=[eos[k].double() for k in D_FIELDS[:3]]
    dg,du,ds,dp,dx,dy=gC-gA,uC-uA,sC-sA,pC-pA,xC-xA,yC-yA
    product_delta=sC*uC-sA*uA
    sC_theory=torch.nn.functional.silu(gC);sA_theory=torch.nn.functional.silu(gA)
    ds_theory=sC_theory-sA_theory
    rho=recomputed['m_product'].double();mu=recomputed['mu'].double();mg=recomputed['mg'].double();new=recomputed['mnorm'].double()
    pred=project(saved_mnorm,dx);actual=project(saved_upstream,dy)
    out={'sign_convention':'prediction_minus_actual','reference_MLP_combined':expected_error,
        'saved_input_prediction':float(pred.sum()),'actual_output_effect':float(actual.sum()),
        'saved_prediction_minus_actual':float((pred-actual).sum()),'recomputed_input_prediction':float(project(new,dx).sum()),'terms':{}}
    def term(name,value):out['terms'][name]=summarize(value,groups)
    term('diagnostic_recompute_to_saved_mnorm',((saved_mnorm.double()-new)*dx))
    # A tokenwise scalar can be represented with a singleton channel here; no
    # per-channel interpretation is claimed for this combined input boundary.
    term('input_projection_GEMM_transfer',(project(new,dx)-project(mu,du)-project(mg,dg)).unsqueeze(-1))
    term('finite_coefficient_rounding',mu*du+mg*dg-rho*(s_mean*du+u_mean*slope_native*dg))
    term('SiLU_scalar_secant_curvature',rho*u_mean*(slope_theory*dg-ds_theory))
    term('SiLU_native_endpoint_secant_rounding',rho*u_mean*(slope_native-slope_theory)*dg)
    term('SiLU_native_output_rounding',rho*u_mean*(ds_theory-ds))
    # This is the exact conditional residual of the symmetric product rule
    # reanchored to actual B1 clean/allEOS (or B2 itself for the control).
    term('bilinear_content_gate_interaction',rho*(ds*du-0.5*(sC-sE)*du-0.5*(uC-uE)*ds))
    term('bilinear_B2_to_B1_endpoint_transfer',rho*((s_mean-0.5*(sE+sC))*du+(u_mean-0.5*(uE+uC))*ds))
    term('native_product_rounding',rho*(product_delta-dp))
    term('output_projection_GEMM_transfer',(project(rho,dp)-actual).unsqueeze(-1))
    out['term_sum']=sum(v['total'] for v in out['terms'].values())
    out['closure_residual']=out['term_sum']-out['saved_prediction_minus_actual']
    out['saved_ledger_difference']=out['saved_prediction_minus_actual']-expected_error
    out['group_saved_error']={name:float((pred-actual)[0,positions].sum()) for name,positions in groups.items()}
    out['zero_B2_gate_delta_channels']=int(((paired['gate_output'][1::2]-paired['gate_output'][0::2])==0).sum())
    out['interpretation']='Recomputed multiplier algebra; numerical transfer is explicit. Bilinear/SiLU errors are conditional on frozen endpoints and coefficients. No candidate or improvement guarantee.'
    r['points'][label]=out;save()
    assert abs(out['closure_residual'])<1e-7,out['closure_residual']
    assert abs(out['saved_ledger_difference'])<1e-7,out['saved_ledger_difference']


try:
    def timeout(*args):raise TimeoutError('Frozen MLP6 mechanism diagnostic budget expired.')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(p['budget']['wall_time_seconds']);r['sources_before']=sources()
    import torch
    import safetensors
    from safetensors import safe_open
    torch.set_num_threads(4)
    from qwen35_decoder_finite import _linear_transpose
    from compiled_swiglu_secant import swiglu_finite_rule
    D_FIELDS=('gate_output','up_output','silu_output','down_input','post_norm_output','mlp_output')
    assert p['fixed_steps']==[1,10,20] and p['layer_index']==6
    metadata=json.loads((A/'weights_metadata.json').read_bytes());cp=Path(p['checkpoint']).resolve()
    assert Path(metadata['index_path']).resolve()==cp/'model.safetensors.index.json'
    raw=Path(metadata['index_path']).read_bytes();assert sha(raw)==metadata['index_sha256'];index=json.loads(raw)
    assert set(metadata['weight_map'])=={f'model.language_model.layers.6.mlp.{part}_proj.weight' for part in ['down','gate','up']}
    assert all(index['weight_map'][key]==shard for key,shard in metadata['weight_map'].items())
    for shard,want in metadata['shard_stats'].items():
        path=(cp/shard).resolve();assert path.parent==cp and path.name==shard
        assert [path.stat().st_size,path.stat().st_mtime_ns]==want==p['expected_weight_stats'][shard]
    raw=Path(p['actual_results_path']).read_bytes();assert sha(raw)==p['actual_results_sha256'];actual_results=json.loads(raw)
    assert actual_results['status']=='decoder19_6_actual_conditional_observation_complete'
    info=actual_results['input'];assert actual_results['private_artifact']['sha256']==p['actual_artifact_sha256']
    assert timed('hash_actual_private_input',lambda:file_sha(p['actual_artifact_path']))==p['actual_artifact_sha256']
    artifact=timed('read_actual_private_CPU_artifact',lambda:torch.load(p['actual_artifact_path'],map_location='cpu',weights_only=True))
    paired={k:artifact['layers']['6']['paired']['d'][k] for k in D_FIELDS}
    points={step:{k:artifact['scoring'][step]['6']['d'][k] for k in D_FIELDS} for step in ['0','1','10','20']}
    saved_upstream=artifact['layers']['6']['coeff']['upstream'];saved_mnorm=artifact['layers']['6']['coeff']['m_mlp_norm_output']
    del artifact;gc.collect()
    batch,length,width=saved_upstream.shape;assert batch==1 and length==info['total_length'] and saved_mnorm.shape==saved_upstream.shape
    intermediate=paired['gate_output'].shape[-1]
    for features in [paired,*points.values()]:
        rows=2 if features is paired else 1
        for name in D_FIELDS[:4]:assert features[name].shape==(rows,length,intermediate)
        for name in D_FIELDS[4:]:assert features[name].shape==(rows,length,width)
    r['weight_receipts']={};r['shard_headers']={}
    def read_three_weights():
        for shard in sorted(metadata['shard_stats']):
            path=cp/shard
            with path.open('rb') as f:
                header_length=struct.unpack('<Q',f.read(8))[0];assert 0<header_length<16*1024*1024;header=f.read(header_length)
            parsed=json.loads(header);r['shard_headers'][shard]={'header_sha256':sha(header),'header_bytes':header_length}
            with safe_open(str(path),framework='pt',device='cpu') as f:
                for key,expected_shard in sorted(metadata['weight_map'].items()):
                    if expected_shard!=shard:continue
                    assert key in parsed
                    value=f.get_tensor(key);r['weight_tensors_read']+=1
                    name=key.split('.')[-2];expected_shape=(width,intermediate) if name=='down_proj' else (intermediate,width)
                    assert value.dtype==torch.bfloat16 and value.shape==expected_shape
                    weight_cpu[name]=value
                    r['weight_receipts'][key]={'shard':str(path),'shape':list(value.shape),'dtype':str(value.dtype),
                        'tensor_sha256_before':tensor_sha(value),'tensor_bytes':value.numel()*value.element_size(),
                        'safetensors_header_entry':parsed[key]}
        assert r['weight_tensors_read']==3
    timed('read_only_three_indexed_MLP_weight_tensors',read_three_weights)
    assert set(weight_cpu)=={'down_proj','gate_proj','up_proj'}
    r['weight_hash_scope']='Fixed index SHA and known shard size/mtime; exact selected tensor hashes measured before/after. No claim of a pre-existing whole-shard cryptographic hash; only3get_tensor calls and no model loading.'
    assert torch.cuda.mem_get_info()[0]>=4*1024**3
    r['GPU_allocated_before']=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
    gpu_weights={name:value.to('cuda') for name,value in weight_cpu.items()}
    args=[paired[name][i::2].to('cuda') for name in ['gate_output','up_output','silu_output'] for i in (0,1)]
    # Match the original graph's argument and arithmetic order exactly; added
    # diagnostic outputs can change compiler scheduling, so drift is reported.
    g0,g1,u0,u1,s0,s1=args;upstream=saved_upstream.to('cuda')
    def diagnostic_graph(g0,g1,u0,u1,s0,s1,upstream,down_weight,up_weight,gate_weight):
        m_product=_linear_transpose(upstream,down_weight)
        mu,mg=swiglu_finite_rule(g0,g1,u0,u1,s0.float(),s1.float(),m_product)
        mnorm=_linear_transpose(mu,up_weight)+_linear_transpose(mg,gate_weight)
        return m_product,mu,mg,mnorm
    compiled=torch.compile(diagnostic_graph,fullgraph=True,dynamic=False,options={'triton.cudagraphs':False,'max_autotune':False})
    r['status']='one_recomputed_3GEMM_diagnostic_graph';save();r['diagnostic_graph_calls_entered']+=1
    values=timed('compiled_recomputed_MLP6_graph_including_compile',lambda:compiled(g0,g1,u0,u1,s0,s1,upstream,
        gpu_weights['down_proj'],gpu_weights['up_proj'],gpu_weights['gate_proj']),gpu=True)
    r['diagnostic_graph_calls_returned']+=1
    recomputed={name:value.detach().to('cpu',copy=True) for name,value in zip(['m_product','mu','mg','mnorm'],values)}
    assert all(bool(value.isfinite().all()) for value in recomputed.values())
    difference=recomputed['mnorm'].double()-saved_mnorm.double();den=float(saved_mnorm.double().norm())
    r['saved_vs_recomputed_mnorm']={'bitwise_equal':bool(torch.equal(recomputed['mnorm'].contiguous().view(torch.uint8),saved_mnorm.contiguous().view(torch.uint8))),
        'difference_L2':float(difference.norm()),'relative_L2':float(difference.norm())/den if den else None,'max_absolute':float(difference.abs().max()),
        'saved_sha256':tensor_sha(saved_mnorm),'recomputed_sha256':tensor_sha(recomputed['mnorm']),
        'policy':'No retroactive precision threshold. Every ledger uses original saved mnorm; its difference from this diagnostic is a separately reported term.'}
    r['peak_GPU_allocated']=torch.cuda.max_memory_allocated();r['GPU_reserved']=torch.cuda.memory_reserved()
    del values,args,g0,g1,u0,u1,s0,s1,upstream,gpu_weights,difference;gc.collect();torch.cuda.synchronize()
    r['GPU_allocated_after_cleanup']=torch.cuda.memory_allocated()
    g0,g1=paired['gate_output'][0::2].double(),paired['gate_output'][1::2].double()
    u0,u1=paired['up_output'][0::2].double(),paired['up_output'][1::2].double()
    s0,s1=paired['silu_output'][0::2].double(),paired['silu_output'][1::2].double()
    s_mean=(s0+s1)*0.5;u_mean=(u0+u1)*0.5;slope_native=scalar_secant(g0,g1,s0,s1)
    slope_theory=scalar_secant(g0,g1,torch.nn.functional.silu(g0),torch.nn.functional.silu(g1))
    def groups_for(deleted):
        eligible=set(info['keep']);selected=set(deleted);assert selected<=eligible
        groups={'eligible_deleted':sorted(selected),'eligible_kept':sorted(eligible-selected),
            'other_prompt':sorted(set(range(info['prompt_length']))-eligible),'fixed_response':list(range(info['prompt_length'],info['total_length']))}
        assert sorted(j for values in groups.values() for j in values)==list(range(info['total_length']))
        return groups
    r['status']='fixed_CPU_MLP6_conditional_algebra';save()
    audit_point('B2', {k:v[1::2] for k,v in paired.items()},{k:v[0::2] for k,v in paired.items()},
        {k:v[0::2] for k,v in paired.items()},groups_for(info['keep']),actual_results['B2_decompositions']['6']['terms']['MLP_combined'])
    for step in p['fixed_steps']:
        expected=actual_results['points'][str(step)]['layer_decompositions']['6']['terms']['MLP_combined']
        audit_point(str(step),points['0'],points[str(step)],points['20'],
            groups_for(actual_results['points'][str(step)]['input_receipt']['deleted_positions']),expected)
    for key,receipt in r['weight_receipts'].items():
        receipt['tensor_sha256_after']=tensor_sha(weight_cpu[key.split('.')[-2]])
        assert receipt['tensor_sha256_after']==receipt['tensor_sha256_before']
    assert file_sha(metadata['index_path'])==metadata['index_sha256']
    for shard,want in metadata['shard_stats'].items():
        q=cp/shard;assert [q.stat().st_size,q.stat().st_mtime_ns]==want
    assert file_sha(p['actual_artifact_path'])==p['actual_artifact_sha256']
    r['sources_after']=sources();assert r['sources_before']==r['sources_after']
    assert r['diagnostic_graph_calls_entered']==r['diagnostic_graph_calls_returned']==1 and r['weight_tensors_read']==3
    r['status']='MLP6_actual_mechanism_diagnostic_complete'
except Exception:
    r['status']='failed';r['error']=traceback.format_exc()
finally:
    signal.alarm(0)
    if recomputed is not None:
        artifact=A/'MLP6_diagnostic_recomputed_coefficients.pt'
        try:
            torch.save({'scope':'Explicit diagnostic recomputation,notoriginalcapturedcoefficients','coefficients':recomputed},artifact)
            r['diagnostic_artifact']={'file':artifact.name,'sha256':file_sha(artifact),'bytes':artifact.stat().st_size}
        except Exception:r['diagnostic_artifact_error']=traceback.format_exc();r['status']='failed'
    r['compilation_count_scope']='Exactly one explicit compiled graph invocation,containing3semanticGEMMs. Compile/default compiler kernel work is included in elapsed cost; no unsupported claim of exactly3physicalGPUlaunches or free compilation.'
    r['job_seconds']=time.perf_counter()-started;save()
    with zipfile.ZipFile(A/'review_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in list(p['files_sha256'])+['protocol.json','results.json']:
            if (A/name).exists():z.write(A/name,name)
    print(json.dumps({'status':r['status'],'graph_calls':r['diagnostic_graph_calls_returned'],'seconds':r['job_seconds'],'error':r.get('error')}),flush=True)

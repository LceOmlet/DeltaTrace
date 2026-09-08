"""One saved-endpoint native adjoint capture, then bounded CPU reference audit."""
import os,sys,json,time,hashlib,signal,traceback,zipfile,gc
from pathlib import Path
A=Path(__file__).resolve().parent;p=json.loads((A/'protocol.json').read_bytes())
os.environ.update(MACA_PATH='/opt/maca',HF_HUB_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',
    TRITON_CACHE_DIR=p['compiler_cache'],TORCHINDUCTOR_CACHE_DIR=p['boundary_compiler_cache'],
    OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
for overlay in p['dependency_overlays']:sys.path.insert(0,overlay)
sys.path.insert(0,str(A))
r={'status':'starting','protocol':p,'model_loads':0,'model_forwards':0,'scorer_calls':0,'FT_calls':0,'FA_calls':0,
   'native_input_adjoints_entered':0,'native_input_adjoints_returned':0,'mixed_entered':0,'mixed_returned':0,'calls':[]}
started=time.perf_counter();retained={};torch=None

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def save():
    f=A/'results.partial';f.write_text(json.dumps(r,indent=2,allow_nan=False));f.replace(A/'results.json')

def cpu(x):
    if isinstance(x,torch.Tensor):return x.detach().to('cpu',copy=True)
    if isinstance(x,dict):return {k:cpu(v) for k,v in x.items()}
    return x

def nbytes(x):
    if isinstance(x,torch.Tensor):return x.numel()*x.element_size()
    if isinstance(x,dict):return sum(nbytes(v) for v in x.values())
    return 0

def drift(x,y):
    assert x.shape==y.shape and bool(torch.isfinite(x).all()) and bool(torch.isfinite(y).all())
    d=x.double()-y.double();den=y.double().norm()
    return {'bitwise_equal':bool(torch.equal(x,y)),'max_absolute':float(d.abs().max()),'relative_L2':float(d.norm()/den) if den else None}

def timed(label,fn,gpu=True):
    if gpu:torch.cuda.synchronize()
    tick=time.perf_counter();row={'kind':label,'status':'entered'};r['calls'].append(row)
    try:
        out=fn()
        if gpu:torch.cuda.synchronize()
        row['status']='returned';return out
    finally:row['seconds']=time.perf_counter()-tick

try:
    def timeout(*args):raise TimeoutError('Frozen native-adjoint and CPU reference audit budget expired.')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(p['budget']['wall_time_seconds'])
    for name,want in p['files_sha256'].items():assert digest(A/name)==want,name
    for item in p['protected_sources']:assert digest(item['path'])==item['sha256'],item['path']
    import numpy as np
    import torch
    torch.set_num_threads(4)
    from finite_fla_gpu import native_input_adjoints,mixed_coefficients,verify_native_sources
    import NI_fla_reference_mismatch_cpu_20260908 as audit
    verify_native_sources(p['native_stage_source_sha256'])
    source=json.loads(Path(p['source_results_path']).read_bytes());assert source['status']=='NI_current_layer0_5replay1finite_internal_measurement_complete'
    boundary=json.loads(Path(p['boundary_results_path']).read_bytes());assert boundary['status']=='NI_current_34boundaries_1DT4score_observation_complete'
    assert source['input']==boundary['input']==p['input']
    path=Path(p['source_private_path']);assert path.stat().st_size==p['source_private_bytes'] and digest(path)==p['source_private_sha256']
    raw=timed('load_saved_actual_private_CPU',lambda:torch.load(path,map_location='cpu',weights_only=True),False)
    paired=raw['native']['B2']['e'];points={step:raw['native'][step]['e'] for step in ('0','1','10','20')}
    saved=raw['coeff']['coeff'];do=raw['coeff']['mo_native'];scale=float(source['points']['B2']['actual_scale'])
    assert scale==p['scale'] and do.dtype==torch.bfloat16
    r['source_operand_CPU_bytes']=nbytes(paired)+nbytes(points)+nbytes(saved)+nbytes(do)
    del raw;gc.collect()
    assert torch.cuda.is_available();torch.cuda.set_device(0);torch.cuda.reset_peak_memory_stats()
    e={name:value.to('cuda').contiguous() for name,value in paired.items()};seed=do.to('cuda').contiguous()
    r['actual_input_layout']={name:{'shape':list(x.shape),'dtype':str(x.dtype),'stride':list(x.stride())} for name,x in e.items()}
    r['status']='one_existing_native_input_adjoints';r['native_input_adjoints_entered']+=1;save()
    with torch.no_grad():adj=timed('existing_native_input_adjoints_two_stages',lambda:native_input_adjoints(e,seed,scale))
    r['native_input_adjoints_returned']+=1
    def mixed(endpoints,adjoints,actual_scale):
        return mixed_coefficients(endpoints,adjoints,actual_scale,False,diagnostics=True)
    compiled=torch.compile(mixed,fullgraph=True,dynamic=False,options={'triton.cudagraphs':False,'max_autotune':False})
    r['status']='one_existing_mixed_with_L_r0';r['mixed_entered']+=1;save()
    with torch.no_grad():coeff,diag=timed('existing_compiled_mixed_with_retained_outputs',lambda:compiled(e,adj,scale))
    r['mixed_returned']+=1;retained={'adjoints':cpu(adj),'diagnostic':cpu(diag),'coeff':cpu(coeff)}
    assert set(diag)=={'L','r0'} and set(coeff)==set(saved)=={'q','k','v','beta','alpha','g'}
    assert torch.equal(retained['adjoints']['do'],do)
    r['coefficient_replay_drift_report_only']={name:drift(retained['coeff'][name],saved[name]) for name in saved}
    r['retained_tensor_CPU_bytes']=nbytes(retained)
    artifact=A/'actual_native_adjoints_mixed_private.pt';torch.save(retained,artifact)
    r['adjoint_capture_artifact']={'file':artifact.name,'sha256':digest(artifact),'bytes':artifact.stat().st_size,'tensor_CPU_bytes':nbytes(retained)}
    r['peak_allocated_GPU_bytes']=torch.cuda.max_memory_allocated();r['peak_reserved_GPU_bytes']=torch.cuda.max_memory_reserved()
    del e,seed,adj,diag,coeff,compiled;gc.collect();torch.cuda.empty_cache()
    r['status']='CPU_actual_reference_content_audit';save()
    summary,arrays=timed('CPU64_original_chunk_reference_algebra',lambda:audit.audit(paired,points,saved,retained['coeff'],
        retained['adjoints'],retained['diagnostic'],scale,p['input'],{step:boundary['points'][step]['input_receipt'] for step in ('0','1','10','20')}),False)
    r['CPU_audit']=summary
    # The helper exposes the same saved-seed error measured in the completed
    # single-layer study, not a scorer/whole-model substitution.
    for step in ('1','10','20','B2'):
        got=summary['points'][step]['saved_error']['net']
        expected=source['conditional_ledgers'][step]['replayed_16term_ledger']['terms']['GDN_FLA_including_raw_g_exp']
        assert abs(got-expected)<1e-7,(step,got,expected)
    np.savez_compressed(A/'vectors.npz',**arrays);r['vectors_sha256']=digest(A/'vectors.npz')
    assert r['native_input_adjoints_entered']==r['native_input_adjoints_returned']==r['mixed_entered']==r['mixed_returned']==1
    for item in p['protected_sources']:assert digest(item['path'])==item['sha256'],item['path']
    assert digest(path)==p['source_private_sha256'];verify_native_sources(p['native_stage_source_sha256'])
    r['status']='NI_FLA_reference_content_native_capture_and_CPU_audit_complete'
except Exception:r['status']='failed';r['error']=traceback.format_exc()
finally:
    signal.alarm(0);r['seconds']=time.perf_counter()-started
    r['partial_count_policy']='One returned native_input_adjoints comprises exactly the existing two native stages. If it fails midcall,the completed stage count is unknown;do not report zero work. No model/forward/FA/scorer/FT is imported or called by this study.'
    r['scope']='Fixed real endpoint1 route/scale/seed/native adjoints. Only real captured reference0 versus A operands substituted into unchanged CPU chunk coefficient algebra;not an A1 model counterfactual,shadowstate recursion,new finite rule or metric comparison.'
    r['remaining_residual_scope']='The A1 primitive paired residual is measured,not assumedzero. It includes native read/write/chunk arithmetic and finite/native-adjoint consistency. Original B1clean-vs-B2input transfer is explicit and common across deletion masks;do not label it purecontent mismatch.'
    save()
    with zipfile.ZipFile(A/'review_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in [*p['files_sha256'],'protocol.json','results.json','vectors.npz']:
            if (A/name).exists():z.write(A/name,name)
    print(json.dumps({'status':r['status'],'seconds':r['seconds'],'adjoints':r['native_input_adjoints_returned'],'mixed':r['mixed_returned'],'error':r.get('error')}),flush=True)

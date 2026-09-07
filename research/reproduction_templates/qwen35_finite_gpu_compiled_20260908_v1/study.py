"""Bounded real-capture GPU finite pullback and matched native backward cost."""
import os
os.environ.update(MACA_PATH='/opt/maca',HF_HUB_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1')
import hashlib,inspect,io,json,sys,time,traceback,zipfile
from collections import Counter
from pathlib import Path
HERE=Path(__file__).resolve().parent;p=json.loads((HERE/'protocol.json').read_bytes())
os.environ['TRITON_CACHE_DIR']=p['compiler_cache']
os.environ['TORCHINDUCTOR_CACHE_DIR']=str(HERE/'inductor_cache')
sha=lambda b:hashlib.sha256(b).hexdigest()
r={'status':'running','protocol':p,'model_loads':0,'model_forwards':0,'generation_calls':0,'quality_queries':0,
   'whole_model_attributions':0,'cached_adjoint_mixed_attempts':0,'finite_pullback_attempts':0,'native_backward_attempts':0,
   'compiled_pullback_attempts':0,'calls':[],'artifacts':[]}
start=time.perf_counter()
def save():
    f=HERE/'results.partial';f.write_text(json.dumps(r,ensure_ascii=False,indent=2));f.replace(HERE/'results.json')
def source_receipt():
    count=0;changed=[];h=hashlib.sha256();site=Path(p['isolated_site'])
    for name,digest in p['wheels'].items():
        raw=Path(name).read_bytes();assert sha(raw)==digest
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            for name in z.namelist():
                if not(name.endswith('.py') and name.startswith(('fla/','transformers/'))):continue
                raw=(site/name).read_bytes()
                if raw!=z.read(name):
                    assert name in p['explicit_native_changes'] and sha(raw)==p['explicit_native_changes'][name];changed.append(name)
                h.update(name.encode()+bytes.fromhex(sha(raw)));count+=1
    assert count==2853 and set(changed)==set(p['explicit_native_changes'])
    return {'count':count,'changes':changed,'sha256':h.hexdigest()}
try:
    for name,digest in p['files_sha256'].items():assert sha((HERE/name).read_bytes())==digest
    r['sources_before']=source_receipt();assert r['sources_before']['sha256']==p['source_tree_sha256']
    import numpy as np
    import torch
    import triton
    from finite_fla_gpu import verify_native_sources,mixed_coefficients,finite_fla_pullback
    chunk=verify_native_sources(p['native_stage_source_sha256'])
    assert sha((Path(p['prior_GPU_directory'])/'results.json').read_bytes())==p['prior_GPU_sha256']
    from finite_fla_gpu import make_compiled_finite_pullback
    compiled_pullback=make_compiled_finite_pullback()
    torch.set_num_threads(4)
    r['versions']={'torch':torch.__version__,'triton':triton.__version__,'backend':triton.runtime.driver.active.get_current_target().backend,
                   'device':torch.cuda.get_device_name()}
    assert r['versions']['backend']=='maca'
    parent=Path(p['parent_directory']);raw=(parent/'results.json').read_bytes();assert sha(raw)==p['parent_sha256']
    parent_result=json.loads(raw)
    def read(name):
        path=parent/(name+'.npz');assert sha(path.read_bytes())==p['input_npz_sha256'][path.name]
        item=next(x for x in parent_result['artifacts'] if x['file']==path.name)
        with np.load(path,allow_pickle=False) as z:
            return {k:torch.from_numpy(z[k].copy()).to(getattr(torch,item['dtypes'][k].split('.')[-1])).to('cuda') for k in z.files}
    endpoints=read('real_paired_FLA_prefix');saved=read('native_input_adjoints')
    do=saved['do'];scale=parent_result['native_scale']
    assert endpoints['q'].shape==(4,129,32,128) and do.shape==(2,129,32,128)
    r['resident_input_bytes']=sum(x.untyped_storage().nbytes() for x in [*endpoints.values(),*saved.values()])
    def persist(name,values):
        path=HERE/(name+'.npz')
        with path.open('wb') as f:np.savez_compressed(f,**{k:v.detach().float().cpu().numpy() for k,v in values.items()})
        r['artifacts'].append({'file':path.name,'sha256':sha(path.read_bytes()),'dtypes':{k:str(v.dtype) for k,v in values.items()}});save()
    codes={inspect.unwrap(fn).__code__:label for label,fn in [
        ('native_backward',chunk.chunk_gated_delta_rule_bwd),('native_dv_local',chunk.chunk_bwd_dv_local),
        ('native_state_adjoint',chunk.chunk_gated_delta_rule_bwd_dhu),('native_state_reconstruction',chunk.chunk_gated_delta_rule_fwd_h),
        ('native_WY_backward',chunk.prepare_wy_repr_bwd)]}
    counts=Counter()
    def event(frame,kind,value):
        if kind=='call' and frame.f_code in codes:counts[codes[frame.f_code]]+=1
    def native_backward():
        values={name:endpoints[name][1::2].contiguous() for name in ['q','k','v','g','beta','A']}
        dq,dk,dv,db,dg,dh0=chunk.chunk_gated_delta_rule_bwd(**values,scale=scale,initial_state=None,do=do,dht=None,cu_seqlens=None)
        assert dh0 is None
        return dict(q=dq,k=dk,v=dv,beta=db,g=dg)
    def run(kind,phase,profile=False):
        counter={'cached':'cached_adjoint_mixed_attempts','finite':'finite_pullback_attempts','native':'native_backward_attempts','compiled':'compiled_pullback_attempts'}[kind]
        ceiling={'cached':0,'finite':5,'native':4,'compiled':5}[kind]
        assert r[counter]<ceiling;r[counter]+=1
        row={'kind':kind,'phase':phase,'profiled':profile,'status':'attempted'};r['calls'].append(row);save()
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        row['before_bytes']=torch.cuda.memory_allocated();previous=Counter(counts)
        tick=time.perf_counter();begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        profiler=None
        try:
            if profile:profiler=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]);profiler.__enter__()
            assert sys.getprofile() is None
            # The passive hook is used only for cold/profiled diagnostics, never formal timings.
            if phase=='cold' or profile:sys.setprofile(event)
            begin.record()
            with torch.no_grad():
                if kind=='cached':out=mixed_coefficients(endpoints,saved,scale)
                elif kind=='finite':out=finite_fla_pullback(endpoints,do,scale)
                elif kind=='compiled':out=compiled_pullback(endpoints,do,scale)
                else:out=native_backward()
            end.record();torch.cuda.synchronize()
            row.update(status='complete',wall_ms=(time.perf_counter()-tick)*1000,event_ms=begin.elapsed_time(end),
                       peak_bytes=torch.cuda.max_memory_allocated(),after_bytes=torch.cuda.memory_allocated())
            return out
        except Exception:
            row['error']=traceback.format_exc();row['status']='failed';raise
        finally:
            sys.setprofile(None);row['native_dispatch']=dict(counts-previous)
            if profiler is not None:
                profiler.__exit__(None,None,None)
                path=HERE/(kind+'_profile.json');profiler.export_chrome_trace(str(path));raw=path.read_bytes()
                kernels=[e for e in json.loads(raw)['traceEvents'] if e.get('cat')=='kernel']
                kc=Counter(e['name'] for e in kernels);duration=Counter()
                for e in kernels:duration[e['name']]+=e.get('dur',0)
                row['profile']={'file':path.name,'sha256':sha(raw),'kernel_counts':dict(kc),'kernel_total_us':dict(duration)}
            save()
    for kind in ['compiled','finite','native']:
        out=run(kind,'cold');persist(kind+'_coefficients',out);del out
    for iteration,order in enumerate([['compiled','finite','native'],['native','finite','compiled'],['finite','compiled','native']]):
        for kind in order:
            out=run(kind,'formal_'+str(iteration))
            assert all(torch.isfinite(v).all() for v in out.values());del out
    for kind in ['compiled','finite']:
        out=run(kind,'diagnostic',profile=True);del out
    r['sources_after']=source_receipt();assert r['sources_before']==r['sources_after']
    assert r['compiled_pullback_attempts']==r['finite_pullback_attempts']==5 and r['native_backward_attempts']==4
    from torch._dynamo.utils import counters
    r['compiler_counters']={str(k):dict(v) for k,v in counters.items()}
    r['compiler_generated_sources']=[]
    for path in (HERE/'inductor_cache').rglob('*.py'):
        raw=path.read_bytes();r['compiler_generated_sources'].append({'file':str(path.relative_to(HERE)),'sha256':sha(raw),'bytes':len(raw)})
    r['status']='official_compiler_fused_mixed_finite_screen_complete'
except Exception:
    sys.setprofile(None);r['status']='failed';r['error']=traceback.format_exc()
finally:
    r['job_seconds']=time.perf_counter()-start;save()
    with zipfile.ZipFile(HERE/'review_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in ['study.py','protocol.json','finite_fla_gpu.py','results.json']+[x.name for x in HERE.glob('*.npz')]:z.write(HERE/name,name)
        for path in (HERE/'inductor_cache').rglob('*.py'):z.write(path,str(path.relative_to(HERE)))
    print(json.dumps({'status':r['status'],'attempts':[r['compiled_pullback_attempts'],r['finite_pullback_attempts'],r['native_backward_attempts']],
                      'error':r.get('error')}),flush=True)

from pathlib import Path
import hashlib,json,os,sys,time,traceback
A=Path(__file__).resolve().parent;plan=json.loads((A/'fa_cached_followup_plan.json').read_bytes());R=Path(plan['release_root']);out=A/'template_validation';out.mkdir(exist_ok=False);s=lambda b:hashlib.sha256(b).hexdigest()
r={'status':'running','driver_sha256':s(Path(__file__).read_bytes()),'plan_sha256':s((A/'fa_cached_followup_plan.json').read_bytes()),'model_calls':0,'attribution_calls':0}
def save():(out/'results.json').write_text(json.dumps(r,indent=2)+'\n')
save();t=time.perf_counter()
try:
    for n,h in plan['files'].items():assert s((A/n).read_bytes())==h,n
    for n,h in plan['runtime_files'].items():assert s((R/n).read_bytes())==h,n
    os.environ.update(MACA_PATH='/opt/maca',OMP_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1',TRITON_CACHE_DIR=str(A/'triton_template_validation'),TORCHINDUCTOR_CACHE_DIR=str(A/'inductor_template_validation'))
    for p in [R/'deltatrace/clean/qwen3',R/'deltatrace/accelerated/qwen3']:sys.path.insert(0,str(p))
    import torch
    torch.set_num_threads(4)
    from check_template_graph_validation_probes import run
    r['checks']=run();r['status']='complete'
except Exception:r['status']='failed';r['error']=traceback.format_exc()
finally:r['seconds']=time.perf_counter()-t;save();print(json.dumps(r),flush=True)

"""Compare all model state bytes under original and common CPU-staged loading."""
from pathlib import Path
import gc,hashlib,importlib.util,inspect,json,os,resource,sys,time,traceback
T=Path(__file__).resolve().parent;R=T/'memory_production_v3_release';B=Path('/tmp/codex_short_b1_efficiency_20260910_v1')
plan_path=T/'cpu_staged_loading_plan.json';plan=json.loads(plan_path.read_bytes());out=T/'cpu_staged_loading';out.mkdir(exist_ok=False)
sha=lambda b:hashlib.sha256(b).hexdigest();assert sha(Path(__file__).read_bytes())==plan['driver_sha256']
env=json.loads((R/'environment.json').read_bytes())['qwen3']
os.environ.update(MACA_PATH='/opt/maca',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
for p in [B/'deps',Path(env['official_root'])]:sys.path.insert(0,str(p))
report={'status':'running','driver_sha256':plan['driver_sha256'],'plan_sha256':sha(plan_path.read_bytes()),'model_loads':0,'model_forwards':0,'model_backwards':0,'attribution_calls':0,'records':[]}
def save():(out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
save()
try:
    source=R/'official_exp1/run_time_curve.py';assert sha(source.read_bytes())==plan['exp1_sha256']
    spec=importlib.util.spec_from_file_location('original_exp1',source);bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench)
    import torch
    torch.set_num_threads(4)
    def state_bytes(model):
        result={}
        for kind,items in [('parameter',model.named_parameters()),('buffer',model.named_buffers())]:
            for name,tensor in items:
                value=tensor.detach().cpu().contiguous().view(torch.uint8).numpy()
                result[kind+'/'+name]={'shape':list(tensor.shape),'stride':list(tensor.stride()),'dtype':str(tensor.dtype),'sha256':sha(value),'requires_grad':tensor.requires_grad,'device':str(tensor.device)}
                del value
        return result
    for mode in ['original_GPU_load','original_CPU_load_then_native_to_GPU']:
        gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
        model,tokenizer=bench.load_model_balanced(env['checkpoint'],'cuda:0' if mode=='original_GPU_load' else 'cpu')
        if mode!='original_GPU_load':model.to('cuda:0')
        torch.cuda.synchronize();elapsed=time.perf_counter()-t;report['model_loads']+=1
        row={'mode':mode,'load_seconds':elapsed,'peak_allocated_gb':torch.cuda.max_memory_allocated()/1e9,'peak_reserved_gb':torch.cuda.max_memory_reserved()/1e9,
             'resident_allocated_gb':torch.cuda.memory_allocated()/1e9,'resident_reserved_gb':torch.cuda.memory_reserved()/1e9,
             'host_load_highwater_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'model_source_sha256':sha(Path(inspect.getfile(type(model))).read_bytes()),
             'hf_device_map':getattr(model,'hf_device_map',None),'tokenizer_special_ids':[tokenizer.eos_token_id,tokenizer.pad_token_id],
             'all_parameters_on_C550':all(str(p.device)=='cuda:0' for p in model.parameters())}
        assert row['model_source_sha256']==env['native_model_sha256'] and row['all_parameters_on_C550']
        t=time.perf_counter();row['state']=state_bytes(model);row['full_state_hash_seconds']=time.perf_counter()-t
        row['host_hash_highwater_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        report['records'].append(row);save();del model,tokenizer;gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
        row['after_release_allocated_gb']=torch.cuda.memory_allocated()/1e9;assert torch.cuda.memory_allocated()==0;save()
    old,new=report['records'];report['full_state_byte_exact']=old['state']==new['state']
    report['state_mismatches']=[n for n in old['state'] if old['state'][n]!=new['state'].get(n)]
    report['tokenizer_special_ids_equal']=old['tokenizer_special_ids']==new['tokenizer_special_ids']
    report['GPU_peak_lower']=new['peak_allocated_gb']<old['peak_allocated_gb'] and new['peak_reserved_gb']<old['peak_reserved_gb']
    assert report['full_state_byte_exact'] and report['tokenizer_special_ids_equal'] and report['GPU_peak_lower']
    report['status']='complete'
except BaseException:report['status']='failed';report['error']=traceback.format_exc()
save();print(json.dumps({'status':report['status'],'model_loads':report['model_loads'],'state_mismatches':report.get('state_mismatches'),'error':report.get('error')}),flush=True)

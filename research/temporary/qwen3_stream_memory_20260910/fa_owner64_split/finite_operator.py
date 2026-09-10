"""Finite operator controls only; no model/root/backward calls."""
from pathlib import Path
import hashlib,json,math,os,time,traceback
import numpy as np
import torch
from vendor_fa_finite_shared_mean_reuse_runtime import VendorFAFiniteP1SharedMeanReuse
A=Path(__file__).resolve().parent;s=lambda b:hashlib.sha256(b).hexdigest();p=json.loads((A/'operator_protocol.json').read_bytes())
assert s(Path(__file__).read_bytes())==p['driver_sha256']
b=json.loads((A/'results.json').read_bytes());assert b['status']=='finite_extension_compiled_not_executed'
old=VendorFAFiniteP1SharedMeanReuse(p['original_library']['path'],p['original_library']['sha256'])
new=VendorFAFiniteP1SharedMeanReuse(A/'libdeltatrace_fa_finite_shared_mean_reuse.so',b['library']['sha256'])
r={'status':'running','protocol_sha256':s((A/'operator_protocol.json').read_bytes()),'driver_sha256':p['driver_sha256'],'build_sha256':s((A/'results.json').read_bytes()),'model_calls':0,'backward_calls':0,'full_attribution_calls':0,'operator_calls':0,'cases':[]}
def save():
    q=A/'operator_results.partial';q.write_text(json.dumps(r,indent=2)+'\n');q.replace(A/'operator_results.json')
save()
try:
    torch.manual_seed(p['seed'])
    for batch in p['batches']:
        for n in p['lengths']:
            qshape=(batch,32,n,128);kshape=(batch,8,n,128)
            q0=torch.randn(qshape,device='cuda',dtype=torch.float16)*.25;k0=torch.randn(kshape,device='cuda',dtype=torch.float16)*.25
            q1=q0+torch.randn_like(q0)*.125;k1=k0+torch.randn_like(k0)*.125
            v0=torch.randn(kshape,device='cuda',dtype=torch.float16);u=torch.randn(qshape,device='cuda',dtype=torch.float32)*.25
            lse0=torch.arange(1,n+1,device='cuda',dtype=torch.float32).log()[None,None,:].expand(batch,32,n).contiguous();lse1=lse0+.03125
            x=dict(q0=q0,k0=k0,q1=q1,k1=k1,v0=v0,u=u,lse0=lse0,lse1=lse1)
            case={'batch':batch,'length':n,'synthetic_finite_operator_control':True,'input_sha256':{k:s(v.cpu().numpy().tobytes()) for k,v in x.items()},'rows':[]};r['cases'].append(case);outputs={}
            for rep in range(5):
                for name in (['old','new'] if rep%2==0 else ['new','old']):
                    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
                    answer=(old if name=='old' else new)(x,1/math.sqrt(128));torch.cuda.synchronize();seconds=time.perf_counter()-t
                    case['rows'].append({'method':name,'repeat':rep,'phase':'warm' if rep<2 else 'measured','seconds':seconds,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()});r['operator_calls']+=1
                    outputs[name]=answer
            case['comparisons']={}
            for k in ['dq','dk','dv','tau','center']:
                a=outputs['old'][k];c=outputs['new'][k]
                case['comparisons'][k]={'exact':bool(torch.equal(a,c)),'max_abs':float((a.float()-c.float()).abs().max()),'old_sha256':s(a.cpu().numpy().tobytes()),'new_sha256':s(c.cpu().numpy().tobytes()),'shape':list(a.shape),'dtype':str(a.dtype)}
            case['all_exact']=all(v['exact'] for v in case['comparisons'].values());save()
            if not case['all_exact']:
                np.savez_compressed(A/('failed_'+str(batch)+'_'+str(n)+'.npz'),**{method+'_'+key:value.cpu().numpy() for method,output in outputs.items() for key,value in output.items()})
                raise AssertionError('Owner64 changes finite outputs')
            del answer,outputs,x,q0,k0,q1,k1,v0,u,lse0,lse1,a,c
    r['status']='complete'
except Exception:r['status']='failed';r['error']=traceback.format_exc()
finally:save();print(json.dumps({'status':r['status'],'operator_calls':r['operator_calls'],'error':r.get('error')}),flush=True)

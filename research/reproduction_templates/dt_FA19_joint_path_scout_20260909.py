"""Three fixed real FA19 cases; joint-QKV midpoint/Gauss2 native-FA path scout.

This integrates the full attention operator's native-definition Jacobian along
one joint Q/K/V segment, instead of composing separate local finite averages.
One/two-node quadrature is approximate. No endpoint correction, mask fitting,
new forward/backward implementation, model counterfactual or quality claim.
"""
import os,json,time,hashlib,traceback,signal,zipfile,gc
from pathlib import Path
os.environ['MACA_PATH']='/opt/maca'
A=Path(__file__).resolve().parent;p=json.loads((A/'protocol.json').read_bytes());started=time.perf_counter()
r={'status':'starting','protocol':p,'native_FA_entered':0,'native_FA_returned':0,'native_autograd_entered':0,'native_autograd_returned':0,
    'model_calls':0,'DT_calls':0,'scorer_calls':0,'FT_calls':0,'generation_calls':0,'compiler_calls':0,'calls':[],'cases':{}};torch=None
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def save():
    f=A/'results.partial';f.write_text(json.dumps(r,indent=2,allow_nan=False));f.replace(A/'results.json')
def stats(x):
    assert torch.isfinite(x).all()
    return {'net':float(x.sum()),'positive':float(x.clamp_min(0).sum()),'negative':float(x.clamp_max(0).sum()),'absolute':float(x.abs().sum())}
def project(m,x):assert m.shape==x.shape;return (m.double()*x.double()).sum((1,3))
try:
    def timeout(*a):raise TimeoutError('Frozen three-case native-FA joint-path scout budget expired')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(p['budget']['wall_seconds'])
    for n,h in p['files_sha256'].items():assert digest(A/n)==h
    import torch,numpy as np,flash_attn
    import flash_attn.flash_attn_interface as fa
    torch.set_num_threads(4)
    assert flash_attn.flash_attn_func is fa.flash_attn_func and digest(fa.__file__)==p['FA_interface_sha256']
    torch.cuda.reset_peak_memory_stats();vectors={}
    assert p['nodes']==[.5,.5-3**.5/6,.5+3**.5/6]
    kwargs=dict(p['native_FA_kwargs']);kwargs['window_size']=tuple(kwargs['window_size'])
    assert kwargs['return_attn_probs'] is False and kwargs['dropout_p']==0 and kwargs['causal'] and kwargs['softmax_scale']==.0625
    for src in p['sources']:
        key=src['case'];r['status']='case_'+key;save()
        assert digest(src['private_path'])==src['private_sha256'] and digest(src['results_path'])==src['results_sha256']
        original=torch.load(src['private_path'],map_location='cpu',mmap=True,weights_only=True)
        if key=='MH0':st=original;paired=st['native']['B2']['c'];points={s:st['native'][s]['c'] for s in ['0','3','10','20']};m=st['coeff']
        elif key=='MH2':st=original['19'];paired=st['native']['B2']['c'];points={s:st['native'][s]['c'] for s in ['0','3','10','20']};m=st['coeff']
        else:st=original['FA19'];paired=st['paired']['c'];points={s:st['B1'][s]['c'] for s in ['0','3','10','20']};m=st['coeff']
        B,H,T,D=m['mcontent'].shape;KH,G=m['FA_layout']['kv_heads'],m['FA_layout']['groups'];assert B==1 and D==256 and H==KH*G
        seed=m['mcontent'].transpose(1,2).contiguous();native_seed=seed.to('cuda',dtype=torch.bfloat16)
        gpu0=[paired[n][0:1].transpose(1,2).to('cuda').contiguous() for n in ['query','key','value']]
        gpu1=[paired[n][1:2].transpose(1,2).to('cuda').contiguous() for n in ['query','key','value']]
        row={'points':{},'source':src,'sequence_length':T};r['cases'][key]=row
        tick=time.perf_counter();r['native_FA_entered']+=1
        with torch.no_grad():replay=fa.flash_attn_func(*gpu1,**kwargs)
        torch.cuda.synchronize();r['native_FA_returned']+=1
        diff=replay.cpu().double()-paired['attention_output'][1:2].double()
        row['input_endpoint_replay']={'seconds':time.perf_counter()-tick,'relative_L2':float(diff.norm()/paired['attention_output'][1:2].double().norm()),'max_absolute':float(diff.abs().max()),'projected_drift':float((seed.double()*diff).sum())};del replay,diff
        gradients=[]
        for index,node in enumerate(p['nodes']):
            entry={'case':key,'node':node,'status':'entered'};r['calls'].append(entry)
            operands=[((1-node)*x0.float()+node*x1.float()).to(torch.bfloat16).detach().requires_grad_(True) for x0,x1 in zip(gpu0,gpu1)]
            assert all(x.is_leaf and x.is_contiguous() for x in operands)
            torch.cuda.synchronize();tick=time.perf_counter();r['native_FA_entered']+=1
            output=fa.flash_attn_func(*operands,**kwargs);torch.cuda.synchronize();r['native_FA_returned']+=1
            entry['forward_seconds']=time.perf_counter()-tick;entry['autograd_function']=type(output.grad_fn).__name__
            assert output.shape==operands[0].shape and output.requires_grad and output.dtype==torch.bfloat16
            tick=time.perf_counter();r['native_autograd_entered']+=1
            grad=torch.autograd.grad(output,operands,native_seed,retain_graph=False,create_graph=False)
            torch.cuda.synchronize();r['native_autograd_returned']+=1;entry['backward_seconds']=time.perf_counter()-tick
            gradients.append({n:g.detach().transpose(1,2).float().to('cpu',copy=True) for n,g in zip(['dq','dk','dv'],grad)})
            entry['status']='returned';del grad,output,operands;save()
        current={'dq':m['FA_coeff']['dq'].double(),'dk':m['FA_coeff']['dk'].double().reshape(B,KH,G,T,D).sum(2),'dv':m['FA_coeff']['dv'].double().reshape(B,KH,G,T,D).sum(2)}
        methods={'current':current,'joint_midpoint':gradients[0],'joint_gauss2':{n:.5*(gradients[1][n]+gradients[2][n]) for n in current}}
        for step in ['3','10','20','B2']:
            left={n:v[1:2] for n,v in paired.items()} if step=='B2' else points['0']
            right={n:v[0:1] for n,v in paired.items()} if step=='B2' else points[step]
            delta={n:left[n].double()-right[n].double() for n in ['query','key','value','attention_output']}
            actual=(seed.double()*delta['attention_output']).sum((2,3));fields={'actual':actual}
            for name,coeff in methods.items():
                qk=project(coeff['dq'],delta['query'])+project(coeff['dk'],delta['key']);value=project(coeff['dv'],delta['value'])
                fields[name+'_qk']=qk;fields[name+'_value']=value;fields[name+'_prediction']=qk+value;fields[name+'_error']=qk+value-actual
            if step!='B2':assert abs(float(fields['current_error'].sum())-src['frozen_core_errors'][step])<1e-7
            row['points'][step]={'fields':{n:stats(v) for n,v in fields.items()}}
            for n,v in fields.items():vectors[key+'_'+step+'_'+n]=v.numpy()
        del methods,current,gradients,gpu0,gpu1,native_seed,seed,points,paired,m,st,original;gc.collect();save()
        assert digest(src['private_path'])==src['private_sha256'] and digest(src['results_path'])==src['results_sha256']
    assert r['native_FA_entered']==r['native_FA_returned']==12
    assert r['native_autograd_entered']==r['native_autograd_returned']==9
    np.savez_compressed(A/'vectors.npz',**vectors);r['vectors_sha256']=digest(A/'vectors.npz')
    assert digest(fa.__file__)==p['FA_interface_sha256']
    r['status']='three_FA19_joint_paths_12nativeFA9nativebackward_complete'
except Exception:r['status']='failed';r['error']=traceback.format_exc()
finally:
    signal.alarm(0);r['seconds']=time.perf_counter()-started
    if torch is not None and torch.cuda.is_initialized():r['peak_allocated_bytes']=torch.cuda.max_memory_allocated();r['peak_reserved_bytes']=torch.cuda.max_memory_reserved()
    r['limits']='One/two-node quadrature approximates the coupled real-arithmetic FA operator path; native BF16 forward/backward used without modification. Does not preserve exact finite endpoints, which are explicitly checked. Private input trajectories are actual; intermediate Q/K/V probes are operator inputs, not full-model counterfactuals. Token-net absolute statistics combine different operand/query roles and are not original-source error norms. No whole propagation or original quality/cost claim.'
    save()
    with zipfile.ZipFile(A/'review_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
        for n in [*p['files_sha256'],'protocol.json','results.json','vectors.npz']:
            if (A/n).is_file():z.write(A/n,n)
    print(json.dumps({'status':r['status'],'seconds':r['seconds'],'error':r.get('error'),'native_FA_returned':r['native_FA_returned'],'native_autograd_returned':r['native_autograd_returned']}),flush=True)

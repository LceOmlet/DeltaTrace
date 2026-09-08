"""Actual public FA input-endpoint VJP scout; no model or finite-kernel replacement."""
import os,json,time,hashlib,traceback,signal,zipfile,gc
from pathlib import Path
os.environ['MACA_PATH']='/opt/maca'
A=Path(__file__).resolve().parent;p=json.loads((A/'protocol.json').read_bytes());start=time.perf_counter()
r={'status':'starting','protocol':p,'native_FA_entered':0,'native_FA_returned':0,'native_autograd_entered':0,'native_autograd_returned':0,
   'model_calls':0,'DT_calls':0,'scorer_calls':0,'FT_calls':0,'generation_calls':0,'points':{}};torch=None
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def stats(x):
    assert torch.isfinite(x).all()
    return {'net':float(x.sum()),'positive':float(x.clamp_min(0).sum()),'negative':float(x.clamp_max(0).sum()),'absolute':float(x.abs().sum())}
def drift(x,y):
    assert x.shape==y.shape
    return {'equal':bool(torch.equal(x,y)),'max_absolute':float((x.double()-y.double()).abs().max()),'relative_L2':float((x.double()-y.double()).norm()/y.double().norm().clamp_min(1e-30))}
def project(m,x):assert m.shape==x.shape;return (m.double()*x.double()).sum((1,3))
try:
    def stop(*a):raise TimeoutError('Frozen native VJP scout budget exceeded')
    signal.signal(signal.SIGALRM,stop);signal.alarm(p['budget']['wall_seconds'])
    for n,h in p['files_sha256'].items():assert digest(A/n)==h
    for x in p['protected_sources']:assert digest(x['path'])==x['sha256']
    src=p['source'];assert digest(src['private'])==src['private_sha256']
    source=json.loads(Path(src['results']).read_bytes());assert source['status']=='MH2_FA19_GDN1_1native10replay2finite_internal_complete'
    import torch,numpy as np,flash_attn
    import flash_attn.flash_attn_interface as fa
    torch.set_num_threads(4)
    assert fa.flash_attn_func is flash_attn.flash_attn_func and digest(fa.__file__)==p['FA_interface_sha256']
    original=torch.load(src['private'],map_location='cpu',weights_only=True,mmap=True);st=original['19'];m=st['coeff'];n=st['native'];meta=m['FA_layout']
    B,H,T,D=m['mcontent'].shape;KH,G=meta['kv_heads'],meta['groups'];assert (B,H,T,D,KH)==(1,16,710,256,4) and H==KH*G
    paired=n['B2']['c'];seed=m['mcontent'].transpose(1,2).contiguous();clean={k:v[1:2] for k,v in paired.items()}
    q,k,v=[clean[name].transpose(1,2).to('cuda').contiguous().detach().requires_grad_(True) for name in ['query','key','value']]
    assert all(x.is_leaf and x.dtype==torch.bfloat16 for x in [q,k,v]);kwargs=dict(p['native_FA_kwargs']);kwargs['window_size']=tuple(kwargs['window_size'])
    assert kwargs['return_attn_probs'] is False and kwargs['dropout_p']==0 and kwargs['causal'] and kwargs['softmax_scale']==.0625
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();tick=time.perf_counter();r['native_FA_entered']+=1
    output=fa.flash_attn_func(q,k,v,**kwargs);torch.cuda.synchronize();r['native_FA_returned']+=1;r['native_FA_seconds']=time.perf_counter()-tick
    assert output.shape==q.shape and output.requires_grad and output.dtype==torch.bfloat16
    r['native_output_drift']=drift(output.detach().cpu(),clean['attention_output'])
    actual_seed=seed.to(device='cuda',dtype=torch.bfloat16);r['upstream_precision']='Stored current finite mcontent converted to native BF16, matching original finite-FA seed conversion.'
    torch.cuda.synchronize();tick=time.perf_counter();r['native_autograd_entered']+=1
    gradient=torch.autograd.grad(output,(q,k,v),actual_seed,retain_graph=False,create_graph=False);torch.cuda.synchronize();r['native_autograd_returned']+=1;r['native_autograd_seconds']=time.perf_counter()-tick
    coeff={name:g.detach().transpose(1,2).to('cpu',copy=True) for name,g in zip(['dq','dk','dv'],gradient)}
    assert coeff['dq'].shape==(1,H,T,D) and coeff['dk'].shape==coeff['dv'].shape==(1,KH,T,D)
    assert all(torch.isfinite(x).all() for x in coeff.values())
    current={'dq':m['FA_coeff']['dq'].double(),'dk':m['FA_coeff']['dk'].double().reshape(B,KH,G,T,D).sum(2),'dv':m['FA_coeff']['dv'].double().reshape(B,KH,G,T,D).sum(2)}
    vectors={};P=source['input']['prompt_length'];keep=set(source['input']['keep'])
    boundary=json.loads(Path(p['boundary_results']).read_bytes())
    for step in ['3','10','20','B2']:
        left={k:v[1:2] for k,v in paired.items()} if step=='B2' else n['0']['c']
        right={k:v[0:1] for k,v in paired.items()} if step=='B2' else n[step]['c']
        change={name:left[name].double()-right[name].double() for name in ['query','key','value','attention_output']}
        actual=(seed.double()*change['attention_output']).sum((2,3));actual_bf16=(seed.to(torch.bfloat16).double()*change['attention_output']).sum((2,3))
        fields={'actual_effect':actual,'seed_cast_effect':actual_bf16-actual}
        for label,c in [('current',current),('native_vjp',coeff)]:
            qk=project(c['dq'],change['query'])+project(c['dk'],change['key']);vp=project(c['dv'],change['value'])
            fields[label+'_qk']=qk;fields[label+'_value']=vp;fields[label+'_prediction']=qk+vp;fields[label+'_error']=qk+vp-actual
        frozen=source['layers']['19']['ledgers'][step]['internal']['terms']['finite_FA_core_including_seed_cast'];assert abs(float(fields['current_error'].sum())-frozen)<1e-7
        deleted=keep if step=='B2' else set(boundary['cases']['morehopqa_2']['points'][step]['input_receipt']['deleted_positions'])
        groups={'deleted':sorted(deleted),'kept':sorted(keep-deleted),'other_prompt':sorted(set(range(P))-keep),'response':list(range(P,T))}
        row={'fields':{name:stats(x) for name,x in fields.items()},'groups':{label:{'count':len(ix),'fields':{name:stats(x[0,ix]) for name,x in fields.items()}} for label,ix in groups.items()}}
        for name,x in fields.items():
            assert abs(sum(y['fields'][name]['net'] for y in row['groups'].values())-float(x.sum()))<1e-7
            vectors[step+'_'+name]=x.numpy()
        r['points'][step]=row
    torch.save({'native_input_VJP':coeff,'source_results_sha256':digest(src['results'])},A/'native_vjp_coefficients_private.pt')
    np.savez_compressed(A/'vectors.npz',**vectors)
    r['artifacts']={name:{'sha256':digest(A/name),'bytes':(A/name).stat().st_size} for name in ['vectors.npz','native_vjp_coefficients_private.pt']}
    for x in p['protected_sources']:assert digest(x['path'])==x['sha256']
    assert digest(src['private'])==src['private_sha256'] and digest(fa.__file__)==p['FA_interface_sha256']
    r['status']='MH2_FA19_one_public_FA_one_native_VJP_scout_complete'
except Exception:r['status']='failed';r['error']=traceback.format_exc()
finally:
    signal.alarm(0);r['seconds']=time.perf_counter()-start
    if torch is not None and torch.cuda.is_initialized():r['peak_allocated_bytes']=torch.cuda.max_memory_allocated();r['peak_reserved_bytes']=torch.cuda.max_memory_reserved()
    (A/'results.json').write_text(json.dumps(r,indent=2,allow_nan=False))
    with zipfile.ZipFile(A/'review_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
        for n in [*p['files_sha256'],'protocol.json','results.json','vectors.npz']:
            if (A/n).exists():z.write(A/n,n)
    print(json.dumps({'status':r['status'],'seconds':r['seconds'],'error':r.get('error'),'points':{s:{k:x['net'] for k,x in v['fields'].items()} for s,v in r['points'].items()}}),flush=True)

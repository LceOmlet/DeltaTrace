"""One native B2 LSE and two unchanged finite FA calls, no model or new kernel.

Reverse both real endpoints, use real V1 in the V0 slot and retain the same
upstream. Partial-deletion activations are CPU evaluation data only. Averaging
the two returned BF16 coefficient sets in FP64 is a diagnostic, not a claimed
single-call production implementation or new whole-input attribution.
"""
import gc,hashlib,json,signal,time,traceback,zipfile
from pathlib import Path
A=Path(__file__).resolve().parent;p=json.loads((A/'protocol.json').read_bytes())
started=time.perf_counter();torch=None;arrays={}
r={'status':'starting','protocol':p,'native_FA_entered':0,'native_FA_returned':0,
   'finite_FA_entered':0,'finite_FA_returned':0,'model_calls':0,'DT_calls':0,'scorer_calls':0,
   'FT_calls':0,'generation_calls':0,'compile_calls':0,'calls':[],'points':{}}

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def save():
    q=A/'results.partial';q.write_text(json.dumps(r,indent=2,allow_nan=False));q.replace(A/'results.json')
def timeout(*args):raise TimeoutError('Frozen1nativeFA2finiteFA PV-order budget expired.')
def cpu(x):return x.detach().to('cpu',copy=True)
def layout(x):return {'shape':list(x.shape),'stride':list(x.stride()),'dtype':str(x.dtype),'device':str(x.device),'contiguous':x.is_contiguous()}
def stats(x):
    x=x.double();assert x.device.type=='cpu' and bool(torch.isfinite(x).all())
    return {'net':float(x.sum()),'positive':float(x.clamp_min(0).sum()),'negative':float(x.clamp_max(0).sum()),'absolute':float(x.abs().sum())}
def difference(a,b):return {name:a[name].double()-b[name].double() for name in NAMES}
def project(c,delta,name):
    m=c[name].double()
    if name!='dq':m=m.reshape(1,KH,G,T,D).sum(2)
    assert m.shape==delta.shape and m.device.type==delta.device.type=='cpu'
    return (m*delta).sum((1,3))
def effects(c,delta):
    q=project(c,delta['query'],'dq');k=project(c,delta['key'],'dk');v=project(c,delta['value'],'dv')
    actual=(seed.double()*delta['attention_output'].transpose(1,2)).sum((1,3))
    cast=((seed.to(torch.bfloat16).double()-seed.double())*delta['attention_output'].transpose(1,2)).sum((1,3))
    return {'q_prediction':q,'k_prediction':k,'qk_prediction':q+k,'v_prediction':v,
        'captured_actual_effect':actual,'core_prediction_minus_actual':q+k+v-actual,
        'core_at_BF16_seed':q+k+v-actual-cast,'BF16_minus_stored_seed_actual_effect':cast}
def numerical_drift(now,old):
    a=now.double();b=old.double();den=float(b.norm());change=a-b
    assert a.shape==b.shape and bool(torch.isfinite(a).all())
    return {'bitwise_equal':bool(torch.equal(now,old)),'relative_L2':float(change.norm()/den) if den else None,
        'max_absolute':float(change.abs().max())}

try:
    signal.signal(signal.SIGALRM,timeout);signal.alarm(p['budget']['wall_time_seconds'])
    for name,want in p['files_sha256'].items():assert digest(A/name)==want,name
    for item in p['protected_sources']:assert digest(item['path'])==item['sha256'],item['path']
    source=json.loads(Path(p['source_results_path']).read_bytes());boundary=json.loads(Path(p['boundary_results_path']).read_bytes())
    assert source['status']=='MH0_current_FA19_1native_kwargs5replay1finite_internal_complete'
    assert source['input']==boundary['cases']['morehopqa_0']['input']
    assert p['frozen_input_receipts']=={s:boundary['cases']['morehopqa_0']['points'][s]['input_receipt'] for s in ('0','3','10','20')}
    artifact=Path(p['private_artifact_path']);assert artifact.stat().st_size==p['private_artifact_bytes']
    assert digest(artifact)==p['private_artifact_sha256']==source['private_artifact']['sha256']
    import numpy as np
    import torch,flash_attn
    import flash_attn.flash_attn_interface as fa
    from vendor_fa_finite_bf16_d256 import VendorFAFiniteP1BF16D256,RightPaddedLengths
    torch.set_num_threads(4)
    assert flash_attn.flash_attn_func is fa.flash_attn_func and digest(fa.__file__)==p['installed_FA_interface_sha256']
    tick=time.perf_counter();private=torch.load(artifact,map_location='cpu',weights_only=True);r['CPU_load_seconds']=time.perf_counter()-tick
    coeff=private['coeff'];seed=coeff['mcontent'].clone();saved={n:coeff['FA_coeff'][n].clone() for n in ('dq','dk','dv')}
    meta=coeff['FA_layout'];B,H,T,D=seed.shape;KH=meta['kv_heads'];G=meta['groups']
    assert (B,H,KH,T,D)==(1,16,4,853,256) and H==KH*G
    assert meta['query_heads']==H and meta['padded_length']==T and meta['valid_lengths']==[T]
    assert meta['mapping']=='query_head // groups = compact_kv_head'
    NAMES=('query','key','value','attention_output')
    paired={name:private['native']['B2']['c'][name].clone() for name in NAMES}
    points={s:{name:private['native'][s]['c'][name].clone() for name in NAMES} for s in ('0','3','10','20')}
    assert paired['query'].shape==(2,H,T,D) and paired['key'].shape==paired['value'].shape==(2,KH,T,D)
    assert paired['attention_output'].shape==(2,T,H,D) and seed.dtype==torch.float32
    assert all(x.dtype==torch.bfloat16 for x in paired.values())
    assert all(saved[n].shape==(1,H,T,D) for n in saved)
    r['input']=source['input'];r['FA_layout']=meta
    del private,coeff;gc.collect()
    r['endpoint_controls']={}
    for label,s,index in [('B1clean_vs_B2input','0',1),('B1allEOS_vs_B2baseline','20',0)]:
        r['endpoint_controls'][label]={name:numerical_drift(points[s][name],paired[name][index:index+1]) for name in NAMES}
    # Actual partial inputs are not moved to GPU or supplied to either operator.
    assert torch.cuda.is_available();torch.cuda.set_device(0);torch.cuda.reset_peak_memory_stats()
    endpoints={name:value.to('cuda').contiguous() for name,value in paired.items() if name!='attention_output'}
    native_qkv=[endpoints[name].transpose(1,2).contiguous() for name in ('query','key','value')]
    assert all(x.dtype==torch.bfloat16 and x.is_cuda and x.is_contiguous() for x in native_qkv)
    kwargs=dict(p['native_FA_kwargs']);kwargs['window_size']=tuple(kwargs['window_size']);kwargs['return_attn_probs']=True
    assert kwargs['dropout_p']==0 and kwargs['causal'] is True and kwargs['softmax_scale']==p['scale']
    r['public_B2_operand_layouts']=[layout(x) for x in native_qkv]
    r['status']='one_public_B2_FA_for_shared_LSE';save();torch.cuda.synchronize();tick=time.perf_counter()
    r['native_FA_entered']+=1
    try:
        with torch.no_grad():out,lse,unused=fa.flash_attn_func(*native_qkv,**kwargs)
        torch.cuda.synchronize();r['native_FA_returned']+=1
    finally:r['public_B2_FA_seconds']=time.perf_counter()-tick;save()
    assert lse.shape==(2,H,T) and lse.dtype==torch.float32 and out.shape==(2,T,H,D)
    r['public_FA_unused_return_kind']='None' if unused is None else type(unused).__name__
    assert unused is None or isinstance(unused,torch.Tensor)
    r['public_FA_unused_return_numel']=0 if unused is None else unused.numel()
    assert r['public_FA_unused_return_numel']==0,'Unexpected explicit attention probability output.'
    lse= lse.contiguous();public_lse=cpu(lse)
    r['public_B2_output_drift']=numerical_drift(cpu(out),paired['attention_output'])
    del out,unused,native_qkv
    u=seed.to('cuda').contiguous()
    control={'q0':endpoints['query'][0:1],'q1':endpoints['query'][1:2],
        'k0':endpoints['key'][0:1],'k1':endpoints['key'][1:2],'v0':endpoints['value'][0:1],
        'u':u,'lse0':lse[0:1],'lse1':lse[1:2]}
    reverse={'q0':control['q1'],'q1':control['q0'],'k0':control['k1'],'k1':control['k0'],
        'v0':endpoints['value'][1:2],'u':u,'lse0':control['lse1'],'lse1':control['lse0']}
    assert reverse['u'] is control['u'] and reverse['q0'] is control['q1'] and reverse['lse0'] is control['lse1']
    op=VendorFAFiniteP1BF16D256(p['finite_FA_library'],p['finite_FA_library_sha256']);lengths=RightPaddedLengths([T],T,u.device)
    methods={'saved_current':saved};raw_returns={}
    for name,operands in [('same_LSE_control',control),('reversed',reverse)]:
        assert r['finite_FA_entered']<2
        assert all(value.is_contiguous() for value in operands.values())
        activity={};call={'name':name,'status':'entered','operand_layouts':{k:layout(v) for k,v in operands.items()},'activity':activity}
        r['calls'].append(call);r['finite_FA_entered']+=1;torch.cuda.synchronize();tick=time.perf_counter();save()
        try:
            with torch.no_grad():result=op(operands,p['scale'],lengths,activity)
            torch.cuda.synchronize();r['finite_FA_returned']+=1;call['status']='returned'
            raw_returns[name]={k:cpu(v) for k,v in result.items()}
            assert all(bool(torch.isfinite(x).all()) for x in raw_returns[name].values())
            methods[name]={n:raw_returns[name][n] for n in ('dq','dk','dv')}
            assert all(x.dtype==torch.bfloat16 for x in methods[name].values())
            assert activity['calls_attempted']==activity['calls_enqueued']==1 and activity['kernel_launches_per_call']==3
        finally:call['seconds_including_CPU_copy']=time.perf_counter()-tick;save()
    methods['diagnostic_average']={n:(methods['same_LSE_control'][n].double()+methods['reversed'][n].double())*.5 for n in ('dq','dk','dv')}
    r['same_LSE_control_vs_saved_coefficient_drift']={n:numerical_drift(methods['same_LSE_control'][n],saved[n]) for n in saved}
    r['reversed_vs_control_coefficient_drift']={n:numerical_drift(methods['reversed'][n],methods['same_LSE_control'][n]) for n in saved}
    P=source['input']['prompt_length'];keep=set(source['input']['keep'])
    for step in ('3','10','20','B2'):
        left={name:paired[name][1:2] for name in NAMES} if step=='B2' else points['0']
        right={name:paired[name][0:1] for name in NAMES} if step=='B2' else points[step]
        delta=difference(left,right);fields={name:effects(c,delta) for name,c in methods.items()}
        original=source['conditional_ledgers'][step]['replayed_9term_ledger']['terms']['finite_FA_core_including_seed_cast']
        assert abs(float(fields['saved_current']['core_prediction_minus_actual'].sum())-original)<1e-7
        receipt=p['frozen_input_receipts']['20' if step=='B2' else step];deleted=set(receipt['deleted_positions'])
        groups={'deleted':sorted(deleted),'kept':sorted(keep-deleted),'other_prompt':sorted(set(range(P))-keep),'response':list(range(P,T))}
        assert sum(map(len,groups.values()))==T
        row={'input_receipt':receipt,'actual_pair':'B2input_minus_B2EOS' if step=='B2' else 'B1clean_minus_B1deleted',
            'original_saved_core':original,'methods':{},'checks':{}}
        for name,values in fields.items():
            closed=values['qk_prediction']+values['v_prediction']-values['captured_actual_effect']-values['core_prediction_minus_actual']
            assert float(closed.abs().max())<1e-7
            row['methods'][name]={'fields':{k:stats(v) for k,v in values.items()},
                'groups':{g:{'count':len(ix),'fields':{k:stats(v[0,ix]) for k,v in values.items()}} for g,ix in groups.items()}}
            for k,v in values.items():
                assert abs(sum(gr['fields'][k]['net'] for gr in row['methods'][name]['groups'].values())-float(v.sum()))<1e-7
                arrays[step+'_'+name+'_'+k]=v.numpy()
        transfers={k:fields['same_LSE_control'][k]-fields['saved_current'][k] for k in ('qk_prediction','v_prediction','core_prediction_minus_actual')}
        row['same_LSE_control_minus_saved_numerical_transfer']={k:stats(v) for k,v in transfers.items()}
        for name in ('reversed','diagnostic_average'):
            row[name+'_minus_same_LSE_control']={k:stats(fields[name][k]-fields['same_LSE_control'][k]) for k in ('qk_prediction','v_prediction','core_prediction_minus_actual')}
        mean=(fields['same_LSE_control']['core_prediction_minus_actual']+fields['reversed']['core_prediction_minus_actual'])*.5
        error=fields['diagnostic_average']['core_prediction_minus_actual']-mean
        assert float(error.abs().max())<1e-7
        row['checks']['max_token_average_closure']=float(error.abs().max())
        r['points'][step]=row;save()
    assert r['native_FA_entered']==r['native_FA_returned']==1 and r['finite_FA_entered']==r['finite_FA_returned']==2
    private_out={'methods':methods,'raw_returned_control_and_reverse':raw_returns,'public_B2_LSE':public_lse,
        'source_private_sha256':p['private_artifact_sha256'],'source_protocol_sha256':p['source_protocol_sha256'],
        'scope':'Actual existing backend coefficients; FP64 mean is diagnostic only. Original source B2 operands/upstream are pinned remotely, not duplicated.'}
    path=A/'PV_orders_coefficients_private.pt';tick=time.perf_counter();torch.save(private_out,path)
    r['private_coefficients']={'file':path.name,'sha256':digest(path),'bytes':path.stat().st_size,'save_seconds':time.perf_counter()-tick}
    np.savez_compressed(A/'signed_token_contrasts.npz',**arrays)
    r['signed_token_contrasts']={'file':'signed_token_contrasts.npz','sha256':digest(A/'signed_token_contrasts.npz')}
    assert digest(artifact)==p['private_artifact_sha256'] and digest(fa.__file__)==p['installed_FA_interface_sha256']
    for item in p['protected_sources']:assert digest(item['path'])==item['sha256']
    r['status']='MH0_FA19_PV_orders_1publicFA2finiteFA_local_complete'
except Exception:r['status']='failed';r['error']=traceback.format_exc()
finally:
    signal.alarm(0);r['seconds']=time.perf_counter()-started
    r['kernel_phase_accounting']={'phases_from_successfully_enqueued_calls':3*sum(x['activity'].get('calls_enqueued',0) for x in r['calls']),
        'phases_inside_unreturned_calls':'unknown' if r['finite_FA_entered']!=r['finite_FA_returned'] else 0}
    if torch is not None and torch.cuda.is_initialized():r['GPU_peak_allocated']=torch.cuda.max_memory_allocated();r['GPU_peak_reserved']=torch.cuda.max_memory_reserved()
    r['scope']='Two actual endpoint operands only create all candidate coefficients; actual deletion coordinates enter CPU evaluation only. Same fresh public LSE controls original/reverse backend calls; saved/current replay drift is separate. No new kernel, explicit attention probabilities, model forward, complete DT, native backward, scorer, FT, generation or original metric. Local averages do not prove whole-input or MAS improvement.'
    save()
    with zipfile.ZipFile(A/'review_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in [*p['files_sha256'],'protocol.json','results.json','signed_token_contrasts.npz']:
            if (A/name).is_file():z.write(A/name,name)
    print(json.dumps({'status':r['status'],'native_FA_entered':r['native_FA_entered'],'native_FA_returned':r['native_FA_returned'],
        'finite_FA_entered':r['finite_FA_entered'],'finite_FA_returned':r['finite_FA_returned'],'seconds':r['seconds'],'error':r.get('error')}),flush=True)

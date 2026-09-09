"""Scoped passive FA capture and existing public-FA operator contrasts.

No model/attention/backward replacement; hybrid operands are operator probes,
never claimed to be complete-model counterfactual states.
"""
import time
import torch
import decoder19_conditional_decomposition_20260908 as ledger
from native_dense_attention_capture import NativeDenseAttentionCapture

class ScopedNativeDenseCapture(NativeDenseAttentionCapture):
    """Limit the existing dense observer to this module's native interface call."""
    def __init__(self,*a,**kw):super().__init__(*a,**kw);self.inside=False
    def profile(self,frame,event,result):
        target=frame.f_code is self.interface.__code__ and frame.f_locals.get('module') is self.module
        if target and event=='call':assert not self.inside;self.inside=True
        if self.inside:super().profile(frame,event,result)
        if target and event=='return':self.inside=False

def stats(x):
    assert torch.isfinite(x).all()
    return {'net':float(x.sum()),'positive':float(x.clamp_min(0).sum()),'negative':float(x.clamp_max(0).sum()),'absolute':float(x.abs().sum())}

def project(m,x):
    assert m.shape==x.shape and m.device.type==x.device.type=='cpu'
    return (m.double()*x.double()).flatten(2).sum(-1).squeeze(0)

def analyze(capture,receipts,results,vectors):
    """Four exact-operand replays and seven hybrids, all public default BF16 FA."""
    import flash_attn.flash_attn_interface as fa
    c=capture['coeff'];points=capture['B1'];paired=capture['paired']
    B,H,T,D=c['mcontent'].shape;meta=c['FA_layout'];KH,G=meta['kv_heads'],meta['groups']
    assert B==1 and D==256 and H==KH*G
    seed=c['mcontent'].transpose(1,2).contiguous()
    dq=c['FA_coeff']['dq'].transpose(1,2)
    dk=c['FA_coeff']['dk'].double().reshape(B,KH,G,T,D).sum(2).transpose(1,2)
    dv=c['FA_coeff']['dv'].double().reshape(B,KH,G,T,D).sum(2).transpose(1,2)
    kwargs=capture['B1_native_arguments']['0'];assert all(v==kwargs for v in capture['B1_native_arguments'].values())
    assert kwargs=={'dropout_p':0.0,'softmax_scale':256**-.5,'causal':True,'return_attn_probs':False}
    out={'calls':[],'native_FA_entered':0,'native_FA_returned':0,'points':{},'replays':{},'B1_B2_drift':{},'native_arguments':kwargs}
    # Expose the mutable counters before calls so even a failed probe is counted.
    results['FA19_native_probe']=out
    gpu={step:tuple(value['c'][n].transpose(1,2).to('cuda').contiguous() for n in ['query','key','value']) for step,value in points.items()}
    value0=paired['c']['value'][0:1].transpose(1,2).to('cuda').contiguous()
    schedule=[*['replay_'+s for s in ['0','3','10','20']],'C0',*sum(([f'A0_{s}',f'CA_{s}'] for s in ['3','10','20']),[])]
    def call(label,operands):
        assert out['native_FA_entered']<11 and label==schedule[out['native_FA_entered']]
        out['native_FA_entered']+=1;tick=time.perf_counter();row={'label':label,'status':'entered'};out['calls'].append(row)
        try:
            with torch.no_grad():y=fa.flash_attn_func(*operands,**kwargs)
            torch.cuda.synchronize();assert y.dtype==torch.bfloat16 and y.shape==(1,T,H,D)
            y=y.detach().to('cpu',copy=True);out['native_FA_returned']+=1;row['status']='returned';return y
        finally:row['seconds']=time.perf_counter()-tick
    for step in ['0','3','10','20']:
        replay=call('replay_'+step,gpu[step]);actual=points[step]['c']['attention_output'];diff=replay.double()-actual.double()
        out['replays'][step]={'equal':torch.equal(replay,actual),'relative_L2':float(diff.norm()/actual.double().norm()),'max_absolute':float(diff.abs().max()),'projected_drift':stats(project(seed,diff))}
    for label,step,index in [('clean','0',1),('EOS','20',0)]:
        out['B1_B2_drift'][label]={}
        for name in ['query','key','value','attention_output']:
            actual=points[step]['c'][name].double();end=paired['c'][name][index:index+1].double()
            out['B1_B2_drift'][label][name]={'relative_L2':float((actual-end).norm()/end.norm()),'max_absolute':float((actual-end).abs().max())}
    xc0=call('C0',(gpu['0'][0],gpu['0'][1],value0));oc=points['0']['c']['attention_output']
    for step in ['3','10','20']:
        d=ledger.difference(points['0'],points[step]);internal=ledger.decompose(c,d)
        xa0=call('A0_'+step,(gpu[step][0],gpu[step][1],value0));xca=call('CA_'+step,(gpu['0'][0],gpu['0'][1],gpu[step][2]));oa=points[step]['c']['attention_output']
        qk=project(dq,d['c']['query'].transpose(1,2))+project(dk,d['c']['key'].transpose(1,2))
        v=project(dv,d['c']['value'].transpose(1,2));actual=project(seed,oc.double()-oa.double())
        r0=project(seed,xc0.double()-xa0.double());ra=project(seed,xca.double()-oa.double());content=project(seed,oc.double()-xca.double())
        fields={'qk_prediction':qk,'v_prediction':v,'actual':actual,'R0':r0,'RA':ra,'content_actual':content,
            'route_error_at_V0':qk-r0,'content_error_at_PC':v-content,'reference_interaction':r0-ra,'core_error':qk+v-actual}
        assert abs(float(fields['core_error'].sum())-internal['terms']['finite_FA_core_including_seed_cast'])<1e-7
        assert (fields['route_error_at_V0']+fields['content_error_at_PC']+fields['reference_interaction']-fields['core_error']).abs().max()<1e-7
        fields['BF16_seed_delta']=project(seed.to(torch.bfloat16).double()-seed.double(),oc.double()-oa.double())
        out['points'][step]={'input_receipt':receipts[step],'internal':internal,'fields':{n:stats(v) for n,v in fields.items()}}
        for n,v in fields.items():vectors['FA19_'+step+'_'+n]=v.numpy()
    assert out['native_FA_entered']==out['native_FA_returned']==11
    out['scope']='All calls use the installed public FA on real saved operands or labeled hybrids. Core terms mix query- and operand-location contractions; only scalar sums form a causal-error decomposition. Their token absolute sums are not original-source absolute attribution errors. No explicit probability matrices, backward or new quality curve.'
    return out

"""Passive native K capture and CPU-only conditional contractions; no forward replacement."""
import torch
from qwen35_gdn_finite import NativeGDNCapture
from input_supported_l2_20260909 import input_supported_l2_pullback

class NativeFreshKCapture(NativeGDNCapture):
    """Reuse native code-object discovery, retaining only actual K boundary tensors."""
    def event(self,frame,kind,value):
        label=self.codes.get(frame.f_code);f=frame.f_locals
        if label=='module' and kind=='call' and f['self'] is self.module:
            assert not self.active and not self.values
            cache=f.get('cache_params')
            previous=False if cache is None else bool(cache.has_previous_state(self.module.layer_idx))
            assert not previous and not f.get('kwargs',{}).get('cu_seq_lens_q')
            self.cache_receipt={'provided':cache is not None,'has_previous_state':previous}
            self.active=True
        if not self.active:return
        if kind=='call' and label:self.calls[label]=self.calls.get(label,0)+1
        if kind=='call' and label=='FLA':self.values['raw_k']=self.copy(f['k'])
        if kind=='return' and label=='stage' and value is not None:
            assert f['initial_state'] is None and f['cu_seqlens'] is None
            self.endpoints['k']=self.copy(f['k'])
        if kind=='return' and label=='module' and f['self'] is self.module:
            assert value is not None;self.active=False

def stats(x):
    assert torch.isfinite(x).all()
    return {'net':float(x.sum()),'positive':float(x.clamp_min(0).sum()),
        'negative':float(x.clamp_max(0).sum()),'absolute':float(x.abs().sum()),'max_absolute':float(x.abs().max())}

def analyze(k,prompt_length):
    saved=k['mk'].double();B,T,H,D=saved.shape
    repeat=k['raw_k'].shape[2]//H
    def compact(x):
        x=x.double().reshape(B,T,H,repeat,D)
        assert torch.equal(x,x[:,:,:,:1].expand_as(x))
        return x[:,:,:,0]
    x0,x1=[compact(k['raw_k'][i:i+1]) for i in [0,1]]
    coeff=k['coeff'].double();f=coeff.reshape(B,T,H,repeat,D).sum(3)
    radius=lambda x:(x.square().sum(-1,keepdim=True)+1e-6).sqrt()
    r0,r1=radius(x0),radius(x1);d=x1-x0
    candidate=input_supported_l2_pullback(x0,x1,f)
    candidate32=input_supported_l2_pullback(k['raw_k'][0:1].float(),k['raw_k'][1:2].float(),k['coeff'].float()).reshape(B,T,H,repeat,D).sum(3).double()
    scale01=.5*(1/r0+1/r1);bar01=.5*(x0+x1)
    radial01=-2*bar01*(bar01*f).sum(-1,keepdim=True)/(r0*r1*(r0+r1))
    analytic_current=f*scale01+radial01
    endpoint=(f*(x1/r1-x0/r0)).sum(-1)
    closure=(candidate*d).sum(-1)-endpoint
    assert closure.abs().max()<1e-7
    vectors={};points={}
    for step in ['3','10','20']:
        xC=compact(k['B1']['0']['raw_k']);xA=compact(k['B1'][step]['raw_k']);delta=xC-xA
        rC,rA=radius(xC),radius(xA);barCA=.5*(xC+xA)
        actual=(coeff*(k['B1']['0']['k'].double()-k['B1'][step]['k'].double())).sum(-1).reshape(B,T,H,repeat).sum(-1)
        native_real=(f*(xC/rC-xA/rA)).sum(-1)
        scaleCA=.5*(1/rC+1/rA)
        radialCA=-2*barCA*(barCA*f).sum(-1,keepdim=True)/(rC*rA*(rC+rA))
        fields={'actual_native':actual,'actual_real':native_real,
            'current_prediction':(saved*delta).sum(-1),
            'candidate_prediction':(candidate*delta).sum(-1),
            'candidate32_prediction':(candidate32*delta).sum(-1),
            'scale_error':(f*(scale01-scaleCA)*delta).sum(-1),
            'radial_error':((radial01-radialCA)*delta).sum(-1),
            'native_precision':native_real-actual,
            'current_coefficient_precision':((saved-analytic_current)*delta).sum(-1)}
        for method in ['current','candidate','candidate32']:fields[method+'_error']=fields[method+'_prediction']-actual
        assert (fields['current_error']-sum(fields[n] for n in ['scale_error','radial_error','native_precision','current_coefficient_precision'])).abs().max()<1e-7
        fields['candidate_minus_current']=fields['candidate_prediction']-fields['current_prediction']
        points[step]={'fields':{n:stats(x) for n,x in fields.items()},'groups':{
            name:{n:stats(x[:,sl]) for n,x in fields.items()} for name,sl in [('prompt',slice(0,prompt_length)),('response',slice(prompt_length,T))]}}
        for name,value in fields.items():vectors['K_'+step+'_'+name]=value.numpy()
    # Small raw K tensors permit an independent CPU formula check without model weights.
    for name,value in {'x0':x0,'x1':x1,'f':f,'current_m':saved,'candidate_m':candidate,'candidate32_m':candidate32}.items():vectors['K_raw_'+name]=value.numpy()
    for step,values in k['B1'].items():vectors['K_raw_x'+step]=compact(values['raw_k']).numpy()
    return {'points':points,'endpoint_closure_max':float(closure.abs().max()),
        'candidate32_vs64_relative_L2':float((candidate32-candidate).norm()/candidate.norm()),
        'coordinate_scope':'same token and compact K head; normalized coefficients sum repeated heads, raw heads verified equal',
        'comparison_scope':'Candidate contracts the same current upstream and actual partial K changes; no candidate model forward, no lower-layer candidate propagation.'},vectors

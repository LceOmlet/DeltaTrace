"""Passive paired-coefficient diagnostic; original B1 states are reused, not synthesized."""
import hashlib,json
from pathlib import Path
import torch

class Observer:
    def __init__(self):self.coeff={};self.paired={};self.k={}
    def boundary(self,name,m,x):
        if name not in ['0','1','2']:return
        self.coeff[name]=m.detach().to('cpu',copy=True);self.paired[name]=x.detach().to('cpu',copy=True)
    def wants_decoder(self,i):return i==1
    def decoder(self,i,d,c,e,m,new,terms):
        assert i==1 and not self.k
        self.k={n:v.detach().to('cpu',copy=True) for n,v in {'raw':c['raw_k'],'native':e['k'],'upstream':terms['mixer']['coeff']['k'],'mk':terms['mixer']['mk']}.items()}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def stats(x):
    assert torch.isfinite(x).all()
    return {'net':float(x.sum()),'positive':float(x.clamp_min(0).sum()),'negative':float(x.clamp_max(0).sum()),'absolute':float(x.abs().sum())}

def compare(observers,vectors,p):
    src=p['native_state_source'];assert sha(src['private_path'])==src['private_sha256'] and sha(src['results_path'])==src['results_sha256']
    private=torch.load(src['private_path'],map_location='cpu',mmap=True,weights_only=True)
    source=json.loads(Path(src['results_path']).read_bytes());sc=source['cases']['morehopqa_3']
    C,K=[observers[n] for n in ['control','candidate']]
    assert torch.equal(C.coeff['2'],K.coeff['2'])
    for n in ['raw','native','upstream']:assert torch.equal(C.k[n],K.k[n]),n
    for n in ['0','1','2']:assert torch.equal(C.paired[n],K.paired[n])
    B,T,H,D=C.k['mk'].shape;repeat=C.k['raw'].shape[2]//H
    def compact(x):
        z=x.double().reshape(B,T,H,repeat,D);assert torch.equal(z,z[:,:,:,:1].expand_as(z));return z[:,:,:,0]
    out={'same_process_upstream_and_native_endpoints_equal':True,'points':{},'source_drift':{},'sign_convention':'candidate minus current prediction; decomposition is an exact change ledger, not an absolute error repair proof'};zout={}
    for name in ['0','1','2']:
        old=private['coefficients'][name].double();new=C.coeff[name].double()
        out['source_drift'][name]={'coefficient_relative_L2':float((new-old).norm()/old.norm()),
            'B2_native_boundary_max_absolute':float((C.paired[name].double()-private['paired_native_boundaries'][name].double()).abs().max())}
    curve=source['protocol']['frozen_capture_receipts']
    for step in ['3','10','20']:
        raw=compact(private['K']['B1']['0']['raw_k'])-compact(private['K']['B1'][step]['raw_k'])
        fields={}
        for n in ['0','1','2']:
            delta=private['B1_native_boundaries']['0'][n].double()-private['B1_native_boundaries'][step][n].double()
            fields['boundary_'+n]=((K.coeff[n].double()-C.coeff[n].double())*delta).sum(-1).squeeze(0)
        fields['K_norm']=((K.k['mk'].double()-C.k['mk'].double())*raw).sum((-1,-2)).squeeze(0)
        fields['GDN1_below_K']=fields['boundary_1']-fields['K_norm']
        fields['GDN0']=fields['boundary_0']-fields['boundary_1']
        ids=curve[step]['deleted_positions']
        cw=torch.from_numpy(vectors['morehopqa_3_control_run0_evaluated']).double()
        kw=torch.from_numpy(vectors['morehopqa_3_candidate_run1_evaluated']).double()
        change=float((kw-cw)[ids].sum());input_map=change-float(fields['boundary_0'].sum())
        terms={n:float(fields[n].sum()) for n in ['K_norm','GDN1_below_K','GDN0']};terms['input_map_and_score_storage']=input_map
        assert abs(sum(terms.values())-change)<1e-8
        actual=sc['points']['0']['original_native_score']-sc['points'][step]['original_native_score']
        out['points'][step]={'prediction_change':change,'terms':terms,'fields':{n:stats(x) for n,x in fields.items()},
            'current_prediction':float(cw[ids].sum()),'candidate_prediction':float(kw[ids].sum()),
            'prior_actual_drop':actual,'current_error_using_prior_score':float(cw[ids].sum())-actual,
            'candidate_error_using_prior_score':float(kw[ids].sum())-actual,'telescoping_error':sum(terms.values())-change}
        for n,x in fields.items():zout[step+'_'+n]=x.numpy()
    out['limits']='No new scorer calls; prior native actual states/score remain source-bound. C/K matched exactly within this pair before K; cross-process drift is reported. Individual coordinate ledger terms are not original-token causal assignments.'
    assert sha(src['private_path'])==src['private_sha256']
    return out,zout

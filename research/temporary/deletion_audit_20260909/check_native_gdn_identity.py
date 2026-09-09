"""Two-step shared-key identity: actual native FLA and frozen finite pullback.

Small operator audit only. The explicit FP64 recurrence below is an analytic
reference, never used to compute model outputs or attribution coefficients.
"""
import argparse,hashlib,inspect,json,os,sys
from pathlib import Path
sha=lambda b:hashlib.sha256(b).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('clean','environment','output'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();env=json.loads(args.environment.read_bytes())['qwen35']
    os.environ.setdefault('MACA_PATH','/opt/maca')
    for path in env.get('dependency_overlays',[]):sys.path.insert(0,path)
    sys.path.insert(0,str(args.clean))
    import torch
    import torch.nn.functional as F
    import fla.ops.gated_delta_rule.chunk as chunk
    from finite_fla_gpu import finite_fla_pullback,verify_native_sources
    from signed_secant_rules import rmsnorm_secant_pullback
    from normalization_geometry import rmsnorm_geometry_pullback
    verify_native_sources(env['native_stage_source_sha256'])
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    source=Path(args.clean/'signed_secant_rules.py')
    x0=torch.tensor([[3.,4.]],dtype=torch.float64);x1=torch.tensor([[0.,10.]],dtype=torch.float64)
    out={'scope':'independent operator audit, not NI benchmark','native_FLA_sources':env['native_stage_source_sha256'],
        'frozen_normalization_sha256':sha(source.read_bytes()),'cases':[]}
    for epsilon in (0.,1e-6):
        norm=lambda x:x/(x.square().sum(-1,keepdim=True)+epsilon).sqrt()
        n0,n1=norm(x0),norm(x1)
        # Explicit two-step GDN: beta=(.5,1), v=(1,0), decay=(1,1), q=(0,1).
        def analytic(x):
            n=norm(x)[0];h=.5*n;h=h+n*(0-torch.dot(n,h));return h[1]
        combined=.5*((1-n0.square().sum())*torch.tensor([[0.,1.]],dtype=torch.float64)-n1[0,1]*(n0+n1))
        c_old=rmsnorm_secant_pullback(x0,x1,2**-.5,combined,epsilon/2)
        c_new=rmsnorm_geometry_pullback(x0,x1,2**-.5,combined,epsilon/2)
        assert abs(float((c_old*(x1-x0)).sum()-(analytic(x1)-analytic(x0))))<1e-12
        assert abs(float((c_new*(x1-x0)).sum()-(analytic(x1)-analytic(x0))))<1e-12
        row={'epsilon':epsilon,'analytic_outputs':[float(analytic(torch.tensor([[a,b]],dtype=torch.float64))) for a in (3.,0.) for b in (4.,10.)],
             'analytic_old_source_allocation':(c_old*(x1-x0)).tolist(),
             'analytic_geometry_source_allocation':(c_new*(x1-x0)).tolist()}
        # Use the original public FLA function and passively retain its actual
        # chunk operands. Default BF16 precision is retained, including drift.
        e={};capture_code=inspect.unwrap(chunk.chunk_gated_delta_rule_fwd).__code__
        def capture(frame,event,value):
            if frame.f_code is capture_code and event=='return' and value is not None:
                for name in ('q','k','v','g','beta','A','w','v_new','o','h'):e[name]=frame.f_locals[name].detach().clone()
        key=F.pad(torch.cat((n0,n1)),(0,126)).to(device='cuda',dtype=torch.bfloat16)
        k=key[:,None,None,:].expand(2,2,1,128).contiguous()
        q=torch.zeros_like(k);q[:,1,0,1]=1
        v=torch.zeros_like(k);v[:,0,0,0]=1
        beta=torch.tensor([.5,1.],device='cuda',dtype=torch.float32)[None,:,None].expand(2,-1,-1).contiguous()
        g=torch.zeros((2,2,1),device='cuda',dtype=torch.float32)
        assert sys.getprofile() is None;sys.setprofile(capture)
        try:
            with torch.no_grad():native,_=chunk.chunk_gated_delta_rule(q,k,v,g,beta,scale=1.,use_qk_l2norm_in_kernel=False)
        finally:sys.setprofile(None)
        assert len(e)==10;e['raw_g']=g
        upstream=torch.zeros((1,2,1,128),device='cuda',dtype=torch.bfloat16);upstream[0,1,0,0]=1
        with torch.no_grad():coeff=finite_fla_pullback(e,upstream,1.)
        # The same source supplies BOTH normalized keys; accumulate both uses.
        mk=coeff['k'][0,:,0].sum(0,keepdim=True).double().cpu()
        padded0=F.pad(x0,(0,126));padded1=F.pad(x1,(0,126))
        old=rmsnorm_secant_pullback(padded0,padded1,128**-.5,mk,epsilon/128)
        new=rmsnorm_geometry_pullback(padded0,padded1,128**-.5,mk,epsilon/128)
        row['native_outputs']=native[:,1,0,0].float().cpu().tolist()
        row['native_finite_key_upstream']=mk[:,:2].tolist()
        row['native_finite_old_source_allocation']=(old*(padded1-padded0))[:,:2].tolist()
        row['native_finite_geometry_source_allocation']=(new*(padded1-padded0))[:,:2].tolist()
        row['note']='BF16 native output is not an exact constant; analytic epsilon limit and native rounding are reported separately.'
        out['cases'].append(row)
    # Finite closure, endpoint-swap symmetry, equal-endpoint Jacobian, and the
    # constant-norm identity across unequal radii, dimensions and epsilon.
    generator=torch.Generator().manual_seed(20260909);errors=[]
    for dimension in (2,7,128):
        for epsilon in (0.,1e-6,.01):
            for repeat in range(8):
                a=torch.randn((3,dimension),dtype=torch.float64,generator=generator)
                b=torch.randn((3,dimension),dtype=torch.float64,generator=generator)*(repeat+1)
                u=torch.randn((3,dimension),dtype=torch.float64,generator=generator)
                weight=torch.randn(dimension,dtype=torch.float64,generator=generator)
                forward=lambda x:weight*x/(x.square().mean(-1,keepdim=True)+epsilon).sqrt()
                c=rmsnorm_geometry_pullback(a,b,weight,u,epsilon)
                closure=((c*(b-a)).sum(-1)-((forward(b)-forward(a))*u).sum(-1)).abs().max()
                swapped=rmsnorm_geometry_pullback(b,a,weight,u,epsilon)
                x=a.clone().requires_grad_(True);truth=torch.autograd.grad((forward(x)*u).sum(),x)[0]
                coincident=rmsnorm_geometry_pullback(a,a,weight,u,epsilon)
                errors.extend([float(closure),float((c-swapped).abs().max()),float((truth-coincident).abs().max())])
    out['geometry_max_algebra_error']=max(errors);assert max(errors)<1e-11
    args.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))

if __name__=='__main__':main()

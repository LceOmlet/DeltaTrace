"""Exercise finite RMS/SiLU at coincident endpoints in the unchanged FLA test.

Only input adjoints use DT, exactly the interface used by GDN finite
propagation. FLA retains forward and weight gradients. Its original FP32
reference and 1e-3 input-gradient tolerances remain unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from fla.modules.fused_norm_gate import FusedRMSNormGated, rms_norm_gated
from qwen35_decoder_finite import FiniteBoundaryOps
from verify_official_kernel_tolerances import load


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    assert hashlib.sha256(args.source.read_bytes()).hexdigest()=='6e52f344d00b9995b2078968766f30f02e567f5e26d69813fa543020dbcd4e22'
    test=load('official_norm_test',args.source)
    boundary=FiniteBoundaryOps(True,dynamic_shapes=True).gdn_norm_gate

    class FiniteInputs(torch.autograd.Function):
        @staticmethod
        def forward(ctx,x,g,weight,eps):
            with torch.enable_grad():
                xx=x.detach().requires_grad_();gg=g.detach().requires_grad_()
                ww=weight.detach().requires_grad_()
                out=rms_norm_gated(xx,gg,ww,None,activation='silu',eps=eps)
            ctx.save_for_backward(xx,gg,ww,out)
            ctx.eps=eps
            return out.detach()

        @staticmethod
        def backward(ctx,upstream):
            x,g,weight,out=ctx.saved_tensors
            dx,dg=boundary(x.repeat_interleave(2,0),g.repeat_interleave(2,0),
                          upstream,weight,ctx.eps,'symmetric')
            with torch.enable_grad():
                dw,=torch.autograd.grad(out,weight,upstream)
            return dx,dg,dw,None

    class FiniteNorm(FusedRMSNormGated):
        def forward(self,x,g):
            assert self.activation=='silu'
            return FiniteInputs.apply(x,g,self.weight,self.eps)

    result=dict(scope=__doc__,source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
                source_url='https://raw.githubusercontent.com/fla-org/flash-linear-attention/v0.4.1/tests/modules/test_layernorm_gated.py',cases=[])
    for owner in ('native','finite_compiled'):
        test.FusedRMSNormGated=FusedRMSNormGated if owner=='native' else FiniteNorm
        for b,h,t,d in ((2,2,1,64),(2,2,2048,1200),(4,32,128,128),(4,32,447,128)):
            row=dict(owner=owner,batch=b,heads=h,length=t,dimension=d,
                     original_parameter_case=(d!=128))
            try:
                test.test_rmsnorm_gated(B=b,H=h,T=t,D=d,activation='silu')
                row['status']='passed'
            except Exception as exc:row.update(status='failed',error=str(exc))
            result['cases'].append(row)
            args.output.write_text(json.dumps(result,indent=2)+'\n')
            print(row,flush=True)
    result['status']='passed' if all(row['status']=='passed' for row in result['cases']) else 'failed'
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    if result['status']=='failed':raise SystemExit(1)


if __name__=='__main__':main()

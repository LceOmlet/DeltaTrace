"""Executable algebraic counterexample, not an attribution quality benchmark.

Calls the exact frozen normalization and matrix-product pullbacks on CPU.
Epsilon zero tests the stated constant-function identity; the native epsilon
case is separately labelled because its function is almost, not exactly, one.
"""
import argparse,hashlib,json,sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--clean',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();sys.path.insert(0,str(args.clean))
    import torch
    import signed_secant_rules as rules
    assert Path(rules.__file__).resolve()==(args.clean/'signed_secant_rules.py').resolve()
    x0=torch.tensor([[3.,4.]],dtype=torch.float64);x1=torch.tensor([[0.,10.]],dtype=torch.float64)
    out={'scope':'algebraic unit audit; not model benchmark or proof of NI root cause',
        'source_sha256':hashlib.sha256(Path(rules.__file__).read_bytes()).hexdigest(),'cases':[]}
    for epsilon in (0.,1e-6):
        normalize=lambda x:x/(x.square().sum(-1,keepdim=True)+epsilon).sqrt()
        n0,n1=normalize(x0),normalize(x1)
        ma,mb=rules.matmul_secant_pullback(n0,n1,n0.T,n1.T,torch.ones((1,1),dtype=torch.float64))
        combined=ma+mb.T
        coefficients=rules.rmsnorm_secant_pullback(x0,x1,2**-.5,combined,epsilon/2)
        allocation=coefficients*(x1-x0)
        fx=lambda x:float(normalize(x).square().sum())
        hybrids=[torch.tensor([[a,b]],dtype=torch.float64) for a in (3.,0.) for b in (4.,10.)]
        native_grad=[]
        for point in (x0,x1):
            x=point.clone().requires_grad_(True);y=normalize(x).square().sum()
            native_grad.append(torch.autograd.grad(y,x)[0].tolist())
        row={'epsilon':epsilon,'x0':x0.tolist(),'x1':x1.tolist(),'actual_outputs':[fx(x) for x in hybrids],
            'combined_repeated_variable_upstream':combined.tolist(),'coefficients':coefficients.tolist(),
            'per_source_allocation':allocation.tolist(),'sum':float(allocation.sum()),'actual_delta':fx(x1)-fx(x0),
            'native_autograd_endpoint_gradients':native_grad,
            'actual_single_coordinate_deletion_effects':[fx(x1)-fx(torch.tensor([[3.,10.]],dtype=torch.float64)),fx(x1)-fx(torch.tensor([[0.,4.]],dtype=torch.float64))]}
        assert abs(row['sum']-row['actual_delta'])<1e-12
        if epsilon==0:
            assert max(row['actual_outputs'])-min(row['actual_outputs'])<1e-14
            assert torch.allclose(allocation,torch.tensor([[-.108,.108]],dtype=torch.float64),rtol=0,atol=1e-14)
        out['cases'].append(row)
    args.output.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))

if __name__=='__main__':main()

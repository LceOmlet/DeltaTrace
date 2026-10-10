"""CPU interface test against installed PyTorch; not a training tolerance test."""
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.utils.clip_grad as owner

from capture_native_preclip_20261010 import observe_native_norm


def main():
    original = owner._get_total_norm
    results = []
    for name, value in [('finite',2.0),('nan',float('nan')),('inf',float('inf'))]:
        plain = [torch.nn.Parameter(torch.zeros(3)),torch.nn.Parameter(torch.zeros(2))]
        watched = [torch.nn.Parameter(torch.zeros(3)),torch.nn.Parameter(torch.zeros(2))]
        for pair in zip(plain,watched):
            for p in pair:
                p.grad = torch.ones_like(p)
        plain[0].grad[0] = watched[0].grad[0] = value
        expected = owner.clip_grad_norm_(plain,1.0)
        seen, errors, identities = [], [], []
        def native(*args,**kwargs):
            result = original(*args,**kwargs)
            identities.append(result)
            return result
        def callback(norm,tensors):
            seen.append(dict(norm=norm,gradients=[t.clone() for t in tensors]))
        try:
            owner._get_total_norm = observe_native_norm(native,lambda:True,callback,errors.append)
            actual = owner.clip_grad_norm_(watched,1.0)
        finally:
            owner._get_total_norm = original
        assert seen[0]['norm'] is identities[0]
        assert not errors and len(seen)==1
        torch.testing.assert_close(actual,expected,rtol=0,atol=0,equal_nan=True)
        for a,b in zip(plain,watched):
            torch.testing.assert_close(a.grad,b.grad,rtol=0,atol=0,equal_nan=True)
        # The observer sees the native gradients before clipping mutates them.
        assert torch.equal(seen[0]['gradients'][1],torch.ones(2))
        results.append(dict(case=name,exact_return_and_native_clipped_gradients=True))
    errors=[]
    def fail(*_):raise RuntimeError('intentional diagnostic callback failure')
    norm = observe_native_norm(original,lambda:True,fail,errors.append)([torch.ones(2)])
    assert torch.isfinite(norm) and len(errors)==1
    assert not torch.cuda.is_initialized()
    print(json.dumps(dict(status='passed',cases=results,callback_error_preserves_return=True,
        torch_version=torch.__version__,owner_path=owner.__file__,
        owner_sha256=hashlib.sha256(Path(owner.__file__).read_bytes()).hexdigest(),
        CUDA_initialized=False,model_calls=0,training_calls=0,
        scope=__doc__)))


if __name__=='__main__':main()

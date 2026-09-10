"""Compile the unchanged finite-FA buffer preparation using installed Inductor.

No finite/native FA operation is changed. The audit mode compares all ten input
buffers to the literal eager preparation before invoking the same finite kernel.
"""
import torch


def eager_inputs(q0,k0,q1,k1,v0,u,lse0,lse1):
    values=[x.detach().to(dtype=torch.float16).contiguous() for x in (q0,k0,q1,k1,v0,u)]
    for first,last in ((q0,q1),(k0,k1)):
        values.append(((first.float()+last.float())*.5).half().contiguous())
    values.extend(x.detach().contiguous() for x in (lse0,lse1))
    return values


compiled_inputs=torch.compile(eager_inputs,fullgraph=True,dynamic=True,
    options={'triton.cudagraphs':False,'max_autotune':False})

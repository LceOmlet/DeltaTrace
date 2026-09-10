"""Typed public tensor contract for the separately named vendor-FA finite kernel.

No private installed-FA ABI, model patch, native-backward substitution, or fallback.
The caller supplies actual endpoint operands; this remains finite attribution.
"""
import ctypes,hashlib
from pathlib import Path
import torch

from compiled_fa_inputs import compiled_inputs,eager_inputs

class CompiledInputsFA:
    def __init__(self,library,expected_sha256):
        path=Path(library)
        assert hashlib.sha256(path.read_bytes()).hexdigest()==expected_sha256
        self.library=ctypes.CDLL(str(path));self.operation=self.library.deltatrace_fa_finite_p1
        self.operation.argtypes=[ctypes.c_void_p]*15+[ctypes.c_int]*3+[ctypes.c_float,ctypes.c_void_p]
        self.operation.restype=ctypes.c_int

    def __call__(self,operands,scale,activity=None):
        # All expanded operands are [B,H,N,128]; H corresponds to query heads.
        names=['q0','k0','q1','k1','v0','u']
        reference=operands['q0'];batch,heads,length,dim=reference.shape
        assert dim==128 and reference.is_cuda and reference.dtype==torch.float16
        for name in names:
            value=operands[name]
            assert value.shape==reference.shape and value.device==reference.device
            assert value.dtype==(torch.float32 if name=='u' else torch.float16)
        for name in ['lse0','lse1']:
            value=operands[name]
            assert value.shape==(batch,heads,length) and value.dtype==torch.float32 and value.device==reference.device
        inputs=[operands[name] for name in names+['lse0','lse1']]
        values=compiled_inputs(*inputs)
        if getattr(self,'audit_inputs',False):
            eager=eager_inputs(*inputs)
            assert len(eager)==len(values)==10
            exact=[bool(torch.equal(a,b)) for a,b in zip(eager,values)]
            assert all(exact),exact
            if activity is not None:activity['compiled_input_buffers_exact']=exact
            del eager
        tau=torch.empty((batch,heads,length),device=reference.device,dtype=torch.float32)
        center=torch.empty_like(tau)
        dq,dk,dv=(torch.empty_like(values[0]) for _ in range(3))
        buffers=values+[tau,center,dq,dk,dv]
        if activity is not None:
            activity['calls_attempted']=activity.get('calls_attempted',0)+1
            activity['buffer_contract']=[{'shape':list(v.shape),'dtype':str(v.dtype),'bytes':v.numel()*v.element_size()} for v in buffers]
        status=self.operation(*[ctypes.c_void_p(v.data_ptr()) for v in buffers],batch,heads,length,scale,
            ctypes.c_void_p(torch.cuda.current_stream(reference.device).cuda_stream))
        assert status==0,('Finite FA launch error',status)
        if activity is not None:activity['calls_enqueued']=activity.get('calls_enqueued',0)+1
        return {'dq':dq,'dk':dk,'dv':dv,'tau':tau,'center':center}

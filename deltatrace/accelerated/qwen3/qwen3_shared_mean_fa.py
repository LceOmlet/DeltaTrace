"""Typed public tensor contract for the separately named vendor-FA finite kernel.

No private installed-FA ABI, model patch, native-backward substitution, or fallback.
The caller supplies actual endpoint operands; this remains finite attribution.
"""
import ctypes,hashlib
from pathlib import Path
import torch

class VendorFAFiniteP1SharedMeanReuse:
    def __init__(self,library,expected_sha256):
        path=Path(library)
        assert hashlib.sha256(path.read_bytes()).hexdigest()==expected_sha256
        self.library=ctypes.CDLL(str(path));self.operation=self.library.deltatrace_fa_finite_p1_shared_mean_reuse
        self.operation.argtypes=[ctypes.c_void_p]*13+[ctypes.c_int]*4+[ctypes.c_float,ctypes.c_void_p]
        self.operation.restype=ctypes.c_int

    def __call__(self,operands,scale,activity=None):
        # Q/U are [B,Hq,N,128]; K/V are compact [B,Hkv,N,128].
        # Query-head outputs keep the existing FP32 GQA sum outside this operator.
        names=['q0','k0','q1','k1','v0','u']
        reference=operands['q0'];batch,heads,length,dim=reference.shape
        assert dim==128 and reference.is_cuda and reference.dtype==torch.float16
        kv_heads=operands['k0'].shape[1]
        assert kv_heads>0 and heads%kv_heads==0
        for name in names:
            value=operands[name]
            expected=(batch,kv_heads,length,dim) if name in ('k0','k1','v0') else reference.shape
            assert value.shape==expected and value.device==reference.device
            assert value.dtype==(torch.float32 if name=='u' else torch.float16)
        values=[operands[name].detach().to(dtype=torch.float16).contiguous() for name in names]
        for name in ['lse0','lse1']:
            value=operands[name]
            assert value.shape==(batch,heads,length) and value.dtype==torch.float32 and value.device==reference.device
            values.append(value.detach().contiguous())
        tau=torch.empty((batch,heads,length),device=reference.device,dtype=torch.float32)
        center=torch.empty_like(tau)
        dq,dk,dv=(torch.empty_like(values[0]) for _ in range(3))
        buffers=values+[tau,center,dq,dk,dv]
        if activity is not None:
            activity['endpoint_mean']='FP32_add_FP16_store_in_FA_shared_tile_from_existing_loads'
            activity['global_endpoint_mean_buffers']=0
            activity['extra_shared_tile']=False
            activity['owner_kernel_geometry']={'phase0':[32,32,2],'phase1':[64,32,4],'phase2':[64,32,4]}
            activity['shared_memory_bytes']={'phase0':8192,'phase1':16384,'phase2':16384}
            activity['all_phase_owner_input_tiles_loaded_once']=True
            activity['owner_input_reuse_in_native_MMA_register_layout']=True
            activity['dead_shared_A_staging_reused_for_B']=True
            activity['query_heads']=heads
            activity['kv_heads']=kv_heads
            activity['GQA_input_expansion']=False
            activity['calls_attempted']=activity.get('calls_attempted',0)+1
            activity['buffer_contract']=[{'shape':list(v.shape),'dtype':str(v.dtype),'bytes':v.numel()*v.element_size()} for v in buffers]
        status=self.operation(*[ctypes.c_void_p(v.data_ptr()) for v in buffers],batch,heads,kv_heads,length,scale,
            ctypes.c_void_p(torch.cuda.current_stream(reference.device).cuda_stream))
        assert status==0,('Finite FA launch error',status)
        if activity is not None:activity['calls_enqueued']=activity.get('calls_enqueued',0)+1
        return {'dq':dq,'dk':dk,'dv':dv,'tau':tau,'center':center}

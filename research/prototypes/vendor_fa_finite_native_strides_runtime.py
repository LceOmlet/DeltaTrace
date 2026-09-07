"""Typed public tensor contract for the separately named vendor-FA finite kernel.

No private installed-FA ABI, model patch, native-backward substitution, or fallback.
The caller supplies actual endpoint operands; this remains finite attribution.
"""
import ctypes,hashlib
from pathlib import Path
import torch

class VendorFAFiniteP1NativeStrides:
    def __init__(self,library,expected_sha256):
        path=Path(library)
        assert hashlib.sha256(path.read_bytes()).hexdigest()==expected_sha256
        self.library=ctypes.CDLL(str(path));self.operation=self.library.deltatrace_fa_finite_p1_native_strides
        self.operation.argtypes=[ctypes.c_void_p]*13+[ctypes.c_int]*4+[ctypes.c_float,ctypes.POINTER(ctypes.c_int64),ctypes.c_void_p]
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
        # Reuse actual endpoint storage. Only U needs the existing FP32 -> FP16 cast.
        values=[operands[name].detach() if name!='u' else operands[name].detach().to(dtype=torch.float16) for name in names]
        for value in values:
            assert value.stride(-1)==1 and value.data_ptr()%16==0
            assert all(s>0 and s%8==0 for s in value.stride()[:3])
        host_strides=(ctypes.c_int64*18)(*(s for value in values for s in value.stride()[:3]))
        for name in ['lse0','lse1']:
            value=operands[name]
            assert value.shape==(batch,heads,length) and value.dtype==torch.float32 and value.device==reference.device
            values.append(value.detach().contiguous())
        tau=torch.empty((batch,heads,length),device=reference.device,dtype=torch.float32)
        center=torch.empty_like(tau)
        dq,dk,dv=(torch.empty(reference.shape,device=reference.device,dtype=torch.float16) for _ in range(3))
        buffers=values+[tau,center,dq,dk,dv]
        if activity is not None:
            activity['input_layout']='native_FA_batch_head_row_strides;last_dimension_contiguous'
            activity['input_storage_reused']={name:values[i].data_ptr()==operands[name].data_ptr() for i,name in enumerate(names)}
            activity['input_strides']={name:list(operands[name].stride()) for name in names}
            activity['kernel_input_strides']={name:list(values[i].stride()) for i,name in enumerate(names)}
            activity['QKV_copy_bytes']=0
            activity['U_cast_bytes']=values[5].numel()*values[5].element_size()
            activity['endpoint_mean']='FP32_add_FP16_store_in_FA_shared_tile_from_existing_loads'
            activity['global_endpoint_mean_buffers']=0
            activity['extra_shared_tile']=False
            activity['query_heads']=heads
            activity['kv_heads']=kv_heads
            activity['GQA_input_expansion']=False
            activity['calls_attempted']=activity.get('calls_attempted',0)+1
            activity['buffer_contract']=[{'shape':list(v.shape),'dtype':str(v.dtype),'bytes':v.numel()*v.element_size(),'strides':list(v.stride())} for v in buffers]
        status=self.operation(*[ctypes.c_void_p(v.data_ptr()) for v in buffers],batch,heads,kv_heads,length,scale,host_strides,
            ctypes.c_void_p(torch.cuda.current_stream(reference.device).cuda_stream))
        assert status==0,('Finite FA launch error',status)
        if activity is not None:activity['calls_enqueued']=activity.get('calls_enqueued',0)+1
        return {'dq':dq,'dk':dk,'dv':dv,'tau':tau,'center':center}

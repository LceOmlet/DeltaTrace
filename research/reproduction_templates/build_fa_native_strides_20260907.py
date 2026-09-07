"""Address-only extension of the verified FA shared-storage midpoint kernel."""
import ast, hashlib, json, subprocess, sys
from pathlib import Path
A=Path(__file__).resolve().parent
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
def once(s,a,b):
    assert s.count(a)==1,(a,s.count(a))
    return s.replace(a,b)

s=(A/'vendor_fa_finite_p1_shared_mean_reuse.cu').read_text()
s=once(s,' * Endpoint means reuse existing RHS tile loads; no global Q/K midpoint buffers.',
    ' * Endpoint means reuse existing RHS tile loads; no global Q/K midpoint buffers.\n'
    ' * Actual input strides reuse native FA batch/head offsets and make_stride(row,_1{}).')
s=once(s,'    float scale;\n','    float scale;\n    int64_t strides[18]; // Six inputs, each [batch,head,row], measured in elements.\n')
s=once(s,'''    const int kv_bh=(bh/p.heads)*p.kv_heads+(bh%p.heads)/(p.heads/p.kv_heads);
    const int64_t kv_offset=int64_t(kv_bh)*length*D;
    const int64_t pair_left_offset=Phase==2?kv_offset:head_offset;
    const int64_t pair_right_offset=Phase==2?head_offset:kv_offset;''','''    auto input_offset=[&](int operand) {
        const bool kv=operand==1||operand==3||operand==4;
        const int head=kv?(bh%p.heads)/(p.heads/p.kv_heads):bh%p.heads;
        return int64_t(bh/p.heads)*p.strides[operand*3]+int64_t(head)*p.strides[operand*3+1];
    };''')
s=once(s,'auto pair=[&](const E *left,const E *right,auto &acc,int endpoint)',
    'auto pair=[&](const E *left,const E *right,int left_id,int right_id,auto &acc,int endpoint)')
s=once(s,'''left+pair_left_offset+int64_t(row0)*D),
                               Shape<Int<M>,Int<D>>{},Stride<Int<D>,_1>{}''',
    '''left+input_offset(left_id)+int64_t(row0)*p.strides[left_id*3+2]),
                               Shape<Int<M>,Int<D>>{},make_stride(p.strides[left_id*3+2],_1{})''')
s=once(s,'''right+pair_right_offset+int64_t(col0)*D),
                               Shape<Int<N>,Int<D>>{},Stride<Int<D>,_1>{}''',
    '''right+input_offset(right_id)+int64_t(col0)*p.strides[right_id*3+2]),
                               Shape<Int<N>,Int<D>>{},make_stride(p.strides[right_id*3+2],_1{})''')
s=once(s,'pair(p.k0,p.q0,a0,0);pair(p.k1,p.q1,a1,1);pair(p.v0,p.u,at,-1);',
    'pair(p.k0,p.q0,1,0,a0,0);pair(p.k1,p.q1,3,2,a1,1);pair(p.v0,p.u,4,5,at,-1);')
s=once(s,'pair(p.q0,p.k0,a0,0);pair(p.q1,p.k1,a1,1);pair(p.u,p.v0,at,-1);',
    'pair(p.q0,p.k0,0,1,a0,0);pair(p.q1,p.k1,2,3,a1,1);pair(p.u,p.v0,5,4,at,-1);')
s=once(s,'''values+(Phase==1?kv_offset:head_offset)+int64_t(col0)*D),
                                   Shape<Int<N>,Int<D>>{},Stride<Int<D>,_1>{}''',
    '''values+input_offset(5)+int64_t(col0)*p.strides[17]),
                                   Shape<Int<N>,Int<D>>{},make_stride(p.strides[17],_1{})''')
s=once(s,'float scale,void *stream_ptr) {','float scale,const int64_t *input_strides,void *stream_ptr) {')
s=once(s,'    dim3 grid(','''    // Copy host metadata by value before launch; no host pointer is used by a kernel.
    if(!input_strides)return -2;
    for(int i=0;i<18;++i) {
        if(input_strides[i]<=0||input_strides[i]%8!=0)return -3;
        p.strides[i]=input_strides[i];
    }
    dim3 grid(''')
s=s.replace('deltatrace_fa_finite_p1_shared_mean_reuse','deltatrace_fa_finite_p1_native_strides')
s=s.replace('deltatrace_fa_finite_p1_kernel','deltatrace_fa_finite_p1_kernel_strided')
(A/'vendor_fa_finite_p1_native_strides.cu').write_text(s)

r=(A/'vendor_fa_finite_shared_mean_reuse_runtime.py').read_text()
r=r.replace('VendorFAFiniteP1SharedMeanReuse','VendorFAFiniteP1NativeStrides')
r=r.replace('deltatrace_fa_finite_p1_shared_mean_reuse','deltatrace_fa_finite_p1_native_strides')
r=once(r,'[ctypes.c_float,ctypes.c_void_p]',
    '[ctypes.c_float,ctypes.POINTER(ctypes.c_int64),ctypes.c_void_p]')
r=once(r,"        values=[operands[name].detach().to(dtype=torch.float16).contiguous() for name in names]",'''        # Reuse actual endpoint storage. Only U needs the existing FP32 -> FP16 cast.
        values=[operands[name].detach() if name!='u' else operands[name].detach().to(dtype=torch.float16) for name in names]
        for value in values:
            assert value.stride(-1)==1 and value.data_ptr()%16==0
            assert all(s>0 and s%8==0 for s in value.stride()[:3])
        host_strides=(ctypes.c_int64*18)(*(s for value in values for s in value.stride()[:3]))''')
r=once(r,'dq,dk,dv=(torch.empty_like(values[0]) for _ in range(3))',
    'dq,dk,dv=(torch.empty(reference.shape,device=reference.device,dtype=torch.float16) for _ in range(3))')
r=once(r,"            activity['endpoint_mean']=",'''            activity['input_layout']='native_FA_batch_head_row_strides;last_dimension_contiguous'
            activity['input_storage_reused']={name:values[i].data_ptr()==operands[name].data_ptr() for i,name in enumerate(names)}
            activity['input_strides']={name:list(operands[name].stride()) for name in names}
            activity['kernel_input_strides']={name:list(values[i].stride()) for i,name in enumerate(names)}
            activity['QKV_copy_bytes']=0
            activity['U_cast_bytes']=values[5].numel()*values[5].element_size()
            activity['endpoint_mean']=''')
r=once(r,"'bytes':v.numel()*v.element_size()", "'bytes':v.numel()*v.element_size(),'strides':list(v.stride())")
r=once(r,'batch,heads,kv_heads,length,scale,','batch,heads,kv_heads,length,scale,host_strides,')
ast.parse(r);(A/'vendor_fa_finite_native_strides_runtime.py').write_text(r)

b=(A/'fa_shared_mean_reuse_build_20260907.py').read_text()
b=b.replace('shared_mean_reuse','native_strides')
(A/'fa_native_strides_build_20260907.py').write_text(b)
p=json.loads((A/'fa_shared_mean_reuse_operator_protocol_20260907.json').read_text())
p.update(purpose='Compile address-only finite FA change using the pinned original FA stride framework. No arithmetic, precision, pass/GEMM count, shared-memory, model or installed FA changes. Zero GPU/model runs in this build.',
    study_sha256=sha(A/'fa_native_strides_build_20260907.py'),
    extension_sha256=sha(A/'vendor_fa_finite_p1_native_strides.cu'),
    sources={},budget={'extension_compiles':1,'native_model_forwards':0,'attributions':0,'quality_queries':0,'local_finite_operator_calls':0},
    next_if_compiled='Bounded captured-real-tensor layout validation, then original B1/B4 whole-vector checks. No quality/FT sweep.')
(A/'fa_native_strides_build_protocol_20260907.json').write_text(json.dumps(p,indent=2))
subprocess.run([sys.executable,str(A/'prepare_remote_experiment.py'),'${ARTIFACT_ROOT}/codex_fa_native_strides_build_20260907_v1',
    'study.py='+str(A/'fa_native_strides_build_20260907.py'),
    'protocol.json='+str(A/'fa_native_strides_build_protocol_20260907.json'),
    'vendor_fa_finite_p1_native_strides.cu='+str(A/'vendor_fa_finite_p1_native_strides.cu'),
    '--request',str(A/'launch_fa_native_strides_build_20260907.json')],check=True)
print(json.dumps({'stride_metadata_int64s':18,'shared_memory_bytes_per_phase':16384,'arithmetic_changed':False}))

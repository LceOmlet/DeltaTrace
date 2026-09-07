"""Reuse native FA backward transpose/accumulation to share finite score tiles."""
import ast,hashlib,json,subprocess,sys
from pathlib import Path
A=Path(__file__).resolve().parent;sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
def once(s,a,b):assert s.count(a)==1,(a,s.count(a));return s.replace(a,b)
s=(A/'vendor_fa_finite_p1_shared_mean_reuse.cu').read_text()
s=once(s,'#include "softmax.h"','#include "softmax.h"\n#include "flash_bwd_preprocess_kernel.h"\n#include "flash_bwd_preprocess_kernel_hdim128_32_32.h"')
s=once(s,'struct FiniteParams {',
    '''// Native FA backward layouts for storing and transposing centered FP16 weights.
using TransposeTraits=Flash_bwd_kernel_traits<128,32,32,2,1,1,1>;
// Match native flash_bwd_launch_template.h's 32x32 dispatch and traits.
// Generic convert_dQ's register load mapping is not valid for this case.
using ConvertTraits=Flash_bwd_kernel_traits<128,32,32,2,2,2,2,true,true,true>;

// Explicit argument adapter to the unmodified templated convert_dQ routine.
// This is NOT the installed FA private ABI and needs no ATen generator header.
struct FiniteDqConvertParams {
    const int *cu_seqlens_q,*cu_seqlens_k,*seqused_k;
    const void *knew_ptr;
    bool is_seqlens_k_cumulative;
    int seqlen_q,seqlen_k,seqlen_knew,seqlen_q_rounded,h,h_k,d,d_rounded;
    void *dq_ptr,*dq_accum_ptr;
    int64_t dq_batch_stride,dq_row_stride,dq_head_stride,dq_accum_split_stride;
    float scale_softmax_rp_dropout;
};

struct FiniteParams {''')
s=once(s,'    int batch,heads,kv_heads,length;', '    float *dq_accum;\n    int batch,heads,kv_heads,length;')
s=once(s,'    const int length=p.length;', '''    static_assert(Phase==0 || Phase==2);
    const int length=p.length;
    const int rounded=((length+31)/32)*32;
    if constexpr(Phase==0) {
        // Same linear dQ accumulation workspace used by native FA backward.
        // Every row, including rounded padding, has exactly one initializer.
        for(int i=tid;i<M*D;i+=Traits::kNThreads) {
            const int row=row0+i/D,d=i%D;
            p.dq_accum[((int64_t(bh/p.heads)*rounded+row)*p.heads+bh%p.heads)*D+d]=0.f;
        }
    }''')
s=once(s,'    auto firstEndpointB=make_fragment_like(toB);',
    '    auto firstEndpointB=make_fragment_like(toB);\n    auto firstEndpointA=make_fragment_like(toA);')
s=once(s,'                if(endpoint==0) cute::copy(toB,firstEndpointB);',
    '                if(endpoint==0) {cute::copy(toB,firstEndpointB);cute::copy(toA,firstEndpointA);}')
s=once(s,'                        firstEndpointB(i)=E((float(firstEndpointB(i))+float(toB(i)))*0.5f);',
    '''                        firstEndpointB(i)=E((float(firstEndpointB(i))+float(toB(i)))*0.5f);
                    #pragma unroll
                    for(int i=0;i<size(firstEndpointA);++i)
                        firstEndpointA(i)=E((float(firstEndpointA(i))+float(toA(i)))*0.5f);''')
s=once(s,'            else {multiply(a0,nullptr,accQ,true);multiply(a1,p.u,accV,false);}', '''            else {
                multiply(a0,nullptr,accQ,true);multiply(a1,p.u,accV,false);
                // Native flash_bwd_kernel.h pattern: store centered FP16 dS,
                // view its transpose in shared memory, then use flash::gemm.
                // Here a0 is finite L*(U V0^T-center), not a model gradient.
                __syncthreads();
                auto sW=make_tensor(sB.data()+size(sB),typename TransposeTraits::SmemLayoutPdS{});
                auto sWt=make_tensor(sW.data(),typename TransposeTraits::SmemLayoutPdStransposed{});
                auto sWtPlain=make_tensor(sW.data(),typename TransposeTraits::SmemLayoutPdStransposedNoSwizzle{});
                auto writeW=make_tiled_copy_C(typename TransposeTraits::SmemCopyAtomPdS{},mma);
                auto writeThread=writeW.get_thread_slice(tid);
                auto destW=writeThread.partition_D(sW);
                CONVERT_TENSOR_TYPE(float,E,a0,half_centered)
                auto weights=make_tensor(half_centered.data(),a0.layout());
                auto sourceW=writeThread.retile_S(weights);
                cute::copy(writeW,sourceW,destW);
                // The original A tile is dead. Reuse it for the K midpoint.
                cute::copy(firstEndpointA,toA);
                auto sAt=make_tensor(sA.data(),typename Traits::SmemLayoutVtransposed{});
                auto sAtPlain=make_tensor(sA.data(),typename Traits::SmemLayoutVtransposedNoSwizzle{});
                auto readW=make_tiled_copy_A(typename Traits::SmemCopyAtomTransposed{},mma);
                auto readK=make_tiled_copy_B(typename Traits::SmemCopyAtomTransposed{},mma);
                auto wt=readW.get_thread_slice(tid);auto kt=readK.get_thread_slice(tid);
                auto wsrc=wt.partition_S(sWt);auto ksrc=kt.partition_S(sAt);
                auto wreg=mma_thread.partition_fragment_A(sWtPlain);
                auto kreg=mma_thread.partition_fragment_B(sAtPlain);
                auto partialQ=partition_fragment_C(mma,Shape<Int<N>,Int<D>>{});
                clear(partialQ);__syncthreads();
                flash::gemm(partialQ,wreg,kreg,wsrc,ksrc,mma,readW,readK,wt,kt);
                // Native FA seq-k-parallel backward uses FP32 atomic dQ
                // accumulation; summation order can vary, as in default FA.
                // Match BOTH sides of the native workspace contract: the
                // accumulation copy layout and its32x32 conversion. This is an
                // encoded register workspace, not a plain logical dQ matrix.
                const int64_t qbase=((int64_t(bh/p.heads)*rounded+col0)*p.heads+bh%p.heads)*D;
                auto gQacc=make_tensor(make_gmem_ptr(p.dq_accum+qbase),
                    Shape<Int<N>,Int<D>>{},make_stride(int64_t(p.heads)*D,_1{}));
                typename ConvertTraits::GmemTiledCopydQaccumAtomicAdd_hdim128_32_32 accumCopy;
                auto accumThread=accumCopy.get_thread_slice(tid);
                auto targetQ=accumThread.partition_D(gQacc);
                CUTE_STATIC_ASSERT_V(size(partialQ)==size(targetQ));
                #pragma unroll
                for(int i=0;i<size(partialQ);++i) atomicAdd(&targetQ(i),partialQ(i));
            }''')
s=once(s,'extern "C" int deltatrace_fa_finite_p1_shared_mean_reuse(',
    '''__global__ void deltatrace_fa_finite_convert_dq(FiniteDqConvertParams p) {
    // Unmodified vendor FA postprocessing, including FP32 scaling and FP16 store.
    flash::convert_dQ_hdim128_32_32<ConvertTraits>(p,1);
}

extern "C" int deltatrace_fa_finite_p1_two_sweeps(''')
s=once(s,'void *dv,int batch', 'void *dv,void *dq_accum,int batch')
s=once(s,'(E*)dq,(E*)dk,(E*)dv,batch,', '(E*)dq,(E*)dk,(E*)dv,(float*)dq_accum,batch,')
s=once(s,'''    deltatrace_fa_finite_p1_kernel<1><<<grid,FiniteTraits::kNThreads,shared,stream>>>(p);
    error=cudaGetLastError();if(error!=cudaSuccess)return int(error);
    deltatrace_fa_finite_p1_kernel<2><<<grid,FiniteTraits::kNThreads,shared,stream>>>(p);
    return int(cudaGetLastError());''', '''    constexpr int weights_shared=size(typename TransposeTraits::SmemLayoutPdS{})*sizeof(E);
    deltatrace_fa_finite_p1_kernel<2><<<grid,FiniteTraits::kNThreads,shared+weights_shared,stream>>>(p);
    error=cudaGetLastError();if(error!=cudaSuccess)return int(error);
    FiniteDqConvertParams convert{};
    convert.h=heads;convert.h_k=kv_heads;convert.seqlen_q=length;convert.seqlen_k=length;
    convert.seqlen_q_rounded=((length+31)/32)*32;convert.d=128;convert.d_rounded=128;
    convert.dq_ptr=dq;convert.dq_accum_ptr=dq_accum;
    convert.dq_batch_stride=int64_t(heads)*length*128;
    convert.dq_head_stride=int64_t(length)*128;convert.dq_row_stride=128;
    convert.scale_softmax_rp_dropout=scale;
    dim3 convert_grid((length+31)/32,heads,batch);
    constexpr int convert_shared=size(typename ConvertTraits::SmemLayoutdQ{})*sizeof(E);
    deltatrace_fa_finite_convert_dq<<<convert_grid,ConvertTraits::kNThreads,convert_shared,stream>>>(convert);
    return int(cudaGetLastError());''')
s=s.replace('deltatrace_fa_finite_p1_kernel','deltatrace_fa_finite_p1_kernel_two_sweeps')
s=s.replace(' * No integral quadrature, no global N-by-N buffer, no model/package patch.',
    ' * Two quadratic sweeps plus unmodified native FA dQ conversion. FP32 atomic\n'
    ' * dQ summation follows native FA backward; default non-deterministic order.\n'
    ' * No integral quadrature, no global N-by-N buffer, no model/package patch.')
(A/'vendor_fa_finite_p1_two_sweeps.cu').write_text(s)
r=(A/'vendor_fa_finite_shared_mean_reuse_runtime.py').read_text()
r=r.replace('VendorFAFiniteP1SharedMeanReuse','VendorFAFiniteP1TwoSweeps')
r=r.replace('deltatrace_fa_finite_p1_shared_mean_reuse','deltatrace_fa_finite_p1_two_sweeps')
r=once(r,'[ctypes.c_void_p]*13','[ctypes.c_void_p]*14')
r=once(r,'        buffers=values+[tau,center,dq,dk,dv]',
    '''        dq_accum=torch.empty((batch,((length+31)//32)*32,heads,dim),device=reference.device,dtype=torch.float32)
        buffers=values+[tau,center,dq,dk,dv,dq_accum]''')
r=once(r,"            activity['extra_shared_tile']=False", '''            activity['extra_shared_tile']='One native-FA-layout32x32FP16 coefficient tile,2048 bytes'
            activity['quadratic_sweeps']=2
            activity['output_conversion']='unmodified_vendor_flash_convert_dQ_hdim128_32_32'
            activity['dq_accumulation']='FP32_atomic_add_as_native_FA_seqk_parallel_backward'
            activity['dq_accumulation_workspace_bytes']=dq_accum.numel()*dq_accum.element_size()
            activity['runtime_kernel_launches']=3''')
ast.parse(r);(A/'vendor_fa_finite_two_sweeps_runtime.py').write_text(r)
b=(A/'fa_shared_mean_reuse_build_20260907.py').read_text().replace('shared_mean_reuse','two_sweeps')
(A/'fa_two_sweeps_build_20260907.py').write_text(b)
p={k:v for k,v in json.loads((A/'fa_shared_mean_operator_protocol_20260907.json').read_text()).items()
   if k in ['vendor_source_parent','vendor_parent_sha256','vendor_git_commit','vendor_source_version','installed_model_FA_version','same_version_source_claim','max_compile_seconds']}
p.update(purpose='Compile two-sweep finite P1 candidate. Reuse native FA backward transpose layouts, FP32 atomic accumulation pattern and unmodified convert_dQ. Keep actual model/default FA and finite centered coefficient unchanged; summation order explicitly changes. No model or GPU execution in build.',
    study_sha256=sha(A/'fa_two_sweeps_build_20260907.py'),extension_sha256=sha(A/'vendor_fa_finite_p1_two_sweeps.cu'),
    budget={'extension_compiles':1,'local_operator_calls':0,'whole_model_attributions':0,'quality_queries':0},
    local_review_if_compiled={'scope':'Old captured real NI0 layer35, not a new benchmark. Six operator calls,1warm+2alternating measured.',
        'unchanged_outputs':'tau,center,dk,dv should remain exact.',
        'dq_gate':'Finite DQ must have relativeL2 error<=0.001 and max_abs_error<=0.01*old_RMS. This is an engineering screen, not a bound on attribution/sign/quality. Retain all per-output errors and sign flips; do not relax gate after results.',
        'performance':'All preparation, FP32 accumulation workspace and native conversion included. No automatic full sweep.'})
(A/'fa_two_sweeps_build_protocol_20260907.json').write_text(json.dumps(p,indent=2))
subprocess.run([sys.executable,str(A/'prepare_remote_experiment.py'),'${ARTIFACT_ROOT}/codex_fa_two_sweeps_build_20260907_v4',
    'study.py='+str(A/'fa_two_sweeps_build_20260907.py'),'protocol.json='+str(A/'fa_two_sweeps_build_protocol_20260907.json'),
    'vendor_fa_finite_p1_two_sweeps.cu='+str(A/'vendor_fa_finite_p1_two_sweeps.cu'),
    '--request',str(A/'launch_fa_two_sweeps_build_20260907.json')],check=True)
print(json.dumps({'quadratic_sweeps':2,'kernel_launches':3,'new_global_workspace':'FP32 B*H*ceil32(N)*128; linear in N','native_dQ_conversion_reused':True}))

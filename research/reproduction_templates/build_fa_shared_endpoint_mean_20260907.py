"""Fuse endpoint means into already-loaded FA shared-memory tiles.

Reuses pinned FA copying, layouts and GEMM. Does not change finite centering,
quantization of attention multipliers, pass order, or any model computation.
"""
import ast,hashlib,json,subprocess,sys
from pathlib import Path
A=Path(__file__).resolve().parent;sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
def once(s,a,b):
    assert s.count(a)==1,(a,s.count(a));return s.replace(a,b)
s=(A/'vendor_fa_finite_p1_gqa.cu').read_text()
s=once(s,'*q0,*k0,*q1,*k1,*v0,*u,*qmean,*kmean;', '*q0,*k0,*q1,*k1,*v0,*u;')
s=once(s,'auto sBtPlain=make_tensor(sB.data(),typename Traits::SmemLayoutVtransposedNoSwizzle{});', '''auto sBtPlain=make_tensor(sB.data(),typename Traits::SmemLayoutVtransposedNoSwizzle{});
    // The RHS tiles of the first two QK products already contain both endpoints.
    // Preserve their FP32 midpoint as FP16 in a third native FA-layout tile.
    auto sMeanB=make_tensor(sB.data()+size(sB),typename Traits::SmemLayoutKV{});
    auto sMeanBt=make_tensor(sMeanB.data(),typename Traits::SmemLayoutVtransposed{});''')
s=once(s,'auto toB=global_thread.partition_D(sB);','''auto toB=global_thread.partition_D(sB);
    auto toMeanB=global_thread.partition_D(sMeanB);
    auto firstEndpointB=make_fragment_like(toB);''')
s=once(s,'auto fromBt=threadBt.partition_S(sBt);','''auto fromBt=threadBt.partition_S(sBt);
    auto fromMeanBt=threadBt.partition_S(sMeanBt);''')
s=once(s,'auto pair=[&](const E *left,const E *right,auto &acc) {', 'auto pair=[&](const E *left,const E *right,auto &acc,int endpoint) {')
s=once(s,'''            __syncthreads();
            clear(acc);''','''            __syncthreads();
            if constexpr(Phase!=0) {
                if(endpoint==0) cute::copy(toB,firstEndpointB);
                if(endpoint==1) {
                    #pragma unroll
                    for(int i=0;i<size(firstEndpointB);++i)
                        firstEndpointB(i)=E((float(firstEndpointB(i))+float(toB(i)))*0.5f);
                    cute::copy(firstEndpointB,toMeanB);
                }
            }
            clear(acc);''')
s=once(s,'pair(p.k0,p.q0,a0);pair(p.k1,p.q1,a1);pair(p.v0,p.u,at);',
    'pair(p.k0,p.q0,a0,0);pair(p.k1,p.q1,a1,1);pair(p.v0,p.u,at,-1);')
s=once(s,'pair(p.q0,p.k0,a0);pair(p.q1,p.k1,a1);pair(p.u,p.v0,at);',
    'pair(p.q0,p.k0,a0,0);pair(p.q1,p.k1,a1,1);pair(p.u,p.v0,at,-1);')
s=once(s,'auto multiply=[&](auto &weights,const E *values,auto &acc) {','auto multiply=[&](auto &weights,const E *values,auto &acc,bool shared_mean) {')
start=s.index('                auto gB=make_tensor(make_gmem_ptr(values+')
end=s.index('                __syncthreads();',start)
s=s[:start]+'                if(!shared_mean) {\n'+s[start:end]+'                }\n'+s[end:]
s=once(s,'flash::gemm_rs(acc,regWeights,regBt,fromBt,mma,copyBt,threadBt);',
    'auto sourceB=shared_mean?fromMeanBt:fromBt;\n                flash::gemm_rs(acc,regWeights,regBt,sourceB,mma,copyBt,threadBt);')
s=once(s,'if constexpr(Phase==1) multiply(a0,p.kmean,accQ);\n            else {multiply(a0,p.qmean,accQ);multiply(a1,p.u,accV);}',
    'if constexpr(Phase==1) multiply(a0,nullptr,accQ,true);\n            else {multiply(a0,nullptr,accQ,true);multiply(a1,p.u,accV,false);}')
s=once(s,'extern "C" int deltatrace_fa_finite_p1_gqa(', 'extern "C" int deltatrace_fa_finite_p1_shared_mean(')
s=once(s,'const void *u,const void *qmean,const void *kmean,const void *lse0,const void *lse1,',
    'const void *u,const void *lse0,const void *lse1,')
s=once(s,'(const E*)u,(const E*)qmean,(const E*)kmean,(const float*)lse0,(const float*)lse1,',
    '(const E*)u,(const float*)lse0,(const float*)lse1,')
s=once(s,'    auto stream=reinterpret_cast<cudaStream_t>(stream_ptr);','''    constexpr int shared_mean=shared+size(typename FiniteTraits::SmemLayoutKV{})*sizeof(E);
    auto stream=reinterpret_cast<cudaStream_t>(stream_ptr);''')
s=once(s,'kernel<1><<<grid,FiniteTraits::kNThreads,shared,stream>>>','kernel<1><<<grid,FiniteTraits::kNThreads,shared_mean,stream>>>')
s=once(s,'kernel<2><<<grid,FiniteTraits::kNThreads,shared,stream>>>','kernel<2><<<grid,FiniteTraits::kNThreads,shared_mean,stream>>>')
s=s.replace(' * No integral quadrature,', ' * Endpoint means reuse existing RHS tile loads; no global Q/K midpoint buffers.\n * No integral quadrature,')
(A/'vendor_fa_finite_p1_shared_mean.cu').write_text(s)
runtime=(A/'vendor_fa_finite_gqa_runtime.py').read_text()
runtime=runtime.replace('class VendorFAFiniteP1CompactGQA:', 'class VendorFAFiniteP1SharedMean:')
runtime=runtime.replace('self.library.deltatrace_fa_finite_p1_gqa','self.library.deltatrace_fa_finite_p1_shared_mean')
runtime=runtime.replace('[ctypes.c_void_p]*15','[ctypes.c_void_p]*13')
runtime=once(runtime,"""        for first,last in [('q0','q1'),('k0','k1')]:
            values.append(((operands[first].float()+operands[last].float())*.5).half().contiguous())
""",'')
runtime=runtime.replace("            activity['query_heads']=heads", "            activity['endpoint_mean']='FP32_add_FP16_store_in_FA_shared_tile_from_existing_loads'\n            activity['global_endpoint_mean_buffers']=0\n            activity['query_heads']=heads",1)
ast.parse(runtime)
(A/'vendor_fa_finite_shared_mean_runtime.py').write_text(runtime)
build=(A/'fa_compact_gqa_build_20260907.py').read_text()
build=build.replace('vendor_fa_finite_p1_gqa.cu','vendor_fa_finite_p1_shared_mean.cu').replace('libdeltatrace_fa_finite_gqa.so','libdeltatrace_fa_finite_shared_mean.so').replace('deltatrace_fa_finite_p1_gqa','deltatrace_fa_finite_p1_shared_mean')
(A/'fa_shared_mean_build_20260907.py').write_text(build)
driver=(A/'fa_compact_gqa_operator_20260907.py').read_text()
driver=driver.replace('from vendor_fa_finite_runtime import VendorFAFiniteP1', 'from vendor_fa_finite_shared_mean_runtime import VendorFAFiniteP1SharedMean')
driver=driver.replace("old=VendorFAFiniteP1(p['old_library'],p['old_library_sha256'])", "old=VendorFAFiniteP1CompactGQA(p['old_library'],p['old_library_sha256'])")
driver=driver.replace("new=VendorFAFiniteP1CompactGQA(A/'libdeltatrace_fa_finite_gqa.so'", "new=VendorFAFiniteP1SharedMean(A/'libdeltatrace_fa_finite_shared_mean.so'")
driver=driver.replace("ops if name=='old' else compact", 'compact')
driver=driver.replace("'compact'", "'shared_mean'")
driver=driver.replace('Address mapping must preserve existing operator outputs.', 'Shared-tile midpoint must preserve the existing coefficient and output quantization.')
driver=driver.replace("'operator_exact_no_end_to_end_claim'", "'shared_mean_operator_exact_no_end_to_end_claim'")
ast.parse(driver)
(A/'fa_shared_mean_operator_20260907.py').write_text(driver)
p=json.loads((A/'fa_compact_gqa_operator_protocol_20260907.json').read_text())
prior=json.loads((A/'fa_compact_gqa_operator_summary_20260907.json').read_text())
p.update(purpose='FA shared-tile endpoint mean: reuse the already-loaded RHS endpoint tiles in current traceable FA finite kernel. FP32 add then FP16 mean store exactly as old preparation; attention centering and coefficient casts unchanged. One build, six local operator calls on pinned original NI0 layer35 operands. No new model/quality calls or benchmarks.',
    study_sha256=sha(A/'fa_shared_mean_operator_20260907.py'),
    extension_sha256=sha(A/'vendor_fa_finite_p1_shared_mean.cu'),
    old_library='${ARTIFACT_ROOT}/codex_fa_compact_gqa_operator_20260907_v1/libdeltatrace_fa_finite_gqa.so',
    old_library_sha256=prior['library_sha256'],
    next_if_compiled='Six local same-input old/new finite operator calls only; no whole-network promotion from local timing.',
    math_change=False,pass_count=3,
    shared_memory_bytes={'phase0':16384,'phase1':24576,'phase2':24576})
files={'build.py':'fa_shared_mean_build_20260907.py','vendor_fa_finite_p1_shared_mean.cu':'vendor_fa_finite_p1_shared_mean.cu',
       'vendor_fa_finite_shared_mean_runtime.py':'vendor_fa_finite_shared_mean_runtime.py',
       'vendor_fa_finite_gqa_runtime.py':'vendor_fa_finite_gqa_runtime.py'}
p['sources']={name:sha(A/file) for name,file in files.items()}
(A/'fa_shared_mean_operator_protocol_20260907.json').write_text(json.dumps(p,indent=2))
cmd=[sys.executable,str(A/'prepare_remote_experiment.py'),'${ARTIFACT_ROOT}/codex_fa_shared_mean_operator_20260907_v1',
    'study.py='+str(A/'fa_shared_mean_operator_20260907.py'),
    'protocol.json='+str(A/'fa_shared_mean_operator_protocol_20260907.json')]
cmd += [name+'='+str(A/file) for name,file in files.items()]
cmd += ['--request',str(A/'launch_fa_shared_mean_operator_20260907.json')]
subprocess.run(cmd,check=True)
assert len(json.loads((A/'launch_fa_shared_mean_operator_20260907.json').read_text())['cmd'].encode())<100000
print(json.dumps({'FA_source_generated':True,'global_midpoint_buffers_removed':2,'finite_passes':3,'native_model_calls_budget':0}))

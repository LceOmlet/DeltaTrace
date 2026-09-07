"""Reuse existing FA shared B storage after score GEMMs finish."""
import ast,hashlib,json,subprocess,sys
from pathlib import Path
A=Path(__file__).resolve().parent;sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
def once(s,a,b):assert s.count(a)==1,(a,s.count(a));return s.replace(a,b)
s=(A/'vendor_fa_finite_p1_shared_mean.cu').read_text()
s=once(s,'''    // The RHS tiles of the first two QK products already contain both endpoints.
    // Preserve their FP32 midpoint as FP16 in a third native FA-layout tile.
    auto sMeanB=make_tensor(sB.data()+size(sB),typename Traits::SmemLayoutKV{});
    auto sMeanBt=make_tensor(sMeanB.data(),typename Traits::SmemLayoutVtransposed{});
''','''    // Retain the FP16 midpoint in registers until the score products finish.
    // Their shared B tile is then dead and is reused for the multiplier GEMM.
''')
s=once(s,'    auto toMeanB=global_thread.partition_D(sMeanB);\n','')
s=once(s,'    auto fromMeanBt=threadBt.partition_S(sMeanBt);\n','')
s=once(s,'                    cute::copy(firstEndpointB,toMeanB);\n','')
s=once(s,'                if(!shared_mean) {', '''                if(shared_mean) {
                    cute::copy(firstEndpointB,toB);
                } else {''')
s=once(s,'''                auto sourceB=shared_mean?fromMeanBt:fromBt;
                flash::gemm_rs(acc,regWeights,regBt,sourceB,mma,copyBt,threadBt);''',
    '                flash::gemm_rs(acc,regWeights,regBt,fromBt,mma,copyBt,threadBt);')
s=once(s,'    constexpr int shared_mean=shared+size(typename FiniteTraits::SmemLayoutKV{})*sizeof(E);\n','')
assert s.count('FiniteTraits::kNThreads,shared_mean,stream')==2
s=s.replace('FiniteTraits::kNThreads,shared_mean,stream','FiniteTraits::kNThreads,shared,stream')
s=s.replace('extern "C" int deltatrace_fa_finite_p1_shared_mean(', 'extern "C" int deltatrace_fa_finite_p1_shared_mean_reuse(')
(A/'vendor_fa_finite_p1_shared_mean_reuse.cu').write_text(s)
runtime=(A/'vendor_fa_finite_shared_mean_runtime.py').read_text().replace('class VendorFAFiniteP1SharedMean:', 'class VendorFAFiniteP1SharedMeanReuse:')
runtime=runtime.replace('self.library.deltatrace_fa_finite_p1_shared_mean','self.library.deltatrace_fa_finite_p1_shared_mean_reuse')
runtime=runtime.replace("            activity['global_endpoint_mean_buffers']=0", "            activity['global_endpoint_mean_buffers']=0\n            activity['extra_shared_tile']=False",1)
ast.parse(runtime);(A/'vendor_fa_finite_shared_mean_reuse_runtime.py').write_text(runtime)
build=(A/'fa_shared_mean_build_20260907.py').read_text().replace('vendor_fa_finite_p1_shared_mean.cu','vendor_fa_finite_p1_shared_mean_reuse.cu').replace('libdeltatrace_fa_finite_shared_mean.so','libdeltatrace_fa_finite_shared_mean_reuse.so').replace("assert 'deltatrace_fa_finite_p1_shared_mean'", "assert 'deltatrace_fa_finite_p1_shared_mean_reuse'")
(A/'fa_shared_mean_reuse_build_20260907.py').write_text(build)
driver=(A/'fa_shared_mean_operator_20260907.py').read_text()
driver=driver.replace('from vendor_fa_finite_gqa_runtime import VendorFAFiniteP1CompactGQA',
    'from vendor_fa_finite_gqa_runtime import VendorFAFiniteP1CompactGQA\nfrom vendor_fa_finite_shared_mean_reuse_runtime import VendorFAFiniteP1SharedMeanReuse',1)
driver=driver.replace("new=VendorFAFiniteP1SharedMean(A/'libdeltatrace_fa_finite_shared_mean.so',b['library']['sha256'])", """extra_tile=VendorFAFiniteP1SharedMean(p['extra_tile_library'],p['extra_tile_library_sha256'])
new=VendorFAFiniteP1SharedMeanReuse(A/'libdeltatrace_fa_finite_shared_mean_reuse.so',b['library']['sha256'])
methods={'old':old,'extra_tile':extra_tile,'reuse_shared':new}""",1)
driver=driver.replace("        for name in (['old','shared_mean'] if repeat!=1 else ['shared_mean','old']):", "        names=['old','extra_tile','reuse_shared'];names=names[repeat:]+names[:repeat]\n        for name in names:",1)
driver=driver.replace("len(r['calls'])<6", "len(r['calls'])<9")
driver=driver.replace("(old if name=='old' else new)(compact,record['scale'],activity)", "methods[name](compact,record['scale'],activity)")
driver=driver.replace("a,c=outputs['old'][key],outputs['shared_mean'][key]", "a,c=outputs['old'][key],outputs['reuse_shared'][key]\n            assert torch.equal(a,outputs['extra_tile'][key])")
driver=driver.replace('shared_mean_operator_exact_no_end_to_end_claim','shared_mean_reuse_operator_exact_no_end_to_end_claim')
ast.parse(driver);(A/'fa_shared_mean_reuse_operator_20260907.py').write_text(driver)
p=json.loads((A/'fa_shared_mean_operator_protocol_20260907.json').read_text())
v1=json.loads((A/'fa_shared_mean_operator_summary_20260907.json').read_text())
p.update(purpose='FA shared-storage reuse adjustment after extra-tile version slowed. Three-way same-job local comparison: original compact GQA, already-tested extra shared tile, and register-held midpoint copied to the now-dead original B tile. Same finite expression/precision/three passes. One compile and9 local calls total,0 model/quality calls.',
    study_sha256=sha(A/'fa_shared_mean_reuse_operator_20260907.py'),
    extension_sha256=sha(A/'vendor_fa_finite_p1_shared_mean_reuse.cu'),
    extra_tile_library='${ARTIFACT_ROOT}/codex_fa_shared_mean_operator_20260907_v1/libdeltatrace_fa_finite_shared_mean.so',
    extra_tile_library_sha256=v1['library_sha256'],
    budget={'native_model_forwards':0,'VJPs':0,'attributions':0,'quality_queries':0,'local_finite_operator_calls':9},
    shared_memory_bytes={'phase0':16384,'phase1':16384,'phase2':16384},
    next_if_compiled='Nine local calls only. Assess all outputs and same-job times; no automatic whole-model run or parameter scan.')
files={'build.py':'fa_shared_mean_reuse_build_20260907.py',
    'vendor_fa_finite_p1_shared_mean_reuse.cu':'vendor_fa_finite_p1_shared_mean_reuse.cu',
    'vendor_fa_finite_shared_mean_reuse_runtime.py':'vendor_fa_finite_shared_mean_reuse_runtime.py',
    'vendor_fa_finite_shared_mean_runtime.py':'vendor_fa_finite_shared_mean_runtime.py',
    'vendor_fa_finite_gqa_runtime.py':'vendor_fa_finite_gqa_runtime.py'}
p['sources']={name:sha(A/file) for name,file in files.items()}
(A/'fa_shared_mean_reuse_operator_protocol_20260907.json').write_text(json.dumps(p,indent=2))
cmd=[sys.executable,str(A/'prepare_remote_experiment.py'),'${ARTIFACT_ROOT}/codex_fa_shared_mean_reuse_operator_20260907_v1',
    'study.py='+str(A/'fa_shared_mean_reuse_operator_20260907.py'),
    'protocol.json='+str(A/'fa_shared_mean_reuse_operator_protocol_20260907.json')]
cmd += [name+'='+str(A/file) for name,file in files.items()]
cmd += ['--request',str(A/'launch_fa_shared_mean_reuse_operator_20260907.json')]
subprocess.run(cmd,check=True)
assert len(json.loads((A/'launch_fa_shared_mean_reuse_operator_20260907.json').read_text())['cmd'].encode())<100000
print(json.dumps({'new_shared_memory_matches_original':True,'calls':9,'model_calls':0}))

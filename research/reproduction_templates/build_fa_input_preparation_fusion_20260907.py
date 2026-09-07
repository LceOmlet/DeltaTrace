"""Same FA operator, compiled preparation; bounded original B1/B4 integration."""
import ast,hashlib,json,subprocess,sys
from pathlib import Path
A=Path(__file__).resolve().parent
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
old=(A/'vendor_fa_finite_gqa_runtime.py').read_text()
old_block="""        values=[operands[name].detach().to(dtype=torch.float16).contiguous() for name in names]
        for first,last in [('q0','q1'),('k0','k1')]:
            values.append(((operands[first].float()+operands[last].float())*.5).half().contiguous())"""
assert old.count(old_block)==1
runtime=old.replace('class VendorFAFiniteP1CompactGQA:', 'class VendorFAFiniteP1CompiledPreparation:')
runtime=runtime.replace('import torch\n', 'import torch\nfrom compiled_fa_finite_inputs import prepare_finite_fa_inputs\n',1)
runtime=runtime.replace(old_block,"""        values=list(prepare_finite_fa_inputs(*(operands[name] for name in names)))
        assert len(values)==8 and all(v.is_contiguous() and v.dtype==torch.float16 for v in values)""")
runtime=runtime.replace("            activity['query_heads']=heads", "            activity['input_preparation']='public_torch_compile_inductor_same_expressions'\n            activity['query_heads']=heads",1)
ast.parse(runtime)
(A/'vendor_fa_finite_compiled_prepare_runtime.py').write_text(runtime)
# Reverse only the declared wrapper edits; the operator contract must be exact.
reverse=runtime.replace('class VendorFAFiniteP1CompiledPreparation:', 'class VendorFAFiniteP1CompactGQA:')
reverse=reverse.replace('from compiled_fa_finite_inputs import prepare_finite_fa_inputs\n','')
reverse=reverse.replace("""        values=list(prepare_finite_fa_inputs(*(operands[name] for name in names)))
        assert len(values)==8 and all(v.is_contiguous() and v.dtype==torch.float16 for v in values)""",old_block)
reverse=reverse.replace("            activity['input_preparation']='public_torch_compile_inductor_same_expressions'\n",'')
assert reverse==old

source=(A/'fa_compact_gqa_integration_20260907.py').read_text()
source=source.replace('from vendor_fa_finite_gqa_runtime import VendorFAFiniteP1CompactGQA',
    'from vendor_fa_finite_gqa_runtime import VendorFAFiniteP1CompactGQA\nfrom vendor_fa_finite_compiled_prepare_runtime import VendorFAFiniteP1CompiledPreparation',1)
source=source.replace("compact_extension=VendorFAFiniteP1CompactGQA(p['compact_library'],p['compact_library_sha256'])\nold_extension=extension;old_pair=finite_entry;old_batch=propagate_batch", """compact_extension=VendorFAFiniteP1CompiledPreparation(p['compact_library'],p['compact_library_sha256'])
old_extension=VendorFAFiniteP1CompactGQA(p['compact_library'],p['compact_library_sha256'])
old_pair=compact_pair;old_batch=compact_batch""",1)
source=source.replace('for repeat in range(2):','for repeat in range(3):',1)
source=source.replace("(['old','compact'] if repeat==0 else ['compact','old'])", "(['old','compact'] if repeat in (0,2) else ['compact','old'])",1)
source=source.replace("report['fresh_attributions']<8","report['fresh_attributions']<12")
source=source.replace("report['native_root_forwards']==8", "report['native_root_forwards']==12")
source=source.replace("report['finite_FA_calls_enqueued']==288", "report['finite_FA_calls_enqueued']==432")
source=source.replace("'compact'", "'fused_prepare'")
source=source.replace('compact_B4_profile', 'fused_prepare_B4_profile')
source=source.replace('compact_GQA_B1_and_real_B4_exact', 'FA_input_preparation_B1_and_real_B4_exact')
source=source.replace('FA_GQA_INTEGRATION','FA_INPUT_PREPARATION')
ast.parse(source)
(A/'fa_input_preparation_integration_20260907.py').write_text(source)
p=json.loads((A/'fa_compact_gqa_integration_protocol_20260907.json').read_text())
p.update(purpose='FA priority: compile the entire existing compact-GQA input preparation with public torch.compile/Inductor, keeping native model/FA, finite shared library and expressions unchanged. Twelve whole-network attributions: original NI2 B1 and original NI0/3/6,MH1 B4, one warm plus two reversed/alternating measured pairs each. No MLP cache or target-head optimization, generation, VJP, FT sweep or new curves.',
    study_sha256=sha(A/'fa_input_preparation_integration_20260907.py'),
    wait_for_pid=161276,wait_for_script='${ARTIFACT_ROOT}/codex_fa_compact_gqa_integration_20260907_v1/study.py',
    runtime_source_parent='${ARTIFACT_ROOT}/codex_fa_compact_gqa_integration_20260907_v1',
    repeats='Per group: one old/new warm pair, then new/old and old/new measured pairs.12 total calls; new B4 warm includes profiler, excluded from measured latency.',
    budget={'native_root_forwards':12,'native_vjps':0,'quality_queries':0,'ft_attribution_forwards':0,
        'manual_passes':12,'extra_layer_replay_calls':432,'extra_native_fa_attention_calls':432,
        'finite_FA_calls_enqueued':432,'native_attribution_endpoint_trajectories':60},
    predeclared_review={'numerics':'Full same-job signed vectors and endpoints checked before continuing; preparation expresses exactly the prior FP16 casts and FP32 endpoint mean. FA default precision allowed, no changed finite algorithm.',
        'memory':'Matched current compact-GQA FA baseline; permit resource tradeoffs, no fixed overhead ceiling.',
        'performance':'Two measured pairs per shape and actual B4 kernel dispatch; report all times and compile/warm work. No FT speed or independent quality claim.',
        'stop':'Stop on failed source/model/finite semantics or inability to run; no variant sweep.'})
new_names=['compiled_fa_finite_inputs.py','vendor_fa_finite_compiled_prepare_runtime.py']
for name,digest in p['sources'].items():assert sha(A/name)==digest,name
p['sources'].update({name:sha(A/name) for name in new_names})
p['preparation_wrapper_reverse_exact_to_compact_GQA']=True
p['finite_library_changed']=False
(A/'fa_input_preparation_integration_protocol_20260907.json').write_text(json.dumps(p,indent=2))
cmd=[sys.executable,str(A/'prepare_remote_experiment.py'),'${ARTIFACT_ROOT}/codex_fa_input_preparation_integration_20260907_v1',
    'study.py='+str(A/'fa_input_preparation_integration_20260907.py'),
    'protocol.json='+str(A/'fa_input_preparation_integration_protocol_20260907.json')]
cmd += [name+'='+str(A/name) for name in new_names]
cmd += ['--request',str(A/'launch_fa_input_preparation_integration_20260907.json')]
subprocess.run(cmd,check=True)
assert len(json.loads((A/'launch_fa_input_preparation_integration_20260907.json').read_text())['cmd'].encode())<100000
print(json.dumps({'source_review':'reverse_exact_except_declared_preparation_and_metadata','FA_library_changed':False,'maximum_attributions':12}))

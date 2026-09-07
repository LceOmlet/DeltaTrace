"""Promising FA shared-storage reuse: bounded original B1/B4 validation."""
import ast,hashlib,json,subprocess,sys
from pathlib import Path
A=Path(__file__).resolve().parent;sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
local=json.loads((A/'fa_shared_mean_reuse_operator_summary_20260907.json').read_text())
assert local['full_outputs_exact'] and local['ratios_to_original']['reuse_shared']<1
s=(A/'fa_input_preparation_integration_20260907.py').read_text()
s=s.replace('from vendor_fa_finite_compiled_prepare_runtime import VendorFAFiniteP1CompiledPreparation',
    'from vendor_fa_finite_shared_mean_reuse_runtime import VendorFAFiniteP1SharedMeanReuse')
s=s.replace("compact_extension=VendorFAFiniteP1CompiledPreparation(p['compact_library'],p['compact_library_sha256'])",
    "compact_extension=VendorFAFiniteP1SharedMeanReuse(p['shared_mean_library'],p['shared_mean_library_sha256'])")
s=s.replace('fused_prepare','shared_mean')
s=s.replace('FA_input_preparation_B1_and_real_B4_exact','FA_shared_mean_B1_and_real_B4_exact')
s=s.replace('FA_INPUT_PREPARATION','FA_SHARED_MEAN')
ast.parse(s);(A/'fa_shared_mean_integration_20260907.py').write_text(s)
p=json.loads((A/'fa_input_preparation_integration_protocol_20260907.json').read_text())
p.update(purpose='FA priority: integrate the verified shared-storage midpoint variant after 15% local operator reduction. Original NI2 B1 and original NI0/3/6,MH1 B4; current compact-GQA versus shared-mean reuse,1warm+2measured each,12 whole-model attributions. Original default FA, P1 finite rules and default precision retained;0 generation,VJP,FT curves or quality queries. No MLP/head changes.',
    study_sha256=sha(A/'fa_shared_mean_integration_20260907.py'),
    wait_for_pid=173068,wait_for_script='${ARTIFACT_ROOT}/codex_fa_shared_mean_reuse_operator_20260907_v1/study.py',
    runtime_source_parent='${ARTIFACT_ROOT}/codex_fa_input_preparation_integration_20260907_v1',
    shared_mean_library='${ARTIFACT_ROOT}/codex_fa_shared_mean_reuse_operator_20260907_v1/libdeltatrace_fa_finite_shared_mean_reuse.so',
    shared_mean_library_sha256=local['library_sha256'],
    shared_mean_local_raw_sha256=local['raw_sha256'],
    finite_library_changed=True,
    predeclared_review={'numerics':'Same-job full signed vectors/endpoints exact expected; midpoint remains FP32 add followed by FP16 cast. No centering, coefficient-cast or pass-order changes.',
        'memory':'Matched current compact-GQA FA baseline; report peak and linear buffers, no fixed excess ceiling.',
        'performance':'Two alternating measured pairs per shape and actual B4 dispatch in existing warm call. No FT or independent quality claim.',
        'stop':'Stop on source/model/finite validity failure; no shape/parameter sweep.'})
p.pop('preparation_wrapper_reverse_exact_to_compact_GQA',None)
for n in ['compiled_fa_finite_inputs.py','vendor_fa_finite_compiled_prepare_runtime.py']:p['sources'].pop(n)
new='vendor_fa_finite_shared_mean_reuse_runtime.py';p['sources'][new]=sha(A/new)
for name,digest in p['sources'].items():assert sha(A/name)==digest
(A/'fa_shared_mean_integration_protocol_20260907.json').write_text(json.dumps(p,indent=2))
cmd=[sys.executable,str(A/'prepare_remote_experiment.py'),'${ARTIFACT_ROOT}/codex_fa_shared_mean_integration_20260907_v1',
    'study.py='+str(A/'fa_shared_mean_integration_20260907.py'),
    'protocol.json='+str(A/'fa_shared_mean_integration_protocol_20260907.json'),new+'='+str(A/new),
    '--request',str(A/'launch_fa_shared_mean_integration_20260907.json')]
subprocess.run(cmd,check=True)
assert len(json.loads((A/'launch_fa_shared_mean_integration_20260907.json').read_text())['cmd'].encode())<100000

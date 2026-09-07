"""Adapt the frozen full-array checks to the explicitly separate operator job."""
from pathlib import Path
A=Path(__file__).resolve().parent
base=(A/'verify_vendor_fa_finite_actual_20260907.py').read_text()
pre='''"""Verify real-tensor finite FA operators without promoting failed capture."""
import ast,hashlib,json,statistics
from pathlib import Path
import numpy as np
A=Path(__file__).resolve().parent;F=A/'snapshot${ARTIFACT_ROOT}/codex_vendor_fa_captured_operator_20260907_v2'
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
d=json.loads((F/'results.json').read_text());p=d['protocol']
assert d['status']=='complete'
assert p==json.loads((A/'vendor_fa_captured_operator_protocol_20260907.json').read_text())
assert sha(F/'study.py')==p['study_sha256']==sha(A/'vendor_fa_captured_operator_20260907.py')
for name,digest in p['sources'].items():assert sha(F/name)==sha(A/name)==digest
source=A/'snapshot'/p['capture_results'].lstrip('/');assert sha(source)==p['capture_results_sha256']
captured=json.loads(source.read_text())
assert captured['status']=='failed' and not captured['operator_attempts']
assert not captured['records'][0]['signed_vector_exact_to_parent']
assert d['capture_exact_parent_guard']=='failed_in_predecessor_not_relaxed'
capture=captured['captured_layers'][0];assert capture==p['actual_operands']
assert d['native_sources_before']==d['native_sources_after']==p['native_source_sha256']
assert d['native_root_forwards']==d['native_vjps']==d['manual_passes']==0
build_path=A/'snapshot'/p['build_result'].lstrip('/');assert sha(build_path)==p['build_result_sha256']
build=json.loads(build_path.read_text());assert build['library']['sha256']==p['library_sha256']
failure=json.loads((A/'vendor_fa_capture_failure_summary_20260907.json').read_text())
assert failure['raw_sha256']==sha(source) and failure['parent_vector_guard']=='failed_preserved'
'''
tail=base[base.index('\ndef load(meta):'):]
tail=tail.replace("'capture_budget':p['capture_budget']", "'capture_budget_spent_in_predecessor':failure['budget'],'failed_capture_preserved':failure")
tail=tail.replace("vendor_fa_finite_actual_summary_20260907.json","vendor_fa_captured_operator_summary_20260907.json")
tail=tail.replace("assert sorted(kernels)==sorted(profile['GPU_kernels'])", "assert sorted(e['name'] for e in events if e.get('cat') in ['kernel','gpu_memcpy'])==sorted(profile['GPU_kernels'])  # CUDA events include separately categorized host copies")
extra='''
out['local_sign_rounding']={}
for key in ['dq','dk','dv']:
    x=outputs['finite_FA'][key].astype(np.float64);y=data['expected_'+key].astype(np.float64)
    zero=(y!=0)&(x==0);flip=x*y<0
    out['local_sign_rounding'][key]={'nonzero_to_zero':int(zero.sum()),
        'zeroed_reference_l2_fraction':float(np.linalg.norm(y[zero])/np.linalg.norm(y)),
        'opposite_nonzero_sign':int(flip.sum()),
        'flipped_reference_l2_fraction':float(np.linalg.norm(y[flip])/np.linalg.norm(y))}
v1=A/'snapshot${ARTIFACT_ROOT}/codex_vendor_fa_captured_operator_20260907_v1/results.json'
failed=json.loads(v1.read_text());assert failed['status']=='failed'
assert len(failed['operator_attempts'])==1 and not failed['operator_attempts'][0]['complete']
assert failed['operator_attempts'][0]['error'].endswith("ModuleNotFoundError: No module named 'signed_secant_rules'\\n")
out['earlier_packaging_failure']={'raw_sha256':sha(v1),'missing_dependency':'signed_secant_rules',
    'dense_attempted_not_executed':1,'finite_operator_calls':0,'model_calls':0,
    'elapsed_seconds':failed['elapsed_seconds'],'preserved':True}
out['limits']=['One actual original NI0 layer35, B1 H32 N601 D128; not end-to-end or long-sequence validation.',
    'Finite half intermediates and native LSE differ numerically from old explicit FP32 normalization.',
    'Profile CUDA kernels include wrapper and output-diagnostic kernels; only three are finite-extension kernels.',
    'Standard FA is native GQA forward+backward cost reference, not the finite mathematical reference.',
    'Ordinary FA repeated gradients are not bitwise identical; no repeat-stability claim for that reference.',
    'Dense local reference is minimal required P1 math with eager logarithmic mean, not a whole production core benchmark.']
'''
tail=tail.replace("(A/'vendor_fa_captured_operator_summary_20260907.json')",extra+"\n(A/'vendor_fa_captured_operator_summary_20260907.json')")
(A/'verify_vendor_fa_captured_operator_20260907.py').write_text(pre+tail,encoding='utf-8')
print('Prepared standalone verifier.')

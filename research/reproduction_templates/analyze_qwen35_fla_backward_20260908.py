"""Audit the real native backward, preserved failure, and reusable WY value branch."""
import hashlib
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path
import numpy as np
A=Path(__file__).resolve().parent;R=A.parent/'DeltaTrace'
sha=lambda b:hashlib.sha256(b).hexdigest()
sys.path.insert(0,str(R/'research/runtime'))
from native_batch_diagnostics import metrics
from fla_wy_upstream_layout_backport import transformed_source
parent=A/'snapshot${ARTIFACT_ROOT}/codex_qwen35_batch_diagnostic_20260908_v1'
roots=[A/('snapshot${ARTIFACT_ROOT}/codex_qwen35_fla_backward_20260908_v'+str(i)) for i in [1,2]]
results=[]; profiles=[]
for version,D in enumerate(roots,1):
    archive=D/('review_compact.zip' if (D/'review_compact.zip').exists() else 'review_bundle.zip')
    if archive.exists():
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None
            for name in z.namelist():
                assert Path(name).name==name
                (D/name).write_bytes(z.read(name))
    raw=(D/'results.json').read_bytes();r=json.loads(raw);p=json.loads((D/'protocol.json').read_bytes())
    assert p==r['protocol'] and sha((D/'study.py').read_bytes())==p['study_sha256']
    assert sha((D/'native_fla_stage_capture.py').read_bytes())==p['stage_runtime_sha256']
    assert sha((parent/'results.json').read_bytes())==p['required_parent_raw_sha256']
    assert sha((parent/p['input_npz']).read_bytes())==p['input_npz_sha256']
    assert all(r[k]==0 for k in ['model_loads','model_load_attempts','root_forward_attempts','attributions','quality_queries','generation_calls'])
    for item in r['artifacts']:
        if (D/item['file']).exists():
            assert sha((D/item['file']).read_bytes())==item['sha256']
        else:
            assert version==2 and item['file'].startswith('completed_stage_')
    assert r['native_stage_references_after_release']==0
    if (D/r['native_profile']['file']).exists():
        profile=(D/r['native_profile']['file']).read_bytes()
        assert sha(profile)==r['native_profile']['sha256']
        events=json.loads(profile)['traceEvents'];counts=Counter(e['name'] for e in events if e.get('cat')=='kernel')
        del events,profile
        profile_scope='Full profile hash and kernel counts recomputed locally.'
    else:
        compact=json.loads((D/'native_operator_kernel_summary.json').read_bytes())
        assert compact['profile_sha256']==r['native_profile']['sha256']
        counts=Counter(compact['all_kernel_counts'])
        profile_scope='Full profile hash verified and counts extracted remotely; compact counts read locally. Full trace retained in remote bundle.'
    assert counts
    groups={term:sum(n for name,n in counts.items() if term in name) for term in [
        'chunk_gated_delta_rule_fwd_kernel_h','chunk_gated_delta_rule_bwd_kernel_dhu',
        'chunk_bwd_kernel_dv','chunk_bwd_kernel_dqkwg','prepare_wy_repr_bwd_kernel','l2norm_bwd']}
    profiles.append({'raw_profile_sha256':r['native_profile']['sha256'],'scope':profile_scope,'selected_kernel_counts':groups,
                     'all_kernel_counts':dict(counts)})
    results.append(r)
first,last=results;D=roots[1]
assert first['status']=='failed' and first['native_forwards_completed']==1 and first['native_backwards_completed']==0
assert 'prepare_wy_repr_bwd' in first['error'] and 'PassManager::run failed' in first['error']
assert first['CPU_forwards_completed']==first['CPU_backwards_completed']==0
assert last['status']=='native_FLA_backward_and_official_CPU_reference_executed',last.get('error')
assert all(last[k]==1 for k in ['native_forward_attempts','native_forwards_completed','native_backward_attempts',
                             'native_backwards_completed','CPU_forward_attempts','CPU_forwards_completed',
                             'CPU_backward_attempts','CPU_backwards_completed'])
assert last['sources_before']==first['sources_before']
assert last['sources_during']==last['sources_after']
assert last['native_stage_counts']=={'forward':1,'state_forward':2,'backward':1,'state_backward':1,'wy_backward':1}
assert not last.get('partial_capture_errors')
assert profiles[1]['selected_kernel_counts']['prepare_wy_repr_bwd_kernel']>0
assert profiles[1]['selected_kernel_counts']['chunk_gated_delta_rule_bwd_kernel_dhu']>0
layout=json.loads((D/'layout_backport_receipt.json').read_bytes())
original=(A/'snapshot${ARTIFACT_ROOT}/codex_qwen35_isolated_import_20260908_v1/env/lib/python3.12/site-packages/fla/ops/gated_delta_rule/wy_fast.py').read_bytes()
assert sha(transformed_source(original,(D/'upstream_wy_fast.py').read_bytes()))==layout['after_sha256']
assert layout==last['layout_backport']

def load(path):
    with np.load(path,allow_pickle=False) as f:return {k:f[k] for k in f.files}
inputs=load(parent/'FLA_prefix_call0.npz')
native_out=load(D/'native_output.npz')['output'];ref_out=load(D/'official_CPU_output.npz')['output']
native_grad=load(D/'native_gradients.npz');ref_grad=load(D/'official_CPU_gradients.npz')
gradient_differences={key:metrics(ref_grad[key],native_grad[key]) for key in native_grad}
for key,value in gradient_differences.items():
    a=ref_grad[key].astype(np.float64).ravel();b=native_grad[key].astype(np.float64).ravel()
    value['cosine']=float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)))
fwd=load(D/'native_stage_forward.npz');state_bwd=load(D/'native_stage_state_backward.npz');wy=load(D/'native_stage_wy_backward.npz')
assert fwd['A'].shape==(1,129,32,64) and state_bwd['dU_WY'].shape==(1,129,32,128)
assert np.array_equal(state_bwd['dU_WY'],wy['du'])
assert np.array_equal(wy['dV'],native_grad['v'])
# Independent CPU64 tile contraction, not a production attention/backward.
branch=np.empty_like(native_grad['v'],dtype=np.float64)
for lo in range(0,129,64):
    hi=min(lo+64,129);width=hi-lo
    tile=fwd['A'][:,lo:hi,:,:width].astype(np.float64).transpose(0,2,1,3)
    adjoint=state_bwd['dU_WY'][:,lo:hi].astype(np.float64).transpose(0,2,1,3)
    transformed=(tile.swapaxes(-1,-2)@adjoint).transpose(0,2,1,3)
    branch[:,lo:hi]=transformed*inputs['beta'][:,lo:hi,:,None]
branch_difference=metrics(branch,native_grad['v'])
seed=inputs['output'].astype(np.float64)
linearity={}
for label,out,grad in [('native',native_out,native_grad['v']),('official_CPU',ref_out,ref_grad['v'])]:
    j=float(np.sum(out.astype(np.float64)*seed));v_dot=float(np.sum(inputs['v'].astype(np.float64)*grad.astype(np.float64)))
    linearity[label]={'J':j,'v_dot_gradient':v_dot,'relative_residual':abs(j-v_dot)/max(abs(j),1e-30)}
summary={'status':'native_backward_with_upstream_layout_backport_verified_on_saved_prefix',
    'raw_sha256':[sha((root/'results.json').read_bytes()) for root in roots],
    'study_sha256':[r['protocol']['study_sha256'] for r in results],
    'protocol_sha256':[sha((root/'protocol.json').read_bytes()) for root in roots],
    'budget':{'model_loads':0,'model_forwards':0,'native_FLA_forward_attempts':2,'native_FLA_forwards_completed':2,
              'native_FLA_backward_attempts':2,'native_FLA_backwards_completed':1,'official_CPU_forwards':1,
              'official_CPU_backwards':1,'attributions':0,'quality_queries':0,'generation_calls':0},
    'failure':{'stage':'prepare_wy_repr_bwd_kernel','error':'arith.mulf encoding mismatch during Triton lowering',
               'CPU_reference_skipped':True,'passive_backward_stage_hook_missed_worker_thread':True,
               'partial_stage_arrays_not_saved':True},
    'backport':layout,'sources_after':last['sources_after'],'native_stage_counts':last['native_stage_counts'],
    'profiles':profiles,'output_vs_reference':metrics(ref_out,native_out),
    'output_vs_actual_parent_prefix':metrics(inputs['output'],native_out),
    'output_before_after_backport':metrics(load(roots[0]/'native_output.npz')['output'],native_out),
    'output_equal_before_after_backport':np.array_equal(load(roots[0]/'native_output.npz')['output'],native_out),
    'gradient_vs_official_CPU':gradient_differences,'value_WY_contraction_vs_native_gradient':branch_difference,
    'native_WY_value_is_native_autograd_value':True,'value_linearity_identity':linearity,
    'retained_native_stage_storage_bytes':last['retained_native_stage_storage_bytes'],
    'native_stage_references_after_release':last['native_stage_references_after_release'],
    'diagnostic_cost_not_steady_performance':[{'job_seconds':r['job_seconds'],'operator_seconds':r['native_diagnostic_seconds'],
         'peak_allocated_bytes':r['native_peak_allocated_bytes']} for r in results],
    'CPU_reference_seconds':last['CPU_reference_seconds'],
    'artifacts_by_attempt':[r['artifacts'] for r in results],
    'limits':['Only actual NI0 layer0 prefix129, one fixed output cotangent, zero initial state; no complete model backward or batch backward.',
              'Native ordinary gradients are not all finite coefficients; mixed q/k/g/beta and endpoint0 states remain unimplemented.',
              'Public autograd engine multithreading was disabled only to observe backward stages; this is diagnostic timing.',
              'CPU reference and NumPy tile checks are explicitly validation only, never used as model or production backward replacements.']}
summary['local_compact_archive_scope']='The seven named output/gradient/native-stage arrays are local and hash-verified; duplicate completed-stage snapshots and the full419MB v2 trace remain in the remote full bundle.'
summary['capture_release_limit']='Executed driver cleared capture.records but retained loop-variable aliases to stage dictionaries. Zero list length is not proof of complete GPU release. Current helper invalidates those borrowed dictionaries; container lifetime checked locally without another GPU run.'
(A/'qwen35_fla_backward_summary_20260908.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps({k:summary[k] for k in ['status','raw_sha256','output_vs_reference','gradient_vs_official_CPU',
    'value_WY_contraction_vs_native_gradient','value_linearity_identity','retained_native_stage_storage_bytes']}))

"""Read-only source/CPU checks supporting FA integration decisions."""
import __future__,ast,hashlib,json,os
from pathlib import Path
import numpy as np
A=Path(__file__).resolve().parent;R=A.parent/'DeltaTrace';sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
source=A/'snapshot${SITE_PACKAGES}/transformers/modeling_flash_attention_utils.py'
tree=ast.parse(source.read_bytes());f=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='_process_flash_attention_kwargs')
namespace={'os':os};exec(compile(ast.Module(body=[f],type_ignores=[]),str(source),'exec',flags=__future__.annotations.compiler_flag),namespace)
args=dict(query_length=601,key_length=601,is_causal=True,dropout=0.0,softmax_scale=128**-0.5,
    supports_mapping={k:True for k in ['dropout_p','window_size','deterministic','softcap','s_aux']})
before=namespace[f.name](**args,return_attn_probs=False)
after=namespace[f.name](**args,return_attn_probs=True)
assert before==after and 'return_attn_probs' not in after
F=A/'snapshot${ARTIFACT_ROOT}/codex_fa_two_sweeps_operator_20260907_v1'
d=json.loads((F/'results.json').read_text());assert d['status']=='failed' and len(d['calls'])==2
assert all(d['comparisons'][0][k]['exact'] for k in ['tau','center','dk','dv'])
a=np.load(F/'old_dq_repeat0.npy',allow_pickle=False).astype(np.float64)
b=np.load(F/'two_sweeps_dq_repeat0.npy',allow_pickle=False).astype(np.float64)
rms=float(np.sqrt(np.mean(a*a)));err=float(np.sqrt(np.mean((a-b)**2)))
assert abs(err/rms-d['comparisons'][0]['dq']['relative_L2'])<1e-12
# Values/coordinates are diagnostic evidence only. No permutation is applied to
# an attribution score, used as a correction, or substituted into evaluation.
top_a=a[abs(a)>0.3];top_b=b[abs(b)>0.3]
assert np.array_equal(np.sort(top_a),np.sort(top_b))
vendor=R/'third_party/metax_fa_2_5_3/csrc/flash_attn/src'
launch=(vendor/'flash_bwd_launch_template.h').read_text()
assert 'flash::convert_dQ_hdim128_32_32<Kernel_traits>(params, nsplits);' in launch
out={'status':'source_and_saved_numeric_review_complete','GPU_calls':0,'model_calls':0,
    'LSE_request':{'source_sha256':sha(source),'function_line':f.lineno,
        'default_kwargs':before,'requested_kwargs':after,'request_discarded':True,
        'decision':'Do not remove auxiliary public FA call through an unsupported model kwarg. No private slot or shadow model replacement introduced.'},
    'first_operator_failure':{'raw_sha256':sha(F/'results.json'),'operator_calls':2,
        'dq_relative_L2':err/rms,'other_outputs_exact':True,
        'same_large_values_different_coordinates':True,
        'old_top_coordinates':np.argwhere(abs(a)>0.3).tolist(),'new_top_coordinates':np.argwhere(abs(b)>0.3).tolist(),
        'diagnosis':'Generic conversion was called despite native launcher selecting the32x32 specialization. Coordinate error, not ordinary FA precision. Correct dispatch and matching upstream traits must be tested; no output-permutation workaround.'},
    'native_dispatch_source_sha256':sha(vendor/'flash_bwd_launch_template.h'),
    'native_specialization_source_sha256':sha(vendor/'flash_bwd_preprocess_kernel_hdim128_32_32.h'),
    'arithmetic_work':{'old_products_per_causal_tile_group':12,'new_products_per_causal_tile_group':9,
        'basis':'Old row-center3 + query3score/1output + key3score/2output; new row-center3 + key3score/3output. All products have32*32*128 multiplications per full tile. Two quadratic sweeps plus one linear native conversion remain.',
        'matrix_multiply_work_reduction_fraction':0.25,'wall_clock_speedup_claim':False,
        'additional_cost':'FP32 atomic dQ accumulation,2KiB shared coefficient tile,linear FP32 workspace and native conversion.'}}
later=A/'fa_two_sweeps_operator_summary_20260907.json'
if later.exists():
    last=json.loads(later.read_text())['attempts'][-1]
    out['subsequent_verified_result']={'summary_sha256':sha(later),
        'native_accumulation_and_conversion_must_be_paired':True,
        'status':last['status'],'numerical_gate_passed':last['numerical_gate_passed'],
        'relative_L2_max':max(c['dq']['relative_L2'] for c in last['comparisons']),
        'new_to_old_median_ratio':last['new_to_old_median_ratio'],
        'whole_model_screen_allowed':last['whole_model_screen_allowed']}
(A/'fa_lse_two_sweeps_source_summary_20260907.json').write_text(json.dumps(out,indent=2))
print(json.dumps({k:out[k] for k in ['status','GPU_calls','model_calls','LSE_request','arithmetic_work']}))

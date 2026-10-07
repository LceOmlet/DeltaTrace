"""Record the measured scope of the saved-operand precision control."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[4]


def binding(p):
    return dict(path=p.as_posix(),bytes=p.stat().st_size,
                sha256=hashlib.sha256(p.read_bytes()).hexdigest())


raw=HERE/'fla-replay-results/results/result.json'
d=json.loads(raw.read_bytes())
launch=json.loads((HERE/'fla-replay-results/launch.json').read_bytes())
assert d['phase']=='complete' and all(v['complete'] for v in d['profiles'].values())
native=d['profiles']['eager_original_native_GEMM']
fp32=d['profiles']['eager_FP32_intermediate_GEMM']
out=dict(status='Intermediate GEMM operand rounding does not explain this observed sign flip; credit not repaired',
    diagnostic_code_commit=launch['code_commit'],helper_sha256=launch['helper_sha256'],
    pid=launch['pid'],birth=launch['birth'],devices=launch['devices'],owners=d['owners'],
    measured=dict(original_saved_B4_point=d['actual_saved_native_point'],
        eager_same_native_GEMM_point=native['contraction_including_same_residual_and_z'],
        eager_FP32_intermediate_GEMM_point=fp32['contraction_including_same_residual_and_z'],
        native_eager_minus_saved=native['contraction_including_same_residual_and_z']-d['actual_saved_native_point'],
        FP32_control_minus_native_eager=fp32['contraction_including_same_residual_and_z']-native['contraction_including_same_residual_and_z']),
    profiles={name:dict(groups=len(v['groups']),operator_seconds=sum(g['seconds'] for g in v['groups']),
        group_end_max_live_allocated_bytes=max(g['allocated_bytes'] for g in v['groups']),
        max_recorded_pss_bytes=max(g['pss_bytes'] for g in v['groups']),
        max_coefficient_error_vs_original_saved_B4={key:max(g['compared_with_original_saved_B4'][key]['maxabs'] for g in v['groups'])
            for key in v['groups'][0]['compared_with_original_saved_B4']}) for name,v in d['profiles'].items()},
    allow_tf32=d['allow_tf32'],float32_matmul_precision=d['float32_matmul_precision'],
    sources=[binding(raw),binding(HERE/'fla-replay-results/transport.json'),
        binding(REPO/'experiments/rl/results_current_extreme_gdn_20261008.json')],
    scope=d['scope'],limitations=[
        'This changes only the finite formula GEMM operand casts. Native FLA adjoint stages, captured low-precision model endpoints, elementwise/Triton stages and symmetric formulas remain unchanged. It does not rule out every other numerical or formula defect.',
        'Selected B1 eager replay is compared numerically to the original saved B4 compiled outputs. No official FA/FLA pass criterion is claimed for this nonzero finite computation.',
        'The two timings include their own cold/warm ordering and are not a production speed comparison.',
        'Group-end allocated and PSS values are observations, not continuous physical VRAM or process-memory peaks.',
    ],official_tolerance_claim=False,production_profile_changed=False,credit_repaired=False,
    formal_restart=False,text_update_released=False,
    operations=dict(model_load=0,full_DT=0,rollout=0,training_backward=0,optimizer=0,checkpoint_restore=0))
target=REPO/'experiments/rl/results_current_extreme_fla_precision_20261008.json'
target.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(receipt=binding(target),measured=out['measured']),indent=2))

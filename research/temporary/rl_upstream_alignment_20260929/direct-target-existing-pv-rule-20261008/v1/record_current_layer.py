"""Bind the current real-token layer result without changing its estimator."""
import hashlib
import json
from pathlib import Path
import re

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[4]


def binding(path):
    return dict(path=path.as_posix(),bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


analysis=json.loads((HERE/'current-layer-analysis.json').read_bytes())
launch=json.loads((HERE/'layer-results/launch.json').read_bytes())
log=HERE/'layer-results/driver.log'
text=re.sub(r'\x1b\[[0-9;]*m','',log.read_text(errors='replace'))
body=re.search(r'"layer_types":\s*\[(.*?)\]',text,re.S).group(1)
types=re.findall(r'"(linear_attention|full_attention)"',body)
assert len(types)==32 and types[30]=='linear_attention' and types[31]=='full_attention'
previous=HERE.parents[1]/'direct-target-credit-sample-20261007/v1/sample-analysis.json'
points=json.loads(previous.read_bytes())['tasks']['appworld']['points']
native=next(p for p in points if p['row']==3 and p['mode']=='most_negative')
assert native['traj_uid']==launch['case_binding']['value']['case']['uid']
r=analysis['ranks'][0]
out=dict(status='Current maximum-negative token localized to decoder30; credit not repaired',
    diagnostic_code_commit=launch['code_commit'],source_sha256=launch['source_sha256'],
    native_sha256=launch['native_sha256'],case_binding=launch['case_binding'],
    geometry=r['candidate'],original_layer_types=types,layer_type_source=binding(log),
    first_negative_boundary=r['first_negative_during_reverse_traversal'],
    measured=dict(single_DT_d=r['single_source_signed'],
        same_job_native_cached_single_target_difference=r['single_source_focused_root'],
        previous_independent_uncached_native_single_target_difference=native['native_single_delete_d'],
        single_DT_minus_same_job_cached_endpoint=r['single_source_signed']-r['single_source_focused_root'],
        cached_minus_previous_uncached_single_effect=r['single_source_focused_root']-native['native_single_delete_d'],
        joint_token_d=r['original_joint_candidate_signed'],
        joint_head_coefficient_times_single_delta=r['boundaries'][0]['joint_coefficient_times_single_deletion_delta'],
        before_decoder30=r['boundaries'][1]['joint_coefficient_times_single_deletion_delta'],
        after_decoder30=r['boundaries'][2]['joint_coefficient_times_single_deletion_delta']),
    controls=analysis['controls'],unchanged_joint_outputs=analysis['unchanged_original_outputs'],
    candidate_factual_endpoints_equal_all_33_boundaries=r['factual_endpoint_all_equal'],
    resources=analysis['resource_scope'],elapsed_by_rank=[x['seconds'] for x in analysis['ranks']],
    finite_owners=[x['finite_owner'] for x in analysis['ranks']],
    sources=[binding(HERE/'current-layer-analysis.json'),binding(HERE/'layer-results/transport.json'),
        binding(HERE/'layer-results/results/effective-config.yaml'),binding(previous)],
    limitations=[
        'A changed contraction localizes where the joint-reference coefficients act differently on a real single-deletion hidden difference. It is not itself proof of a numerically faulty GDN kernel or a complete repair.',
        'The same-job native cached single effect and a previous independent uncached effect differ. They are separately reported, not relabeled as the same endpoint or hidden by a correction.',
        'The single-DT value also differs from its current native cached endpoint difference. No invented tolerance labels this discrepancy acceptable.',
        'Factual endpoint equality concerns the selected candidate row. Whole joint signed equality concerns this evidenced B4 and does not replace FA/FLA tolerance tests.',
        'Native-root identity/cache hooks did not observe the root model.__call__ path. Their absence is not a zero-difference claim; original boundary endpoints were captured separately.',
        'Physical VRAM was observed at polls, not continuously sampled by this diagnostic; recorded Torch allocator/PSS observations are not physical-VRAM peak measurements.',
    ],
    operations=dict(DT_per_rank=2,rollout=0,backward=0,optimizer=0,checkpoint_restore=0),
    formal_restart=False,text_update_released=False,production_profile_changed=False,
    credit_repaired=False,official_tolerance_claim=False)
target=REPO/'experiments/rl/results_current_extreme_layer_20261008.json'
target.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(receipt=binding(target),measured=out['measured'],
                     joint_outputs_exact_equal=[x['whole_signed_equal'] for x in out['unchanged_joint_outputs']]),indent=2))

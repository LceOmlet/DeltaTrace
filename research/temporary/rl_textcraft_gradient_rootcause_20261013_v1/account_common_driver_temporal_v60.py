"""Join saved native trend and common-group exposure; no model or regrouping."""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

HERE=Path(__file__).resolve().parent
native_path=Path('F:/DeltaTraceBackups/plots/gradient-rootcause-20261014/native-gradient-growth-85.csv')
groups_path=HERE/'account_success_state_groups_v6.json'
fixed_path=HERE/'native-fixed-seed-propagation-summary-v33.json'
middle_path=HERE/'native-midpoint-fixed-seed-summary-v35.json'
local_path=HERE/'account_common_update_feedback_v57-dt.json'
interface_path=HERE/'verify_DT_group_multiplier_v40-dt.json'
native={int(r['iteration']):float(r['native_grad_norm']) for r in csv.DictReader(native_path.open(encoding='utf-8-sig'))}
groups=json.loads(groups_path.read_text())['report']
rows={r['iteration']:r for r in groups['rows']}
blocks=[]
for start,end in [(1,10),(11,20),(21,30),(31,40),(41,50),(51,60),(61,70),(71,80),(81,85)]:
    values=[rows[i] for i in range(start,end+1)]
    blocks.append(dict(first=start,last=end,native_grad_norm_median=statistics.median(native[i] for i in range(start,end+1)),
        eight_success_state_count_median=statistics.median(r['states_with_eight_successes'] for r in values),
        retained_positive_target_fraction_in_eight_success_states_median=statistics.median(r['retained_target_fraction_in_eight_success_states'] for r in values),
        observed_positive_state_collisions_over8=sum(r['state_collisions_over_eight'] for r in values)))
fixed=json.loads(fixed_path.read_text());middle=json.loads(middle_path.read_text())
g5=fixed['cases']['late5_seed80']['parameter_VJP_norm'];g80=fixed['cases']['late80_native']['parameter_VJP_norm']
# The completed midpoint summary exports its measured VJP directly as G.
assert middle['checkpoint']==40 and middle['status']=='completed_exited_native_VJP_not_repair'
g40=middle['G']
local=json.loads(local_path.read_text())['report']
interface=json.loads(interface_path.read_text())['report']
report=dict(scope=__doc__,sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
    (native_path,groups_path,fixed_path,middle_path,local_path,interface_path)},
    native_blocks=blocks,native81_state_to_UID_calibration=groups['native81_group_calibration'],
    conditional_update_driver=local['observed_local_feedback_signs'],
    fixed_input_fixed_seed_parameter_VJP={5:g5,40:g40,80:g80},
    descriptive_fixed_seed_log_growth_per_iteration={'5_to40':math.log(g40/g5)/35,'40_to80':math.log(g80/g40)/40},
    native_interface_fix_evidence={k:interface[k] for k in ('rows','positive_reward_zero_group_rows',
        'actual_zero_group_DT_path_zero','DT_advantage_boundary_exact_passthrough','raw_rewards_unchanged','no_shadow_GRPO')},
    rejected_complete_explanation='Instantaneous count of successful constant-return groups alone: their exposure decreases after61-70 while native gradient medians continue to rise. Counts are not parameter gradients.',
    supported_chain='Old DTPO retains updates on constant-positive groups that the official GRPO multiplier nulls. Native old-state ablation changes the sign of subsequent fixed-adjoint VJP growth. Historical same-input/same-seed sensitivity accelerates in the late stage, so the relevant mechanism includes accumulated parameter/state sensitivity, not only instantaneous coefficient or sample count.',
    limits=groups['limits']+[
        'Earlier initial-state groups are descriptive, except where native UIDs were actually calibrated. Zero observed collisions over8 does not prove one-to-one mapping at every iteration.',
        'The log rates describe two completed fixed-input/seed intervals; they do not establish a global exponential or quadratic law.',
        'A late-state conditional intervention does not identify the entire historical cumulative update path.',
        'Official zero group multipliers block the measured constant-positive component in the new method, but this is not acceptance of the complete long-term root-cause repair.'],
    new_model_calls=0,new_optimizer_calls=0,GRPO_reruns=0,rootcause_repair_verified=False)
path=HERE/'common-driver-temporal-evidence-v60.json';path.write_text(json.dumps(report,indent=2)+'\n')
print('midpoint_norm',g40)
print('blocks',[(b['first'],b['last'],b['native_grad_norm_median'],b['eight_success_state_count_median']) for b in blocks])
print('log_rates',report['descriptive_fixed_seed_log_growth_per_iteration'])
print('native_interface',report['native_interface_fix_evidence'])
print('receipt',path)

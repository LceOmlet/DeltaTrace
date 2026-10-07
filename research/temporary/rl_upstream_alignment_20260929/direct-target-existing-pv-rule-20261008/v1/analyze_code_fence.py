"""Analyze actual target-score isolation; no new credit rule or tolerance."""
import hashlib
import json
from pathlib import Path
import re

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
FOLDER = HERE / 'code-fence-results'
MODE = 'original_next_code_fence_score_only'


def binding(path):
    return dict(path=path.as_posix(), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


transport = json.loads((FOLDER/'transport.json').read_bytes())
for item in transport:
    assert binding(Path(item['local_path']))['sha256'] == item['sha256']
launch = json.loads((FOLDER/'launch.json').read_bytes())
reports = [json.loads((FOLDER/f'results/rank{rank}.json').read_bytes()) for rank in (0, 1)]
traces = [torch.load(FOLDER/f'results/rank{rank}-{MODE}.pt',
                     map_location='cpu', weights_only=False) for rank in (0, 1)]
full_path = HERE/'memory-results/results/rank0-original_symmetric_memory.pt'
full = torch.load(full_path, map_location='cpu', weights_only=False)
native_path = HERE.parents[1]/'direct-target-credit-sample-20261007/v1/results-appworld/appworld-rank0-most_negative-targets.pt'
native = torch.load(native_path, map_location='cpu', weights_only=False)
actual_targets = native['samples'].eq(3)
selected = actual_targets & native['predictor_positions'].eq(2883) & native['labels'].eq(71093)
assert int(selected.sum()) == 1
native_d = native['factual_target_logp'].double() - native['reference_target_logp'].double()
full_d, partial_d = [float(value['signed'][3, 2883]) for value in (full, traces[0])]
for report, trace in zip(reports, traces):
    assert report['phase'] == 'complete' and report['operations']['DT'] == 1
    assert trace['diagnostic_only'] and 'values' not in trace and 'report' not in trace
    assert trace['source_sha256'] == full['source_sha256'] == launch['source_sha256']
    assert trace['native_sha256'] == full['native_sha256'] == launch['native_sha256']
    selection = trace['target_selection']
    assert selection['original_reference_sources_exact'] and selection['original_target_input_tokens_retained']
    assert selection['score_target_counts'] == [702, 1119, 883, 1]
    assert not selection['training_target_changed']
    assert trace['detail']['norm_gate_rules'] == full['detail']['norm_gate_rules']
    assert trace['detail']['finite_fla_by_layer'] == full['detail']['finite_fla_by_layer']
    assert trace['detail']['attention_pv_rules'] == full['detail']['attention_pv_rules']
physical = []
for line in (FOLDER/'results/physical-mx-smi.jsonl').read_bytes().splitlines():
    event = json.loads(line)
    gpu = None
    for text in event['stdout'].splitlines():
        board = re.match(r'^\|\s*(\d+)\s+MetaX\s', text)
        if board:
            gpu = int(board[1])
        usage = re.search(r'(\d+)/(\d+) MiB', text)
        if usage and gpu in (4, 5):
            physical.append(dict(gpu=gpu, used_mib=int(usage[1])))
details = traces[0]['detail']['per_sample'][3]
other_target_count = sum(traces[0]['target_selection']['score_target_counts'][:3])
endpoint_controls = {}
for key in ('target_logp0', 'target_logp1'):
    original = torch.tensor(full['detail'][key][:other_target_count], dtype=torch.float64)
    isolated = torch.tensor(traces[0]['detail'][key][:other_target_count], dtype=torch.float64)
    endpoint_controls[key] = dict(exact=torch.equal(original, isolated),
                                 maxabs=float((original-isolated).abs().max()))
receipt = dict(
    status='Original target-score isolation complete; no credit repair or production change',
    launch=launch,
    sources=[binding(FOLDER/'transport.json'), binding(full_path), binding(native_path)],
    original_reference_and_all_target_input_IDs_unchanged=True,
    original_target_selection=traces[0]['target_selection'],
    controls=dict(cross_rank_signed_exact=torch.equal(traces[0]['signed'], traces[1]['signed']),
                  other_three_rows_exact_equal=torch.equal(traces[0]['signed'][:3], full['signed'][:3]),
                  other_three_rows_maxabs=float((traces[0]['signed'][:3]-full['signed'][:3]).abs().max()),
                  other_three_target_endpoint_controls=endpoint_controls,
                  selected_code_fence_endpoint_logps_exact=all(
                      full['detail'][key][other_target_count] == traces[0]['detail'][key][other_target_count]
                      for key in ('target_logp0', 'target_logp1')),
                  native_head_input_shapes_by_run=dict(complete=full['detail']['actual_head_input_shapes'],
                                                       isolated=traces[0]['detail']['actual_head_input_shapes']),
                  score_from_future_input_positions_maxabs=float(traces[0]['signed'][3, 2884:].abs().max()),
                  same_original_finite_rules=True),
    selected_newline=dict(packed_slot=2883, input_token_id=198,
                         complete_joint_target_DT_d=full_d,
                         next_code_fence_target_DT_d=partial_d,
                         complete_minus_next_score_DT_d=full_d-partial_d,
                         native_single_delete_next_code_fence_d=float(native_d[selected].sum()),
                         native_single_delete_remaining_targets_d=float(native_d[actual_targets & ~selected].sum()),
                         native_single_delete_complete_joint_target_d=float(native_d[actual_targets].sum()),
                         native_reference_scope='Previously measured independent uncached single-token EOS deletion; original complete joint target retained. It is not the all-source-EOS reference used by this DT diagnostic.'),
    next_code_fence_joint_endpoint=dict(
        factual_logp=details['factual_target_logp'],
        all_source_EOS_reference_logp=details['reference_target_logp'],
        factual_probability=float(torch.tensor(details['factual_target_logp'],dtype=torch.float64).exp()),
        all_source_EOS_reference_probability=float(torch.tensor(details['reference_target_logp'],dtype=torch.float64).exp()),
        root_effect=details['root_effect'], signed_sum=details['policy_credit_signed_sum'],
        conservation_residual=details['conservation_residual'],
        interpretation='The all-source joint endpoint and actual single-deletion endpoint are different contexts. Conservation of this root effect is not evidence of a correct individual-token probability ratio.'),
    measurements=dict(DT_calls_per_rank=1,
                      seconds_by_rank=[report['modes'][MODE]['seconds'] for report in reports],
                      physical_sampled_peak_mib={str(gpu):max(x['used_mib'] for x in physical if x['gpu']==gpu) for gpu in (4,5)},
                      checkpoint_restore=0, rollout=0, optimizer=0, backward=0),
    interpretation='The immediate code-fence score assigns the newline positive DT credit. Its negative complete-target score appears when later actual targets are included. Independently measured native single deletion gives positive net effects both for the code fence and for the remaining targets. The existing joint finite allocation therefore disagrees with this actual deletion at the later-target contribution level. This locates an interaction-allocation problem; it does not justify dropping later targets, changing Y, clipping, scaling, or replacing Q/V/A.',
    limitations=[
        'The full-minus-selected number is an algebraic comparison of two original DT runs, not an independently scored remaining-target run or an exact low-precision linearity proof.',
        'Unchanged-target rows have a measured maximum signed difference of 0.0144215 and endpoint-logp difference below 4.7e-7. Target selection changed the native head workload shape. This residual has not been localized to one operator or tested against an invented full-DT tolerance; it prevents claiming bitwise-isolated linearity. The selected code-fence endpoint logps themselves are exactly equal across these two runs.',
        'This is one already identified extreme token and real B4. It supplements the unchanged original author cumulative deletion/RISE/MAS; it does not replace those metrics or establish population accuracy.',
        'The original producer printed its usual report while its trace-boundary selection was isolated. Those diagnostic log Q/V/A and original target-count fields are not complete-event training quantities. Only signed score contributions and explicit score target counts are exported by this worker.',
        'The native individual-deletion endpoint is from a separate uncached run; native target-score differences and joint DT endpoint values have distinct recorded scopes.',
        'No numerical pass threshold was invented and no FA/FLA official accuracy pass is claimed.',
    ],
    official_tolerance_changed=False, training_target_changed=False,
    QVA_changed=False, credit_clipping_added=False, credit_repaired=False,
    formal_restart=False, TextCraft_update_released=False,
    local_cuda_initialized=torch.cuda.is_initialized(),
)
target = REPO/'experiments/rl/results_code_fence_target_isolation_20261008.json'
target.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
(HERE/'code-fence-analysis.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(receipt=binding(target), controls=receipt['controls'],
                      token=receipt['selected_newline'], endpoint=receipt['next_code_fence_joint_endpoint'],
                      measurements=receipt['measurements']),ensure_ascii=False,indent=2))

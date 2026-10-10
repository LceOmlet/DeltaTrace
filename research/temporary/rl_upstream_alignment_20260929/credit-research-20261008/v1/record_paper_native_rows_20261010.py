"""Bind original native observations without inventing a numerical threshold."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
OUT = HERE/'paper-native-rows-20261010-v2'


def artifact(path):
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size)


def main():
    result = json.loads((OUT/'result.json').read_bytes())
    launch = json.loads((OUT/'launch.json').read_bytes())
    completed = json.loads((OUT/'completed.json').read_bytes())
    assert result['phase'] == completed['phase'] == 'complete'
    assert result['native_forward_calls'] == completed['native_forward_calls'] == 1
    assert result['pid'] == launch['pid'] == completed['pid']
    assert result['birth'] == launch['birth'] == completed['birth']
    assert all(v == 0 for v in result['operations'].values())
    observation = result['operator_observation']
    first = []
    for row,v in observation['first_unequal_by_row'].items():
        event = observation['events'][v['event']]
        first.append(dict(pair=int(row),**v,operation_event=event))
    snapshots = sorted(OUT.glob('observation-*.json'))
    terminal = json.loads(snapshots[-1].read_bytes())
    assert not terminal['alive'],'Check the actual process handle after the saved completion'
    failed = HERE/'paper-native-rows-20261010-v1'
    previous = json.loads((failed/'result.json').read_bytes())
    assert previous['phase']=='failed' and previous['native_forward_calls']==0
    assert "'NoneType' object has no attribute 'transpose'" in previous['exception']['message']
    owner = HERE/'inspect_native_identity_operators.py'
    patched = OUT/'inspect_native_identity_operators.py'
    expected = owner.read_text().replace(
        "v.transpose(1,2) if k in ('x','initial_states') else v",
        "v.transpose(1,2) if k in ('x','initial_states') and isinstance(v,torch.Tensor) else v")
    assert patched.read_text() == expected
    receipt = dict(status='Completed single native forward localization of paper row differences',
        scope=result['scope'],source_commit_numerical='26bef6c8b2e49db118f46e3c05e86944dcf8e293',
        numerical_version='fla-early-output-scale-20261009-v1',
        inputs=[artifact(OUT/name) for name in ('launch.json','result.json','completed.json','rank0-batch0-operators.json')],
        terminal_observation=artifact(snapshots[-1]),actual_owners=result['owners'],
        first_unequal=first,scores=result['scores'],previous_scores=result['previous_scores'],
        maximum_score_difference_vs_previous=result['maximum_score_difference_vs_previous'],
        paired_score_differences=result['paired_score_differences'],
        saved_operand_artifacts=observation['artifacts'],operator_owners=observation['owners'],
        resource=result['resources'],diagnostic_seconds=result['seconds'],
        operations=result['operations'],native_forward_calls=1,
        observer_optional_state_fix=dict(original=artifact(owner),patched=artifact(patched),
            scope='Observation only: preserve optional native initial_states=None instead of transposing it. Native call and returned results unchanged.'),
        failed_attempt=dict(launch=artifact(failed/'launch.json'),result=artifact(failed/'result.json'),
            completed_native_forwards=0,accepted=False),
        tolerance_status='Not inferred from model-level score differences; apply the corresponding owner to captured original operands.',
        claims_excluded=['whole-model tolerance','DT accuracy certificate','attribution repair','population negative-tail recall','production change'])
    check = OUT/'linear-roundoff.json'
    if check.exists():
        checked = json.loads(check.read_bytes())
        checks = [v for c in checked['cases'] for v in c['native_pair_checks']]
        checks += [v for c in checked['cases']
                   for e in c['all_unequal_cells_vs_FP64_rounded_BF16_checks'] for v in e['endpoints']]
        receipt['linear_owner_check'] = dict(receipt=artifact(check),result=checked,
            unchanged_owner_assertions=len(checks),statuses=[v['status'] for v in checks],
            scope='Original PyTorch assert_close BF16 defaults, eight full paired outputs and six unequal-coordinate endpoint comparisons. Not FA/FLA or whole-model tolerances.')
    path = REPO/'experiments/rl/results_paper_native_rows_20261010.json'
    path.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(path=str(path),sha256=artifact(path)['sha256'],
        first_unequal=[{k:v for k,v in item.items() if k!='operation_event'} for item in first],
        maximum_score_difference_vs_previous=receipt['maximum_score_difference_vs_previous'],
        native_forward_calls=1,seconds=receipt['diagnostic_seconds'])))


if __name__ == '__main__':main()

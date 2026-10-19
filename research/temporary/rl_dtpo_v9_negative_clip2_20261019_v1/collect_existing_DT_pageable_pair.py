"""Collect the bounded comparison through the existing metadata collector."""
from pathlib import Path

owner = Path(__file__).with_name('collect_existing_DT_timings.py')
code = owner.read_text()
code = code.replace('native_DT_existing_timing', 'native_DT_existing_pageable_pair')
code = code.replace('existing-DT-native-stage-timings.json', 'existing-DT-pageable-pair-results.json')
code = code.replace('results_dtpo_v9_negative2_existing_DT_timings_20261019_v4.json',
                    'results_dtpo_v9_negative2_pageable_pair_20261019_v4.json')
code = code.replace("row['value']['additional_DT_calls'] == 0", "row['value']['additional_DT_calls'] == 1")
code = code.replace('additional_calls=dict(model=0,DT=0,optimizer=0,environment=0,profiling=0)',
                    "additional_calls=dict(model='native forward/replay inside the validation DT calls',DT=sum(bool(row['value'].get('optional_same_input_pageable_pair')) for row in rows),optimizer=0,environment=0,profiling=0)")
anchor = 'if rows or scopes or errors:'
assert code.count(anchor) == 1
code = code.replace(anchor, '''report['candidate_deployed']=False
report['numerical_acceptance_claimed']=False
report['comparison_status']='comparison_pending'
report['same_input_pairs']=[dict(path=row['path'],line=row['line'],rank=row['value']['rank'],
    minibatch=row['value']['minibatch'],paired_shape=row['value']['paired_shape'],
    baseline_wall_seconds=row['value']['finished']-row['value']['started'],
    pair=row['value']['optional_same_input_pageable_pair'])
    for row in rows if row['value'].get('optional_same_input_pageable_pair')]
report['exclude_from_production_efficiency_comparison']=[dict(rank=row['value']['rank'],
    start=row['value']['started'],end=row['value']['finished'],
    reason='Readout includes one additional validation-only DT call; training receives original result.') for row in scopes]
if complete and len(report['same_input_pairs'])==8:
    successful=all(row['pair']['status']=='same_input_owner_pair_completed_not_deployed'
        for row in report['same_input_pairs'])
    report['comparison_status']=('same_input_pairs_completed_production_unchanged' if successful
        else 'optional_comparison_failed_original_training_results_retained')
report['pair_failures']=[dict(rank=row['rank'],status=row['pair']['status'],
    error=row['pair'].get('error')) for row in report['same_input_pairs']
    if row['pair']['status']!='same_input_owner_pair_completed_not_deployed']
report['successful_same_input_pair_count']=sum(
    row['pair']['status']=='same_input_owner_pair_completed_not_deployed'
    for row in report['same_input_pairs'])
report['interpretation'].append('This readout and its whole iteration contain validation-only extra work. Do not count higher Util or its timing as a production speedup; use separately timed paired calls only as preliminary cost evidence.')
''' + anchor)
anchor = 'print(json.dumps(brief, ensure_ascii=False))'
assert code.count(anchor) == 1
code = code.replace(anchor, '''brief['pairs']=[dict(rank=row['rank'],shape=row['paired_shape'],baseline=row['baseline_wall_seconds'],
    candidate=row['pair'].get('candidate_wall_seconds'),equal=row['pair'].get('raw_DT_vector_equal'),
    max_difference=row['pair'].get('raw_DT_max_abs_difference'),status=row['pair']['status'],
    flags_restored=row['pair'].get('original_options_restored')) for row in report['same_input_pairs']]
''' + anchor)
exec(compile(code, str(owner), 'exec'), globals())

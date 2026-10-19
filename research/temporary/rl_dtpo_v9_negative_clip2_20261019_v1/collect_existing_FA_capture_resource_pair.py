"""Collect the A-B-A resource comparison through the original metadata reader."""
from pathlib import Path

owner = Path(__file__).with_name('collect_existing_DT_pageable_pair.py')
code = owner.read_text()
for old, new in (
    ('native_DT_existing_pageable_pair', 'native_DT_FA_capture_resource_pair'),
    ('existing-DT-pageable-pair-results.json', 'existing-DT-FA-capture-resource-pair-results.json'),
    ('results_dtpo_v9_negative2_pageable_pair_20261019_v4.json', 'results_dtpo_v9_negative2_FA_capture_pair_20261019_v4.json'),
    ('optional_same_input_pageable_pair', 'optional_same_input_FA_capture_pair'),
    ("row['value']['additional_DT_calls'] == 1", "row['value']['additional_DT_calls'] in (0,1,2)"),
    ("DT=sum(bool(row['value'].get('optional_same_input_FA_capture_pair')) for row in rows)",
     "DT=sum((row['value'].get('optional_same_input_FA_capture_pair') or {}).get('additional_DT_calls',0) for row in rows)"),
    ('Readout includes one additional validation-only DT call; training receives original result.',
     'Readout includes validation-only control collectives and up to two additional DT calls; training receives original result.'),
):
    assert old in code, old
    code = code.replace(old, new)
anchor = "if complete and len(report['same_input_pairs'])==8:"
assert code.count(anchor) == 1
code = code.replace(anchor, '''if complete and not report['same_input_pairs']:
    report['comparison_status']='not_executed_no_common_memory_eligible_long_B4'
report['training_iterations_excluded_from_production_efficiency']=sorted({r['value']['training_iteration'] for r in scopes})
report['validation_control_collectives_by_rank']={str(r['value']['rank']):r['value']['validation_guard_collectives'] for r in scopes}
report['memory_guard_errors']=[dict(path=r['path'],line=r['line'],rank=r['value']['rank'],guard=r['value']['FA_capture_memory_guard']) for r in rows if r['value']['FA_capture_memory_guard'].get('error')]
''' + anchor)
anchor = "brief['pairs']=[dict(rank=row['rank'],shape=row['paired_shape'],baseline=row['baseline_wall_seconds'],"
assert code.count(anchor) == 1
code = code.replace(anchor, "brief['pairs']=[dict(rank=row['rank'],shape=row['paired_shape'],baseline=row['baseline_wall_seconds'],\n    training_iteration=row['pair']['training_iteration'],warm_baseline=row['pair'].get('warm_baseline_wall_seconds'),warm_equal=row['pair'].get('warm_baseline_vector_equal'),")
exec(compile(code, str(owner), 'exec'), globals())

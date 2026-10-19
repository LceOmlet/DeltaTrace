"""Read the small GPU-resident pair using the original comparison collector."""
from pathlib import Path

owner = Path(__file__).with_name('collect_existing_DT_pageable_pair.py')
code = owner.read_text()
for old, new in (
    ('native_DT_existing_pageable_pair', 'native_DT_existing_gpu_resident_pair'),
    ('existing-DT-pageable-pair-results.json', 'existing-DT-gpu-resident-pair-results.json'),
    ('results_dtpo_v9_negative2_pageable_pair_20261019_v4.json',
     'results_dtpo_v9_negative2_gpu_resident_pair_20261019_v4.json'),
    ('optional_same_input_pageable_pair', 'optional_same_input_gpu_resident_pair'),
    ("row['value']['additional_DT_calls'] == 1", "row['value']['additional_DT_calls'] in (0,1)"),
):
    assert old in code
    code = code.replace(old, new)
anchor = "if complete and len(report['same_input_pairs'])==8:"
assert code.count(anchor) == 1
code = code.replace(anchor, "if complete and not report['same_input_pairs']:\n    report['comparison_status']='not_executed_no_common_eligible_small_B4'\n" + anchor)
exec(compile(code, str(owner), 'exec'), globals())

"""Select the post-change observations with the existing native collector."""
from pathlib import Path

owner = Path(__file__).with_name('collect_existing_DT_host_counters.py')
code = owner.read_text()
for old, new in (
    ('native_DT_existing_host_counters', 'native_DT_root_pageable_host_counters'),
    ('existing-DT-native-host-counters.json', 'root-pageable-DT-native-host-counters.json'),
    ('results_dtpo_v9_negative2_existing_DT_host_counters_20261019_v4.json',
     'results_dtpo_v9_negative2_root_pageable_DT_host_counters_20261019_v4.json'),
):
    assert old in code
    code = code.replace(old, new)
exec(compile(code, str(owner), 'exec'), globals())

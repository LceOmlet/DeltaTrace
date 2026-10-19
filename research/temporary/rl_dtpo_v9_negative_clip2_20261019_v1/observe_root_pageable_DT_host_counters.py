"""Reuse the single natural-readout observer after the accepted storage change."""
import hashlib
from pathlib import Path

owner = Path(__file__).with_name('observe_existing_DT_host_counters.py')
code = owner.read_text()
for old, new in (
    ('_existing_DT_host_counters_scope_installed', '_root_pageable_DT_host_counters_scope_installed'),
    ('existing-DT-host-counters', 'root-pageable-DT-host-counters'),
    ('native_DT_existing_host_counters', 'native_DT_root_pageable_host_counters'),
):
    assert old in code
    code = code.replace(old, new)
# The owner constructs and installs the same passive wrapper. Only its receipt
# identity and the now-required deployed resource flags differ.
anchor = "exec(compile(code, str(owner), 'exec'), globals())"
assert code.count(anchor) == 1
guard = "        import json,time\n        assert (runner.pin_root_host,runner.pin_replay_host)==(False,True)"
code = code.replace(anchor,
    "assert code.count('        import json,time') == 1\n"
    "code=code.replace('        import json,time'," + repr(guard) + ")\n" + anchor)
exec(compile(code, str(owner), 'exec'), globals())

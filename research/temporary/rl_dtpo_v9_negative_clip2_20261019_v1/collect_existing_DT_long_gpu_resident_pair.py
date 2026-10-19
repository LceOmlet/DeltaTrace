"""Reuse the original metadata collector for the native long GPU-storage pair."""
import hashlib
from pathlib import Path

owner = Path(__file__).with_name('collect_existing_FA_capture_resource_pair.py')
code = owner.read_text()
for old, new in (
    ('native_DT_FA_capture_resource_pair', 'native_DT_long_GPU_resident_resource_pair'),
    ('existing-DT-FA-capture-resource-pair-results.json', 'existing-DT-long-GPU-resident-resource-pair-results.json'),
    ('results_dtpo_v9_negative2_FA_capture_pair_20261019_v4.json',
     'results_dtpo_v9_negative2_long_GPU_resident_pair_20261019_v4.json'),
    ('optional_same_input_FA_capture_pair', 'optional_same_input_long_GPU_resident_pair'),
    ('FA_capture_memory_guard', 'long_GPU_resident_memory_guard'),
):
    assert old in code, old
    code = code.replace(old, new)
exec(compile(code, str(owner), 'exec'), globals())

"""Read the measured-source A-B-A comparison through the original collector."""
import hashlib
from pathlib import Path

owner = Path(__file__).with_name('collect_existing_DT_long_gpu_resident_pair.py')
assert hashlib.sha256(owner.read_bytes()).hexdigest() == '07105b9364da1427cf763964f414a6d8676b1cdcc5dcd6005fd99cb040286a7a'
code = owner.read_text()
for old, new in (
    ('native_DT_long_GPU_resident_resource_pair', 'native_DT_measured_source_resident_pair'),
    ('existing-DT-long-GPU-resident-resource-pair-results.json', 'existing-DT-measured-source-resident-pair-results.json'),
    ('results_dtpo_v9_negative2_long_GPU_resident_pair_20261019_v4.json',
     'results_dtpo_v9_negative2_measured_source_resident_pair_20261019_v4.json'),
    ('optional_same_input_long_GPU_resident_pair', 'optional_same_input_measured_source_resident_pair'),
    ('long_GPU_resident_memory_guard', 'measured_source_memory_guard'),
):
    assert old in code, old
    code = code.replace(old, new)
exec(compile(code, str(owner), 'exec'), globals())

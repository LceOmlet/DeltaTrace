"""Select the saved native storage A-B-A calls in the original Util reducer."""
from pathlib import Path

owner = Path(__file__).with_name('summarize_FA_capture_pair_utilization.py')
code = owner.read_text()
for old, new in (
    ('existing-DT-FA-capture-resource-pair-results.json', 'existing-DT-measured-source-resident-pair-v2-results.json'),
    ('optional_same_input_FA_capture_pair', 'optional_same_input_measured_source_resident_pair_v2'),
    ('FA-capture-pair-', 'measured-source-resident-v2-pair-'),
):
    assert old in code, old
    code = code.replace(old, new)
exec(compile(code, str(owner), 'exec'), globals())

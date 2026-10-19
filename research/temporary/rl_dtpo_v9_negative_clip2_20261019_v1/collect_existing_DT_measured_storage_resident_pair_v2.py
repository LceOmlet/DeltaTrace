"""Read the native-limit comparison through the unchanged original collector."""
import hashlib
from pathlib import Path

owner = Path(__file__).with_name('collect_existing_DT_measured_storage_resident_pair.py')
assert hashlib.sha256(owner.read_bytes()).hexdigest() == '0995f3842f22ab90bfed827af79aa89c13f8ec1ef8ddae0959825ae461133be2'
code = owner.read_text()
anchor = "exec(compile(code, str(owner), 'exec'), globals())"
assert code.count(anchor) == 1
code = code.replace(anchor,
    "code = code.replace('native_DT_measured_source_resident_pair', 'native_DT_measured_source_resident_pair_v2')\n"
    "code = code.replace('optional_same_input_measured_source_resident_pair', 'optional_same_input_measured_source_resident_pair_v2')\n"
    "code = code.replace('existing-DT-measured-source-resident-pair-results.json', 'existing-DT-measured-source-resident-pair-v2-results.json')\n"
    "code = code.replace('results_dtpo_v9_negative2_measured_source_resident_pair_20261019_v4.json', 'results_dtpo_v9_negative2_measured_source_resident_pair_v2_20261019_v4.json')\n"
    + anchor)
exec(compile(code, str(owner), 'exec'), globals())

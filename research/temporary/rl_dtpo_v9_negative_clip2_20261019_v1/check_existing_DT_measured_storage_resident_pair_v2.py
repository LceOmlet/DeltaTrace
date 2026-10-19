"""Reuse the measured-source owner comparison within its native 32768 limit.

Only remove the earlier validation-only 12288 cutoff. The measured-source,
workspace, two-native-KV and 2 GiB physical-free guards remain unchanged.
Private/default allocator caches receive no speculative capacity credit.
"""
import hashlib
from pathlib import Path

owner = Path(__file__).with_name('check_existing_DT_measured_storage_resident_pair.py')
assert hashlib.sha256(owner.read_bytes()).hexdigest() == '7c91fda489d94b7f3df997f0750aaba32bddaf7240fc21cbea2ed5c66f074ac3'
code = owner.read_text()
code = code.replace('measured_source_resident_pair', 'measured_source_resident_pair_v2')
code = code.replace('measured-source-resident-pair-v1', 'measured-source-resident-pair-v2')
anchor = "wrapper = HERE / 'inspect_native_workers.py'"
assert code.count(anchor) == 1
code = code.replace(anchor, "assert observation.count('2048<paired.shape[1]<=12288') == 1\n"
    "observation = observation.replace('2048<paired.shape[1]<=12288', '2048<paired.shape[1]<=32768')\n"
    + anchor)
exec(compile(code, str(owner), 'exec'), globals())

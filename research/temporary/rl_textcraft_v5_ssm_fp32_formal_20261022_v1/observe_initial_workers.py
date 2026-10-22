"""Compose the original once-per-phase read-only worker observer for DP8."""
from prepare_campaign import ROOT, AUDIT, VERSION, sha

owner = ROOT / 'receipts/textcraft-v5-two-gpu-reproduction-20261021-v1/source/observe_initial_workers.py'
assert sha(owner) == '4895c206919ac21371f8a38b00b07f85e1c3848c08e6c11a3e320145e49d135e'
code = owner.read_text()
for before, after in {
    "'waiting_original_two_gpu_workers'": "'waiting_original_eight_gpu_workers'",
    'len(workers)==2': 'len(workers)==8',
    "str(script),'dt','--passive'": "str(script),'dt'",
    "'original_initial_metadata_and_passive_recorder_requested'": "'original_initial_read_only_metadata_requested'",
    "'original_initial_metadata_and_passive_recorder_completed'": "'original_initial_read_only_metadata_completed'",
}.items():
    assert code.count(before) == 1, before
    code = code.replace(before, after)
exec(compile(code, str(owner), 'exec'), globals())

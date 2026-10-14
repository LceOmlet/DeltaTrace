"""Add actual full-PG readouts to the prepared upper-parameter interventions.

Same two parameter interventions and original native owners. Reuse the actual
saved iteration81 full-PG carrier as well as the unchanged fixed-seed carrier.
No new credit/DT, input construction, training checkpoint or model call.
"""
import copy
import hashlib
import json
import time
from pathlib import Path

from verl import DataProto

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT = ROOT / 'receipts/textcraft-gradient-rootcause-20261013-v1'
DEST = OUT / 'native-upper-adapter-partition-v81'
original_path = DEST / 'case-manifest.json'
source = json.loads(original_path.read_bytes())
assert source['status'] == 'prepared_only_no_GPU_submission'
assert not (DEST / 'job.json').exists() and not (DEST / 'result.json').exists()
assert len(source['cases']) == 2
driver_path = OUT / 'native-PG-driver-components-v38/case-manifest.json'
driver = json.loads(driver_path.read_bytes())
full_case = driver['cases'][0]
path = Path(full_case['path'])
assert hashlib.sha256(path.read_bytes()).hexdigest() == full_case['SHA256']
full = DataProto.load_from_disk(str(path))
assert len(full) == 64
assert (full.batch['diagnostic_full_DT'] == full.batch['advantages']).all().item()

report = copy.deepcopy(source)
report.update(unix=time.time(), diagnostic_revision='full-policy-readout-v83',
    original_target_only_preparation=dict(path=str(original_path),
        SHA256=hashlib.sha256(original_path.read_bytes()).hexdigest(), GPU_submitted=False),
    actual_original81_full_PG_carrier=dict(path=str(path), SHA256=full_case['SHA256'],
        original_native_decomposition_manifest=str(driver_path),
        original_native_decomposition_manifest_SHA256=hashlib.sha256(driver_path.read_bytes()).hexdigest(),
        advantages_equal_original_saved_whitened_DT=True, rows=len(full)),
    scope=__doc__, planned_native_fixed_seed_derivatives=2, planned_native_full_policy_derivatives=2,
    parameter_interventions=2, total_native_derivative_readouts=4,
    counterevidence=dict(
        same_saved81_input_full_policy_norm40=2.0620788290896104,
        same_saved81_input_full_policy_norm80=2.214119191862583,
        full_policy_norm80_over40=1.073731595818816,
        same_seed_conditional_parameter_VJP80_over40=12.7854,
        meaning='Large conditional internal VJP growth cannot alone establish the actual full policy-gradient time growth. Keep the original full-PG direction/cancellation as a separate readout under each same parameter intervention.'),
    decision='Read fixed-seed internal transfer AND the actual original81 full policy gradient under each of the same two upper-state swaps. A change in only the probe is localization, not a repair of actual PG growth. No owner or credit change is accepted from this contrast alone.')
for case in source['cases']:
    added = copy.deepcopy(case)
    added.update(name=case['name'] + '_full_original81_PG', path=str(path),
                 SHA256=full_case['SHA256'], diagnostic_advantage_views={'full_DT': 'diagnostic_full_DT'})
    for key in ('boundary_seed_source_case', 'boundary_seed_source_directory',
                'boundary_seed_SHA256_by_rank_micro', 'capture_boundary_seed'):
        added.pop(key, None)
    report['cases'].append(added)
report['limits'].extend([
    'The full-PG readouts reuse the actual complete first global64 of original iteration81, including source/target/observation mask and original full256 whitening. They are not the15 matched-target carrier or a GRPO algorithm.',
    'Original owner compute_log_prob supplies each intervened model old-policy point (rho1). This is a read-only full policy-gradient comparison, not its four later clipped optimizer steps or a validation-score test.',
    'The original two-readout preparation is preserved. This v83 revision adds the full-PG observable required by the completed counterevidence; it does not change the parameter intervention, formal method or native mathematical implementation.'
])
dest = DEST / 'case-manifest.full-policy-v83.json'
assert not dest.exists()
dest.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(dict(receipt=str(dest), report=report)))

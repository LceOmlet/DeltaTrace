"""Prepare one same-session whole-adjoint comparison from saved native states.

The completed v99 measures actual natural PG after saved native updates. The
older v90 fixed-adjoint result has an unresolved cross-run forward-point gap.
Capture the actual complete baseline adjoint once in a new run, then replay
that same saved tensor at the baseline and four already saved after-states.
No new optimizer, DT, sampling, quality test, or formal training is prepared.
"""
import copy
import hashlib
import json
from pathlib import Path
import time

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT = ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1'
DEST = OUT/'native-same-session-saved-update-feedback-v113'
source_path = OUT/'native-saved-update-natural-PG-v99/case-manifest.json'
original = json.loads(source_path.read_bytes())
assert not (DEST/'case-manifest.json').exists() and not (DEST/'job.json').exists()
DEST.mkdir(exist_ok=True)
cases = []
for source in original['cases']:
    case = copy.deepcopy(source)
    if case['name'] == 'old80_before_repeat':
        case['name'] = 'old80_same_session_fixed_seed'
    for key in ('capture_boundary_seed', 'read_boundary_seed_reference_case',
                'boundary_seed_source_case', 'boundary_seed_source_directory',
                'boundary_seed_SHA256_by_rank_micro', 'boundary_seed_hash_receipt'):
        case.pop(key, None)
    if case['name'] == 'old80_before':
        case['capture_boundary_seed'] = True
    else:
        case.update(capture_boundary_seed=False, boundary_seed_source_case='old80_before',
                    boundary_seed_source_directory=str(DEST))
    assert hashlib.sha256(Path(case['path']).read_bytes()).hexdigest() == case['SHA256']
    for artifact in case['diagnostic_parameter_partition']['rank_artifacts'].values():
        assert hashlib.sha256(Path(artifact['path']).read_bytes()).hexdigest() == artifact['SHA256']
    cases.append(case)
assert len(cases) == 6 and cases[0]['name'] == 'old80_before'
assert sum(bool(c.get('boundary_seed_source_case')) for c in cases) == 5
report = dict(unix=time.time(), status='prepared_only_no_GPU_submission', scope=__doc__,
    destination=str(DEST), checkpoint=80, checkpoint_path=original['checkpoint_path'], cases=cases,
    diagnostic_advantage_views={'full_DT':'diagnostic_full_DT'},
    passive_boundary_readout_enabled=True, native_ratio_chain_readout=True,
    decoder_branch_layers=[], save_native_probability_carrier=False,
    original_manifest=dict(path=str(source_path), SHA256=hashlib.sha256(source_path.read_bytes()).hexdigest()),
    question='After actual saved native original, same-reward-ablated and approved-v3 updates, does complete-direction internal propagation differ when the complete baseline seed and forward-point metadata are recorded in the same run?',
    why_needed='v99 actual natural PG declines after original updates. v90 fixed-vector comparisons cannot close the mechanism because its original80 rank1 forward point differs materially from the older v86 reference. Static saved-state/owner checks do not repair that numerical evidence gap.',
    discriminating_quantity='One actual full-PG baseline capture, its same-session identical-seed replay, and four replay passes at already saved native after-states. Compare whole global VJP, full-layer adjoints, actual B4 input hashes and native buffer hashes.',
    decision='Use within-run baseline capture/replay and complete metadata to assess whether the extra common update changes internal gain relative to the saved native ablation. If it does not, withdraw that driver claim; neither a relative effect nor a lower candidate norm alone accepts the temporal root repair.',
    counterfactual_scope='Same-session replay of the actual complete baseline adjoint at four saved native after-states; no new optimizer outcomes, trajectory sampling, or from-scratch training comparison.',
    diagnostic_revision='same-session-complete-adjoint-saved-native-updates-v113',
    seed_identity_scope='The unchanged existing observer captures16 BF16 tensors once, never overwrites them, and replays those same run-owned files. Preserve actual source paths/shape/dtype and hash the16 retained files in the completion receipt; do not invent an external bit-exact replay gate.',
    existing_observer='observe_saved_update_natural_PG_v99.py',
    existing_observer_SHA256='ee24d145579ea3cdb14892c78da2dea65761f3253da1ed29152f0fd473605290',
    planned_actual_native_full_PG_derivatives=1, planned_same_session_fixed_seed_derivatives=5,
    planned_native_full_policy_derivatives=1, planned_native_fixed_seed_derivatives=5,
    optimizer_updates=0, scheduler_updates=0, DT_calls=0, rollout_calls=0,
    validation_score_calls=0, GRPO_training_reruns=0, formal_training_changes=0,
    resource_scope='Only the previously authorized temporary PPO pair may be used after a completed native checkpoint preserves current progress; restore its latest native full state immediately after any terminal outcome. No formal DTPO/GRPO/other-project GPU is available by assumption.',
    preparation_changes_no_training_or_processes=True, rootcause_repair_verified=False,
    limits=['The four after-states and native inputs already exist; there are no new optimizer outcomes or trajectory-quality tests.',
            'Five replay cases are conditional VJPs, not actual post-update natural policy gradients. The completed v99 natural gradients remain a separate result.',
            'A relative internal-gain difference on one frozen late batch is not a reconstruction of all85 historical iterations or a from-scratch result.',
            'Forward/RNG/source equivalence and observed numerical variation are recorded; no arbitrary zero-error kernel or all-layer-decline condition is imposed.',
            'The official actor/core/1/16/B4/32k/LoRA8/16/optimizer/credit methods are unchanged. Prepared-only diagnostics are not deployed training repairs.'])
path = DEST/'case-manifest.json'
path.write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(dict(receipt=str(path), report=report)))

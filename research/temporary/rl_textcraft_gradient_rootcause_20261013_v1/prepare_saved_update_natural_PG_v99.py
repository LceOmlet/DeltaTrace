"""Bind saved native update outcomes to actual complete native PG readouts.

CPU preparation only. The completed fixed-adjoint experiment omitted the
post-update change in the head adjoint. Reuse the same native inputs, original
advantages, original Torch partial-state artifacts and official PPO owner;
execute no new optimizer/DT/rollout/validation call and no GRPO rerun.
"""
import copy
import hashlib
import json
from pathlib import Path
import time

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT = ROOT / 'receipts/textcraft-gradient-rootcause-20261013-v1'
DEST = OUT / 'native-saved-update-natural-PG-v99'
old_path = OUT / 'native-full-policy-saved-update-feedback-v90/case-manifest.json'
old = json.loads(old_path.read_bytes())
assert not (DEST / 'case-manifest.json').exists() and not (DEST / 'job.json').exists()
DEST.mkdir(exist_ok=True)
cases = []
for case in old['cases']:
    if case['name'] == 'GRPO160_same_full80_seed':
        continue
    new = copy.deepcopy(case)
    for name in ('boundary_seed_source_case', 'boundary_seed_source_directory',
                 'boundary_seed_SHA256_by_rank_micro', 'boundary_seed_hash_receipt'):
        new.pop(name, None)
    assert hashlib.sha256(Path(new['path']).read_bytes()).hexdigest() == new['SHA256']
    for artifact in new['diagnostic_parameter_partition']['rank_artifacts'].values():
        assert hashlib.sha256(Path(artifact['path']).read_bytes()).hexdigest() == artifact['SHA256']
    new['capture_boundary_seed'] = new['name'] == 'old80_before'
    if not new['capture_boundary_seed']:
        new['read_boundary_seed_reference_case'] = 'old80_before'
    cases.append(new)
assert len(cases) == 5 and cases[0]['name'] == 'old80_before'
repeat = copy.deepcopy(cases[0])
repeat.update(name='old80_before_repeat', capture_boundary_seed=False,
              read_boundary_seed_reference_case='old80_before')
cases.insert(1, repeat)
report = dict(unix=time.time(), status='prepared_only_no_GPU_submission', scope=__doc__,
    destination=str(DEST), checkpoint=80, checkpoint_path=old['checkpoint_path'], cases=cases,
    diagnostic_advantage_views={'full_DT': 'diagnostic_full_DT'},
    passive_boundary_readout_enabled=True, native_ratio_chain_readout=True,
    decoder_branch_layers=[], save_native_probability_carrier=False,
    question='On identical original81 inputs and frozen complete original advantages, how does the actual native rho1 PG change at the already saved native original, same-reward-ablated and approved-v3 after-parameters, allowing its head adjoint to change naturally?',
    why_needed='v90 holds the complete head adjoint fixed. Its original-update VJP decline cannot by itself determine the change of the actual PG, whose head adjoint also changes with parameters. The saved after-parameters make this direct missing readout possible without new optimizer, DT, generation or validation.',
    discriminating_quantity='Six complete official native PG passes: original80, same original80 repetition, and four already saved native after-states. Observe actual incoming BF16 head adjoint norms/dots against the same-session original capture, complete global gradient and B4 increment Gram. No boundary adjoint is replaced.',
    decision='If actual original PG also declines, reject a local positive-feedback claim for this complete frozen-credit minibatch. If it rises while fixed-direction propagation declined, the changing head adjoint is a necessary part of that finite increase; quantify it before proposing a causal repair. A lower candidate norm alone never accepts the historical repair.',
    counterfactual_scope='Retrospective fixed-data/fixed-credit original rho1 gradient at saved parameter states. Original old/ref log-probabilities are recomputed at each point. This is not the historical next rollout, a new training run, a validation-quality test or a from-scratch outcome.',
    diagnostic_revision='actual-native-PG-after-saved-updates-v99',
    planned_native_full_policy_derivatives=6, planned_native_fixed_seed_derivatives=0,
    original_manifest=dict(path=str(old_path), SHA256=hashlib.sha256(old_path.read_bytes()).hexdigest()),
    optimizer_updates=0, scheduler_updates=0, DT_calls=0, rollout_calls=0,
    validation_score_calls=0, GRPO_training_reruns=0, formal_training_changes=0,
    rootcause_repair_verified=False,
    limits=['All six cases use original global64, full256-original-whitened coefficients, pinned native 1/16/B4/32k/LoRA8/16. Nothing is renormalized or re-credited.',
        'The four after-states are already saved exact outcomes of original native four-update experiments v47/v69, not new optimized states.',
        'A same-original-point repetition reports observed native variation, without an invented tolerance or universal exact numerical replay gate.',
        'The boundary hook observes the native adjoint and returns None. Its norm/dot summaries are not an operator norm, model implementation or independent causal share.',
        'Parameters/optimizer/scheduler remain unchanged within each readout. Diagnostic model-only artifacts are not training checkpoints or formal backup selections.',
        'No conclusion from this finite frozen-credit minibatch alone accepts the complete85-iteration root-cause repair.'])
path = DEST / 'case-manifest.json'
path.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(dict(receipt=str(path), report=report)))

"""Prepare the full approved-v3 bad-state regression; no GPU submission."""
import hashlib
import json
from pathlib import Path
import time

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT = ROOT / 'receipts/textcraft-gradient-rootcause-20261013-v1'
DEST = OUT / 'native-full-current-v3-feedback-v69'
old = json.loads((OUT / 'native-common-update-feedback-v47/case-manifest.json').read_bytes())
prepared = json.loads((OUT / 'saved-bad-state-v3-DT-contract-v68/preparation.json').read_bytes())
current_path = ROOT / 'runs/textcraft-dt-grpo-multiplier-variance-only-20261014-v3/launch.json'
current = json.loads(current_path.read_bytes())
assert prepared['required_artifacts_available'] and prepared['all_original_retained_token_identity_checks_passed']
assert prepared['captured_completed_step'] == 80 and prepared['rows'] == 256
assert prepared['model_calls'] == prepared['DT_worker_calls'] == prepared['optimizer_calls'] == 0
assert prepared['native_zero_group_rows'] == prepared['positive_reward_zero_group_rows'] == 128
path = Path(prepared['payload']['path'])
assert hashlib.sha256(path.read_bytes()).hexdigest() == prepared['payload']['SHA256']
report = dict(old)
report.update(unix=time.time(), destination=str(DEST),
    scope='Full approved DTPO v3 at original checkpoint80 and actual original iteration81 input. Original native PPO/AdamW/scheduler and captured RNG; fixed-seed gradient/hidden-propagation regression, not quality scoring.',
    status='prepared_only_no_GPU_submission', current_v3_launch=str(current_path),
    worker_effective_environment=current['effective_environment'],
    v3_token_ids=prepared['token_ids'], prepared_native_request_rows=prepared['prepared_native_request_rows'],
    native_DT_preparation_receipt=str(OUT / 'saved-bad-state-v3-DT-contract-v68/preparation.json'),
    native_DT_preparation_SHA256=hashlib.sha256((OUT / 'saved-bad-state-v3-DT-contract-v68/preparation.json').read_bytes()).hexdigest(),
    source_bindings={p: digest for p, digest in current['source_bindings'].items() if p.endswith((
        '/verl/workers/actor/dp_actor.py', '/verl/trainer/ppo/core_algos.py', '/verl/workers/fsdp_workers.py',
        '/verl/utils/checkpoint/fsdp_checkpoint_manager.py', '/verl/trainer/ppo/ray_trainer.py',
        '/verl/utils/torch_functional.py', '/entry/dt_training_batch.py'))},
    arms=[old['arms'][0], dict(name='approved_v3_full_method', path=str(path),
        SHA256=prepared['payload']['SHA256'], rows=256, prepare_current_v3=True)],
    cases=[dict(case, name=case['name'].replace('without_constant_positive_component', 'approved_v3_full_method'))
           for case in old['cases']],
    new_generation_calls=0, GRPO_model_reruns=0, validation_score_calls=0,
    why_needed='The prior late-state intervention removed a subset of old advantages; it did not compute the full approved v3, which needs DT for35 zero-reward/nonzero-group rows. The original full token/mask/target artifacts survive, so no new trajectory or decoded-text reconstruction is needed.',
    question='At the actual saved bad state, does the complete approved v3 change the measured native-update-to-sensitivity feedback, while the original native full update reproduces the old effect?',
    measurement='Two arms, four original native optimizer updates per arm. Before/after original fixed-input/fixed-BF16-seed parameter VJP and native hidden-adjoint layer gains; actual native update metrics and parameter displacement. Recompute v3 DT with the existing owner, then restore captured training RNG before update.',
    decision='Only compare the observed conditional feedback. A smaller first gradient alone is insufficient. Neither success nor failure proves the complete from-scratch counterfactual trajectory.',
    limits=[
        'Old checkpoint80 weights, optimizer history, original iteration81 inputs and reference/old log probabilities remain fixed across arms.',
        'The full new method changes group multiplier and uses variance-only scaling as approved by the user; it is not the previous partial-row ablation.',
        'Exact old DT credit is not reused for new nonzero multipliers. The deployed native request/DT/advantage components compute all new values.',
        'The fixed output seed probes a specific directional sensitivity; hidden-layer and parameter-coordinate changes are reported separately.',
        'This is a finite diagnostic, not old formal training continuation, a GRPO rerun, a validation-score test, or a long-run stability proof.',
        'No diagnostic parameter artifact is a formal training checkpoint or part of formal backup progress.',
    ], accepted_repair=None, deployed_repair=None, rootcause_repair_verified=False)
for p, digest in report['source_bindings'].items():
    assert hashlib.sha256(Path(p).read_bytes()).hexdigest() == digest
DEST.mkdir(exist_ok=True)
path = DEST / 'case-manifest.json'
assert not path.exists()
path.write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(dict(receipt=str(path), report=report)))

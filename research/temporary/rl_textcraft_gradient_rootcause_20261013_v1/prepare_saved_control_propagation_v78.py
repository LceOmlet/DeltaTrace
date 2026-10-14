"""Prepare a read-only native derivative comparison at saved GRPO weights.

This does not run GRPO advantages, sampling, training or evaluation. It reuses
the exact input, coefficients and BF16 adjoint from the completed DTPO probes.
"""
import hashlib
import json
from pathlib import Path
import time

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT = ROOT / 'receipts/textcraft-gradient-rootcause-20261013-v1'
DEST = OUT / 'native-trained-control-propagation-v78'
prior = json.loads((OUT / 'native-fixed-seed-propagation-v33/case-manifest.json').read_bytes())
stage = ROOT / 'backup-access/local-snapshots/1792008986210692700'
selection_path = stage / 'selection.json'
selection = json.loads(selection_path.read_bytes())
assert selection['completed_checkpoints']['grpo'] == 160
relative = Path('runs/textcraft-grpo-formal-native-graph-fresh-20261012-v1/textcraft-grpo/checkpoints/global_step_160/actor')
checkpoint = stage / 'files' / ROOT.relative_to('/') / relative
bindings = {}
for prefix in ('model', 'optim', 'extra_state'):
    for rank in (0, 1):
        path = checkpoint / f'{prefix}_world_size_2_rank_{rank}.pt'
        original_relative = str(ROOT.relative_to('/') / relative / path.name)
        assert path.is_file() and path.stat().st_size > 0
        bindings[str(path)] = dict(SHA256=selection['hashes'][original_relative], bytes=path.stat().st_size,
            verification='Original backup PREPARE hashed this immutable completed-checkpoint snapshot.')

original_launch = ROOT / 'runs/textcraft-dt-formal-native-graph-fresh-20261012-v1/launch.json'
control_launch = ROOT / 'runs/textcraft-grpo-formal-native-graph-fresh-20261012-v1/launch.json'
launch = json.loads(original_launch.read_bytes())
control = json.loads(control_launch.read_bytes())
config_keys = ('actor_rollout_ref.model.path', 'actor_rollout_ref.model.lora_rank',
    'actor_rollout_ref.model.lora_alpha', 'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu',
    'actor_rollout_ref.actor.ppo_mini_batch_size', 'actor_rollout_ref.actor.ppo_epochs',
    'actor_rollout_ref.actor.ppo_max_token_len_per_gpu')
for key in config_keys:
    assert launch['options'].get(key) == control['options'].get(key), key

carrier = next(c for c in prior['cases'] if c['name'] == 'late5_seed80')
completed_seed_binding = json.loads((OUT / 'native-full-current-v3-feedback-v69/case-manifest.json').read_bytes())['cases'][0]
assert completed_seed_binding['boundary_seed_source_case'] == 'late80_native'
assert hashlib.sha256(Path(carrier['path']).read_bytes()).hexdigest() == carrier['SHA256']
cases = []
for name, fixed in (('control160_native_target', False), ('control160_fixed_seed80', True)):
    case = dict(carrier, name=name, checkpoint=160, checkpoint_path=str(checkpoint),
        capture_boundary_seed=False, boundary_seed_source_case='late80_native' if fixed else None)
    case.pop('expected_trainable_SHA256_by_rank', None)
    if fixed:
        for key in ('boundary_seed_source_directory', 'boundary_seed_SHA256_by_rank_micro'):
            case[key] = completed_seed_binding[key]
        assert len(case['boundary_seed_SHA256_by_rank_micro']) == 16
        for rank_micro in case['boundary_seed_SHA256_by_rank_micro']:
            rank, micro = rank_micro.split(':')
            assert (Path(case['boundary_seed_source_directory']) / f'seed-late80_native-rank{rank}-micro{micro}.pt').is_file()
    else:
        case.pop('boundary_seed_source_directory', None)
        case.pop('boundary_seed_SHA256_by_rank_micro', None)
    cases.append(case)

report = dict(unix=time.time(), status='prepared_only_no_GPU_submission', scope=__doc__,
    destination=str(DEST), checkpoint=160, checkpoint_path=str(checkpoint), cases=cases,
    question='On the same factual input and exact saved decoder31 BF16 adjoint, do already-trained stable-control weights have the same large internal propagation as old DTPO weights?',
    why_needed='Existing DTPO fixed-input/fixed-seed global VJP grows0.0604→1.1039 at checkpoints5→80. Existing gradient comparisons apply GRPO advantages at DTPO weights; they do not measure propagation at the trained control weights. The saved complete control checkpoint160 survives in an immutable original backup snapshot, so no control training is repeated.',
    discriminating_quantity='Two original actor derivative passes at saved control checkpoint160: native target PG and the exact unchanged old decoder31 seed. Compare global parameter VJP, layer adjoints and original B4 Gram to completed DTPO5/40/80 records.',
    decision='Use the actual control propagation to decide whether the observed DTPO growth is specific to its learned internal geometry, or whether stable control gradients coexist with similar sensitivity and therefore require stronger attention to the advantage/update projection. No owner or credit change follows from a norm quotient alone.',
    counterfactual_scope='A read-only comparison of actual trained states, not a from-scratch randomized causal training experiment. Control160 and DTPO80 have different histories and completed-step counts.',
    original_launch=str(original_launch), control_launch=str(control_launch),
    original_launch_SHA256=hashlib.sha256(original_launch.read_bytes()).hexdigest(),
    control_launch_SHA256=hashlib.sha256(control_launch.read_bytes()).hexdigest(),
    checkpoint_snapshot_selection=dict(path=str(selection_path), SHA256=hashlib.sha256(selection_path.read_bytes()).hexdigest()),
    checkpoint_snapshot_files=bindings,
    fixed_configuration={k:launch['options'].get(k) for k in config_keys},
    passive_boundary_readout_enabled=True, native_ratio_chain_readout=True,
    decoder_branch_layers=[29, 30, 31], save_native_probability_carrier=True,
    natural_target_PG_passes=1, fixed_seed_VJP_passes=1,
    optimizer_updates=0, scheduler_updates=0, GRPO_algorithm_calls=0,
    GRPO_training_reruns=0, generation_calls=0, DT_calls=0, validation_score_calls=0,
    trained_control_checkpoint_model_derivative_passes=2, formal_training_changes=0,
    source_bindings={str(Path(launch['effective_environment']['VERL_ROOT']) / suffix):digest for suffix,digest in (
        ('verl/workers/actor/dp_actor.py','3a65e173300be82a7a9e056a96227c4f746eabc3be778ef41c8d50138d52ce6c'),
        ('verl/trainer/ppo/core_algos.py','fc2f992b16fb7fb23aebc683ad5f00136cf61fc9013cd426f5046983babe7299'),
        ('verl/workers/fsdp_workers.py','9dfd6ce795384b98faa962269e1eac98ad05ec65e42af2aac434c41db41641d3'),
    )},
    limits=[
        'No new GRPO training, advantage computation, rollout, validation score or DT query; only the saved actor weights are reused.',
        'The exact same original64-row carrier has15 matched factual targets and49 unchanged zero-derivative rows. It is not a complete control optimizer batch.',
        'The fixed BF16 adjoint is reused without normalization, scaling or reconstruction; its parameter VJP is not the control policy gradient.',
        'All original input tokens, masks, B4/native1/16, LoRA8/16 and32k capacity settings are retained.',
        'Different training histories and checkpoint counts prevent interpreting a between-method norm difference as a precise causal fraction.',
        'Reporting variation and low-precision native propagation are retained; no new tolerances or universal singular-value claims.',
    ], rootcause_repair_verified=False, accepted_repair=None, deployed_repair=None)
for path, expected in report['source_bindings'].items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
DEST.mkdir(exist_ok=True)
path = DEST / 'case-manifest.json'
if path.exists():
    assert not (DEST / 'job.json').exists() and not (DEST / 'result.json').exists()
    previous = DEST / 'case-manifest.before-explicit-seed-binding.json'
    assert not previous.exists()
    previous.write_bytes(path.read_bytes())
    report['prepared_only_binding_correction'] = dict(path=str(previous),
        SHA256=hashlib.sha256(previous.read_bytes()).hexdigest(),
        reason='The prior probe used its own destination as implicit seed root; this new destination needs the original completed seed path and16 exact SHA bindings.')
path.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(dict(receipt=str(path), report=report)))

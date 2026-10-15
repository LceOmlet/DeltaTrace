"""Bind saved native optimizer outcomes to the unchanged actual whole-loss seed.

CPU only. Convert existing local LoRA shards into the original Torch DTensor
state-dict representation; no model/loss/optimizer implementation is provided.
The six planned native VJPs execute no training update or new DT computation.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import time

import torch

torch.set_num_threads(4)
ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT = ROOT / 'receipts/textcraft-gradient-rootcause-20261013-v1'
DEST = OUT / 'native-full-policy-saved-update-feedback-v90'
assert not (DEST / 'case-manifest.json').exists() and not (DEST / 'job.json').exists()
DEST.mkdir(exist_ok=True)
direction = json.loads((OUT / 'native-full-policy-direction-exchange-v86/case-manifest.json').read_bytes())
natural = next(c for c in direction['cases'] if c['name'] == 'full80_native')
seed_root = OUT / 'native-full-policy-direction-exchange-v86'
seed_receipt_path = seed_root / 'captured-full80-seed-bindings.json'
seed_receipt = json.loads(seed_receipt_path.read_bytes())
assert len(seed_receipt['files']) == 16
for item in seed_receipt['files']:
    with Path(item['path']).open('rb') as f:
        assert hashlib.file_digest(f, 'sha256').hexdigest() == item['SHA256']
assert hashlib.sha256(Path(natural['path']).read_bytes()).hexdigest() == natural['SHA256']

def digest(model):
    h = hashlib.sha256()
    for name, value in model.items():
        if '.lora_' not in name:
            continue
        local = value.to_local() if hasattr(value, 'to_local') else value
        h.update(name.encode())
        h.update(local.detach().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()

def new_case(name, checkpoint, checkpoint_path, origin):
    case = copy.deepcopy(natural)
    case.update(name=name, checkpoint=checkpoint, checkpoint_path=str(checkpoint_path),
        capture_boundary_seed=False, boundary_seed_source_case='full80_native',
        boundary_seed_source_directory=str(seed_root), boundary_seed_SHA256_by_rank_micro=seed_receipt['hashes'],
        expected_trainable_SHA256_by_rank={}, saved_native_parameter_origin=origin,
        diagnostic_parameter_partition=dict(layers=list(range(32)), base_checkpoint=checkpoint,
            donor_checkpoint='saved_native_model_only', owner='Original Torch/FSDP sharded model.load_state_dict(strict=False)', rank_artifacts={}))
    case.pop('captured_seed_receipt', None)
    return case

cases = [new_case('old80_before', 80, natural['checkpoint_path'], dict(type='original_complete_checkpoint', checkpoint=80))]
outcome_bindings = []
for tag, directory, arms in (
    ('v47', 'native-common-update-feedback-v47', ('full_original', 'without_constant_positive_component')),
    ('v69', 'native-full-current-v3-feedback-v69', ('full_original', 'approved_v3_full_method')),
):
    path = OUT / directory / 'result.json'
    state = json.loads(path.read_bytes())
    assert state['phase'].startswith('complete_') and len(state['observations']) == 2
    for arm in arms:
        obs = next(o for o in state['observations'] if o['arm'] == arm)
        before, after = [sorted(obs[k], key=lambda x: x['rank']) for k in ('before', 'after')]
        assert all(x['scheduler'] == 80 and x['optimizer_steps'] == [320.] for x in before)
        assert all(x['scheduler'] == 81 and x['optimizer_steps'] == [324.] for x in after)
        origin = dict(type='saved_actual_native_four_update_model', arm=arm, revision=tag,
            native_result_path=str(path), native_result_SHA256=hashlib.sha256(path.read_bytes()).hexdigest(),
            actual_before=before, actual_after=after,
            scope='Only exact saved model parameters are replayed; this is not a resumable checkpoint or an optimizer replay.')
        cases.append(new_case(f'{tag}_{arm}_after', 80, natural['checkpoint_path'], origin))
        outcome_bindings.append(origin)

# Preserve the original completed control actor by native hard links. This is
# not a new trained checkpoint, a GRPO rerun or an independent storage copy.
control_manifest = json.loads((OUT / 'native-trained-control-propagation-v78/case-manifest.json').read_bytes())
control_source = Path(control_manifest['checkpoint_path'])
control = DEST / 'original-completed-GRPO160-actor-links'
control.mkdir(exist_ok=True)
control_links = []
for original_path, binding in control_manifest['checkpoint_snapshot_files'].items():
    source = Path(original_path)
    assert source.parent == control_source and source.is_file()
    assert source.stat().st_size == binding['bytes']
    target = control / source.name
    if not target.exists():
        os.link(source, target)
    assert os.path.samefile(source, target)
    control_links.append(dict(source=str(source), path=str(target), inode=target.stat().st_ino,
        bytes=target.stat().st_size, original_immutable_snapshot_binding=binding,
        SHA256_scope='Original immutable snapshot PREPARE hash retained; actual LoRA values checked again below.'))
assert len(control_links) == 6
cases.append(new_case('GRPO160_same_full80_seed', 160, control,
    dict(type='saved_original_complete_control_actor', original_manifest=str(OUT / 'native-trained-control-propagation-v78/case-manifest.json'),
         checkpoint=160, links=control_links, GRPO_training_rerun=False)))

parameter_bindings = []
for rank in (0, 1):
    path = Path(natural['checkpoint_path']) / f'model_world_size_2_rank_{rank}.pt'
    original_model = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    keys = [name for name in original_model if '.lora_' in name]
    assert len(keys) == 496
    native_chain_path = seed_root / f'iterationfull80_native-rank{rank}-chain.json'
    native_chain = json.loads(native_chain_path.read_bytes())
    base_digest = digest(original_model)
    assert base_digest == native_chain['trainable_SHA256_before'] == native_chain['trainable_SHA256_after']
    template = DEST / f'native-DTensor-template-rank{rank}.pt'
    assert not template.exists()
    torch.save({name: original_model[name] for name in keys}, template)
    del original_model
    for case in cases:
        origin = case['saved_native_parameter_origin']
        raw = None
        if origin['type'] == 'saved_actual_native_four_update_model':
            for phase in ('actual_before', 'actual_after'):
                item = next(x for x in origin[phase] if x['rank'] == rank)
                snapshot = Path(item['trainable_snapshot'])
                assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == item['snapshot_SHA256']
                values = torch.load(snapshot, map_location='cpu', weights_only=False)
                assert list(values) == keys and digest(values) == item['trainable_SHA256']
                if phase == 'actual_before':
                    assert digest(values) == base_digest
                else:
                    raw = values
            wanted = next(x for x in origin['actual_after'] if x['rank'] == rank)['trainable_SHA256']
            before_digest = base_digest
        elif origin['type'] == 'saved_original_complete_control_actor':
            m = torch.load(control / f'model_world_size_2_rank_{rank}.pt', map_location='cpu', weights_only=False, mmap=True)
            wanted = digest(m)
            expected = ('5aa6992792766c2813f484172a5d62a1fcd60eb960f4bc1caf332e1829f382bf',
                        '550f381444b2a6258a3b7c3b64cc15a55c57ad64159904eafd55cce34aaaf24b')[rank]
            assert wanted == expected
            raw = {name: m[name].to_local() for name in keys}
            before_digest = wanted
        else:
            wanted = before_digest = base_digest
        # No mmap here: copy_ writes only newly owned diagnostic CPU storage.
        partial = torch.load(template, map_location='cpu', weights_only=False)
        assert list(partial) == keys
        if raw is not None:
            with torch.no_grad():
                for name, value in partial.items():
                    local = value.to_local()
                    assert local.shape == raw[name].shape and local.dtype == raw[name].dtype
                    local.copy_(raw[name])
        assert digest(partial) == wanted
        artifact_path = DEST / f"{case['name']}-rank{rank}-native-partial-state.pt"
        assert not artifact_path.exists()
        torch.save(partial, artifact_path)
        artifact = dict(path=str(artifact_path), SHA256=hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
            selected_keys=keys, selected_parameter_count=len(keys),
            selected_local_elements=sum(v.to_local().numel() for v in partial.values()),
            bytes=artifact_path.stat().st_size, expected_base_trainable_SHA256=before_digest,
            expected_result_trainable_SHA256=wanted, actual_donor_trainable_SHA256=wanted)
        case['diagnostic_parameter_partition']['rank_artifacts'][str(rank)] = artifact
        case['expected_trainable_SHA256_by_rank'][str(rank)] = wanted
        parameter_bindings.append(dict(case=case['name'], rank=rank, **artifact))
        del partial, raw

report = dict(unix=time.time(), status='prepared_only_no_GPU_submission', scope=__doc__, destination=str(DEST),
    checkpoint=80, checkpoint_path=natural['checkpoint_path'], cases=cases,
    diagnostic_advantage_views={'full_DT': 'diagnostic_full_DT'},
    passive_boundary_readout_enabled=True, native_ratio_chain_readout=True, decoder_branch_layers=[29, 30, 31],
    save_native_probability_carrier=False, original_seed_binding=dict(path=str(seed_receipt_path), SHA256=hashlib.sha256(seed_receipt_path.read_bytes()).hexdigest()),
    question='Do the already executed native original, zero-group-ablated and full-v3 updates change internal sensitivity along the actual complete PG direction in the same way as the selected-target probe?',
    why_needed='Completed v86 establishes 11.897/14.672-fold sensitivity growth for both actual full-loss directions. Existing update-feedback evidence v47/v69 used a selected15-target adjoint, so it cannot establish its effect on the whole-loss direction. The exact native after-parameters and full80 adjoint are already saved; no new optimizer or DT call is needed.',
    discriminating_quantity='Six original-native VJPs using the identical complete full80 BF16 decoder31 adjoint: original80, four actual saved native update outcomes, and saved GRPO160 actor. Compare global parameter VJP, layer propagation and B4 Gram.',
    decision='Determine whether the same-reward-group update driver and approved v3 change the actual whole-direction sensitivity feedback. Keep original v47/v69 repetitions separate and report actual control geometry without calling it a GRPO policy gradient.',
    counterfactual_scope='Model-only replay of finite saved native updates at original80, plus a saved trained-state reference; not a new optimizer experiment or a from-scratch proof.',
    saved_native_outcomes=outcome_bindings, parameter_artifacts=parameter_bindings, control_original_links=control_links,
    planned_native_full_policy_derivatives=0, planned_native_fixed_seed_derivatives=6,
    optimizer_updates=0, scheduler_updates=0, DT_calls=0, generation_calls=0, validation_score_calls=0,
    GRPO_training_reruns=0, formal_training_changes=0, rootcause_repair_verified=False,
    limits=['All cases use exactly the original81 complete global64 carrier, original full256 advantages, B4/1/16/32k/LoRA8/16. The replaced boundary means these are conditional VJPs, not natural post-update policy gradients.',
        'Actual native v47/v69 optimizer outcomes contain all original clipping, adaptive state, entropy and KL interactions; differences are not linear optimizer-component contributions.',
        'Model-only after-state artifacts are never formal checkpoints, training restore targets or formal backup selections. Optimizer/scheduler remain untouched during the readout.',
        'Original repetitions have measured native variation; no invented tolerance, universal all-layer-decrease gate or operator-norm claim.',
        'Control160 and DT80 have different training histories. This fixed-direction reference does not rerun GRPO or identify a precise between-algorithm causal fraction.'])
assert not torch.cuda.is_initialized()
dest = DEST / 'case-manifest.json'
dest.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(dict(receipt=str(dest), report=report)))

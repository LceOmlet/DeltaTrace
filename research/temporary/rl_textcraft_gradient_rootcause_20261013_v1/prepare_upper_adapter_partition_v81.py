"""Prepare two saved-parameter interventions for the measured upper gain.

CPU checkpoint reads and small LoRA artifacts only. The two derivatives keep
the same factual carrier/BF16 seed and swap only actual layers24-31 LoRA
values between the completed original DTPO40/80 checkpoints. No new training
checkpoint, credit, optimizer, input or model implementation is made.
"""
import copy
import hashlib
import json
import re
import time
from pathlib import Path

import torch

torch.set_num_threads(4)
ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT = ROOT / 'receipts/textcraft-gradient-rootcause-20261013-v1'
DEST = OUT / 'native-upper-adapter-partition-v81'
assert not (DEST / 'case-manifest.json').exists()
assert not (DEST / 'job.json').exists()
DEST.mkdir(exist_ok=True)

mid = json.loads((OUT / 'native-midpoint-fixed-seed-v35/case-manifest.json').read_bytes())
late = json.loads((OUT / 'native-fixed-seed-propagation-v33/case-manifest.json').read_bytes())
seed_case = mid['cases'][0]
paths = {40: Path(seed_case['checkpoint_path']),
         80: ROOT / 'runs/textcraft-dt-formal-native-graph-fresh-20261012-v1/textcraft-dt/checkpoints/global_step_80/actor'}
assert hashlib.sha256(Path(seed_case['path']).read_bytes()).hexdigest() == seed_case['SHA256']
for identity, sha in seed_case['boundary_seed_SHA256_by_rank_micro'].items():
    rank, micro = identity.split(':')
    p = Path(seed_case['boundary_seed_source_directory']) / f'seed-late80_native-rank{rank}-micro{micro}.pt'
    assert hashlib.sha256(p.read_bytes()).hexdigest() == sha

def selected(name):
    match = re.search(r'\.layers\.(\d+)\.', name)
    return '.lora_' in name and match is not None and 24 <= int(match[1]) <= 31

def trainable_digest(model):
    result = hashlib.sha256()
    for name, value in model.items():
        if '.lora_' not in name:
            continue
        local = value.to_local() if hasattr(value, 'to_local') else value
        result.update(name.encode())
        result.update(local.detach().contiguous().view(torch.uint8).numpy().tobytes())
    return result.hexdigest()

cases = []
for base, donor in ((80, 40), (40, 80)):
    case = copy.deepcopy(seed_case)
    case.update(name=f'DT{base}_upper24_31_from{donor}', checkpoint=base,
                checkpoint_path=str(paths[base]), expected_trainable_SHA256_by_rank={},
                diagnostic_parameter_partition=dict(layers=list(range(24, 32)),
                    base_checkpoint=base, donor_checkpoint=donor,
                    owner='Original Torch/FSDP sharded model.load_state_dict(strict=False)',
                    rank_artifacts={}))
    cases.append(case)

readbacks = []
for rank in (0, 1):
    models = {}
    native_digests = {}
    for step in (40, 80):
        chain_path = OUT / ('native-midpoint-fixed-seed-v35' if step == 40 else 'native-fixed-seed-propagation-v33') / (
            f'iterationlate40_seed80-rank{rank}-chain.json' if step == 40 else f'iterationlate80_native-rank{rank}-chain.json')
        chain = json.loads(chain_path.read_bytes())
        assert chain['trainable_SHA256_before'] == chain['trainable_SHA256_after']
        path = paths[step] / f'model_world_size_2_rank_{rank}.pt'
        models[step] = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        native_digests[step] = trainable_digest(models[step])
        assert native_digests[step] == chain['trainable_SHA256_before']
        readbacks.append(dict(checkpoint=step, rank=rank, path=str(path),
                              native_trainable_SHA256=native_digests[step],
                              completed_native_chain=str(chain_path),
                              completed_native_chain_SHA256=hashlib.sha256(chain_path.read_bytes()).hexdigest()))
    assert tuple(models[40]) == tuple(models[80])
    keys = [name for name in models[40] if selected(name)]
    assert {int(re.search(r'\.layers\.(\d+)\.', name)[1]) for name in keys} == set(range(24, 32))
    for name in keys:
        a, b = models[40][name], models[80][name]
        assert a.shape == b.shape and a.dtype == b.dtype
        assert tuple(str(p) for p in a.placements) == tuple(str(p) for p in b.placements) == ('S(0)',)
    for case in cases:
        spec = case['diagnostic_parameter_partition']
        base, donor = spec['base_checkpoint'], spec['donor_checkpoint']
        partial = {name: models[donor][name] for name in keys}
        p = DEST / f"{case['name']}-rank{rank}-partial-lora-state.pt"
        assert not p.exists()
        torch.save(partial, p)
        composite = {name: models[donor][name] if selected(name) else value
                     for name, value in models[base].items()}
        expected = trainable_digest(composite)
        case['expected_trainable_SHA256_by_rank'][str(rank)] = expected
        spec['rank_artifacts'][str(rank)] = dict(path=str(p), SHA256=hashlib.sha256(p.read_bytes()).hexdigest(),
            selected_keys=keys, selected_parameter_count=len(keys),
            selected_local_elements=sum(v.to_local().numel() for v in partial.values()),
            expected_base_trainable_SHA256=native_digests[base],
            expected_result_trainable_SHA256=expected,
            actual_donor_checkpoint_path=str(paths[donor]),
            actual_donor_trainable_SHA256=native_digests[donor])
    del models

report = dict(unix=time.time(), status='prepared_only_no_GPU_submission', scope=__doc__, destination=str(DEST),
    checkpoint=80, checkpoint_path=str(paths[80]), cases=cases,
    checkpoint_readbacks=readbacks, passive_boundary_readout_enabled=True,
    native_ratio_chain_readout=True, decoder_branch_layers=[29, 30, 31],
    question='Does replacing the actual learned upper24-31 LoRA state transfer the observed large internal gain, while the remainder of the model and the incoming BF16 seed are held fixed?',
    why_needed='Same input/seed native global VJP40-to80 grows0.08634→1.10390, while upper24-31 directional transfer grows7.558x. The saved stable GRPO160 comparison has VJP0.10587 and7.532x less upper gain than DTPO80. Layer norm quotients alone cannot tell whether upper adapter values or the forward state inherited from lower layers creates this difference.',
    discriminating_quantity='Two native fixed-seed derivatives: DTPO80 body with original40 upper24-31 LoRA, and DTPO40 body with original80 upper24-31 LoRA. Compare actual upper hidden-adjoint transfer and lower parameter VJP with the already completed40/80 baselines.',
    decision='If the gain follows the exchanged upper state, focus on the actual historical update of that partition; if it stays with the unchanged body, follow the inherited forward state. No official model or method repair is accepted from this localization alone.',
    counterfactual_scope='Parameter intervention on saved original DTPO states, not a rollout/quality/credit experiment or a formal training restart.',
    limits=['Only actual LoRA parameters of the evidence-located upper8 layers are exchanged; original frozen weights, all other trainable coordinates, factual input, mask, loss carrier and BF16 seed stay fixed in each comparison.',
        'This does not identify the historical optimizer/advantage source of a transferred gain or prove complete long-run repair.',
        'Low-precision propagation variation remains recorded. Original native B4/global64/1/16 and32k/LoRA8/16 settings remain.',
        'Both interventions are diagnostic model states, never formal checkpoints, restore targets, or backup selections.'],
    formal_training_changes=0, model_calls=0, optimizer_updates=0, scheduler_updates=0,
    DT_calls=0, generation_calls=0, validation_score_calls=0, GRPO_calls=0,
    planned_native_fixed_seed_derivatives=2, rootcause_repair_verified=False,
    accepted_repair=None, deployed_repair=None)
assert not torch.cuda.is_initialized()
p = DEST / 'case-manifest.json'
p.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(dict(receipt=str(p), report=report)))

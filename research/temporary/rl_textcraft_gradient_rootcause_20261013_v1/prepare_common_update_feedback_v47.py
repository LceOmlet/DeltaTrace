"""Prepare a conditional native-update intervention, not a training-quality test.

Reuse original iteration81's complete256 rows and checkpoint80. The only
diagnostic intervention removes the already measured constant-positive-group
PG component; all other frozen advantages and original entropy/KL stay intact.
"""
import hashlib
import inspect
import json
import os
from pathlib import Path
import time
import torch
from verl import DataProto
from verl.trainer.ppo.ray_trainer import compute_advantage, AdvantageEstimator
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager

torch.set_num_threads(4)
ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT=ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1'
DEST=OUT/'native-common-update-feedback-v47'
assert not (DEST/'case-manifest.json').exists(), 'Preserve prepared inputs; no duplicate preparation'
old_launch=json.loads((ROOT/'runs/textcraft-dt-formal-native-graph-fresh-20261012-v1/launch.json').read_bytes())
verl=Path(old_launch['effective_environment']['VERL_ROOT'])
bindings={'verl/workers/actor/dp_actor.py':'3a65e173300be82a7a9e056a96227c4f746eabc3be778ef41c8d50138d52ce6c',
          'verl/trainer/ppo/core_algos.py':'fc2f992b16fb7fb23aebc683ad5f00136cf61fc9013cd426f5046983babe7299',
          'verl/workers/fsdp_workers.py':'9dfd6ce795384b98faa962269e1eac98ad05ec65e42af2aac434c41db41641d3',
          'verl/utils/checkpoint/fsdp_checkpoint_manager.py':'f65b65383a7265b4d2808441427fb8a2ee891b84fcda4cdf1d961a3d96a47a01'}
for name,sha in bindings.items():assert hashlib.sha256((verl/name).read_bytes()).hexdigest()==sha
native_rng_methods={name:dict(source=inspect.getsourcefile(function),signature=str(inspect.signature(function)))
                    for name,function in inspect.getmembers(FSDPCheckpointManager,inspect.isfunction) if 'rng' in name.lower()}
assert 'load_rng_state' in native_rng_methods, native_rng_methods
receipts=[json.loads((OUT/f'dt/rank{rank}-capture.json').read_bytes()) for rank in (0,1)]
for receipt in receipts:
    assert receipt['checkpoint_step']==80 and receipt['status']=='saved'
    assert hashlib.sha256(Path(receipt['input']).read_bytes()).hexdigest()==receipt['input_sha256']
data=DataProto.concat([DataProto.load_from_disk(receipt['input']) for receipt in receipts])
assert len(data)==256
raw=data.batch['token_level_rewards'].sum(-1)
assert set(raw.tolist()) <= {0.,1.}
group=compute_advantage(data.select(batch_keys=['responses','token_level_rewards','response_mask','loss_mask'],
    non_tensor_batch_keys=['uid','traj_uid']),adv_estimator=AdvantageEstimator.GRPO,multi_turn=True,norm_adv_by_std_in_grpo=True)
constant_positive=group.batch['advantages'].abs().sum(-1).eq(0)&raw.gt(0)
first64=torch.cat((torch.arange(32),torch.arange(128,160)))
assert int(constant_positive[first64].sum())==21, 'Must match the measured v26 component, not invent a new grouping'
DEST.mkdir(exist_ok=True)
arms=[]
for name in ('full_original','without_constant_positive_component'):
    view=data.select(deepcopy=True);view.batch=data.batch.clone(recurse=False)
    if name!='full_original':
        view.batch['advantages']=torch.where(constant_positive[:,None],torch.zeros_like(data.batch['advantages']),data.batch['advantages'])
    else:
        assert torch.equal(view.batch['advantages'],data.batch['advantages'])
    for key in ('old_log_probs','ref_log_prob','responses','loss_mask','dt_token_advantages','dt_q_estimates','dt_v_estimates','token_level_rewards'):
        assert torch.equal(view.batch[key],data.batch[key]), key
    path=DEST/(name+'-original81.pkl');view.save_to_disk(str(path))
    arms.append(dict(name=name,path=str(path),SHA256=hashlib.sha256(path.read_bytes()).hexdigest(),rows=len(view)))
seed_spec=json.loads((OUT/'native-midpoint-fixed-seed-v35/case-manifest.json').read_bytes())['cases'][0]
original80=json.loads((OUT/'native-fixed-seed-propagation-v33/case-manifest.json').read_bytes())
point80=next(c for c in original80['cases'] if c['name']=='late80_native')
carrier=Path(point80['path']);assert hashlib.sha256(carrier.read_bytes()).hexdigest()==point80['SHA256']
cases=[]
for arm in arms:
    for suffix,step in [('before',80),('after',81)]:
        case=dict(name=arm['name']+'_'+suffix,path=str(carrier),SHA256=point80['SHA256'],rows=64,
            checkpoint=step,state_kind='original_completed_checkpoint80' if suffix=='before' else 'diagnostic_after_four_native_optimizer_steps_not_a_saved_checkpoint',
            boundary_seed_source_case='late80_native',boundary_seed_source_directory=seed_spec['boundary_seed_source_directory'],
            boundary_seed_SHA256_by_rank_micro=seed_spec['boundary_seed_SHA256_by_rank_micro'])
        if suffix=='before':case['expected_trainable_SHA256_by_rank']=point80.get('expected_trainable_SHA256_by_rank',original80.get('expected_trainable_SHA256_by_rank'))
        cases.append(case)
manifest=dict(unix=time.time(),status='prepared_only_no_GPU_submission',scope=__doc__,destination=str(DEST),
    original_launch=str(ROOT/'runs/textcraft-dt-formal-native-graph-fresh-20261012-v1/launch.json'),
    checkpoint=80,checkpoint_path=str(ROOT/'runs/textcraft-dt-formal-native-graph-fresh-20261012-v1/textcraft-dt/checkpoints/global_step_80/actor'),
    source_bindings={str(verl/k):v for k,v in bindings.items()},arms=arms,cases=cases,
    captured_training_RNG_by_rank={str(rank):dict(path=receipt['rng'],SHA256=hashlib.sha256(Path(receipt['rng']).read_bytes()).hexdigest()) for rank,receipt in enumerate(receipts)},
    native_rng_restore_interface=native_rng_methods['load_rng_state'],native_ratio_chain_readout=True,
    passive_boundary_readout_enabled=True,decoder_branch_layers=[29,30,31],
    removed_component_rows=int(constant_positive.sum()),removed_component_policy_tokens=int(data.batch['response_mask'][constant_positive].sum()),
    original_first_global64_removed_rows=21,exact_old_logprob_reference_inputs_retained=True,
    diagnostic_native_optimizer_steps=8,diagnostic_native_scheduler_steps=2,formal_training_changes=0,
    new_generation_calls=0,new_DT_calls=0,validation_score_calls=0,GRPO_model_reruns=0,
    why_needed='v26 measured constant-positive-group PG2.22119 versus full2.22405. v42 held exact incoming seed and measured last-MLP feedback from-0.3908 at40 to+0.15267 at80. v43/v44 show adaptive step size falls while directions turn. Large component norm and raw-moment cosine cannot determine its effect through the actual native optimizer on internal gain.',
    question='Conditional on original checkpoint80, original captured iteration81 inputs/RNG/state and unchanged PPO, does the measured constant-positive-group component increase internal fixed-seed gain after the four actual native updates?',
    measurement='Native parameter displacement and fixed-input/fixed-BF16-seed parameter VJP/last-MLP residual feedback before versus after, with and without that one existing gradient component.',
    decision='A local causal update-to-gain comparison; no validation reward score. If full original updates do not increase gain, or component removal does not reduce the increase, do not claim common-component reinforcement at this point.',
    limits=['This conditional old-state intervention cannot prove that the same component caused every step of the85-iteration trajectory, or prove the new method is repaired from scratch.',
            'Whole-batch original whitening stays frozen in both diagnostic arms; no advantage recomputation or new baseline is introduced.',
            'The removal is only the component already measured in v26: native GRPO-zero AND positive raw reward rows. It does not remove zero-reward whitening coefficients or change entropy/KL.',
            'After native updates, the probe alone uses a fixed saved adjoint. It is not a training advantage or a full-policy gradient above block31.',
            'No formal old run is resumed, no diagnostic checkpoint is treated as formal, no sampling/validation/GRPO rerun occurs.',
            'GPU resources are not selected or occupied during preparation. Current formal DTPO/GRPO/PPO are not touched.'],
    accepted_repair=None,deployed_repair=None)
path=DEST/'case-manifest.json';path.write_text(json.dumps(manifest,indent=2)+'\n')
assert not torch.cuda.is_initialized()
print(json.dumps(dict(receipt=str(path),report=manifest)))

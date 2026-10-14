"""Partition existing DT loss coefficients by the original native reward groups.

No new credit calculation or model query. Full native inputs are unchanged.
The original GRPO computation is called only on CPU to identify its exact zero
signal; no GRPO training or model run is made.
"""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time
import torch
from verl import DataProto
from verl.trainer.ppo.ray_trainer import compute_advantage

torch.set_num_threads(4)
ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT=ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1'
DEST=OUT/'native-DT-group-gradient-components-v26'
receipts=[json.loads((OUT/f'dt/rank{rank}-capture.json').read_bytes()) for rank in (0,1)]
full=DataProto.concat([DataProto.load_from_disk(r['input']) for r in receipts])
rewards=full.batch['token_level_rewards'].sum(-1)
groups=defaultdict(list)
for i,uid in enumerate(full.non_tensor_batch['uid']):groups[str(uid)].append(i)
same=torch.zeros(len(full),dtype=torch.bool)
for rows in groups.values():
    r=rewards[rows]
    if float(r[0])>0 and bool(torch.all(r==r[0])):same[rows]=True
reference=full.select(deepcopy=True);reference.batch=full.batch.clone(recurse=False)
reference=compute_advantage(reference,adv_estimator='grpo',multi_turn=True,norm_adv_by_std_in_grpo=True)
assert not reference.batch['advantages'][same].count_nonzero()
idx=torch.cat((torch.arange(32),torch.arange(128,160)))
data=full.select_idxs(idx);data.batch=data.batch.clone(recurse=False)
a=data.batch['advantages'];selected=same[idx,None]
data.batch['diagnostic_full_DT']=a.clone()
data.batch['diagnostic_same_reward_DT']=a*selected
data.batch['diagnostic_other_groups_DT']=a*~selected
assert torch.equal(data.batch['diagnostic_same_reward_DT']+data.batch['diagnostic_other_groups_DT'],a)
digests={}
for rank,r in enumerate(receipts):
    tensors=torch.load(r['lora_local'],map_location='cpu',weights_only=False)
    h=hashlib.sha256()
    for name,value in tensors.items():h.update(name.encode());h.update(value.contiguous().view(torch.uint8).numpy().tobytes())
    digests[str(rank)]=h.hexdigest()
DEST.mkdir(exist_ok=True)
path=DEST/'original-iteration81-first-global64.pkl';assert not path.exists();data.save_to_disk(str(path))
report=dict(unix=time.time(),scope=__doc__,destination=str(DEST),checkpoint=80,
    checkpoint_path=str(ROOT/'runs/textcraft-dt-formal-native-graph-fresh-20261012-v1/textcraft-dt/checkpoints/global_step_80/actor'),
    expected_trainable_SHA256_by_rank=digests,
    question='Does the actual DT parameter-gradient excess concentrate in sampled same-positive-reward groups whose original GRPO advantage is exactly zero?',
    why_needed='The CPU original-loss coefficient audit finds16 of32 same-positive-reward groups. They contain only14128 of135023 policy tokens but a large DT coefficient-energy share, while native GRPO zeros them. Coefficient energy is not parameter-gradient share; this native partition supplies the missing actual norms and cross terms.',
    selection='Original saved first global64, all rows retained in native order. Group classification uses only original UID/rewards across the full256, not gradient/advantage magnitudes.',
    diagnostic_advantage_views=dict(full_DT='diagnostic_full_DT',same_reward_DT='diagnostic_same_reward_DT',other_groups_DT='diagnostic_other_groups_DT'),
    cases=[dict(name='original81_groups',path=str(path),SHA256=hashlib.sha256(path.read_bytes()).hexdigest(),rows=64)],
    selected_same_reward_rows=int(selected.sum()),selected_other_rows=int((~selected).sum()),
    full_DT_advantage_exact_partition=True,original_GRPO_same_reward_advantage_zero=True,
    passive_boundary_readout_enabled=False,
    limits=['These are components of the unchanged original DT loss at one saved state; none is a new training advantage or proposed method.',
        'Original input tensors, observations, masks and per-token DT advantages are retained. No trajectory is shortened, regenerated or replaced.',
        'Parameter gradients may cancel; norms and inner products must be reported together. Coefficient energy is not a causal share.',
        'This does not prove the entire85-iteration root cause, a new-training outcome, or a repair.'],
    GPU_model_calls=0,new_optimizer_scheduler_DT_rollout_validation_GRPO_calls=0,accepted_repair=None,deployed_repair=None)
assert not torch.cuda.is_initialized()
dest=DEST/'case-manifest.json';assert not dest.exists();dest.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(receipt=str(dest),report=report)))

"""Check the two integration changes against native GRPO on a saved real batch.

No model or training trial. There is no replica of GRPO, PPO or DT here.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT = ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1'
PREPARED = OUT/'DT-group-multiplier-v40-prepared'
launch = json.loads((PREPARED/'launch.json').read_bytes())
env = launch['effective_environment']
sys.path[:0] = env['PYTHONPATH'].split(':')
os.environ.update(env)
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import inspect
import torch
from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.ray_trainer import compute_advantage, AdvantageEstimator
from dt_training_batch import compute_direct_target_credit, CREDIT_KEYS

torch.set_num_threads(4)
assert hashlib.sha256(Path(inspect.getsourcefile(core_algos)).read_bytes()).hexdigest() == 'fc2f992b16fb7fb23aebc683ad5f00136cf61fc9013cd426f5046983babe7299'
assert str(Path(inspect.getsourcefile(compute_advantage))) == env['VERL_ROOT']+'/verl/trainer/ppo/ray_trainer.py'
assert str(Path(inspect.getsourcefile(compute_direct_target_credit))) == env['PYTHONPATH'].split(':')[0]+'/dt_training_batch.py'
receipts = [json.loads((OUT/f'dt/rank{rank}-capture.json').read_bytes()) for rank in (0,1)]
data = DataProto.concat([DataProto.load_from_disk(r['input']) for r in receipts])
original_rewards = data.batch['token_level_rewards'].clone()
group = compute_advantage(data.select(batch_keys=['responses','token_level_rewards','response_mask','loss_mask'],
    non_tensor_batch_keys=['uid','traj_uid']), adv_estimator=AdvantageEstimator.GRPO,
    norm_adv_by_std_in_grpo=True, multi_turn=True)
mask = data.batch['response_mask'].bool()
first_policy = data.batch['response_mask'].long().argmax(-1, keepdim=True)
multiplier = group.batch['advantages'].gather(1,first_policy).squeeze(1)
assert torch.equal(group.batch['advantages'][mask], multiplier[:,None].expand_as(mask)[mask])
assert torch.equal(data.batch['token_level_rewards'],original_rewards)
data.batch['dt_grpo_multipliers'] = multiplier

# The already saved real DT values are sufficient to check the removal of
# whitening at the official advantage boundary, without another DT call.
view = data.select(deepcopy=True)
view.batch = data.batch.clone(recurse=False)
new = compute_advantage(view,adv_estimator=AdvantageEstimator.DELTATRACE,multi_turn=True)
assert torch.equal(new.batch['advantages'][mask],data.batch['dt_token_advantages'][mask])
assert not new.batch['advantages'][~mask].any()

# All-equal groups have a native zero multiplier, even when raw rewards are
# positive. Exercise the actual analytical DT path, which needs no RPC/model.
zero = multiplier.eq(0).nonzero().flatten()
assert len(zero) > 0
zero_view = data.select_idxs(zero)
analytic = compute_direct_target_credit(zero_view,None,eos_token_id=151645,pad_token_id=151643)
assert all(not analytic.batch[k].any() for k in CREDIT_KEYS)
raw = original_rewards.sum(-1)
report = dict(unix=time.time(),scope=__doc__,rows=len(data),
    native_GRPO_valid_tokens_exact_broadcast=True,raw_rewards_unchanged=True,
    DT_advantage_boundary_exact_passthrough=True,observation_padding_zero=True,
    actual_zero_group_DT_path_zero=True,zero_multiplier_rows=len(zero),
    raw_zero_nonzero_group_rows=int(((raw==0)&(multiplier!=0)).sum()),
    positive_reward_zero_group_rows=int(((raw>0)&(multiplier==0)).sum()),
    multiplier_min=float(multiplier.min()),multiplier_max=float(multiplier.max()),
    source_bindings=launch['source_bindings'],configuration_changes=launch['configuration_changes'],
    fixed_configuration=launch['fixed_configuration'],model_calls=0,optimizer_steps=0,
    no_shadow_GRPO=True,rootcause_repair_verified=False)
assert report['raw_zero_nonzero_group_rows'] > 0
assert not torch.cuda.is_initialized()
dest = OUT/'DT-group-multiplier-interface-verification-v40.json'
assert not dest.exists()
dest.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(receipt=str(dest),report=report)))

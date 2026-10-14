"""Real saved batch regression of v3's official variance-only DT boundary."""
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT=ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1'
PREPARED=OUT/'DTPO-variance-v3-prepared'
launch=json.loads((PREPARED/'launch.json').read_bytes());env=launch['effective_environment']
sys.path[:0]=env['PYTHONPATH'].split(':');os.environ.update(env);os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
from verl import DataProto
from verl.trainer.ppo.ray_trainer import compute_advantage,AdvantageEstimator
from verl.utils import torch_functional as F
torch.set_num_threads(4)
folder=OUT/'live-native-update-input-v61'
receipts=[json.loads((folder/f'approved_DTPO_actual_update/rank{r}-capture.json').read_bytes()) for r in (0,1)]
data=DataProto.concat([DataProto.load_from_disk(r['input']) for r in receipts])
raw=data.batch['dt_token_advantages'].clone();mask=data.batch['response_mask'].bool()
preserved={k:data.batch[k].clone() for k in ('dt_q_estimates','dt_v_estimates','token_level_rewards','dt_grpo_multipliers')}
scale=torch.rsqrt(F.masked_var(raw,mask)+1e-8)
result=compute_advantage(data,adv_estimator=AdvantageEstimator.DELTATRACE,multi_turn=True)
actual=result.batch['advantages'];valid_nonzero=mask & raw.ne(0)
checks=dict(zero_and_padding_preserved=not bool(actual[raw.eq(0)].any()),
    no_sign_flips=torch.equal(actual[valid_nonzero].sign(),raw[valid_nonzero].sign()),
    variance_only_contract=torch.equal(actual,raw*scale),
    input_Q_V_rewards_multipliers_unchanged=all(torch.equal(data.batch[k],v) for k,v in preserved.items()),
    finite_original_epsilon_path=bool(torch.isfinite(actual).all()),
    observed_nonzero_mean_retained=bool(F.masked_mean(actual,mask)!=0))
assert all(checks.values()),checks
old_api=F.masked_whiten(raw,mask,shift_mean=False)
report=dict(unix=time.time(),status='v3_real_saved_batch_boundary_regression_passed',rows=len(data),
 checks=checks,raw_masked_mean=float(F.masked_mean(raw,mask)),raw_masked_std=float(F.masked_var(raw,mask).sqrt()),
 output_masked_mean=float(F.masked_mean(actual,mask)),output_masked_std=float(F.masked_var(actual,mask).sqrt()),
 scale=float(scale),native_shift_mean_false_changes_zero_values=bool(old_api[raw.eq(0)&mask].any()),
 source_bindings=launch['source_bindings'],configuration_changes=launch['configuration_changes'],
 PPO_reduction=launch['PPO_reduction'],new_model_calls=0,new_backward_calls=0,optimizer_calls=0,
 rootcause_repair_verified=False)
assert not torch.cuda.is_initialized()
dest=PREPARED/'interface-verification.json';assert not dest.exists();dest.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(receipt=str(dest),report=report)))

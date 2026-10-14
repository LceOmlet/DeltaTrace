"""Read actual advantage scale and unchanged native accumulation configuration.

Original masked statistics only; no alternate advantages, model or update.
"""
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT=ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1'
RUN=ROOT/'runs/textcraft-dt-grpo-multiplier-no-whiten-20261014-v1'
env=json.loads((RUN/'launch.json').read_bytes())['effective_environment']
sys.path[:0]=env['PYTHONPATH'].split(':');os.environ.update(env);os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
from verl import DataProto
from verl.utils import torch_functional as F
from verl.workers.actor.dp_actor import DataParallelPPOActor

torch.set_num_threads(4)
folder=OUT/'live-native-update-input-v61'
receipts=[json.loads((folder/f'approved_DTPO_actual_update/rank{r}-capture.json').read_bytes()) for r in (0,1)]
assert all(r['status']=='saved' and r['original_update_restored'] for r in receipts)
assert len({r['completed_step_before_update'] for r in receipts})==1
data=DataProto.concat([DataProto.load_from_disk(r['input']) for r in receipts])
adv=data.batch['advantages'];mask=data.batch['loss_mask'][:,-adv.shape[-1]:]
mean=F.masked_mean(adv,mask);variance=F.masked_var(adv,mask)
actor_path=Path(inspect.getsourcefile(DataParallelPPOActor))
assert hashlib.sha256(actor_path.read_bytes()).hexdigest()=='3a65e173300be82a7a9e056a96227c4f746eabc3be778ef41c8d50138d52ce6c'
configs=[r['config']['actor'] for r in receipts]
assert all(c['ppo_mini_batch_size']==32 and c['ppo_micro_batch_size_per_gpu']==4
           and not c['use_dynamic_bsz'] and c['loss_agg_mode']=='token-mean' for c in configs)
report=dict(unix=time.time(),scope=__doc__,completed_step_before_update=receipts[0]['completed_step_before_update'],
    rows=len(data),policy_tokens=int(mask.sum()),actual_masked_advantage_mean=float(mean),
    actual_native_masked_advantage_variance=float(variance),actual_native_masked_advantage_std=float(variance.sqrt()),
    actual_advantages_equal_DT_output=torch.equal(adv,data.batch['dt_token_advantages']),
    actor_source=dict(path=str(actor_path),SHA256=hashlib.sha256(actor_path.read_bytes()).hexdigest()),
    native_masked_statistics_source=dict(path=inspect.getsourcefile(F.masked_var),
        SHA256=hashlib.sha256(Path(inspect.getsourcefile(F.masked_var)).read_bytes()).hexdigest()),
    actual_configurations=[{k:c[k] for k in ('ppo_mini_batch_size','ppo_micro_batch_size_per_gpu',
        'use_dynamic_bsz','loss_agg_mode','ppo_epochs','entropy_coeff','kl_loss_coef')} for c in configs],
    native_accumulation_per_card=8,data_parallel_ranks=2,original_micro_mean_weight='1/8 * 1/2 = 1/16; unchanged',
    new_model_calls=0,new_optimizer_calls=0,training_changes=0,
    limits=['Removing whitening changes the coefficient scale and centering; changing raw-return to native group multipliers also changes the coefficients.',
        'A coefficient standard deviation does not determine the parameter-gradient norm or isolate either method change.',
        'This does not establish that the historical accelerating gradient anomaly has been repaired.'])
assert not torch.cuda.is_initialized()
path=folder/'actual-advantage-scale-and-native-normalization-v64.json';assert not path.exists()
path.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(receipt=str(path),report=report)))

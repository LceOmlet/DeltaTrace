"""Compare the same complete credit population through native DP2 and DP8.

Sentinel d values test transport only. The separate eight-GPU real-payload
check verifies actual DT. No model, optimizer, or quality evaluation is called.
"""
from copy import deepcopy
import json
import os
import sys

import torch
from prepare_campaign import ROOT, AUDIT

sys.path.insert(0, str(ROOT/'receipts/textcraft-weighted-reward-targets-20261015-v4/source'))
from test_multi_target_v5 import EventTransportWorker, events, artifact, native_training_data
from dt_training_batch import CREDIT_KEYS, compute_direct_target_credit, _v6_retained_policy_mask
from verl.trainer.ppo.ray_trainer import compute_advantage, AdvantageEstimator
from verl.utils.torch_functional import masked_var

assert os.environ['DT_CREDIT_MODE'] == 'reward_target_exponential_v5'
rows = []
for credits in [(-.1, 1.), (-1., 1.)]:
    items = [events(artifact(i), credits) for i in range(5)]
    repeated = deepcopy(items[0])
    repeated['retained_response_positions'] = [3, 4, 5, -1]
    data = native_training_data(items+[repeated],
        identities=[f'uid-{i}' for i in range(5)]+['uid-0'], multipliers=[sum(credits)]*6)
    mask = _v6_retained_policy_mask(data)
    outputs = []
    for world in [2, 8]:
        group = EventTransportWorker()
        group._workers = [None]*world
        raw = compute_direct_target_credit(data, group, eos_token_id=99, pad_token_id=0)
        exact = deepcopy(data)
        for key in CREDIT_KEYS:
            exact.batch[key] = raw.batch[key].clone()
        exact.batch['response_mask'] = mask
        used = compute_advantage(exact, adv_estimator=AdvantageEstimator.DELTATRACE)
        expected = torch.where(mask,
            raw.batch['dt_token_advantages']*torch.rsqrt(masked_var(raw.batch['dt_token_advantages'], mask)+1e-8), 0.)
        assert torch.equal(used.batch['advantages'], expected)
        assert not bool(used.batch['advantages'][~mask].any())
        outputs.append((raw, used))
        rows.append(dict(credits=credits, world_size=world,
            chunks=[len(c) for c in group.calls[0]],
            native_post_group_masked_variance_exact=True))
    for key in CREDIT_KEYS:
        assert torch.equal(outputs[0][0].batch[key], outputs[1][0].batch[key]), key
    assert torch.equal(outputs[0][1].batch['advantages'], outputs[1][1].batch['advantages'])

result = dict(status='passed', same_complete_training_batch_DP2_DP8_bitwise_equal=True,
    raw_QVA_bitwise_equal=True, actual_trainer_advantages_bitwise_equal=True,
    original_masked_var_applied_once_after_retained_scatter=True,
    RPC_padding_excluded_from_variance=True, sentinel_scope='transport only; native GPU evidence separate',
    model_calls=0, optimizer_updates=0, rows=rows)
(AUDIT/'two-eight-rank-credit-transport.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result))

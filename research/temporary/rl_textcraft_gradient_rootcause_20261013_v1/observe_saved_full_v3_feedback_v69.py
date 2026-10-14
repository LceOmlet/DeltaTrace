"""Regress the approved full v3 method at the saved bad state, via native owners.

The existing diagnostic owns checkpoint/RNG restore, fixed-seed probes and
unchanged native PPO updates. This adapter only prepares the approved v3 DT
advantages from original saved trajectories; no generation or quality score.
"""
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time

import torch
from verl.trainer.ppo.ray_trainer import compute_advantage, AdvantageEstimator
from verl.utils import torch_functional as verl_F
import dt_training_batch
import observe_common_update_feedback_v47 as native


def read_actual_DT_owner(worker):
    import hashlib, inspect, os, psutil, sys
    from pathlib import Path
    from verl.workers.fsdp_workers import ActorRolloutRefWorker
    result = dict(pid=os.getpid(), birth=psutil.Process().create_time(), rank=worker.rank,
                  actor_path=inspect.getsourcefile(type(worker.actor)),
                  worker_path=inspect.getsourcefile(ActorRolloutRefWorker), modules={})
    for name in ('reward_readout', 'qwen35_gdn_finite', 'qwen35_dense_finite_runner', 'qwen35_decoder_finite'):
        module = sys.modules.get(name)
        path = None if module is None else getattr(module, '__file__', None)
        result['modules'][name] = dict(path=path, SHA256=None if not path else hashlib.sha256(Path(path).read_bytes()).hexdigest())
    assert result['modules']['qwen35_gdn_finite']['SHA256'] == '3f51f5f7569a4d6b410e7acef89635a50f2e6ecadfa43eb94d1fd4fdd655dfd1'
    return result


def prepare_current_v3(group, arm, data):
    if not arm.get('prepare_current_v3'):
        return data
    started = time.time()
    bindings = {}
    for obj, expected in (
        (dt_training_batch, 'a4e1f70b42082972324199b328975fb87ec6600ab9a572d5ca2b2040bebc54f7'),
        (compute_advantage, 'b4b0187ef747d1ffb91dbfb8d1b8f914f93d01d90ff58d24e386f2cb1b7eb943'),
        (verl_F, '079a6d20696a861340687d6e61fb4162cc1d436837ce747ac9ef318b738c1701'),
    ):
        path = Path(inspect.getsourcefile(obj))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == expected
        bindings[str(path)] = digest
    assert not any(k in data.batch for k in dt_training_batch.CREDIT_KEYS), 'Never reuse old DT credit for the new multiplier.'
    preserved = {k: data.batch[k].clone() for k in ('token_level_rewards', 'old_log_probs', 'ref_log_prob')}
    stored_g = data.batch['dt_grpo_multipliers'].clone()
    view = data.select(batch_keys=['responses', 'token_level_rewards', 'response_mask', 'loss_mask'],
                       non_tensor_batch_keys=['uid', 'traj_uid'], deepcopy=True)
    view.batch = view.batch.clone(recurse=False)
    grouped = compute_advantage(view, adv_estimator=AdvantageEstimator.GRPO,
                                norm_adv_by_std_in_grpo=True, multi_turn=True)
    mask = data.batch['response_mask'].bool()
    g = grouped.batch['advantages'].gather(1, mask.long().argmax(-1, keepdim=True)).squeeze(1)
    assert torch.equal(g, stored_g)
    native.phase('computing_full_current_v3_DT_with_native_owner', arm=arm['name'])
    credits = dt_training_batch.compute_direct_target_credit(data, group, **native.SPEC['v3_token_ids'])
    data = data.union(credits)
    raw = data.batch['dt_token_advantages'].clone()
    data = compute_advantage(data, adv_estimator=AdvantageEstimator.DELTATRACE, multi_turn=True)
    mask = data.batch['response_mask'].bool()
    actual = data.batch['advantages']
    zero = g.eq(0)
    checks = dict(
        official_group_multiplier_unchanged=torch.equal(data.batch['dt_grpo_multipliers'], g),
        original_rewards_oldlogp_reference_unchanged=all(torch.equal(data.batch[k], v) for k, v in preserved.items()),
        zero_group_all_credit_zero=all(not bool(data.batch[k][zero].any()) for k in (*dt_training_batch.CREDIT_KEYS, 'advantages')),
        observation_padding_zero=not bool(actual[~mask].any()),
        variance_only_contract=torch.equal(actual, torch.where(mask, raw * torch.rsqrt(verl_F.masked_var(raw, mask)+1e-8), 0.0)),
        signs_preserved=torch.equal(actual[mask & raw.ne(0)].sign(), raw[mask & raw.ne(0)].sign()),
        finite_advantages=bool(torch.isfinite(actual).all()),
    )
    assert all(checks.values()), checks
    # The original probe binding verifies the entire trainable parameter digest
    # and scheduler80 again after DT. DT is not an optimizer update.
    unchanged = group.set_case(arm['name']+'_before')
    assert all(row['scheduler'] == 80 for row in unchanged)
    owners = native.rpc(group, read_actual_DT_owner)
    rng = native.rpc(group, native.restore_original_training_rng,
                     receipts=native.SPEC['captured_training_RNG_by_rank'])
    path = native.DEST / 'full-current-v3-actual-advantages.pkl'
    assert not path.exists()
    data.save_to_disk(str(path))
    record = dict(unix=time.time(), elapsed_seconds=time.time()-started, checks=checks,
                  source_bindings=bindings, actual_DT_owners=owners,
                  rows=len(data), zero_group_rows=int(zero.sum()),
                  raw_masked_std=float(verl_F.masked_var(raw, mask).sqrt()),
                  output_masked_mean=float(verl_F.masked_mean(actual, mask)),
                  output_masked_std=float(verl_F.masked_var(actual, mask).sqrt()),
                  original_RNG_restored_after_DT=rng,
                  payload=str(path), payload_SHA256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  native_DT_request_rows=native.SPEC['prepared_native_request_rows'],
                  new_generation_calls=0, GRPO_model_reruns=0, validation_score_calls=0,
                  rootcause_repair_verified=False)
    (native.DEST / 'full-current-v3-DT-preparation.json').write_text(json.dumps(record, indent=2)+'\n')
    native.STATE['DT_calls'] += 1
    native.STATE['approved_v3_preparation'] = record
    return data


native.prepare_native_update = prepare_current_v3
if __name__ == '__main__':
    native.main()

"""Reuse the native actor diagnostic on literal saved inputs, then full B8/32k.

This is a test driver. The existing worker, VERL update, optimizer, loss and
vLLM synchronization own execution. Only the original owner padding assertion
is used for the same-weight comparison; there is no whole-update tolerance.
"""
import ast
import hashlib
import json
import os
from pathlib import Path
import time

import ray
import torch
from omegaconf import OmegaConf
from verl import DataProto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.utils.torch_functional import masked_mean
from verify_owner_response_padding import ObservedWorker

OUT = Path(os.environ['PADDING_DIAGNOSTIC_DIR'])


def native_batch(rows, pad_id, *, full=False):
    """Place literal original IDs; synthetic repetition is capacity-only."""
    ids = torch.full((8, 32768), pad_id, dtype=torch.long)
    mask = torch.zeros_like(ids)
    actions = torch.zeros_like(ids)
    for index, row in enumerate(rows):
        literal = torch.cat((row['prompt'], row['actions']))
        if full:
            ids[index] = literal.repeat((32768 + len(literal) - 1) // len(literal))[:32768]
            mask[index] = 1
            actions[index, 1:] = 1
        else:
            assert len(literal) <= 32768
            ids[index, :len(literal)] = literal
            mask[index, :len(literal)] = 1
            actions[index, len(row['prompt']):len(literal)] = 1
    result = DataProto.from_single_dict(dict(input_ids=ids, attention_mask=mask,
        loss_mask=actions, position_ids=(mask.cumsum(-1) - 1).clamp_min(0), responses=ids[:, 1:]))
    result.meta_info.update(multi_turn=True, global_token_num=mask.sum(-1).tolist())
    return result


if __name__ == '__main__':
    source = json.loads((OUT/'source.json').read_bytes())
    cfg = OmegaConf.load(Path(os.environ['VERL_ROOT'])/'verl/trainer/config/ppo_trainer.yaml')
    for key, value in json.loads(Path(source['formal_launch']).read_bytes())['options'].items():
        OmegaConf.update(cfg, key.lstrip('+'), value, force_add=True)
    cfg.actor_rollout_ref.actor.optim.total_training_steps = source['original_total_training_steps']
    (OUT/'effective-config.yaml').write_text(OmegaConf.to_yaml(cfg))
    assertion_path = Path(source['official_padding_test'])
    tree = ast.parse(assertion_path.read_text())
    fn = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'test_hf_casual_models')
    assertion = next(x for x in ast.walk(fn) if isinstance(x, ast.Expr) and isinstance(x.value, ast.Call)
                     and ast.unparse(x.value.func) == 'torch.testing.assert_close')
    check = compile(ast.fix_missing_locations(ast.Module(body=[assertion], type_ignores=[])), str(assertion_path), 'exec')
    rows, inputs = [], []
    for rank in (0, 1):
        path = Path(source['original_request_artifacts'])/f'actual-requests-rank{rank}.pt'
        requests = torch.load(path, map_location='cpu', weights_only=False)['requests']
        # Select the longest four literal histories below the observed AppWorld
        # padded width. This selection is a model fixture, not task sampling.
        candidates = [(index, row) for index, row in enumerate(requests)
                      if len(row['prompt']) + len(row['actions']) <= 25383]
        chosen = sorted(candidates, key=lambda pair: len(pair[1]['prompt']) + len(pair[1]['actions']), reverse=True)[:4]
        assert len(chosen) == 4
        rows.extend(row for _, row in chosen)
        inputs.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            rows=[dict(index=index, prompt_tokens=len(row['prompt']), action_tokens=len(row['actions'])) for index, row in chosen]))
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(os.environ['MODEL_PATH'], local_files_only=True)
    data = native_batch(rows, tokenizer.pad_token_id)
    data.meta_info['temperature'] = cfg.actor_rollout_ref.rollout.temperature
    state = dict(scope=__doc__, source=source, inputs=inputs, readouts=[], phase='initializing',
                 official_padding_assertion=ast.unparse(assertion), official_test_sha256=hashlib.sha256(assertion_path.read_bytes()).hexdigest())
    def record(phase, **values):
        state.update(phase=phase, observed_unix=time.time(), **values)
        (OUT/'result.json').write_text(json.dumps(state, indent=2)+'\n')
        print(json.dumps(dict(phase=phase, **values)), flush=True)
    ray.init(num_cpus=8, include_dashboard=False)
    try:
        class_dict = {'actor_rollout': RayClassWithInitArgs(
            ObservedWorker, config=cfg.actor_rollout_ref, role='actor_rollout')}
        colocated = create_colocated_worker_cls(class_dict=class_dict)
        owner_group = RayWorkerGroup(
            resource_pool=RayResourcePool([2], use_gpu=True, max_colocate_count=1),
            ray_cls_with_init=colocated)
        group = owner_group.spawn(prefix_set=class_dict.keys())['actor_rollout']
        record('native_worker_init_start', rollout_mode=cfg.actor_rollout_ref.rollout.mode)
        group.init_model()
        manager = None
        if cfg.actor_rollout_ref.rollout.mode == 'async':
            from verl.workers.rollout.async_server import AsyncLLMServerManager
            record('native_async_server_init_start')
            manager = AsyncLLMServerManager(config=cfg.actor_rollout_ref, worker_group=group)
            manager.wake_up()
            manager.sleep()
            record('native_async_server_sleep_complete')
        scores = {}
        for mode in ('reference', 'candidate', 'reference', 'candidate'):
            state['selected_owners'] = group.select_padding_owner(mode)
            record(mode+'_logprob_start')
            tick = time.perf_counter()
            scores[mode] = group.compute_log_prob(data).batch['old_log_probs']
            assert scores[mode].shape == (8, 32767) and torch.isfinite(scores[mode]).all()
            state['readouts'].append(dict(mode=mode, seconds=time.perf_counter()-tick))
            record(mode+'_logprob_complete')
        try:
            exec(check, dict(torch=torch, masked_mean=masked_mean, log_probs=scores['candidate'],
                 origin_log_probs=scores['reference'], attention_mask=data.batch['attention_mask'], response_length=32767))
        except AssertionError as error:
            record('original_owner_padding_assertion_failed', padding_assertion_passed=False, error=str(error))
            raise
        policy_mask = data.batch['loss_mask'][:, 1:].bool()
        delta = (scores['candidate']-scores['reference'])[policy_mask]
        record('original_owner_padding_assertion_passed', padding_assertion_passed=True,
               policy_token_max_abs=float(delta.abs().max()), policy_token_mean_abs=float(delta.abs().mean()),
               policy_tokens=int(policy_mask.sum()), mean_reference=float(masked_mean(scores['reference'],data.batch['attention_mask'][:, :-1])),
               mean_candidate=float(masked_mean(scores['candidate'],data.batch['attention_mask'][:, :-1])))
        payload = dict(batch=data.batch.to_dict(), reference=scores['reference'], candidate=scores['candidate'])
        payload = torch.utils._pytree.tree_map(lambda value: value.detach().cpu().clone() if torch.is_tensor(value) else value, payload)
        torch.save(payload, OUT/'same-input-readouts.pt')
        for label, batch in [('literal_padded', data), ('full_B8_32768', native_batch(rows, tokenizer.pad_token_id, full=True))]:
            batch.meta_info['temperature'] = cfg.actor_rollout_ref.rollout.temperature
            if label == 'literal_padded':
                old = scores['candidate']
            else:
                record(label+'_old_logprob_start')
                old = group.compute_log_prob(batch).batch['old_log_probs']
            batch.batch['old_log_probs'] = old
            batch.batch['ref_log_prob'] = old
            batch.batch['advantages'] = torch.linspace(-.2, .3, 32767).expand(8, -1).clone()
            record(label+'_native_update_start')
            tick = time.perf_counter()
            metrics = group.update_actor(batch).meta_info['metrics']
            state.setdefault('updates', []).append(dict(fixture=label, seconds=time.perf_counter()-tick, metrics=metrics))
            record(label+'_native_update_complete')
        if manager is not None:
            record('post_update_native_async_sync_start')
            manager.wake_up()
            manager.sleep()
        workers = group.finish_padding_observation()
        record('complete', workers=workers, post_update_native_async_sync_sleep=manager is not None)
    finally:
        ray.shutdown()

"""Check composed AppWorld DT capacity through the existing VERL worker RPC.

One exact32768 B8 and the original failed B8, split by native DP into B4/card.
The saved-case capacity builder and original padding utilities own the inputs.
No worker subclass, forward, attribution, reward, PPO or tolerance is copied.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time

import numpy as np
import psutil
import ray
import torch
from omegaconf import OmegaConf

from check_capacity_lifetime import capacity_row
from inspect_extreme_endpoint import check_imports
from reward_readout import DirectActionTargetReadout
from verl import DataProto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.utils.model import compute_position_id_with_mask
from verl.utils.torch_functional import pad_2d_list_to_length, postprocess_data
from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker
from verl.workers.rollout.async_server import AsyncLLMServerManager
from transformers import AutoTokenizer


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def native_data(rows, tokenizer):
    """Representation only: retain exact IDs/masks, calling original padding."""
    prepared = [DirectActionTargetReadout._prepare_row(row, i) for i, row in enumerate(rows)]
    prompt_width = max(item['prompt_length'] for item in prepared)
    prompts, masks = zip(*(postprocess_data(
        item['selected'][:item['prompt_length']][None, :],
        torch.ones((1, item['prompt_length']), dtype=torch.long),
        prompt_width, tokenizer.pad_token_id, left_pad=True, truncation='error') for item in prepared))
    responses = pad_2d_list_to_length(
        [item['selected'][item['prompt_length']:].tolist() for item in prepared], tokenizer.pad_token_id)
    response_mask = torch.arange(responses.shape[1])[None, :] < torch.tensor(
        [item['suffix_positions'].numel() for item in prepared])[:, None]
    attention = torch.cat((torch.cat(masks), response_mask.long()), -1)
    policy = pad_2d_list_to_length([item['valid_policy'].tolist() for item in prepared], 0).bool()
    target = pad_2d_list_to_length([item['valid_target'].tolist() for item in prepared], 0).bool()
    data = DataProto.from_dict(tensors=dict(
        input_ids=torch.cat((torch.cat(prompts), responses), -1), responses=responses,
        attention_mask=attention, position_ids=compute_position_id_with_mask(attention),
        policy_mask=policy, target_mask=target,
        dt_direct_reward=torch.tensor([item['reward'] for item in prepared], dtype=torch.float64)),
        non_tensors=dict(traj_uid=np.array([str(row['traj_uid']) for row in rows], dtype=object)))
    data.meta_info.update(eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
                          dt_target_semantics='native_joint_action_target')
    for actual, expected in enumerate(prepared):
        # Compare the transport with the same owner's row preparation.
        row = {key: value[actual] for key, value in data.batch.items()}
        restored = DirectActionTargetReadout._prepare_row(row, actual)
        for key in ('selected', 'valid_policy', 'valid_target'):
            assert torch.equal(restored[key], expected[key])
    return data


def inspect_worker(worker):
    import importlib
    owners = [owner for owner in worker.worker_dict.values() if getattr(owner, '_is_actor', False)]
    assert len(owners) == 1
    owner = owners[0]
    modules = {}
    for name in ('qwen35_dense_finite_runner', 'qwen35_gdn_finite', 'deltatrace_rollout', 'verl.workers.fsdp_workers'):
        module = importlib.import_module(name)
        modules[name] = dict(path=module.__file__, resolved=str(Path(module.__file__).resolve()), sha256=digest(module.__file__))
    assert modules['qwen35_dense_finite_runner']['sha256'] == '7d6f57f61ecde7ce3506b8ef58d61268859fc04357c35de99a1892928c4ba7a8'
    assert modules['qwen35_gdn_finite']['sha256'] == '3f51f5f7569a4d6b410e7acef89635a50f2e6ecadfa43eb94d1fd4fdd655dfd1'
    assert (owner.config.model.lora_rank, owner.config.model.lora_alpha) == (8, 16)
    assert owner.config.actor.ppo_micro_batch_size_per_gpu == 4
    return dict(pid=os.getpid(), birth=psutil.Process().create_time(), rank=owner.rank, modules=modules,
        allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved(),
        PSS=psutil.Process().memory_full_info().pss,
        original_readout_report=(owner._deltatrace_producer.direct_readout.last_report
                                if hasattr(owner, '_deltatrace_producer') else None))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--failed-batch', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    source = json.loads(args.source.read_bytes())
    assert digest(args.source) == '58209daa0fccfea4b70645465e96ea5d203f9d309187b7a64b405cbd9fd47da0'
    check_imports(source)
    cfg = OmegaConf.load(Path(source['verl_root'])/'verl/trainer/config/ppo_trainer.yaml')
    for key, value in source['startup_options'].items():
        OmegaConf.update(cfg, key.lstrip('+'), value, force_add=True)
    cfg.actor_rollout_ref.actor.optim.total_training_steps = cfg.trainer.total_training_steps
    tokenizer = AutoTokenizer.from_pretrained(cfg.actor_rollout_ref.model.path, local_files_only=True)
    saved = [torch.load(args.failed_batch/f'rank{rank}-failed-batch.pt', map_location='cpu', weights_only=False) for rank in (0, 1)]
    rows = [row for item in saved for row in item['rows']]
    assert len(rows) == 8
    expanded = [capacity_row(row, DirectActionTargetReadout._prepare_row) for row in rows]
    inputs = [('exact32768', native_data([item[0] for item in expanded], tokenizer)),
              ('original_failed_B8', native_data(rows, tokenizer))]
    assert inputs[0][1].batch['attention_mask'].sum(-1).tolist() == [32768]*8
    preparation = dict(unix=time.time(), script_sha256=digest(__file__), source_sha256=digest(args.source),
        inputs=[dict(name=name, rows=len(data), causal_lengths=data.batch['attention_mask'].sum(-1).tolist(),
                     shapes={key:list(value.shape) for key,value in data.batch.items()}) for name,data in inputs],
        source_artifacts=[dict(path=str(args.failed_batch/f'rank{rank}-failed-batch.pt'),
                               sha256=digest(args.failed_batch/f'rank{rank}-failed-batch.pt')) for rank in (0, 1)],
        exact_ID_mask_transport_verified=True, CUDA_initialized=torch.cuda.is_initialized())
    (args.output/'prepared-inputs.json').write_text(json.dumps(preparation, indent=2)+'\n')
    if args.prepare_only:
        assert not preparation['CUDA_initialized']
        print(json.dumps(preparation))
        raise SystemExit(0)
    stop = threading.Event()
    phase = ['initializing']
    def emit(event, **values):
        with (args.output/'driver-phases.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(unix=time.time(), event=event, phase=phase[0], **values))+'\n')
    def sample_physical():
        with (args.output/'physical-mx-smi.jsonl').open('a', buffering=1) as stream:
            while not stop.is_set():
                result = subprocess.run(['mx-smi'], capture_output=True, text=True, timeout=20)
                stream.write(json.dumps(dict(unix=time.time(), phase=phase[0], stdout=result.stdout,
                    stderr=result.stderr, returncode=result.returncode))+'\n')
                stop.wait(2)
    sampler = threading.Thread(target=sample_physical, daemon=True)
    sampler.start()
    ray.init(num_cpus=8, include_dashboard=False)
    try:
        pool = RayResourcePool([2], use_gpu=True, max_colocate_count=1)
        actor = RayClassWithInitArgs(ray.remote(AsyncActorRolloutRefWorker), cfg.actor_rollout_ref, 'actor_rollout')
        owner_group = RayWorkerGroup(resource_pool=pool,
            ray_cls_with_init=create_colocated_worker_cls(class_dict={'actor_rollout': actor}),
            device_name=cfg.trainer.device)
        group = owner_group.spawn(prefix_set={'actor_rollout'})['actor_rollout']
        group.init_model()
        async_manager = AsyncLLMServerManager(config=cfg.actor_rollout_ref, worker_group=group)
        async_manager.wake_up()
        async_manager.sleep()
        emit('original_async_vllm_initialized_and_asleep', workers=ray.get([
            worker.execute_with_func_generator.remote(inspect_worker) for worker in group.workers]))
        for name, data in inputs:
            phase[0] = name
            emit('native_DT_RPC_begin', causal_lengths=data.batch['attention_mask'].sum(-1).tolist())
            started = time.perf_counter()
            result = group.compute_dt_token_advantages(data)
            seconds = time.perf_counter()-started
            assert all(torch.isfinite(value).all() for value in result.batch.values())
            torch.save(dict(result=result.batch.to_dict(), input=data.batch.to_dict(),
                            identities=data.non_tensor_batch, meta_info=data.meta_info), args.output/f'{name}.pt')
            emit('native_DT_RPC_complete', seconds=seconds, rows=len(result), workers=ray.get([
                worker.execute_with_func_generator.remote(inspect_worker) for worker in group.workers]))
        emit('complete', rollout_requests=0, PPO_updates=0, checkpoint_restore=False,
             numerical_tolerance_tests='Previously accepted unchanged FA/FLA receipts; this is capacity/finite regression only')
        (args.output/'completed.json').write_text(json.dumps(dict(status='complete_capacity_and_original_failed_B8',
            script_sha256=digest(__file__), source_sha256=digest(args.source), operations=dict(DT_RPC=2,
                per_rank_B4_DT=2, PPO_updates=0, rollout_requests=0, checkpoint_restore=False)), indent=2)+'\n')
    finally:
        stop.set()
        sampler.join(timeout=25)
        ray.shutdown()

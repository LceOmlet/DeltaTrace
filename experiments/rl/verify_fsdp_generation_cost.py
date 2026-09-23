"""Measure the pinned VERL generation owner on a fixed 32k capacity input.

Uses a synthetic long prefix and deterministic decoding for a capacity/cost
comparison, not task training or success-rate evaluation. The default HF test
compares the public FSDP reshard setter. The vLLM test uses the worker's native
rollout, LoRA synchronization, and sleep/wake lifecycle.
"""
import argparse
import hashlib
import inspect
import json
import os
import statistics
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from torch.distributed.fsdp import FSDPModule
from verl import DataProto
from verl.workers.fsdp_workers import ActorRolloutRefWorker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--backend', choices=('hf', 'vllm'), default='hf',
                        help='Use the pinned VERL owner for this generation backend.')
    parser.add_argument('--compare-active-mask', action='store_true',
                        help='Compare original all-row generation with skipping finished rows.')
    parser.add_argument('--handoff-only', action='store_true',
                        help='Inspect the native LoRA and allocator handoff without generation.')
    args = parser.parse_args()
    if args.backend == 'vllm' and args.compare_active_mask:
        parser.error('active-mask comparison currently exercises the HF owner')
    result = dict(scope=__doc__, prompt_tokens=32256, response_cap=512,
                  context_cap=32768, rollout_batch=4, cases=[],
                  backend=args.backend,
                  allocator_config=os.environ.get('PYTORCH_CUDA_ALLOC_CONF'))
    if args.compare_active_mask:
        result['scope'] = ('Same pinned HF/FSDP2 owner, real Qwen weights, fixed 32k capacity fixture; '
                           'compare all-row generation with collector active-mask transport. '
                           'No claim about task success or complete iteration throughput.')
    try:
        torch.manual_seed(2026)
        torch.use_deterministic_algorithms(True)
        cfg = OmegaConf.load(Path(os.environ['VERL_ROOT'])/'verl/trainer/config/ppo_trainer.yaml')
        c = cfg.actor_rollout_ref
        c.model.path = os.environ['MODEL_PATH']
        c.model.lora_rank, c.model.lora_alpha = 1, 2
        c.model.trust_remote_code = True
        c.actor.strategy = 'fsdp2'
        c.actor.ppo_mini_batch_size = 4
        c.actor.ppo_micro_batch_size_per_gpu = 1
        c.actor.use_torch_compile = False
        c.actor.optim.total_training_steps = 3
        c.actor.fsdp_config.model_dtype = 'bfloat16'
        c.actor.fsdp_config.reshard_after_forward = True
        c.rollout.name = args.backend
        c.rollout.n = 1
        c.rollout.tensor_model_parallel_size = 1
        c.rollout.micro_batch_size = 4
        c.rollout.prompt_length = result['prompt_tokens']
        c.rollout.response_length = result['response_cap']
        if args.backend == 'vllm':
            # Use the owner's existing single-GPU actor/rollout memory lifecycle
            # and native LoRA synchronization, without modifying either engine.
            c.actor.fsdp_config.param_offload = True
            c.actor.fsdp_config.optimizer_offload = True
            c.rollout.load_format = 'safetensors'
            c.rollout.max_model_len = result['context_cap']
            c.rollout.max_num_seqs = result['rollout_batch']
            c.rollout.max_num_batched_tokens = result['context_cap']
            c.rollout.gpu_memory_utilization = 0.75
            # These tasks use the checkpoint's text model only. Use the
            # existing VERL engine-kwargs boundary and vLLM text-only mode.
            c.rollout.engine_kwargs.vllm.limit_mm_per_prompt = {'image': 0, 'video': 0}
            result['scope'] = ('Pinned VERL vLLM backend readiness and cost on the same '
                               '32k fixture; includes owner LoRA sync and sleep/wake. '
                               'Not a DT/PPO integration or numerical-parity result.')
            result['owner_config'] = OmegaConf.to_container(c, resolve=True)
        result['phase'] = 'owner_model_and_rollout_init'
        args.output.write_text(json.dumps(result, indent=2)+'\n')
        print('GENERATION_COST_PHASE', result['phase'], flush=True)
        worker = ActorRolloutRefWorker(c, 'actor_rollout')
        worker.init_model()
        if args.backend == 'vllm':
            from vllm_metax.device_allocator.cumem import CuMemAllocator
            allocator = CuMemAllocator.get_instance()
            executor = worker.rollout.inference_engine.llm_engine.model_executor
            native_worker = executor.driver_worker.worker
            def native_memory_state():
                torch.cuda.synchronize()
                free, total = torch.cuda.mem_get_info()
                return dict(free_gib=free/2**30, device_used_gib=(total-free)/2**30,
                    torch_allocated_gib=torch.cuda.memory_allocated()/2**30,
                    pool_bytes=allocator.get_current_usage(),
                    pool_allocations=len(allocator.pointer_to_data),
                    backed_up_allocations=sum(d.cpu_backup_tensor is not None for d in allocator.pointer_to_data.values()))
            result['native_sleep_owner'] = dict(worker_type=str(type(native_worker)),
                sleep_source=inspect.getsourcefile(native_worker.sleep),
                pool_source=inspect.getsourcefile(native_worker._maybe_get_memory_pool_context),
                sleeping_after_init=native_memory_state())
            # Inspect the actual native loader: successful add_lora alone can
            # silently leave every layer reset if HF/vLLM prefixes differ.
            with worker.rollout_sharding_manager:
                result['native_sleep_owner']['awake'] = native_memory_state()
                manager = worker.rollout_sharding_manager.model_runner.lora_manager._adapter_manager
                adapters = manager.list_adapters()
                assert len(adapters) == 1, adapters.keys()
                adapter = next(iter(adapters.values()))
                unused = sorted(set(adapter.loras) - set(manager.modules))
                result['native_lora_binding'] = dict(
                    actor_model_type=worker.actor_module_fsdp.config.model_type,
                    rollout_model_type=worker.rollout_sharding_manager.model_config.model_type,
                    loaded_layers=len(adapter.loras), available_layers=len(manager.modules),
                    unused_layers=unused, layer_examples=sorted(adapter.loras)[:5],
                )
                args.output.write_text(json.dumps(result, indent=2)+'\n')
                assert adapter.loras and not unused, result['native_lora_binding']
            result['native_sleep_owner']['after_exit'] = native_memory_state()
            print('NATIVE_SLEEP_OWNER', result['native_sleep_owner'], flush=True)
            if args.handoff_only:
                result.update(status='recorded', scope='Native LoRA binding and sleep/wake diagnostic; no DT/PPO or generation claim')
                return
        result['generation_eos_token_id'] = worker.generation_config.eos_token_id
        fixture = json.loads((Path(os.environ['DT_RUNTIME_ROOT'])/'receipts/rollout-fixtures.json').read_text())
        row = fixture['tasks']['Sokoban']['rows'][-1]
        width = len(row['responses'])
        original = torch.tensor(row['input_ids'][:-width])[torch.tensor(row['attention_mask'][:-width]).bool()]
        filler_id = worker.tokenizer.encode(' context', add_special_tokens=False)[0]
        fill = result['prompt_tokens']-original.numel()
        assert fill > 0
        ids = torch.cat((torch.full((fill,), filler_id), original)).repeat(4, 1)
        mask = torch.ones_like(ids)
        prompts = DataProto(batch=TensorDict(dict(input_ids=ids, attention_mask=mask,
            position_ids=mask.cumsum(-1)-1), batch_size=[4]), meta_info={'do_sample': False})
        modules = [m for m in worker.actor_module_fsdp.modules() if isinstance(m, FSDPModule)]
        assert modules
        reference = None
        # Default FSDP2 root does not reshard; change only descendant settings.
        cases = [(True, None), (False, None)] if not args.compare_active_mask else [
            (False, None), (False, [True, False, True, False]), (False, [False]*4)]
        if args.backend == 'vllm':
            cases = [(False, None)]
        for enabled, active in cases:
            for m in modules[1:]:
                m.set_reshard_after_forward(enabled, recurse=False)
            case = dict(reshard_after_forward=enabled, runs=[])
            owner_active_reference = None
            if active is not None and any(active):
                # Same-shape owner comparison: ordinary HF on these exact rows,
                # with no mask extension. Cross-batch long greedy continuations
                # can diverge due to the owner's own BF16 rounding; record that
                # difference, rather than inventing a bitwise cross-batch gate.
                prompts.non_tensor_batch.pop('rollout_active_mask', None)
                selected_prompts = prompts.select_idxs(np.flatnonzero(active))
                owner_active_reference = worker.generate_sequences(selected_prompts).batch['responses'].cpu()
                case['original_owner_selected_rows'] = int(sum(active))
                case['original_owner_changed_tokens_vs_batch4'] = int(
                    (owner_active_reference != reference[torch.tensor(active)]).sum())
            if active is None:
                prompts.non_tensor_batch.pop('rollout_active_mask', None)
            else:
                prompts.non_tensor_batch['rollout_active_mask'] = np.array(active, dtype=bool)
            if args.compare_active_mask:
                case['active_rows'] = active or [True]*4
            result['cases'].append(case)
            for repeat in range(2):
                for m in reversed(modules):
                    m.reshard()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                result['phase'] = f'generation_repeat_{repeat}'
                args.output.write_text(json.dumps(result, indent=2)+'\n')
                print('GENERATION_COST_PHASE', result['phase'], flush=True)
                output = worker.generate_sequences(prompts)
                torch.cuda.synchronize()
                elapsed = time.perf_counter()-started
                responses = output.batch['responses'].cpu()
                effective = output.batch['attention_mask'][:, -result['response_cap']:].sum(-1).tolist()
                if reference is None:
                    reference = responses.clone()
                elif args.backend == 'vllm':
                    case['repeat_changed_tokens'] = int((responses != reference).sum())
                elif owner_active_reference is not None:
                    torch.testing.assert_close(responses[torch.tensor(active)], owner_active_reference, rtol=0, atol=0)
                else:
                    selected = torch.tensor(active if active is not None else [True]*4)
                    torch.testing.assert_close(responses[selected], reference[selected], rtol=0, atol=0)
                if active is not None:
                    assert responses[~torch.tensor(active)].eq(worker.tokenizer.pad_token_id).all()
                assert output.batch['input_ids'].shape == (4, 32768)
                rec = dict(repeat=repeat, warmup=repeat == 0, seconds=elapsed,
                    actual_response_lengths=effective,
                    peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                    peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                    response_sha256=hashlib.sha256(responses.numpy().tobytes()).hexdigest())
                case['runs'].append(rec)
                args.output.write_text(json.dumps(result, indent=2)+'\n')
                print('GENERATION_COST', enabled, rec, flush=True)
            case['warm_seconds'] = statistics.median(r['seconds'] for r in case['runs'][1:])
        key = 'warm_ratio_active2_over_all4' if args.compare_active_mask else 'warm_ratio_false_over_true'
        if len(result['cases']) > 1:
            result[key] = result['cases'][1]['warm_seconds']/result['cases'][0]['warm_seconds']
        if args.backend == 'hf':
            result['active_generated_ids_exactly_equal_to_same_shape_owner' if args.compare_active_mask else 'all_generated_ids_exactly_equal'] = True
        result['status'] = 'passed'
    except Exception as exc:
        result.update(status='failed', error=str(exc), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        args.output.write_text(json.dumps(result, indent=2)+'\n')
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    if result['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()

"""Measure the pinned VERL/vLLM worker with identical recorded task text.

The optional owner text dump removes special tokens; recorded RequestOutput
prompt IDs can instead be replayed directly. Neither is a regeneration of
identical on-policy trajectories or an evaluation score.
Forced equal output lengths isolate throughput from sampled stopping lengths.
"""
import argparse
import hashlib
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from verl import DataProto
from verl.workers.fsdp_workers import ActorRolloutRefWorker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-num-seqs', type=int, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--rollouts', type=Path)
    source.add_argument('--request-json', type=Path, help='Exact prompt_ids recorded from original RequestOutput.')
    parser.add_argument('--requests', type=int, default=32)
    parser.add_argument('--enforce-eager', action=argparse.BooleanOptionalAction, default=True,
                        help='Forward the existing owner setting; disabling it permits native vLLM graph execution.')
    parser.add_argument('--greedy', action='store_true', help='Use the original owner do_sample=False path.')
    parser.add_argument('--decode-tokens', type=int, default=128)
    parser.add_argument('--mamba-cache-mode', choices=('none', 'align', 'all'), default=None,
                        help='Pass the native vLLM engine option through the existing engine_kwargs interface.')
    parser.add_argument('--enable-prefix-caching', action=argparse.BooleanOptionalAction, default=None,
                        help='Pass the native vLLM cache toggle; omitted preserves the owner default.')
    parser.add_argument('--profile-dir', type=Path,
                        help='Use the original vLLM profiler for a bounded extra call; no training kernel changes.')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = dict(scope=__doc__, max_num_seqs=args.max_num_seqs, actor_minibatch=4,
                  model_context_cap=32768, enforce_eager=args.enforce_eager,
                  greedy=args.greedy, stages=[])
    started = time.perf_counter()

    def record(phase, **data):
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        row = dict(phase=phase, elapsed=time.perf_counter()-started,
                   device_used_bytes=total-free, **data)
        result['stages'].append(row)
        print(json.dumps(row), flush=True)
        args.output.write_text(json.dumps(result, indent=2)+'\n')

    try:
        torch.manual_seed(2026)
        cfg = OmegaConf.load(Path(os.environ['VERL_ROOT'])/'verl/trainer/config/ppo_trainer.yaml')
        c = cfg.actor_rollout_ref
        c.model.path = os.environ['MODEL_PATH']
        c.model.lora_rank, c.model.lora_alpha = 1, 2
        c.model.trust_remote_code = True
        c.model.enable_activation_offload = True
        c.actor.strategy = 'fsdp2'
        c.actor.ppo_mini_batch_size = 4
        c.actor.ppo_micro_batch_size_per_gpu = 4
        c.actor.use_dynamic_bsz = False
        c.actor.ppo_max_token_len_per_gpu = 32768
        c.actor.use_torch_compile = False
        c.actor.optim.total_training_steps = 3
        c.actor.fsdp_config.model_dtype = 'bfloat16'
        c.actor.fsdp_config.optimizer_offload = True
        c.actor.fsdp_config.reshard_after_forward = True
        c.actor.fsdp_config.offload_policy = True
        c.actor.fsdp_config.param_offload = True
        c.rollout.name = 'vllm'
        c.rollout.n = 1
        c.rollout.tensor_model_parallel_size = 1
        c.rollout.log_prob_micro_batch_size_per_gpu = 4
        c.rollout.micro_batch_size = 4
        c.rollout.prompt_length = 31744
        c.rollout.response_length = 1024
        c.rollout.load_format = 'safetensors'
        c.rollout.max_model_len = 32768
        c.rollout.max_num_seqs = args.max_num_seqs
        c.rollout.max_num_batched_tokens = 32768
        c.rollout.gpu_memory_utilization = 0.75
        c.rollout.enforce_eager = args.enforce_eager
        if not args.enforce_eager:
            # The pinned VERL constructor requires this original option pair.
            # free_cache_engine only controls its legacy vLLM 0.5/0.6 calls;
            # the current sharding manager still owns native sleep/wake.
            c.rollout.free_cache_engine = False
        c.rollout.engine_kwargs.vllm.limit_mm_per_prompt = {'image': 0, 'video': 0}
        if args.mamba_cache_mode is not None:
            c.rollout.engine_kwargs.vllm.mamba_cache_mode = args.mamba_cache_mode
        if args.enable_prefix_caching is not None:
            c.rollout.engine_kwargs.vllm.enable_prefix_caching = args.enable_prefix_caching
        if args.profile_dir:
            c.rollout.engine_kwargs.vllm.profiler_config = dict(
                profiler='torch', torch_profiler_dir=str(args.profile_dir.resolve()),
                torch_profiler_with_stack=False, torch_profiler_record_shapes=True,
                ignore_frontend=True, max_iterations=16)
        worker = ActorRolloutRefWorker(c, 'actor_rollout')
        worker.init_model()
        engine_config = worker.rollout.inference_engine.llm_engine.vllm_config
        result['native_cache_config'] = engine_config.cache_config.metrics_info()
        record('owner_init')

        # Read the owner's real RequestOutput counters; do not reconstruct
        # cache hits or call this interval pure decode (it includes prefill).
        engine_calls = []
        original_generate = worker.rollout.inference_engine.generate
        profile_next = False

        def measured_generate(*a, **kw):
            if profile_next:
                worker.rollout.inference_engine.start_profile()
            tick = time.perf_counter()
            outputs = original_generate(*a, **kw)
            torch.cuda.synchronize()
            seconds = time.perf_counter()-tick
            if profile_next:
                worker.rollout.inference_engine.stop_profile()
            engine_calls.append(dict(seconds=seconds, profiled=profile_next, requests=[dict(
                request_id=out.request_id,
                prompt_ids_sha256=hashlib.sha256(json.dumps(out.prompt_token_ids).encode()).hexdigest(),
                prompt_tokens=len(out.prompt_token_ids),
                cached_tokens=out.num_cached_tokens,
                generated_tokens=sum(len(sample.token_ids) for sample in out.outputs),
                first_tokens=[dict(token_id=sample.token_ids[0],
                    logprob=sample.logprobs[0][sample.token_ids[0]].logprob)
                    for sample in out.outputs if sample.token_ids],
            ) for out in outputs]))
            return outputs

        worker.rollout.inference_engine.generate = measured_generate

        if args.request_json:
            lines = json.loads(args.request_json.read_text())
            chosen = [i % len(lines) for i in range(args.requests)]
            prompts = [lines[i]['prompt_ids'] for i in chosen]
            result['input_protocol'] = 'Exact recorded prompt IDs, repeated to requested concurrency; no text reconstruction.'
        else:
            lines = [json.loads(line) for line in args.rollouts.read_text().splitlines()]
            texts = sorted((row['input'] for row in lines), key=len)
            chosen = np.linspace(0, len(texts)-1, args.requests).astype(int).tolist()
            prompts = [worker.tokenizer.encode(texts[i], add_special_tokens=False) for i in chosen]
            result['input_protocol'] = 'Owner decoded text sorted by length and re-tokenized; throughput fixture only.'
        assert max(map(len, prompts)) + 1024 <= 32768
        result.update(input_source=str(args.request_json or args.rollouts), source_row_indices=chosen,
                      input_token_lengths=list(map(len, prompts)),
                      input_ids_sha256=hashlib.sha256(json.dumps(prompts).encode()).hexdigest())
        if args.rollouts:
            result['source_row_indices_by_sorted_length'] = chosen

        def batch(active=None):
            rows = prompts
            width = max(map(len, rows))
            ids = torch.full((len(rows), width), worker.tokenizer.pad_token_id, dtype=torch.long)
            mask = torch.zeros_like(ids)
            for index, tokens in enumerate(rows):
                ids[index, -len(tokens):] = torch.tensor(tokens)
                mask[index, -len(tokens):] = 1
            pos = (mask.cumsum(-1)-1).clamp_min(0)
            nt = {} if active is None else {'rollout_active_mask': np.asarray(active, dtype=bool)}
            return DataProto(batch=TensorDict(dict(input_ids=ids, attention_mask=mask, position_ids=pos),
                                             batch_size=[len(rows)]), non_tensor_batch=nt)

        def run(phase, tokens, active=None):
            samples = worker.rollout.sampling_params
            samples.max_tokens = samples.min_tokens = tokens
            samples.ignore_eos = True
            inputs = batch(active)
            if args.greedy:
                inputs.meta_info['do_sample'] = False
            tick = time.perf_counter()
            output = worker.generate_sequences(inputs)
            torch.cuda.synchronize()
            seconds = time.perf_counter()-tick
            count = len(prompts) if active is None else sum(active)
            engine = engine_calls[-1]
            actual_tokens = sum(row['generated_tokens'] for row in engine['requests'])
            assert actual_tokens == count*tokens
            # Preserve exact owner outputs for an execution-path comparison.
            artifact = args.output.with_name(f'{args.output.stem}-{phase}-outputs.pt')
            torch.save({key: output.batch[key].detach().cpu() for key in
                        ('responses', 'attention_mask', 'rollout_log_probs') if key in output.batch}, artifact)
            if active is not None:
                inactive = ~torch.as_tensor(active)
                assert not output.batch['attention_mask'][inactive, -1024:].any()
                assert output.batch['responses'][inactive].eq(worker.tokenizer.pad_token_id).all()
            record(phase, seconds=seconds, requests=count, generated_tokens=actual_tokens,
                   generated_tokens_per_second=actual_tokens/seconds,
                   prompt_tokens=sum(row['prompt_tokens'] for row in engine['requests']),
                   engine_generate=engine, outside_engine_seconds=seconds-engine['seconds'],
                   forced_decode_tokens=tokens, output_shape=list(output.batch['input_ids'].shape),
                   output_artifact=str(artifact))

        run('warmup', 16)
        run('all_active', args.decode_tokens)
        run('all_active_repeat', args.decode_tokens)
        # Same transport; vLLM receives only the actual active requests.
        run('eight_active' if len(prompts) == 32 else 'quarter_active', args.decode_tokens,
            [i % 4 == 0 for i in range(len(prompts))])
        if args.profile_dir:
            profile_next = True
            run('profiled_owner_call_not_throughput_baseline', 32)
        result['status'] = 'measured'
    except Exception as exc:
        result.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        result['elapsed'] = time.perf_counter()-started
        args.output.write_text(json.dumps(result, indent=2)+'\n')
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()

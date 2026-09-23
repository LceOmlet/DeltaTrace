"""Measure the pinned VERL/vLLM worker with identical recorded task text.

The owner dump removes special tokens. This is a throughput replay of that text,
not a regeneration of identical on-policy trajectories or an evaluation score.
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
    parser.add_argument('--rollouts', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = dict(scope=__doc__, max_num_seqs=args.max_num_seqs, actor_minibatch=4,
                  model_context_cap=32768, stages=[])
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
        c.rollout.enforce_eager = True
        c.rollout.engine_kwargs.vllm.limit_mm_per_prompt = {'image': 0, 'video': 0}
        worker = ActorRolloutRefWorker(c, 'actor_rollout')
        worker.init_model()
        record('owner_init')

        # Read the owner's real RequestOutput counters; do not reconstruct
        # cache hits or call this interval pure decode (it includes prefill).
        engine_calls = []
        original_generate = worker.rollout.inference_engine.generate

        def measured_generate(*a, **kw):
            tick = time.perf_counter()
            outputs = original_generate(*a, **kw)
            torch.cuda.synchronize()
            engine_calls.append(dict(seconds=time.perf_counter()-tick, requests=[dict(
                prompt_tokens=len(out.prompt_token_ids),
                cached_tokens=out.num_cached_tokens,
                generated_tokens=sum(len(sample.token_ids) for sample in out.outputs),
            ) for out in outputs]))
            return outputs

        worker.rollout.inference_engine.generate = measured_generate

        lines = [json.loads(line) for line in args.rollouts.read_text().splitlines()]
        texts = sorted((row['input'] for row in lines), key=len)
        chosen = np.linspace(0, len(texts)-1, 32).astype(int).tolist()
        prompts = [worker.tokenizer.encode(texts[i], add_special_tokens=False) for i in chosen]
        assert max(map(len, prompts)) + 1024 <= 32768
        result.update(input_source=str(args.rollouts), source_row_indices_by_sorted_length=chosen,
                      input_token_lengths=list(map(len, prompts)),
                      input_ids_sha256=hashlib.sha256(json.dumps(prompts).encode()).hexdigest())

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
            tick = time.perf_counter()
            output = worker.generate_sequences(inputs)
            torch.cuda.synchronize()
            seconds = time.perf_counter()-tick
            count = 32 if active is None else sum(active)
            engine = engine_calls[-1]
            actual_tokens = sum(row['generated_tokens'] for row in engine['requests'])
            assert actual_tokens == count*tokens
            if active is not None:
                inactive = ~torch.as_tensor(active)
                assert not output.batch['attention_mask'][inactive, -1024:].any()
                assert output.batch['responses'][inactive].eq(worker.tokenizer.pad_token_id).all()
            record(phase, seconds=seconds, requests=count, generated_tokens=actual_tokens,
                   generated_tokens_per_second=actual_tokens/seconds,
                   prompt_tokens=sum(row['prompt_tokens'] for row in engine['requests']),
                   engine_generate=engine, outside_engine_seconds=seconds-engine['seconds'],
                   forced_decode_tokens=tokens, output_shape=list(output.batch['input_ids'].shape))

        run('warmup', 16)
        run('all_active', 128)
        run('all_active_repeat', 128)
        # Same 32-row transport, but vLLM must receive only these 8 requests.
        run('eight_active', 128, [i % 4 == 0 for i in range(32)])
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

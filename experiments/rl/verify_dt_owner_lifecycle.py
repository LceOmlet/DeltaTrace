"""Compare actual DT vectors before/after FSDP parameter-lifetime changes.

The prior adapter is an archived source file, not a second production path.
Uses the real checkpoint and recorded official fixture, with exact comparison.
This numerical regression does not replace the separate exact-32k capacity run.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import statistics
import traceback
from pathlib import Path

import torch
from omegaconf import OmegaConf
from verl.workers.fsdp_workers import ActorRolloutRefWorker


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--previous-adapter', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--context-length', type=int, default=0,
                   help='Explicit filler fixture for matched cost comparison; 0 keeps native fixture length')
    p.add_argument('--repetitions', type=int, default=3,
                   help='First trace is warmup; compare remaining traces')
    args = p.parse_args()
    if args.repetitions < 2:
        p.error('At least one warmup and one measured trace are required')
    result = {'scope': __doc__, 'previous_adapter_sha256': hashlib.sha256(args.previous_adapter.read_bytes()).hexdigest()}
    torch.manual_seed(2026)
    torch.use_deterministic_algorithms(True)
    try:
        cfg = OmegaConf.load(Path(os.environ['VERL_ROOT'])/'verl/trainer/config/ppo_trainer.yaml')
        c = cfg.actor_rollout_ref
        c.model.path = os.environ['MODEL_PATH']
        c.model.lora_rank, c.model.lora_alpha = 1, 2
        c.model.trust_remote_code = True
        c.actor.strategy = 'fsdp2'
        c.actor.ppo_mini_batch_size = 4
        c.actor.ppo_micro_batch_size_per_gpu = 1
        c.actor.use_torch_compile = False
        c.actor.fsdp_config.model_dtype = 'bfloat16'
        c.actor.fsdp_config.reshard_after_forward = False
        c.actor.optim.total_training_steps = 3
        c.rollout.name = 'hf'
        c.rollout.tensor_model_parallel_size = 1
        worker = ActorRolloutRefWorker(c, 'actor_rollout')
        worker.init_model()
        # Exercise nonzero PEFT weights; keep the same actual actor for both runs.
        with torch.no_grad():
            for name, param in worker.actor_module_fsdp.named_parameters():
                if 'lora_B' in name:
                    local = param.to_local() if hasattr(param, 'to_local') else param
                    local.normal_(std=.002)
        fixture = json.loads((Path(os.environ['DT_RUNTIME_ROOT'])/'receipts/rollout-fixtures.json').read_text())
        row = fixture['tasks']['Sokoban']['rows'][-1]
        row = {k: torch.tensor(v) if k in ('input_ids', 'attention_mask', 'responses') else v for k, v in row.items()}
        if args.context_length:
            from reward_readout import RewardAlphabet
            width = row['responses'].numel()
            count = int(row['attention_mask'][-width:].sum())
            query = RewardAlphabet.for_task('Sokoban').query_ids(worker.tokenizer,
                current_step=int(row['env_step']), event_step=int(row['env_step']), max_steps=15)
            fill = args.context_length-len(query)-1-int(row['attention_mask'].sum())
            assert fill >= 0
            filler_id = worker.tokenizer.encode(' context', add_special_tokens=False)[0]
            row['input_ids'] = torch.cat((torch.full((fill,), filler_id), row['input_ids']))
            row['attention_mask'] = torch.cat((torch.ones(fill, dtype=torch.long), row['attention_mask']))
            result['synthetic_filler_tokens'] = fill
        spec = importlib.util.spec_from_file_location('previous_owner_adapter', args.previous_adapter)
        prior = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(prior)
        from deltatrace_rollout import DeltaTraceRolloutProducer
        outputs, vectors = [], []
        for name, producer_type in [('previous', prior.DeltaTraceRolloutProducer), ('layerwise', DeltaTraceRolloutProducer)]:
            producer = producer_type(worker.actor_module_fsdp,
                eos_token_id=worker.tokenizer.eos_token_id, pad_token_id=worker.tokenizer.pad_token_id)
            native = producer.runner.attribute
            signed = []
            def recorded(*a, **kw):
                value, detail = native(*a, **kw)
                signed.append(value.cpu())
                return value, detail
            producer.runner.attribute = recorded
            repeated, reports = [], []
            for iteration in range(args.repetitions):
                repeated.append(producer.attribute_episode([row], float(row['rewards'])))
                reports.append(producer.readout.last_report)
            outputs.append(repeated)
            vectors.append(signed)
            result[name] = reports
            result[name+'_warm_median_seconds'] = statistics.median(r['seconds'] for r in reports[1:])
            from torch.distributed.tensor import DTensor
            assert all(isinstance(param, DTensor) for param in worker.actor_module_fsdp.parameters())
            del producer
        torch.testing.assert_close(vectors[0], vectors[1], rtol=0, atol=0)
        torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
        result['signed_and_qva_exactly_equal'] = True
        result['all_parameters_resharded_after_each_trace'] = True
        result['warm_time_ratio_new_over_previous'] = result['layerwise_warm_median_seconds']/result['previous_warm_median_seconds']
        result['status'] = 'passed'
    except Exception as exc:
        result.update(status='failed', error=str(exc), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        args.output.write_text(json.dumps(result, indent=2, default=str)+'\n')
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    if result['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()

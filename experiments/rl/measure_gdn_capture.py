"""Isolate the measured GDN capture bottleneck using one original 9B layer.

All cases call the same native layer, weights and B8/32768 BF16 hidden input
(four interleaved DT endpoint pairs). The hidden input is a seeded cost fixture,
not a task trajectory. Compare saved operands exactly outside the timed region.
"""
import argparse
import json
import os
import time
import traceback
from pathlib import Path

import torch
from omegaconf import OmegaConf
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from accelerated.qwen35.qwen35_code_local_capture import NativeGDNCapture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = dict(scope=__doc__, runs=[])
    try:
        torch.manual_seed(2026)
        cfg = OmegaConf.load(Path(os.environ['VERL_ROOT'])/'verl/trainer/config/ppo_trainer.yaml')
        c = cfg.actor_rollout_ref
        c.model.path = os.environ['MODEL_PATH']
        c.model.lora_rank, c.model.lora_alpha = 1, 2
        c.model.trust_remote_code = True
        c.actor.strategy = 'fsdp2'
        c.actor.ppo_mini_batch_size = 4
        c.actor.ppo_micro_batch_size_per_gpu = 1
        c.actor.fsdp_config.model_dtype = 'bfloat16'
        c.actor.fsdp_config.reshard_after_forward = True
        c.actor.optim.total_training_steps = 3
        c.rollout.name = 'hf'
        c.rollout.tensor_model_parallel_size = 1
        worker = ActorRolloutRefWorker(c, 'actor_rollout')
        worker.init_model()
        actor = worker.actor_module_fsdp
        actor.eval()
        with torch.no_grad():
            actor(input_ids=torch.tensor([[1, 2, 3]], device='cuda'), use_cache=False, logits_to_keep=1)
        layer = next(m for m in actor.modules() if hasattr(m, 'linear_attn') and m.linear_attn.layer_idx == 30)
        layer.unshard()
        x = torch.randn(8, 32768, 4096, device='cuda', dtype=torch.bfloat16)
        reference = None
        cases = [('all_pageable', True, False), ('needed_pageable', False, False),
                 ('needed_pinned', False, True)]
        for repeat in range(3):
            for name, outputs, pinned in (cases if repeat < 2 else reversed(cases)):
                capture = NativeGDNCapture(layer.linear_attn, device='cpu', copy_tensors=False,
                    preserve_strides=True, capture_module_outputs=outputs, pinned_host=pinned)
                torch.cuda.synchronize()
                start = time.perf_counter()
                with torch.no_grad(), capture:
                    y = layer.linear_attn(x, attention_mask=None)
                torch.cuda.synchronize()
                elapsed = time.perf_counter()-start
                current = dict(values=capture.values, endpoints=capture.endpoints)
                if reference is None:
                    reference = current
                for group, values in current.items():
                    for key, value in values.items():
                        expected = reference[group][key]
                        if value is None:
                            assert expected is None
                        else:
                            assert value.stride() == expected.stride(), (group, key)
                            assert torch.equal(value, expected), (group, key)
                record = dict(name=name, repeat=repeat, seconds=elapsed, exact_common_operands=True,
                              copied_bytes=sum(v.numel()*v.element_size() for values in current.values()
                                               for v in values.values() if v is not None),
                              calls=capture.calls)
                result['runs'].append(record)
                args.output.write_text(json.dumps(result, indent=2)+'\n')
                print('GDN_CAPTURE_RESULT', record, flush=True)
                del y, current, capture
        layer.reshard()
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

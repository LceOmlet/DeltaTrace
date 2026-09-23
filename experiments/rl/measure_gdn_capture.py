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
    parser.add_argument('--pullback', action='store_true',
                        help='Profile only the existing finite pullback of this one layer after a pinned capture.')
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
        if args.pullback:
            # Resolve the production profile through its existing factory;
            # do not substitute the content1 rule for the selected symmetric rule.
            from deltatrace_rollout import DeltaTraceRolloutProducer
            from qwen35_gdn_finite import gdn_finite_pullback
            producer = DeltaTraceRolloutProducer(actor,
                eos_token_id=worker.tokenizer.eos_token_id, pad_token_id=worker.tokenizer.pad_token_id)
            runner = producer.runner
            capture = NativeGDNCapture(layer.linear_attn, device='cpu', copy_tensors=False,
                preserve_strides=True, capture_module_outputs=False, pinned_host=True)
            with torch.no_grad(), capture:
                y = layer.linear_attn(x, attention_mask=None)
            del x, y
            upstream = torch.randn(4, 32768, 4096, device='cuda', dtype=torch.float32)
            # Compare the two existing Torch copy routes on the captured head
            # slices, before changing the owner. Compare values/strides outside
            # the timing window; no finite arithmetic is involved here.
            from native_attention_capture import copy_capture_tensor
            for repeat in range(2):
                for route in ('pageable_contiguous', 'pinned_strided'):
                    elapsed = 0.0
                    for head in range(0, 32, 8):
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        group = {}
                        for name, operand in capture.endpoints.items():
                            if name == 'o':
                                continue
                            part = operand[:, :, head:head+8]
                            if route == 'pageable_contiguous':
                                group[name] = copy_capture_tensor(part.contiguous(), 'cuda', preserve_strides=True)
                            else:
                                assert part.is_pinned()
                                group[name] = torch.empty(part.shape, device='cuda', dtype=part.dtype).copy_(part, non_blocking=True)
                        torch.cuda.synchronize()
                        elapsed += time.perf_counter()-start
                        for name, actual in group.items():
                            expected = capture.endpoints[name][:, :, head:head+8].contiguous()
                            assert actual.stride() == expected.stride()
                            assert torch.equal(actual.cpu(), expected), (route, name, head)
                        del group
                    record = dict(name='head_group_transport', route=route, repeat=repeat,
                                  seconds=elapsed, exact_operands_and_strides=True)
                    result['runs'].append(record)
                    args.output.write_text(json.dumps(result, indent=2)+'\n')
                    print('GDN_HEAD_TRANSPORT', json.dumps(record), flush=True)
            for repeat in range(2):
                # CPU activity diagnoses packing/allocation/dispatch costs. It
                # is not a kernel-time profile or a replacement speed baseline.
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    with torch.no_grad():
                        value, _ = gdn_finite_pullback(layer.linear_attn,
                            dict(capture.values), dict(capture.endpoints), upstream, capture.scale,
                            runner.finite_fla_by_layer[30], norm_gate_rule=runner.norm_gate_rules[30],
                            offload_endpoints=True, fla_head_batch_size=8,
                            norm_gate_pullback=runner.boundaries.gdn_norm_gate,
                            conv_silu_pullback=runner.boundaries.gdn_conv_silu)
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter()-start
                operators = sorted(profile.key_averages(), key=lambda e:e.self_cpu_time_total, reverse=True)
                record = dict(name='finite_pullback_cpu_profile', repeat=repeat, seconds=elapsed,
                    finite=bool(torch.isfinite(value).all()), operators=[dict(name=e.key, calls=e.count,
                        self_cpu_seconds=e.self_cpu_time_total/1e6,
                        inclusive_cpu_seconds=e.cpu_time_total/1e6) for e in operators[:25]])
                result['runs'].append(record)
                args.output.write_text(json.dumps(result, indent=2)+'\n')
                print('GDN_PULLBACK_PROFILE', json.dumps(record), flush=True)
                del value
            layer.reshard()
            result['status'] = 'profiled_single_layer_not_a_training_acceptance'
            return
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
    if result['status'] == 'failed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()

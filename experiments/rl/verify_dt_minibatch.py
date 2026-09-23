"""Check real Qwen DT minibatching and borrowed replay operands.

The copy/no-copy comparison holds batch and inputs fixed and expects exact
values. Serial/batched differences are recorded separately; this script does
not invent a whole-model FA tolerance or replace the exact-32k capacity test.
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--candidate-fa', type=Path)
    p.add_argument('--parameter-offload-policy', action='store_true')
    p.add_argument('--snapshot-output', type=Path,
                   help='Save real actor weight fingerprints and the first B4 native root inputs for offload diagnosis.')
    p.add_argument('--execution-reuse', action='store_true',
                   help='Compare reused retained/local-event captures and deferred diagnostics at fixed batch geometry.')
    args = p.parse_args()
    if args.snapshot_output:
        # First-root hooks save once; require a fresh destination so an older
        # run cannot silently supply the supposedly current tensors.
        args.snapshot_output.mkdir(parents=True, exist_ok=False)
    result = dict(scope=__doc__, max_length=32768,
                  parameter_offload_policy=args.parameter_offload_policy, runs=[])
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
        c.actor.fsdp_config.offload_policy = args.parameter_offload_policy
        c.actor.optim.total_training_steps = 3
        c.rollout.name = 'hf'
        c.rollout.tensor_model_parallel_size = 1
        worker = ActorRolloutRefWorker(c, 'actor_rollout')
        worker.init_model()
        # Reproduce the real zero-reward iteration ordering: actor log-probs
        # can initialize FSDP before the first nonzero-reward DT invocation.
        worker.actor_module_fsdp.eval()
        with torch.no_grad():
            worker.actor_module_fsdp(input_ids=torch.tensor([[1, 2, 3]], device='cuda'),
                                    use_cache=False, logits_to_keep=1)
        result['actor_forward_before_dt'] = True
        from deltatrace_rollout import DeltaTraceRolloutProducer
        producer = DeltaTraceRolloutProducer(worker.actor_module_fsdp,
            eos_token_id=worker.tokenizer.eos_token_id, pad_token_id=worker.tokenizer.pad_token_id)
        fixture = json.loads((Path(os.environ['DT_RUNTIME_ROOT'])/'receipts/rollout-fixtures.json').read_text())
        row = fixture['tasks']['Sokoban']['rows'][-1]
        row = {k: torch.tensor(v) if k in ('input_ids', 'attention_mask', 'responses') else v for k,v in row.items()}
        filler = worker.tokenizer.encode(' context', add_special_tokens=False)[0]
        episodes = []
        for index, count in enumerate((0, 3, 8, 17)):
            sample = dict(row, traj_uid=f'minibatch-{index}')
            sample['input_ids'] = torch.cat((torch.full((count,), filler), row['input_ids']))
            sample['attention_mask'] = torch.cat((torch.ones(count, dtype=torch.long), row['attention_mask']))
            episodes.append([sample])
        outputs = {}
        if args.snapshot_output:
            import hashlib
            from torch.distributed.tensor import DTensor
            fingerprints = {}
            for name, parameter in worker.actor_module_fsdp.named_parameters():
                local = parameter.to_local() if isinstance(parameter, DTensor) else parameter
                raw = local.detach().cpu().contiguous()
                fingerprints[name] = dict(shape=list(raw.shape), dtype=str(raw.dtype),
                    sha256=hashlib.sha256(raw.view(torch.uint8).numpy().tobytes()).hexdigest())
            (args.snapshot_output/'weights.json').write_text(json.dumps(fingerprints, indent=2)+'\n')
        # Test-only first-boundary comparison isolates transport/layout errors
        # from error accumulated across the full finite network.
        import qwen35_dense_finite_runner as owner
        mixer_baselines = {}
        mixer_seen = set()
        current_run = ''
        for kind in ('attention', 'gdn'):
            function_name = kind+'_finite_pullback'
            original = getattr(owner, function_name)
            def check_mixer(*a, _kind=kind, _original=original, **kw):
                value = _original(*a, **kw)
                if current_run in ('batch_head8', 'batch_offloaded') and (current_run, _kind) not in mixer_seen:
                    mixer_seen.add((current_run, _kind))
                    tensors = {f'{index}.{key}': (tensor.detach().cpu(), tensor.stride())
                        for index, arg in enumerate(a) if isinstance(arg, dict)
                        for key, tensor in arg.items() if isinstance(tensor, torch.Tensor)}
                    tensors['output'] = (value[0].detach().cpu(), value[0].stride())
                    if current_run == 'batch_head8':
                        mixer_baselines[_kind] = tensors
                    else:
                        result.setdefault('first_mixer_differences', {})[_kind] = {
                            key: dict(max_abs=float((tensor.double()-mixer_baselines[_kind][key][0].double()).abs().max()),
                                stride=stride, baseline_stride=mixer_baselines[_kind][key][1])
                            for key, (tensor, stride) in tensors.items()}
                return value
            setattr(owner, function_name, check_mixer)
        calls = []
        native = producer.runner.attribute
        def record(*a, **kw):
            handles = []
            if args.snapshot_output and current_run == 'batch_borrowed':
                torch.save(dict(ids=a[0].cpu(), mask=a[1].cpu()), args.snapshot_output/'inputs.pt')
                layers = producer.runner.model.model.language_model.layers
                def save_input(module, inputs, kwargs, *, name):
                    path = args.snapshot_output/(name+'.pt')
                    if not path.exists():
                        hidden = inputs[0] if inputs else kwargs['hidden_states']
                        torch.save(hidden.detach().cpu(), path)
                from functools import partial
                for index, layer in enumerate(layers):
                    handles.append(layer.register_forward_pre_hook(
                        partial(save_input, name=f'layer-{index:02d}'), with_kwargs=True))
                def save_logits(module, inputs, output):
                    path = args.snapshot_output/'logits.pt'
                    if not path.exists():
                        torch.save(output.detach().cpu(), path)
                handles.append(producer.runner.model.lm_head.register_forward_hook(save_logits))
            try:
                value, detail = native(*a, **kw)
            finally:
                for handle in handles:
                    handle.remove()
            calls.append(dict(paired_shape=list(a[0].shape), signed=value.cpu(),
                              target_logp0=detail['target_logp0'], target_logp1=detail['target_logp1']))
            return value, detail
        producer.runner.attribute = record
        if args.execution_reuse:
            from accelerated.qwen35 import qwen35_retained_capture as retained
            from accelerated.qwen35 import qwen35_code_local_capture as local
            from accelerated.native_capture_events import check_runtime
            result['local_event_runtime'] = check_runtime()
            result['scope'] = 'Real actor, identical B4 inputs and finite rules; reused capture/scheduling controls only.'
            producer.readout.minibatch_size = 4
            producer.runner.copy_replay_captures = False
            reference = None
            configurations = [('borrowed', None, False), ('retained', retained, False),
                              ('local_events', local, False), ('deferred', local, True)]
            for repeat in range(3):
                for name, backend, deferred in (configurations if repeat != 2 else reversed(configurations)):
                    current_run = name
                    producer.runner.capture_backend = backend
                    producer.runner.defer_diagnostics = deferred
                    calls.clear()
                    start = time.perf_counter()
                    value = producer.attribute_episodes(episodes, [float(row['rewards'])]*4)
                    elapsed = time.perf_counter()-start
                    actual = (value, list(calls))
                    if reference is None: reference = actual
                    torch.testing.assert_close(reference, actual, rtol=0, atol=0)
                    result['runs'].append(dict(name=name, repeat=repeat, seconds=elapsed,
                        paired_shapes=[v['paired_shape'] for v in calls], exact_match=True))
                    args.output.write_text(json.dumps(result, indent=2)+'\n')
            result['status'] = 'passed_reused_execution_parity'
            return
        configurations = [('serial', 1, False), ('batch_copied', 4, True), ('batch_borrowed', 4, False), ('batch_compiled_gate', 4, False), ('batch_head8', 4, False), ('batch_offloaded', 4, False)]
        if producer.runner.fa_coefficient_suffix:
            configurations.append(('batch_fa_suffix', 4, False))
        for name, batch, copies in configurations:
            current_run = name
            producer.runner.capture_backend = None
            producer.runner.defer_diagnostics = False
            producer.readout.minibatch_size = batch
            producer.runner.copy_replay_captures = copies
            producer.runner.fa_coefficient_suffix = name == 'batch_fa_suffix'
            producer.runner.offload_replay_mixer = name in ('batch_offloaded', 'batch_fa_suffix')
            producer.runner.pin_replay_host = name in ('batch_offloaded', 'batch_fa_suffix')
            producer.runner.compile_gdn_scalar_rules = name in ('batch_compiled_gate', 'batch_head8', 'batch_offloaded', 'batch_fa_suffix')
            producer.runner.gdn_head_batch_size = 8 if name in ('batch_head8', 'batch_offloaded', 'batch_fa_suffix') else None
            calls.clear()
            start = time.perf_counter()
            value = producer.attribute_episodes(episodes, [float(row['rewards'])]*4)
            outputs[name] = (value, list(calls))
            assert all(torch.isfinite(r['dt_token_advantages']).all() for ep in value for r in ep)
            assert any(r['dt_token_advantages'].count_nonzero() for ep in value for r in ep)
            result['runs'].append(dict(name=name, seconds=time.perf_counter()-start,
                paired_shapes=[v['paired_shape'] for v in calls], report=producer.readout.last_report))
            args.output.write_text(json.dumps(result, indent=2)+'\n')
        torch.testing.assert_close(outputs['batch_copied'], outputs['batch_borrowed'], rtol=0, atol=0)
        result['borrowed_matches_copied_exactly'] = True
        torch.testing.assert_close(outputs['batch_head8'], outputs['batch_offloaded'], rtol=0, atol=0)
        result['offloaded_matches_same_head_partition_exactly'] = True
        if 'batch_fa_suffix' in outputs:
            torch.testing.assert_close(outputs['batch_offloaded'], outputs['batch_fa_suffix'], rtol=0, atol=0)
            result['fa_coefficient_suffix_matches_full_exactly'] = True
        assert len(outputs['batch_borrowed'][1]) == 1
        assert outputs['batch_borrowed'][1][0]['paired_shape'][0] == 8
        result['compiled_gate_differences'] = {}
        for key in ('dt_q_estimates', 'dt_v_estimates', 'dt_token_advantages'):
            plain = torch.cat([e[0][key] for e in outputs['batch_borrowed'][0]]).double()
            fused = torch.cat([e[0][key] for e in outputs['batch_compiled_gate'][0]]).double()
            result['compiled_gate_differences'][key] = dict(max_abs=float((plain-fused).abs().max()),
                relative_l2=float((plain-fused).norm()/plain.norm().clamp_min(1e-30)))
        result['head_partition_differences'] = {}
        for key in ('dt_q_estimates', 'dt_v_estimates', 'dt_token_advantages'):
            full = torch.cat([e[0][key] for e in outputs['batch_compiled_gate'][0]]).double()
            split = torch.cat([e[0][key] for e in outputs['batch_head8'][0]]).double()
            result['head_partition_differences'][key] = dict(max_abs=float((full-split).abs().max()),
                relative_l2=float((full-split).norm()/full.norm().clamp_min(1e-30)))
        result['serial_batch_differences'] = {}
        for key in ('dt_q_estimates', 'dt_v_estimates', 'dt_token_advantages'):
            serial = torch.cat([e[0][key] for e in outputs['serial'][0]]).double()
            batch = torch.cat([e[0][key] for e in outputs['batch_borrowed'][0]]).double()
            result['serial_batch_differences'][key] = dict(
                max_abs=float((serial-batch).abs().max()),
                relative_l2=float((serial-batch).norm()/serial.norm().clamp_min(1e-30)))
        if args.candidate_fa:
            import hashlib
            from vendor_fa_finite_bf16_d256 import VendorFAFiniteP1BF16D256
            digest=hashlib.sha256(args.candidate_fa.read_bytes()).hexdigest()
            producer.runner.finite_fa=VendorFAFiniteP1BF16D256(args.candidate_fa,digest)
            producer.runner.offload_replay_mixer=False
            producer.runner.gdn_head_batch_size=None
            producer.runner.compile_gdn_scalar_rules=False
            current_run='candidate_fa'
            start=time.perf_counter()
            candidate=producer.attribute_episodes(episodes,[float(row['rewards'])]*4)
            result['candidate_fa']=dict(sha256=digest,seconds=time.perf_counter()-start,differences={})
            for key in ('dt_q_estimates','dt_v_estimates','dt_token_advantages'):
                reference=torch.cat([e[0][key] for e in outputs['batch_borrowed'][0]]).double()
                actual=torch.cat([e[0][key] for e in candidate]).double()
                result['candidate_fa']['differences'][key]=dict(max_abs=float((reference-actual).abs().max()),
                    relative_l2=float((reference-actual).norm()/reference.norm().clamp_min(1e-30)))
        result['status'] = 'passed_memory_copy_parity_and_batch_interface'
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

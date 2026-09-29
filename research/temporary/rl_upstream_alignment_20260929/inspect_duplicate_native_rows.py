"""Locate duplicate-row differences in the original Qwen forward and cache.

Read-only saved-token diagnostic. It neither implements a layer nor imposes
an acceptance threshold on full-model numerics.
"""
import json
import os
from pathlib import Path
import re
import sys
import time
import torch
from verify_author_rollout import configuration
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from deltatrace_rollout import DeltaTraceRolloutProducer

root = Path(os.environ['DT_RUNTIME_ROOT']) / 'receipts/upstream-alignment-20260929'
output = root / 'native-duplicate-row-diagnostic.json'
credit_probe = os.environ.get('DT_DUPLICATE_CREDIT_PROBE') == '1'
boundary_probe = os.environ.get('DT_DUPLICATE_BOUNDARY_PROBE') == '1'
credit_probe = credit_probe or boundary_probe
if credit_probe:
    output = root / 'native-duplicate-credit-diagnostic.json'
if boundary_probe:
    output = root / 'native-duplicate-boundary-diagnostic.json'
started = time.perf_counter()
result = dict(scope=__doc__, stages=[])
def save(phase):
    result.update(phase=phase, seconds=time.perf_counter()-started)
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(phase=phase, seconds=result['seconds'])), flush=True)

for raw in (root/'native-two-gpu-credit-native-lora.log').read_text().splitlines():
    if '[DT EOS minimum] ' in raw:
        entry = json.loads(re.sub(r'\x1b\[[0-9;]*m', '', raw).split('[DT EOS minimum] ', 1)[1])
        samples = [s for s in entry['samples'] if s['traj_uid']=='Sokoban-1']
        if samples:
            sample = samples[0]
            break
cfg = configuration().actor_rollout_ref
cfg.rollout.name = 'hf'
worker = ActorRolloutRefWorker(cfg, 'actor')
worker.init_model()
worker.actor_module_fsdp.eval()
producer = DeltaTraceRolloutProducer(worker.actor_module_fsdp,
    eos_token_id=worker.tokenizer.eos_token_id, pad_token_id=worker.tokenizer.pad_token_id,
    invalid_action_penalty_coef=cfg.actor.invalid_action_penalty_coef)
runner = producer.runner
model = runner.model
layers = model.model.language_model.layers
model.model.language_model.set_attn_implementation('flash_attention_2')
if credit_probe:
    fixtures = json.loads((Path(os.environ['DT_RUNTIME_ROOT'])/'receipts/rollout-fixtures.json').read_text())
    row = [r for r in fixtures['tasks']['Sokoban']['rows'] if r['active_masks']][-1]
    row = {k: torch.tensor(v) if k in ('input_ids', 'attention_mask', 'responses') else v for k,v in row.items()}
    episodes = [[{**row, 'traj_uid': f'duplicate-{i}'}] for i in range(4)]
    result['runs'] = []
    for mode in (('passive_boundaries',) if boundary_probe else ('plain', 'plain_repeat', 'synchronize_layers')):
        handles = []
        observations = []
        if boundary_probe:
            from accelerated.qwen35.qwen35_code_local_capture import NativeGDNCapture
            original_attribute = runner.attribute
            def record_owner_details(*args, **kwargs):
                signed, details = original_attribute(*args, **kwargs)
                result['owner_details'] = details
                return signed, details
            runner.attribute = record_owner_details
            capture_state = {'calls': 0}
            def start_gdn_capture(module, args):
                capture_state['calls'] += 1
                if capture_state['calls'] == 2:
                    capture_state['capture'] = NativeGDNCapture(module, device='cpu')
                    capture_state['capture'].__enter__()
            def finish_gdn_capture(module, args, out):
                capture = capture_state.pop('capture', None)
                if capture is not None:
                    capture.__exit__(None,None,None)
                    torch.save(dict(values=capture.values,endpoints=capture.endpoints,scale=capture.scale),
                               root/'actual-dt-cached-layer2-gdn.pt')
            handles.append(layers[2].linear_attn.register_forward_pre_hook(start_gdn_capture))
            handles.append(layers[2].linear_attn.register_forward_hook(finish_gdn_capture))
            visits = {}
            saved_first = set()
            def observe(name):
                def capture(module, args, out):
                    value = out[0] if isinstance(out, tuple) else out
                    if not isinstance(value, torch.Tensor) or value.ndim != 3 or value.shape[0] not in (4, 8):
                        return
                    visits[name] = visits.get(name, 0) + 1
                    # Pair0 has a different scored return label. Compare only
                    # pairs1..3, whose complete IDs/labels are identical.
                    for side, rows in ([('prefix', value[1:])] if value.shape[0] == 4 else
                                       [('reference', value[2::2]), ('factual', value[3::2])]):
                        delta = rows.float() - rows[:1].float()
                        maximum = float(delta.abs().max())
                        observations.append(dict(module=name, call=visits[name], side=side,
                            shape=list(value.shape), dtype=str(value.dtype), duplicate_max_abs=maximum))
                        key = (visits[name], side)
                        if maximum and key not in saved_first:
                            saved_first.add(key)
                            torch.save(dict(name=name, call=visits[name], side=side,
                                input=args[0].detach().cpu() if args and isinstance(args[0],torch.Tensor) else None,
                                output=value.detach().cpu(),
                                weight=module.weight.detach().cpu() if isinstance(getattr(module,'weight',None),torch.Tensor)
                                    and not hasattr(module.weight,'placements') else None),
                                root/f'duplicate-first-difference-{visits[name]}-{side}.pt')
                return capture
            for index, layer in enumerate(layers):
                handles.append(layer.register_forward_hook(observe(f'{index}.output')))
                for name, module in layer.named_modules():
                    if name in ('input_layernorm', 'post_attention_layernorm', 'linear_attn',
                                'linear_attn.in_proj_qkv', 'linear_attn.in_proj_a', 'linear_attn.in_proj_b',
                                'linear_attn.in_proj_z', 'linear_attn.norm', 'linear_attn.out_proj', 'mlp'):
                        handles.append(module.register_forward_hook(observe(f'{index}.{name}')))
        if mode == 'synchronize_layers':
            def synchronize(module, args, out):
                torch.cuda.synchronize()
            handles = [layer.register_forward_hook(synchronize) for layer in layers]
        try:
            producer.attribute_episodes(episodes, [float(row['rewards'])]*4,
                complete_returns=[[10.799999237060547]]+[[10.899999618530273]]*3)
            report = producer.readout.last_report
            result['runs'].append(dict(mode=mode, seconds=report['seconds'], traces=report['traces'],
                source_vectors=[x['source_signed'] for x in report['minimum_log_ratio_batch']['samples']]))
            if boundary_probe:
                result['observations'] = observations
                result['first_differences'] = [next(item for item in observations
                    if item['call'] == call and item['side'] == side and item['duplicate_max_abs'] > 0)
                    for call, side in sorted(saved_first)]
            save(mode)
        finally:
            for handle in handles: handle.remove()
            if boundary_probe:
                runner.attribute = original_attribute
    save('complete')
    torch.distributed.destroy_process_group()
    sys.exit(0)
from accelerated.qwen35.native_fla_precision import native_fla_fp16
from accelerated.qwen35.qwen35_code_local_capture import NativeGDNCapture

selected = torch.tensor(sample['selected_input_ids'][:-1], device='cuda')[None].repeat(4, 1)
reference = selected.clone()
reference[:, sample['source_start']:sample['source_end']] = worker.tokenizer.eos_token_id
pair = torch.stack((reference, selected), 1).flatten(0, 1)
cut = sample['source_start']//64*64
observations = []
phase = ''
def hook(name):
    def capture(module, args, out):
        value = out[0] if isinstance(out, tuple) else out
        if not isinstance(value, torch.Tensor) or value.ndim < 3:
            return
        stride = 1 if phase == 'prefix' else 2
        rows = value.detach()[stride-1::stride].float()
        difference = rows-rows[:1]
        observations.append(dict(phase=phase, layer=name, shape=list(value.shape),
            duplicate_max_abs=float(difference.abs().max()),
            duplicate_relative_l2=float(difference.norm()/rows.norm().clamp_min(1e-30))))
    return capture
handles = [layer.register_forward_hook(hook(str(i))) for i,layer in enumerate(layers)]
for name in ('input_layernorm','post_attention_layernorm'):
    handles.append(getattr(layers[0], name).register_forward_hook(hook('0.'+name)))
for name in ('in_proj_qkv','in_proj_a','in_proj_b','in_proj_z','out_proj'):
    handles.append(getattr(layers[0].linear_attn, name).register_forward_hook(hook('0.'+name)))
save('model_ready')
try:
    with native_fla_fp16(worker.actor_module_fsdp), torch.no_grad():
        phase = 'full'
        capture = NativeGDNCapture(layers[0].linear_attn, device='cpu')
        with capture:
            full = runner.read_outcomes(pair, entry['outcome_token_ids'])
        torch.save(dict(values=capture.values, endpoints=capture.endpoints, scale=capture.scale),
            root/'native-duplicate-first-gdn.pt')
        del capture
        result['full_logp'] = full.cpu().tolist()
        save('native_full_complete')
        phase = 'prefix'
        prefix = model.forward_root(input_ids=selected[:, :cut], use_cache=True, logits_to_keep=1)
        cache = prefix.past_key_values
        del prefix
        cache.reorder_cache(torch.arange(4,device='cuda').repeat_interleave(2))
        phase = 'suffix'
        cached = runner.read_outcomes(pair[:, cut:], entry['outcome_token_ids'], past_key_values=cache)
        result['cached_logp'] = cached.cpu().tolist()
        result['full_cached_max_abs'] = float((full-cached).abs().max())
        result['observations'] = observations
        save('complete')
finally:
    for handle in handles:
        handle.remove()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

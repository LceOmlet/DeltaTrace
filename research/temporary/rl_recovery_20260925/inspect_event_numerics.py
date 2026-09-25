"""Read-only DT comparison on recorded owner IDs and a completed actor checkpoint.

No environment reward or training update is generated here. The single-token
check measures a fixed-text deletion endpoint, not full environment causality.
"""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import time
import torch
from verify_author_rollout import configuration
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from deltatrace_rollout import DeltaTraceRolloutProducer
from deltatrace_credit import trace_token_attribution


def native_prefix_diagnostic(runtime, selected, reference, cases, labels, producer, output):
    """Observe actual HF cache composition; no DT finite propagation here."""
    from qwen35_answer_finite import selected_target_log_probs
    from accelerated.qwen35.qwen35_code_local_capture import NativeGDNCapture
    pair = torch.stack((reference, selected), 1).flatten(0, 1)
    selection = producer.packed_answer_targets(cases, [[0]]*4, pair.shape[1], pair.device,
                                               outcome_token_ids=labels)
    positions = selection.positions.unique(sorted=True)
    changed = reference != selected
    cut = int(torch.where(changed, torch.arange(pair.shape[1], device=pair.device),
                          pair.shape[1]).amin())//64*64
    assert cut > 0
    layers = runtime.model.model.language_model.layers
    full, leaf_inputs, leaf_weights = {}, {}, {}
    observations = []
    phase = 'full'
    handles = []
    def hook(name, save_operand=False):
        def capture(module, args, out):
            if phase == 'prefix':
                return
            value = out[0] if isinstance(out, tuple) else out
            if phase == 'full':
                full[name] = value[:, cut:].detach().cpu().clone()
                if save_operand:
                    leaf_inputs[name] = args[0][:, cut:].detach().cpu().clone()
                    weight = module.weight
                    if hasattr(weight, 'full_tensor'): weight = weight.full_tensor()
                    leaf_weights[name] = weight.detach().cpu().clone()
            else:
                actual = value.detach().cpu()
                original = full[name]
                delta = actual.float()-original.float()
                record = dict(name=name, shape=list(actual.shape), equal=torch.equal(original, actual),
                    max_abs=float(delta.abs().max()), relative_l2=float(delta.norm()/original.float().norm().clamp_min(1e-30)))
                if save_operand:
                    x = args[0].detach().cpu()
                    record['input_equal'] = torch.equal(leaf_inputs[name], x)
                    record['input_max_abs'] = float((x.float()-leaf_inputs[name].float()).abs().max())
                    if record['input_equal'] and not record['equal'] and 'first_leaf_operands' not in saved:
                        saved['first_leaf_operands'] = dict(name=name, input=x, weight=leaf_weights[name],
                            full_output=original, cached_output=actual)
                observations.append(record)
        return capture
    saved = {}
    for i, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(hook(f'layer.{i}')))
        if i < 3:
            handles.append(layer.input_layernorm.register_forward_hook(hook(f'layer.{i}.input_norm')))
            for name in ('in_proj_a','in_proj_b','in_proj_qkv','in_proj_z','out_proj'):
                module = getattr(layer.linear_attn, name)
                # Capture native leaf outputs only; LoRA parent output still
                # determines the real layer result above.
                leaf = module.base_layer if hasattr(module, 'base_layer') else module
                handles.append(leaf.register_forward_hook(hook(f'layer.{i}.{name}', True)))
    try:
        with torch.no_grad():
            capture = NativeGDNCapture(layers[0].linear_attn, device='cpu')
            with capture:
                a = runtime.model.forward_root(input_ids=pair, attention_mask=torch.ones_like(pair),
                    use_cache=False, logits_to_keep=positions)
            saved['full_gdn'] = dict(values=capture.values, endpoints=capture.endpoints, scale=capture.scale)
            del capture
            full_logp = selected_target_log_probs(a.logits[selection.paired_samples,
                torch.searchsorted(positions,selection.paired_positions)],selection).cpu()
            del a
            phase = 'prefix'
            prefix = runtime.model.forward_root(input_ids=selected[:, :cut],use_cache=True,logits_to_keep=1)
            cache = prefix.past_key_values
            del prefix
            cache.reorder_cache(torch.arange(4,device=pair.device).repeat_interleave(2))
            phase = 'suffix'
            capture = NativeGDNCapture(layers[0].linear_attn, device='cpu')
            with capture:
                a = runtime.model.forward_root(input_ids=pair[:,cut:],attention_mask=torch.ones_like(pair),
                    use_cache=True,past_key_values=cache,logits_to_keep=positions-cut)
            saved['cached_gdn'] = dict(values=capture.values, endpoints=capture.endpoints, scale=capture.scale)
            saved['cut'] = cut
            del capture
            cached_logp = selected_target_log_probs(a.logits[selection.paired_samples,
                torch.searchsorted(positions,selection.paired_positions)],selection).cpu()
            del a,cache
    finally:
        for h in handles:h.remove()
    if saved:
        torch.save(saved,output.with_suffix('.pt'))
    return dict(scope='Original HF full forward versus official cache prefix/reorder/suffix, same trained actor and exact IDs; no finite DT, no alternate attention.',
        cut=cut, length=pair.shape[1], observations=observations,
        full_logp=full_logp.tolist(), cached_logp=cached_logp.tolist(),
        first_leaf_operands=saved.get('first_leaf_operands',{}).get('name'))


def main(stack):
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--native-prefix-only', action='store_true')
    p.add_argument('--causal-prefix-only', action='store_true')
    p.add_argument('--native-fla-fp16', action='store_true',
                   help='Diagnostic native FLA dtype boundary; keep Qwen weights and output BF16')
    args = p.parse_args()
    start = time.perf_counter()
    result = dict(scope=__doc__, checkpoint=str(args.checkpoint), cases=[])
    def save(phase):
        result.update(phase=phase, seconds=time.perf_counter()-start)
        args.output.write_text(json.dumps(result, indent=2)+'\n')
        print(phase, result['seconds'], flush=True)
    torch.set_num_threads(8)
    torch.manual_seed(2026)
    worker = ActorRolloutRefWorker(configuration(1024).actor_rollout_ref, 'actor')
    worker.init_model()
    worker.load_checkpoint(str(args.checkpoint), del_local_after_load=False)
    # Keep the owner's CPU-offloaded checkpoint state for FSDP lazy init.
    # FSDP owns materialization on the first real forward.
    worker.actor_module_fsdp.eval()
    producer = DeltaTraceRolloutProducer(worker.actor_module_fsdp,
        eos_token_id=worker.tokenizer.eos_token_id, pad_token_id=worker.tokenizer.pad_token_id)
    save('loaded_original_actor_checkpoint')
    from profiles.official import make_qwen35_runner
    from qwen35_answer_finite import selected_target_log_probs
    runtime = producer.runner
    if args.native_fla_fp16:
        from accelerated.qwen35.native_fla_precision import native_fla_fp16
        precision = native_fla_fp16(worker.actor_module_fsdp)
        stack.enter_context(precision)
        result['fla_compute_dtype'] = 'float16 (diagnostic boundary; native operator)'
    # Official factory and current finite callbacks; no copied propagation.
    control = make_qwen35_runner(runtime.model, runtime.finite_fa, runtime.finite_fla,
                                answer_compiled=False, dynamic_shapes=True)
    examples = [episode[0] for episode in json.loads(args.input.read_text())[:4]]
    alphabet = producer.readout.alphabet
    labels = alphabet.label_ids(worker.tokenizer)
    cases, endpoints, spans = [], [], []
    for row in examples:
        prompt, response = row['prompt'], row['response']
        target = labels[alphabet.observed_index(row['reward'])]
        query = alphabet.query_ids(worker.tokenizer, current_step=0, event_step=0, max_steps=15)
        seq = prompt+response+query+[target]
        spans.append((len(prompt),len(prompt)+len(response)))
        endpoints.append(seq)
        cases.append(dict(target_ids=torch.tensor([target]), prompt_length=len(seq)-1))
    selected = torch.full((4,max(map(len,endpoints))),worker.tokenizer.eos_token_id,device='cuda',dtype=torch.long)
    for i,seq in enumerate(endpoints): selected[i,:len(seq)] = torch.tensor(seq,device='cuda')
    reference = selected.clone()
    for i,(left,right) in enumerate(spans): reference[i,left:right] = worker.tokenizer.eos_token_id
    if args.native_prefix_only or args.causal_prefix_only:
        text_model = runtime.model.model.language_model
        original_attention = text_model.config._attn_implementation
        text_model.set_attn_implementation('flash_attention_2')
        try:
            if args.causal_prefix_only:
                from inspect_prefix_causality import inspect
                result['causal_prefix'] = inspect(runtime, selected, reference, cases, labels, producer)
            else:
                result['native_prefix'] = native_prefix_diagnostic(runtime, selected, reference, cases,
                    labels, producer, args.output)
            save('completed_native_prefix_diagnostic')
        finally:
            text_model.set_attn_implementation(original_attention)
            if torch.distributed.is_initialized():torch.distributed.destroy_process_group()
        return
    outputs = {}
    for name,runner in [('rl_runtime',runtime),('rl_without_prefix_reuse',runtime),('official_default',control)]:
        if name == 'rl_without_prefix_reuse':
            runtime.reuse_native_prefix = False
        save(name+'_start')
        signed, roots, detail = trace_token_attribution(runner,reference,selected,cases,[[0]]*4,
            packed_answer_targets=producer.packed_answer_targets,outcome_token_ids=labels)
        outputs[name] = signed.cpu()
        result[name] = dict(seconds=detail['complete_attribution_seconds_with_diagnostics'],
            roots=roots.tolist(), signed_sums=signed.sum(-1).tolist(), signed_min=signed.min(-1).values.tolist(),
            signed_max=signed.max(-1).values.tolist(), finite=bool(torch.isfinite(signed).all()),
            residuals=[x['conservation_residual'] for x in detail['per_sample']])
        del signed
        save(name+'_end')
    old,new=outputs['official_default'],outputs['rl_runtime']
    result['difference'] = dict(max_abs=float((new-old).abs().max()),
                               relative_l2=float((new-old).norm()/old.norm().clamp_min(1e-30)))
    other = outputs['rl_without_prefix_reuse']
    result['difference_without_prefix_reuse'] = dict(max_abs=float((other-old).abs().max()),
                               relative_l2=float((other-old).norm()/old.norm().clamp_min(1e-30)))
    # Query original forward once for four selected single-token deletions.
    reference=selected.clone()
    for i,(left,right) in enumerate(spans):
        pos=left+int(outputs['rl_runtime'][i,left:right].abs().argmax())
        reference[i,pos]=worker.tokenizer.eos_token_id
        result['cases'].append(dict(token_position=pos, token_id=int(selected[i,pos]),
                                   full_eos_dt=float(outputs['rl_runtime'][i,pos]), reward=examples[i]['reward']))
    pair=torch.stack((reference,selected),1).flatten(0,1)
    sel=producer.packed_answer_targets(cases,[[0]]*4,pair.shape[1],pair.device,outcome_token_ids=labels)
    positions=sel.positions.unique(sorted=True)
    from qwen35_answer_finite import categorical_head_logits
    packed_rows=(sel.paired_samples,torch.searchsorted(positions,sel.paired_positions))
    captured=[]
    handle=runtime.model.lm_head.register_forward_pre_hook(
        lambda _m,args:captured.append(args[0][packed_rows].detach()))
    try:
        with torch.no_grad():
            native=runtime.model.forward_root(input_ids=pair, attention_mask=torch.ones_like(pair),
                                             use_cache=False, logits_to_keep=positions)
            logits=native.logits[packed_rows]
            (hidden,)=captured
            event_logits=categorical_head_logits(runtime.model.lm_head,hidden,sel.outcome_token_ids)
            lp=selected_target_log_probs(logits,sel,outcome_logits=event_logits)
    finally:
        handle.remove()
    result['single_token_readout']='original captured head input and weight; categorical FP32, same as DT'
    del native
    effects=(lp[1::2]-lp[0::2]).tolist()
    for case,effect in zip(result['cases'],effects):case['native_single_token_effect']=effect
    save('completed_diagnostic_no_new_acceptance_threshold')
    torch.save(outputs,args.output.with_suffix('.pt'))
    if torch.distributed.is_initialized():torch.distributed.destroy_process_group()


if __name__=='__main__':
    with ExitStack() as stack:
        main(stack)

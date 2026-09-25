"""Read-only DT comparison on recorded owner IDs and a completed actor checkpoint.

No environment reward or training update is generated here. The single-token
check measures a fixed-text deletion endpoint, not full environment causality.
"""
import argparse
import json
from pathlib import Path
import time
import torch
from verify_author_rollout import configuration
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from deltatrace_rollout import DeltaTraceRolloutProducer
from deltatrace_credit import trace_token_attribution


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
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
    with torch.no_grad():
        native=runtime.model.forward_root(input_ids=pair, attention_mask=torch.ones_like(pair),
                                         use_cache=False, logits_to_keep=positions)
        logits=native.logits[sel.paired_samples,torch.searchsorted(positions,sel.paired_positions)]
        lp=selected_target_log_probs(logits,sel)
    del native
    effects=(lp[1::2]-lp[0::2]).tolist()
    for case,effect in zip(result['cases'],effects):case['native_single_token_effect']=effect
    save('completed_diagnostic_no_new_acceptance_threshold')
    torch.save(outputs,args.output.with_suffix('.pt'))
    if torch.distributed.is_initialized():torch.distributed.destroy_process_group()


if __name__=='__main__':main()

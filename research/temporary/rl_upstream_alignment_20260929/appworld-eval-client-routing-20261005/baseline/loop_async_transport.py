"""Exact LOOP requests/replies through VERL's native asynchronous servers.

LOOP owns task/runner/cancellation state; vLLM owns active requests, batching,
decoding and LoRA. This module transports native final RequestOutput objects
and represents them as the existing VERL DataProto credit carrier.
"""
import time


def response_carrier(prompt, output, response_length, pad_token_id, eos_token_id):
    from vllm import SamplingParams
    from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import process_request_outputs

    metadata = dict(prompt.non_tensor_batch)
    metadata.pop('raw_prompt_ids',None)
    overrides = metadata.pop('owner_sampling_kwargs')
    return process_request_outputs(
        [output], prompt.batch['input_ids'], prompt.batch['attention_mask'],
        prompt.batch['position_ids'], metadata, eos_token_id=eos_token_id,
        sampling_params=SamplingParams(n=1), pad_token_id=pad_token_id,
        max_response_length=response_length, owner_overrides=overrides)


def collect(owner,collector,gen_batch):
    import numpy as np
    import ray
    import torch
    from verl import DataProto
    from owner_environment_transport import policy_reply

    pool=owner.processes
    pool.start()
    servers=collector.async_rollout_manager.async_llm_servers
    results,records,inflight,aborted={},{},{},set()
    calls=tokens=0
    started=time.monotonic()
    while len(results)<len(pool.processes) or inflight:
        event=pool.event()
        if event[0]=='result':
            results[event[1]]=event[2]
        elif event[0]=='completion':
            _,rank,key,raw=event
            if pool.cancellations[rank].is_set():continue
            options=dict(raw);ids=options.pop('prompt')
            options.pop('model');options.pop('stream');options['detokenize']=True
            carrier=DataProto.from_dict(tensors=dict(input_ids=torch.zeros(1,1,dtype=torch.long)),
                non_tensors=dict(data_source=np.array(['appworld'],dtype=object)),meta_info=gen_batch.meta_info)
            prompt=collector.preprocess_batch(carrier,dict(raw_prompt_ids=[ids],sampling_kwargs=[options]))
            ref=servers[rank].generate_tokens.remote(ids,options,key)
            inflight[key]=(rank,prompt,ref)
            ref.future().add_done_callback(lambda _,key=key:pool.output.put(('async_ready',key)))
            calls+=1
        elif event[0]=='async_ready':
            key=event[1];rank,prompt,ref=inflight.pop(key)
            if key in aborted:
                # The owner's cancellation ended this environment request.
                continue
            output=ray.get(ref)
            row=response_carrier(prompt,output,owner.config.data.max_response_length,
                owner.tokenizer.pad_token_id,owner.tokenizer.eos_token_id)
            reply=policy_reply(row,0);records[key]=row
            pool.replies[rank].put((key,reply));tokens+=len(reply.token_ids)
            print(f'[loop_transport] calls={calls} rpc_mode=native_async batch_requests=1 '
                  f'requests={len(records)} generated_tokens={tokens} '
                  f'elapsed={time.monotonic()-started:.1f}s completed_ranks={len(results)}',flush=True)
        else:
            raise RuntimeError(f'Unexpected LOOP transport event: {event[0]}')
        # Cancellation is LOOP's existing signal. vLLM owns request abort.
        for key,(rank,_,_) in inflight.items():
            if key not in aborted and pool.cancellations[rank].is_set():
                servers[rank].abort_request.remote(key);aborted.add(key)
    owner.records=records
    rows=[]
    from uuid import uuid4
    for rank in sorted(results):
        scenarios,groups=results[rank]
        for scenario,group in zip(scenarios,groups):
            uid=str(uuid4());rows.extend((scenario,rollout,uid) for rollout in group)
    return owner.to_batch(rows,records)

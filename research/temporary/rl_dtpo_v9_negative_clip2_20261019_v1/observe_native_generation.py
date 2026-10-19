"""Observe eight existing generation calls per rank, then restore the owner.

Use the original metadata RPC and ConsoleLogger stream. No new rollout,
environment call, RNG operation, engine configuration, or training tensor.
"""
import hashlib
from pathlib import Path

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
wrapper=Path(__file__).with_name('inspect_native_workers.py')
assert hashlib.sha256(wrapper.read_bytes()).hexdigest()=='4ce78954d689164245c6ec9ea7580680754ed841a85dc4e4cc4ebcc0019f59ba'
code=wrapper.read_text()
anchor="exec(compile(code,str(owner),'exec'),globals())"
assert code.count(anchor)==1
observation='''    engine=getattr(getattr(actor,'rollout',None),'inference_engine',None)
    if engine is not None and not getattr(engine,'_native_efficiency_observer_installed',False):
        import json,time
        native_generate=engine.generate
        count=[0]
        def observe_existing_generate(*args,**kwargs):
            started=time.time()
            outputs=native_generate(*args,**kwargs)
            finished=time.time()
            try:
                prompts=kwargs.get('prompts')
                if prompts is None:prompts=kwargs.get('prompt_token_ids',[])
                lengths=[len(p.get('prompt_token_ids',[])) if isinstance(p,dict) else len(p) for p in prompts]
                generated=[len(sample.token_ids) for output in outputs for sample in output.outputs]
                cached=[getattr(output,'num_cached_tokens',None) for output in outputs]
                cached=[x for x in cached if isinstance(x,int)]
                metrics=[getattr(output,'metrics',None) for output in outputs]
                ttft=[];latencies=[]
                for metric in metrics:
                    arrival=getattr(metric,'arrival_time',None)
                    first=getattr(metric,'first_token_time',None)
                    finish=getattr(metric,'finished_time',None)
                    if arrival is not None and first is not None:ttft.append(first-arrival)
                    if arrival is not None and finish is not None:latencies.append(finish-arrival)
                record=dict(pid=os.getpid(),rank=actor.rank,call=count[0]+1,started=started,finished=finished,
                    native_generate_seconds=finished-started,requests=len(lengths),
                    prompt_token_lengths=lengths,generated_token_lengths=generated,
                    cached_token_counts=cached,cache_statistics_present=len(cached)==len(outputs),
                    native_request_TTFT_seconds=ttft,native_request_latency_seconds=latencies,
                    additional_generation_calls=0,original_outputs_returned_unchanged=True)
                print('[native_generation_efficiency] '+json.dumps(record),flush=True)
            except Exception as error:
                print('[native_generation_efficiency_observer_error] '+repr(error),flush=True)
            finally:
                count[0]+=1
                if count[0]>=8:engine.generate=native_generate
            return outputs
        engine.generate=observe_existing_generate
        engine._native_efficiency_observer_installed=True
    result['generation_observer_installed']=bool(engine is not None)
    result['generation_observer_max_existing_calls']=8
    result['additional_generation_calls']=0
    return result'''
code=code.replace(anchor,"assert code.count('    return result')==1\ncode=code.replace('    return result',"+repr(observation)+")\n"+anchor)
exec(compile(code,str(wrapper),'exec'),globals())

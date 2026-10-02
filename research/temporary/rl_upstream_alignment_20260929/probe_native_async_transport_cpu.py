"""Compare the async boundary with native owner conversion on CPU only.

Actual RequestOutput/SamplingParams and pinned owner methods are used. Engine
outputs are recorded fixtures: this is not a model or numerical speed test.
"""
import argparse
import ast
import asyncio
from copy import deepcopy
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--entry', type=Path, required=True)
    parser.add_argument('--original-spmd', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.path[:0] = [str(args.entry), str(args.candidate)]
    import numpy as np
    import psutil
    import torch
    from omegaconf import OmegaConf
    from vllm import SamplingParams
    from vllm.outputs import RequestOutput, CompletionOutput
    from vllm.logprobs import Logprob
    from vllm.v1.engine.async_llm import AsyncLLM
    from verl import DataProto
    from verl.utils.debug import performance
    from verl.workers.rollout.vllm_rollout import vllm_rollout_spmd as owner
    from verl.workers.rollout.vllm_rollout import vllm_async_server as server_module
    from loop_async_transport import response_carrier
    from owner_environment_transport import policy_reply
    from owner_rollout_scope import native_rollout_scope

    started = time.time()
    receipt = dict(started_unix=started, scope=__doc__, checks=[], sources={})
    for module in [owner,server_module,sys.modules['loop_async_transport'],
                   sys.modules['owner_rollout_scope']]:
        path = Path(module.__file__)
        receipt['sources'][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    original_source = args.original_spmd.read_text()
    tree = ast.parse(original_source)
    cls = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='vLLMRollout')
    namespace = dict(vars(owner))
    exec(compile(ast.Module(body=[cls],type_ignores=[]),str(args.original_spmd),'exec'),namespace)
    baseline_class = namespace['vLLMRollout']
    performance._get_current_mem_info = lambda: (0.,0.,0.,0.)

    def artifact(key, ids, reason, text):
        sample = CompletionOutput(index=0,text=text,token_ids=ids,cumulative_logprob=-.25*len(ids),
            logprobs=[{t:Logprob(logprob=-.25,rank=1)} for t in ids],finish_reason=reason)
        return RequestOutput(request_id=key,prompt='unused',prompt_token_ids=[11,12],
            prompt_logprobs=None,outputs=[sample],finished=True)

    class RecordedOutputEngine:
        def generate(self, *, prompts, sampling_params, **kwargs):
            count = sampling_params.n if isinstance(sampling_params,SamplingParams) else 1
            result=[]
            for row,prompt in enumerate(prompts):
                ids = [7,31] if row%2 == 0 else [41,42,9]
                value=artifact(str(row),ids,'length' if row%2==0 else 'stop','native exact text')
                if count>1:
                    value.outputs = [deepcopy(value.outputs[0]) for _ in range(count)]
                result.append(value)
            return result

    def instance(cls, n=1):
        value=object.__new__(cls)
        value.config=OmegaConf.create(dict(response_length=8,free_cache_engine=False,
            val_kwargs=dict(top_k=-1,top_p=1.,temperature=0.)))
        value.pad_token_id=0
        value.sampling_params=SamplingParams(n=n,max_tokens=8,logprobs=0)
        value.lora_kwargs={}
        value.inference_engine=RecordedOutputEngine()
        return value

    def prompts(active=None, overrides=True, dimensions=2, mode='sample'):
        ids=torch.tensor([[0,0,11,12],[0,11,12,13]])
        mask=ids.ne(0).long()
        pos=(mask.cumsum(-1)-1).clamp_min(0)
        if dimensions==3:pos=pos[:,None,:].expand(-1,3,-1).clone()
        metadata={}
        if overrides:
            metadata['owner_sampling_kwargs']=np.array([dict(max_tokens=8,stop_token_ids=[9])]*2,dtype=object)
        if active is not None:metadata['rollout_active_mask']=np.array(active)
        meta_info=dict(eos_token_id=7,do_sample=mode!='greedy',validate=mode=='validate')
        return DataProto.from_dict(tensors=dict(input_ids=ids,attention_mask=mask,position_ids=pos),
            non_tensors=metadata,meta_info=meta_info)

    def equal(actual, expected):
        assert actual.batch.keys()==expected.batch.keys()
        for key in expected.batch.keys():
            torch.testing.assert_close(actual.batch[key],expected.batch[key],rtol=0,atol=0)
        assert actual.non_tensor_batch.keys()==expected.non_tensor_batch.keys()
        for key in actual.non_tensor_batch:
            np.testing.assert_array_equal(actual.non_tensor_batch[key],expected.non_tensor_batch[key])
        assert actual.meta_info==expected.meta_info

    cases=0
    for mode in ['sample','greedy','validate']:
        for overrides in [False,True]:
            for active in [None,[True,True],[False,True],[False,False]]:
                for dim in [2,3]:
                    a=instance(owner.vLLMRollout).generate_sequences(prompts(active,overrides,dim,mode))
                    b=instance(baseline_class).generate_sequences(prompts(active,overrides,dim,mode))
                    equal(a,b);cases+=1
    for active in [None,[False,True],[False,False]]:
        equal(instance(owner.vLLMRollout,2).generate_sequences(prompts(active,False)),
              instance(baseline_class,2).generate_sequences(prompts(active,False)))
        cases+=1
    receipt['checks'].append(dict(name='Original sync path and moved owner conversion',cases=cases,
        assert_type='Exact transport tensor/metadata equality, not numerical tolerance'))

    for ids,reason in [([31,32,9],'stop'),([7,31],'length'),([],'stop')]:
        value=artifact('carrier',ids,reason,'actual engine text')
        prepared=prompts().select_idxs([0])
        class FixedOutput:
            def generate(self,**kwargs):return [value]
        control=instance(baseline_class);control.inference_engine=FixedOutput()
        expected=control.generate_sequences(deepcopy(prepared))
        actual=response_carrier(prepared,value,8,0,7)
        equal(actual,expected)
        assert policy_reply(actual,0).token_ids==ids
        assert actual.batch['attention_mask'][0,-8:].sum()==len(ids)
    receipt['checks'].append('Async raw RequestOutput calls the same native conversion; custom stops/length/empty preserved')

    async def exercise_raw_api():
        cls=server_module.AsyncvLLMServer.__ray_metadata__.modified_class
        service=object.__new__(cls)
        service.owner_sampling_params=SamplingParams(n=1,max_tokens=8,logprobs=0)
        service.owner_lora_request=object()
        release_long=asyncio.Event()
        values={key:artifact(key,[41,42],'length','exact') for key in ['short','long']}
        calls=[]
        class NativeAPIFixture:
            async def generate(self,**kwargs):
                inspect.signature(AsyncLLM.generate).bind(object(),**kwargs)
                calls.append(kwargs)
                if kwargs['request_id']=='long':await release_long.wait()
                yield values[kwargs['request_id']]
            async def abort(self,key):calls.append(('abort',key))
        service.engine=NativeAPIFixture()
        first=asyncio.create_task(service.generate_tokens([11,12],dict(max_tokens=8),'long'))
        second=asyncio.create_task(service.generate_tokens([11,12],dict(max_tokens=8),'short'))
        assert await second is values['short']
        assert not first.done()
        release_long.set()
        assert await first is values['long']
        await service.abort_request('native-request')
        assert calls[-1]==('abort','native-request')
        for call in calls[:-1]:
            assert call['prompt']==dict(prompt_token_ids=[11,12])
            assert call['lora_request'] is service.owner_lora_request
            assert call['sampling_params'].max_tokens==8
    asyncio.run(exercise_raw_api())
    receipt['checks'].append('Actual native AsyncLLM signature bound; separate requests return without a batch barrier; native abort composed')

    events=[]
    class Manager:
        def wake_up(self):events.append('manager-wake')
        def sleep(self):events.append('manager-sleep')
    @native_rollout_scope
    def collect(self,gen_batch,worker):
        events.append('owner-loop')
        raise ValueError('recorded failure')
    try:collect(SimpleNamespace(async_rollout_manager=Manager()),None,object())
    except ValueError as error:assert str(error)=='recorded failure'
    else:raise AssertionError('Owner error hidden')
    assert events==['manager-wake','owner-loop','manager-sleep']
    receipt['checks'].append('Original manager owns whole-rollout wake/sleep, including original failure propagation')
    receipt.update(passed=True,seconds=time.time()-started,
        process_pss_bytes=psutil.Process().memory_full_info().pss,
        original_spmd_sha256=hashlib.sha256(args.original_spmd.read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()

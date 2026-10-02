"""Native owner LoRA boundary and actual Ray completion delivery on CPU.

Recorded outputs are fixtures, not LLM inference. No model is constructed;
PPO, DT, tolerance, budgets and formal jobs are untouched.
"""
import argparse
import asyncio
from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
from queue import Queue
import sys
from threading import Event
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['candidate','entry','root','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    sys.path[:0]=[str(args.entry),str(args.candidate)]
    import psutil
    import ray
    import torch
    from omegaconf import OmegaConf
    from peft import LoraConfig
    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest
    from vllm.v1.worker.worker_base import WorkerWrapperBase
    from vllm.outputs import RequestOutput,CompletionOutput
    from vllm.logprobs import Logprob
    from verl import DataProto
    from verl.utils.debug import performance
    from verl.workers.sharding_manager.fsdp_vllm import FSDPVLLMShardingManager
    from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import vLLMAsyncRollout
    from verl.workers.rollout.vllm_rollout.vllm_async_server import AsyncvLLMServer
    from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector
    from loop_async_transport import collect

    started=time.time();receipt=dict(started_unix=started,scope=__doc__,checks=[],sources={})
    performance._get_current_mem_info=lambda:(0.,0.,0.,0.)
    params=OrderedDict([('fixture.lora_A.weight',torch.arange(16.).reshape(8,2)),
                        ('fixture.lora_B.weight',torch.arange(16.).reshape(2,8))])
    class AdapterRecording:
        request=None
        def add_lora(self,request):self.request=request;return True
        def list_loras(self):return {self.request.lora_int_id}
    registry=AdapterRecording()
    wrapper=WorkerWrapperBase();wrapper.worker=registry
    sharder=object.__new__(FSDPVLLMShardingManager)
    sharder.inference_engine=wrapper
    sharder.model_runner=SimpleNamespace(model=object())
    sharder.base_sync_done=True
    sharder.update_params(params,peft_config=LoraConfig(r=8,lora_alpha=16))
    assert registry.request.lora_tensors is params
    assert registry.request.peft_config['r']==8 and registry.request.peft_config['lora_alpha']==16
    events=[]
    class ContextRecording:
        def __enter__(self):events.append('original-sharder-enter')
        def __exit__(self,*exc):events.append('original-sharder-exit')
    rollout=vLLMAsyncRollout()
    rollout.sharding_manager=ContextRecording()
    class EngineAPIRecording:
        async def reset_prefix_cache(self):events.append('native-reset-prefix')
        async def sleep(self):rollout.execute_method('sleep')
        async def wake_up(self):rollout.execute_method('wake_up')
        async def collective_rpc(self,method):
            assert method=='list_loras'
            return [wrapper.list_loras()]
    service=object.__new__(AsyncvLLMServer.__ray_metadata__.modified_class)
    service.config=OmegaConf.create(dict(model=dict(lora_rank=8)))
    service.engine=EngineAPIRecording()
    asyncio.run(service.sleep());assert rollout.is_sleep
    asyncio.run(service.wake_up());assert not rollout.is_sleep
    assert isinstance(service.owner_lora_request,LoRARequest)
    assert service.owner_lora_request.lora_int_id==registry.request.lora_int_id
    assert events==['native-reset-prefix','original-sharder-exit','original-sharder-enter']
    receipt['checks'].append('Original update_params forwards exact CPU tensors/rank8/alpha16 through native WorkerWrapperBase; server reads the same adapter ID')
    receipt['checks'].append('Actual original server/rollout sleep-wake methods use existing sharder callbacks; no second state machine')

    active=json.loads((args.root/'active-training.json').read_text())
    job=next(j for j in active['jobs'] if j['task']=='AppWorld')
    driver=psutil.Process(job['pid'])
    assert abs(driver.create_time()-job['observed_process_created_unix'])<.02
    gcs=next(p for p in driver.children(recursive=True) if p.name()=='gcs_server')
    port=next(a.split('=',1)[1] for a in gcs.cmdline() if a.startswith('--gcs_server_port='))
    ray.init(address=f'127.0.0.1:{port}',namespace=f'async-delivery-cpu-{os.getpid()}',ignore_reinit_error=True)
    @ray.remote(num_cpus=0)
    class NativeCompletionFixture:
        def __init__(self):self.release=asyncio.Event()
        async def generate_tokens(self,ids,options,key):
            if key=='long':await self.release.wait()
            token=ids[-1]+10
            sample=CompletionOutput(index=0,text=key,token_ids=[token],cumulative_logprob=-.25,
                logprobs=[{token:Logprob(logprob=-.25,rank=1)}],finish_reason='length')
            return RequestOutput(request_id=key,prompt='fixture',prompt_token_ids=ids,
                prompt_logprobs=None,outputs=[sample],finished=True)
        async def allow_long(self):self.release.set()
        async def abort_request(self,key):self.release.set()
    actor=NativeCompletionFixture.remote()
    class Requests:
        processes=[object()]
        def __init__(self):
            self.output=Queue();self.replies=[Queue()];self.cancellations=[Event()]
        def start(self):pass
        def event(self):return self.output.get(timeout=20)
        def completion(self,key):self.output.put(('completion',0,key,dict(
            prompt=[11 if key=='long' else 12],model='native-fixture',stream=False,
            max_tokens=1,logprobs=1)))
    requests=Requests()
    owner=SimpleNamespace(processes=requests,config=SimpleNamespace(data=SimpleNamespace(max_response_length=8)),
        tokenizer=SimpleNamespace(pad_token_id=0,eos_token_id=7),to_batch=lambda rows,records:records)
    collector=object.__new__(TrajectoryCollector)
    collector.tokenizer=owner.tokenizer
    collector.config=OmegaConf.create(dict(data=dict(max_prompt_length=8)))
    collector.async_rollout_manager=SimpleNamespace(async_llm_servers=[actor])
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future=executor.submit(collect,owner,collector,SimpleNamespace(meta_info={}))
            requests.completion('long');requests.completion('short')
            key,reply=requests.replies[0].get(timeout=20)
            assert key=='short' and reply.token_ids==[22] and reply.text=='short'
            assert not future.done()
            ray.get(actor.allow_long.remote())
            key,reply=requests.replies[0].get(timeout=20)
            assert key=='long' and reply.token_ids==[21]
            requests.output.put(('result',0,([],[])))
            records=future.result(timeout=20)
            assert set(records)=={'short','long'}
        receipt['checks'].append('Actual Ray futures + original collector + exact output converter return short request while same-server long request remains unfinished')
    finally:
        ray.kill(actor);ray.shutdown()
    for module in ['loop_async_transport','verl.workers.rollout.vllm_rollout.vllm_async_server',
                   'verl.workers.sharding_manager.fsdp_vllm']:
        p=Path(sys.modules[module].__file__);receipt['sources'][str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    receipt.update(passed=True,seconds=time.time()-started,
        process_pss_bytes=psutil.Process().memory_full_info().pss,
        formal_driver=dict(pid=driver.pid,created_unix=driver.create_time()))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()

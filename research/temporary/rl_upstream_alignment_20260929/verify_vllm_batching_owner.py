"""Pinned vLLM hybrid test_batching on captured AppWorld inputs, without HF.

Original generation helpers and comparison execute unchanged. The real model,
saved token prompts and MetaX engine configuration are additional test fixtures.
"""
import ast
import argparse
from contextlib import contextmanager
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time
from types import MethodType, SimpleNamespace
from uuid import uuid4
import warnings

import torch
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

root=Path(__file__).resolve().parent
report=dict(scope=__doc__,status='running',sources={},cases=[])
path=root/'vllm-batching-owner.json'
start=time.perf_counter()
def record(phase):
    report.update(phase=phase,elapsed=time.perf_counter()-start)
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(phase=phase,elapsed=report['elapsed'])),flush=True)
ns=dict(torch=torch,warnings=warnings,SamplingParams=SamplingParams)
def owner(filename,name,class_name=None):
    source=root/filename
    report['sources'][filename]=hashlib.sha256(source.read_bytes()).hexdigest()
    nodes=ast.parse(source.read_text()).body
    if class_name:
        nodes=next(n for n in nodes if isinstance(n,ast.ClassDef) and n.name==class_name).body
    node=next(n for n in nodes if isinstance(n,ast.FunctionDef) and n.name==name)
    node.decorator_list=[]
    tree=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),node],type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree),str(source),'exec'),ns)
    return ns[name]
compare=owner('vllm015-test-utils.py','check_logprobs_close')


class ExistingNativeAsync:
    """Test-only RPC carrier; the formal native engine owns all inference.

    Concurrent formal requests may coexist. Sequential and concurrent fixture
    submission are compared by the original criterion; this is not an isolated
    batch-one benchmark, and no engine or lifecycle is created here.
    """

    def __init__(self, source_root):
        import psutil
        import ray
        self.ray=ray
        job=next(j for j in json.loads((source_root/'active-training.json').read_text())['jobs']
                 if j['task']=='AppWorld')
        driver=psutil.Process(job['pid'])
        assert driver.create_time()==job['observed_process_created_unix']
        launch=json.loads((Path(job['output'])/'launch.json').read_text())
        assert launch['options']['actor_rollout_ref.rollout.mode']=='async'
        sys.path.insert(0,job['entry'])
        from owner_environment_transport import owner_sampling_params
        self.owner_sampling_params=owner_sampling_params
        gcs=next(p for p in driver.children(recursive=True) if p.name()=='gcs_server')
        port=next(a.split('=',1)[1] for a in gcs.cmdline() if a.startswith('--gcs_server_port='))
        ray.init(address=f'127.0.0.1:{port}',namespace='native-batching-comparison-'+uuid4().hex,
                 log_to_driver=False)
        names=[n for n in ray.util.list_named_actors(all_namespaces=True)
               if n['name']=='async_llm_server_0']
        assert len(names)==1,names
        self.server=ray.get_actor(names[0]['name'],namespace=names[0]['namespace'])
        report['formal_native_engine']=dict(driver_pid=driver.pid,driver_created_unix=driver.create_time(),
            entry=job['entry'],verl_root=job['verl_root'],server=names[0],
            max_num_seqs=launch['options']['actor_rollout_ref.rollout.max_num_seqs'],
            lora_rank=launch['options']['actor_rollout_ref.model.lora_rank'],
            lora_alpha=launch['options']['actor_rollout_ref.model.lora_alpha'])
        report['native_request_outputs']=[]

    def generate(self, inputs, sampling_params, **kwargs):
        fields=inspect.signature(SamplingParams.from_optional).parameters
        payload={name:getattr(sampling_params,name) for name in fields}
        # Reuse the actual transport constructor. Every public field from the
        # original test helper is sent, not new hand-picked sampling defaults.
        clone=self.owner_sampling_params(SamplingParams(),[payload],None)[0]
        assert all(getattr(clone,k)==getattr(sampling_params,k)
                   for k in sampling_params.__struct_fields__)
        refs=[self.server.generate_tokens.remote(p['prompt_token_ids'],payload,
            'official-batching-'+uuid4().hex) for p in inputs]
        outputs=self.ray.get(refs)
        for output in outputs:
            adapter=getattr(output,'lora_request',None)
            report['native_request_outputs'].append(dict(request_id=output.request_id,
                lora_id=adapter.lora_int_id if adapter is not None else None,
                prompt_tokens=len(output.prompt_token_ids),cached_tokens=output.num_cached_tokens,
                generated_tokens=sum(len(s.token_ids) for s in output.outputs)))
        return outputs

    def close(self):
        self.ray.shutdown()


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-async-root',type=Path,
                        help='Use an already awake formal native server; no engine or wake/sleep call.')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.native_async_root:
        assert args.output is not None,'Keep native async evidence separate from existing baseline receipts'
        assert not args.output.exists(),'Preserve an existing comparison instead of overwriting it'
        path=args.output
        path.parent.mkdir(parents=True,exist_ok=True)
        report['scope']=(__doc__+' Existing formal AsyncLLM transport; original sampling/helpers/assertion; '
            'serial versus concurrent submission under live load, not isolated batch-one or throughput.')
    use_lora=os.environ.get('DT_BATCH_LORA')=='1'
    if use_lora and not args.native_async_root:
        path=root/'vllm-batching-owner-lora.json'
    if args.native_async_root:
        report['adapter_fixture']='current_formal_owner_adapter'
    else:
        report['nonzero_lora_fixture']=use_lora
    original=root/'vllm015-hybrid.py'
    report['sources'][original.name]=hashlib.sha256(original.read_bytes()).hexdigest()
    source_fn=next(n for n in ast.parse(original.read_text()).body
        if isinstance(n,ast.FunctionDef) and n.name=='test_batching')
    parameters={d.args[0].value:ast.literal_eval(d.args[1]) for d in source_fn.decorator_list
        if isinstance(d,ast.Call) and isinstance(d.args[0],ast.Constant)
        and d.args[0].value in ('max_tokens','num_logprobs')}
    max_tokens,num_logprobs=parameters['max_tokens'][0],parameters['num_logprobs'][0]
    saved=torch.load(root/'native-probability-gap.pt',map_location='cpu',weights_only=True)
    width=saved['responses'].shape[-1]
    prompts=[ids[:-width][mask[:-width].bool()].tolist()
             for ids,mask in zip(saved['input_ids'][:4],saved['attention_mask'][:4])]
    report['prompt_tokens']=list(map(len,prompts))
    report['max_tokens']=max_tokens
    report['num_logprobs']=num_logprobs
    record('load_original_engine')
    llm=(ExistingNativeAsync(args.native_async_root) if args.native_async_root else
         LLM(model=os.environ['MODEL_PATH'],dtype='bfloat16',max_model_len=32768,max_num_seqs=4,
             max_num_batched_tokens=32768,gpu_memory_utilization=.75,enforce_eager=True,
             enable_prefix_caching=True,enable_lora=use_lora,max_lora_rank=8,
             limit_mm_per_prompt={'image':0,'video':0}))
    runner=SimpleNamespace(llm=llm,get_inputs=lambda inputs,**kwargs:[dict(prompt_token_ids=p) for p in inputs])
    runner._final_steps_generate_w_logprobs=owner('vllm015-conftest.py','_final_steps_generate_w_logprobs','VllmRunner')
    for name in ('generate_w_logprobs','generate_greedy_logprobs'):
        setattr(runner,name,MethodType(owner('vllm015-conftest.py',name,'VllmRunner'),runner))
    individual=[]
    extra={'lora_request':LoRARequest('owner_fixture',1,str(root/'vllm-logprob-test-adapter-bfloat16'))} if use_lora and not args.native_async_root else {}
    try:
        for i,prompt in enumerate(prompts):
            record(f'original_individual_generate_{i}')
            single,=runner.generate_greedy_logprobs([prompt],max_tokens,num_logprobs,**extra)
            individual.append(single)
        record('original_batched_generate')
        batched=runner.generate_greedy_logprobs(prompts,max_tokens,num_logprobs,**extra)
        torch.save(dict(individual=individual,batched=batched,prompts=prompts),path.with_suffix('.pt'))
        record('original_batching_comparison')
        compare(outputs_0_lst=individual,outputs_1_lst=batched,
            name_0='for_loop_vllm',name_1='batched_vllm')
        report['status']='passed_original_batching_comparison'
    except Exception as exc:
        report.update(status=('failed_original_batching_comparison'
            if report['phase']=='original_batching_comparison' else 'failed_generation_before_comparison'),
            error=str(exc))
        raise
    finally:
        if args.native_async_root:llm.close()
        record('finished')

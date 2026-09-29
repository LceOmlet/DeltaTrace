"""Pinned vLLM hybrid test_batching on captured AppWorld inputs, without HF.

Original generation helpers and comparison execute unchanged. The real model,
saved token prompts and MetaX engine configuration are additional test fixtures.
"""
import ast
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import time
from types import MethodType, SimpleNamespace
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
if __name__ == "__main__":
    use_lora=os.environ.get('DT_BATCH_LORA')=='1'
    if use_lora:
        path=root/'vllm-batching-owner-lora.json'
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
    llm=LLM(model=os.environ['MODEL_PATH'],dtype='bfloat16',max_model_len=32768,max_num_seqs=4,
        max_num_batched_tokens=32768,gpu_memory_utilization=.75,enforce_eager=True,
        enable_prefix_caching=True,enable_lora=use_lora,max_lora_rank=8,
        limit_mm_per_prompt={'image':0,'video':0})
    runner=SimpleNamespace(llm=llm,get_inputs=lambda inputs,**kwargs:[dict(prompt_token_ids=p) for p in inputs])
    runner._final_steps_generate_w_logprobs=owner('vllm015-conftest.py','_final_steps_generate_w_logprobs','VllmRunner')
    for name in ('generate_w_logprobs','generate_greedy_logprobs'):
        setattr(runner,name,MethodType(owner('vllm015-conftest.py',name,'VllmRunner'),runner))
    individual=[]
    extra={'lora_request':LoRARequest('owner_fixture',1,str(root/'vllm-logprob-test-adapter-bfloat16'))} if use_lora else {}
    for i,prompt in enumerate(prompts):
        record(f'original_individual_generate_{i}')
        single,=runner.generate_greedy_logprobs([prompt],max_tokens,num_logprobs,**extra)
        individual.append(single)
    record('original_batched_generate')
    batched=runner.generate_greedy_logprobs(prompts,max_tokens,num_logprobs,**extra)
    torch.save(dict(individual=individual,batched=batched,prompts=prompts),path.with_suffix('.pt'))
    try:
        compare(outputs_0_lst=individual,outputs_1_lst=batched,
            name_0='for_loop_vllm',name_1='batched_vllm')
        report['status']='passed_original_batching_comparison'
    except AssertionError as exc:
        report.update(status='failed_original_batching_comparison',error=str(exc))
        raise
    finally:
        record('finished')

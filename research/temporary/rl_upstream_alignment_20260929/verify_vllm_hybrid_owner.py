"""Apply pinned vLLM hybrid model/sleep tests to the real Qwen3.5 runtime.

The official hybrid test explicitly uses HF. Original HF readout, generation
helpers and model/sleep comparisons execute from pinned test source. Only model
loading, native saved token inputs, PEFT fixture and result transport are local.
This adds a Qwen3.5/MetaX case, not a claim that upstream CI tested this machine.
"""
import ast
import gc
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
from types import MethodType, SimpleNamespace
import warnings
import torch
import torch.nn.functional as F
from transformers import AutoModelForImageTextToText, AutoTokenizer
from peft import PeftModel
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


def main():
    root = Path(__file__).resolve().parent
    report = dict(scope=__doc__, status='running', sources={}, cases=[], max_tokens=64,
                  num_logprobs=5, dtype='bfloat16', context_cap=32768,
                  owner_test='v0.15.0/tests/models/language/generation/test_hybrid.py::test_models',
                  numerical_scope='Original model comparison tests token/top-k membership, not scalar logprob allclose. The earlier additional FP16-sampler assertion on BF16 remains a separate failed diagnostic.')
    path = root/'vllm-hybrid-owner.json'
    start = time.perf_counter()
    def record(phase):
        report.update(phase=phase, elapsed=time.perf_counter()-start)
        path.write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(dict(phase=phase,elapsed=report['elapsed'])),flush=True)
    ns = dict(torch=torch, F=F, warnings=warnings, SamplingParams=SamplingParams)
    def owner(filename, name, class_name=None):
        source = root/filename
        report['sources'][filename] = hashlib.sha256(source.read_bytes()).hexdigest()
        nodes = ast.parse(source.read_text()).body
        if class_name:
            nodes = next(n for n in nodes if isinstance(n,ast.ClassDef) and n.name==class_name).body
        node = next(n for n in nodes if isinstance(n,ast.FunctionDef) and n.name==name)
        node.decorator_list = []
        tree = ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),node],type_ignores=[])
        exec(compile(ast.fix_missing_locations(tree),str(source),'exec'),ns)
        return ns[name]
    compare = owner('vllm015-test-utils.py','check_logprobs_close')
    # The exact upstream sleep assertion, without model-specific memory limits.
    source = root/'vllm015-cumem.py'
    report['sources'][source.name]=hashlib.sha256(source.read_bytes()).hexdigest()
    fn=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='test_end_to_end')
    assertion=next(n for n in ast.walk(fn) if isinstance(n,ast.Assert) and 'output2' in ast.unparse(n.test))
    sleep_assert=compile(ast.fix_missing_locations(ast.Module(body=[assertion],type_ignores=[])),str(source),'exec')
    # Derive owner test length/number from its original parameter decorators.
    hybrid = root/'vllm015-hybrid.py'
    report['sources'][hybrid.name]=hashlib.sha256(hybrid.read_bytes()).hexdigest()
    hybrid_fn=next(n for n in ast.parse(hybrid.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='test_models')
    params={d.args[0].value:ast.literal_eval(d.args[1]) for d in hybrid_fn.decorator_list
            if isinstance(d,ast.Call) and isinstance(d.args[0],ast.Constant) and d.args[0].value in ('max_tokens','num_logprobs')}
    max_tokens,num_logprobs=params['max_tokens'][0],params['num_logprobs'][0]
    names=['native-fresh-rollout-parity-request-1.json','native-fresh-rollout-parity-Webshop-request-1.json']
    ids=[json.loads((root/name).read_text())[0]['prompt_ids'] for name in names]
    report['prompts']=[dict(source=name,tokens=len(tokens)) for name,tokens in zip(names,ids)]
    model_path=os.environ['MODEL_PATH']
    tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
    record('load_hf_reference')
    hf=AutoModelForImageTextToText.from_pretrained(model_path,dtype=torch.bfloat16,
        device_map='cuda:1',local_files_only=True,attn_implementation='sdpa').eval()
    hf_runner=SimpleNamespace(model=hf,tokenizer=tokenizer,
        get_inputs=lambda prompts,**kwargs:[dict(input_ids=torch.tensor([p]),attention_mask=torch.ones(1,len(p),dtype=torch.long)) for p in prompts],
        wrap_device=lambda inputs:{k:v.to('cuda:1') for k,v in inputs.items()})
    for name in ('_hidden_states_to_seq_logprobs','_hidden_states_to_logprobs','generate_greedy_logprobs_limit'):
        setattr(hf_runner,name,MethodType(owner('vllm015-conftest.py',name,'HfRunner'),hf_runner))
    references={}
    adapter=root/'vllm-logprob-test-adapter-bfloat16'
    with torch.no_grad():
        references['base']=hf_runner.generate_greedy_logprobs_limit(ids,max_tokens,num_logprobs)
        record('hf_nonzero_lora')
        hf=PeftModel.from_pretrained(hf,adapter).eval()
        hf_runner.model=hf
        references['lora']=hf_runner.generate_greedy_logprobs_limit(ids,max_tokens,num_logprobs)
    torch.save(references,root/'vllm-hybrid-owner-hf.pt')
    del hf_runner,hf
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.set_device(0)
    record('load_vllm_owner')
    llm=LLM(model=model_path,dtype='bfloat16',max_model_len=32768,max_num_seqs=32,
        max_num_batched_tokens=32768,gpu_memory_utilization=.75,enforce_eager=True,
        enable_prefix_caching=True,enable_sleep_mode=True,enable_lora=True,max_lora_rank=8,
        limit_mm_per_prompt={'image':0,'video':0})
    runner=SimpleNamespace(llm=llm,get_inputs=lambda prompts,**kwargs:[dict(prompt_token_ids=p) for p in prompts])
    runner._final_steps_generate_w_logprobs=owner('vllm015-conftest.py','_final_steps_generate_w_logprobs','VllmRunner')
    for name in ('generate_w_logprobs','generate_greedy_logprobs'):
        setattr(runner,name,MethodType(owner('vllm015-conftest.py',name,'VllmRunner'),runner))
    def generate(kind, phase):
        record(phase)
        kwargs={} if kind=='base' else dict(lora_request=LoRARequest('numeric-fixture',1,str(adapter)))
        output=runner.generate_greedy_logprobs(ids,max_tokens,num_logprobs,**kwargs)
        torch.save(output,root/('vllm-hybrid-owner-'+phase+'.pt'))
        result=dict(phase=phase,tokens=[len(v[0]) for v in output])
        try:
            compare(outputs_0_lst=references[kind],outputs_1_lst=output,name_0='hf',name_1='vllm')
            result['official_model_check_passed']=True
        except AssertionError:
            result.update(official_model_check_passed=False,failure=traceback.format_exc())
        report['cases'].append(result)
        record(phase+':finished')
        return output
    for kind in ('base','lora'):
        first=generate(kind,kind+'_fresh')
        llm.sleep(level=1)
        llm.wake_up(tags=['weights'])
        llm.wake_up(tags=['kv_cache'])
        second=generate(kind,kind+'_wake')
        result=dict(phase=kind+'_sleep_output')
        try:
            for left,right in zip(first,second):
                exec(sleep_assert,dict(output=[SimpleNamespace(outputs=[SimpleNamespace(text=left[1])])],
                    output2=[SimpleNamespace(outputs=[SimpleNamespace(text=right[1])])]))
            result['official_sleep_assertion_passed']=True
        except AssertionError:
            result.update(official_sleep_assertion_passed=False,failure=traceback.format_exc())
        report['cases'].append(result)
    report['status']='passed' if all(v.get('official_model_check_passed',v.get('official_sleep_assertion_passed')) for v in report['cases']) else 'failed'
    record('finished')


if __name__=='__main__':
    main()

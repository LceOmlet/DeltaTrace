"""Run vLLM 0.15's unmodified numerical validator on the actual Qwen/MetaX path.

Only test plumbing is local. The HF hidden-state readout and all assertions are
compiled from pinned upstream test source, without copying or relaxing them.
Two actual collector prompts, five greedy tokens (the owner's numerical test
length), BF16 runtime, 32768 context cap. This is not a task success-rate test.
"""
import ast
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace
import traceback
import warnings
import regex as re
import torch
import torch.nn.functional as F
from transformers import AutoModelForImageTextToText, AutoTokenizer
from vllm import LLM, SamplingParams

def lora_fingerprints(model):
    """Read owner tensors through LLM.apply_model; no execution path changes."""
    result = {}
    for name, module in model.named_modules():
        for attr in ('lora_a_stacked', 'lora_b_stacked'):
            values = getattr(module, attr, ())
            values = (values,) if isinstance(values, torch.Tensor) else values
            for index, tensor in enumerate(values):
                value = tensor.detach().contiguous().view(torch.uint8).cpu().numpy()
                result[f'{name}.{attr}.{index}'] = hashlib.sha256(value.tobytes()).hexdigest()
    return result

def install_first_layer_trace(model):
    """Diagnostic PyTorch hooks only; retain first invocation of each owner op."""
    model._dt_diagnostic_trace = {}
    def tensors(value, prefix=''):
        if isinstance(value, torch.Tensor):
            return {prefix: value.detach().cpu().clone()}
        result = {}
        if isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                result.update(tensors(item, prefix + '/' + str(index)))
        elif isinstance(value, dict):
            for name, item in value.items():
                result.update(tensors(item, prefix + '/' + str(name)))
        return result
    handles = []
    for name, module in model.named_modules():
        if '.layers.0.' not in name and not name.endswith('.embed_tokens'):
            continue
        def before(module, args, kwargs, name=name):
            key = name + ':input'
            if key not in model._dt_diagnostic_trace:
                model._dt_diagnostic_trace[key] = tensors((args, kwargs))
        def after(module, args, kwargs, output, name=name):
            key = name + ':output'
            if key not in model._dt_diagnostic_trace:
                model._dt_diagnostic_trace[key] = tensors(output)
        handles.append(module.register_forward_pre_hook(before, with_kwargs=True))
        handles.append(module.register_forward_hook(after, with_kwargs=True))
    model._dt_diagnostic_handles = handles
    return len(handles)

def save_first_layer_trace(model, path):
    torch.save(model._dt_diagnostic_trace, path)
    count = len(model._dt_diagnostic_trace)
    model._dt_diagnostic_trace.clear()
    return count

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dtype', choices=['bfloat16', 'half'], default='bfloat16')
    parser.add_argument('--replay', action='store_true', help='Reuse the saved unchanged HF reference and PEFT adapter')
    parser.add_argument('--inspect-lora-state', action='store_true')
    parser.add_argument('--trace-first-layer', action='store_true')
    args = parser.parse_args()
    dtype = torch.bfloat16 if args.dtype == 'bfloat16' else torch.float16
    root = Path(__file__).resolve().parent
    label = args.dtype + ('-pool' if args.replay else '')
    label += '-state' if args.inspect_lora_state else ''
    label += '-trace' if args.trace_first_layer else ''
    report_path = root / f'vllm-owner-logprobs-{label}.json'
    report = dict(scope=__doc__, status='running', cases=[], sources={}, context_cap=32768,
                  dtype=args.dtype, max_tokens=5,
                  numerical_scope='Original numeric fixture uses FP16 Llama. BF16 Qwen is an explicit additional case, not an official published BF16 Qwen tolerance. Original assertions remain unchanged.',
                  adapter_scope='Native PEFT checkpoint names from the multimodal HF owner match the native vLLM owner. Text-only PEFT names in the earlier diagnostic were not a valid vLLM adapter reference.')
    started = time.perf_counter()

    def record(phase):
        report.update(phase=phase, elapsed=time.perf_counter()-started)
        report_path.write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(dict(phase=phase, elapsed=report['elapsed'])), flush=True)

    def owner_function(filename, name, namespace, class_name=None):
        path = root / filename
        source = path.read_text()
        nodes = ast.parse(source).body
        if class_name:
            nodes = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == class_name).body
        node = next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == name)
        module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), node], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), namespace)
        report['sources'][filename] = hashlib.sha256(path.read_bytes()).hexdigest()
        return namespace[name]

    ns = dict(torch=torch, F=F, re=re, warnings=warnings)
    for name in ('assert_incr_detok_str_matches_non_incr_detok_str', 'compute_correct_cumulative_logprob'):
        owner_function('vllm015-sample-utils.py', name, ns)
    validate = owner_function('vllm015-logprobs.py', '_run_and_validate', ns)
    compare_model = owner_function('vllm015-test-utils.py', 'check_logprobs_close', ns)
    readout = owner_function('vllm015-conftest.py', '_hidden_states_to_seq_logprobs', ns, 'HfRunner')

    class Capture:
        def __init__(self, llm):
            self.llm, self.outputs, self.lora_request = llm, None, None
        def generate(self, *args, **kwargs):
            self.outputs = self.llm.generate(*args, lora_request=self.lora_request, **kwargs)
            return self.outputs

    model_path = os.environ['MODEL_PATH']
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    sources = ['native-fresh-rollout-parity-request-1.json', 'native-fresh-rollout-parity-Webshop-request-1.json']
    ids = [json.loads((root/p).read_text())[0]['prompt_ids'] for p in sources]
    prompts = [dict(prompt_token_ids=x) for x in ids]
    report['prompts'] = [dict(source=p, tokens=len(x)) for p, x in zip(sources, ids)]
    record('load_hf_reference')
    hf = None if args.replay else AutoModelForImageTextToText.from_pretrained(model_path, dtype=dtype,
        device_map='cuda:1', local_files_only=True, attn_implementation='sdpa').eval()

    def reference():
        outputs, logprobs = [], []
        with torch.no_grad():
            for prompt in ids:
                inputs = torch.tensor([prompt], device='cuda:1')
                result = hf.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                    use_cache=True, do_sample=False, max_new_tokens=5,
                    output_hidden_states=True, return_dict_in_generate=True)
                outputs.append((result.sequences[0].tolist(), tokenizer.decode(result.sequences[0])))
                # Original vLLM HF reference projection/log-softmax, unchanged.
                logprobs.append([v.cpu() for v in readout(SimpleNamespace(model=hf), result.hidden_states)])
        return outputs, logprobs

    if args.replay:
        saved = torch.load(root/(args.dtype+'-base_fresh.pt'), map_location='cpu', weights_only=False)
        hf_outputs, hf_logprobs = saved['hf_outputs'], saved['hf_logprobs']
    else:
        hf_outputs, hf_logprobs = reference()
    torch.cuda.set_device(0)
    record('load_vllm_owner')
    llm = LLM(model=model_path, dtype=args.dtype, max_model_len=32768,
        max_num_seqs=32, max_num_batched_tokens=32768, gpu_memory_utilization=.75,
        enforce_eager=True, enable_prefix_caching=True, enable_sleep_mode=True,
        enable_lora=True, max_lora_rank=8, limit_mm_per_prompt={'image':0, 'video':0})
    capture = Capture(llm)

    def check(phase):
        record(phase)
        case = dict(phase=phase)
        try:
            validate(vllm_model=SimpleNamespace(llm=capture), test_prompts=prompts,
                vllm_sampling_params=SamplingParams(max_tokens=5, logprobs=5, temperature=0, seed=1984),
                hf_logprobs=hf_logprobs, hf_outputs=hf_outputs,
                logprob_prompt_logprob_list=[(5,None)]*len(prompts), temperature=0., max_tokens=5, do_apc=True)
            case['passed'] = True
        except AssertionError:
            case.update(passed=False, failure=traceback.format_exc())
        # Preserve native artifacts, including failures; no numerical corrections.
        if capture.outputs is not None:
            torch.save(dict(native=capture.outputs, hf_outputs=hf_outputs, hf_logprobs=hf_logprobs), root/(label+'-'+phase+'.pt'))
            case['tokens'] = [v.outputs[0].token_ids for v in capture.outputs]
            case['hf_tokens'] = [v[0][-5:] for v in hf_outputs]
            hf_compare, native_compare = [], []
            for prompt, generated, probabilities, native in zip(ids, hf_outputs, hf_logprobs, capture.outputs):
                generated_ids = generated[0][len(prompt):]
                top = [v[-1].topk(5) for v in probabilities]
                hf_compare.append((generated_ids, tokenizer.decode(generated_ids),
                    [dict(zip(v.indices.tolist(), v.values.tolist())) for v in top]))
                native_compare.append((native.outputs[0].token_ids, native.outputs[0].text, native.outputs[0].logprobs))
            try:
                compare_model(outputs_0_lst=hf_compare, outputs_1_lst=native_compare, name_0='HF', name_1='vLLM')
                case['official_model_generation_check_passed'] = True
            except AssertionError:
                case['official_model_generation_check_passed'] = False
            if args.replay:
                prior_phase = 'lora_fresh' if phase == 'lora_repeat' else phase
                prior = torch.load(root/(args.dtype+'-'+prior_phase+'.pt'), map_location='cpu', weights_only=False)['native']
                # Original vLLM test_end_to_end sleep/wake output assertion.
                sleep_path = root/'vllm015-cumem.py'
                fn = next(n for n in ast.parse(sleep_path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='test_end_to_end')
                assertion = next(n for n in ast.walk(fn) if isinstance(n,ast.Assert) and 'output2' in ast.unparse(n.test))
                code = compile(ast.fix_missing_locations(ast.Module(body=[assertion],type_ignores=[])), str(sleep_path), 'exec')
                try:
                    for left,right in zip(prior,capture.outputs):
                        exec(code, dict(output=[left],output2=[right]))
                    case['official_sleep_output_assertion_vs_before_pool_fix'] = True
                except AssertionError:
                    case['official_sleep_output_assertion_vs_before_pool_fix'] = False
                case['max_native_logprob_difference_vs_before_pool_fix'] = max(abs(lp.logprob-prior[row].outputs[0].logprobs[t][token].logprob)
                    for row,output in enumerate(capture.outputs) for t,values in enumerate(output.outputs[0].logprobs)
                    for token,lp in values.items() if token in prior[row].outputs[0].logprobs[t])
        report['cases'].append(case)
        if args.trace_first_layer and phase.startswith('lora_'):
            from functools import partial
            case['trace_entries'] = llm.apply_model(partial(save_first_layer_trace,
                path=str(root/(label+'-'+phase+'-layer0.pt'))))
        record(phase+':finished')

    try:
        check('base_fresh')
        for cycle in range(2):
            tick = time.perf_counter()
            llm.sleep(level=1)
            llm.wake_up(tags=['weights'])
            llm.wake_up(tags=['kv_cache'])
            report.setdefault('sleep_wake_seconds', []).append(time.perf_counter()-tick)
            check(f'base_wake_{cycle+1}')
        # Real PEFT/vLLM public interfaces; explicit nonzero test adapter, not a
        # claimed trained checkpoint or a substitute for VERL's synchronization.
        from peft import LoraConfig, get_peft_model
        from vllm.lora.request import LoRARequest
        adapter = root/('vllm-logprob-test-adapter-'+args.dtype)
        record('nonzero_peft_reference')
        if args.replay:
            saved = torch.load(root/(args.dtype+'-lora_fresh.pt'), map_location='cpu', weights_only=False)
            hf_outputs, hf_logprobs = saved['hf_outputs'], saved['hf_logprobs']
        else:
            torch.manual_seed(20260929)
            text_linears = [name for name, module in hf.named_modules()
                            if name.startswith('model.language_model.') and isinstance(module, torch.nn.Linear)]
            assert text_linears
            hf = get_peft_model(hf, LoraConfig(r=1, lora_alpha=2, target_modules=text_linears, task_type='CAUSAL_LM'))
            with torch.no_grad():
                for name, param in hf.named_parameters():
                    if 'lora_B' in name:
                        param.normal_(std=0.001)
            hf.eval()
            hf.save_pretrained(adapter)
            hf_outputs, hf_logprobs = reference()
        capture.lora_request = LoRARequest('numeric-fixture', 1, str(adapter))
        if args.trace_first_layer:
            report['trace_hooks'] = llm.apply_model(install_first_layer_trace)
        check('lora_fresh')
        if args.inspect_lora_state:
            report['lora_state_before_sleep'] = llm.apply_model(lora_fingerprints)
            llm.reset_prefix_cache()
            check('lora_repeat')
        llm.sleep(level=1)
        llm.wake_up(tags=['weights'])
        llm.wake_up(tags=['kv_cache'])
        if args.inspect_lora_state:
            report['lora_state_after_wake'] = llm.apply_model(lora_fingerprints)
            report['lora_tensors_byte_equal_after_wake'] = report['lora_state_before_sleep'] == report['lora_state_after_wake']
        check('lora_wake')
        report['status'] = 'passed' if all(c['passed'] for c in report['cases']) else 'failed_official_assertions'
    finally:
        record('finished')


if __name__ == '__main__':
    main()

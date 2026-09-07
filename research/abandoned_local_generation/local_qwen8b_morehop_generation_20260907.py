"""Native local Qwen3-8B generation of a frozen unused original MoreHopQA cohort.

Original author prompt construction, format parser and span helpers are called
unchanged. Generator and answer-review provenance intentionally differ from the
author sampler defaults; no external API, model judge or attribution runs here.
"""
import dataclasses
import hashlib
import importlib.util
import inspect
import json
import os
import sys
import time
import traceback
import zipfile
from pathlib import Path

os.environ['MACA_PATH'] = '/opt/maca'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
p = json.loads((HERE/'protocol.json').read_text())
sha = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()
assert sha(Path(__file__)) == p['study_sha256']
ROOT = Path(p['original_repository'])
sys.path.insert(0, str(ROOT))
for name, digest in p['original_normalized_sources'].items():
    assert hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n', b'\n')).hexdigest() == digest
source = Path(p['sampling_source'])
assert sha(source) == p['sampling_source_sha256']
assert not (HERE/'results.json').exists(), 'Inspect prior results before any restart.'
spec = importlib.util.spec_from_file_location('original_morehop_sampler', ROOT/'exp/exp2/sample_and_filter.py')
sampler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sampler)
from exp.exp2.dataset_utils import CachedExample, DatasetLoader, attach_spans_from_answer, split_boxed_generation
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import transformers
import flash_attn.flash_attn_interface as fa_native

def checkpoint_receipt():
    path = Path(p['checkpoint_receipt'])
    assert sha(path) == p['checkpoint_receipt_sha256']
    receipt = json.loads(path.read_text())
    assert receipt['status'] == 'complete' and receipt['checkpoint'] == p['checkpoint']
    rows = []
    for expected in receipt['files']:
        f = Path(p['checkpoint'])/expected['name']
        before = f.stat()
        digest = sha(f)
        after = f.stat()
        assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
        assert digest == expected['sha256'] and after.st_size == expected['bytes']
        rows.append({'file': expected['name'], 'sha256': digest})
    return rows

started = time.time()
report = {'status': 'initializing', 'protocol': p, 'records': [], 'batches': [],
          'API_calls': 0, 'judge_model_calls': 0, 'attribution_calls': 0,
          'quality_evaluation_calls': 0, 'native_generation_calls': 0,
          'native_root_forwards': 0, 'native_decoder_layer_calls': 0,
          'native_forward_trajectories': 0, 'native_decoder_layer_trajectories': 0}

def save():
    tmp = HERE/'results.partial'
    tmp.write_text(json.dumps(report, ensure_ascii=False, separators=(',', ':')))
    tmp.replace(HERE/'results.json')

def source_receipt():
    modules = [type(model), type(model).generate, fa_native, fa_native.flash_attn_cuda]
    paths = sorted({inspect.getfile(m) for m in modules})
    return {name: sha(Path(name)) for name in paths}

save()
try:
    report['checkpoint_before'] = checkpoint_receipt()
    torch.manual_seed(p['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    model = AutoModelForCausalLM.from_pretrained(p['checkpoint'], device_map={'': 0},
        torch_dtype=torch.float16, attn_implementation='flash_attention_2', local_files_only=True)
    model.eval().requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(p['checkpoint'], local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    assert sha(Path(inspect.getfile(type(model)))) == p['native_model_source_sha256']
    assert model.config._attn_implementation == 'flash_attention_2'
    assert all('forward' not in m.__dict__ for m in model.modules())
    assert not any(m._forward_hooks or m._forward_pre_hooks or m._backward_hooks for m in model.modules())
    original_methods = {name: type(m).forward for name, m in model.named_modules()}
    report['native_sources_before'] = source_receipt()
    report.update(torch_version=torch.__version__, transformers_version=transformers.__version__,
                  device=torch.cuda.get_device_name(), dtype=str(next(model.parameters()).dtype),
                  template_sha256=hashlib.sha256(tokenizer.chat_template.encode()).hexdigest())
    generation_config = GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=p['max_new_tokens'],
        bos_token_id=model.generation_config.bos_token_id,
        eos_token_id=model.generation_config.eos_token_id, pad_token_id=tokenizer.pad_token_id,
        use_cache=True, return_dict_in_generate=True, output_scores=False, output_logits=False)
    report['resolved_generation_config'] = generation_config.to_dict()
    eos = generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else eos
    raw = DatasetLoader(seed=p['seed']).load_raw(str(source), sample=None)
    assert len(raw) == p['maximum_raw_examples']
    selected = p['executed_source_offsets']
    assert selected == list(range(64))
    report['status'] = 'generating'
    save()
    for batch_start in range(0, len(selected), p['batch_size']):
        offsets = selected[batch_start:batch_start+p['batch_size']]
        examples = [raw[i] for i in offsets]
        messages = [sampler.build_gen_messages(ex.prompt) for ex in examples]
        input_lists = [tokenizer.apply_chat_template(m, tokenize=True, add_generation_prompt=True,
                      enable_thinking=False) for m in messages]
        inputs = tokenizer.pad({'input_ids': input_lists}, padding=True, return_tensors='pt').to(model.device)
        assert inputs.input_ids.shape[0] == len(offsets) == p['batch_size']
        for i, ids in enumerate(input_lists):
            assert inputs.input_ids[i][inputs.attention_mask[i].bool()].tolist() == ids
        counters = {'root_forwards': 0, 'root_trajectories': 0, 'decoder_calls': 0, 'decoder_trajectories': 0}
        public_codes = {fa_native.flash_attn_func.__code__: 'flash_attn_func',
                        fa_native.flash_attn_varlen_func.__code__: 'flash_attn_varlen_func'}
        fa_calls = {'flash_attn_func': 0, 'flash_attn_varlen_func': 0}
        fa_examples = []
        def observe_fa(frame, event, result):
            if event == 'return' and frame.f_code in public_codes:
                assert isinstance(result, torch.Tensor), 'Expected real public default FA output.'
                name = public_codes[frame.f_code]
                fa_calls[name] += 1
                if len(fa_examples) < 4:
                    fa_examples.append({'function': name, 'q_shape': list(frame.f_locals['q'].shape),
                        'output_shape': list(result.shape), 'dtype': str(result.dtype)})
        def count_root(_module, args, kwargs):
            value = kwargs.get('input_ids')
            if value is None: value = kwargs.get('inputs_embeds')
            if value is None: value = args[0]
            counters['root_forwards'] += 1
            counters['root_trajectories'] += value.shape[0]
        def count_layer(_module, args):
            counters['decoder_calls'] += 1
            counters['decoder_trajectories'] += args[0].shape[0]
        handles = [model.register_forward_pre_hook(count_root, with_kwargs=True)]
        handles += [layer.register_forward_pre_hook(count_layer) for layer in model.model.layers]
        report['active_offsets'] = offsets
        report['native_generation_calls'] += 1
        save()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        try:
            assert sys.getprofile() is None
            sys.setprofile(observe_fa)
            with torch.no_grad():
                output = model.generate(**inputs, generation_config=generation_config)
            torch.cuda.synchronize()
            seconds = time.perf_counter()-t
            peak = torch.cuda.max_memory_allocated()
        finally:
            sys.setprofile(None)
            for h in handles: h.remove()
        assert counters['decoder_calls'] == counters['root_forwards']*len(model.model.layers)
        assert counters['root_trajectories'] == counters['root_forwards']*len(offsets)
        assert sum(fa_calls.values()) == counters['decoder_calls']
        sequences = output.sequences.cpu().tolist()
        prefix_len = inputs.input_ids.shape[1]
        padded_inputs = inputs.input_ids.cpu().tolist()
        attention_mask = inputs.attention_mask.cpu().tolist()
        for j, (offset, ex, sequence) in enumerate(zip(offsets, examples, sequences)):
            assert sequence[:prefix_len] == padded_inputs[j]
            generated = sequence[prefix_len:]
            end = next((i for i, token in enumerate(generated) if token in eos), None)
            actual_tokens = generated[:end] if end is not None else generated
            generation = tokenizer.decode(actual_tokens, skip_special_tokens=True).strip()
            parsed = split_boxed_generation(generation)
            record = {'source_offset': offset, 'original_source_index': p['sampling_original_indices'][offset],
                'original_id': ex.metadata.get('id'), 'prompt': ex.prompt,
                'reference_answer': ex.metadata.get('reference_answer') or ex.target or '',
                'messages': messages[j], 'actual_generation_input_ids': input_lists[j],
                'actual_left_padded_input_ids': padded_inputs[j], 'actual_attention_mask': attention_mask[j],
                'generated_tokens_including_batch_padding': generated, 'actual_content_tokens': actual_tokens,
                'stop_reason': 'eos' if end is not None else 'length', 'generation': generation,
                'original_format_pass': parsed is not None, 'original_span_pass': False,
                'answer_review_status': 'pending_blind_gold_review', 'candidate_cache_row': None}
            if parsed:
                reasoning, _, answer = parsed
                metadata = dict(ex.metadata)
                metadata.update(reference_answer=record['reference_answer'], judge_response=None,
                    generation_origin='native_local_Qwen3-8B', answer_review_status='pending_blind_gold_review')
                new_ex = CachedExample(prompt=ex.prompt, target=f'{reasoning}\n{answer}' if reasoning else answer,
                    indices_to_explain=None, attr_mask_indices=ex.attr_mask_indices,
                    sink_span=None, thinking_span=None, metadata=metadata)
                new_ex = attach_spans_from_answer(new_ex, tokenizer, answer)
                span_ok = isinstance(new_ex.sink_span, list) and len(new_ex.sink_span) == 2
                record.update(boxed_answer=answer, original_span_pass=span_ok)
                if span_ok:
                    new_ex = dataclasses.replace(new_ex, indices_to_explain=new_ex.sink_span)
                    record['candidate_cache_row'] = dataclasses.asdict(new_ex)
            report['records'].append(record)
        report['batches'].append({'offsets': offsets, 'seconds': seconds, 'peak_allocated_bytes': peak,
            'physical_batch_size': len(offsets), 'padded_prompt_length': prefix_len,
            'returned_new_token_columns': len(sequences[0])-prefix_len,
            'actual_public_FA_returns': fa_calls, 'actual_public_FA_examples': fa_examples, **counters})
        report['native_root_forwards'] += counters['root_forwards']
        report['native_forward_trajectories'] += counters['root_trajectories']
        report['native_decoder_layer_calls'] += counters['decoder_calls']
        report['native_decoder_layer_trajectories'] += counters['decoder_trajectories']
        del output, inputs
        save()
        print('LOCAL_QWEN8B_GENERATED', offsets, 'seconds', seconds, 'format',
              sum(r['original_format_pass'] for r in report['records']), flush=True)
    for name, module in model.named_modules():
        assert type(module).forward is original_methods[name] and 'forward' not in module.__dict__
    assert not any(m._forward_hooks or m._forward_pre_hooks or m._backward_hooks for m in model.modules())
    report['native_sources_after'] = source_receipt()
    assert report['native_sources_before'] == report['native_sources_after']
    report['checkpoint_after'] = checkpoint_receipt()
    assert report['checkpoint_before'] == report['checkpoint_after']
    assert len(report['records']) == 64 and report['native_generation_calls'] == 16
    report['status'] = 'generation_complete_pending_blind_answer_review_and_cache_freeze'
    report.pop('active_offsets', None)
except Exception:
    report['status'] = 'failed'
    report['error'] = traceback.format_exc()
    raise
finally:
    report['elapsed_seconds'] = time.time()-started
    save()
    with zipfile.ZipFile(HERE/'review_bundle.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for name in ['study.py', 'protocol.json', 'results.json']: z.write(HERE/name, name)

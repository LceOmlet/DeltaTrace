"""At most four actual default FA/FLA model forwards on official fixed text."""
import os
os.environ['MACA_PATH'] = '/opt/maca'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import contextlib
import hashlib
import inspect
import json
import signal
import sys
import time
import traceback
import zipfile
from collections import Counter
from pathlib import Path
HERE = Path(__file__).resolve().parent
os.environ['TRITON_CACHE_DIR'] = str(HERE / 'triton_cache')
p = json.loads((HERE / 'protocol.json').read_text())
sha = lambda b: hashlib.sha256(b).hexdigest()
assert sha((HERE / 'study.py').read_bytes()) == p['study_sha256']
assert sha((HERE / 'official_fixed_text_inputs.py').read_bytes()) == p['input_runtime_sha256']
sys.path.insert(0, p['author_root'])
r = {'status': 'running', 'protocol': p, 'model_load_attempts': 0, 'model_loads': 0,
     'root_forward_attempts': 0, 'root_forwards_entered': 0, 'root_forwards_completed': 0,
     'quality_queries': 0, 'generation_calls': 0, 'attributions': 0, 'calls': []}
start = time.time()


def save():
    temp = HERE / 'results.partial'
    temp.write_text(json.dumps(r, ensure_ascii=False, separators=(',', ':')))
    temp.replace(HERE / 'results.json')


def source_receipt():
    site = Path(p['isolated_site'])
    count = 0; changed = []; h = hashlib.sha256()
    for path, digest in p['wheels'].items():
        raw = Path(path).read_bytes(); assert sha(raw) == digest
        with zipfile.ZipFile(__import__('io').BytesIO(raw)) as z:
            for name in z.namelist():
                if name.endswith('.py') and name.startswith(('fla/', 'transformers/')):
                    actual = (site / name).read_bytes(); original = z.read(name)
                    if actual != original:
                        assert name == 'fla/utils.py' and sha(actual) == p['fla_mapped_utils_sha256']
                        changed.append(name)
                    h.update(name.encode() + bytes.fromhex(sha(actual))); count += 1
    assert count == 2853 and changed == ['fla/utils.py']
    return {'files_compared': count, 'explicit_changes': changed, 'source_tree_sha256': h.hexdigest()}


try:
    r['sources_before'] = source_receipt()
    root = Path(p['author_root'])
    for name, digest in p['author_source_sha256'].items():
        assert sha((root / name).read_bytes().replace(b'\r\n', b'\n')) == digest
    checkpoint = Path(p['checkpoint'])
    r['checkpoint_stats_before'] = {x.name: {'bytes': x.stat().st_size, 'mtime_ns': x.stat().st_mtime_ns}
        for x in checkpoint.glob('*.safetensors')}
    assert {k: v['bytes'] for k, v in r['checkpoint_stats_before'].items()} == p['verified_shard_sizes']
    for name, digest in p['checkpoint_config_tokenizer_sha256'].items():
        assert sha((checkpoint / name).read_bytes()) == digest
    import torch
    import triton
    import transformers
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    from transformers.models.qwen3_5 import modeling_qwen3_5 as native
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
    from fla import utils as fla_utils
    import flash_attn.flash_attn_interface as fa
    import causal_conv1d
    from official_fixed_text_inputs import prepare_fixed_text, right_pad_batch, load_author_preparer
    AuthorTextPreparer = load_author_preparer(p['author_root'], p['author_source_sha256'])
    r['text_preparation_source'] = 'Unchanged hash-verified author LLMAttribution class and two constants extracted from original ASTs; unrelated top-level imports omitted. No model implementation copied or replaced.'
    assert sha(Path(native.__file__).read_bytes()) == p['native_model_sha256']
    assert native.chunk_gated_delta_rule is chunk_gated_delta_rule
    assert native.fused_recurrent_gated_delta_rule is fused_recurrent_gated_delta_rule
    assert native.is_fast_path_available and fla_utils.device_platform == 'maca'
    assert not fla_utils.IS_NVIDIA and not hasattr(torch, 'maca')
    r['versions'] = {'torch': torch.__version__, 'triton': triton.__version__, 'transformers': transformers.__version__,
        'device': torch.cuda.get_device_name(), 'real_triton_backend': triton.runtime.driver.active.get_current_target().backend}
    torch.manual_seed(73)
    r['model_load_attempts'] += 1; save()
    tick = time.perf_counter()
    model, loading = Qwen3_5ForConditionalGeneration.from_pretrained(
        checkpoint, dtype=torch.bfloat16, attn_implementation='flash_attention_2',
        device_map={'': 'cuda:0'}, local_files_only=True, output_loading_info=True)
    model.eval().requires_grad_(False)
    r['model_loads'] += 1
    r['loading'] = {k: v for k, v in loading.items() if v}
    assert not r['loading'], r['loading']
    r['loading_seconds'] = time.perf_counter() - tick
    assert type(model) is Qwen3_5ForConditionalGeneration
    layers = model.model.language_model.layers
    assert len(layers) == 32
    for layer in layers:
        if layer.block_type == 'linear_attention':
            assert layer.linear_attn.chunk_gated_delta_rule is chunk_gated_delta_rule
            assert layer.linear_attn.causal_conv1d_fn is causal_conv1d.causal_conv1d_fn
        else:
            assert layer.self_attn.config._attn_implementation == 'flash_attention_2'
    original_methods = {n: type(m).forward for n, m in model.named_modules()}

    def method_audit():
        for name, module in model.named_modules():
            assert 'forward' not in module.__dict__, name
            assert type(module).forward is original_methods[name]
            assert getattr(module.forward, '__func__', None) is original_methods[name]
            assert not module._forward_hooks and not module._forward_pre_hooks and not module._backward_hooks

    method_audit()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False)
    tokenizer.pad_token = tokenizer.eos_token  # Exact author run_exp.py policy.
    assert tokenizer.eos_token_id == tokenizer.pad_token_id == 248046
    cases = []
    for dataset, index in p['selection']:
        raw = (root / 'exp/exp2/data' / (dataset + '.jsonl')).read_bytes()
        assert sha(raw) == p['cache_sha256'][dataset]
        ex = json.loads(raw.decode().splitlines()[index])
        engine = AuthorTextPreparer(model, tokenizer)
        case = prepare_fixed_text(engine, ex['prompt'], ex['target'])
        case['metadata'].update(dataset=dataset, index=index)
        case['metadata']['input_ids_sha256'] = sha(case['input_ids'].numpy().tobytes())
        cases.append(case)
        del engine
    r['input_metadata'] = [case['metadata'] for case in cases]
    save()
    code_names = {}
    for label, func in [('FLA_chunk', chunk_gated_delta_rule), ('FLA_recurrent', fused_recurrent_gated_delta_rule),
                        ('FA_dense', fa.flash_attn_func), ('FA_varlen', fa.flash_attn_varlen_func),
                        ('causal_conv', causal_conv1d.causal_conv1d_fn),
                        ('torch_chunk_fallback', native.torch_chunk_gated_delta_rule),
                        ('torch_recurrent_fallback', native.torch_recurrent_gated_delta_rule)]:
        code_names[inspect.unwrap(func).__code__] = label

    for call_index, selected in enumerate(p['call_selections']):
        assert r['root_forward_attempts'] < 4
        batch_cases = [cases[i] for i in selected]
        row = {'selection': selected, 'profiled': call_index in [0, 2], 'status': 'attempted',
               'python_dispatch': {}, 'layer_calls': [], 'dispatch_shapes': []}
        r['calls'].append(row); r['root_forward_attempts'] += 1; save()
        handles = []
        counts = Counter()

        def python_event(frame, event, arg):
            if event == 'call' and frame.f_code in code_names:
                label = code_names[frame.f_code]; counts[label] += 1
                q = frame.f_locals.get('q')
                row['dispatch_shapes'].append({'kind': label, 'q_shape': list(q.shape) if isinstance(q, torch.Tensor) else None,
                                              'source': frame.f_code.co_filename, 'line': frame.f_code.co_firstlineno})

        def root_enter(module, args):
            r['root_forwards_entered'] += 1

        handles.append(model.register_forward_pre_hook(root_enter))
        for i, layer in enumerate(layers):
            handles.append(layer.register_forward_pre_hook(lambda m, a, _i=i: row['layer_calls'].append(_i)))
        torch.cuda.empty_cache(); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        tick = time.perf_counter()
        profiler = torch.profiler.profile(activities=list(torch.profiler.supported_activities())) if row['profiled'] else None
        prior_profile = sys.getprofile(); assert prior_profile is None
        try:
            with profiler if profiler is not None else contextlib.nullcontext():
                inputs = right_pad_batch(batch_cases, tokenizer.pad_token_id, model.device)
                row['input_shape'] = list(inputs['input_ids'].shape)
                row['valid_lengths'] = inputs['attention_mask'].sum(-1).tolist()
                sys.setprofile(python_event)
                try:
                    with torch.no_grad():
                        outputs = model(**inputs, use_cache=False, logits_to_keep=0, return_dict=True)
                    r['root_forwards_completed'] += 1
                finally:
                    sys.setprofile(prior_profile)
                assert outputs.past_key_values is None
                assert outputs.logits.dtype == torch.bfloat16 and outputs.logits.shape[-1] == 248320
                row['target_logprobs'] = []
                for j, case in enumerate(batch_cases):
                    plen = case['prompt_length']; target_ids = case['target_ids'].to(model.device)
                    logits = outputs.logits[j, plen-1:plen-1+len(target_ids)].float()
                    logprob = logits.log_softmax(-1).gather(-1, target_ids[:, None]).squeeze(-1)
                    assert bool(torch.isfinite(logprob).all())
                    row['target_logprobs'].append(logprob.cpu().tolist())
                    del logits, logprob, target_ids
                del outputs, inputs
                torch.cuda.synchronize()
            row['status'] = 'complete'
        except Exception:
            row['status'] = 'failed'; row['error'] = traceback.format_exc()
            raise
        finally:
            sys.setprofile(prior_profile)
            row['elapsed_seconds_including_profile_if_enabled'] = time.perf_counter() - tick
            row['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
            row['python_dispatch'] = dict(counts)
            for handle in handles: handle.remove()
            if profiler is not None:
                trace = HERE / f'call_{call_index}_profile.json'
                try:
                    profiler.export_chrome_trace(str(trace))
                    row['profile'] = {'file': trace.name, 'sha256': sha(trace.read_bytes())}
                except Exception:
                    row['profile_export_error'] = traceback.format_exc()
            save()
        assert row['layer_calls'] == list(range(32))
        assert counts['FLA_chunk'] == counts['causal_conv'] == 24
        assert counts['FA_dense'] + counts['FA_varlen'] == 8
        assert counts['FLA_recurrent'] == counts['torch_chunk_fallback'] == counts['torch_recurrent_fallback'] == 0
        method_audit()
        print('QWEN35_NATIVE', call_index, row['input_shape'], row['python_dispatch'], flush=True)
    r['batch_comparisons'] = []
    for row in r['calls'][2:]:
        for j, ref_index in enumerate(row['selection']):
            a = torch.tensor(r['calls'][ref_index]['target_logprobs'][0], dtype=torch.float64)
            b = torch.tensor(row['target_logprobs'][j], dtype=torch.float64)
            delta = (a-b).abs()
            check = {'call_selection': row['selection'], 'reference': ref_index,
                     'max_abs_logprob_difference': delta.max().item(), 'mean_abs_logprob_difference': delta.mean().item(),
                     'relative_L2': float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(a)),
                     'tokens': len(a)}
            r['batch_comparisons'].append(check)
            assert check['relative_L2'] <= p['relative_L2_limit'] and check['max_abs_logprob_difference'] <= p['max_abs_logprob_limit']
    r['sources_after'] = source_receipt(); assert r['sources_before'] == r['sources_after']
    r['checkpoint_stats_after'] = {x.name: {'bytes': x.stat().st_size, 'mtime_ns': x.stat().st_mtime_ns}
        for x in checkpoint.glob('*.safetensors')}
    assert r['checkpoint_stats_before'] == r['checkpoint_stats_after']
    r['status'] = 'four_native_FA_FLA_forwards_and_B2_target_consistency_passed'
except Exception:
    r['status'] = 'failed'; r['error'] = traceback.format_exc()
    raise
finally:
    r['job_seconds'] = time.time() - start
    save()
    with zipfile.ZipFile(HERE / 'review_bundle.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for name in ['study.py', 'protocol.json', 'official_fixed_text_inputs.py', 'results.json'] + [x.name for x in HERE.glob('*_profile.json')]:
            z.write(HERE / name, name)

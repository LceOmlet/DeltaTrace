"""Run the complete unchanged official FlashTrace public API on one fixed NI0.

This is an external driver, not an attribution implementation. No AST execution,
function replacement, custom content operator or controller is used. Hooks and
Python profiling only record/check actual execution. Official B1/eager and its
identity-valued FLA probes retain their full costs and limitations.
"""
import os, sys, json, hashlib, time, traceback, zipfile, signal
from pathlib import Path
from collections import Counter

A = Path(__file__).resolve().parent
p = json.loads((A / 'protocol.json').read_bytes())
sha = lambda b: hashlib.sha256(b).hexdigest()
assert sha((A / 'study.py').read_bytes()) == p['study_sha256']
os.environ.update(MACA_PATH='/opt/maca', HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
                  PYTHONDONTWRITEBYTECODE='1', TRITON_CACHE_DIR=p['compiler_cache'])
sys.dont_write_bytecode = True
for overlay in p['dependency_overlays']:
    sys.path.insert(0, overlay)
sys.path.insert(0, p['official_root'])
r = {'status': 'running', 'protocol': p, 'model_load_attempts': 0, 'model_loads': 0,
     'root_forwards_entered': 0, 'root_forwards_completed': 0, 'trace_calls': 0,
     'original_recovery_calls': 0, 'generation_calls': 0, 'events': [], 'timings': {}}
start = time.perf_counter()
handles = []
counts = Counter()

def save():
    tmp = A / 'results.partial'
    tmp.write_text(json.dumps(r, ensure_ascii=False, indent=2))
    tmp.replace(A / 'results.json')

def timeout(signum, frame):
    raise TimeoutError('Frozen 600-second one-sample execution ceiling reached.')

def official_sources():
    root = Path(p['official_root'])
    assert {str(f.relative_to(root)) for f in (root / 'flashtrace').rglob('*.py')} == set(p['package_blob_sha1'])
    receipt = {}
    for name, digest in p['package_blob_sha1'].items():
        raw = (root / name).read_bytes()
        assert hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest() == digest, name
        receipt[name] = sha(raw)
    return receipt

def observe(frame, event, arg):
    if event != 'call':
        return
    name = frame.f_code.co_name
    if name not in {'calculate_ifr_multi_hop_both', '_capture_model_state', 'build_layer_inputs',
                    '_linear_layer_input', '_full_layer_input', 'compute_ifr_sentence_aggregate',
                    'chunk_gated_delta_rule', 'eager_attention_forward'}:
        return
    filename = frame.f_code.co_filename
    if filename.startswith(p['official_root'] + '/'):
        counts['official.' + name] += 1
    elif name == 'chunk_gated_delta_rule' and '/fla/ops/gated_delta_rule/chunk.py' in filename:
        counts['native_FLA.chunk_gated_delta_rule'] += 1
        value = frame.f_locals.get('v')
        if value is not None:
            r['events'].append({'kind': 'native_FLA_call', 'value_shape': list(value.shape)})
    elif name == 'eager_attention_forward' and '/transformers/models/qwen3_5/' in filename:
        counts['native_model.eager_attention_forward'] += 1

try:
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(600)
    r['official_sources_before'] = official_sources()
    entry_raw = (Path(p['entry_directory']) / 'results.json').read_bytes()
    assert sha(entry_raw) == p['entry_results_sha256']
    entry = json.loads(entry_raw)
    assert entry['status'] == 'complete_official_package_imported'
    import torch
    import numpy as np
    import flashtrace
    from flashtrace import FlashTrace
    from flashtrace.improved import keep_token_indices, evaluate_attr_recovery_skip_tokens
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    from transformers.models.qwen3_5 import modeling_qwen3_5 as native
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    import causal_conv1d
    assert sha(Path(native.__file__).read_bytes()) == p['native_model_sha256']
    assert native.chunk_gated_delta_rule is chunk_gated_delta_rule and native.is_fast_path_available
    site = Path(p['isolated_site'])
    r['runtime_source_sha256'] = {}
    for name, digest in p['runtime_source_sha256'].items():
        actual = sha((site / name).read_bytes())
        assert actual == digest, name
        r['runtime_source_sha256'][name] = actual
    cp = Path(p['checkpoint'])
    for name, digest in p['checkpoint_config_tokenizer_sha256'].items():
        assert sha((cp / name).read_bytes()) == digest, name
    stats = {f.name: {'bytes': f.stat().st_size, 'mtime_ns': f.stat().st_mtime_ns} for f in cp.glob('*.safetensors')}
    assert {k: v['bytes'] for k, v in stats.items()} == p['verified_shard_sizes']
    r['checkpoint_stats_before'] = stats
    raw = Path(p['cache_path']).read_bytes()
    assert sha(raw) == p['cache_sha256']
    record = json.loads(raw.decode().splitlines()[0])
    span_raw = Path(p['spans_path']).read_bytes()
    assert sha(span_raw) == p['spans_sha256']
    case = next(c for c in json.loads(span_raw)['cases'] if c['dataset'] == 'niah_mq_q2' and c['index'] == 0)
    meta, mapping = case['input_metadata'], case['mapping']
    tokenizer = AutoTokenizer.from_pretrained(cp, local_files_only=True, trust_remote_code=False)
    tokenizer.pad_token = tokenizer.eos_token
    r['gpu_before_load'] = list(torch.cuda.mem_get_info())
    assert r['gpu_before_load'][0] >= 30 * 1024**3
    torch.manual_seed(73)
    r.update(status='loading_actual_model', model_load_attempts=1); save()
    tick = time.perf_counter()
    model, info = Qwen3_5ForConditionalGeneration.from_pretrained(
        cp, dtype=torch.bfloat16, attn_implementation='eager', device_map={'': 'cuda:0'},
        local_files_only=True, output_loading_info=True)
    model.eval().requires_grad_(False)
    r['timings']['model_load'] = time.perf_counter() - tick
    r['model_loads'] = 1
    r['nonempty_loading_info'] = {k: v for k, v in info.items() if v}
    assert not r['nonempty_loading_info'], r['nonempty_loading_info']
    layers = model.model.language_model.layers
    assert len(layers) == 32
    for layer in layers:
        if layer.block_type == 'linear_attention':
            assert layer.linear_attn.chunk_gated_delta_rule is chunk_gated_delta_rule
            assert layer.linear_attn.causal_conv1d_fn is causal_conv1d.causal_conv1d_fn
        else:
            assert layer.self_attn.config._attn_implementation == 'eager'
    r['backend'] = {'full_attention': 'unchanged native Qwen3.5 eager',
                    'linear_attention': 'actual installed FLA', 'batch': 1,
                    'FT_package': flashtrace.__file__, 'DT_backend_identical': False}

    def root_enter(module, args, kwargs):
        r['root_forwards_entered'] += 1
        assert r['root_forwards_entered'] == 1
        ids = kwargs['input_ids'].detach().cpu()
        digest = sha(ids.numpy().tobytes())
        r['actual_input'] = {'shape': list(ids.shape), 'sha256': digest,
                             'output_attentions': kwargs.get('output_attentions'),
                             'use_cache': kwargs.get('use_cache')}
        assert digest == meta['input_ids_sha256'] and list(ids.shape) == [1, meta['input_length']]
        assert kwargs.get('output_attentions') is True and kwargs.get('use_cache') is False
        assert kwargs['attention_mask'].eq(1).all()
        save()

    def root_exit(module, args, kwargs, output):
        r['root_forwards_completed'] += 1
        r['actual_attention_shapes'] = [list(a.shape) for a in output.attentions]
        assert len(output.attentions) == 8
        save()

    handles.append(model.register_forward_pre_hook(root_enter, with_kwargs=True))
    handles.append(model.register_forward_hook(root_exit, with_kwargs=True))
    tracer = FlashTrace(model, tokenizer, use_chat_template=True)
    r.update(status='executing_complete_official_trace', trace_calls=1); save()
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    tick = time.perf_counter(); sys.setprofile(observe)
    result = tracer.trace(prompt=record['prompt'], target=record['target'],
                          output_span=tuple(mapping['new_sink_span']),
                          reasoning_span=tuple(mapping['new_thinking_span']),
                          hops=3, method='flashtrace')
    sys.setprofile(None); torch.cuda.synchronize()
    r['timings']['complete_trace_with_observation'] = time.perf_counter() - tick
    r['peak_allocated'] = torch.cuda.max_memory_allocated()
    r['peak_reserved'] = torch.cuda.max_memory_reserved()
    r['calls'] = dict(counts)
    assert r['root_forwards_completed'] == 1 and counts['official.calculate_ifr_multi_hop_both'] == 1
    assert counts['official._linear_layer_input'] == 24 and counts['official._full_layer_input'] == 8
    assert counts['official.compute_ifr_sentence_aggregate'] <= 4
    assert len(result.prompt_tokens) == len(meta['author_user_positions'])
    keep = keep_token_indices(result.prompt_tokens)
    assert keep == mapping['keep_local_indices']
    ifr = result.metadata['ifr']; obs = ifr['observation_projected']
    prefix = obs['base'].clone(); cumulative = [prefix.clone()]
    for contribution in obs['per_hop']:
        prefix = prefix + contribution
        cumulative.append(prefix.clone())
    assert len(cumulative) == 4
    assert torch.equal(prefix, obs['sum'])
    prompt_len = len(result.prompt_tokens)
    assert np.array_equal(prefix[:prompt_len].numpy(), np.asarray(result.scores, dtype=np.float32))
    scores = torch.stack(cumulative)[:, :prompt_len]
    assert torch.isfinite(scores).all()
    np.savez_compressed(A / 'official_outputs.npz', cumulative_projected=torch.stack(cumulative).numpy(),
                        scores=scores.numpy(), base=obs['base'].numpy(),
                        per_hop=torch.stack(obs['per_hop']).numpy(),
                        sum=obs['sum'].numpy(),
                        raw_projected=torch.stack(ifr['per_hop_projected']).numpy())
    r['artifacts'] = {'official_outputs.npz': sha((A / 'official_outputs.npz').read_bytes())}
    r['status'] = 'official_outputs_saved'; save()
    r['needle'] = []
    for hop, score in enumerate(scores):
        recovery = evaluate_attr_recovery_skip_tokens(score[None], keep_prompt_token_indices=keep,
            gold_prompt_token_indices=mapping['gold_user_token_indices'], top_fraction=0.1)
        r['original_recovery_calls'] += 1
        selected = [keep[i] for i in torch.topk(score[keep].clamp(min=0), 31).indices.tolist()]
        hits = sorted(set(selected) & set(mapping['gold_user_token_indices']))
        assert recovery == len(hits) / 40
        r['needle'].append({'hop': hop, 'recovery': recovery, 'hits': hits, 'selected': selected})
    r['thinking_ratios'] = ifr['thinking_ratios']
    r['official_span_metadata'] = {k: ifr[k] for k in ['sink_span_generation', 'thinking_span_generation', 'all_gen_span_generation', 'n_hops', 'stop_config', 'note']}
    r['result_semantics'] = 'One unchanged public trace(hops=3). Prefix scores are sums of the official returned base and hop observation contributions; no custom recurrence or extra trace runs.'
    r['official_sources_after'] = official_sources()
    assert r['official_sources_before'] == r['official_sources_after']
    r['checkpoint_stats_after'] = {f.name: {'bytes': f.stat().st_size, 'mtime_ns': f.stat().st_mtime_ns} for f in cp.glob('*.safetensors')}
    assert r['checkpoint_stats_after'] == stats
    r['status'] = 'unchanged_official_FT_NI0_trace_completed'
except Exception:
    sys.setprofile(None)
    r.update(status='failed_unmodified_official_run', error=traceback.format_exc(), calls=dict(counts))
finally:
    signal.alarm(0)
    for handle in handles:
        handle.remove()
    r['seconds_before_bundle'] = time.perf_counter() - start; save()
    with zipfile.ZipFile(A / 'review_bundle.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for name in ['study.py', 'protocol.json', 'results.json', 'official_outputs.npz']:
            if (A / name).exists(): z.write(A / name, name)
    print(json.dumps({'status': r['status'], 'calls': dict(counts), 'needle': r.get('needle'),
                      'seconds': r['seconds_before_bundle'], 'error': r.get('error')}), flush=True)

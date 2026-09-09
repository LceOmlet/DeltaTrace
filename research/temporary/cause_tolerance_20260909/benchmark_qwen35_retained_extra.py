"""Freeze six original author inputs through the existing formal prepare_case.

Then measure unchanged deferred B2 against replay retention. No generation,
FT attribution or metric curves. Full vector equality can preserve score inputs;
otherwise original score validation remains outstanding.
"""
import argparse
import ast
import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace

sha = lambda b: hashlib.sha256(b).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in ('release', 'environment', 'output'): p.add_argument('--' + n, type=Path, required=True)
    p.add_argument('--formal-sha256', required=True)
    a = p.parse_args(); env = json.loads(a.environment.read_bytes())['qwen35']
    formal = a.release / 'experiments/official/evaluate.py'; raw = formal.read_bytes(); assert sha(raw) == a.formal_sha256
    protocol = json.loads((formal.parent / 'protocol.json').read_bytes()); official = Path(env['official_root'])
    for name, digest in protocol['official_normalized_sources'].items():
        assert sha((official / name).read_bytes().replace(b'\r\n', b'\n')) == digest
    os.environ.update(MACA_PATH='/opt/maca', HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false', TRITON_ENABLE_PERSISTENT_AUTOTUNE_CONFIGS='0')
    os.environ.setdefault('TRITON_CACHE_DIR', '/tmp/deltatrace_clean_v1_triton')
    os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR', '/tmp/deltatrace_clean_v1_inductor')
    sys.path.insert(0, str(official))
    for d in env.get('dependency_overlays', []): sys.path.insert(0, d)
    for d in ('deltatrace/clean/qwen35', 'deltatrace/accelerated/qwen35', 'deltatrace/accelerated', 'experiments/official'):
        sys.path.insert(0, str(a.release / d))
    import numpy as np
    import torch
    from torch._dynamo.utils import counters
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    import ft_ifr_improve as ft
    from exp.exp2 import dataset_utils as data_utils
    from llm_attr_eval import LLMAttributionEvaluator
    from batching import attribute_batch, group_cases
    from deferred import make_deferred_qwen35
    from dynamic_finite import configure_dynamic_finite
    from qwen35_retained_controller import Qwen35DenseFiniteRunner
    from vendor_fa_finite_bf16_d256 import VendorFAFiniteP1BF16D256
    from finite_fla_gpu import verify_native_sources
    torch.set_num_threads(4); torch.manual_seed(73); torch.backends.cuda.matmul.allow_tf32 = False
    torch._dynamo.config.cache_size_limit = max(128, torch._dynamo.config.cache_size_limit)
    torch._dynamo.config.accumulated_cache_size_limit = max(512, torch._dynamo.config.accumulated_cache_size_limit)
    a.output.mkdir(exist_ok=False)
    report = {'status': 'loading', 'script_sha256': sha(Path(__file__).read_bytes()), 'formal_sha256': a.formal_sha256,
              'candidate_sources': {n: sha((Path(__file__).parent / n).read_bytes()) for n in ('qwen35_retained_controller.py', 'qwen35_retained_capture.py')},
              'calls': [], 'cases': [], 'groups': [], 'FT_calls': 0, 'metric_calls': 0, 'generation_calls': 0}
    vectors = {}
    def save():
        p = a.output / 'results.partial'; p.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n'); p.replace(a.output / 'results.json')
    def timed(name, fn):
        report['status'] = name; save(); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        tick = time.perf_counter(); call = {'name': name, 'compiler_before': dict(counters['stats'])}; report['calls'].append(call)
        value = fn(); torch.cuda.synchronize()
        call.update(status='returned', seconds=time.perf_counter() - tick, peak_allocated=torch.cuda.max_memory_allocated(), compiler_after=dict(counters['stats']))
        save(); return value, call
    try:
        tokenizer = AutoTokenizer.from_pretrained(env['checkpoint'], local_files_only=True)
        model, _ = timed('model_load', lambda: Qwen3_5ForConditionalGeneration.from_pretrained(env['checkpoint'], dtype=torch.bfloat16,
                         attn_implementation='eager', device_map={'': 'cuda:0'}, local_files_only=True))
        model.eval().requires_grad_(False); assert sha(Path(inspect.getfile(type(model))).read_bytes()) == env['native_model_sha256']
        identities = {n: (type(m).forward, m.forward) for n, m in model.named_modules()}
        evaluator = LLMAttributionEvaluator(model, tokenizer)
        node, = [node for node in ast.walk(ast.parse(raw)) if isinstance(node, ast.FunctionDef) and node.name == 'prepare_case']
        namespace = {'copy': copy, 'ft': ft, 'torch': torch, 'args': SimpleNamespace(family='qwen35', selection='paper'),
                     'data_utils': data_utils, 'tokenizer': tokenizer, 'model': model, 'evaluator': evaluator, 'sha': sha}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(formal), 'exec'), namespace)
        groups = []; cache_refs = []; preparation_calls = []
        hook = model.register_forward_pre_hook(lambda *_: preparation_calls.append(True))
        try:
            for task, count in [('vt_h2_c3', 2), ('hotpotqa_long', 4)]:
                cache = official / 'exp/exp2/data' / (task + '.jsonl'); data = cache.read_bytes()
                assert sha(data) == protocol['tasks'][task]['cache_sha256']
                records = [json.loads(line) for line in data.decode().splitlines()]; loaded = data_utils.load_cached(cache)
                cases = [namespace['prepare_case'](task, i, records[i], loaded) for i in range(count)]
                groups.extend(group_cases(cases, 2)); report['cases'].extend(c['row'] for c in cases)
                cache_refs.append({'task': task, 'cache_sha256': sha(data), 'indices': list(range(count))})
        finally: hook.remove()
        assert not preparation_calls and len(groups) == 3 and len(report['cases']) == 6
        report['input_preparation_native_calls'] = 0
        report['groups'] = [[c['key'] for c in g] for g in groups]
        frozen = {'status': 'frozen_before_attribution', 'family': 'qwen35', 'cases': report['cases'], 'cache_references': cache_refs,
                  'formal_sha256': a.formal_sha256, 'method_selection_holdout': False}
        data = (json.dumps(frozen, indent=2) + '\n').encode(); (a.output / 'inputs.json').write_bytes(data)
        report['frozen_inputs_sha256'] = sha(data); save()
        with torch.no_grad():
            ids = groups[0][0]['ids'].to(model.device)
            timed('native_initialization', lambda: model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False))
            del ids
        model.set_attn_implementation('flash_attention_2'); verify_native_sources(env['native_stage_source_sha256'])
        fa = VendorFAFiniteP1BF16D256(env['finite_library'], env['finite_library_sha256'])
        baseline, report['baseline_sources'] = make_deferred_qwen35(a.release, model, fa, None)
        candidate = configure_dynamic_finite(Qwen35DenseFiniteRunner(model, fa, None, checkpoint_device='cuda'))
        roots = {}
        for phase in ('warm', 'r0', 'r1'):
            for i in (range(len(groups)) if phase != 'r1' else reversed(range(len(groups)))):
                for mode in (('baseline', 'candidate') if phase != 'r1' else ('candidate', 'baseline')):
                    name = f'{phase}/{mode}/{i}'
                    (signed, details, root), call = timed(name, lambda: attribute_batch(baseline if mode == 'baseline' else candidate, groups[i], tokenizer.eos_token_id, model.device))
                    if i in roots: assert root == roots[i]
                    else: roots[i] = root
                    call.update(root=root, details=details)
                    for c, v in zip(groups[i], signed): vectors[name + '/' + c['key']] = v.numpy()
                    for n, m in model.named_modules(): assert (type(m).forward, m.forward) == identities[n]
                    np.savez_compressed(a.output / 'vectors.npz', **vectors); save()
        report['comparisons'] = []
        for i, group in enumerate(groups):
            for c in group:
                names = [f'{p}/{m}/{i}/' + c['key'] for p in ('warm', 'r0', 'r1') for m in ('baseline', 'candidate')]
                report['comparisons'].append({'case': c['key'], 'all_six_complete_vectors_equal': all(np.array_equal(vectors[names[0]], vectors[n]) for n in names)})
        total = lambda m: sum(c['seconds'] for c in report['calls'] if c['name'].startswith(('r0/' + m, 'r1/' + m))) / 2
        report.update(baseline_seconds_per_pass=total('baseline'), candidate_seconds_per_pass=total('candidate'), reduction_fraction=1 - total('candidate') / total('baseline'))
        report['efficiency_gate_passed'] = report['reduction_fraction'] >= .03
        report['vectors_sha256'] = sha((a.output / 'vectors.npz').read_bytes()); report['status'] = 'complete'
    except Exception:
        report['status'] = 'failed'; report['error'] = traceback.format_exc(); raise
    finally: save()


if __name__ == '__main__': main()

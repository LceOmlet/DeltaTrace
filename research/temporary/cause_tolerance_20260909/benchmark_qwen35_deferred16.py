"""Compare current and deferred-diagnostic B2 DT through the unchanged formal batch helper.

Use the exact input groups from the completed signed-view quality run. Warm
every B1/B2 shape, then run two interleaved rounds with reversed order. This
script performs no quality, FT, or generation calls and changes no operators.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time
import traceback

sha = lambda data: hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--environment', type=Path, required=True)
    parser.add_argument('--quality', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert sha(args.plan.read_bytes()) == args.plan_sha256
    plan = json.loads(args.plan.read_bytes())
    assert plan['status'] == 'prepared_not_measured'
    assert sha(args.quality.read_bytes()) == plan['quality_report_sha256']
    quality = json.loads(args.quality.read_bytes())
    assert quality['status'] == 'complete' and quality['family'] == 'qwen35'
    assert sha((args.quality.parent / 'vectors.npz').read_bytes()) == plan['quality_vectors_sha256']
    assert sha((args.release / 'experiments/official/batching.py').read_bytes()) == plan['batching_sha256']
    manifest_raw = (args.release / 'deltatrace/clean/sources.json').read_bytes()
    assert sha(manifest_raw) == plan['clean_sources_sha256']
    for name, row in json.loads(manifest_raw)['models']['qwen35']['files'].items():
        assert sha((args.release / name).read_bytes()) == row['sha256']
    environment = json.loads(args.environment.read_bytes())['qwen35']
    os.environ.update(MACA_PATH='/opt/maca', HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
                      TRITON_ENABLE_PERSISTENT_AUTOTUNE_CONFIGS='0')
    os.environ.setdefault('TRITON_CACHE_DIR', '/tmp/deltatrace_clean_v1_triton')
    os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR', '/tmp/deltatrace_clean_v1_inductor')
    sys.path.insert(0, environment['official_root'])
    for overlay in environment.get('dependency_overlays', []):
        sys.path.insert(0, overlay)
    sys.path.insert(0, str(args.release / 'deltatrace/clean/qwen35'))
    sys.path.insert(0, str(args.release / 'experiments/official'))
    sys.path.insert(0, str(args.release / 'deltatrace/accelerated/qwen35'))
    import numpy as np
    import torch
    from torch._dynamo.utils import counters
    from transformers import Qwen3_5ForConditionalGeneration
    from batching import make_accelerated_runner, attribute_batch
    from qwen35_clean_runner import make_qwen35_clean_runner
    from finite_fla_gpu import make_compiled_finite_pullback, verify_native_sources
    from vendor_fa_finite_bf16_d256 import VendorFAFiniteP1BF16D256
    torch.set_num_threads(4)
    torch.manual_seed(73)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 48)
    torch._dynamo.config.accumulated_cache_size_limit = max(torch._dynamo.config.accumulated_cache_size_limit, 192)
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'loading', 'driver_sha256': sha(Path(__file__).read_bytes()),
              'plan_sha256': args.plan_sha256, 'quality_report_sha256': plan['quality_report_sha256'],
              'batching_sha256': plan['batching_sha256'], 'batches': plan['batches'],
              'calls': [], 'metric_calls': 0, 'generation_calls': 0,
              'measurement_scope': 'Complete original attribute_batch API, including packing and original diagnostics; load and all-shape warmup separate.',
              'memory_scope': 'Maximum allocated at API boundaries and from existing controller counters, including resident model.',
              'quality_not_recomputed': True}
    vectors = {}

    def save():
        temporary = args.output / 'results.partial'
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        temporary.replace(args.output / 'results.json')

    def timed(name, function):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        call = {'name': name, 'status': 'entered', 'allocated_before': torch.cuda.memory_allocated(),
                'compiler_before': dict(counters['stats'])}
        report['calls'].append(call)
        started = time.perf_counter()
        try:
            result = function()
            torch.cuda.synchronize()
            call['status'] = 'returned'
            return result, call
        finally:
            call.update(seconds=time.perf_counter()-started, peak_allocated=torch.cuda.max_memory_allocated(),
                        peak_reserved=torch.cuda.max_memory_reserved(), compiler_after=dict(counters['stats']))

    save()
    try:
        model, _ = timed('model_load', lambda: Qwen3_5ForConditionalGeneration.from_pretrained(
            environment['checkpoint'], dtype=torch.bfloat16, attn_implementation='eager',
            device_map={'': 'cuda:0'}, local_files_only=True))
        model.eval().requires_grad_(False)
        assert sha(Path(inspect.getfile(type(model))).read_bytes()) == environment['native_model_sha256']
        original_forwards = {name: (type(module).forward, module.forward) for name, module in model.named_modules()}
        cases = {}
        for row in quality['cases']:
            key = f"{row['dataset']}_{row['index']}"
            ids = torch.tensor(row['input_ids'], dtype=torch.long)[None]
            assert sha(ids.numpy().tobytes()) == row['input_sha256']
            cases[key] = {'key': key, 'row': row, 'ids': ids, 'prompt_len': row['prompt_length'],
                          'gen_len': row['target_length'], 'eval_target': ids[:, row['prompt_length']:],
                          'eligible': [row['user_positions'][j] for j in row['keep']]}
        groups = [[cases[key] for key in batch['cases']] for batch in plan['batches']]
        assert len(cases) == 16 and len(groups) == 8
        eos = int(groups[0][0]['ids'][0, -1])
        assert all(int(case['ids'][0, -1]) == eos for case in cases.values())
        with torch.no_grad():
            initial, _ = timed('native_eager_initialization', lambda: model(
                input_ids=groups[0][0]['ids'].to(model.device),
                attention_mask=torch.ones_like(groups[0][0]['ids'], device=model.device), use_cache=False))
        del initial
        model.set_attn_implementation('flash_attention_2')
        verify_native_sources(environment['native_stage_source_sha256'])
        finite_fa = VendorFAFiniteP1BF16D256(environment['finite_library'], environment['finite_library_sha256'])
        finite_fla = make_compiled_finite_pullback(reuse_scalar_products=False)
        baseline, sources = make_accelerated_runner(args.release, model, finite_fa, finite_fla)
        from qwen35_deferred_controller import Qwen35DenseFiniteRunner as DeferredRunner
        from dynamic_finite import configure_dynamic_finite
        accelerated = configure_dynamic_finite(DeferredRunner(model,finite_fa,finite_fla,checkpoint_device='cuda'))
        report['deferred_controller_sha256'] = sha(Path(inspect.getfile(DeferredRunner)).read_bytes())
        assert sources == plan['acceleration_sources']
        report['acceleration_sources'] = sources

        def execute(name, runner, group, expected_root=None):
            report['status'] = name
            save()
            (signed, details, root), call = timed(name, lambda: attribute_batch(runner, group, eos, model.device))
            if expected_root is not None:
                assert root == expected_root
            for module_name, module in model.named_modules():
                assert (type(module).forward, module.forward) == original_forwards[module_name]
            assert details['norm_gate_rules'] == details['attention_pv_rules'] == {}
            assert details['finite_fla_by_layer'] == details['key_norm_by_layer'] == []
            call.update(cases=[case['key'] for case in group], actual_root=root, details=details)
            for case, vector in zip(group, signed):
                assert torch.isfinite(vector).all()
                vectors[name + '/' + case['key']] = vector.numpy()
            np.savez_compressed(args.output / 'vectors.npz', **vectors)
            save()

        for index,group in enumerate(groups):
            execute(f'warm_baseline/{index}',baseline,group,plan['batches'][index]['actual_root'])
            execute(f'warm_accelerated/{index}',accelerated,group,plan['batches'][index]['actual_root'])
        for repeat in range(2):
            order=range(8) if repeat==0 else reversed(range(8))
            for index in order:
                modes=('baseline','accelerated') if repeat==0 else ('accelerated','baseline')
                for mode in modes:
                    runner=baseline if mode=='baseline' else accelerated
                    execute(f'measured_r{repeat}_{mode}/{index}',runner,groups[index],plan['batches'][index]['actual_root'])
        report['status'] = 'complete'
        report['vectors_sha256'] = sha((args.output / 'vectors.npz').read_bytes())
        report['measured_calls_without_new_dynamo_graphs'] = all(
            call['compiler_before'].get('unique_graphs', 0) == call['compiler_after'].get('unique_graphs', 0)
            for call in report['calls'] if call['name'].startswith('measured_'))
    except Exception:
        report.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        save()


if __name__ == '__main__':
    main()

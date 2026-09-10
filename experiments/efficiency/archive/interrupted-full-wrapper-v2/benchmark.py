"""Isolated Qwen3 rollout-length cost probe with the frozen DT implementation.

The pinned author's exp1 module supplies token construction; exp2 supplies
the unmodified complete FT evaluation wrapper, including all three views.
No attribution, model, attention, or compiler function is patched.
"""
import argparse
import gc
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sha = lambda data: hashlib.sha256(data).hexdigest()


def save(path, value):
    temp = path.with_suffix('.partial')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def worker(args):
    protocol = json.loads((HERE / 'protocol.json').read_bytes())
    env = json.loads(args.environment.read_bytes())['qwen3']
    report = dict(status='setup', method=args.method, output_tokens=args.length,
                  timing_scope=protocol['timing_scope'],
                  driver_sha256=sha(Path(__file__).read_bytes()), calls=[],
                  protocol_sha256=sha((HERE / 'protocol.json').read_bytes()))
    args.output.mkdir(parents=True, exist_ok=False)
    result_path = args.output / 'result.json'
    save(result_path, report)
    try:
        sources = json.loads((ROOT / 'deltatrace/clean/sources.json').read_bytes())
        for relative, receipt in sources['models']['qwen3']['files'].items():
            assert sha((ROOT / relative).read_bytes()) == receipt['sha256'], relative
        official = Path(env['official_root'])
        author_protocol = json.loads((ROOT / 'experiments/official/protocol.json').read_bytes())
        for relative, digest in author_protocol['official_normalized_sources'].items():
            assert sha((official / relative).read_bytes().replace(b'\r\n', b'\n')) == digest, relative
        assert sha(args.author_script.read_bytes()) == protocol['upstream_script_sha256']
        os.environ.setdefault('MACA_PATH', '/opt/maca')
        os.environ.update(HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
        os.environ.setdefault('TRITON_CACHE_DIR', str(args.output.parent / 'triton_cache'))
        os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR', str(args.output.parent / 'inductor_cache'))
        sys.path.insert(0, str(official))
        sys.path.insert(0, str(ROOT / 'deltatrace/clean/qwen3'))
        import numpy as np
        import torch
        import transformers
        import ft_ifr_improve as ft
        spec = importlib.util.spec_from_file_location('author_exp1', args.author_script)
        author = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(author)
        wrapper_spec = importlib.util.spec_from_file_location('author_exp2', official / 'exp/exp2/run_exp.py')
        wrapper = importlib.util.module_from_spec(wrapper_spec)
        wrapper_spec.loader.exec_module(wrapper)
        assert Path(ft.__file__).resolve().is_relative_to(official.resolve())
        assert Path(sys.modules['llm_attr'].__file__).resolve().is_relative_to(official.resolve())
        torch.set_num_threads(4)
        torch.manual_seed(protocol['seed'])
        np.random.seed(protocol['seed'])
        torch.backends.cuda.matmul.allow_tf32 = False
        free, total = torch.cuda.mem_get_info()
        if free < total * .9:
            raise RuntimeError(f'GPU occupied before model load: {free}/{total} bytes free')
        report['environment'] = dict(torch=torch.__version__, transformers=transformers.__version__,
            device=torch.cuda.get_device_name(), total_memory_bytes=total, free_before_load_bytes=free,
            source_manifest_sha256=sha((ROOT / 'deltatrace/clean/sources.json').read_bytes()),
            native_model_sha256=env['native_model_sha256'], finite_library_sha256=env['finite_library_sha256'])
        model, tokenizer = author.load_model_balanced(env['checkpoint'], 'cuda:0')
        assert sha(Path(inspect.getfile(type(model))).read_bytes()) == env['native_model_sha256']
        # DT has no gradient graph; FT retains the author's loader defaults.
        if args.method == 'DT':
            model.requires_grad_(False)
        native_forwards = {name: module.forward for name, module in model.named_modules()}
        base_path = official / 'data/ruler_multihop/8192/vt_h10_c1/validation.jsonl'
        base_text = author.load_ruler_base(base_path, fallback='RULER fallback text. ')
        prompt, _ = author.build_prompt_to_length(tokenizer, base_text, protocol['prompt_tokens'])
        target, _ = author.build_output_to_length(tokenizer, protocol['target_text'], args.length)
        report['input'] = dict(prompt=prompt, target_sha256=sha(target.encode()),
            base_text_sha256=sha(base_text.encode()), ruler_file_available=base_path.exists(),
            **author.estimate_model_lengths(tokenizer, prompt, target))
        receipt_engine = ft.LLMIFRAttributionBoth(model, tokenizer, show_progress=False)
        expected_ids, _, prompt_len, gen_len = receipt_engine._ensure_generation(prompt, target)
        expected_digest = sha(expected_ids.cpu().numpy().tobytes())
        report['input'].update(input_ids_sha256=expected_digest, input_ids=expected_ids[0].cpu().tolist(),
            user_positions=list(receipt_engine.user_prompt_indices),
            keep_local_indices=ft.keep_token_indices(receipt_engine.user_prompt_tokens))
        del expected_ids, receipt_engine
        native_inputs = []

        def observe_input(module, positional, keyword):
            native_inputs.append(keyword['input_ids'] if 'input_ids' in keyword else positional[0])

        input_hook = model.register_forward_pre_hook(observe_input, with_kwargs=True)
        if args.method == 'DT':
            from qwen_signed_secant_paired_public_fa import capture_checkpoint_pair
            from qwen_signed_secant_paired_vendor_fa import propagate_paired_secant
            from vendor_fa_finite_runtime import VendorFAFiniteP1
            finite_fa = VendorFAFiniteP1(env['finite_library'], env['finite_library_sha256'])
            model.set_attn_implementation('flash_attention_2')

            def run():
                engine = ft.LLMIFRAttributionBoth(model, tokenizer, show_progress=False)
                ids, mask, pl, gl = engine._ensure_generation(prompt, target)
                eligible = [engine.user_prompt_indices[j] for j in ft.keep_token_indices(engine.user_prompt_tokens)]
                base = ids.clone()
                base[0, eligible] = tokenizer.eos_token_id
                before, after = capture_checkpoint_pair(model, base, ids, mask, pl)
                result = propagate_paired_secant(model, before, after, pv_rule='content_P1', finite_attention=finite_fa)
                signed = torch.tensor(result['signed_full_sequence'], dtype=torch.float64)
                result['source_scores'] = signed[engine.user_prompt_indices].clamp_min(0).float().tolist()
                return result
        else:
            example = SimpleNamespace(prompt=prompt, target=target,
                indices_to_explain=[0, gen_len - 1], sink_span=None, thinking_span=None)
            configuration = dict(model=model, tokenizer=tokenizer, attr_func='ifr_multi_hop_both',
                chunk_tokens=protocol['chunk_tokens'], sink_chunk_tokens=protocol['sink_chunk_tokens'], n_hops=1)

            def run():
                values, _, positions, keep = wrapper.run_attribution(configuration, example, target)
                scores = values[0][:, :len(positions)].sum(0).cpu().float().tolist()
                del values
                assert positions == report['input']['user_positions']
                assert keep == report['input']['keep_local_indices']
                return dict(source_scores=scores)

        gc.collect()
        torch.cuda.empty_cache()
        for index in range(protocol['warmups'] + protocol['repeats']):
            report.update(status='running', phase=f'call_{index}', phase_started=time.time())
            save(result_path, report)
            torch.cuda.synchronize()
            before_allocated = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            native_inputs.clear()
            started = time.perf_counter()
            value = run()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            row = dict(index=index, warmup=index < protocol['warmups'], seconds=elapsed,
                allocated_before=before_allocated, peak_allocated=torch.cuda.max_memory_allocated(),
                peak_reserved=torch.cuda.max_memory_reserved())
            assert len(native_inputs) == 1
            assert native_inputs[0].shape[0] == (2 if args.method == 'DT' else 1)
            actual_digest = sha(native_inputs[0][-1:].cpu().numpy().tobytes())
            assert actual_digest == expected_digest
            row['input_ids_sha256'] = actual_digest
            scores = np.asarray(value['source_scores'], dtype=np.float64)
            assert scores.shape == (len(report['input']['user_positions']),)
            assert np.isfinite(scores).all()
            row['scores_sha256'] = sha(scores.tobytes())
            row['scores_shape'] = list(scores.shape)
            row['native_model_calls'] = len(native_inputs)
            native_inputs.clear()
            for name, module in model.named_modules():
                assert module.forward == native_forwards[name], name
            report['calls'].append(row)
            save(result_path, report)
            del value
            # Release Python objects but retain the allocator pool, as in warm use.
            gc.collect()
        warm = report['calls'][protocol['warmups']:]
        input_hook.remove()
        report.update(status='ok', median_seconds=statistics.median(x['seconds'] for x in warm),
            min_seconds=min(x['seconds'] for x in warm), max_seconds=max(x['seconds'] for x in warm),
            peak_allocated_bytes=max(x['peak_allocated'] for x in warm),
            native_forward_unchanged=True, generation_calls=0)
    except Exception as exc:
        message = str(exc)
        report.update(status='oom' if 'out of memory' in message.lower() else 'error',
            error_type=type(exc).__name__, error=message, traceback=traceback.format_exc())
    save(result_path, report)
    print(json.dumps({key: report.get(key) for key in ('method','output_tokens','status','median_seconds','error')}, allow_nan=False), flush=True)
    return 0 if report['status'] == 'ok' else 1


def controller(args):
    protocol = json.loads((HERE / 'protocol.json').read_bytes())
    args.output.mkdir(parents=True, exist_ok=False)
    state = dict(status='running', pid=os.getpid(), cells=[])
    state_path = args.output / 'controller.json'
    save(state_path, state)
    for length in args.lengths:
        for method in args.methods:
            destination = args.output / f'{method}_{length}'
            command = [sys.executable, '-B', str(Path(__file__).resolve()), '--worker',
                '--environment', str(args.environment), '--author-script', str(args.author_script),
                '--output', str(destination), '--method', method, '--length', str(length)]
            with (args.output / f'{method}_{length}.log').open('w') as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                cell = dict(method=method, output_tokens=length, pid=process.pid, status='running', started=time.time())
                state['cells'].append(cell)
                save(state_path, state)
                while process.poll() is None:
                    time.sleep(2)
                    report_path = destination / 'result.json'
                    report = json.loads(report_path.read_bytes()) if report_path.exists() else {}
                    phase_start = report.get('phase_started', cell['started'])
                    limit = protocol['call_timeout_seconds'] if report.get('status') == 'running' else protocol['setup_timeout_seconds']
                    if time.time() - phase_start > limit:
                        process.kill()
                        process.wait()
                        report.update(status='timeout', timeout_seconds=limit, method=method, output_tokens=length)
                        save(report_path, report)
                result = json.loads((destination / 'result.json').read_bytes())
                cell.update(status=result['status'], exit_code=process.returncode, elapsed=time.time()-cell['started'])
                save(state_path, state)
            if result.get('error','').startswith('GPU occupied'):
                state['status'] = 'blocked_gpu_occupied'
                save(state_path, state)
                return 1
    state['status'] = 'complete'
    save(state_path, state)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--environment', type=Path, required=True)
    parser.add_argument('--author-script', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--method', choices=['DT','FT'])
    parser.add_argument('--length', type=int)
    parser.add_argument('--methods', nargs='+', default=['DT','FT'])
    parser.add_argument('--lengths', nargs='+', type=int, default=[10,100,500,1000,2000,5000,10000])
    arguments = parser.parse_args()
    raise SystemExit(worker(arguments) if arguments.worker else controller(arguments))

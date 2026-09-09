"""Bounded Qwen3 NI0/MH0 native target-logit optimization pilot.

One cold call per mode and ABBA warmed full DT calls per case, sample B1/E2.
Native head events are separate from complete synchronized wall time. No FT
rerun; fresh author signed RISE/positive MAS if either candidate vector differs.
"""
import argparse
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('release', 'environment', 'reference', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--reference-sha256', required=True)
    args = parser.parse_args()
    sha = lambda b: hashlib.sha256(b).hexdigest()
    assert sha(args.reference.read_bytes()) == args.reference_sha256
    reference = json.loads(args.reference.read_bytes())
    assert reference['family'] == 'qwen3' and reference['status'] == 'complete'
    rows = [next(r for r in reference['cases'] if r['dataset'] == task and r['index'] == 0)
            for task in ('niah_mq_q2', 'morehopqa')]
    sources = json.loads((args.release / 'deltatrace/clean/sources.json').read_bytes())
    for name, receipt in sources['models']['qwen3']['files'].items():
        assert sha((args.release / name).read_bytes()) == receipt['sha256'], name
    env = json.loads(args.environment.read_bytes())['qwen3']
    os.environ.update(MACA_PATH='/opt/maca', HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    os.environ.setdefault('TRITON_CACHE_DIR', '/tmp/deltatrace_clean_v1_triton')
    os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR', '/tmp/deltatrace_clean_v1_inductor')
    sys.path.insert(0, env['official_root'])
    sys.path.insert(0, str(args.release / 'deltatrace/clean/qwen3'))
    sys.path.insert(0, str(args.release / 'experiments/official'))
    import numpy as np
    import torch
    from exp.exp2 import run_exp as author, dataset_utils
    from llm_attr_eval import LLMAttributionEvaluator
    import ft_ifr_improve as ft
    from qwen_signed_secant_paired_public_fa import capture_checkpoint_pair as baseline
    from qwen3_target_capture import capture_checkpoint_pair as candidate
    from qwen_signed_secant_paired_vendor_fa import propagate_paired_secant
    from vendor_fa_finite_runtime import VendorFAFiniteP1
    from score_views import signed_rise_equals_positive_curve
    torch.set_num_threads(4)
    torch.manual_seed(73)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 32)
    torch._dynamo.config.accumulated_cache_size_limit = max(torch._dynamo.config.accumulated_cache_size_limit, 128)
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'loading', 'script_sha256': sha(Path(__file__).read_bytes()),
              'candidate_sha256': sha(Path(inspect.getfile(candidate)).read_bytes()),
              'reference_sha256': args.reference_sha256, 'sample_batch': 1, 'endpoint_batch': 2,
              'generation_calls': 0, 'FT_calls': 0, 'metric_root_calls': 0,
              'calls': [], 'root_calls': [], 'cases': [], 'comparisons': [], 'metrics': {}}
    vectors = {}

    def save():
        path = args.output / 'results.partial'
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        path.replace(args.output / 'results.json')

    def timed(name, fn):
        report['status'] = name
        row = {'name': name, 'status': 'entered'}
        report['calls'].append(row)
        save()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        try:
            result = fn()
            torch.cuda.synchronize()
            row['status'] = 'returned'
            return result, row
        finally:
            row.update(seconds=time.perf_counter()-start,
                       peak_allocated=torch.cuda.max_memory_allocated(),
                       peak_reserved=torch.cuda.max_memory_reserved())
            save()

    try:
        (model, tokenizer), _ = timed('model_load', lambda: author.load_model(env['checkpoint'], 'cuda:0'))
        model.eval().requires_grad_(False)
        report['native_dtype'] = str(next(model.parameters()).dtype)
        assert sha(Path(inspect.getfile(type(model))).read_bytes()) == env['native_model_sha256']
        identities = {n: (type(m).forward, m.forward) for n, m in model.named_modules()}
        finite_fa = VendorFAFiniteP1(env['finite_library'], env['finite_library_sha256'])
        evaluator = LLMAttributionEvaluator(model, tokenizer)
        model.set_attn_implementation('flash_attention_2')

        def check_identity():
            for n, m in model.named_modules():
                assert (type(m).forward, m.forward) == identities[n], n

        for i, row in enumerate(rows):
            case = f"{row['dataset']}_{row['index']}"
            report['cases'].append({k: row[k] for k in ('dataset', 'index', 'input_sha256', 'prompt_length', 'target_length')})
            ids_cpu = torch.tensor(row['input_ids'], dtype=torch.long)[None]
            assert sha(ids_cpu.numpy().tobytes()) == row['input_sha256']
            eligible = [row['user_positions'][j] for j in row['keep']]
            for repeat, mode in enumerate(('baseline', 'candidate', 'baseline', 'candidate', 'candidate', 'baseline')):
                name = case + '_' + mode + ('_cold' if repeat < 2 else f'_measured{repeat}')
                head_events = []
                root_log = []

                def pre_root(_module, call_args, kwargs):
                    root_log.append({'input_shape': list(kwargs['input_ids'].shape),
                                     'logits_to_keep': kwargs.get('logits_to_keep', 0),
                                     'use_cache': kwargs.get('use_cache')})

                def pre_head(_module, call_args):
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    head_events.append((start, end, list(call_args[0].shape)))

                def post_head(_module, call_args, output):
                    head_events[-1][1].record()

                handles = [model.register_forward_pre_hook(pre_root, with_kwargs=True),
                           model.lm_head.register_forward_pre_hook(pre_head),
                           model.lm_head.register_forward_hook(post_head)]
                stages = {}

                def attribute():
                    ids = ids_cpu.to('cuda')
                    mask = torch.ones_like(ids)
                    base = ids.clone()
                    base[0, eligible] = tokenizer.eos_token_id
                    capture = baseline if mode == 'baseline' else candidate
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    before, after = capture(model, base, ids, mask, row['prompt_length'])
                    torch.cuda.synchronize()
                    stages['native_root_and_checkpoint_seconds'] = time.perf_counter()-start
                    stages['native_root_peak_allocated'] = torch.cuda.max_memory_allocated()
                    stages['native_root_peak_reserved'] = torch.cuda.max_memory_reserved()
                    stages['scores32'] = [before['score32_sum64'], after['score32_sum64']]
                    start = time.perf_counter()
                    result = propagate_paired_secant(model, before, after, pv_rule='content_P1', finite_attention=finite_fa)
                    torch.cuda.synchronize()
                    stages['replay_and_finite_seconds'] = time.perf_counter()-start
                    vector = np.asarray(result['signed_full_sequence'], dtype=np.float64)
                    assert np.isfinite(vector).all()
                    return vector

                try:
                    vector, cost = timed(name, attribute)
                finally:
                    for handle in handles:
                        handle.remove()
                check_identity()
                assert len(root_log) == len(head_events) == 1
                cost.update(stages=stages, head_seconds=head_events[0][0].elapsed_time(head_events[0][1])/1000,
                            head_input_shape=head_events[0][2], warm=repeat >= 2)
                # The frozen finite driver resets peak counters internally.
                # Retain the root-stage peak instead of underreporting the API.
                cost['peak_allocated'] = max(cost['peak_allocated'], stages['native_root_peak_allocated'])
                cost['peak_reserved'] = max(cost['peak_reserved'], stages['native_root_peak_reserved'])
                report['root_calls'].append({'name': name, **root_log[0]})
                vectors[name] = vector
                np.savez_compressed(args.output / 'vectors.npz', **vectors)
                save()
                print(json.dumps({'name': name, 'seconds': cost['seconds'], 'head_seconds': cost['head_seconds']}), flush=True)
                del vector
                gc.collect()
            basevec = vectors[case+'_baseline_measured5']
            newvec = vectors[case+'_candidate_measured4']
            ids_positions = row['user_positions']
            b = torch.tensor(basevec[ids_positions], dtype=torch.float32)
            c = torch.tensor(newvec[ids_positions], dtype=torch.float32)
            active = torch.tensor(row['keep'])
            comparison = {'case': case, 'equal_full_vector': bool(np.array_equal(basevec, newvec)),
                          'relative_l2': float(np.linalg.norm(newvec-basevec)/max(np.linalg.norm(basevec), 1e-30)),
                          'max_absolute': float(np.abs(newvec-basevec).max()),
                          'signed_full_order_equal': bool(torch.equal(torch.argsort(b[active], descending=True), torch.argsort(c[active], descending=True))),
                          'baseline_repeat_equal': bool(np.array_equal(basevec, vectors[case+'_baseline_measured2'])),
                          'candidate_repeat_equal': bool(np.array_equal(newvec, vectors[case+'_candidate_measured3']))}
            report['comparisons'].append(comparison)
            # No stale scalar is used to evaluate a changed candidate. The author
            # function is untouched; signed-MAS outputs are explicitly invalid.
            if not comparison['equal_full_vector']:
                model.set_attn_implementation('eager')
                example = dataset_utils.load_cached(Path(env['official_root'])/'exp/exp2/data'/(row['dataset']+'.jsonl'))[0]
                for label, vector in (('baseline', b), ('candidate', c)):
                    curves = {}
                    for view, scores in (('positive', vector.clamp_min(0)), ('signed', vector)):
                        if view == 'signed':
                            proof = signed_rise_equals_positive_curve(vector, row['keep'], curves['positive'])
                            if proof is not None:
                                curves['signed_reuse_proof'] = proof
                                break
                        curve = {'actual_input_hashes': [], 'deleted_user_indices': []}

                        def before_score(_module, call_args, kwargs):
                            actual = kwargs['input_ids'].detach().cpu()
                            assert actual.shape == ids_cpu.shape
                            changed = (actual[0] != ids_cpu[0]).nonzero().flatten().tolist()
                            assert set(changed).issubset(eligible)
                            assert all(int(actual[0,j]) == tokenizer.eos_token_id for j in changed)
                            curve['actual_input_hashes'].append(sha(actual.numpy().tobytes()))
                            curve['deleted_user_indices'].append([ids_positions.index(j) for j in changed])
                            report['metric_root_calls'] += 1

                        def observe(frame, event, value):
                            if frame.f_code is ft.faithfulness_test_skip_tokens.__code__ and event == 'return' and value is not None:
                                for key in ('scores', 'density', 'normalized_model_response', 'alignment_penalty', 'corrected_scores'):
                                    curve[key] = np.asarray(frame.f_locals[key]).tolist()

                        handle = model.register_forward_pre_hook(before_score, with_kwargs=True)
                        assert sys.getprofile() is None
                        sys.setprofile(observe)
                        try:
                            with torch.no_grad():
                                values, _ = timed(case+'_'+label+'_'+view+'_metrics', lambda: ft.faithfulness_test_skip_tokens(
                                    evaluator, scores[None], example.prompt, example.target,
                                    keep_prompt_token_indices=row['keep'], user_prompt_indices=ids_positions, k=20))
                        finally:
                            sys.setprofile(None)
                            handle.remove()
                        assert len(curve['actual_input_hashes']) == 21 and curve['actual_input_hashes'][0] == row['input_sha256']
                        curve.update(zip(('rise', 'mas', 'rise_plus_ap'), map(float, values)))
                        curve['MAS_valid'] = view == 'positive'
                        curves[view] = curve
                    curves['needle'] = float(ft.evaluate_attr_recovery_skip_tokens(vector.clamp_min(0)[None],
                        keep_prompt_token_indices=row['keep'], gold_prompt_token_indices=row['gold'], top_fraction=.1)) if row['gold'] else None
                    report['metrics'][case+'_'+label] = curves
                    save()
                model.set_attn_implementation('flash_attention_2')
                check_identity()
        report['status'] = 'complete'
        report['vectors_sha256'] = sha((args.output/'vectors.npz').read_bytes())
    except Exception:
        report['status'] = 'failed'
        report['error'] = traceback.format_exc()
        raise
    finally:
        save()


if __name__ == '__main__':
    main()

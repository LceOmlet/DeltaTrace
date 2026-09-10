"""Small Recall pilot derived from the frozen, verified source-v2 adapter.

Source-v2 uses the 11 complete evidence task caches with live, matched FT.
Released-v1 retains the historical full-prompt masks and published comparison.
No model, attention, FT method, or evaluator function is replaced here.
"""
import argparse
import copy
import gc
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import sys
import time
import traceback

PILOT = Path(__file__).resolve().parent
ROOT = PILOT.parents[2]
HERE = ROOT / 'experiments/official'
sys.path.insert(0, str(HERE))
sha = lambda data: hashlib.sha256(data).hexdigest()


def arguments(argv=None):
    parser = argparse.ArgumentParser(description='Small matched-input Recall experiment; no deletion sweep.')
    parser.add_argument('--environment', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', choices=['development', 'validation'], required=True)
    parser.add_argument('--datasets', nargs='+', required=True)
    parser.add_argument('--choice', type=Path)
    args = parser.parse_args(argv)
    args.family, args.selection = 'qwen3', 'paper'
    args.evaluation_protocol, args.ft = 'source-v2', 'live'
    args.sentence_recovery, args.paired_reference_audit = False, False
    if args.stage == 'validation':
        assert args.choice is not None, 'Freeze a candidate before validation'
    else:
        assert args.choice is None
    return args


def main():
    args = arguments()
    protocol = json.loads((HERE / 'protocol.json').read_bytes())
    from evidence_protocol import (configure_run, source_span, select_source_tokens,
        recovery_curve, sentence_recovery_curve, reference_token_ids, SOURCE_PROTOCOL)
    configure_run(args, protocol)
    source_mode = args.evaluation_protocol == SOURCE_PROTOCOL
    evaluation_path = HERE / ('source_protocol.json' if source_mode else 'protocol.json')
    evaluation_settings = json.loads(evaluation_path.read_bytes())
    sources = json.loads((ROOT / 'deltatrace/clean/sources.json').read_bytes())
    env = json.loads(args.environment.read_bytes())[args.family]
    development_inputs = json.loads((HERE/'development16_inputs.json').read_bytes())
    for path, receipt in sources['models'][args.family]['files'].items():
        assert sha((ROOT / path).read_bytes()) == receipt['sha256'], path
    official = Path(env['official_root'])
    if args.family == 'qwen35':
        for relative, digest in env['official_extension_blob_sha1'].items():
            raw = (Path(env['ft_extension_root']) / relative).read_bytes()
            assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest() == digest
    for path, digest in protocol['official_normalized_sources'].items():
        assert sha((official / path).read_bytes().replace(b'\r\n', b'\n')) == digest, path
    weight_identity = None
    if args.family == 'qwen3' and args.selection == 'paper':
        receipt_raw = Path(env['checkpoint_receipt']).read_bytes()
        assert sha(receipt_raw) == env['checkpoint_receipt_sha256']
        receipt = json.loads(receipt_raw)
        assert receipt['status'] == 'complete'
        for file in receipt['files']:
            path = Path(env['checkpoint']) / file['name']
            assert path.stat().st_size == file['bytes']
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(8*1024*1024), b''):
                    digest.update(block)
            assert digest.hexdigest() == file['sha256'], file['name']
        weight_identity = {'receipt_sha256': sha(receipt_raw), 'files_verified': len(receipt['files'])}
    split = json.loads((PILOT / 'answer_split.json').read_bytes())
    assert split['plan_sha256'] == sha((PILOT / 'ANSWER_PILOT.md').read_bytes())
    choice = json.loads(args.choice.read_bytes()) if args.choice else None
    if choice:
        assert choice['split_sha256'] == sha((PILOT / 'answer_split.json').read_bytes())
        assert choice['status'] == 'frozen_for_validation'
    caches = {}
    for dataset in args.datasets:
        raw = (official / 'exp/exp2/data' / (dataset + '.jsonl')).read_bytes()
        expected = protocol['tasks'][dataset]
        assert sha(raw) == expected['cache_sha256']
        records = [json.loads(line) for line in raw.decode().splitlines()]
        assert len(records) == expected['count']
        indices = split['tasks'][dataset][args.stage]
        assert indices and len(indices) == len(set(indices))
        assert sha(raw) == split['tasks'][dataset]['cache_sha256']
        caches[dataset] = [(i, records[i]) for i in indices]
        for reference in expected['FT_references'].values():
            assert sha((ROOT / reference['path']).read_bytes()) == reference['sha256']
    if Path('/opt/maca').exists():
        os.environ.setdefault('MACA_PATH', '/opt/maca')
    os.environ.update(HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    os.environ.setdefault('TRITON_CACHE_DIR', env.get('triton_cache', '/tmp/deltatrace_clean_v1_triton'))
    os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR', env.get('inductor_cache', '/tmp/deltatrace_clean_v1_inductor'))
    sys.path.insert(0, str(official))
    for overlay in env.get('dependency_overlays', []):
        sys.path.insert(0, overlay)
    sys.path.insert(0, str(ROOT / 'deltatrace/clean' / args.family))
    if args.family == 'qwen35':
        sys.path.append(env['ft_extension_root'])
    import numpy as np
    import torch
    from recovery_diagnostics import reported_recovery_diagnostics
    from transformers import AutoTokenizer
    from exp.exp2 import run_exp as author
    from exp.exp2 import dataset_utils as data_utils
    import ft_ifr_improve as ft
    from ft_target_control import InitialTargetFT
    from llm_attr_eval import LLMAttributionEvaluator
    # Imported locations must be the checked paper-era source tree.
    for module in (author, data_utils, ft, sys.modules['llm_attr'], sys.modules['llm_attr_eval']):
        assert Path(module.__file__).resolve().is_relative_to(official.resolve())
    torch.set_num_threads(4)
    torch.manual_seed(73)
    torch.backends.cuda.matmul.allow_tf32 = False
    # The frozen method uses static finite graphs. Different original sequence
    # lengths must not hit Dynamo's default eight-variant cap midway through a
    # dataset. This only sizes the official compiler cache; fullgraph remains
    # enabled, with no fallback, method change or model/FT modification.
    compiler_cache_before = {name: getattr(torch._dynamo.config, name) for name in
                            ('cache_size_limit', 'accumulated_cache_size_limit')}
    required_variants = 2 * sum(len(rows) for rows in caches.values()) + 16
    torch._dynamo.config.cache_size_limit = max(compiler_cache_before['cache_size_limit'], required_variants)
    torch._dynamo.config.accumulated_cache_size_limit = max(compiler_cache_before['accumulated_cache_size_limit'], 4 * required_variants)
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'loading', 'family': args.family, 'selection': args.selection,
        'protocol_sha256': sha((HERE / 'protocol.json').read_bytes()),
        'clean_sources_sha256': sha((ROOT / 'deltatrace/clean/sources.json').read_bytes()),
        'driver_sha256': sha(Path(__file__).read_bytes()),
        'recovery_diagnostics_sha256': sha((HERE / 'recovery_diagnostics.py').read_bytes()),
        'evaluation_protocol': args.evaluation_protocol,
        'evaluation_protocol_sha256': sha(evaluation_path.read_bytes()),
        'evaluation_settings': evaluation_settings,
        'evidence_protocol_sha256': sha((HERE / 'evidence_protocol.py').read_bytes()),
        'sentence_recovery_enabled': args.sentence_recovery,
        'paired_reference_audit': args.paired_reference_audit,
        'ft_source': args.ft,
        'cases': [], 'costs': [],
        'weight_identity': weight_identity,
        'published_number_comparison': not source_mode and args.family == 'qwen3' and args.selection == 'paper',
        'generation_calls': 0, 'sample_batch': 1,
        'compiler_cache_before': compiler_cache_before,
        'compiler_cache_limits': {name: getattr(torch._dynamo.config, name) for name in compiler_cache_before},
        'selected_counts': {key: len(value) for key, value in caches.items()}}
    report.update(experiment='answer-pilot-v1', stage=args.stage, split_sha256=sha((PILOT / 'answer_split.json').read_bytes()), plan_sha256=split['plan_sha256'], choice=choice)
    vectors = {}

    def save():
        temp = args.output / 'results.partial'
        temp.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        temp.replace(args.output / 'results.json')

    def timed(name, function):
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        row = {'name': name, 'status': 'entered', 'allocated_before': before}
        report['costs'].append(row)
        try:
            value = function()
            torch.cuda.synchronize()
            row['status'] = 'returned'
            return value
        finally:
            row.update(seconds=time.perf_counter()-started, peak_allocated=torch.cuda.max_memory_allocated(),
                       peak_reserved=torch.cuda.max_memory_reserved())

    save()
    try:
        if args.family == 'qwen3':
            model, tokenizer = timed('model_load', lambda: author.load_model(env['checkpoint'], 'cuda:0'))
        else:
            from transformers import Qwen3_5ForConditionalGeneration
            tokenizer = AutoTokenizer.from_pretrained(env['checkpoint'], local_files_only=True)
            tokenizer.pad_token = tokenizer.eos_token
            model = timed('model_load', lambda: Qwen3_5ForConditionalGeneration.from_pretrained(env['checkpoint'],
                dtype=torch.bfloat16, attn_implementation='eager', device_map={'': 'cuda:0'}, local_files_only=True))
        model.eval().requires_grad_(False)
        assert sha(Path(inspect.getfile(type(model))).read_bytes()) == env['native_model_sha256']
        original_forwards = {name: type(module).forward for name, module in model.named_modules()}
        original_bound_forwards = {name: module.forward for name, module in model.named_modules()}
        evaluator = LLMAttributionEvaluator(model, tokenizer)
        if source_mode:
            assert evaluator._ensure_pad_token_id() == tokenizer.eos_token_id

        def method_identity():
            for name, module in model.named_modules():
                assert type(module).forward is original_forwards[name]
                assert module.forward == original_bound_forwards[name]

        if args.family == 'qwen3':
            from qwen_signed_secant_paired_public_fa import capture_checkpoint_pair
            from qwen_signed_secant_paired_vendor_fa import propagate_paired_secant
            from weighted_paired import propagate_paired_secant as propagate_weighted
            from vendor_fa_finite_runtime import VendorFAFiniteP1
            finite_fa = VendorFAFiniteP1(env['finite_library'], env['finite_library_sha256'])
        else:
            from qwen35_clean_runner import make_qwen35_clean_runner
            from qwen35_answer_finite import PackedAnswerTargets
            from finite_fla_gpu import make_compiled_finite_pullback, verify_native_sources
            from vendor_fa_finite_bf16_d256 import VendorFAFiniteP1BF16D256
            verify_native_sources(env['native_stage_source_sha256'])
            finite_fa = VendorFAFiniteP1BF16D256(env['finite_library'], env['finite_library_sha256'])
            dt_runner = make_qwen35_clean_runner(model, finite_fa, make_compiled_finite_pullback(reuse_scalar_products=False))
            for field in ('norm_gate_rules','finite_fla_by_layer','attention_pv_rules','key_norm_by_layer'):
                assert getattr(dt_runner, field) == {}
            from flashtrace import FlashTrace

        report['FT_initial_target_adapter'] = {'sha256': sha((PILOT/'ft_target_control.py').read_bytes()), 'addendum_sha256': sha((PILOT/'ANSWER_HOPS_ADDENDUM.md').read_bytes())}
        if choice:
            assert choice['FT_initial_target_adapter_sha256'] == report['FT_initial_target_adapter']['sha256']
        report['weighted_sources'] = {name: sha((PILOT / name).read_bytes()) for name in ('weighted_secant.py', 'weighted_paired.py')}
        for dataset, raw_cases in caches.items():
            loaded = data_utils.load_cached(official / 'exp/exp2/data' / (dataset + '.jsonl'))
            for index, raw_case in raw_cases:
                modes = ['full', 'answer_conditioned', 'answer_only'] if choice is None else ['full', choice['target_mode']]
                modes = list(dict.fromkeys(modes))
                for target_mode in modes:
                    ex = copy.deepcopy(loaded[index])
                    original_target = ex.target
                    encoded_target = tokenizer(ex.target, add_special_tokens=False, return_offsets_mapping=True)
                    answer_start, answer_end = map(int, ex.indices_to_explain)
                    assert 0 <= answer_start <= answer_end < len(encoded_target['input_ids'])
                    char_start = encoded_target['offset_mapping'][answer_start][0]
                    char_end = encoded_target['offset_mapping'][answer_end][1]
                    extracted_answer = ex.target[char_start:char_end]
                    assert extracted_answer.strip()
                    if target_mode == 'answer_only':
                        ex.target = extracted_answer
                        new_length = len(tokenizer(ex.target, add_special_tokens=False)['input_ids'])
                        ex.sink_span = ex.indices_to_explain = [0, new_length - 1]
                        ex.thinking_span = None
                    assert ex.prompt == raw_case['prompt'] and original_target == raw_case['target']
                    if args.family == 'qwen35':
                        ex = copy.deepcopy(ex)
                        ex.indices_to_explain = ex.sink_span = ex.thinking_span = None
                        ex = data_utils.attach_spans_from_answer(ex, tokenizer)
                        ex.indices_to_explain = list(ex.sink_span)
                    key = f'{dataset}_{index}_{target_mode}'
                    engine = ft.LLMIFRAttributionBoth(model, tokenizer, show_progress=False)
                    ids, mask, prompt_len, gen_len = engine._ensure_generation(ex.prompt, ex.target)
                    positions = list(engine.user_prompt_indices)
                    keep = ft.keep_token_indices(engine.user_prompt_tokens)
                    formatted = evaluator.format_prompt(' ' + ex.prompt)
                    assert engine.prompt == formatted
                    eval_prompt = tokenizer(formatted, add_special_tokens=False, return_tensors='pt').input_ids.to(model.device)
                    eval_target = tokenizer(ex.target+tokenizer.eos_token, add_special_tokens=False, return_tensors='pt').input_ids.to(model.device)
                    assert torch.equal(ids, torch.cat((eval_prompt, eval_target), dim=1))
                    # Preserve the author's mapping. Standalone tokenization can
                    # merge a boundary differently; it is not the model input.
                    standalone_ids = engine.user_prompt_ids[0].tolist()
                    actual_user_ids = ids[0, positions].tolist()
                    boundary_differences = [j for j,(a,b) in enumerate(zip(actual_user_ids,standalone_ids)) if a != b]
                    if not source_mode and args.family == 'qwen3' and args.selection != 'paper':
                        reference = development_inputs[key]
                        # The historical fixture hashes JSON token lists, whereas
                        # actual-call receipts below hash tensor bytes.
                        assert sha(json.dumps(ids[0].cpu().tolist()).encode()) == reference['input_ids_sha256']
                        assert prompt_len == reference['prompt_len'] and positions == reference['user_positions']
                        assert keep == reference['keep_local_indices']
                    gold = data_utils.ruler_gold_prompt_token_indices(ex, tokenizer)
                    author_keep = list(keep)
                    source = offsets = None
                    if source_mode:
                        source = source_span(dataset, ex.prompt)
                        encoded = tokenizer(' ' + ex.prompt, add_special_tokens=False, return_offsets_mapping=True)
                        assert list(encoded['input_ids']) == standalone_ids
                        offsets = encoded['offset_mapping']
                        keep = select_source_tokens(source, offsets, author_keep, gold)
                        assert not set(boundary_differences) & set(keep), 'Source token mapping differs from the actual model input'
                    eligible = [positions[j] for j in keep]
                    row = {'dataset': dataset, 'index': index, 'input_ids': ids[0].cpu().tolist(),
                        'input_sha256': sha(ids.cpu().numpy().tobytes()), 'prompt_length': prompt_len,
                        'target_length': gen_len, 'user_positions': positions, 'keep': keep, 'gold': gold,
                        'input_matches_unmodified_author_evaluator': True, 'metrics': {}, 'status': 'entered'}
                    row['standalone_token_boundary_differences'] = boundary_differences
                    row['evaluation_protocol'] = args.evaluation_protocol
                    if source_mode:
                        row['author_keep'] = author_keep
                        row['source_span'] = source
                        row['source_eligible_positions'] = eligible
                        row['source_eligible_count'] = len(keep)
                        row['source_gold_count'] = len(set(gold) & set(keep))
                        reference_ids = reference_token_ids(row['input_ids'], eligible, tokenizer.eos_token_id)
                        row['reference_input_sha256'] = sha(np.asarray(reference_ids, dtype=np.int64).tobytes())
                    row.update(target_mode=target_mode, original_target_sha256=sha(original_target.encode()),
                        target=ex.target, original_answer_token_span=[answer_start, answer_end],
                        original_answer_char_span=[char_start, char_end], sink_span=ex.sink_span,
                        thinking_span=ex.thinking_span)
                    assert len(engine.generation_tokens) == gen_len
                    selected_start, selected_end = (0, gen_len - 2) if target_mode == 'full' else ex.sink_span
                    target_weights = [float(selected_start <= j <= selected_end and not ft.is_stop_token(token))
                        for j, token in enumerate(engine.generation_tokens)]
                    assert sum(target_weights) > 0 and target_weights[-1] == 0
                    row['target_weights'] = target_weights
                    report['cases'].append(row)
                    report['status'] = 'attribute_' + key
                    save()
                    if args.family == 'qwen35' and not report.get('initialized'):
                        with torch.no_grad():
                            init = timed('original_eager_initialization', lambda: model(input_ids=ids, attention_mask=mask, use_cache=False))
                        del init
                        report['initialized'] = True
                    model.set_attn_implementation('flash_attention_2')

                    def attribute(reference_override=None, weights=None):
                        if reference_override is not None:
                            base = ids.new_tensor([reference_override])
                        elif source_mode:
                            base = ids.new_tensor([reference_ids])
                            assert sha(base.cpu().numpy().tobytes()) == row['reference_input_sha256']
                        else:
                            base = ids.clone()
                            base[0, eligible] = tokenizer.eos_token_id
                        if args.family == 'qwen3':
                            left_ids, right_ids = base, ids
                            before, after = capture_checkpoint_pair(model, left_ids, right_ids, mask, prompt_len)
                            result = (propagate_paired_secant(model, before, after, pv_rule='content_P1', finite_attention=finite_fa) if weights is None else
                                propagate_weighted(model, before, after, pv_rule='content_P1', finite_attention=finite_fa, target_weights=weights))
                            signed = torch.tensor(result['signed_full_sequence'], dtype=torch.float64)
                        else:
                            pair = torch.cat((base, ids))
                            selection = PackedAnswerTargets([{'target_ids': eval_target[0].cpu(), 'prompt_length': prompt_len}],
                                [list(range(gen_len))], ids.shape[1], model.device)
                            signed, result = dt_runner.attribute(pair, torch.ones_like(pair), selection, select_output_rows=True, observer=None)
                            signed = signed[0]
                        assert bool(torch.isfinite(signed).all())
                        result['output_seed'] = 'original_all_tokens' if weights is None else 'matched_nonstop_target'
                        result['endpoint_target_logprobs32'] = [before['target_logprobs32'].cpu().tolist(), after['target_logprobs32'].cpu().tolist()]
                        return signed.cpu(), result

                    from retrieval_views import sentence_density_order

                    def score(method, scores):
                        values = np.maximum(np.asarray(scores, dtype=np.float32), 0)
                        density_order = sentence_density_order(' ' + ex.prompt, offsets, values, keep)
                        assert len(density_order) == len(keep) and set(density_order) == set(keep)
                        rank_values = np.zeros_like(values)
                        rank_values[density_order] = np.arange(len(keep), 0, -1)
                        row['metrics'][method] = {
                            'raw': recovery_curve(values, keep, gold, [.05, .1, .2]),
                            'density': recovery_curve(rank_values, keep, gold, [.05, .1, .2])}

                    full_reference = reference_token_ids(row['input_ids'],
                        [positions[j] for j in author_keep], tokenizer.eos_token_id)
                    row['references'] = {'full': sha(np.asarray(full_reference, dtype=np.int64).tobytes())}
                    seeds = [('DT_target', target_weights)]
                    if target_mode == 'full':
                        seeds.insert(0, ('DT_original', None))
                        if args.stage == 'development' and index == raw_cases[0][0]:
                            seeds.append(('DT_weighted_identity', [1.] * gen_len))
                    for method, weights in seeds:
                        signed, detail = timed(key + '_' + method,
                            lambda weights=weights: attribute(full_reference, weights))
                        vectors[key + '_' + method + '_signed_full'] = signed.numpy()
                        row[method + '_details'] = detail
                        score(method, signed.numpy()[positions])
                        if method == 'DT_weighted_identity':
                            assert np.array_equal(vectors[key + '_DT_original_signed_full'], signed.numpy()), 'Weighted identity changed the frozen attribution'
                            row['weighted_identity_bitwise_equal'] = True
                    method_identity()
                    model.set_attn_implementation('eager')

                    if args.ft == 'live':
                        for hops in ([1,3] if gold else [1]):
                            def trace():
                                if args.family == 'qwen3':
                                    tracer_class = InitialTargetFT if target_mode == 'answer_conditioned' else ft.LLMIFRAttributionBoth
                                    original = tracer_class(model, tokenizer, chunk_tokens=128, sink_chunk_tokens=32, show_progress=False)
                                    seed_kwargs = {'initial_target_mask': target_weights} if target_mode == 'answer_conditioned' else {}
                                    result = original.calculate_ifr_multi_hop_both(ex.prompt, target=ex.target,
                                        sink_span=tuple(ex.sink_span), thinking_span=tuple(ex.thinking_span) if ex.thinking_span is not None else None, n_hops=hops, **seed_kwargs)
                                    seq, _, _ = result.get_all_token_attrs(ex.indices_to_explain)
                                    assert seq.shape[1]-seq.shape[0] == len(positions)
                                    return seq[:, :len(positions)].sum(0).detach().cpu()
                                tracer = FlashTrace(model, tokenizer, use_chat_template=False)
                                result = tracer.trace(prompt=formatted, target=ex.target, output_span=tuple(ex.sink_span),
                                    reasoning_span=tuple(ex.thinking_span), hops=hops, method='flashtrace')
                                assert len(result.scores) == prompt_len
                                return torch.tensor(result.scores, dtype=torch.float32)[positions]
                            expected_ft_inputs = []
                            def verify_ft_input(_module, call_args, kwargs):
                                assert torch.equal(kwargs['input_ids'], ids)
                                expected_ft_inputs.append(row['input_sha256'])
                            handle = model.register_forward_pre_hook(verify_ft_input, with_kwargs=True)
                            aggregation_calls = []
                            aggregate_code = inspect.unwrap(ft.compute_ifr_sentence_aggregate).__code__
                            def observe_aggregation(frame, event, _arg):
                                if event == 'call' and frame.f_code is aggregate_code:
                                    values = frame.f_locals
                                    weights = values.get('sink_weights')
                                    aggregation_calls.append({'start': int(values['sink_start']) - prompt_len,
                                        'end': int(values['sink_end']) - prompt_len,
                                        'weights': weights.detach().float().cpu().tolist() if weights is not None else None})
                            assert sys.getprofile() is None
                            sys.setprofile(observe_aggregation)
                            try:
                                ft_score = timed(key+f'_original_FT_K{hops}', trace)
                            finally:
                                sys.setprofile(None)
                                handle.remove()
                            assert aggregation_calls, 'No actual FT target aggregation was observed'
                            assert all(item['start'] == 0 and item['end'] == gen_len-2 for item in aggregation_calls), 'Reasoning-hop support was reduced'
                            first = aggregation_calls[0]
                            actual_weights = [0.] * gen_len
                            start, end = first['start'], first['end']
                            assert 0 <= start <= end < gen_len - 1
                            actual_weights[start:end+1] = first['weights'] if first['weights'] is not None else [1.] * (end-start+1)
                            assert actual_weights == target_weights, 'DT/FT target seed positions differ'
                            row[f'FT_K{hops}_actual_target_aggregation'] = aggregation_calls
                            assert len(expected_ft_inputs) == 1
                            vectors[key+f'_FT_K{hops}_prompt'] = ft_score.numpy()
                            score(f'FT_K{hops}', ft_score.numpy())
                    else:
                        row['published_FT_reference'] = protocol['tasks'][dataset]['FT_references']
                    row['status'] = 'complete'
                    np.savez_compressed(args.output/'vectors.npz', **vectors)
                    method_identity()
                    save()
                    print(json.dumps({'case': key, 'stage': args.stage, 'status': 'complete'}), flush=True)
                    del engine, ids, mask, signed, detail
                    gc.collect()
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

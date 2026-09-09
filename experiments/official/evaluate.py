"""Evaluate frozen DT with the author's experiment inputs, targets and metrics.

The default paper mode uses whole released task caches. Development16 is a
separate selection and never compares its mean against full-paper CSV means.
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
from score_views import signed_rise_equals_positive_curve

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sha = lambda data: hashlib.sha256(data).hexdigest()


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family', choices=['qwen3', 'qwen35'], required=True)
    parser.add_argument('--environment', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--selection', choices=['paper', 'development16', 'smoke'], required=True)
    parser.add_argument('--datasets', nargs='+', help='Paper: all released tasks by default; development/smoke: NI and MH.')
    parser.add_argument('--ft', choices=['live', 'published'], help='Defaults to published FT for Qwen3 paper runs; otherwise live.')
    parser.add_argument('--dt-backend', choices=['clean','accelerated_qwen35'], default='clean')
    parser.add_argument('--sample-batch', type=int, default=1, help='Real examples per DT call; acceleration only.')
    return parser.parse_args()


def main():
    args = arguments()
    if args.sample_batch < 1 or (args.dt_backend == 'clean' and args.sample_batch != 1):
        raise ValueError('Clean DT retains sample batch1; batching requires the explicit acceleration backend.')
    if args.dt_backend == 'accelerated_qwen35' and args.family != 'qwen35':
        raise ValueError('The accelerated batch backend has only been implemented for Qwen3.5.')
    protocol = json.loads((HERE / 'protocol.json').read_bytes())
    if args.datasets is None:
        args.datasets = list(protocol['tasks']) if args.selection == 'paper' else ['niah_mq_q2', 'morehopqa']
    if args.ft is None:
        args.ft = 'published' if args.family == 'qwen3' and args.selection == 'paper' else 'live'
    if len(set(args.datasets)) != len(args.datasets) or any(name not in protocol['tasks'] for name in args.datasets):
        raise ValueError('Datasets must be unique tasks from the original released table.')
    sources = json.loads((ROOT / 'deltatrace/clean/sources.json').read_bytes())
    env = json.loads(args.environment.read_bytes())[args.family]
    development_inputs = json.loads((HERE/'development16_inputs.json').read_bytes())
    if args.ft == 'published' and (args.family != 'qwen3' or args.selection != 'paper'):
        raise ValueError('Published FT means require Qwen3 and the complete published task selection.')
    if args.selection != 'paper' and not set(args.datasets).issubset({'niah_mq_q2', 'morehopqa'}):
        raise ValueError('Development/smoke runs only select the frozen NI and MH tasks.')
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
    caches = {}
    for dataset in args.datasets:
        raw = (official / 'exp/exp2/data' / (dataset + '.jsonl')).read_bytes()
        expected = protocol['tasks'][dataset]
        assert sha(raw) == expected['cache_sha256']
        records = [json.loads(line) for line in raw.decode().splitlines()]
        assert len(records) == expected['count']
        count = len(records) if args.selection == 'paper' else 8 if args.selection == 'development16' else 1
        caches[dataset] = records[:count]
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
    from transformers import AutoTokenizer
    from exp.exp2 import run_exp as author
    from exp.exp2 import dataset_utils as data_utils
    import ft_ifr_improve as ft
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
        'score_views_sha256': sha((HERE/'score_views.py').read_bytes()),
        'result_schema': 2, 'cases': [], 'costs': [],
        'weight_identity': weight_identity,
        'published_number_comparison': args.family == 'qwen3' and args.selection == 'paper',
        'generation_calls': 0, 'sample_batch': args.sample_batch, 'dt_backend': args.dt_backend, 'DT_batches': [],
        'native_autotune_environment': {name: os.environ.get(name) for name in
            ('TRITON_ENABLE_PERSISTENT_AUTOTUNE_CONFIGS','TRITON_AUTOTUNE_CONFIG_PATH')},
        'compiler_cache_before': compiler_cache_before,
        'compiler_cache_limits': {name: getattr(torch._dynamo.config, name) for name in compiler_cache_before},
        'selected_counts': {key: len(value) for key, value in caches.items()}}
    vectors = {}

    def save():
        temp = args.output / 'results.partial'
        temp.write_text(json.dumps(report, ensure_ascii=False, separators=(',', ':'), allow_nan=False), encoding='utf-8')
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

        def method_identity():
            for name, module in model.named_modules():
                assert type(module).forward is original_forwards[name]
                assert module.forward == original_bound_forwards[name]

        if args.family == 'qwen3':
            from qwen_signed_secant_paired_public_fa import capture_checkpoint_pair
            from qwen_signed_secant_paired_vendor_fa import propagate_paired_secant
            from vendor_fa_finite_runtime import VendorFAFiniteP1
            finite_fa = VendorFAFiniteP1(env['finite_library'], env['finite_library_sha256'])
        else:
            from qwen35_clean_runner import make_qwen35_clean_runner
            from qwen35_answer_finite import PackedAnswerTargets
            from finite_fla_gpu import make_compiled_finite_pullback, verify_native_sources
            from vendor_fa_finite_bf16_d256 import VendorFAFiniteP1BF16D256
            verify_native_sources(env['native_stage_source_sha256'])
            finite_fa = VendorFAFiniteP1BF16D256(env['finite_library'], env['finite_library_sha256'])
            finite_fla = make_compiled_finite_pullback(reuse_scalar_products=False)
            if args.dt_backend == 'clean':
                dt_runner = make_qwen35_clean_runner(model, finite_fa, finite_fla)
            else:
                from batching import make_accelerated_runner, group_cases, attribute_batch
                dt_runner, report['acceleration_sources'] = make_accelerated_runner(ROOT, model, finite_fa, finite_fla)
                report['batching_sha256'] = sha((HERE/'batching.py').read_bytes())
            for field in ('norm_gate_rules','finite_fla_by_layer','attention_pv_rules','key_norm_by_layer'):
                assert getattr(dt_runner, field) == {}
            from flashtrace import FlashTrace

        def prepare_case(dataset, index, raw_case, loaded):
            ex = loaded[index]
            assert ex.prompt == raw_case['prompt'] and ex.target == raw_case['target']
            if args.family == 'qwen35':
                ex = copy.deepcopy(ex)
                ex.indices_to_explain = ex.sink_span = ex.thinking_span = None
                ex = data_utils.attach_spans_from_answer(ex, tokenizer)
                ex.indices_to_explain = list(ex.sink_span)
            key = f'{dataset}_{index}'
            engine = ft.LLMIFRAttributionBoth(model, tokenizer, show_progress=False)
            ids, mask, prompt_len, gen_len = engine._ensure_generation(ex.prompt, ex.target)
            positions = list(engine.user_prompt_indices)
            keep = ft.keep_token_indices(engine.user_prompt_tokens)
            eligible = [positions[j] for j in keep]
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
            if args.family == 'qwen3' and args.selection != 'paper':
                reference = development_inputs[key]
                # The historical fixture hashes JSON token lists, whereas
                # actual-call receipts below hash tensor bytes.
                assert sha(json.dumps(ids[0].cpu().tolist()).encode()) == reference['input_ids_sha256']
                assert prompt_len == reference['prompt_len'] and positions == reference['user_positions']
                assert keep == reference['keep_local_indices']
            gold = data_utils.ruler_gold_prompt_token_indices(ex, tokenizer)
            row = {'dataset': dataset, 'index': index, 'input_ids': ids[0].cpu().tolist(),
                'input_sha256': sha(ids.cpu().numpy().tobytes()), 'prompt_length': prompt_len,
                'target_length': gen_len, 'user_positions': positions, 'keep': keep, 'gold': gold,
                'input_matches_unmodified_author_evaluator': True, 'metrics': {}, 'status': 'entered'}
            row['standalone_token_boundary_differences'] = boundary_differences
            return {'key':key,'ex':ex,'ids':ids.cpu(),'mask':mask.cpu(),'prompt_len':prompt_len,
                'gen_len':gen_len,'positions':positions,'keep':keep,'eligible':eligible,'formatted':formatted,
                'eval_target':eval_target.cpu(),'gold':gold,'row':row}

        for dataset, raw_cases in caches.items():
            loaded = data_utils.load_cached(official / 'exp/exp2/data' / (dataset + '.jsonl'))
            if args.dt_backend == 'clean':
                groups = ([prepare_case(dataset,index,raw_case,loaded)] for index,raw_case in enumerate(raw_cases))
            else:
                prepared = [prepare_case(dataset,index,raw_case,loaded) for index,raw_case in enumerate(raw_cases)]
                groups = group_cases(prepared,args.sample_batch)
            for group in groups:
                if args.dt_backend != 'clean':
                    if not report.get('initialized'):
                        with torch.no_grad():
                            initial=timed('original_eager_initialization',lambda:model(input_ids=group[0]['ids'].to(model.device),attention_mask=group[0]['mask'].to(model.device),use_cache=False))
                        del initial
                        report['initialized']=True
                    model.set_attn_implementation('flash_attention_2')
                    batch_id=len(report['DT_batches'])
                    report['cases'].extend(c['row'] for c in group)
                    report['status']=f'attribute_batch_{batch_id}'
                    save()
                    batch_signed,batch_detail,root_call=timed(f'DT_batch{batch_id}',lambda:attribute_batch(dt_runner,group,tokenizer.eos_token_id,model.device))
                    report['DT_batches'].append({'batch_id':batch_id,'cases':[c['key'] for c in group],
                        'actual_root':root_call,'details':batch_detail})
                    for j,state in enumerate(group):
                        state['signed']=batch_signed[j]
                        state['DT_details']={'batch_id':batch_id,'sample_index':j,'details_scope':'Details and finite conservation totals belong to the complete batch.'}
                    del batch_signed,batch_detail
                    method_identity()
                for state in group:
                    key,ex=state['key'],state['ex']
                    ids,mask=state['ids'].to(model.device),state['mask'].to(model.device)
                    prompt_len,gen_len=state['prompt_len'],state['gen_len']
                    positions,keep,eligible=state['positions'],state['keep'],state['eligible']
                    formatted,eval_target=state['formatted'],state['eval_target'].to(model.device)
                    gold,row=state['gold'],state['row']
                    if args.dt_backend == 'clean': report['cases'].append(row)
                    report['status'] = 'attribute_' + key
                    save()
                    if args.family == 'qwen35' and not report.get('initialized'):
                        with torch.no_grad():
                            init = timed('original_eager_initialization', lambda: model(input_ids=ids, attention_mask=mask, use_cache=False))
                        del init
                        report['initialized'] = True
                    model.set_attn_implementation('flash_attention_2')

                    def attribute():
                        base = ids.clone()
                        base[0, eligible] = tokenizer.eos_token_id
                        if args.family == 'qwen3':
                            before, after = capture_checkpoint_pair(model, base, ids, mask, prompt_len)
                            result = propagate_paired_secant(model, before, after, pv_rule='content_P1', finite_attention=finite_fa)
                            signed = torch.tensor(result['signed_full_sequence'], dtype=torch.float64)
                        else:
                            pair = torch.cat((base, ids))
                            selection = PackedAnswerTargets([{'target_ids': eval_target[0].cpu(), 'prompt_length': prompt_len}],
                                [list(range(gen_len))], ids.shape[1], model.device)
                            signed, result = dt_runner.attribute(pair, torch.ones_like(pair), selection, select_output_rows=True, observer=None)
                            signed = signed[0]
                        assert bool(torch.isfinite(signed).all())
                        return signed.cpu(), result

                    if args.dt_backend == 'clean':
                        signed, detail = timed(key+'_DT', attribute)
                    else:
                        signed, detail = state['signed'], state['DT_details']
                    row['DT_details'] = detail
                    dt_score = signed[positions].float().clamp_min(0)
                    vectors[key+'_DT_signed_full'] = signed.numpy()
                    vectors[key+'_DT_positive_prompt'] = dt_score.numpy()
                    method_identity()
                    model.set_attn_implementation('eager')

                    def score(method, scores):
                        curve = {'actual_input_hashes': [], 'deleted_user_indices': [], 'raw_response': []}
                        def before_score(_module, call_args, kwargs):
                            actual = kwargs['input_ids'].detach().cpu()
                            assert actual.shape == ids.shape
                            changed = (actual[0] != ids[0].cpu()).nonzero().flatten().tolist()
                            assert set(changed).issubset(eligible)
                            assert all(int(actual[0,j]) == tokenizer.eos_token_id for j in changed)
                            assert torch.equal(actual[:, prompt_len:], eval_target.cpu())
                            curve['actual_input_hashes'].append(sha(actual.numpy().tobytes()))
                            curve['deleted_user_indices'].append([positions.index(j) for j in changed])
                        def observe(frame, event, arg):
                            if frame.f_code is ft.faithfulness_test_skip_tokens.__code__ and event == 'return' and arg is not None:
                                for field in ('scores','density','normalized_model_response','alignment_penalty','corrected_scores'):
                                    curve[field] = np.asarray(frame.f_locals[field]).tolist()
                        handle = model.register_forward_pre_hook(before_score, with_kwargs=True)
                        assert sys.getprofile() is None
                        sys.setprofile(observe)
                        try:
                            with torch.no_grad():
                                values = timed(key+'_'+method+'_original_metrics', lambda: ft.faithfulness_test_skip_tokens(
                                    evaluator, scores[None], ex.prompt, ex.target, keep_prompt_token_indices=keep,
                                    user_prompt_indices=positions, k=20))
                        finally:
                            sys.setprofile(None)
                            handle.remove()
                        assert len(curve['actual_input_hashes']) == 21
                        assert curve['actual_input_hashes'][0] == row['input_sha256']
                        curve['rise'], curve['mas'], curve['rise_plus_ap'] = map(float, values)
                        curve['needle'] = float(ft.evaluate_attr_recovery_skip_tokens(scores[None],
                            keep_prompt_token_indices=keep, gold_prompt_token_indices=gold, top_fraction=.1)) if gold else None
                        row['metrics'][method] = curve

                    # RISE uses signed ordering; MAS requires its own positive view.
                    # Keep both curves explicit rather than attaching a signed RISE
                    # scalar to the positive curve's response/density arrays.
                    score('DT_positive', dt_score)
                    signed_prompt=signed[positions].float()
                    identity_proof=signed_rise_equals_positive_curve(signed_prompt,keep,row['metrics']['DT_positive'])
                    if identity_proof is None:
                        score('DT_signed', signed_prompt)
                        row['metrics']['DT_signed']['MAS_is_valid_for_this_view']=False
                        signed_rise=row['metrics']['DT_signed']['rise']
                    else:
                        signed_rise=row['metrics']['DT_positive']['rise']
                    row['metrics']['DT']={
                        'rise':signed_rise,'mas':row['metrics']['DT_positive']['mas'],
                        'needle':row['metrics']['DT_positive']['needle'],
                        'views':{'rise':'signed','mas':'positive_part','needle':'positive_part'},
                        'signed_RISE_reuse_proof':identity_proof}
                    if args.ft == 'live':
                        for hops in ([1,3] if gold else [1]):
                            def trace():
                                if args.family == 'qwen3':
                                    original = ft.LLMIFRAttributionBoth(model, tokenizer, chunk_tokens=128, sink_chunk_tokens=32, show_progress=False)
                                    result = original.calculate_ifr_multi_hop_both(ex.prompt, target=ex.target,
                                        sink_span=tuple(ex.sink_span), thinking_span=tuple(ex.thinking_span), n_hops=hops)
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
                            try:ft_score = timed(key+f'_original_FT_K{hops}', trace)
                            finally:handle.remove()
                            assert len(expected_ft_inputs) == 1
                            vectors[key+f'_FT_K{hops}_prompt'] = ft_score.numpy()
                            if hops == 1:score('FT_K1', ft_score)
                            else:row['FT_K3_needle'] = float(ft.evaluate_attr_recovery_skip_tokens(ft_score[None],
                                keep_prompt_token_indices=keep, gold_prompt_token_indices=gold, top_fraction=.1))
                    else:
                        row['published_FT_reference'] = protocol['tasks'][dataset]['FT_references']
                    row['status'] = 'complete'
                    np.savez_compressed(args.output/'vectors.npz', **vectors)
                    method_identity()
                    save()
                    print(json.dumps({'case':key,'DT_RISE':row['metrics']['DT']['rise'],'DT_MAS':row['metrics']['DT']['mas'],
                        'DT_needle':row['metrics']['DT']['needle']}), flush=True)
                    del ids, mask, signed, detail, dt_score
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

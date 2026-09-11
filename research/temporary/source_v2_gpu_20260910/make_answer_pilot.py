"""Freeze disjoint samples and derive a weighted-seed experiment without editing frozen methods."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
release = json.loads((ROOT / 'experiments/official/protocol.json').read_bytes())
tasks = ['vt_h2_c3', 'vt_h4_c1', 'vt_h6_c1', 'vt_h10_c1', 'hotpotqa_long']
split = {'version': 'answer-pilot-v1', 'plan_sha256': sha(HERE / 'ANSWER_PILOT.md'), 'tasks': {}}
old_split = json.loads((HERE / 'recall_split.json').read_bytes())
for task in tasks:
    order = sorted(range(release['tasks'][task]['count']), key=lambda i:
        hashlib.sha256(f'recall-pilot-v1|{task}|{i}'.encode()).hexdigest())
    dev = sorted(order[:8]) if task in ('vt_h2_c3', 'hotpotqa_long') else []
    val = sorted(order[24:40])
    assert not set(val) & set(old_split['tasks'][task]['validation'] + old_split['tasks'][task]['development'] + dev)
    split['tasks'][task] = {'development': dev, 'validation': val,
        'cache_sha256': release['tasks'][task]['cache_sha256']}
path = HERE / 'answer_split.json'
if path.exists():
    assert json.loads(path.read_bytes()) == split
else:
    path.write_text(json.dumps(split, indent=2) + '\n', encoding='utf-8')

source = ROOT / 'deltatrace/clean/qwen3/qwen_signed_secant_vendor_fa.py'
manifest = json.loads((ROOT / 'deltatrace/clean/sources.json').read_bytes())
assert sha(source) == manifest['models']['qwen3']['files'][source.relative_to(ROOT).as_posix()]['sha256']
weighted = source.read_text(encoding='utf-8')
old = "def propagate_signed_secant(model, before, after, variant='rescale', progress=None, pv_rule='content_P1', finite_attention=None, finite_activity=None):"
assert weighted.count(old) == 1
weighted = weighted.replace(old, old[:-2] + ', target_weights=None):')
old = "        g_delta = after['score32_sum64'] - before['score32_sum64']"
new = old + '''
        if target_weights is not None:
            weights = torch.as_tensor(target_weights, device=device, dtype=seed.dtype)
            assert weights.shape == after['target'].shape and torch.isfinite(weights).all()
            assert (weights >= 0).all() and (weights > 0).any()
            seed = seed * weights[:, None]
            g_delta = float(((after['target_logprobs32'].double() - before['target_logprobs32'].double()) * weights.double()).sum())
'''
assert weighted.count(old) == 1
weighted = weighted.replace(old, new)
weighted = weighted.replace("'target_delta_score16': after['score16'] - before['score16']", "'target_delta_score16': after['score16'] - before['score16'] if target_weights is None else None")
weighted = '# Experimental output-seed selection; derived from frozen source SHA256 ' + sha(source) + '\n' + weighted
compile(weighted, 'weighted_secant.py', 'exec')
(HERE / 'weighted_secant.py').write_text(weighted, encoding='utf-8')

source = ROOT / 'deltatrace/clean/qwen3/qwen_signed_secant_paired_vendor_fa.py'
wrapper = source.read_text(encoding='utf-8')
wrapper = wrapper.replace('from qwen_signed_secant_vendor_fa import propagate_signed_secant', 'from weighted_secant import propagate_signed_secant')
wrapper = wrapper.replace('finite_activity=None):', 'finite_activity=None,target_weights=None):')
wrapper = wrapper.replace('finite_activity=finite_activity)', 'finite_activity=finite_activity,target_weights=target_weights)')
compile(wrapper, 'weighted_paired.py', 'exec')
(HERE / 'weighted_paired.py').write_text(wrapper, encoding='utf-8')

source = HERE / 'evaluate_recall.py'
assert sha(source) == '2be875544d8b90dbaf63706ac080111b2e8a6e6a0026369375051043cb00b3f0'
code = source.read_text(encoding='utf-8')
code = code.replace('recall-pilot-v1', 'answer-pilot-v1').replace('recall_split.json', 'answer_split.json').replace('RECALL_PILOT.md', 'ANSWER_PILOT.md')
code = code.replace('from qwen_signed_secant_paired_vendor_fa import propagate_paired_secant',
    'from qwen_signed_secant_paired_vendor_fa import propagate_paired_secant\n            from weighted_paired import propagate_paired_secant as propagate_weighted')
code = code.replace('        for dataset, raw_cases in caches.items():', '''        report['weighted_sources'] = {name: sha((PILOT / name).read_bytes()) for name in ('weighted_secant.py', 'weighted_paired.py')}
        for dataset, raw_cases in caches.items():''')
start = code.index('                ex = loaded[index]')
end = code.index("        report['status'] = 'complete'", start)
body = code[start:end]
body = body.replace('                ex = loaded[index]', '''                ex = copy.deepcopy(loaded[index])
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
                elif target_mode == 'answer_conditioned':
                    ex.thinking_span = list(ex.sink_span)''')
body = body.replace('assert ex.prompt == raw_case[\'prompt\'] and ex.target == raw_case[\'target\']',
                    'assert ex.prompt == raw_case[\'prompt\'] and original_target == raw_case[\'target\']')
body = body.replace("key = f'{dataset}_{index}'", "key = f'{dataset}_{index}_{target_mode}'")
body = body.replace("                report['cases'].append(row)", '''                row.update(target_mode=target_mode, original_target_sha256=sha(original_target.encode()),
                    target=ex.target, original_answer_token_span=[answer_start, answer_end],
                    original_answer_char_span=[char_start, char_end], sink_span=ex.sink_span,
                    thinking_span=ex.thinking_span)
                assert len(engine.generation_tokens) == gen_len
                selected_start, selected_end = (0, gen_len - 2) if target_mode == 'full' else ex.sink_span
                target_weights = [float(selected_start <= j <= selected_end and not ft.is_stop_token(token))
                    for j, token in enumerate(engine.generation_tokens)]
                assert sum(target_weights) > 0 and target_weights[-1] == 0
                row['target_weights'] = target_weights
                report['cases'].append(row)''')
body = body.replace('def attribute(reference_override=None, reverse=False):', 'def attribute(reference_override=None, weights=None):')
body = body.replace('left_ids, right_ids = (ids, base) if reverse else (base, ids)', 'left_ids, right_ids = base, ids')
body = body.replace("result = propagate_paired_secant(model, before, after, pv_rule='content_P1', finite_attention=finite_fa)",
                    "result = (propagate_paired_secant(model, before, after, pv_rule='content_P1', finite_attention=finite_fa) if weights is None else\n                            propagate_weighted(model, before, after, pv_rule='content_P1', finite_attention=finite_fa, target_weights=weights))")
body = body.replace("result['pilot_reported_sign'] = -1 if reverse else 1\n                    return (-signed if reverse else signed).cpu(), result",
                    "result['output_seed'] = 'original_all_tokens' if weights is None else 'matched_nonstop_target'\n                    result['endpoint_target_logprobs32'] = [before['target_logprobs32'].cpu().tolist(), after['target_logprobs32'].cpu().tolist()]\n                    return signed.cpu(), result")
vstart = body.index("                references = {'body': reference_ids")
vend = body.index('                method_identity()', vstart)
body = body[:vstart] + '''                full_reference = reference_token_ids(row['input_ids'],
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
''' + body[vend:]
body = body.replace('thinking_span=tuple(ex.thinking_span)', 'thinking_span=tuple(ex.thinking_span) if ex.thinking_span is not None else None')
body = body.replace("                        try:ft_score = timed(key+f'_original_FT_K{hops}', trace)\n                        finally:handle.remove()", '''                        aggregation_calls = []
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
                        first = aggregation_calls[0]
                        actual_weights = [0.] * gen_len
                        start, end = first['start'], first['end']
                        assert 0 <= start <= end < gen_len - 1
                        actual_weights[start:end+1] = first['weights'] if first['weights'] is not None else [1.] * (end-start+1)
                        assert actual_weights == target_weights, 'DT/FT target seed positions differ'
                        row[f'FT_K{hops}_actual_target_aggregation'] = aggregation_calls''')
body = body.replace('del engine, ids, mask, signed, detail, signed_by_method', 'del engine, ids, mask, signed, detail')
code = code[:start] + "                modes = ['full', 'answer_conditioned', 'answer_only'] if choice is None else ['full', choice['target_mode']]\n                modes = list(dict.fromkeys(modes))\n                for target_mode in modes:\n" + ''.join('    ' + line if line.strip() else line for line in body.splitlines(keepends=True)) + code[end:]
compile(code, 'evaluate_answer.py', 'exec')
(HERE / 'evaluate_answer.py').write_text(code, encoding='utf-8')
print(json.dumps({'status': 'answer_pilot_frozen', 'split': split,
    'files': {n: sha(HERE / n) for n in ('evaluate_answer.py', 'weighted_secant.py', 'weighted_paired.py')}}))

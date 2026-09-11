"""Create the small Recall experiment from the frozen, verified GPU adapter."""
from pathlib import Path
import hashlib

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
source = ROOT / 'experiments/official/evaluate.py'
assert hashlib.sha256(source.read_bytes()).hexdigest() == 'adeed3f872dc299ebf7ba86cd167e383989609d3a0631f9eff72a199357d8857'
code = source.read_text(encoding='utf-8')


def replace(old, new):
    global code
    assert code.count(old) == 1, old[:80]
    code = code.replace(old, new)


replace('ROOT = Path(__file__).resolve().parents[2]\nHERE = Path(__file__).resolve().parent',
    "PILOT = Path(__file__).resolve().parent\nROOT = PILOT.parents[2]\nHERE = ROOT / 'experiments/official'\nsys.path.insert(0, str(HERE))")
start = code.index('def arguments(')
end = code.index('\n\ndef main():', start)
code = code[:start] + '''def arguments(argv=None):
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
''' + code[end:]
replace('    caches = {}', '''    split = json.loads((PILOT / 'recall_split.json').read_bytes())
    assert split['plan_sha256'] == sha((PILOT / 'RECALL_PILOT.md').read_bytes())
    choice = json.loads(args.choice.read_bytes()) if args.choice else None
    if choice:
        assert choice['split_sha256'] == sha((PILOT / 'recall_split.json').read_bytes())
        assert choice['status'] == 'frozen_for_validation'
    caches = {}''')
replace("        count = len(records) if args.selection == 'paper' else 8 if args.selection == 'development16' else 1\n        caches[dataset] = records[:count]",
    "        indices = split['tasks'][dataset][args.stage]\n        assert indices and len(indices) == len(set(indices))\n        assert sha(raw) == split['tasks'][dataset]['cache_sha256']\n        caches[dataset] = [(i, records[i]) for i in indices]")
replace("    vectors = {}", "    report.update(experiment='recall-pilot-v1', stage=args.stage, split_sha256=sha((PILOT / 'recall_split.json').read_bytes()), plan_sha256=split['plan_sha256'], choice=choice)\n    vectors = {}")
replace('            for index, raw_case in enumerate(raw_cases):', '            for index, raw_case in raw_cases:')
replace('                def attribute(reference_override=None):', '                def attribute(reference_override=None, reverse=False):')
replace('                        before, after = capture_checkpoint_pair(model, base, ids, mask, prompt_len)',
    '                        left_ids, right_ids = (ids, base) if reverse else (base, ids)\n                        before, after = capture_checkpoint_pair(model, left_ids, right_ids, mask, prompt_len)')
replace('                    return signed.cpu(), result', '                    result[\'pilot_reported_sign\'] = -1 if reverse else 1\n                    return (-signed if reverse else signed).cpu(), result')
start = code.index('                signed, detail = timed(')
end = code.index("                if args.ft == 'live':", start)
code = code[:start] + '''                from retrieval_views import sentence_density_order

                def score(method, scores):
                    values = np.maximum(np.asarray(scores, dtype=np.float32), 0)
                    density_order = sentence_density_order(' ' + ex.prompt, offsets, values, keep)
                    assert len(density_order) == len(keep) and set(density_order) == set(keep)
                    rank_values = np.zeros_like(values)
                    rank_values[density_order] = np.arange(len(keep), 0, -1)
                    row['metrics'][method] = {
                        'raw': recovery_curve(values, keep, gold, [.05, .1, .2]),
                        'density': recovery_curve(rank_values, keep, gold, [.05, .1, .2])}

                references = {'body': reference_ids, 'full': reference_token_ids(row['input_ids'],
                    [positions[j] for j in author_keep], tokenizer.eos_token_id)}
                row['references'] = {name: sha(np.asarray(ref, dtype=np.int64).tobytes())
                    for name, ref in references.items()}
                pairs = [('body', False), ('body', True), ('full', False), ('full', True)]
                if choice:
                    ref, direction, _ = choice['candidate'].split('/')
                    needed = {(ref, False)} if direction == 'forward' else {(ref, True)} if direction == 'reverse' else {(ref, False), (ref, True)}
                    pairs = [p for p in pairs if p in needed | {('body', False), ('full', False)}]
                signed_by_method = {}
                for ref, reverse in pairs:
                    method = ref + ('_reverse' if reverse else '_forward')
                    signed, detail = timed(key + '_DT_' + method,
                        lambda ref=ref, reverse=reverse: attribute(references[ref], reverse))
                    assert len(signed) == ids.shape[1]
                    signed_by_method[method] = signed.numpy()
                    vectors[key + '_DT_' + method + '_signed_full'] = signed.numpy()
                    row['DT_' + method + '_details'] = detail
                    score('DT_' + method, signed.numpy()[positions])
                for ref in references:
                    if ref + '_forward' in signed_by_method and ref + '_reverse' in signed_by_method:
                        signed = .5 * (signed_by_method[ref + '_forward'] + signed_by_method[ref + '_reverse'])
                        vectors[key + '_DT_' + ref + '_symmetric_signed_full'] = signed
                        score('DT_' + ref + '_symmetric', signed[positions])
                method_identity()
                model.set_attn_implementation('eager')

''' + code[end:]
start = code.index("                        if hops == 1:score('FT_K1', ft_score)")
end = code.index("                else:\n                    row['published_FT_reference']", start)
code = code[:start] + "                        score(f'FT_K{hops}', ft_score.numpy())\n" + code[end:]
start = code.index("                print(json.dumps({'case':key,'DT_RISE'")
end = code.index('                gc.collect()', start)
code = code[:start] + "                print(json.dumps({'case': key, 'stage': args.stage, 'status': 'complete'}), flush=True)\n                del engine, ids, mask, signed, detail, signed_by_method\n" + code[end:]
replace('"""Evaluate frozen DT under source-v2 or the explicit legacy released-v1 protocol.',
    '"""Small Recall pilot derived from the frozen, verified source-v2 adapter.')
compile(code, 'evaluate_recall.py', 'exec')
(Path(__file__).resolve().parent / 'evaluate_recall.py').write_text(code, encoding='utf-8')
print('RECALL_DRIVER_CREATED')

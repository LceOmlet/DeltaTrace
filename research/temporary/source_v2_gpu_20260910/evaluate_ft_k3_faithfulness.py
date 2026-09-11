"""Evaluate saved live FT K3 scores with the unchanged native deletion evaluator."""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OFFICIAL = ROOT / 'experiments/official'


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True, help='One complete paired task shard')
    parser.add_argument('--environment', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(OFFICIAL))
    from evidence_protocol import source_span, select_source_tokens
    from summarize import summarize
    release = json.loads((OFFICIAL / 'protocol.json').read_bytes())
    parent = json.loads((args.parent / 'results.json').read_bytes())
    assert parent['status'] == 'complete' and parent['family'] == 'qwen3'
    assert parent['evaluation_protocol'] == 'source-v2' and parent['paired_reference_audit']
    assert parent['selection'] == 'paper' and len(parent['selected_counts']) == 1
    task = next(iter(parent['selected_counts']))
    assert task.startswith('vt_') or task == 'hotpotqa_long'
    assert parent['selected_counts'][task] == release['tasks'][task]['count']
    summarize(parent, release)
    assert digest(args.parent / 'vectors.npz') == parent['vectors_sha256']
    assert digest(OFFICIAL / 'evaluate.py') == parent['driver_sha256']
    assert digest(OFFICIAL / 'evidence_protocol.py') == parent['evidence_protocol_sha256']
    manifest = json.loads((ROOT / 'deltatrace/clean/sources.json').read_bytes())
    assert digest(ROOT / 'deltatrace/clean/sources.json') == parent['clean_sources_sha256']
    for model in manifest['models'].values():
        for relative, receipt in model['files'].items():
            assert digest(ROOT / relative) == receipt['sha256']
    env = json.loads(args.environment.read_bytes())['qwen3']
    author_root = Path(env['official_root'])
    for relative, expected in release['official_normalized_sources'].items():
        assert hashlib.sha256((author_root / relative).read_bytes().replace(b'\r\n', b'\n')).hexdigest() == expected
    checkpoint_receipt = Path(env['checkpoint_receipt'])
    assert digest(checkpoint_receipt) == env['checkpoint_receipt_sha256'] == parent['weight_identity']['receipt_sha256']
    checkpoint = json.loads(checkpoint_receipt.read_bytes())
    assert checkpoint['status'] == 'complete'
    for receipt in checkpoint['files']:
        path = Path(env['checkpoint']) / receipt['name']
        assert path.stat().st_size == receipt['bytes'] and digest(path) == receipt['sha256']
    cache_path = author_root / 'exp/exp2/data' / (task + '.jsonl')
    assert digest(cache_path) == release['tasks'][task]['cache_sha256']
    os.environ.update(HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    os.environ.setdefault('MACA_PATH', '/opt/maca')
    sys.path.insert(0, str(author_root))
    for overlay in env.get('dependency_overlays', []):
        sys.path.insert(0, overlay)
    import numpy as np
    import torch
    from exp.exp2 import run_exp as author
    from exp.exp2 import dataset_utils
    import ft_ifr_improve as ft
    from llm_attr_eval import LLMAttributionEvaluator
    for module in (author, dataset_utils, ft, sys.modules['llm_attr'], sys.modules['llm_attr_eval']):
        assert Path(module.__file__).resolve().is_relative_to(author_root.resolve())
    torch.set_num_threads(4)
    torch.manual_seed(73)
    torch.backends.cuda.matmul.allow_tf32 = False
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'loading', 'scope': 'FT_K3_faithfulness_supplement', 'dataset': task,
        'selected_count': len(parent['cases']), 'script_sha256': digest(Path(__file__)),
        'focus_addendum_sha256': digest(HERE / 'FOCUS.md'),
        'parent_results_sha256': digest(args.parent / 'results.json'), 'parent_vectors_sha256': parent['vectors_sha256'],
        'weight_identity': parent['weight_identity'], 'native_model_sha256': env['native_model_sha256'],
        'clean_sources_sha256': parent['clean_sources_sha256'], 'cache_sha256': digest(cache_path),
        'generation_calls': 0, 'new_attribution_calls': 0,
        'score_source': 'saved live FT K3 vector from this exact paired task',
        'K1_repeat_check': {'indices': [0], 'relative_tolerance': 1e-8, 'absolute_tolerance': 1e-8},
        'cases': [], 'costs': []}

    def save():
        temporary = args.output / 'results.partial'
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        temporary.replace(args.output / 'results.json')

    def timed(name, function):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        cost = {'name': name, 'status': 'entered'}
        report['costs'].append(cost)
        try:
            value = function()
            torch.cuda.synchronize()
            cost['status'] = 'returned'
            return value
        finally:
            cost.update(seconds=time.perf_counter() - started,
                        peak_allocated=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved())

    save()
    try:
        model, tokenizer = timed('model_load', lambda: author.load_model(env['checkpoint'], 'cuda:0'))
        model.eval().requires_grad_(False)
        model.set_attn_implementation('eager')
        assert digest(Path(inspect.getfile(type(model)))) == env['native_model_sha256']
        original_forwards = {name: module.forward for name, module in model.named_modules()}
        evaluator = LLMAttributionEvaluator(model, tokenizer)
        assert evaluator._ensure_pad_token_id() == tokenizer.eos_token_id
        examples = dataset_utils.load_cached(cache_path)
        with np.load(args.parent / 'vectors.npz', allow_pickle=False) as vectors:
            for parent_row in parent['cases']:
                evaluate_case(parent_row, examples[parent_row['index']], vectors, model, tokenizer, evaluator,
                              ft, np, torch, source_span, select_source_tokens, task, report, timed, save)
                assert all(module.forward == original_forwards[name] for name, module in model.named_modules())
        report['status'] = 'complete'
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
        raise
    finally:
        save()


def evaluate_case(parent_row, example, vectors, model, tokenizer, evaluator, ft, np, torch,
                  source_span, select_source_tokens, task, report, timed, save):
    index = parent_row['index']
    key = f'{task}_{index}'
    engine = ft.LLMIFRAttributionBoth(model, tokenizer, show_progress=False)
    ids, _, prompt_length, _ = engine._ensure_generation(example.prompt, example.target)
    positions = list(engine.user_prompt_indices)
    assert ids[0].cpu().tolist() == parent_row['input_ids']
    assert positions == parent_row['user_positions'] and prompt_length == parent_row['prompt_length']
    author_keep = ft.keep_token_indices(engine.user_prompt_tokens)
    assert author_keep == parent_row['author_keep']
    encoding = tokenizer(' ' + example.prompt, add_special_tokens=False, return_offsets_mapping=True)
    span = source_span(task, example.prompt)
    assert span == parent_row['source_span']
    keep = select_source_tokens(span, encoding['offset_mapping'], author_keep, parent_row['gold'])
    assert keep == parent_row['keep']
    eligible = {positions[i] for i in keep}
    inverse_positions = {position: i for i, position in enumerate(positions)}
    original_ids = ids.cpu()
    row = {'index': index, 'status': 'entered', 'input_sha256': parent_row['input_sha256'],
           'reference_input_sha256': parent_row['reference_input_sha256'], 'metrics': {}}
    report['cases'].append(row)
    report['status'] = 'evaluate_' + key
    save()

    def score(method, vector_method):
        values = np.maximum(vectors[key + '_' + vector_method + '_prompt'], 0).astype(np.float32)
        assert values.shape == (len(positions),) and np.isfinite(values).all()
        weights = torch.from_numpy(values)
        curve = {'actual_input_hashes': [], 'deleted_user_indices': []}

        def before_score(_module, _call_args, kwargs):
            actual = kwargs['input_ids'].detach().cpu()
            assert actual.shape == original_ids.shape
            changed = (actual[0] != original_ids[0]).nonzero().flatten().tolist()
            assert set(changed) <= eligible
            assert all(int(actual[0, j]) == tokenizer.eos_token_id for j in changed)
            assert torch.equal(actual[:, prompt_length:], original_ids[:, prompt_length:])
            curve['actual_input_hashes'].append(hashlib.sha256(actual.numpy().tobytes()).hexdigest())
            curve['deleted_user_indices'].append([inverse_positions[j] for j in changed])

        def observe(frame, event, result):
            if frame.f_code is ft.faithfulness_test_skip_tokens.__code__ and event == 'return' and result is not None:
                for field in ('scores', 'density', 'normalized_model_response', 'alignment_penalty', 'corrected_scores'):
                    curve[field] = np.asarray(frame.f_locals[field]).tolist()

        handle = model.register_forward_pre_hook(before_score, with_kwargs=True)
        assert sys.getprofile() is None
        sys.setprofile(observe)
        try:
            with torch.no_grad():
                result = timed(key + '_' + method, lambda: ft.faithfulness_test_skip_tokens(
                    evaluator, weights[None], example.prompt, example.target, keep_prompt_token_indices=keep,
                    user_prompt_indices=positions, k=20))
        finally:
            sys.setprofile(None)
            handle.remove()
        assert len(curve['actual_input_hashes']) == min(20, len(keep)) + 1
        assert curve['actual_input_hashes'][0] == row['input_sha256']
        assert curve['actual_input_hashes'][-1] == row['reference_input_sha256']
        curve['rise'], curve['mas'], curve['rise_plus_ap'] = map(float, result)
        curve['reference_matches_final_deletion'] = True
        return curve

    if index == 0:
        repeated = row['metrics']['FT_K1_repeat'] = score('FT_K1_repeat', 'FT_K1')
        previous = parent_row['metrics']['FT_K1']
        assert repeated['actual_input_hashes'] == previous['actual_input_hashes']
        assert repeated['deleted_user_indices'] == previous['deleted_user_indices']
        differences = {}
        for field in ('scores', 'density', 'normalized_model_response', 'alignment_penalty', 'corrected_scores',
                      'rise', 'mas', 'rise_plus_ap'):
            new, old = np.asarray(repeated[field]), np.asarray(previous[field])
            differences[field] = float(np.max(np.abs(new - old)))
            assert np.allclose(new, old, rtol=1e-8, atol=1e-8), ('K1 repeat mismatch', field, differences[field])
        row['K1_repeat_max_abs'] = differences
        report['K1_repeat_check']['passed'] = True
    row['metrics']['FT_K3'] = score('FT_K3', 'FT_K3')
    row['status'] = 'complete'
    save()
    print(json.dumps({'case': key, 'FT_K3_RISE': row['metrics']['FT_K3']['rise'],
                      'FT_K3_MAS': row['metrics']['FT_K3']['mas']}), flush=True)


if __name__ == '__main__':
    main()

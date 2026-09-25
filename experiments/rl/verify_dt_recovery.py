"""Matched recovery regression for official DT and its RL execution controls.

Reuse frozen Qwen3.5 input IDs, targets, gold and existing scoring functions.
The extra source-reference VT cases exercise actual cached-prefix continuation;
they are reported separately from the unchanged paper-reference protocol.
This is a recovery test, not an RL reward/advantage implementation.
"""
import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmark-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cases-per-task', type=int, default=8)
    p.add_argument('--candidate-fla-fp16', action='store_true',
                   help='Compare the existing BF16 official default to the isolated native-FLA FP16 RL candidate')
    args = p.parse_args()
    if args.candidate_fla_fp16:
        from accelerated.qwen35.native_fla_precision import native_fla_fp16
    benchmark = args.benchmark_root
    sys.path.insert(0, str(benchmark/'repo_dynamic/experiments/qwen35_comparison'))
    from paper_recovery_common import score_recovery, load_protocol, ids_sha, save_json
    from evidence_protocol import reference_token_ids
    from recovery_diagnostics import recovery_diagnostics
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    from profiles.official import make_qwen35_runner
    from accelerated.qwen35 import qwen35_code_local_capture
    from finite_fla_gpu import make_compiled_finite_pullback, verify_native_sources
    from qwen35_answer_finite import PackedAnswerTargets
    from vendor_fa_finite_bf16_d256 import VendorFAFiniteP1BF16D256
    import qwen35_dense_finite_runner as dense_owner
    import qwen35_gdn_finite as gdn_owner

    # A staged source draft once shadowed the candidate through the script
    # directory. Record the actual imported owner, not only the intended root.
    import marshal
    imported_capture = dense_owner.NativeGDNCapture.event
    capture_provenance = dict(
        module_file=gdn_owner.__file__,
        event_code_file=imported_capture.__code__.co_filename,
        event_code_sha256=hashlib.sha256(marshal.dumps(imported_capture.__code__)).hexdigest())
    print(json.dumps(dict(phase='capture_owner', **capture_provenance)), flush=True)

    env_path = Path(os.environ['DT_ENVIRONMENT_JSON'])
    env = json.loads(env_path.read_text())['qwen35']
    verify_native_sources(env['native_stage_source_sha256'])
    protocol = load_protocol()
    prepared_path = benchmark/'receipts/paper_recovery_inputs.json'
    prepared = json.loads(prepared_path.read_text())
    tokenizer = AutoTokenizer.from_pretrained(env['checkpoint'], local_files_only=True)
    cases = []
    for row in prepared['cases']:
        if row['index'] >= args.cases_per_task:
            continue
        assert ids_sha(row['input_ids']) == row['input_sha256']
        assert ids_sha(row['reference_ids']) == row['reference_sha256']
        cases.append(dict(row, protocol='paper_reference', scoring='paper_recovery',
                          fraction=protocol['tasks'][row['dataset']]['fraction']))
        if row['dataset'].startswith('vt_'):
            # The existing body-eligible token set and reference owner, with
            # unchanged gold/target/budget. This reference has a long solved
            # demonstration before the first changed token.
            reference = reference_token_ids(row['input_ids'],
                [row['user_positions'][j] for j in row['keep']], tokenizer.eos_token_id)
            cases.append(dict(row, protocol='source_reference_cache_check', scoring='paper_recovery',
                              reference_ids=reference, reference_sha256=ids_sha(reference),
                              fraction=protocol['tasks'][row['dataset']]['fraction']))
    needle_tasks = ['niah_mq_q2', 'niah_mq_q4', 'niah_mq_q8',
                    'niah_mv_v2', 'niah_mv_v4', 'niah_mv_v8']
    for task in needle_tasks:
        report = json.loads((benchmark/'full_dynamic_with_ifr'/task/'results.json').read_text())
        for row in report['cases'][:args.cases_per_task]:
            assert ids_sha(row['input_ids']) == row['input_sha256']
            reference = reference_token_ids(row['input_ids'],
                [row['user_positions'][j] for j in row['keep']], tokenizer.eos_token_id)
            cases.append(dict(row, protocol='paper_reference', scoring='released_token_recovery',
                fraction=.1, reference_ids=reference, reference_sha256=ids_sha(reference),
                target_offsets=list(range(row['target_length']))))
    assert cases and all(len(row['input_ids']) <= 32768 for row in cases)
    args.output.mkdir(parents=True, exist_ok=False)
    result = dict(scope=__doc__, status='loading', cases_per_task=args.cases_per_task,
                  cases=[], context_cap=32768, cases_planned=len(cases),
                  environment_sha256=hashlib.sha256(env_path.read_bytes()).hexdigest(),
                  prepared_sha256=hashlib.sha256(prepared_path.read_bytes()).hexdigest(),
                  driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  profile='gdn-symmetric-v1', generation_calls=0)
    result['capture_provenance'] = capture_provenance
    result['candidate_fla_compute_dtype'] = 'float16' if args.candidate_fla_fp16 else 'bfloat16'
    save_json(args.output/'results.json', result)
    torch.set_num_threads(4)
    torch.manual_seed(2026)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(env['checkpoint'],
        dtype=torch.bfloat16, attn_implementation='flash_attention_2',
        device_map={'': 'cuda:0'}, local_files_only=True).eval().requires_grad_(False)
    finite = VendorFAFiniteP1BF16D256(env['finite_library'], env['finite_library_sha256'])
    execution = dict(dynamic_shapes=env['dt_dynamic_shapes'], compiler_options=env['dt_compiler_options'])
    runners = {}
    for name in ('official_default', 'rl_accelerated'):
        options = dict(answer_compiled=env['dt_answer_compiled'], **execution)
        if name == 'rl_accelerated':
            options.update(copy_replay_captures=False, capture_backend=qwen35_code_local_capture,
                defer_diagnostics=True, **{key: env['dt_'+key] for key in (
                    'offload_replay_mixer', 'gdn_head_batch_size', 'compile_gdn_scalar_rules',
                    'pin_replay_host', 'gdn_gpu_capture_names', 'fa_coefficient_suffix',
                    'gdn_coefficient_suffix', 'compact_gdn_captures', 'pin_root_host', 'reuse_native_prefix')})
        runners[name] = make_qwen35_runner(model, finite,
            make_compiled_finite_pullback(reuse_scalar_products=False, **execution), **options)
    vectors = {}
    # Keep each protocol/task's four-sample groups intact. No synthetic context.
    groups = sorted({(r['protocol'], r['dataset']) for r in cases})
    for group in groups:
        rows = sorted([r for r in cases if (r['protocol'], r['dataset']) == group], key=lambda r:r['index'])
        for start in range(0, len(rows), 4):
            batch = rows[start:start+4]
            length = max(len(r['input_ids']) for r in batch)
            pair = torch.full((len(batch)*2, length), tokenizer.eos_token_id, dtype=torch.long, device='cuda')
            for index, row in enumerate(batch):
                for endpoint, field in enumerate(('reference_ids', 'input_ids')):
                    pair[2*index+endpoint, :len(row[field])] = torch.tensor(row[field], device='cuda')
            selection = PackedAnswerTargets([
                dict(target_ids=torch.tensor(r['input_ids'][r['prompt_length']:]), prompt_length=r['prompt_length'])
                for r in batch], [r['target_offsets'] for r in batch], length, model.device)
            output, details = {}, {}
            for name, runner in runners.items():
                print(json.dumps(dict(phase='attribution_start', method=name, group=group,
                                      indices=[r['index'] for r in batch], length=length)), flush=True)
                boundary = (native_fla_fp16(model) if args.candidate_fla_fp16 and name == 'rl_accelerated'
                            else nullcontext())
                with boundary:
                    signed, detail = runner.attribute(pair, torch.ones_like(pair), selection,
                                                      select_output_rows=True, observer=None)
                assert bool(torch.isfinite(signed).all())
                output[name] = signed.cpu().numpy()
                details[name] = dict(seconds=detail['complete_attribution_seconds_with_diagnostics'],
                                    cached_prefix_tokens=detail.get('native_shared_prefix_length', 0))
            for index, row in enumerate(batch):
                key = f"{row['protocol']}/{row['dataset']}/{row['index']}"
                entry = dict(protocol=row['protocol'], dataset=row['dataset'], index=row['index'],
                    input_sha256=row['input_sha256'], reference_sha256=row['reference_sha256'],
                    target_offsets=row['target_offsets'], length=len(row['input_ids']),
                    batch_costs=details, metrics={})
                for name, signed in output.items():
                    vector = signed[index, :len(row['input_ids'])]
                    vectors[key+'/'+name] = vector
                    values = vector[row['user_positions']]
                    if row['scoring'] == 'paper_recovery':
                        metric = score_recovery(row, values, row['fraction'])
                    else:
                        metric = recovery_diagnostics(np.maximum(values.astype(np.float32), 0),
                            row['keep'], row['gold'], row['fraction'])
                        metric['fraction'] = row['fraction']
                    entry['metrics'][name] = metric
                old, new = (output[n][index, :len(row['input_ids'])] for n in runners)
                delta = new.astype(np.float64)-old.astype(np.float64)
                entry['vector_difference'] = dict(max_abs=float(np.max(np.abs(delta))),
                    relative_l2=float(np.linalg.norm(delta)/max(np.linalg.norm(old), 1e-30)))
                result['cases'].append(entry)
            result['status'] = 'running'
            save_json(args.output/'results.json', result)
            np.savez_compressed(args.output/'vectors.npz', **vectors)
            print(json.dumps(dict(phase='batch_complete', completed=len(result['cases']), costs=details)), flush=True)
    result['status'] = 'complete'
    result['vectors_sha256'] = hashlib.sha256((args.output/'vectors.npz').read_bytes()).hexdigest()
    save_json(args.output/'results.json', result)


if __name__ == '__main__':
    main()

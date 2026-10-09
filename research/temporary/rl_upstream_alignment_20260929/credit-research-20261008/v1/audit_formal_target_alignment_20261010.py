"""Read saved formal target/source alignment through the existing packing owner.

This is a CPU data/interface audit, not a model, DT, numerical-tolerance or
single-deletion accuracy test. It does not alter the formal training process.
"""
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time

import torch


ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
FORMAL = ROOT / 'runs/textcraft-formal-stable-20261009-v1'
SOURCE_SHA = '1c08b57bf83506d3e69d865f74377e3baa624358c678e0ac92cb6ebfb2d73658'
PACKING_SHA = 'd47333ea68fb7a332e7d1dce7913c989d875ea262dfe49cfa4f20f7c35ebe03e'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    started = time.perf_counter()
    source = FORMAL / 'source.json'
    assert digest(source) == SOURCE_SHA
    manifest = json.loads(source.read_bytes())
    clean = Path(manifest['dt_root']) / 'clean/qwen35'
    sys.path.insert(0, str(clean))
    packing = importlib.import_module('qwen35_answer_finite')
    assert digest(Path(packing.__file__)) == PACKING_SHA
    cohorts = {}
    for ordinal in ['second', 'third', 'fourth']:
        audit_path = ROOT / 'receipts/direct-credit-records-20261009-v1' / (
            f'complete-{ordinal}-DT-record-audit.json')
        audit = json.loads(audit_path.read_bytes())
        paths = sorted({item['path'] for item in audit['rows']})
        rows = []
        for filename in paths:
            path = Path(filename)
            stored = torch.load(path, map_location='cpu', weights_only=True)
            batch = stored['rows']
            cases = [dict(prompt_length=row['prompt_length'],
                          target_ids=row['input_ids'][row['prompt_length']:])
                     for row in batch]
            selected = packing.PackedAnswerTargets(
                cases, [row['target_offsets'] for row in batch],
                max(row['input_ids'].numel() for row in batch), 'cpu')
            for index, row in enumerate(batch):
                ids = row['input_ids']
                slots = row['suffix_positions']
                policy, target, prior, d = [row[name] for name in (
                    'policy_mask', 'target_mask', 'prior_source_mask',
                    'source_log_ratios')]
                offsets = torch.tensor(row['target_offsets'], dtype=torch.long)
                labels = selected.labels[selected.samples == index]
                predictors = selected.positions[selected.samples == index]
                eos_source = prior[slots] & (
                    ids[row['prompt_length']:] == stored['eos_token_id'])
                exact = dict(
                    original_packing_labels_equal_saved_target_IDs=torch.equal(
                        labels, ids[row['prompt_length'] + offsets]),
                    original_packing_predictors_one_before_target=torch.equal(
                        predictors, row['prompt_length'] + offsets - 1),
                    target_offsets_equal_original_response_mask=torch.equal(
                        offsets, target[slots].nonzero().flatten()),
                    suffix_scatter_slots_unique_ordered=torch.equal(
                        slots, slots.unique(sorted=True)),
                    target_and_prior_disjoint=not bool((target & prior).any()),
                    prior_is_policy=not bool((prior & ~policy).any()),
                    target_is_policy=not bool((target & ~policy).any()),
                    non_source_raw_d_zero=not bool(d[~prior].any()),
                    already_EOS_source_raw_d_zero=not bool(
                        d[slots[eos_source]].any()),
                    source_vector_finite=bool(torch.isfinite(d).all()),
                    original_response_lengths_match=(
                        ids.numel() == row['prompt_length'] + slots.numel()),
                    normalized_factual_target_score=(
                        row['factual_target_logp'] is not None
                        and row['factual_target_logp'] <= 0),
                    normalized_joint_reference_target_score=(
                        row['reference_target_logp'] is not None
                        and row['reference_target_logp'] <= 0),
                )
                rows.append(dict(
                    path=filename, traj_uid=row['traj_uid'], rank_row=index,
                    input_tokens=ids.numel(), source_tokens=int(prior.sum()),
                    target_tokens=len(offsets), already_EOS_source_tokens=int(eos_source.sum()),
                    factual_target_logp=row['factual_target_logp'],
                    reference_target_logp=row['reference_target_logp'],
                    source_signed_sum=float(d.double().sum()),
                    endpoint_minus_source_sum=(
                        row['factual_target_logp'] - row['reference_target_logp']
                        - float(d.double().sum())),
                    exact_interface_checks=exact))
        checks = {key:sum(not row['exact_interface_checks'][key] for row in rows)
                  for key in rows[0]['exact_interface_checks']}
        cohorts[ordinal] = dict(
            existing_audit=dict(path=str(audit_path), sha256=digest(audit_path)),
            files=len(paths), original_rank_rows=len(rows),
            unique_traj_uids=len({row['traj_uid'] for row in rows}),
            failed_rank_rows_by_check=checks,
            all_exact_interface_checks_passed=not any(checks.values()),
            maximum_abs_endpoint_minus_source_sum=max(
                abs(row['endpoint_minus_source_sum']) for row in rows),
            rows=rows)
    result = dict(
        scope=__doc__, unix=time.time(), seconds=time.perf_counter()-started,
        source=dict(path=str(source), sha256=digest(source)),
        packing_owner=dict(path=packing.__file__, resolved=str(Path(packing.__file__).resolve()),
                           sha256=digest(Path(packing.__file__))),
        script_sha256=digest(Path(__file__)), cohorts=cohorts,
        cuda_initialized=torch.cuda.is_initialized(),
        model_calls=0, DT_calls=0, optimizer_calls=0, production_changes=0,
        limits=[
            'Endpoint/source residuals are reported without a new pass threshold or correction.',
            'Per-target-token endpoint log-probs and raw pre-scatter signed vectors are not saved; their numerical values cannot be independently recomputed here.',
            'Full joint reference scores do not supply factual single-deletion probabilities.',
            'This audit does not replace original author deletion curves or the frozen probability-sample single-deletion comparison.',
        ])
    assert not result['cuda_initialized']
    output = Path(sys.argv[1])
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key:value for key,value in result.items() if key != 'cohorts'}))
    print(json.dumps({name:{key:value for key,value in data.items() if key != 'rows'}
                      for name,data in cohorts.items()}))


if __name__ == '__main__':
    main()

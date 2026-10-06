"""Read existing Adam JSONs only; repeat scalar arithmetic without Torch."""
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).parent
BRANCHES = ('dt', 'grpo', 'regularizers_only')


def identity(path):
    raw = path.read_bytes()
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))


def main():
    path = ROOT / 'native-adam-analysis.json'
    analysis = json.loads(path.read_bytes())
    sources = dict(analysis=identity(path), script=identity(Path(__file__)))
    raw_updates, counters = {}, []
    for rank in range(2):
        raw_updates[rank] = {}
        for branch in BRANCHES:
            name = f'rank{rank}-{branch}-native-adam.json'
            source = ROOT / name
            record = json.loads(source.read_bytes())
            sources[name] = identity(source)
            raw_updates[rank][branch] = record
            counters.append(dict(rank=rank, branch=branch, status=record['status'],
                actual_optimizer_calls=record['optimizer_step_calls'],
                actual_scheduler_calls=record['scheduler_step_calls'],
                Adam_before=record['optimizer_before']['step_counters'],
                Adam_after=record['optimizer_after']['step_counters'],
                scheduler_epoch_before=record['scheduler_before']['last_epoch'],
                scheduler_epoch_after=record['scheduler_after']['last_epoch'],
                scheduler_counter_before=record['scheduler_before']['_step_count'],
                scheduler_counter_after=record['scheduler_after']['_step_count'],
                raw_SHA_equals_analysis_input=sources[name]['sha256'] == analysis['inputs'][name]['sha256'],
                raw_counter_fields_equal_analysis=all(record[key] == analysis['actual_native_updates'][str(rank)][branch][key]
                    for key in ('optimizer_step_calls', 'scheduler_step_calls', 'optimizer_before',
                                'optimizer_after', 'scheduler_before', 'scheduler_after'))))
    geometry = []
    for item in analysis['local_parameter_geometry']:
        gram = item['parameter_delta_Gram']
        norms = {branch: math.sqrt(gram[f'{branch}:{branch}']) for branch in BRANCHES}
        cos = {f'{a}:{b}': gram[f'{a}:{b}'] / (norms[a] * norms[b])
               for a, b in (('dt', 'grpo'), ('dt', 'regularizers_only'), ('grpo', 'regularizers_only'))}
        comparisons = {}
        for branch in BRANCHES:
            records = item['parameters']
            comparisons[branch] = dict(parameters=len(records),
                before_native_storage_hash_equal_to_DT=sum(
                    row['branches'][branch]['before'] == row['branches']['dt']['before'] for row in records),
                delta_nonfinite_elements=item['branches'][branch]['delta_nonfinite_elements'],
                gradient_nonfinite_elements=item['branches'][branch]['gradient_nonfinite_elements'])
        difference = item['pairwise_parameter_delta_difference_norms']['dt:regularizers_only']
        geometry.append(dict(rank=item['rank'], native_local_storage_scope=True,
            repeated_norms=norms, repeated_cosines=cos,
            repeated_cosine_minus_recorded={key: value - item['parameter_delta_cosines'][key] for key, value in cos.items()},
            DT_minus_reg_local_difference_norm=difference,
            DT_minus_reg_difference_over_DT_norm=difference/norms['dt'], before=comparisons,
            arithmetic_source='Original saved JSON Gram scalars; original .pt values were analyzed by the bound native CPU analyzer, not reloaded here.'))
    observations = {}
    for branch in BRANCHES:
        ranks = analysis['H_LP_observations'][branch]['rank_reductions']
        result = []
        for rank in ranks:
            rows = rank['B4']
            means = {key: math.fsum(row[key] for row in rows) / 8 for key in (
                'H_before', 'H_after', 'H_delta', 'LP_delta', 'LP_delta_abs_mean',
                'before_LP_minus_saved_original_abs_mean')}
            result.append(dict(rank=rank['rank'], B4_rows=len(rows),
                original_policy_denominators=[row['policy_mask_denominator'] for row in rows],
                native_equal_B4_mean=means,
                repeated_minus_recorded={key: value-rank['native_equal_B4_sum_divided_by_8'][key] for key, value in means.items()}))
        means = {key: math.fsum(row['native_equal_B4_mean'][key] for row in result)/2
                 for key in result[0]['native_equal_B4_mean']}
        observations[branch] = dict(per_rank=result, equal_rank_mean=means,
            repeated_minus_recorded={key: value-analysis['H_LP_observations'][branch]['two_rank_equal_mean_of_native_reductions'][key]
                                      for key, value in means.items()})
    output = dict(scope='Independent stdlib review of one saved global64 native Adam update per branch, same checkpoint25; no new model/Torch/forward/update.',
        sources=sources, actual_updates=counters, local_parameter_geometry=geometry,
        native_B4_observations=observations,
        before_readout_native_tensor_hashes={branch: {
            field: analysis['initial_readout_comparisons'][branch][field]['tensor']['logical_contiguous_native_dtype_bytes_sha256']
            for field in ('entropys', 'old_log_probs')} for branch in BRANCHES},
        limits=[
            'Local parameter norms/cosines are per-rank coordinates; no global theta norm or cross-rank gather is inferred.',
            'Regularizers-only zeroes current advantages but retains common Adam history and weight decay; its update is not a pure entropy/KL contribution.',
            'Differences between Adam branches are paired observed differences, not a linear component attribution or historical degradation proof.',
            'Fresh before log-probs differ from the saved old-log-probs (native B4 mean absolute difference about 0.012436); old/ref were retained, so ratios are not asserted to be exactly one.',
            'Three fresh before readout tensors and native before parameter hashes are equal across branches; this supports the matched comparison, not identical arithmetic with the earlier pre-Adam gradient probe.',
            'H_delta is the original loss-mask/token-mean B4 reduction of per-token after-minus-before, then original B4 sum/8 and equal-rank mean; it is not a pooled token mean or a new rollout metric.',
            'No numerical tolerance gate, production edit, single-token replacement for cumulative-deletion/RISE/MAS, or additional computation on model tensors.'],
        operations=dict(model_calls=0, DT_calls=0, backward=0, optimizer_updates=0, Torch_imports=0, production_edits=0))
    destination = ROOT / 'independent-native-adam-review.json'
    with destination.open('x', encoding='utf8') as stream:
        json.dump(output, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(identity(destination)))


if __name__ == '__main__':
    main()

"""Describe four real native endpoints without altering the training estimator."""
import hashlib
import json
import math
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
AUDIT = HERE.parents[1]
ENDPOINT = AUDIT / 'direct-target-extreme-token-endpoint-20261007/v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize_pair(path, row, slot):
    data = torch.load(path, map_location='cpu', weights_only=False)
    mask = data['samples'].eq(row)
    future = mask & data['predictor_positions'].ge(slot)
    earlier = mask & ~future
    effect = data['factual_target_logp'].double() - data['reference_target_logp'].double()
    other = ~mask
    assert torch.count_nonzero(effect[other]) == 0
    assert torch.count_nonzero(effect[earlier]) == 0
    top = []
    for index in future.nonzero().flatten():
        top.append(dict(predictor=int(data['predictor_positions'][index]),
            target_id=int(data['labels'][index]),
            kept_logp=float(data['factual_target_logp'][index]),
            removed_logp=float(data['reference_target_logp'][index]),
            kept_minus_removed=float(effect[index])))
    return dict(future_tokens=int(future.sum()), earlier_tokens=int(earlier.sum()),
        earlier_maxabs=float(effect[earlier].abs().max()),
        other_rows_maxabs=float(effect[other].abs().max()),
        signed_future_effect=float(effect[future].sum()),
        top=sorted(top, key=lambda item: abs(item['kept_minus_removed']), reverse=True)[:15])


def main():
    dest = HERE / 'actual-results'
    transport = json.loads((dest / 'transport.json').read_bytes())
    for name, record in transport['files'].items():
        assert sha(dest / name) == record['sha256']
    factual = json.loads((ENDPOINT / 'appworld/rank0.json').read_bytes())
    old = json.loads((ENDPOINT / 'endpoint-analysis.json').read_bytes())['cases'][1]
    ranks = [json.loads((dest / f'results/rank{rank}.json').read_bytes()) for rank in (0, 1)]
    pairs = [summarize_pair(dest / f'results/rank{rank}-target-logp.pt', 0, 7260) for rank in (0, 1)]
    assert pairs[0] == pairs[1]
    assert old['candidate'] == ranks[0]['geometry']['candidate']
    for record, pair in zip(ranks, pairs):
        assert record['phase'] == 'complete'
        assert record['native_sha256'] == factual['prepared']['native_sha256']
        assert record['source_sha256'] == factual['prepared']['source_sha256']
        assert record['restored_minus_reference'][1:] == [0.0] * 3
        assert not any(shard['nonzero'] for shard in record['lora_B_local_shards'])
        assert pair['signed_future_effect'] == record['restored_minus_reference'][0]
        assert record['operations'] == dict(native_paired_forward=1, DT=0,
            backward=0, optimizer=0, checkpoint_restore=0, rollout=0)
    null_effect = pairs[0]['signed_future_effect']
    result = dict(candidate=old['candidate'],
        endpoints=dict(
            F_factual_all_sources=old['factual_joint_lp'],
            D_factual_delete_only_candidate=old['single_eos_joint_lp'],
            B_all_prior_sources_EOS=ranks[0]['reference_joint_logp'][0],
            C_only_candidate_restored=ranks[0]['token_restored_joint_logp'][0]),
        factual_context=dict(d=old['native_d'],
            deleted_to_kept_probability_ratio=old['native_deleted_to_factual_ratio'],
            reward1_sample_coefficient=old['same_reward1_native_coefficient'],
            top=old['top_target_effects']),
        joint_EOS_context=dict(d=null_effect,
            deleted_to_kept_probability_ratio=math.exp(-null_effect),
            reward1_sample_coefficient=-math.expm1(-null_effect), **pairs[0]),
        difference_between_context_marginals=old['native_d'] - null_effect,
        original_joint_DT=dict(d=old['saved_DT_d'],
            reward1_sample_coefficient=old['saved_DT_reward1_coefficient'],
            note='Original joint finite allocation; not a measurement of either marginal above.'),
        controls=dict(ranks_equal=True, earlier_target_maxabs=0.0,
            other_three_row_maxabs=0.0, lora_B_shards_all_zero=True),
        numerical_scope=dict(native_logits='torch.bfloat16', target_logp='torch.float32',
            accumulation='torch.float64',
            factual_drift_from_original_cached_root=old['factual_drift_from_original_saved'],
            note='Two native paired forwards with unchanged actor/source owners. Context effects are descriptive, not an FA/FLA tolerance test. Separate cached-root drift is retained.'),
        interpretation='The sign reverses when the other prior sources are jointly replaced by EOS. A negative joint-reference allocation therefore does not establish a negative factual single-token deletion effect. This confirms a substantial context interaction, but does not assign all joint finite-allocation error to one cause or diagnose DT quality overall.',
        scope='One actual extreme source token, four native endpoint scores. No new reference policy, production queries, reward target, credit formula, clipping, whitening or PPO change. Does not replace cumulative deletion, RISE or MAS.',
        provenance=dict(source_sha256=ranks[0]['source_sha256'],
            native_sha256=ranks[0]['native_sha256'],
            old_endpoint_analysis_sha256=sha(ENDPOINT / 'endpoint-analysis.json'),
            transport_sha256=sha(dest / 'transport.json'),
            launch=json.loads((dest / 'launch.json').read_bytes()),
            owners=ranks[0]['owners']),
        state='TextCraft first update held; AppWorld formal job terminal. Completed isolated native diagnosis on GPU4/5, zero optimizer steps. No formal restart.')
    (HERE / 'analysis.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    assert not torch.cuda.is_initialized()
    print(json.dumps({key: result[key] for key in ('endpoints', 'difference_between_context_marginals', 'controls')}, ensure_ascii=False))


if __name__ == '__main__':
    main()

"""Re-evaluate original PPO losses on already saved native before/after LP/H.

CPU only. No model, new prediction, backward, update, reward or credit is made.
The original core owns PG clipping, KL and reductions. Both saved DT and GRPO
signals are evaluated on each actual matched Adam branch. This measures frozen
batch objective changes, not new success rates or linear attribution of Adam.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import time

from analyze_textcraft_native_adam import identity, source, resources, BRANCHES, PROTO_SHA, CORE_SHA


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--minibatch-path', type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    import torch
    from verl.protocol import DataProto
    from verl.trainer.ppo.core_algos import compute_policy_loss, agg_loss, kl_penalty
    torch.set_num_threads(1)
    assert not torch.cuda.is_initialized() and not torch.distributed.is_initialized()
    started = time.time()
    output = args.input_dir / 'native-objective-effect.json'
    assert not output.exists(), 'Preserve the first complete raw CPU observation'
    sources = {key: source(fn) for key, fn in dict(protocol_load=DataProto.load_from_disk,
        protocol_chunk=DataProto.chunk, policy_loss=compute_policy_loss, aggregate=agg_loss, kl=kl_penalty).items()}
    assert sources['protocol_load']['sha256'] == sources['protocol_chunk']['sha256'] == PROTO_SHA
    assert all(sources[key]['sha256'] == CORE_SHA for key in ('policy_loss','aggregate','kl'))
    sources.update(analyzer=identity(__file__), analysis_helper=identity(Path(__file__).with_name('analyze_textcraft_native_adam.py')))
    complete_path = args.input_dir / 'completed.json'
    inspected_path = args.input_dir / 'native-owner-inspection.json'
    complete = json.loads(complete_path.read_bytes())
    inspected = json.loads(inspected_path.read_bytes())
    inputs = dict(minibatch=identity(args.minibatch_path), completed=identity(complete_path), inspection=identity(inspected_path))
    assert inputs['minibatch']['sha256'] == complete['source_minibatch_sha256']
    config = inspected['actor_config']['actor']
    assert config['ppo_mini_batch_size'] == 64 and config['ppo_micro_batch_size_per_gpu'] == 4
    assert config['policy_loss']['loss_mode'] == 'vanilla'
    assert config['entropy_coeff'] == config['kl_loss_coef'] == .001 and config['use_kl_loss']
    minibatch = DataProto.load_from_disk(str(args.minibatch_path))
    assert len(minibatch) == 64 and all(t.device.type == 'cpu' for t in minibatch.batch.values())
    original_ranks = minibatch.chunk(2)
    branches, saved_branches = {}, {x['branch']:x for x in complete['branches']}
    before_resources = resources(torch)
    with torch.no_grad():
        for branch in BRANCHES:
            observations = {}
            for when in ('before','after'):
                path = args.input_dir / f'{branch}-{when}-logprob.pkl'
                inputs[path.name] = identity(path)
                assert inputs[path.name]['sha256'] == saved_branches[branch][when]['sha256']
                observations[when] = DataProto.load_from_disk(str(path)).chunk(2)
            ranks = []
            for rank, original in enumerate(original_ranks):
                assert len(original) == 32
                rows = []
                for index, batch in enumerate(original.batch.split(4)):
                    width = batch['responses'].shape[-1]
                    mask = batch['loss_mask'][:, -width:]
                    row = dict(rank=rank, microbatch_index=index, policy_mask_denominator=float(mask.sum()), outcomes={})
                    for when in ('before','after'):
                        observed = observations[when][rank].batch[index*4:(index+1)*4]
                        lp, H = observed['old_log_probs'], observed['entropys']
                        assert lp.shape == H.shape == mask.shape
                        values = dict(native_log_prob_dtype=str(lp.dtype), native_entropy_dtype=str(H.dtype))
                        H_loss = agg_loss(loss_mat=H, loss_mask=mask, loss_agg_mode=config['loss_agg_mode'])
                        KL_loss = agg_loss(loss_mat=kl_penalty(logprob=lp, ref_logprob=batch['ref_log_prob'],
                            kl_penalty=config['kl_loss_type']), loss_mask=mask, loss_agg_mode=config['loss_agg_mode'])
                        values.update(weighted_entropy=float(-config['entropy_coeff']*H_loss),
                            weighted_kl=float(config['kl_loss_coef']*KL_loss))
                        for signal, field in (('dt','advantages'), ('grpo','diagnostic_grpo_advantages')):
                            loss, clip, ppo_kl, lower = compute_policy_loss(old_log_prob=batch['old_log_probs'], log_prob=lp,
                                advantages=batch[field], response_mask=mask, cliprange=config['clip_ratio'],
                                cliprange_low=config['clip_ratio_low'], cliprange_high=config['clip_ratio_high'],
                                clip_ratio_c=config['clip_ratio_c'], loss_agg_mode=config['loss_agg_mode'])
                            values[signal+'_pg'] = float(loss)
                            values[signal+'_pg_clipfrac'] = float(clip)
                            values[signal+'_pg_clipfrac_lower'] = float(lower)
                            values[signal+'_ppo_kl'] = float(ppo_kl)
                            values[signal+'_total'] = float(loss - config['entropy_coeff']*H_loss + config['kl_loss_coef']*KL_loss)
                        row['outcomes'][when] = values
                    row['changes_after_minus_before'] = {key:row['outcomes']['after'][key]-value
                        for key,value in row['outcomes']['before'].items() if isinstance(value, float)}
                    rows.append(row)
                assert len(rows) == 8
                means = {when:{key:sum(row['outcomes'][when][key] for row in rows)/8
                    for key,value in rows[0]['outcomes'][when].items() if isinstance(value,float)} for when in ('before','after')}
                ranks.append(dict(rank=rank, microbatches=rows, means=means,
                    changes_after_minus_before={key:means['after'][key]-value for key,value in means['before'].items()}))
            means = {when:{key:sum(rank['means'][when][key] for rank in ranks)/2 for key in ranks[0]['means'][when]} for when in ('before','after')}
            branches[branch] = dict(ranks=ranks, mean_of_two_native_rank_reductions=means,
                changes_after_minus_before={key:means['after'][key]-value for key,value in means['before'].items()})
            del observations
            gc.collect()
    result = dict(scope=__doc__, sources=sources, inputs=inputs, effective_original_actor_config=config,
        original_partition='DataProto.chunk(2): local32 -> 8 original B4 token-means -> /8; then mean of two ranks.',
        old_policy_source='Unchanged original saved old_log_probs and ref_log_prob, not replaced by the measured before arrays.',
        branches=branches, operations=dict(model_initializations=0, model_forwards=0, backward=0, optimizer=0, DT=0),
        interpretation=['Negative after-minus-before loss means improved frozen original objective, not improved task success.',
            'All three actual updates retain restored common historical Adam moments. The zero-current-PG branch is not pure entropy.',
            'This re-evaluates stored LP/H with the original CPU core, without the GPU forward/autocast. It is descriptive, not an official numerical tolerance claim.',
            'DT and GRPO PG loss changes have different units; compare each frozen signal across matched actual branches, not raw magnitudes as an error ratio.'],
        runtime=dict(started_unix=started, completed_unix=time.time(), before=before_resources, after=resources(torch)))
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output=identity(output), changes={branch:value['changes_after_minus_before'] for branch,value in branches.items()}, runtime=result['runtime'])))

"""Freeze a passive suboperation diagnostic from the measured whole collection.

This is an observation protocol, not an RL method or a candidate credit rule.
No model/DT call, new source selection, or training setting is introduced.
"""
import json
import hashlib
from pathlib import Path

HERE=Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_bytes())


def ref(path):
    raw=path.read_bytes()
    return dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))


def main():
    cfg_path=HERE/'gdn-owner-readonly.json'
    cfg=read(cfg_path)['model_config']['text_config']
    plan_path=HERE/'layer-collection-inputs.json'
    plan=read(plan_path)
    families=('linear_attention','full_attention')
    selection={}; union=set(); inputs=[ref(cfg_path),ref(plan_path)]
    costs={}
    for task in ('textcraft','appworld'):
        ranks=[]; points=[]
        for rank in (0,1):
            path=HERE/f'layer-factual-controls-{task}-observations/rank{rank}.json'
            inputs.append(ref(path));report=read(path)
            assert report['phase']=='complete'
            ranks.append(report)
            points.extend(p for b in report['batches'] for p in b['points'])
        assert len(points)==165
        selection[task]={}
        for cohort in ('uniform','predicted_tail_census'):
            rows=[p for p in points if any(c['cohort']==cohort for c in p['previous_comparisons'])]
            selection[task][cohort]={}
            for family in families:
                indices=[i for i,t in enumerate(cfg['layer_types']) if t==family]
                states={}
                for p in rows:
                    winner=max(indices,key=lambda i:abs(p['matched_factual_residuals'][i]))
                    states.setdefault(p['initial_state_sha256'],{}).setdefault(p['traj_uid'],[]).append(winner)
                frequencies={i:sum(sum(sum(w==i for w in winners)/len(winners)
                    for winners in trajectories.values())/len(trajectories)
                    for trajectories in states.values())/len(states) for i in indices}
                leader=max(indices,key=lambda i:frequencies[i]);union.add(leader)
                selection[task][cohort][family]=dict(points=len(rows),states=len(states),
                    frequencies=frequencies,leader=leader,
                    weighting='source frequency within trajectory, trajectory mean within initial state, equal initial-state mean; tasks/cohorts never pooled',
                    scope='Bounded localization frequency, not an error share, population rate or proof of kernel defect')
        costs[task]=dict(
            original_DT_calls=sum(r['operations']['DT'] for r in ranks),
            original_native_forward_calls_per_rank=[r['operations']['native_forward'] for r in ranks],
            original_elapsed_seconds_per_rank=[r['elapsed_seconds'] for r in ranks],
            largest_original_bank_bytes=max(b['retained_bank_bytes'] for r in ranks for b in r['batches']),
            largest_original_head_output_bytes=max(b['DT_detail']['actual_output_bytes'] for r in ranks for b in r['batches']))
    layers=sorted(union)
    assert layers==[18,30,31], 'Recomputed selection differs; do not silently change the staged protocol'
    base=sorted(set([32]+[i for layer in layers for i in (layer,layer+1)]))
    for task,cost in costs.items():
        # Existing bank has 33*(FP32 coefficient+BF16 factual state) rows.
        # Retain only the needed original boundaries and three internal
        # coefficient/state pairs per selected decoder.
        pairs=len(base)+3*len(layers)
        # Additional factual mixer/MLP outputs expose native BF16 residual-add
        # rounding, rather than silently treating these additions as exact.
        extra_factual=2*len(layers)
        decoder_upper=cost['largest_original_bank_bytes']*(pairs+extra_factual/3)/33
        # Packed targets are a subset of B4 times the original union of
        # predictor rows. FP32 seed+BF16 factual logits cost <=1.5 times
        # the original interleaved BF16 full-vocabulary output. Head hidden
        # banks add two FP32 coefficients and one BF16 factual state.
        vocab=cfg['vocab_size']; hidden=cfg['hidden_size']
        head_upper=cost['largest_original_head_output_bytes']*(1.5+2.5*hidden/vocab)
        cost.update(retained_decoder_coefficient_state_pairs=pairs,
                    additional_factual_only_states=extra_factual,
                    decoder_host_bank_upper_bytes=int(decoder_upper),
                    head_host_bank_upper_bytes=int(head_upper),
                    combined_host_bank_upper_bytes=int(decoder_upper+head_upper),
                    additional_full_model_calls=0, additional_DT_calls=0,
                    additional_head_only_reverse_projection_per_DT=1,
                    GPU_seed_readout_rows=128,
                    estimated_head_readout_temporary_upper_bytes=128*vocab*4*8,
                    estimate_scope='Storage/work bounds from actual owner shapes, not an observed memory peak or promised speed')
    result=dict(scope=__doc__,inputs=inputs,model_config=dict(hidden=cfg['hidden_size'],vocab=cfg['vocab_size']),
        selections=selection,selected_decoder_layers=layers,required_original_boundaries=base,costs=costs,
        unchanged_queries_plan_sha256=ref(plan_path)['sha256'],
        queries_per_task=165,cohorts=dict(uniform=128,predicted_tail_census=37),
        observation_design=[
            'Preserve owner consume_captures=True; do not enable the diagnostics branch that retains MLP-width tensors.',
            'Observe the original boundary callbacks and native module hooks, returning their original result objects.',
            'Decompose selected decoders into norm1/mixer/norm2/MLP contractions; retain factual endpoint controls at each boundary.',
            'Keep both native residual-add rounding terms explicitly; do not assume BF16 forward additions are exact real-number identities.',
            'Decompose output into final norm, independently recomputed same-owner seed/projection difference, linear projection and log-softmax terms.',
            'Recomputed head seed uses the original finite seed and transpose; its measured difference from the actual compiled hidden coefficient is retained, never assumed zero.',
            'Head readout slices use128 rows only to bound diagnostic temporaries; actual DT/actor B4, Q/V/A, LoRA and native calls stay unchanged.',
            'Read all fixed165 queries per task, including source-uniform and complete predicted-tail cohorts. No top-k token rescue or training correction.',
            'Run TextCraft once first; observe actual phase cost and memory before launching AppWorld. No kernel or candidate rule is changed.'
        ], production_modified=False,accepted_candidate=False,CUDA_initialized=False,
        model_calls=0,DT_calls=0,recorder=ref(Path(__file__)))
    output=HERE/'suboperation-protocol.json'
    output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(output=str(output),selected_decoder_layers=layers,
                         required_original_boundaries=base,costs=costs)))


if __name__=='__main__':
    main()

"""Read only the original 176-request metadata; report owner layout extents."""
import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import statistics


def identity(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def describe(values):
    values = sorted(values)
    def quantile(p):
        position = (len(values)-1)*p
        lower = int(position)
        return values[lower] + (values[min(lower+1, len(values)-1)]-values[lower])*(position-lower)
    return dict(count=len(values), minimum=values[0], p25=quantile(.25),
                median=statistics.median(values), p75=quantile(.75), maximum=values[-1],
                mean=statistics.mean(values), sum=sum(values))


def summarize(batches):
    names = ('old_suffix_slots', 'row_suffix_slots', 'saved_suffix_slots',
             'action_query_target_slots', 'alignment_slots', 'old_common_history_tail_slots',
             'old_right_padding_slots', 'row_right_padding_slots',
             'old_KV_carrier_slots', 'row_KV_carrier_slots', 'row_internal_prefix_hole_slots')
    totals = {name: sum(batch[name] for batch in batches) for name in names}
    totals.update(
        suffix_slot_reduction_fraction=totals['saved_suffix_slots']/totals['old_suffix_slots'],
        row_native_KV_carrier_ratio=totals['row_KV_carrier_slots']/totals['old_KV_carrier_slots'],
        row_suffix_padding_fraction=totals['row_right_padding_slots']/totals['row_suffix_slots'],
        row_64_alignment_fraction=totals['alignment_slots']/totals['row_suffix_slots'],
        paired_root_suffix_slots_old=2*totals['old_suffix_slots'],
        paired_root_suffix_slots_row=2*totals['row_suffix_slots'],
        paired_32_decoder_replay_layer_token_slots_old=2*32*totals['old_suffix_slots'],
        paired_32_decoder_replay_layer_token_slots_row=2*32*totals['row_suffix_slots'],
    )
    return dict(batches=len(batches), totals=totals,
        smaller_suffix_batches=sum(b['saved_suffix_slots']>0 for b in batches),
        unchanged_suffix_batches=sum(b['saved_suffix_slots']==0 for b in batches),
        suffix_reduction_fraction_distribution=describe([b['suffix_slot_reduction_fraction'] for b in batches]),
        saved_suffix_slots_distribution=describe([b['saved_suffix_slots'] for b in batches]),
        row_true_suffix_length_distribution=describe([n for b in batches for n in b['row_true_suffix_lengths']]),
        largest_savings=[{k:b[k] for k in ('rank','batch_index','offset','old_suffix_width','row_suffix_width',
            'saved_suffix_slots','suffix_slot_reduction_fraction')} for b in sorted(batches,
                key=lambda b:(-b['saved_suffix_slots'],b['rank'],b['batch_index']))[:5]],
        largest_remaining_suffix_padding=[{k:b[k] for k in ('rank','batch_index','offset',
            'row_suffix_width','row_true_suffix_lengths','action_tokens','row_right_padding_slots')}
            for b in sorted(batches,key=lambda b:(-b['row_right_padding_slots'],b['rank'],b['batch_index']))[:5]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--geometry', type=Path, default=Path(__file__).resolve().parents[5]/
                        'experiments/rl/results_actual_prefix_request_geometry_20261004.json')
    parser.add_argument('--output-directory', type=Path, default=Path(__file__).resolve().parent/
                        'replay-slot-distribution-20261007')
    args = parser.parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    json_path = args.output_directory/'prefix-replay-slot-distribution.json'
    csv_path = args.output_directory/'prefix-replay-slot-distribution.csv'
    if json_path.exists() or csv_path.exists():
        raise FileExistsError('Keep the original diagnostic receipts; choose another output directory.')
    data = json.loads(args.geometry.read_bytes())
    base = Path(__file__).resolve().parent
    owner_paths = dict(
        readout=base.parents[4]/'experiments/rl/reward_readout.py',
        row_runner=base/'native-representation-candidate/candidate/qwen35_dense_finite_runner.py',
        row_targets=base/'native-representation-candidate/candidate/qwen35_answer_finite.py',
        row_leases=base/'native-representation-candidate/candidate/native_prefix_leases.py',
        row_artifacts=base/'native-representation-candidate/candidate/qwen35_native_prefix_artifacts.py',
        combined_storage_leases=base/'combined-storage-row-v1/candidate/native_prefix_leases.py',
        combined_storage_artifacts=base/'combined-storage-row-v1/candidate/qwen35_native_prefix_artifacts.py')
    # Sources are inspected and hashed, never imported or executed.
    source_records = {}
    for name, path in owner_paths.items():
        tree = ast.parse(path.read_text(encoding='utf-8'))
        source_records[name] = dict(**identity(path), functions={n.name:dict(line=n.lineno,end_line=n.end_lineno)
            for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in
            ('episodes','prepare_native_prefix_leases','_pack_native_cached_suffix_rows',
             '_compact_native_cached_kv','suffix_rows','_compose_row_prefix_cache')})
    ranks = []
    all_batches = []
    for rank, record in enumerate(data['ranks']):
        rows = record['rows']
        assert len(rows)==record['request_count']==88
        assert [r['sorted_row'] for r in rows]==list(range(88))
        assert [r['context_tokens'] for r in rows]==sorted(r['context_tokens'] for r in rows)
        batches = []
        for original in record['batches']:
            indices = original['sorted_rows']
            assert indices==list(range(original['batch_index']*4, original['batch_index']*4+4))
            selected = [rows[i] for i in indices]
            lengths = [r['context_tokens'] for r in selected]
            starts = [r['source_start'] for r in selected]
            cuts = [start//64*64 for start in starts]
            assert all(r['context_tokens']==r['sum_saved_components']==r['prompt_tokens']+
                r['action_tokens']+r['query_tokens']+r['target_tokens'] for r in selected)
            assert starts==[r['prompt_tokens'] for r in selected]
            common = min(cuts)
            assert common==original['local_common_prefix_boundary']>0
            old_width = max(lengths)-common
            true_suffix = [length-cut for length,cut in zip(lengths,cuts)]
            width = max(true_suffix)
            prefix_width = max(cuts)
            core = sum(length-start for length,start in zip(lengths,starts))
            align = sum(start-cut for start,cut in zip(starts,cuts))
            history = sum(cut-common for cut in cuts)
            old_pad = 4*max(lengths)-sum(lengths)
            new_pad = 4*width-sum(true_suffix)
            holes = sum(prefix_width-cut for cut in cuts)
            assert old_width==original['local_suffix_tokens']
            assert 4*old_width==core+align+history+old_pad
            assert 4*width==core+align+new_pad
            assert 4*(prefix_width+width)-sum(lengths)==holes+new_pad
            assert width<=old_width
            batch = dict(rank=rank,batch_index=original['batch_index'],offset=indices[0],
                sorted_rows=indices,original_row_indices=[r['row_index'] for r in selected],
                traj_uids=[r['traj_uid'] for r in selected],source_steps=[r['source_step'] for r in selected],
                logical_context_lengths=lengths,source_starts=starts,action_tokens=[r['action_tokens'] for r in selected],
                query_tokens=[r['query_tokens'] for r in selected],target_tokens=[r['target_tokens'] for r in selected],
                common_cut=common,row_cuts=cuts,old_true_suffix_lengths=[length-common for length in lengths],
                row_true_suffix_lengths=true_suffix,old_suffix_width=old_width,row_suffix_width=width,
                old_suffix_slots=4*old_width,row_suffix_slots=4*width,saved_suffix_slots=4*(old_width-width),
                suffix_slot_reduction_fraction=(old_width-width)/old_width,
                action_query_target_slots=core,alignment_slots=align,old_common_history_tail_slots=history,
                old_right_padding_slots=old_pad,row_right_padding_slots=new_pad,
                old_KV_width=max(lengths),row_prefix_carrier_width=prefix_width,row_KV_width=prefix_width+width,
                old_KV_carrier_slots=4*max(lengths),row_KV_carrier_slots=4*(prefix_width+width),
                row_internal_prefix_hole_slots=holes,
                row_native_KV_carrier_ratio=(prefix_width+width)/max(lengths))
            batches.append(batch)
        assert len(batches)==22
        total = summarize(batches)
        last = batches[-1]
        total['offset84_concentration'] = dict(last_batch_saved_suffix_slots=last['saved_suffix_slots'],
            last_batch_fraction_of_rank_savings=last['saved_suffix_slots']/total['totals']['saved_suffix_slots'],
            excluding_last_batch=summarize(batches[:-1]))
        ranks.append(dict(rank=rank,literal_requests=dict(path=record['path'],sha256=record['sha256'],bytes=record['bytes']),
                          summary=total,batches=batches))
        all_batches.extend(batches)
    fields = [k for k in all_batches[0] if k not in ('traj_uids',)]
    with csv_path.open('w',encoding='utf-8',newline='') as handle:
        writer = csv.DictWriter(handle,fieldnames=fields)
        writer.writeheader()
        for batch in all_batches:
            writer.writerow({key:json.dumps(batch[key],separators=(',',':')) if isinstance(batch[key],list)
                             else batch[key] for key in fields})
    result = dict(scope='All original 88 requests/rank and original22 B4 groups/rank; descriptive CPU metadata geometry only.',
        script=identity(__file__),metadata=identity(args.geometry),csv=identity(csv_path),owner_sources=source_records,
        formulas=dict(L_i='Original saved context_tokens includes literal query and target; no query reconstruction.',
            p_i='floor(source_start_i/64)*64',p_common='min(p_i) within the original local B4; no historical cross-rank MIN.',
            old_suffix_width='max(L_i)-p_common',row_suffix_width='max(L_i-p_i)',row_KV_width='max(p_i)+max(L_i-p_i)',
            old_decomposition='B*old_width = action_query_target + 64_alignment + common_min_history_tail + right_padding',
            row_decomposition='B*row_width = action_query_target + 64_alignment + right_padding',
            row_carrier_padding='B*(P+S)-sum(L_i) = sum(P-p_i) + B*S-sum(L_i-p_i)',
            replay_layer_token_slots='2 endpoints *32 decoder replays * single-endpoint suffix slots; layout extent, not FLOPs.'),
        ranks=ranks,pooled_176=summarize(all_batches),
        scope_limits=[
            'This population retains the historical literal query IDs/lengths. Current clock-query hot probes have different query lengths; no full176 current-query timing is inferred.',
            'These are dense suffix/carrier extents. Original FA varlen masking/unpadding and compact finite KV use valid logical sequences; carrier extent is not measured FA memory traffic.',
            'The reported32 replay layer-token extents do not equate compute across layer types, finite kernels, backward, communication or memory copy.',
            'Offset84 is the only measured hot B8 here; no other batch wall time, FLOP ratio or speedup is inferred from slot ratios.',
            'Remaining suffix padding is observable work amplification, not proof all of it is removable at equal behavior/resources.',
            'Bank preparation and its separate storage/segment accounting are excluded; no task, scheduler, model, GPU, DT, optimizer or checkpoint is run.'])
    json_path.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(json=identity(json_path),csv=identity(csv_path),pooled_176=result['pooled_176'],
        ranks=[dict(rank=r['rank'],summary=r['summary']) for r in ranks])))


if __name__=='__main__':
    main()

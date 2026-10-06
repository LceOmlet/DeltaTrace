"""CPU-only descriptive comparison of the completed real-B8-v3 saved Q/V/A."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def identity(path):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.environ.get('MACA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU-only comparison requires empty CUDA/MACA visibility')
    import psutil
    import torch
    torch.set_num_threads(1)
    started = time.time()
    process = psutil.Process()
    prepared_path = args.source/'prepared.json'
    prepared = json.loads(prepared_path.read_bytes())
    labels = ('shared_cold','shared_warm','shared_individual_rows_cold',
              'shared_individual_rows_warm','shared_individual_rows_observed')
    pairs = (('shared_cold','shared_warm','Same scalar bank, cold/warm repeats.'),
             ('shared_individual_rows_cold','shared_individual_rows_warm','Same row bank, cold/warm repeats.'),
             ('shared_individual_rows_warm','shared_individual_rows_observed','Same row bank, observed call additionally saves operands.'),
             ('shared_warm','shared_individual_rows_warm','Different scalar/row banks, native layouts and finite representation; cannot isolate one kernel.'))
    keys = ('dt_token_advantages','dt_q_estimates','dt_v_estimates')
    def stats(values):
        flat = values.detach().double().reshape(-1)
        finite = flat[torch.isfinite(flat)]
        result = dict(count=flat.numel(), finite=int(torch.isfinite(flat).sum()),
                      nonfinite=int((~torch.isfinite(flat)).sum()))
        if finite.numel():
            result.update(min=float(finite.min()), max=float(finite.max()),
                mean=float(finite.mean()), mean_absolute=float(finite.abs().mean()),
                sum=float(finite.sum()), absolute_sum=float(finite.abs().sum()),
                rms=float(finite.square().mean().sqrt()),
                median_absolute=float(finite.abs().median()))
        return result
    ranks = []
    for rank in range(2):
        rank_path = args.source/f'rank{rank}.json'
        record = json.loads(rank_path.read_bytes())
        assert record['phase'] == 'native_prefix_lease_diagnostic_complete'
        assert all(label in record['reports'] for label in labels)
        vector_path = Path(record['vectors']['path'])
        assert identity(vector_path)['sha256'] == record['vectors']['sha256']
        vectors = torch.load(vector_path, map_location='cpu', weights_only=False)
        request_path = args.source/f'actual-requests-rank{rank}.pt'
        expected = prepared['literal_request_files'][str(request_path)]
        assert identity(request_path)['sha256'] == expected
        payload = torch.load(request_path, map_location='cpu', weights_only=False)
        requests = sorted(payload['requests'], key=lambda r:r['context_tokens'])
        offset = int(prepared['selected_observation']['offset'])
        chosen = requests[offset:offset+4]
        rows = [payload['rows'][r['row_index']] for r in chosen]
        mask_rows = []
        selected = []
        for index,(request,row) in enumerate(zip(chosen,rows)):
            width = row['responses'].numel()
            valid = row['attention_mask'].bool()[-width:]
            assert bool(row['active_masks'])
            assert valid.nonzero().flatten().tolist() == list(range(request['actions'].numel()))
            assert torch.equal(row['responses'][:request['actions'].numel()],request['actions'])
            assert torch.equal(row['input_ids'][-width:],row['responses'])
            mask_rows.append(valid)
            selected.append(dict(local_row=index, traj_uid=request['traj_uid'], source_step=request['source_step'],
                original_row_index=request['row_index'], original_source_start=request['start'],
                original_source_end=request['end'], valid_action_tokens=request['actions'].numel(),
                response_vector_width=width, observed_return=request['observed_return']))
        mask = torch.stack(mask_rows)
        variants = {}
        for label in labels:
            report = record['reports'][label]
            traces = report['original_readout_report']['traces']
            assert len(traces) == 4
            per_row = []
            for index,(request,trace) in enumerate(zip(chosen,traces)):
                assert trace['source_step'] == request['source_step']
                values = {key:vectors[label][key][index] for key in keys}
                assert all(value.shape == mask[index].shape for value in values.values())
                per_row.append(dict(**selected[index],
                    action_values={key:stats(value[mask[index]]) for key,value in values.items()},
                    outside_action_values={key:stats(value[~mask[index]]) for key,value in values.items()},
                    original_trace={key:trace.get(key) for key in ('context_tokens','compute_tokens','query_tokens',
                        'root_effect','signed_sum','conservation_residual','factual_target_logp','reference_target_logp')},
                    native_shared_prefix_length=report['original_readout_report'].get('shared_native_prefix')))
            variants[label] = dict(shape={key:list(vectors[label][key].shape) for key in keys},
                dtype={key:str(vectors[label][key].dtype) for key in keys},
                original_wall_seconds=report['total_wall_seconds'],
                original_runner_phase_seconds=report['original_runner_phase_seconds'],
                original_runner_phase_counts=report['original_runner_phase_counts'],
                action_values={key:stats(vectors[label][key][mask]) for key in keys}, per_row=per_row)
        comparisons = []
        for left_label,right_label,scope in pairs:
            fields = []
            for key in keys:
                left,right = vectors[left_label][key],vectors[right_label][key]
                assert left.shape == right.shape == mask.shape and left.dtype == right.dtype
                delta = right.double()-left.double()
                a,b = left[mask],right[mask]
                finite = torch.isfinite(a)&torch.isfinite(b)
                nonzero = (a != 0)&(b != 0)&finite
                rows_out = []
                for index,request in enumerate(chosen):
                    values = delta[index][mask[index]]
                    count = min(5,values.numel())
                    tops = torch.topk(values.abs(),count).indices.tolist()
                    rows_out.append(dict(**selected[index], difference=stats(values),
                        top_absolute_differences=[dict(action_index=i, source_input_position=request['start']+i,
                            literal_token_id=int(request['actions'][i]), left=float(left[index,i]),
                            right=float(right[index,i]), right_minus_left=float(delta[index,i])) for i in tops]))
                fields.append(dict(key=key,dtype=str(left.dtype), shape=list(left.shape),
                    complete_literal_bytes_equal=bool(torch.equal(left.contiguous().view(torch.uint8),right.contiguous().view(torch.uint8))),
                    complete_numerical_equal=bool(torch.equal(left,right)),
                    action_difference=stats(delta[mask]), outside_action_difference=stats(delta[~mask]),
                    changed_action_values=int((a!=b).sum()),
                    action_sign_counts=dict(both_nonzero=int(nonzero.sum()),
                        opposite=int(((torch.sign(a)!=torch.sign(b))&nonzero).sum()),
                        same_nonzero=int(((torch.sign(a)==torch.sign(b))&nonzero).sum()),
                        left_zero_right_nonzero=int(((a==0)&(b!=0)&finite).sum()),
                        left_nonzero_right_zero=int(((a!=0)&(b==0)&finite).sum())), per_row=rows_out))
            comparisons.append(dict(left=left_label,right=right_label,scope=scope,fields=fields,
                endpoint_differences=[dict(local_row=index,**{
                    key:None if (variants[left_label]['per_row'][index]['original_trace'][key] is None or
                        variants[right_label]['per_row'][index]['original_trace'][key] is None) else
                        variants[right_label]['per_row'][index]['original_trace'][key]-variants[left_label]['per_row'][index]['original_trace'][key]
                    for key in ('factual_target_logp','reference_target_logp','root_effect','signed_sum')}) for index in range(4)]))
        ranks.append(dict(rank=rank,rank_json=identity(rank_path),vectors=identity(vector_path),
            original_request_file=identity(request_path), imported_sources=record['imported_sources'],
            original_raw_value_observations=record.get('raw_value_observations'),
            action_mask_source='Original row attention_mask response tail and active_masks; literal actions checked against original response IDs.',
            valid_action_tokens=int(mask.sum()),outside_action_positions=int((~mask).sum()),
            selected_rows=selected,variants=variants,comparisons=comparisons))
        del record,vectors,payload
    assert not torch.cuda.is_initialized()
    result = dict(scope='Saved actual B8-v3 Q/V/A CPU descriptive comparison, no numerical acceptance threshold.',
        script=identity(Path(__file__)),prepared=identity(prepared_path),
        original_job=identity(args.source/'job.json'),ranks=ranks,
        execution=dict(pid=os.getpid(),pid_birth=process.create_time(),elapsed_seconds=time.time()-started,
            PSS_bytes=process.memory_full_info().pss,RSS_bytes=process.memory_info().rss,
            torch_version=torch.__version__,CUDA_initialized=False,CUDA_VISIBLE_DEVICES='',MACA_VISIBLE_DEVICES=''),
        limitations=['Different banks/layouts/operators are jointly changed in baseline-versus-row; differences cannot be assigned to one kernel.',
            'Cold/warm/observed comparisons describe actual saved repeatability only; no full-network tolerance is invented.',
            'Action signs and Q/V/A differences do not establish real-world token counterfactual correctness or learning quality.',
            'No d is reconstructed from Q/V/A; native endpoint fields are the original stored head observations.',
            'This CPU script performs zero model, attention, FLA, DT, backward, optimizer, checkpoint or GPU calls.'])
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(output=str(args.output),sha256=identity(args.output)['sha256'],
        ranks=[dict(rank=r['rank'],valid_action_tokens=r['valid_action_tokens'],
            comparisons=[dict(left=c['left'],right=c['right'],values=[dict(key=f['key'],
                numerical_equal=f['complete_numerical_equal'],difference=f['action_difference'],signs=f['action_sign_counts'])
                for f in c['fields']]) for c in r['comparisons']]) for r in ranks])))


if __name__ == '__main__':
    main()

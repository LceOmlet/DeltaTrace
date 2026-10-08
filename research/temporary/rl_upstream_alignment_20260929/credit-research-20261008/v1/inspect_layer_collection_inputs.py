"""CPU-only exact owner preparation and cost inventory; never run a model."""
import hashlib
import inspect
import json
import os
from pathlib import Path
import time

import psutil
import torch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    import reward_readout
    torch.set_num_threads(2)
    started = time.perf_counter()
    plan_path, task, source_path, output = map(str, __import__('sys').argv[1:])
    plan = json.loads(Path(plan_path).read_bytes())
    source = json.loads(Path(source_path).read_bytes())
    owner_path = inspect.getsourcefile(reward_readout.DirectActionTargetReadout)
    assert sha(owner_path) == source['entry_sha256']['reward_readout.py']
    entries = plan['tasks'][task]['entries']
    inventory = []
    for entry in entries:
        path = entry['native']['path']
        assert sha(path) == entry['native']['sha256']
        native = torch.load(path, map_location='cpu', weights_only=False)
        row = next(r for r in native['rows'] if str(r['traj_uid']) == entry['traj_uid'])
        prepared = reward_readout.DirectActionTargetReadout._prepare_row(row['row'], 0)
        for key in ('selected', 'suffix_positions', 'prior', 'target'):
            assert torch.equal(prepared[key], row[key]), (entry['traj_uid'], key)
        assert prepared['target_offsets'] == row['target_offsets']
        assert torch.equal(prepared['case']['target_ids'], row['case']['target_ids'])
        assert prepared['selected'].numel() == entry['selected_tokens'] <= 32768
        positions = row['prompt_length'] + row['prior'][row['suffix_positions']].nonzero().flatten()
        for query in entry['queries']:
            assert int(positions[query['source_index']]) == query['packed_slot']
            assert int(row['selected'][query['packed_slot']]) == query['token_id']
        detail = native['detail']
        inventory.append(dict(traj_uid=entry['traj_uid'], context=entry['selected_tokens'],
            original_native_path=path, original_batch_width=native['native_signed'].shape[1],
            original_DT_seconds=detail.get('complete_attribution_seconds_with_diagnostics'),
            original_peak_allocated=detail.get('peak_allocated'),
            original_peak_reserved=detail.get('peak_reserved'),
            original_prefix=detail.get('native_shared_prefix_length'),
            original_row_prefix=detail.get('native_row_prefix_lengths')))
        del native, row, prepared
    batches = []
    for batch in plan['tasks'][task]['batches']:
        width = batch['padded_width']
        # Conservative full-width inventories. The actual retained suffix is
        # no larger; no all-query hidden snapshots or model copies are needed.
        batches.append(dict(batch, retained_FP64_coefficient_upper_bytes=33*4*width*4096*8,
            retained_BF16_factual_endpoint_upper_bytes=33*4*width*4096*2,
            largest_single_row_FP64_contraction_operand_bytes=width*4096*8,
            scope='Host bank estimates from actual widths and Qwen3.5-9B H=4096, not physical GPU peaks or a capacity acceptance. Retain one B4 bank at a time.'))
    assert not torch.cuda.is_initialized()
    result = dict(scope=__doc__, status='Exact original owner CPU preparation verified; no model/DT launch',
        task=task, source_path=source_path, source_sha256=sha(source_path),
        input_plan_sha256=sha(plan_path), owner=dict(path=owner_path, sha256=sha(owner_path)),
        entries=inventory, batches=batches,
        elapsed_seconds=time.perf_counter()-started, pid=os.getpid(),
        birth=psutil.Process().create_time(), cuda_initialized=False,
        process_pss_bytes=psutil.Process().memory_full_info().pss,
        host_available_bytes=psutil.virtual_memory().available,
        operations=dict(model=0, DT=0, native_forward=0, backward=0, optimizer=0, rollout=0),
        limitation='Stored original times cover original full batches, not newly grouped diagnostic cost. They must not be added once per trajectory or presented as newly measured timings.')
    Path(output).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(task=task, status=result['status'], trajectories=len(entries),
        DT_B4_calls=len(batches), native_paired_forwards=sum(b['native_paired_forwards'] for b in batches),
        max_host_bank_upper_bytes=max(b['retained_FP64_coefficient_upper_bytes']+b['retained_BF16_factual_endpoint_upper_bytes'] for b in batches),
        elapsed_seconds=result['elapsed_seconds'], cuda_initialized=False)))


if __name__ == '__main__':
    main()

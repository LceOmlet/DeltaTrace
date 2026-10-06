"""CPU description of saved old/new native credit, using original B4 aggregation."""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import pickle
import time


def identity(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size)


def source_identity(function):
    return identity(inspect.getsourcefile(function))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', required=True, type=Path)
    parser.add_argument('--minibatch-path', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    import numpy as np
    import psutil
    import torch
    from verl import DataProto
    from verl.trainer.ppo import core_algos as core
    from analyze_saved_native_b4_denominators import original_denominator_contributions

    started = time.monotonic()
    assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
    assert not torch.cuda.is_initialized() and not torch.distributed.is_initialized()
    directory = args.input_dir
    new_path = directory / 'clock-recomputed-native-minibatch.pkl'
    old, new = [DataProto.load_from_disk(str(p)) for p in (args.minibatch_path, new_path)]
    changed = {'dt_token_advantages', 'dt_q_estimates', 'dt_v_estimates', 'advantages', 'returns'}
    fields = {key: dict(old_shape=list(old.batch[key].shape), new_shape=list(new.batch[key].shape),
        old_dtype=str(old.batch[key].dtype), new_dtype=str(new.batch[key].dtype),
        equal=torch.equal(old.batch[key], new.batch[key])) for key in old.batch.keys()}
    metadata = {key: bool(np.array_equal(value, new.non_tensor_batch[key]))
        for key, value in old.non_tensor_batch.items()}
    comparisons = dict(fields=fields, unchanged_noncredit_fields=all(
        record['equal'] for key, record in fields.items() if key not in changed),
        non_tensor_fields=metadata, old_meta_info_keys=sorted(old.meta_info), new_meta_info_keys=sorted(new.meta_info),
        existing_meta_info_pickle_equal={key: pickle.dumps(value) == pickle.dumps(new.meta_info[key])
            for key, value in old.meta_info.items()},
        row_returns_old=old.batch['token_level_rewards'].sum(-1).tolist(),
        row_returns_new=new.batch['token_level_rewards'].sum(-1).tolist())
    micro_records, rank_records = [], []
    for rank, (old_local, new_local) in enumerate(zip(old.chunk(2), new.chunk(2))):
        records = []
        for index, (a, b) in enumerate(zip(old_local.batch.split(4), new_local.batch.split(4))):
            width = a['responses'].shape[-1]
            mask = a['loss_mask'][:, -width:]
            record = dict(rank=rank, microbatch_index=index,
                original_mask_denominator=int(mask.sum()), native_accumulation_divisor=8, methods={})
            for name, micro in (('old', a), ('new', b)):
                values, q = micro['advantages'], micro['dt_q_estimates']
                subsets = dict(all=torch.ones_like(mask, dtype=torch.bool), Q_zero=q == 0, Q_nonzero=q != 0)
                record['methods'][name] = {label: dict(
                    action_tokens=int(((mask != 0) & subset).sum()),
                    original_denominator=original_denominator_contributions(
                        values, mask, subset, torch, core.agg_loss)) for label, subset in subsets.items()}
            records.append(record)
            micro_records.append(record)
        rank_records.append(dict(rank=rank, B4_count=len(records), methods={name: {label: dict(
            mean_abs_A=sum(r['methods'][name][label]['original_denominator']['abs_mean'] for r in records) / 8,
            signed_mean_A=sum(r['methods'][name][label]['original_denominator']['mean'] for r in records) / 8,
            action_tokens=sum(r['methods'][name][label]['action_tokens'] for r in records))
            for label in ('all', 'Q_zero', 'Q_nonzero')} for name in ('old', 'new')}))
    reports = {}
    report_inputs = {}
    for rank in (0, 1):
        path = directory / f'rank{rank}-clock-native-dt-report.json'
        native = json.loads(path.read_bytes())
        reports[str(rank)] = dict(query_method_restored=native['query_method_restored'],
            scalars={key: value for key, value in native['report'].items() if not isinstance(value, (dict, list))},
            trace_count=len(native['report']['traces']))
        report_inputs[str(rank)] = identity(path)
    value = dict(scope=__doc__, inputs=dict(old=identity(args.minibatch_path), new=identity(new_path), native_DT=report_inputs),
        sources=dict(analyzer=identity(__file__), protocol_load=source_identity(DataProto.load_from_disk),
            protocol_chunk=source_identity(DataProto.chunk), core_aggregate=source_identity(core.agg_loss),
            original_denominator_helper=source_identity(original_denominator_contributions)),
        comparisons=comparisons, B4_records=micro_records, rank_native_B4_averages=rank_records,
        mean_of_two_rank_native_B4_averages={name: sum(
            r['methods'][name]['all']['mean_abs_A'] for r in rank_records) / 2 for name in ('old', 'new')},
        native_DT_reports=reports, runtime=dict(pid=os.getpid(), pid_birth=psutil.Process().create_time(),
            rss_bytes=psutil.Process().memory_info().rss, elapsed_seconds=time.monotonic()-started,
            cuda_initialized=torch.cuda.is_initialized(), distributed_initialized=torch.distributed.is_initialized()),
        model_calls=0, DT_calls=0, backward_calls=0, optimizer_steps=0, scheduler_steps=0,
        limits=['B4 token means retain each full original loss_mask denominator and eight-way accumulation; no global pooled mass replaces it.',
                'Credit magnitude and support statistics are not gradient norms or success-rate changes.',
                'Native internal conservation diagnostics are not FA/FLA official tolerance assertions.'])
    args.output.write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps(dict(status='CPU_saved_credit_described', output=str(args.output),
        source_sha256=value['sources']['analyzer']['sha256'], unchanged_noncredit_fields=comparisons['unchanged_noncredit_fields'],
        B4_mean_abs_A=value['mean_of_two_rank_native_B4_averages'], runtime=value['runtime'])))


if __name__ == '__main__':
    main()

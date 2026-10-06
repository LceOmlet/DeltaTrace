"""CPU accounting of existing formal reports and an existing native DT profile.

No model, task, RPC, GPU, or training calls. Do not add concurrent rank times.
"""
import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    raw = path.read_bytes()
    return json.loads(raw), dict(path=path.as_posix(), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reports', type=Path, required=True)
    p.add_argument('--profile', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    reports, report_source = read(args.reports)
    profile, profile_source = read(args.profile)
    ranks = []
    for index, row in enumerate(reports['records']):
        groups = row['groups']
        complete_seconds = sum(g['seconds'] for g in groups)
        bank_seconds = sum(g['shared_native_prefix']['capture_and_preparation_seconds'] for g in groups)
        batches = [b for g in groups for b in g['preceding_logged_batches']]
        times = sorted(b['seconds'] for b in batches)
        median = (times[(len(times)-1)//2]+times[len(times)//2])/2 if times else None
        ranks.append(dict(rank=index, completed_groups=len(groups),
            completed_contrasts=sum(g['event_contrasts'] for g in groups),
            completed_B4_calls=sum(g['finite_trace_calls'] for g in groups),
            pending_returned_B4_calls=len(row['pending_logged_batches']),
            completed_group_seconds=complete_seconds, bank_seconds=bank_seconds,
            bank_fraction=bank_seconds/complete_seconds, logged_B4_median_seconds=median,
            logged_B4_max_seconds=max(times) if times else None,
            capture_token_slots=sum(g['shared_native_prefix']['capture_token_slots'] for g in groups),
            previous_per_batch_prefix_token_slots=sum(g['shared_native_prefix']['original_shared_prefix_token_slots'] for g in groups),
            actual_context_token_sum=sum(g['trace_summary']['context_tokens_sum'] for g in groups),
            dense_token_slots=sum(g['trace_summary']['compute_token_slots'] for g in groups),
            max_context=max(g['max_readout_length'] for g in groups)))
    warm = []
    for rank in profile['summaries']:
        w = rank['rank_reports']['shared_warm']
        phase = w['original_runner_phase_seconds']
        warm.append(dict(rank=rank['rank'], wall_seconds=w['total_wall_seconds'],
            root_seconds=phase['native_root_with_CPU_checkpoints'],
            replay_seconds=sum(v for k, v in phase.items() if k.startswith('native_replay_')),
            finite_decoder_seconds=sum(v for k, v in phase.items() if k.startswith('finite_decoder_')),
            public_FA_LSE_seconds=sum(v for k, v in phase.items() if k.startswith('public_FA_LSE_')),
            phase_scope='Original CUDA-event regions of one saved checkpoint20 warm B4; not a formal-iteration attribution of time.',
            profile_collectives=rank['collectives'],
            profile_scope='Only the existing profiled warm attribute; device collective duration can include peer waiting, not removable bandwidth time.'))
    result = dict(scope=__doc__, observed_unix=reports['observed_unix'],
        original_reports=report_source, existing_profile=profile_source, formal_ranks=ranks,
        representative_warm_B4=warm,
        not_claimed=['Complete iteration wall time or exact generation time',
                     'Current trained-weight numerical acceptance',
                     'Speedup of the retained-action candidate',
                     'That all replay time can be removed without memory or transfer cost'])
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(dict(output=args.output.as_posix(), formal_ranks=ranks, warm_B4=[{k:v for k,v in r.items() if k not in ('profile_collectives','profile_scope','phase_scope')} for r in warm])))


if __name__ == '__main__':
    main()

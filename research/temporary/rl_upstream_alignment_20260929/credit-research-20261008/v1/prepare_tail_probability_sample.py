"""One probability sample for negative-credit confusion counts and recall.

Reads existing development artifacts on CPU. No model, DT, native deletion,
candidate, training or held-out-data evaluation is run. The sample is frozen
before obtaining any new single-deletion outcome.
"""
import argparse
import bisect
import hashlib
import json
import math
from pathlib import Path
import random
import time


SEED = '20261009-unified-negative-credit-probability-sample-v1'
BODY_SAMPLE = 1024  # A measurement budget, not a claim of adequate precision.
BINS = ('ratio_le_1', 'ratio_1_to_2', 'ratio_2_to_10',
        'ratio_10_to_100', 'ratio_gt_100')


def ref(path):
    path = Path(path)
    raw = path.read_bytes()
    return dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def allocate(counts, budget):
    """Proportional allocation, then largest remainder; no outcome is read."""
    total = sum(counts)
    if total <= budget:
        return list(counts)
    quota = [budget*n/total for n in counts]
    answer = [int(x) for x in quota]
    for i in sorted(range(len(counts)), key=lambda i: (-(quota[i]-answer[i]), i))[:budget-sum(answer)]:
        answer[i] += 1
    assert sum(answer) == budget and all(0 < n <= N for n, N in zip(answer, counts) if N)
    return answer


def main():
    import psutil
    import torch
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    base = Path(args.directory)
    output = Path(args.output)
    assert not output.exists(), 'Do not redraw or replace the frozen sample'
    started = time.perf_counter()
    corpus = json.loads((base/'corpus.json').read_bytes())
    manifest = json.loads((base/'manifest.json').read_bytes())
    strata = json.loads((base/'credit-strata.json').read_bytes())
    for name in ('corpus.json', 'manifest.json'):
        assert ref(base/name)['sha256'] == strata['provenance'][name]
    result = dict(scope=__doc__, seed=SEED, sample_budget_per_task=BODY_SAMPLE,
        source_commit=json.loads((base/'source.json').read_bytes())['commit'],
        sources=[ref(base/n) for n in ('corpus.json', 'manifest.json', 'credit-strata.json')],
        script=ref(__file__), tasks={},
        design=dict(
            population='All eligible source tokens in completed, original frozen development captures only; no held-out results or task-wide claim.',
            prediction='The frozen saved DT d for each token. Repeated-estimate instability is separate, not a third classifier label.',
            diagnostic_boundaries='c=exp(-d), A/r=1-c; existing PLAN bins and cumulative thresholds c>2, c>10, c>100. Not training gates.',
            strata='Two non-tail prediction strata sampled uniformly without replacement over the complete task frame; predicted c>2 is a census.',
            inclusion_probability='n_h/N_h for non-tail strata; 1 for predicted-tail census. No selection on native d or outcomes.',
            primary_estimand='Token-total confusion counts in this finite saved frame; ratios use Horvitz-Thompson weighted totals.',
            secondary_estimand='Equal initial state, then equal completed trajectory, then equal eligible source; use the separately recorded population and sampling weights.',
            grouping='Task, initial state, prior exposure, and crossed predicted/native magnitude bins. Do not pool unbounded advantage magnitudes.',
            sampling_uncertainty='Invert the finite-population hypergeometric distribution for each sampled stratum; combine simultaneous FN bounds with census TP bounds for recall. Zero sampled positives do not give a zero-width interval.',
            reference_uncertainty='Native dtype, same-hidden FP32 head, and rounded FP32 head are recorded separately. Their observed range is numerical sensitivity, not a confidence interval or exact world oracle. Threshold-crossing labels remain unresolved.',
            missingness='Missing original captures remain listed and are excluded only from the explicitly completed-artifact frame. Never impute a negative label.',
            precision_claim='1024 is a fixed collection budget. Report the resulting uncertainty, positive counts and state coverage; it does not guarantee adequate recall precision for every group or decade.',
            reuse='Existing 128-source and tail diagnostics remain localization evidence; do not append their hand-dependent outcome history to this new probability-sample estimator.'),
        operations=dict(model=0, DT=0, GPU=0, native_forward=0, optimizer=0,
            rollout=0, checkpoint_restore=0, production_patch=0))
    checked_files = {}
    peak_pss = 0
    for task, task_manifest in manifest['tasks'].items():
        groups = [g for g in task_manifest['groups'] if g['split'] == 'development']
        group_by_uid = {uid:g for g in groups for uid in g['trajectory_uids']}
        records = {r['traj_uid']:r for r in corpus['tasks'][task]['records'] if r['traj_uid'] in group_by_uid}
        completed = {uid:r for uid, r in records.items() if r['native_occurrences']}
        files = {f['sha256']:f for f in corpus['tasks'][task]['native_files']}
        saved = strata['tasks'][task]
        assert set(completed) == set(saved['rows'])
        totals = [saved['summary'][b]['tokens'] for b in BINS]
        sizes = allocate(totals[:2], BODY_SAMPLE) + totals[2:]
        # Ordinal frame: trajectory UID, then source order within that prediction
        # stratum. Known population counts are checked against original tensors.
        starts = {b:[] for b in BINS}
        offsets = {b:0 for b in BINS}
        for uid in sorted(completed):
            for b in BINS:
                N = saved['rows'][uid]['strata'][b]['tokens']
                starts[b].append((offsets[b], offsets[b]+N, uid))
                offsets[b] += N
        assert [offsets[b] for b in BINS] == totals
        requests = {uid:{} for uid in completed}
        for b, N, n in zip(BINS, totals, sizes):
            seed = int(hashlib.sha256((SEED+'\0'+task+'\0'+b).encode()).hexdigest(), 16)
            ordinals = sorted(random.Random(seed).sample(range(N), n)) if n < N else range(N)
            ends = [end for _, end, _ in starts[b]]
            for ordinal in ordinals:
                index = bisect.bisect_right(ends, ordinal)
                first, end, uid = starts[b][index]
                assert first <= ordinal < end
                requests[uid].setdefault(b, []).append(ordinal-first)
        state_completed = {g['initial_state_sha256']:sum(uid in completed for uid in g['trajectory_uids']) for g in groups}
        entries = {}
        wanted = {}
        for uid, record in completed.items():
            occ = record['native_occurrences'][0]
            wanted.setdefault(occ['file_sha256'], []).append(uid)
        for digest, uids in wanted.items():
            binding = files[digest]
            actual = ref(binding['path'])
            assert actual['sha256'] == digest and actual['bytes'] == binding['bytes']
            checked_files[digest] = actual
            native = torch.load(binding['path'], map_location='cpu', weights_only=False)
            for uid in sorted(uids):
                r = completed[uid]
                occ = r['native_occurrences'][0]
                row = next(v for v in native['rows'] if v['batch_row'] == occ['batch_row'])
                assert str(row['traj_uid']) == uid
                pos = row['prompt_length']+row['prior'][row['suffix_positions']].nonzero().flatten()
                d = native['native_signed'][row['batch_row'], pos]
                assert len(pos) == occ['source_tokens'] and torch.isfinite(d).all()
                h = hashlib.sha256(str((str(d.dtype), tuple(d.shape))).encode())
                h.update(d.contiguous().numpy().tobytes())
                assert h.hexdigest() == occ['native_signed_sha256']
                masks = [d >= 0, (d < 0) & (d >= -math.log(2)),
                    (d < -math.log(2)) & (d >= -math.log(10)),
                    (d < -math.log(10)) & (d >= -math.log(100)), d < -math.log(100)]
                group = group_by_uid[uid]
                queries = []
                for b, N, n, mask in zip(BINS, totals, sizes, masks):
                    indices = mask.nonzero().flatten()
                    assert len(indices) == saved['rows'][uid]['strata'][b]['tokens']
                    for ordinal in requests[uid].get(b, []):
                        source_index = int(indices[ordinal])
                        slot = int(pos[source_index])
                        pi = n/N
                        queries.append(dict(source_index=source_index, packed_slot=slot,
                            token_id=int(row['selected'][slot]), saved_d=float(d[source_index]),
                            prediction_stratum=b, stratum_population=N, stratum_sample=n,
                            inclusion_probability=pi, token_total_weight=1/pi,
                            state_trajectory_source_weight=1/(len(groups)*state_completed[group['initial_state_sha256']]*len(pos)*pi)))
                entries[uid] = dict(traj_uid=uid, initial_state_sha256=group['initial_state_sha256'],
                    previously_examined=group['previously_examined'], native=binding,
                    occurrence=occ, selected_tokens=len(row['selected']), source_tokens=len(pos),
                    source_strata_populations={b:saved['rows'][uid]['strata'][b]['tokens'] for b in BINS},
                    target_predictor_positions=[row['prompt_length']+int(j)-1 for j in row['target_offsets']],
                    reward=r['reward'], queries=queries)
            del native
            peak_pss = max(peak_pss, psutil.Process().memory_full_info().pss)
        points = [dict(q, traj_uid=uid) for uid in sorted(entries) for q in entries[uid]['queries']]
        assert len(points) == sum(sizes)
        assert len({(q['traj_uid'], q['packed_slot']) for q in points}) == len(points)
        old_tail = {(q['traj_uid'], q['packed_slot']) for q in saved['complete_observed_ratio_gt_2_tail']}
        assert {(q['traj_uid'], q['packed_slot']) for q in points if q['prediction_stratum'] in BINS[2:]} == old_tail
        # Length sorting changes execution order only. Every selected point is
        # scored once; four genuine query pairs use the original native B8 call.
        points.sort(key=lambda q:(entries[q['traj_uid']]['selected_tokens'], q['traj_uid'], q['packed_slot']))
        midpoint = (len(points)+1)//2
        chunks = []
        for number, chunk_points in enumerate((points[:midpoint], points[midpoint:])):
            batches = []
            for start in range(0, len(chunk_points), 4):
                actual_queries = chunk_points[start:start+4]
                uids = [q['traj_uid'] for q in actual_queries]
                width = max(entries[uid]['selected_tokens'] for uid in uids)
                predictor_rows = {j for uid in uids for j in entries[uid]['target_predictor_positions']}
                batches.append(dict(uids=uids, actual_rows=len(uids), padded_width=width,
                    native_paired_forwards=1, probability_sample_queries=actual_queries,
                    head_union_predictor_rows=len(predictor_rows)))
            if len(batches) % 2:
                # Equal FSDP call count. This is an identity control, never a
                # sampled token or a confusion-matrix observation.
                last = batches[-1]
                batches.append(dict(last, actual_rows=0, probability_sample_queries=[]))
            chunks.append(dict(chunk=number, queries=len(chunk_points), batches=batches,
                native_B8_calls_per_rank=len(batches)//2,
                note='Original diagnostic worker 1500-second budget remains; no automatic retry or outcome-dependent expansion.'))
        result['tasks'][task] = dict(declared_trajectories=len(records), completed_trajectories=len(completed),
            missing_uids=sorted(set(records)-set(completed)), initial_states=len(groups),
            completed_frame_sources=sum(totals), strata=[dict(name=b, population=N, sample=n,
                inclusion_probability=n/N if N else None) for b, N, n in zip(BINS, totals, sizes)],
            sampled_sources=len(points), sampled_trajectories=len({q['traj_uid'] for q in points}),
            sampled_states=len({entries[q['traj_uid']]['initial_state_sha256'] for q in points}),
            state_completed_trajectories=state_completed, entries=list(entries.values()), chunks=chunks)
    result['checked_native_files'] = list(checked_files.values())
    result['runtime'] = dict(seconds=time.perf_counter()-started, unix=time.time(),
        peak_sampled_pss_bytes=peak_pss, cuda_initialized=torch.cuda.is_initialized())
    assert not result['runtime']['cuda_initialized']
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output=ref(output), runtime=result['runtime'],
        tasks={t:{k:v[k] for k in ('completed_frame_sources','sampled_sources','sampled_trajectories','sampled_states','strata')} for t,v in result['tasks'].items()})))


if __name__ == '__main__':
    main()

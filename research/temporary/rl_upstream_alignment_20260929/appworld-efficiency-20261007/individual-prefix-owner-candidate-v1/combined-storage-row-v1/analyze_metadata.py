"""Stdlib accounting of frozen 88-row lease producer schedules; no model."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = next(p for p in ROOT.parents if (p/'experiments/rl/PLAN.md').exists())
GEOMETRY = REPO/'experiments/rl/results_actual_prefix_request_geometry_20261004.json'
RUNTIME = ROOT.parents[2]/'native-prefix-boundary-rows-20261007-v1/runtime-v1/runtime-summary.json'


def identity(path):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def account(rows, individual, bytes_per_capture_token, bytes_per_gdn_row, gdn_layers):
    batches = [rows[i:i+4] for i in range(0, len(rows), 4)]
    needed = {}
    for batch in batches:
        common = min(r['source_start'] for r in batch)//64*64
        if common:
            for row in batch:
                needed.setdefault(row['traj_uid'], set()).add(
                    row['source_start']//64*64 if individual else common)
    canonical = {}
    for row in rows:
        uid = row['traj_uid']
        if uid in needed and (uid not in canonical or row['prompt_tokens'] > canonical[uid]['prompt_tokens']):
            canonical[uid] = row
    records = list(canonical.items())
    indices = sorted(range(len(records)), key=lambda i: max(needed[records[i][0]]))
    # Both actual ranks have 18 canonical rows. Original MAX therefore gives
    # five B4 captures. Official DataProto padding appends first two indices.
    assert len(records) == 18
    indices += indices[:(-len(indices)) % 4]
    last_slots = {records[row][0]: slot for slot, row in enumerate(indices)}
    captures = []
    for offset in range(0, len(indices), 4):
        indices_batch = indices[offset:offset+4]
        lengths = sorted({n for row in indices_batch for n in needed[records[row][0]]})
        final_rows = [i for i, row in enumerate(indices_batch)
                      if last_slots[records[row][0]] == offset+i]
        consumed = sum(len(needed[records[indices_batch[i]][0]]) for i in final_rows)
        # All five artifacts remain referenced after sources last-write.
        assert final_rows
        captures.append(dict(artifact=offset//4, capture_width=max(lengths),
            boundaries=lengths, canonical_indices=indices_batch,
            original_rows_still_referenced=final_rows,
            stored_gdn_rows_per_layer=4*len(lengths), consumed_gdn_rows_per_layer=consumed,
            unconsumed_gdn_rows_per_layer=4*len(lengths)-consumed,
            full_batch_state_segment_calls_per_gdn_layer=len(lengths)-1))
    tokens = sum(4*c['capture_width'] for c in captures)
    stored = sum(c['stored_gdn_rows_per_layer'] for c in captures)
    consumed = sum(c['consumed_gdn_rows_per_layer'] for c in captures)
    common = tokens*bytes_per_capture_token
    return dict(canonical_histories=len(records), B4_native_capture_calls=len(captures),
        consumer_B4_calls=len(batches), distinct_uid_boundary_pairs=sum(map(len, needed.values())),
        distinct_boundary_lengths=len(set.union(*needed.values())),
        capture_union_boundaries=sum(len(c['boundaries']) for c in captures),
        capture_token_slots=tokens, stored_gdn_rows_per_layer=stored,
        consumed_gdn_rows_per_layer=consumed, unconsumed_gdn_rows_per_layer=stored-consumed,
        full_batch_state_segment_calls_all_gdn_layers=sum(
            c['full_batch_state_segment_calls_per_gdn_layer'] for c in captures)*gdn_layers,
        full_storage_bytes_from_actual_owner_geometry=common+stored*gdn_layers*bytes_per_gdn_row,
        consumed_only_storage_bytes_from_actual_owner_geometry=common+consumed*gdn_layers*bytes_per_gdn_row,
        captures=captures)


def main():
    geometry = json.loads(GEOMETRY.read_bytes())
    runtime = json.loads(RUNTIME.read_bytes())
    assert identity(GEOMETRY)['sha256'] == 'a61c185751434edb4d2fbbac4305e3dbbcaf53c81f705693cacc9a75d42f97a0'
    assert identity(RUNTIME)['sha256'] == '277e697051c23f2a9ab741676ea91cf16c08c0216704ccd470158eb7fad0205d'
    ranks = []
    for index, source in enumerate(geometry['ranks']):
        rows = source['rows']
        assert [r['sorted_row'] for r in rows] == list(range(88))
        assert all(rows[i]['context_tokens'] <= rows[i+1]['context_tokens'] for i in range(87))
        original = runtime['ranks'][index]['inventories']['OFF']
        assert original['original_request_sha256'] == source['sha256']
        total = original['totals']
        gdn_layers = 24
        capture_tokens = original['original_preparation']['capture_token_slots']
        per_token = (total['input_ids_bytes']+total['fa_keys_bytes']+total['fa_values_bytes'])//capture_tokens
        per_gdn_row = (total['gdn_conv_bytes']+total['gdn_state_bytes'])//total['gdn_stored_boundary_rows']
        scalar = account(rows, False, per_token, per_gdn_row, gdn_layers)
        row = account(rows, True, per_token, per_gdn_row, gdn_layers)
        assert scalar['full_storage_bytes_from_actual_owner_geometry'] == total['distinct_live_storage_bytes']
        assert scalar['consumed_gdn_rows_per_layer'] == original['distinct_consumed_artifact_boundary_rows']
        assert scalar['capture_token_slots'] == capture_tokens
        keys = ('capture_union_boundaries', 'capture_token_slots', 'stored_gdn_rows_per_layer',
                'consumed_gdn_rows_per_layer', 'unconsumed_gdn_rows_per_layer',
                'full_batch_state_segment_calls_all_gdn_layers',
                'full_storage_bytes_from_actual_owner_geometry',
                'consumed_only_storage_bytes_from_actual_owner_geometry')
        ranks.append(dict(rank=index, original_requests=dict(path=source['path'], sha256=source['sha256']),
            byte_geometry=dict(gdn_layers=gdn_layers, gdn_conv_plus_state_bytes_per_row=per_gdn_row,
                full_batch_FA_KV_plus_inputID_bytes_per_capture_token_slot=per_token),
            scalar=scalar, individual=row, row_minus_scalar={k:row[k]-scalar[k] for k in keys},
            original_scalar_inventory_exactly_reproduced=True))
    inputs = [identity(GEOMETRY), identity(RUNTIME), identity(ROOT/'prepare_sources.py'),
              identity(ROOT/'source-manifest.json')]
    for name in ('native_prefix_leases.py', 'qwen35_native_prefix_artifacts.py'):
        inputs.extend(identity(ROOT/directory/name) for directory in ('baseline', 'candidate'))
    result = dict(scope='Frozen actual 176-request metadata, original producer grouping/padding/last-write accounting only.',
        script=identity(Path(__file__)), inputs=inputs, ranks=ranks,
        derivation='Original stable canonical sort by max needed boundary; original B4 grouping; official pad appends first two of 18 histories; final sources[uid] last-write determines consumed rows.',
        limits=['This is not a new bank capture, numerical acceptance, measured time or formal deployment.',
            'The row candidate adds boundary union and wider original capture work; storage compression removes saved unused rows but does not reduce those native forwards or state-segment calls.',
            'State segments carry original full-batch FP32 previous_state and process each segment once; boundary count is not a repeated full-prefix count or a wall-time multiplier.',
            'Byte estimates use actual same-model scalar inventory geometry, including unchanged FA KV and input IDs; Python metadata/allocator/cache/RSS are excluded.',
            'Original saved query predates the accepted clock wording, but prompt/source boundaries are literal unchanged fields; no query or tokens are reconstructed.'])
    output = ROOT/'metadata-counts.json'
    output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps(dict(output=str(output), sha256=identity(output)['sha256'],
        ranks=[dict(rank=r['rank'], differences=r['row_minus_scalar']) for r in ranks])))


if __name__ == '__main__':
    main()

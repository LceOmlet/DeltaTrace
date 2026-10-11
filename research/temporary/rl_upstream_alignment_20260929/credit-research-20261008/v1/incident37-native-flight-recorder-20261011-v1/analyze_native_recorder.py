"""Summarize the unmodified official analyzer and exact native metadata.

This analysis tests collective metadata alignment, never parameter contents
or asynchronous storage safety. It creates no numerical acceptance rule.
"""
from collections import Counter
from pathlib import Path
import hashlib
import json
import pickle
import sys
import time

BASE = Path(__file__).resolve().parent.parent
RAW = BASE / 'direct-credit-records-20261009-v1'
OWNER = BASE / 'incident37-flight-recorder-capability-20261011-v1/pytorch-v2.8.0-analyzer'
RESULT = RAW / 'native-flight-recorder-downloaded-20261011-v1/result'

def main():
    sys.path.insert(0, str(OWNER))
    database_path = RAW / 'native-flight-recorder-official-analyzer-20261011-v1.database.pkl'
    _, database = pickle.loads(database_path.read_bytes())
    traces = [pickle.loads((RESULT / f'rank{r}-native-flight-recorder.pkl').read_bytes()) for r in (0, 1)]
    fields = ['process_group', 'collective_seq_id', 'p2p_seq_id', 'op_id',
              'profiling_name', 'input_sizes', 'input_dtypes', 'output_sizes', 'output_dtypes', 'state']
    entries = [d['entries'] for d in traces]
    maps = [{(e['process_group'][0], e['collective_seq_id']): e for e in rows} for rows in entries]
    common = sorted(maps[0].keys() & maps[1].keys())
    mismatches = [dict(key=key, fields=[f for f in fields if maps[0][key].get(f) != maps[1][key].get(f)])
                  for key in common if any(maps[0][key].get(f) != maps[1][key].get(f) for f in fields)]
    ranks = []
    for rank, (trace, rows) in enumerate(zip(traces, entries)):
        actor = json.loads((RESULT / f'rank{rank}.json').read_bytes())
        roots = [e for e in rows if e['profiling_name'] == 'nccl:_all_gather_base'
                 and e['input_sizes'] == [[508561408]] and e['output_sizes'] == [[1017122816]]]
        tail_roots = roots[-3:]
        since = tail_roots[0]['collective_seq_id'] if len(tail_roots) == 3 else None
        ranks.append(dict(rank=rank, native_version=trace['version'],
            entry_count=len(rows), first_record_id=rows[0]['record_id'], last_record_id=rows[-1]['record_id'],
            sequence_range=[rows[0]['collective_seq_id'], rows[-1]['collective_seq_id']],
            ring_capacity=2000, ring_overwrote_earlier_entries=rows[0]['record_id'] > 0,
            states=dict(Counter(e['state'] for e in rows)),
            operation_counts=dict(Counter(e['profiling_name'] for e in rows)),
            last_three_root_geometry_collectives=tail_roots,
            decoder_geometry_collectives_since_first_of_these_roots=[e for e in rows
                if since is not None and e['collective_seq_id'] >= since
                and e['profiling_name'] == 'nccl:_all_gather_base'
                and e['input_sizes'] == [[109547872]] and e['output_sizes'] == [[219095744]]],
            actor_phase=actor.get('phase'), microbatches=actor.get('microbatches'),
            actual_optimizer_steps=actor.get('actual_optimizer_steps')))
    out = dict(unix=time.time(), ranks=ranks,
        native_metadata_cross_rank=dict(compared_fields=fields, common_sequence_count=len(common),
            rank0_only=sorted(maps[0].keys()-maps[1].keys()), rank1_only=sorted(maps[1].keys()-maps[0].keys()),
            mismatches=mismatches),
        original_official_analyzer=dict(owner='pytorch/pytorch v2.8.0 tools/flight_recorder',
            source_receipt=str(OWNER / 'source-receipt.json'),
            command_receipt=str(RAW / 'native-flight-recorder-official-analyzer-20261011-v1.json'),
            database=dict(path=str(database_path), bytes=database_path.stat().st_size,
                sha256=hashlib.sha256(database_path.read_bytes()).hexdigest()),
            groups=len(database.groups), rank_calls=len(database.ncclcalls),
            collectives=len(database.collectives), pass_check_true=sum(c.pass_check for c in database.collectives),
            failed_collectives=[c._asdict() for c in database.collectives if not c.pass_check], modified_owner_files=0),
        production_changes=0,
        limits=['The ring covers only its retained suffix; no claim about overwritten earlier collectives.',
                'Matched sizes, dtype and sequence do not verify send/receive contents, memcpy completion, or storage lifetime.',
                'The official analyzer version accepts native trace version 2.9; this is metadata checking, not a numerical tolerance.',
                'Root classification uses the verified norm/head packed geometry; decoder-size matches include multiple GDN layers and cannot identify layer0 by size alone. The raw trace contains no module FQN.',
                'A successful analyzer exit does not imply the actor update is correct; this execution reproduced nonfinite gradients.'])
    path = RAW / 'native-flight-recorder-analysis-20261011-v1.json'
    path.write_bytes((json.dumps(out, indent=2) + '\n').encode())
    print(json.dumps(dict(path=str(path), ranks=[{k:v for k,v in r.items() if k not in
        ('last_three_root_geometry_collectives','decoder_geometry_collectives_since_first_of_these_roots')}
        for r in ranks], metadata=out['native_metadata_cross_rank'],
        official={k:v for k,v in out['original_official_analyzer'].items() if k not in ('database','source_receipt','command_receipt')},
        root_sequences=[[e['collective_seq_id'] for e in r['last_three_root_geometry_collectives']] for r in ranks])))

if __name__ == '__main__':
    main()

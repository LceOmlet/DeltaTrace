"""Read already published native DT timings; never run a model or profiler."""
from pathlib import Path
import json
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'rl_dtpo_v9_clip_u_20261019_v1'))
from transport import run_python

report = json.loads(run_python(r'''
from pathlib import Path
import collections, hashlib, json, time
R = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
A = R / 'receipts/dtpo-v9-negative-clip2-postgroup-variance-20261019-v4'
first = json.loads((A/'first-complete-iteration-source-snapshot.json').read_bytes())
session = Path(first['jobs'][0]['ray_session'])/'logs'
rows, scopes, errors = [], [], []
markers = {'[native_DT_existing_timing] ': rows,
           '[native_DT_existing_timing_scope] ': scopes,
           '[native_DT_existing_timing_observer_error] ': errors}
for path in sorted(session.glob('worker-*.out')):
    for number, line in enumerate(path.read_text(errors='replace').splitlines(), 1):
        for marker, destination in markers.items():
            if marker not in line:
                continue
            tail = line.split(marker, 1)[1]
            if destination is errors:
                decoded = {'error': tail}
            else:
                decoded, _ = json.JSONDecoder().raw_decode(tail)
            destination.append(dict(path=str(path), line=number, value=decoded))

def summarize(records):
    host, stream, nested_host, nested_stream = collections.Counter(), collections.Counter(), 0., 0.
    wall = 0.
    for row in records:
        value = row['value']
        wall += value['finished'] - value['started']
        for call in value.get('calls') or []:
            kind = call['kind']
            is_nested = kind.startswith(('restore_mixer_captures_', 'public_FA_LSE_'))
            group = ('native_replay' if kind.startswith('native_replay_') else
                     'finite_decoder' if kind.startswith('finite_decoder_') else kind)
            h = call.get('host_enqueue_seconds', call.get('seconds'))
            s = call.get('stream_elapsed_seconds')
            if is_nested:
                nested_host += h or 0.
                nested_stream += s or 0.
            else:
                if h is not None: host[group] += h
                if s is not None: stream[group] += s
    return dict(native_B4_calls=len(records), original_attribute_wall_seconds=wall,
                top_level_host_enqueue_seconds=dict(host),
                top_level_stream_elapsed_seconds=dict(stream),
                nested_host_seconds_excluded_from_sum=nested_host,
                nested_stream_seconds_excluded_from_sum=nested_stream,
                wall_minus_top_level_host_enqueue_seconds=wall-sum(host.values()))

by_rank = collections.defaultdict(list)
for row in rows: by_rank[str(row['value']['rank'])].append(row)
complete = {row['value']['rank'] for row in scopes} == set(range(8)) and len(scopes) == 8 and not errors and all(
    row['value']['restored_original_methods'] and row['value']['additional_DT_calls'] == 0
    for row in scopes)
if complete:
    assert all(len(by_rank[str(row['value']['rank'])]) == row['value']['native_minibatches']
               for row in scopes)
    assert all(row['value']['original_observer_is_none'] for row in rows)
report = dict(collected=time.time(), status='completed' if complete else 'partial_existing_native_metadata',
    run=json.loads((A/'first-complete-iteration-efficiency-audit.json').read_bytes())['current_run'],
    source_snapshot=str(A/'first-complete-iteration-source-snapshot.json'),
    observed_B4_calls=len(rows), restored_rank_scopes=len(scopes), errors=errors,
    per_rank={rank:summarize(records) for rank,records in sorted(by_rank.items())},
    native_timing_records=rows, restored_scopes=scopes,
    interpretation=[
        'Original finite-runner metadata, from one naturally scheduled training readout per rank.',
        'Host enqueue includes blocking native calls; it is not exclusive CPU computation.',
        'Stream elapsed includes idle launch gaps; it is not active kernel time or a FLOPS fraction.',
        'Nested mixer restore/FA timing is excluded from the top-level sum.',
        'Attribute wall excludes readout-level B4 host-cache release and PPO updates.',
        'Do not add rank times to driver wall times or compare different effective DT work as pure kernel speed.'
    ], additional_calls=dict(model=0,DT=0,optimizer=0,environment=0,profiling=0))
if rows or scopes or errors:
    target=A/'existing-DT-native-stage-timings.json'
    target.write_text(json.dumps(report,indent=2)+'\n')
    report['remote_receipt']=str(target)
print(json.dumps(report))
'''))

if report['observed_B4_calls'] or report['restored_rank_scopes'] or report['errors']:
    target = Path('experiments/rl/results_dtpo_v9_negative2_existing_DT_timings_20261019_v4.json')
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
brief = {key:report[key] for key in
    ('collected','status','observed_B4_calls','restored_rank_scopes','errors')}
brief['per_rank'] = {rank:dict(B4=value['native_B4_calls'],
    attribute_wall=value['original_attribute_wall_seconds'],
    host_enqueue={key:value['top_level_host_enqueue_seconds'].get(key)
        for key in ('native_root_with_CPU_checkpoints','native_replay','finite_decoder')})
    for rank,value in report['per_rank'].items()}
print(json.dumps(brief, ensure_ascii=False))

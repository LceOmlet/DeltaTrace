"""Reuse the original existing-DT metadata collector for native counters."""
from pathlib import Path

owner = Path(__file__).with_name('collect_existing_DT_timings.py')
code = owner.read_text()
code = code.replace('native_DT_existing_timing', 'native_DT_existing_host_counters')
code = code.replace('existing-DT-native-stage-timings.json', 'existing-DT-native-host-counters.json')
code = code.replace('results_dtpo_v9_negative2_existing_DT_timings_20261019_v4.json',
                    'results_dtpo_v9_negative2_existing_DT_host_counters_20261019_v4.json')
anchor = 'if rows or scopes or errors:'
addition = '''for scope in scopes:
    value=scope['value']
    rank=str(value['rank'])
    before=value['native_host_counters_before']
    after=value['native_host_counters_after']
    if 'host_alloc_time.total' not in before or 'host_alloc_time.total' not in after:
        report['per_rank'][rank]['native_host_counters_unavailable']={'before':before,'after':after}
        continue
    report['per_rank'][rank]['native_readout_wall_seconds']=value['finished']-value['started']
    report['per_rank'][rank]['native_host_allocator_seconds']={
        'allocation':(after['host_alloc_time.total']-before['host_alloc_time.total'])/1e6,
        'free':(after['host_free_time.total']-before['host_free_time.total'])/1e6}
    report['per_rank'][rank]['native_host_allocator_calls']={
        'allocation':after['num_host_alloc']-before['num_host_alloc'],
        'free':after['num_host_free']-before['num_host_free']}
report['interpretation'].append('Native host time counters are cumulative allocator call time; overlap with queued GPU work and rank synchronization prevents equating the sum to recoverable wall time.')
'''
assert code.count(anchor) == 1
code = code.replace(anchor, addition + '\n' + anchor)
anchor = 'print(json.dumps(brief, ensure_ascii=False))'
assert code.count(anchor) == 1
code = code.replace(anchor, '''for rank,value in report['per_rank'].items():
    brief['per_rank'][rank]['readout_wall']=value.get('native_readout_wall_seconds')
    brief['per_rank'][rank]['allocator_seconds']=value.get('native_host_allocator_seconds')
''' + anchor)
exec(compile(code, str(owner), 'exec'), globals())

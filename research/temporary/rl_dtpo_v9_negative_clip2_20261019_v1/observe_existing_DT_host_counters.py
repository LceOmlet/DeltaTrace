"""Add native host allocator counters to the existing one-readout observer.

Only the next naturally scheduled readout is observed. The original call,
arguments, returned objects and automatic method restoration are preserved.
No tensor, model, DT, optimizer, synchronization or sampling call is added.
"""
import hashlib
from pathlib import Path

owner = Path(__file__).with_name('observe_existing_DT_metadata.py')
assert hashlib.sha256(owner.read_bytes()).hexdigest() == '20c0de1e0d2b7f89082e372fbdac292cab0bea41866612f8f0555d3afe94649c'
code = owner.read_text()
code = code.replace('AUDIT=ROOT/', 'DT_STAGE_OBSERVATION_AUDIT=ROOT/')
code = code.replace("(AUDIT/'existing-DT", "(DT_STAGE_OBSERVATION_AUDIT/'existing-DT")
code = code.replace('_existing_DT_timing_scope_installed', '_existing_DT_host_counters_scope_installed')
code = code.replace('existing-DT-metadata-observer-completed.json', 'existing-DT-host-counters-observer-completed.json')
code = code.replace('native_DT_existing_timing', 'native_DT_existing_host_counters')
code = code.replace('code=wrapper.read_text()', "code=wrapper.read_text().replace('v9-negative2-eight-GPU-initial-v1','existing-DT-host-counters-v1')")

anchor = '        def observe_one_existing_readout(*args,**kwargs):'
assert code.count(anchor) == 1
code = code.replace(anchor, '''        def native_host_counters():
            import torch
            try:
                counters=torch.cuda.memory.host_memory_stats()
                return {key:value for key,value in counters.items()
                    if key.startswith(('host_alloc_time.','host_free_time.'))
                    or key in ('num_host_alloc','num_host_free','allocated_bytes.all.current',
                        'reserved_bytes.all.current','active_bytes.all.current')}
            except Exception as error:
                return {'observation_unavailable':type(error).__name__}
''' + anchor)
replacements = {
    '            count=[0]': '            scope_before=native_host_counters()\n            count=[0]',
    '                began=time.time()': '                before=native_host_counters()\n                began=time.time()',
    '                ended=time.time()': '                ended=time.time()\n                after=native_host_counters()',
    '                        started=began,finished=ended,paired_shape=list(paired.shape),':
    "                        started=began,finished=ended,paired_shape=list(paired.shape),\n                        native_host_counters_before=before,native_host_counters_after=after,\n                        host_counter_time_unit='microseconds as documented by original Torch owner',",
    '                        started=started,finished=time.time(),native_minibatches=count[0],':
    '                        started=started,finished=time.time(),native_minibatches=count[0],\n                        native_host_counters_before=scope_before,native_host_counters_after=native_host_counters(),',
}
for old, new in replacements.items():
    assert code.count(old) == 1, old
    code = code.replace(old, new)
exec(compile(code, str(owner), 'exec'), globals())

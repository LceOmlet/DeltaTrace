"""Read all eight native B4/16384 capacity receipts with the existing reader."""
from pathlib import Path

owner = Path(__file__).with_name('collect_root_pageable_capacity.py')
code = owner.read_text()
for old, new in (
    ('root-pageable', 'resident-16384'),
    ('root_capacity', 'resident_16384_capacity'),
    ('32768', '16384'),
    ('if d[\'rank\'] in (0,1):\n            assert d[\'input_shape\']==[4,16384]\n            assert d[\'native_report\'][\'actual_context_lengths\']==[16384]*4\n            assert d[\'native_report\'][\'minibatch_size\']==4',
     "assert d['input_shape']==[4,16384]\n        assert d['native_report']['actual_context_lengths']==[16384]*4\n        assert d['native_report']['minibatch_size']==4"),
    ('Original B4/exact16384 on ranks0/1, small original B4 on ranks2..7; not simultaneous eight-rank all-32k capacity',
     'All eight ranks simultaneously use original artificial B4/exact16384 and original GPU replay storage; not a 32768 GPU-storage capacity claim'),
):
    assert old in code, old
    code = code.replace(old, new)
exec(compile(code, str(owner), 'exec'), globals())

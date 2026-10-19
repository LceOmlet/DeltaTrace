"""Use one existing native capacity call on all eight loaded workers.

The existing fixture/producer owns Q/V/A and all DT arithmetic. This selects
only its original GPU replay-storage flag and restores the production route.
"""
import hashlib
from pathlib import Path

owner = Path(__file__).with_name('check_root_pageable_native_capacity.py')
assert hashlib.sha256(owner.read_bytes()).hexdigest() == 'f4db62988a53f7364bd2cff2506f9ea98e886079a6fde7b1c7025e80147ba499'
code = owner.read_text()
for old, new in (
    ('root-pageable', 'resident-16384'),
    ('root_pageable', 'resident_16384'),
    ('32768', '16384'),
    ("assert (runner.pin_root_host,runner.pin_replay_host)==(True,True)",
     "assert (runner.pin_root_host,runner.pin_replay_host,runner.offload_replay_mixer)==(False,True,True)"),
    ('flags=(runner.pin_root_host,runner.pin_replay_host)',
     'flags=(runner.pin_root_host,runner.pin_replay_host,runner.offload_replay_mixer)'),
    ("fixture_kind='capacity' if actor.rank in (0,1) else 'small'", "fixture_kind='capacity'"),
    ('if actor.rank in (0,1):assert state[\'input_shape\']==[4,16384]',
     "assert state['input_shape']==[4,16384]"),
    ('runner.pin_root_host=False;runner.pin_replay_host=True',
     'runner.pin_root_host=False;runner.pin_replay_host=True;runner.offload_replay_mixer=False'),
    ("if actor.rank in (0,1):\n                    assert state['native_report']['actual_context_lengths']==[16384]*4\n                    assert state['native_report']['minibatch_size']==4",
     "assert state['native_report']['actual_context_lengths']==[16384]*4\n                assert state['native_report']['minibatch_size']==4"),
    ('runner.pin_root_host,runner.pin_replay_host=flags',
     'runner.pin_root_host,runner.pin_replay_host,runner.offload_replay_mixer=flags'),
    ('(runner.pin_root_host,runner.pin_replay_host)==flags',
     '(runner.pin_root_host,runner.pin_replay_host,runner.offload_replay_mixer)==flags'),
    ('B4/rank exact16384 on ranks0/1; other ranks participate using original small fixture. Not all-eight-rank 32k capacity or production speed.',
     'All eight ranks use original artificial B4/exact16384 with GPU replay storage. Not a 32768 GPU-storage capacity or a production speed test.'),
    ('One next existing DT RPC; two ranks B4/16384, other six original small B4; no new model process or optimizer call',
     'One next existing DT RPC; eight simultaneous B4/exact16384 calls; no new model process or optimizer call'),
):
    assert old in code, old
    code = code.replace(old, new)
exec(compile(code, str(owner), 'exec'), globals())

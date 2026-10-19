"""Check only root checkpoint storage on the fifth natural native B4.

The measured both-pageable candidate slowed the long replay/finite stages.
Keep replay storage pinned; compose the same bounded native-owner comparison.
Release the baseline's unused host buffers through the existing B4 owner
before timing the candidate, as at the normal B4 boundary. No active buffers,
device/compiler caches, training outputs, optimizer or rollout are changed.
"""
import hashlib
from pathlib import Path

owner = Path(__file__).with_name('check_existing_DT_pageable_pair.py')
assert hashlib.sha256(owner.read_bytes()).hexdigest() == 'e9a63d4f72b05d567c2d36255fca2fa5d8fa447d22cdcbb114dc314ce8995ec7'
code = owner.read_text()
for old, new in (
    ('_existing_DT_pageable_pair_scope_installed', '_existing_DT_root_pageable_pair_scope_installed'),
    ('native_DT_existing_pageable_pair', 'native_DT_existing_root_pageable_pair'),
    ('existing-DT-pageable-pair', 'existing-DT-root-pageable-pair'),
    ('count[0]==2', 'count[0]==4'),
    ('int(count[0]>=3)', 'int(count[0]>=5)'),
    ('third natural B4', 'fifth natural B4, root storage only'),
    ('candidate_flags=[False,False]', 'candidate_flags=[False,True]'),
    ('runner.pin_replay_host=False', 'runner.pin_replay_host=True'),
):
    assert old in code, old
    code = code.replace(old, new)

old = '                        assert flags==(True,True)'
assert code.count(old) == 1
code = code.replace(old, old + '''
                        from native_host_memory import release_unused_host_cache
                        release_started=time.time()
                        release_unused_host_cache('validation_same_input_between_DT_pairs')
                        pair_info['between_pair_owner_release_seconds']=time.time()-release_started
                        pair_info['before_candidate_host_memory_stats']=torch.cuda.memory.host_memory_stats()
                        candidate_started=time.time()
                        pair_info['candidate_started_unix']=candidate_started''')
old = "                        pair_info['candidate_wall_seconds']=time.time()-candidate_started"
assert code.count(old) == 1
code = code.replace(old, old + '''
                        pair_info['candidate_finished_unix']=time.time()
                        pair_info['after_candidate_host_memory_stats']=torch.cuda.memory.host_memory_stats()''')
exec(compile(code, str(owner), 'exec'), globals())

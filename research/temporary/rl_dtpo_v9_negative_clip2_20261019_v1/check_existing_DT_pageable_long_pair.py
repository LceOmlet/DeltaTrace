"""Run the same owner comparison on the fifth natural B4, the observed long tail.

Only the bounded validation call index and receipt identity change. If fewer
than five original batches exist, this scope adds no DT call. Production
continues to receive its original result and options are restored.
"""
import hashlib
from pathlib import Path

owner = Path(__file__).with_name('check_existing_DT_pageable_pair.py')
assert hashlib.sha256(owner.read_bytes()).hexdigest() == 'e9a63d4f72b05d567c2d36255fca2fa5d8fa447d22cdcbb114dc314ce8995ec7'
code = owner.read_text()
for old,new in (
    ('_existing_DT_pageable_pair_scope_installed','_existing_DT_pageable_long_pair_scope_installed'),
    ('native_DT_existing_pageable_pair','native_DT_existing_pageable_long_pair'),
    ('existing-DT-pageable-pair','existing-DT-pageable-long-pair'),
    ('count[0]==2','count[0]==4'),
    ('int(count[0]>=3)','int(count[0]>=5)'),
    ('third natural B4','fifth natural B4'),
):
    assert old in code, old
    code=code.replace(old,new)
exec(compile(code,str(owner),'exec'),globals())

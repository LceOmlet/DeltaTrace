"""Reuse the same source-preserving collector for the long native B4 pair."""
from pathlib import Path

owner=Path(__file__).with_name('collect_existing_DT_pageable_pair.py')
code=owner.read_text()
code=code.replace('native_DT_existing_pageable_pair','native_DT_existing_pageable_long_pair')
code=code.replace('existing-DT-pageable-pair','existing-DT-pageable-long-pair')
code=code.replace('results_dtpo_v9_negative2_pageable_pair_20261019_v4.json',
                  'results_dtpo_v9_negative2_pageable_long_pair_20261019_v4.json')
exec(compile(code,str(owner),'exec'),globals())

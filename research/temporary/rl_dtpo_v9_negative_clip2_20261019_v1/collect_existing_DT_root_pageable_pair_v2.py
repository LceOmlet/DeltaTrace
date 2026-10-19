"""Read the corrected comparison without replacing the failed v1 receipt."""
from pathlib import Path

owner = Path(__file__).with_name('collect_existing_DT_root_pageable_pair.py')
code = owner.read_text()
code = code.replace('native_DT_existing_root_pageable_pair', 'native_DT_existing_root_pageable_pair_v2')
code = code.replace('existing-DT-root-pageable-pair', 'existing-DT-root-pageable-pair-v2')
code = code.replace('results_dtpo_v9_negative2_root_pageable_pair_20261019_v4.json',
                    'results_dtpo_v9_negative2_root_pageable_pair_v2_20261019_v4.json')
exec(compile(code, str(owner), 'exec'), globals())

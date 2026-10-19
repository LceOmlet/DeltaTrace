"""Correct the validation adapter's package path before any candidate call.

Keep the failed v1 receipt intact. Import the existing resource owner during
installation, so a missing integration fails before waiting for a DT batch.
"""
import hashlib
from pathlib import Path

owner = Path(__file__).with_name('check_existing_DT_root_pageable_pair.py')
assert hashlib.sha256(owner.read_bytes()).hexdigest() == '35d75864115e6e90dcf3b6c6a439e88214a7edd7ca1aa6b62f8b96c879dada04'
code = owner.read_text()
code = code.replace('from native_host_memory import release_unused_host_cache',
                    'from verl.utils.native_host_memory import release_unused_host_cache')
code = code.replace('_existing_DT_root_pageable_pair_scope_installed',
                    '_existing_DT_root_pageable_pair_v2_scope_installed')
code = code.replace('native_DT_existing_root_pageable_pair',
                    'native_DT_existing_root_pageable_pair_v2')
code = code.replace('existing-DT-root-pageable-pair', 'existing-DT-root-pageable-pair-v2')
anchor = "exec(compile(code, str(owner), 'exec'), globals())"
assert code.count(anchor) == 1
# At this seam `code` is the existing comparison adapter. Its owner text is
# the existing metadata observer, whose observation body owns installation.
preflight = '''    from verl.utils.native_host_memory import release_unused_host_cache
    import inspect
    resource_source=Path(inspect.getfile(release_unused_host_cache))
    result['root_pair_resource_owner_preflight']={
        'path':str(resource_source),
        'SHA256':hashlib.sha256(resource_source.read_bytes()).hexdigest(),
        'callable':callable(release_unused_host_cache),
        'called_during_installation':False}
    installed=False'''
insertion = "code = owner.read_text()\ncode = code.replace('    installed=False', " + repr(preflight) + ")"
code = code.replace(anchor,
    "assert code.count('code = owner.read_text()') == 1\n"
    "code = code.replace('code = owner.read_text()', " + repr(insertion) + ")\n" + anchor)
exec(compile(code, str(owner), 'exec'), globals())

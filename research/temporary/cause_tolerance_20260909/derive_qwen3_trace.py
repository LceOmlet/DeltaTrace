"""Add only read-only observer calls to our pinned finite propagation."""
import ast
import difflib
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sha = lambda b: hashlib.sha256(b).hexdigest()
manifest = json.loads((ROOT / 'deltatrace/accelerated/deferred_sources.json').read_bytes())
files = manifest['files']
outputs = {}
for name in ('finite', 'pair'):
    relative = f'deltatrace/accelerated/qwen3/qwen3_deferred_{name}.py'
    raw = (ROOT / relative).read_bytes()
    assert sha(raw) == files[relative]
    source = raw.decode().replace('\r\n', '\n')
    if name == 'finite':
        replacements = [
            ('validation=None):', 'validation=None,observer=None):'),
            ('        del m_final\n', '        if observer is not None:\n            observer.final(m, m_final, after)\n        del m_final\n'),
            ('            m_mid = m\n', '            m_layer_output = m if observer is not None else None\n            m_mid = m\n'),
            ('            if progress is not None:\n',
             "            if observer is not None:\n                observer.decoder(li, raw1, m_layer_output, m, m_mid, (mqa, mka, mva), (m_x_gate, m_x_up))\n            if progress is not None:\n")]
    else:
        replacements = [
            ('from qwen3_deferred_finite import', 'from qwen3_trace_finite import'),
            ('finite_activity=None):', 'finite_activity=None,observer=None):'),
            ('finite_activity=finite_activity,validation=validation)', 'finite_activity=finite_activity,validation=validation,observer=observer)')]
    derived = source
    for old, new in replacements:
        assert derived.count(old) == 1, (name, old)
        derived = derived.replace(old, new)
    ast.parse(derived)
    # Undo only declared observer changes to prove every arithmetic statement is retained.
    restored = derived
    for old, new in reversed(replacements): restored = restored.replace(new, old)
    assert restored == source
    output = HERE / f'qwen3_trace_{name}.py'
    output.write_text(derived, newline='\n')
    outputs[output.name] = {'original_path': relative, 'original_sha256': sha(raw),
                            'derived_sha256': sha(output.read_bytes()),
                            'diff': ''.join(difflib.unified_diff(source.splitlines(True), derived.splitlines(True)))}
(HERE / 'qwen3_trace_derivation.json').write_text(json.dumps(outputs, indent=2)+'\n', newline='\n')
print(json.dumps({k: v['derived_sha256'] for k,v in outputs.items()}))

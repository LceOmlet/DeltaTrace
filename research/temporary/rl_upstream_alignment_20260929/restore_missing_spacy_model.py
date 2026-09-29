"""Repair the demonstrated missing WebShop model only; reuse the Python stack."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import urllib.request

root = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
audit = root / 'receipts/upstream-alignment-20260929'
name = 'en_core_web_sm-3.8.0-py3-none-any.whl'
url = 'https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/' + name
expected = '1932429db727d4bff3deed6b34cfc05df17794f4a52eeb26cf8928f7c1a0fb85'
wheel = root / 'repair-assets' / name
if importlib.util.find_spec('en_core_web_sm') is None:
    wheel.parent.mkdir(parents=True, exist_ok=True)
    if not wheel.exists():
        with urllib.request.urlopen(url, timeout=30) as response:
            wheel.write_bytes(response.read())
    assert hashlib.sha256(wheel.read_bytes()).hexdigest() == expected
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', '--report',
                    str(audit / 'spacy-model-install.json'), str(wheel)], check=True)
import spacy
nlp = spacy.load('en_core_web_sm')
result = dict(spacy=spacy.__version__, model=nlp.meta['version'],
              tokens=[token.text for token in nlp('A red shopping bag.')],
              official_release='https://github.com/explosion/spacy-models/releases/tag/en_core_web_sm-3.8.0',
              wheel_sha256=expected, wheel=str(wheel), python=sys.executable)
(audit / 'spacy-model-restored.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result), flush=True)

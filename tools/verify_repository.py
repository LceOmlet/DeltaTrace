"""Verify backed-up bytes and parse templates without executing model code."""
import ast,hashlib,json,subprocess
from pathlib import Path,PurePosixPath
ROOT=Path(__file__).resolve().parents[1]
manifest=json.loads((ROOT/'evidence/export_manifest.json').read_text(encoding='utf-8'))
for entry in manifest['artifacts']:
    relative=PurePosixPath(entry['path'])
    assert not relative.is_absolute() and '..' not in relative.parts
    raw=(ROOT/relative).read_bytes()
    assert len(raw)==entry['bytes'] and hashlib.sha256(raw).hexdigest()==entry['public_sha256']
    # Staged Git objects must preserve the same bytes on every platform.
    staged=subprocess.run(['git','show',':'+entry['path']],cwd=ROOT,capture_output=True)
    if staged.returncode==0:assert staged.stdout==raw,entry['path']
    if entry['byte_identical']:assert entry['public_sha256']==entry['private_original_sha256']
    if relative.suffix=='.py':ast.parse(raw.decode('utf-8'))
    if relative.suffix=='.json':json.loads(raw)
for f in ROOT.rglob('*.py'):
    if '.git' not in f.parts:ast.parse(f.read_text(encoding='utf-8'))
assert len(list((ROOT/'core').glob('*.py')))==21
vendor=ROOT/'third_party/metax_fa_2_5_3'
upstream=json.loads((vendor/'UPSTREAM.json').read_text())
for name,receipt in upstream['files'].items():
    raw=(vendor/name).read_bytes()
    assert len(raw)==receipt['bytes'] and hashlib.sha256(raw).hexdigest()==receipt['sha256']
    relative=(vendor/name).relative_to(ROOT).as_posix()
    staged=subprocess.run(['git','show',':'+relative],cwd=ROOT,capture_output=True)
    if staged.returncode==0:assert staged.stdout==raw,relative
print('Verified',len(upstream['files']),'unmodified vendor source files pinned to',upstream['commit'])
print('Verified',len(manifest['artifacts']),'exported artifacts and21exact core files;zero model calls.')

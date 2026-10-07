"""Reuse the existing isolated owner builder with only candidate/output paths."""
import hashlib
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
source = HERE.parent/'individual-prefix-owner-candidate-v1/compile_isolated_owner.py'
spec = importlib.util.spec_from_file_location('existing_isolated_owner_builder', source)
owner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(owner)
owner.HERE = HERE
owner.OUT = owner.entry.ROOT+'/candidates/finite-fa-query-shared-A-candidate-20261007-v1'
if __name__ == '__main__':
    print('Reusing builder SHA256 '+hashlib.sha256(source.read_bytes()).hexdigest(), flush=True)
    owner.main()

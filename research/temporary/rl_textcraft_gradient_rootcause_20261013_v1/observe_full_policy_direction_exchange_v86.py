"""Use the unchanged native derivative observer with captured-seed identities.

Only seed-file provenance is composed between its existing capture and replay
interfaces. No official model, loss, optimizer or backward body is copied.
"""
import hashlib
from pathlib import Path

source = Path(__file__).with_name('observe_native_gradient_chain_v4.py')
expected = '3cbefaf36f2a33712094cd017f0cdf026d1a94fee44d2d1c973eb74ef3251117'
assert hashlib.sha256(source.read_bytes()).hexdigest() == expected
code = source.read_text()
old = "        self.case_checkpoint=self.case.get('checkpoint',CHECKPOINT_STEP)"
new = """        receipt_path = self.case.get('boundary_seed_hash_receipt')
        if receipt_path:
            receipt = json.loads(Path(receipt_path).read_bytes())
            self.case['boundary_seed_SHA256_by_rank_micro'] = receipt['hashes']
        self.case_checkpoint=self.case.get('checkpoint',CHECKPOINT_STEP)"""
assert code.count(old) == 1
code = code.replace(old, new)
old = "            phase('case_complete',iteration=iteration)"
new = """            if case.get('capture_boundary_seed'):
                files = []
                for rank in (0, 1):
                    original_chain_path = DEST/f'iteration{iteration}-rank{rank}-chain.json'
                    original_chain = json.loads(original_chain_path.read_bytes())
                    assert len(original_chain['diagnostic_boundary_seeds']) == 8
                    for seed in original_chain['diagnostic_boundary_seeds']:
                        assert seed['mode'] == 'capture_unmodified_native_seed'
                        seed_path = Path(seed['path'])
                        with seed_path.open('rb') as handle:
                            sha = hashlib.file_digest(handle, 'sha256').hexdigest()
                        files.append(dict(rank=rank, **seed, SHA256=sha,
                            native_chain_path=str(original_chain_path),
                            native_chain_SHA256=hashlib.sha256(original_chain_path.read_bytes()).hexdigest()))
                receipt_path = Path(case['captured_seed_receipt'])
                assert not receipt_path.exists()
                receipt = dict(case=iteration, files=files,
                    hashes={f"{s['rank']}:{s['micro']}":s['SHA256'] for s in files})
                receipt_path.write_text(json.dumps(receipt, indent=2)+'\\n')
                state['observations'][-1]['captured_seed_identity_receipt'] = dict(
                    path=str(receipt_path), SHA256=hashlib.sha256(receipt_path.read_bytes()).hexdigest())
            phase('case_complete',iteration=iteration)"""
assert code.count(old) == 1
code = code.replace(old, new)
exec(compile(code, str(source), 'exec'), globals())

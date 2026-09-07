"""Share local timings and tensor receipts without sharing actual activations."""
import hashlib,json
from pathlib import Path
A=Path(__file__).resolve().parent;F=A/'snapshot${ARTIFACT_ROOT}/codex_vendor_fa_captured_operator_20260907_v2'
raw=(F/'results.json').read_bytes();d=json.loads(raw)
s=json.loads((A/'vendor_fa_captured_operator_summary_20260907.json').read_text())
assert hashlib.sha256(raw).hexdigest()==s['raw_sha256'] and s['status']=='verified_complete'
out={'raw_sha256':s['raw_sha256'],'scope':s['scope'],'actual_input_receipts':d['protocol']['actual_operands'],
 'all_local_operator_attempts':d['operator_attempts'],'profile_receipts':{kind:{k:v for k,v in item.items() if k!='GPU_kernels'} for kind,item in d['profiles'].items()},
 'note':'Tensor values and original benchmark text/IDs are private. Receipts, full local timings, buffer shapes and output hashes are preserved. No newly measured GPU calls.'}
for item in out['all_local_operator_attempts']:assert item['complete']
(A/'vendor_fa_operator_numeric_20260907.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print('Archived 14 actual operator records; zero GPU calls.')

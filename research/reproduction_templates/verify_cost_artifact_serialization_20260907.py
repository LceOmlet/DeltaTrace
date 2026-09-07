"""Verify actual full-report payload identity; retain measured host timings."""
import hashlib,json
from pathlib import Path
A=Path(__file__).resolve().parent;F=A/'snapshot${ARTIFACT_ROOT}/codex_cost_artifact_serialization_20260907_v1';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(F/'study.py')==sha(A/'probe_cost_artifact_serialization_20260907.py')
s=json.loads((F/'results.json').read_text());source=A/'snapshot${ARTIFACT_ROOT}/codex_finite_FA_FT_cost16_20260907_v1/results.json'
assert sha(source)==s['source_sha256'] and s['model_calls']==0
d=json.loads(source.read_text());assert d['status']=='complete'
for r in s['records']:
    options={'indent':2} if r['format']=='pretty' else {'separators':(',',':')}
    value=json.dumps(d,ensure_ascii=False,**options).encode()
    assert len(value)==r['bytes'] and hashlib.sha256(value).hexdigest()==r['sha256'] and r['decoded_payload_identical']
    assert r['encoding_seconds']+r['atomic_write_seconds']==r['total_seconds']
    del value
s.update(status='verified_complete',raw_sha256=sha(F/'results.json'),study_sha256=sha(F/'study.py'))
(A/'cost_artifact_serialization_summary_20260907.json').write_text(json.dumps(s,indent=2))
print('Verified identical full payload for both encodings; observed single-host serialization measurement only.')

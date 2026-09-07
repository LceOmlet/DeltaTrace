import hashlib,json,statistics,zipfile
from pathlib import Path
A=Path(__file__).resolve().parent;F=A/'snapshot${ARTIFACT_ROOT}/codex_fa_shared_mean_operator_20260907_v1'
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
with zipfile.ZipFile(F/'review_bundle.zip') as z:
    for name in z.namelist():
        assert '/' not in name and '\\' not in name
        (F/name).write_bytes(z.read(name))
d=json.loads((F/'results.json').read_text());p=d['protocol']
assert p==json.loads((F/'protocol.json').read_text())
assert sha(F/'study.py')==p['study_sha256']
for name,digest in p['sources'].items():assert sha(F/name)==digest
b=json.loads((F/'build_results.json').read_text())
assert b['vendor_sources_before']==b['vendor_sources_after']
assert d['status']=='shared_mean_operator_exact_no_end_to_end_claim' and len(d['calls'])==6
assert d['attributions']==d['native_model_forwards']==d['VJPs']==d['quality_queries']==0
for repeat in range(3):
    old,new=[next(c for c in d['calls'] if c['repeat']==repeat and c['name']==n) for n in ['old','shared_mean']]
    assert old['outputs']==new['outputs']
    assert all(c['exact'] and c['max_abs_difference']==0 for c in d['comparisons'][repeat].values())
    assert new['activity']['query_heads']==32 and new['activity']['kv_heads']==8
    assert new['activity']['GQA_input_expansion'] is False
    assert new['activity']['global_endpoint_mean_buffers']==0
    assert len(old['activity']['buffer_contract'])==15 and len(new['activity']['buffer_contract'])==13
    for c in [old,new]:assert c['activity']['calls_attempted']==c['activity']['calls_enqueued']==1
times={n:statistics.median(c['seconds'] for c in d['calls'] if c['name']==n and not c['warm']) for n in ['old','shared_mean']}
buffers={n:sum(x['bytes'] for x in next(c for c in d['calls'] if c['name']==n)['activity']['buffer_contract']) for n in ['old','shared_mean']}
out={'status':'local_operator_source_build_and_full_output_hashes_verified',
     'raw_sha256':sha(F/'results.json'),'protocol_sha256':sha(F/'protocol.json'),
     'library_sha256':b['library']['sha256'],'compile_seconds':b['wall_seconds'],
     'operator_calls':6,'attributions':0,'quality_queries':0,'full_outputs_exact':True,
     'median_operator_seconds':times,'buffer_contract_bytes':buffers,
     'not_claimed':'No new benchmark, model speed, full-model equivalence or new B4 validation. Old capture failure remains historical.',
     'next':'Assess paired local timings and proof of shared-memory midpoint reuse before any small whole-network screen. No new curves/FT/VJPs.'}
(A/'fa_shared_mean_operator_summary_20260907.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out))

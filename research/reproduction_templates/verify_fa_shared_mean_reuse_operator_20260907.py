import hashlib,json,statistics,zipfile
from pathlib import Path
A=Path(__file__).resolve().parent;F=A/'snapshot${ARTIFACT_ROOT}/codex_fa_shared_mean_reuse_operator_20260907_v1'
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
assert d['status']=='shared_mean_reuse_operator_exact_no_end_to_end_claim' and len(d['calls'])==9
assert d['attributions']==d['native_model_forwards']==d['VJPs']==d['quality_queries']==0
for repeat in range(3):
    rows=[next(c for c in d['calls'] if c['repeat']==repeat and c['name']==n) for n in ['old','extra_tile','reuse_shared']]
    assert rows[0]['outputs']==rows[1]['outputs']==rows[2]['outputs']
    assert all(c['exact'] and c['max_abs_difference']==0 for c in d['comparisons'][repeat].values())
    assert rows[2]['activity']['extra_shared_tile'] is False
    for n,c in enumerate(rows):
        assert c['activity']['calls_attempted']==c['activity']['calls_enqueued']==1
        assert len(c['activity']['buffer_contract'])==(15 if n==0 else 13)
        assert c['activity']['query_heads']==32 and c['activity']['kv_heads']==8
names=['old','extra_tile','reuse_shared']
measured={n:[c['seconds'] for c in d['calls'] if c['name']==n and not c['warm']] for n in names}
times={n:statistics.median(v) for n,v in measured.items()}
out={'status':'shared_mean_three_way_source_build_and_outputs_verified',
    'raw_sha256':sha(F/'results.json'),'protocol_sha256':sha(F/'protocol.json'),
    'library_sha256':b['library']['sha256'],'compile_seconds':b['wall_seconds'],
    'operator_calls':9,'attributions':0,'quality_queries':0,'full_outputs_exact':True,
    'measured_operator_seconds':measured,'median_operator_seconds':times,
    'ratios_to_original':{n:v/times['old'] for n,v in times.items()},
    'buffer_contract_bytes':{n:sum(x['bytes'] for x in next(c for c in d['calls'] if c['name']==n)['activity']['buffer_contract']) for n in names},
    'scope':'One old captured NI0 layer35; not an end-to-end, quality or batch validation.',
    'shared_storage_change':'Mean retained as FP16 registers then copied into the existing FA shared B tile after prior score GEMMs synchronize; no extra shared tile.'}
(A/'fa_shared_mean_reuse_operator_summary_20260907.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))

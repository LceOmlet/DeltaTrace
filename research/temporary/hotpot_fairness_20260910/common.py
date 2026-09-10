"""Explicit artifact readers for the corrected HotpotQA entry point."""
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OLD=HERE.parent/'source_v2_gpu_20260910'
sys.path.insert(0,str(ROOT/'experiments/official'))
sys.path.insert(0,str(OLD))
TASK='hotpotqa_long'
sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
byte_sha=lambda b: hashlib.sha256(b).hexdigest()

def read_bytes(folder,name):
    path=folder/name
    return path.read_bytes() if path.exists() else gzip.decompress(path.with_name(path.name+'.gz').read_bytes())

def read_run(folder):
    rb=read_bytes(folder,'results.json');vb=read_bytes(folder,'vectors.npz')
    report=json.loads(rb)
    assert report['status']=='complete' and byte_sha(vb)==report['vectors_sha256']
    return report, np.load(io.BytesIO(vb)), dict(results_sha256=byte_sha(rb),vectors_sha256=byte_sha(vb))

def full_cases():
    """Only the 48 origins in the previous completely verified benchmark."""
    plan=json.loads((HERE/'protocol.json').read_bytes())
    summary=OLD/'full_recall/analysis.json'
    assert sha(summary)==plan['prior_full_analysis_sha256']
    analysis=json.loads(summary.read_bytes())
    origins={r['index']:r['origin'] for r in analysis['case_origins'] if r['dataset']==TASK}
    assert set(origins)==set(range(48))
    rows={};receipts=[]
    for origin in sorted(set(origins.values())):
        folder=OLD/'raw'/('target_scope_v1' if origin=='reserved_validation' else 'full_recall_v1/'+origin)
        report,vectors,hashes=read_run(folder)
        if origin=='reserved_validation':
            assert hashes['results_sha256']==analysis['parent_results_sha256']
            parent=json.loads((OLD/'target_scope/analysis.json').read_bytes())
            assert hashes['vectors_sha256']==parent['run_vectors_sha256']
        else:
            receipt=next(r for r in analysis['verified_shards'] if r['chunk']==origin)
            assert all(hashes[k]==receipt[k] for k in hashes)
        receipts.append(dict(origin=origin, **hashes))
        for row in report['cases']:
            if row['dataset']==TASK and origins[row['index']]==origin:
                index=row['index'];prefix=f'{TASK}_{index}_full_'
                values={m:vectors[prefix+m+('_signed_full' if m=='DT_target' else '_prompt')].copy()
                        for m in ('DT_target','FT_K1','FT_K3')}
                if index in rows:raise ValueError('Duplicate case')
                rows[index]=(row,values)
        vectors.close()
    assert set(rows)==set(range(48))
    return rows,receipts

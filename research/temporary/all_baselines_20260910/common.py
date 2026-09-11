"""Read all 448 verified DT/FT origins without running either algorithm again."""
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
HOTPOT=HERE.parent/'hotpot_context_v3_20260910'
sys.path.insert(0,str(ROOT/'experiments/official'))
TASKS=('vt_h2_c3','vt_h4_c1','vt_h6_c1','vt_h10_c1','hotpotqa_long')
BASELINES=('Perturbation','REAGENT','CLP','IFR','AttnLRP')
METHODS=('DT','FT_K3')+BASELINES
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
byte_sha=lambda b:hashlib.sha256(b).hexdigest()

def read_bytes(folder,name):
    p=folder/name
    return p.read_bytes() if p.exists() else gzip.decompress((folder/(name+'.gz')).read_bytes())

def read_run(folder):
    rb=read_bytes(folder,'results.json');vb=read_bytes(folder,'vectors.npz');r=json.loads(rb)
    assert r['status']=='complete' and r['vectors_sha256']==byte_sha(vb)
    return r,np.load(io.BytesIO(vb)),dict(results_sha256=byte_sha(rb),vectors_sha256=byte_sha(vb))

def previous_cases():
    analysis=json.loads((OLD/'full_recall/analysis.json').read_bytes())
    assert analysis['status']=='verified_complete_benchmark' and analysis['case_count']==448
    origins={(r['dataset'],r['index']):r['origin'] for r in analysis['case_origins']}
    assert set(origins)=={(t,i) for t in TASKS for i in range(48 if t=='hotpotqa_long' else 100)}
    out={};receipts=[]
    for origin in sorted(set(origins.values())):
        folder=OLD/'raw'/('target_scope_v1' if origin=='reserved_validation' else 'full_recall_v1/'+origin)
        r,v,h=read_run(folder)
        if origin=='reserved_validation':
            assert h['results_sha256']==analysis['parent_results_sha256']
            assert h['vectors_sha256']==json.loads((OLD/'target_scope/analysis.json').read_bytes())['run_vectors_sha256']
        else:
            item=next(x for x in analysis['verified_shards'] if x['chunk']==origin)
            assert all(h[k]==item[k] for k in h)
        receipts.append(dict(origin=origin,**h))
        for row in r['cases']:
            t,i=row['dataset'],row['index']
            if origins[t,i]!=origin:continue
            prefix=f'{t}_{i}_{row["target_mode"]}_'
            values={m:v[prefix+suffix].copy() for m,suffix in [('DT','DT_target_signed_full'),('FT_K3','FT_K3_prompt'),('FT_K1','FT_K1_prompt')]}
            assert (t,i) not in out;out[t,i]=(row,values)
        v.close()
    assert len(out)==448
    return out,receipts

"""Verify real calls through the promoted factories and preserve a failed test receipt."""
import hashlib
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()


def main():
    root=HERE/'promoted_api'
    manifest=HERE.parents[2]/'deltatrace/accelerated/deferred_sources.json'
    rows=[]
    for family,sub in [('qwen3','qwen3'),('qwen35','qwen35_recovered')]:
        p=root/sub;r=json.loads((p/'results.json').read_bytes());z=np.load(p/'vectors.npz',allow_pickle=False)
        assert r['status']=='complete' and r['vectors_sha256']==sha((p/'vectors.npz').read_bytes())
        assert r['script_sha256']==sha((HERE/f'verify_promoted_{family}.py').read_bytes())
        assert r['promoted_receipt']['manifest_sha256']==sha(manifest.read_bytes())
        assert r['FT_calls']==r['generation_calls']==0 and r.get('metric_root_calls',0)==0
        for c in r['comparisons']:
            keys=[n for n in z.files if n.startswith(c['case']+'_')]
            assert len(keys)==6 and all(np.array_equal(z[keys[0]],z[n]) for n in keys)
        assert len(z.files)==12
        rows.append({'family':family,'cases':2,'full_DT_calls':12,'equal_vectors':12,'manifest_sha256':sha(manifest.read_bytes())})
    failed=json.loads((root/'qwen35/results.json').read_bytes())
    assert failed['status']=='failed' and 'is a built-in class' in failed['error']
    assert failed['script_sha256']==sha((HERE/'verify_promoted_qwen35_before_path_fix.py').read_bytes())
    assert len(failed['calls'])==2 and failed['calls'][-1]['name']=='native_initialization'
    summary={'status':'verified','families':rows,
             'initial_check_failure':'Qwen3.5 test attempted inspect.getfile on a dynamically loaded class. Changed only the test to inspect its attribute method; method/backend/framework unchanged. Failure occurred after native initialization, before DT attribution. Qwen3 was not rerun.',
             'claim':'Both explicit factories execute the measured operators and retain equal complete vectors on NI0/MH0. These are integration checks; primary cost evidence remains the paired 16-case runs.'}
    (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',newline='\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':main()

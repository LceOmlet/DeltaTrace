"""Verify all pilot vectors, native head selection and complete-stage peaks."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('directory', type=Path)
args = p.parse_args()
sha = lambda b: hashlib.sha256(b).hexdigest()
root = args.directory
r = json.loads((root/'results.json').read_bytes())
assert r['status'] == 'complete' and r['metric_root_calls'] == r['FT_calls'] == r['generation_calls'] == 0
assert sha((root/'vectors.npz').read_bytes()) == r['vectors_sha256']
vectors = np.load(root/'vectors.npz')
assert len(vectors.files) == len(r['root_calls']) == 12
summary = {'status':'complete','raw_sha256':sha((root/'results.json').read_bytes()),
           'vectors_sha256':r['vectors_sha256'],'cases':[],
           'interpretation':'Two-case pilot. Identical vectors imply unchanged metric inputs; metrics not rerun. Small wall-time differences are not a demonstrated robust speedup.'}
for row in r['cases']:
    name = row['dataset']+'_'+str(row['index'])
    names = [n for n in vectors.files if n.startswith(name+'_')]
    assert len(names) == 6
    assert all(np.array_equal(vectors[n],vectors[names[0]]) for n in names)
    for call in r['root_calls']:
        if not call['name'].startswith(name+'_'):
            continue
        assert call['input_shape'][0] == 2 and call['use_cache'] is False
        assert call['logits_to_keep'] == (row['target_length']+1 if '_candidate_' in call['name'] else 0)
    out = {'case':name,'all_six_vectors_identical':True}
    for mode in ('baseline','candidate'):
        calls = [c for c in r['calls'] if c.get('warm') and c['name'].startswith(name+'_'+mode+'_')]
        assert len(calls) == 2
        assert all(c['status']=='returned' and c['peak_allocated']>=c['stages']['native_root_peak_allocated'] for c in calls)
        out[mode] = {'mean_seconds':statistics.mean(c['seconds'] for c in calls),
                     'seconds_each':[c['seconds'] for c in calls],
                     'head_mean_seconds':statistics.mean(c['head_seconds'] for c in calls),
                     'peak_allocated_bytes':max(c['peak_allocated'] for c in calls)}
    out['mean_latency_reduction_fraction'] = 1-out['candidate']['mean_seconds']/out['baseline']['mean_seconds']
    out['peak_reduction_bytes'] = out['baseline']['peak_allocated_bytes']-out['candidate']['peak_allocated_bytes']
    summary['cases'].append(out)
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))

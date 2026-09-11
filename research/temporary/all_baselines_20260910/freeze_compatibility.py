"""Register only the 23 immutable successful technical pilots before offload."""
import json
import numpy as np
from common import HERE,sha

def main():
    receipts=[];identity=None;methods=set()
    for path in sorted((HERE/'raw').glob('*/*/*/results.json')):
        record=json.loads(path.read_bytes());assert record['index']==0 and record['status']=='complete'
        if identity is None:identity=record['identity']
        assert record['identity']==identity and record['vectors_sha256']==sha(path.with_name('vectors.npz'))
        receipts.append(sha(path));methods.add(record['method'])
    assert len(receipts)==23 and len(methods)==5
    controls=[]
    for method in ('Perturbation','CLP','IFR'):
        old=HERE/'initial_pilot'/method/'vt_h2_c3/000';new=HERE/'raw'/method/'vt_h2_c3/000'
        left=np.load(old/'vectors.npz');right=np.load(new/'vectors.npz')
        assert set(left.files)==set(right.files)
        for name in left.files:assert np.array_equal(left[name],right[name],equal_nan=True)
        controls.append(dict(method=method,dataset='vt_h2_c3',index=0,all_vectors_bitwise_equal=True,
            first_results_sha256=sha(old/'results.json'),second_results_sha256=sha(new/'results.json')))
        left.close();right.close()
    manifest=dict(status='frozen_before_saved_tensor_offload_and_quality',
        accepted=[dict(name='numeric_fix_gpu_saved_tensors_pilot23',identity=identity,methods=sorted(methods),
            result_sha256=receipts,source_commit='6f967e4',reason='Only tensor storage changes; immutable successful pilot inputs and arithmetic reused.')],
        initial_three_pilot_bitwise_controls=controls)
    path=HERE/'execution_compatibility.json';assert not path.exists()
    path.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(registered=23,bitwise_controls=len(controls),sha256=sha(path))))

if __name__=='__main__':main()

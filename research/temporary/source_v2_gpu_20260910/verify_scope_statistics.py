"""Independent known-answer checks for paired scope inference."""
import hashlib
import json
from pathlib import Path

import numpy as np

from scope_statistics import TASKS,FRACTIONS,VIEWS,summarize_scope


def main():
    rows=[];split={'tasks':{t:{'validation':list(range(16))} for t in TASKS}}
    for ti,t in enumerate(TASKS):
        for i in range(16):
            for view in VIEWS:
                for f in FRACTIONS:
                    base=.2+.05*np.sin(i)
                    delta=([.1,.05,-.025,.075,0.][ti] if view=='raw' else -.15 if ti==4 else 0.)
                    ft=base if view=='raw' else .5
                    for m,value in [('DT_target',ft+delta),('FT_K1',ft),('FT_K3',ft)]:
                        rows.append(dict(dataset=t,index=i,method=m,view=view,fraction=f,recall=value,ceiling=.5))
    r=summarize_scope(rows,split);p=r['primary_comparisons']
    assert np.allclose(p['VT_raw']['adjusted_ci9875'],[.05,.05],atol=1e-14,rtol=0)
    assert p['VT_raw']['adjusted_verdict']=='advantage'
    assert p['VT_density']['adjusted_verdict']=='ceiling_parity'
    assert p['HotpotQA_raw']['adjusted_verdict']=='inconclusive'
    assert np.allclose(p['HotpotQA_density']['adjusted_ci9875'],[-.15,-.15],atol=1e-14,rtol=0)
    assert r['confirmed_scoped_advantages']==['VT_raw']
    assert r['confirmed_scoped_disadvantages']==['HotpotQA_density']
    assert r['confirmed_same_view_shared_advantage']==[]
    assert r==summarize_scope(list(reversed(rows)),split)
    report=dict(status='passed',cases='Synthetic varying paired baselines with fixed known differences',
        checks=['pairing preserves a constant difference despite varying baselines','equal-task macro',
                'ceiling parity is not superiority','null difference is inconclusive',
                'one scoped win cannot become a shared claim','input row-order invariance'],
        statistics_sha256=hashlib.sha256((Path(__file__).parent/'scope_statistics.py').read_bytes()).hexdigest(),
        verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (Path(__file__).parent/'scope_statistics_verification.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()

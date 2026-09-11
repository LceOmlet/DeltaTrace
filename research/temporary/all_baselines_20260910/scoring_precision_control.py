"""Preserve the failed CPU scoring draft and verify restoration of v3 precision."""
import argparse
from collections import Counter
import gzip
from hashlib import sha256
import json
from pathlib import Path
from common import HERE,HOTPOT,sha

CONTROL=HERE/'scoring_precision_control'
FILES=('per_case.csv','summary.csv','primary_recall10.csv','selections.json','analysis.json')

def prior_comparison(predictions):
    prior={(r['index'],r['method'],r['budget_unit'],r['budget_value']):r
        for r in json.loads((HOTPOT/'selections.json').read_bytes())
        if r['target_mode']=='full' and r['pooling']=='signed_sum'}
    mismatch=Counter();checked=0;example=None
    for row in predictions:
        if row['dataset']!='hotpotqa_long' or row['method'] not in ('DT','FT_K3','FT_K1'):continue
        method='DT_target' if row['method']=='DT' else row['method']
        old=prior[row['index'],method,row['budget_unit'],row['fraction']];checked+=1
        for key in ('ranked_units','ranked_scores','selected_units','selected_tokens'):
            if row[key]!=old[key]:
                mismatch[row['method']+':'+key]+=1
                if example is None and key=='ranked_scores':
                    example=dict(index=row['index'],method=row['method'],budget_unit=row['budget_unit'],fraction=row['fraction'],
                        first_different_pair=next([a,b] for a,b in zip(old[key],row[key]) if a!=b))
    assert checked==48*3*6
    return dict(checked_rows=checked,mismatches=dict(mismatch),example=example)

def main():
    p=argparse.ArgumentParser();p.add_argument('--capture',action='store_true');a=p.parse_args()
    current=json.loads((HERE/'selections.json').read_bytes())
    if a.capture:
        CONTROL.mkdir(exist_ok=True);assert not (CONTROL/'before.json').exists()
        for name in FILES:(CONTROL/(name+'.gz')).write_bytes(gzip.compress((HERE/name).read_bytes(),mtime=0))
        (CONTROL/'score_cases_before.py.txt').write_bytes((HERE/'score_cases.py').read_bytes())
        result=dict(status='initial_cpu_draft_failed_independent_precision_check',
            output_sha256={name:sha(HERE/name) for name in FILES},scorer_sha256=sha(HERE/'score_cases.py'),
            hotpot_selections_sha256=sha(HOTPOT/'selections.json'),comparison=prior_comparison(current))
        (CONTROL/'before.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    else:
        before=json.loads((CONTROL/'before.json').read_bytes())
        for name in FILES:assert sha256(gzip.decompress((CONTROL/(name+'.gz')).read_bytes())).hexdigest()==before['output_sha256'][name]
        assert sha(CONTROL/'score_cases_before.py.txt')==before['scorer_sha256']
        original=json.loads(gzip.decompress((CONTROL/'selections.json.gz').read_bytes()))
        assert len(original)==len(current)==34944
        assert before['hotpot_selections_sha256']==sha(HOTPOT/'selections.json')
        for name in ('per_case.csv','summary.csv','primary_recall10.csv'):assert sha(HERE/name)==before['output_sha256'][name],name
        for left,right in zip(original,current):
            for key in set(left)|set(right):
                if key=='ranked_scores':continue
                assert left[key]==right[key],(key,left['dataset'],left['index'],left['method'])
        comparison=prior_comparison(current);assert not comparison['mismatches']
        result=dict(status='original_hotpot_precision_restored_metrics_and_selections_unchanged',
            unchanged_score_rows=34944,unchanged_summary_rows=390,unchanged_selection_rows=34944,
            prior_hotpot_rankings_and_scores_bitwise_equal_rows=comparison['checked_rows'],
            before_sha256=sha(CONTROL/'before.json'),scorer_sha256=sha(HERE/'score_cases.py'),
            control_sha256=sha(Path(__file__)),
            output_sha256={name:sha(HERE/name) for name in FILES})
        (CONTROL/'verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()

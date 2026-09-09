"""Compare existing actual early deletion sets using one frozen DT vector.

This measures a ranking discrepancy; it cannot identify the responsible rule.
The DT set maximizes fixed score mass by construction, so this is deliberately
not reported as independent causal evidence for a particular GDN mechanism.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshots',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();here=Path(__file__).resolve().parent
    receipt=json.loads((here/'initial_evidence.json').read_bytes());rows=[]
    sha=lambda b:hashlib.sha256(b).hexdigest()
    for ref in receipt['raw_references']:
        path=args.snapshots/ref['path'].lstrip('/')
        assert sha(path.read_bytes())==ref['sha256']
        assert sha((path.parent/'vectors.npz').read_bytes())==ref['vectors_sha256']
        raw=json.loads(path.read_bytes());z=np.load(path.parent/'vectors.npz',allow_pickle=False)
        for c in raw['cases']:
            if c['status']!='complete':continue
            name=f"{c['dataset']}_{c['index']}";signed=z[name+'_DT_signed_full'][c['user_positions']]
            for step in (1,2,4):
                selected={};actual={}
                for method in ('DT','FT_K1'):
                    curve=c['metrics'][method];deleted=curve['deleted_user_indices'][step]
                    if method=='DT':assert np.all(signed[deleted]>0)
                    ids=np.array(c['input_ids'],dtype=np.int64)
                    ids[[c['user_positions'][j] for j in deleted]]=ids[-1]
                    assert sha(ids.tobytes())==curve['actual_input_hashes'][step]
                    selected[method]=float(signed[deleted].sum())
                    actual[method]=curve['scores'][0]-curve['scores'][step]
                assert len(c['metrics']['DT']['deleted_user_indices'][step])==len(c['metrics']['FT_K1']['deleted_user_indices'][step])
                rows.append({'family':ref['family'],'dataset':c['dataset'],'index':c['index'],
                    'step':step,'fraction':step/20,'input_sha256':c['input_sha256'],
                    'DT_mass_of_DT_set':selected['DT'],'DT_mass_of_FT_set':selected['FT_K1'],
                    'actual_drop_DT_set':actual['DT'],'actual_drop_FT_set':actual['FT_K1'],
                    'DT_set_less_destructive':actual['DT']<actual['FT_K1'],
                    'fixed_mass_prefers_DT':selected['DT']>selected['FT_K1'],
                    'normalized_drop_gap':(actual['DT']-actual['FT_K1'])/(c['metrics']['DT']['scores'][0]-c['metrics']['DT']['scores'][-1])})
    summaries=[]
    for family in ('qwen3','qwen35'):
        for dataset in ('niah_mq_q2','morehopqa'):
            for step in (1,2,4):
                r=[x for x in rows if (x['family'],x['dataset'],x['step'])==(family,dataset,step)]
                assert len(r)==8
                summaries.append({'family':family,'dataset':dataset,'step':step,'n':len(r),
                    'DT_set_less_destructive':sum(x['DT_set_less_destructive'] for x in r),
                    'DT_set_more_destructive':sum(x['actual_drop_DT_set']>x['actual_drop_FT_set'] for x in r),
                    'mean_normalized_drop_gap':statistics.mean(x['normalized_drop_gap'] for x in r)})
    out={'status':'verified_existing_records','new_model_calls':0,'new_metric_calls':0,
         'raw_references':receipt['raw_references'],'cases':rows,'summaries':summaries,
         'limitation':'Frozen positive curves supply the original deleted sets. These are the same early positive ranks; no signed-tail metric is inferred. Counting a ranking reversal is descriptive, not proof of a particular rule cause.'}
    args.output.write_text(json.dumps(out,indent=2)+'\n',newline='\n')
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':main()

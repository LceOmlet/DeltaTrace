"""Recheck paired original NI curves and bound only final-score rounding.

No new metric/model run. This preserves original AUC and signed-view provenance;
the rounding envelope is a diagnostic, not a replacement benchmark or FP32 run.
It does NOT bound native log-softmax, model/FA/FLA or attribution rounding.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import statistics
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sha=lambda b:hashlib.sha256(b).hexdigest()


def final_rounding_envelope(scores,mantissa_bits):
    s=np.asarray(scores,dtype=np.float64);assert s[0]>s[-1] and np.all(s<0)
    ulp=np.exp2(np.floor(np.log2(np.abs(s)))-mantissa_bits)
    assert np.all(s/ulp==np.round(s/ulp))
    lo,hi=s-.5*ulp,s+.5*ulp
    assert lo[0]>hi[-1]
    lower=[];upper=[]
    for i in range(len(s)):
        if i==0:a=b=1.
        elif i==len(s)-1:a=b=0.
        else:
            corners=[np.clip((x-baseline)/(actual-baseline),0,1) for x,actual,baseline in
                     itertools.product((lo[i],hi[i]),(lo[0],hi[0]),(lo[-1],hi[-1]))]
            a,b=min(corners),max(corners)
        lower.append(a);upper.append(b)
    return [float(np.trapezoid(np.minimum.accumulate(v),dx=.05)) for v in (lower,upper)]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--snapshots',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();assembly_path=ROOT/'research/temporary/acceleration_20260909/clean_signed_views32.json'
    assembly=json.loads(assembly_path.read_bytes())
    replay_path=args.snapshots/'tmp/codex_signed_rise_audit_20260909_v1/run/results.json'
    assert sha(replay_path.read_bytes())==assembly['signed_replay_sha256']
    replay=json.loads(replay_path.read_bytes());extra={(r['dataset'],r['index']):r for r in replay['cases']}
    view={(f,r['dataset'],r['index']):r for f,rows in assembly['models'].items() for r in rows}
    rows=[];curves=[]
    for ref in assembly['references']:
        source=args.snapshots/ref['path'].lstrip('/');assert sha(source.read_bytes())==ref['sha256']
        raw=json.loads(source.read_bytes());family=ref['family']
        for c in raw['cases']:
            if c['status']!='complete' or c['dataset']!='niah_mq_q2':continue
            identity=view[(family,c['dataset'],c['index'])]
            dt=c['metrics']['DT'] if identity['RISE_reuse_proof'] else extra[(c['dataset'],c['index'])]['curve']
            ft=c['metrics']['FT_K1'];out={'family':family,'index':c['index'],'input_sha256':c['input_sha256']}
            bounds={};ys={}
            for name,curve in [('DT',dt),('FT',ft)]:
                y=np.asarray(curve['normalized_model_response'],dtype=np.float64);s=np.asarray(curve['scores'])
                assert len(y)==len(s)==21 and np.isclose(y[0],1) and np.isclose(y[-1],0)
                expected=np.minimum.accumulate(np.clip((s-s[-1])/(s[0]-s[-1]),0,1))
                assert np.allclose(expected,y,atol=1e-12,rtol=0)
                for deleted,digest in zip(curve['deleted_user_indices'],curve['actual_input_hashes']):
                    ids=np.array(c['input_ids'],dtype=np.int64);ids[[c['user_positions'][j] for j in deleted]]=ids[-1]
                    assert sha(ids.tobytes())==digest
                auc=float(np.trapezoid(y,dx=.05));target=identity['DT']['rise'] if name=='DT' else identity['FT_K1']['rise']
                assert np.isclose(auc,target,atol=1e-12,rtol=0)
                out[name+'_RISE']=auc;out[name+'_zero_step']=int(np.flatnonzero(y==0)[0])
                bounds[name]=final_rounding_envelope(s,10 if family=='qwen3' else 7);ys[name]=y
                curves.append({'family':family,'index':c['index'],'method':name,'scores':s.tolist(),'normalized_response':y.tolist(),
                               'final_rounding_auc_envelope':bounds[name]})
            difference=ys['DT']-ys['FT'];out['gap']=out['DT_RISE']-out['FT_RISE']
            out['gap_first20pct']=float(np.trapezoid(difference[:5],dx=.05))
            out['gap_rest']=float(np.trapezoid(difference[4:],dx=.05))
            assert np.isclose(out['gap_first20pct']+out['gap_rest'],out['gap'],atol=1e-12,rtol=0)
            out['final_rounding_gap_envelope']=[bounds['DT'][0]-bounds['FT'][1],bounds['DT'][1]-bounds['FT'][0]];rows.append(out)
    assert len(rows)==16
    summaries=[]
    for family in ('qwen3','qwen35'):
        a=[r for r in rows if r['family']==family]
        summaries.append({'family':family,'n':len(a),'mean_gap':statistics.mean(r['gap'] for r in a),
            'mean_gap_first20pct':statistics.mean(r['gap_first20pct'] for r in a),'mean_gap_rest':statistics.mean(r['gap_rest'] for r in a),
            'final_rounding_mean_gap_envelope':[statistics.mean(r['final_rounding_gap_envelope'][i] for r in a) for i in (0,1)],
            'DT_zero_by10pct':sum(r['DT_zero_step']<=2 for r in a),'FT_zero_by10pct':sum(r['FT_zero_step']<=2 for r in a),
            'DT_median_zero_step':statistics.median(r['DT_zero_step'] for r in a),'FT_median_zero_step':statistics.median(r['FT_zero_step'] for r in a)})
    report={'status':'verified','assembly_sha256':sha(assembly_path.read_bytes()),'signed_replay_sha256':assembly['signed_replay_sha256'],
            'references':assembly['references'],'new_model_calls':0,'new_metric_calls':0,'cases':rows,'curves':curves,'summaries':summaries,
            'limitation':'Only the final native scalar score rounding is bounded. Internal log-softmax/model/attribution numerical error and cross-model architectural causation are not bounded.'}
    args.output.write_text(json.dumps(report,indent=2)+'\n',newline='\n');print(json.dumps(summaries,indent=2))


if __name__=='__main__':main()

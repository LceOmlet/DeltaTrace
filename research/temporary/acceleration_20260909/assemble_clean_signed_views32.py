"""Combine verified original curves and the existing original signed rescoring.

No new attribution, model evaluation, metric formula or dataset is introduced.
The actual formal-entry proof helper decides whether the positive RISE is reusable.
"""
import argparse,hashlib,json,sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--entry',type=Path,required=True)
    p.add_argument('--signed-replay',type=Path,required=True)
    p.add_argument('--qwen3',type=Path,required=True)
    p.add_argument('--qwen35',type=Path,nargs=2,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();sha=lambda b:hashlib.sha256(b).hexdigest()
    sys.path.insert(0,str(args.entry));from score_views import signed_rise_equals_positive_curve
    import numpy as np
    import torch
    replay_raw=args.signed_replay.read_bytes();replay=json.loads(replay_raw)
    assert replay['status']=='complete' and replay['DT_calls']==replay['FT_calls']==replay['generation_calls']==0
    signed={(c['dataset'],c['index']):c for c in replay['cases']}
    expected={r['path']:r for r in replay['plan']['references']}
    result={'scope':'Both clean development16 sets, assembled from source-matched original runs and existing signed rescoring; not a new full16 execution or paper table',
        'source_sha256':sha(Path(__file__).read_bytes()),'score_views_sha256':sha((args.entry/'score_views.py').read_bytes()),
        'signed_replay_sha256':sha(replay_raw),'new_model_calls':0,'views':{'rise':'signed','mas':'positive_part','needle':'positive_part'},
        'models':{},'references':[],'proof_reuse_count':0,'signed_replay_reuse_count':0}
    for family,paths in [('qwen3',[args.qwen3]),('qwen35',args.qwen35)]:
        output=[]
        for path in paths:
            raw=path.read_bytes();d=json.loads(raw);assert d['family']==family
            vector_raw=(path.parent/'vectors.npz').read_bytes();digest=sha(vector_raw)
            if d['status']=='complete':assert digest==d['vectors_sha256']
            else:assert family=='qwen35' and d['status']=='failed'
            if family=='qwen35':
                assert sha(raw)==expected[str(path)]['sha256'] and digest==expected[str(path)]['vectors_sha256']
            result['references'].append({'family':family,'path':str(path),'sha256':sha(raw),'vectors_sha256':digest})
            vectors=np.load(path.parent/'vectors.npz',allow_pickle=False)
            for c in d['cases']:
                if c['status']!='complete':continue
                key=f"{c['dataset']}_{c['index']}";original=np.asarray(c['input_ids'],dtype=np.int64)
                assert sha(original.tobytes())==c['input_sha256']
                w=torch.tensor(vectors[key+'_DT_signed_full'][c['user_positions']],dtype=torch.float32)
                assert np.array_equal(w.clamp_min(0).numpy(),vectors[key+'_DT_positive_prompt'])
                positive=c['metrics']['DT'];proof=signed_rise_equals_positive_curve(w,c['keep'],positive)
                curve=positive if proof else signed[(c['dataset'],c['index'])]['curve']
                if proof:
                    rise=positive['rise'];result['proof_reuse_count']+=1
                else:
                    extra=signed[(c['dataset'],c['index'])]
                    assert extra['status']=='complete' and extra['input_sha256']==c['input_sha256']
                    rise=extra['signed_RISE'];result['signed_replay_reuse_count']+=1
                assert len(curve['actual_input_hashes'])==21
                for deleted,digest in zip(curve['deleted_user_indices'],curve['actual_input_hashes']):
                    assert set(deleted)<=set(c['keep'])
                    actual=original.copy();actual[[c['user_positions'][j] for j in deleted]]=original[-1]
                    assert sha(actual.tobytes())==digest
                drift=[]
                if not proof:
                    for i,h in enumerate(curve['actual_input_hashes']):
                        matches=[j for j,old in enumerate(positive['actual_input_hashes']) if h==old]
                        for j in matches:
                            if curve['scores'][i]!=positive['scores'][j]:
                                drift.append({'signed_step':i,'positive_step':j,'input_sha256':h,
                                              'signed_score':curve['scores'][i],'positive_score':positive['scores'][j]})
                output.append({'dataset':c['dataset'],'index':c['index'],'input_sha256':c['input_sha256'],
                    'source_vector_sha256':sha(vectors[key+'_DT_signed_full'].tobytes()),
                    'DT':{'rise':rise,'mas':positive['mas'],'needle':positive['needle']},
                    'FT_K1':{k:c['metrics']['FT_K1'][k] for k in ('rise','mas')},'FT_K3_needle':c.get('FT_K3_needle'),
                    'positive_RISE_history':positive['rise'],'RISE_reuse_proof':proof,
                    'signed_RISE_source':'existing_positive_curve_with_proof' if proof else 'existing_original_signed_replay',
                    'shared_input_scoring_drift':drift})
        assert len(output)==16 and {(c['dataset'],c['index']) for c in output}=={(task,i) for task in ('niah_mq_q2','morehopqa') for i in range(8)}
        result['models'][family]=output
    assert result['proof_reuse_count']+result['signed_replay_reuse_count']==32
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('models','references')}))


if __name__=='__main__':main()

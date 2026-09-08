"""Five immutable vectors, one original NI0 recovery metric; CPU only."""
import os
os.environ.update(PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1')
import ast,hashlib,json,math,time,traceback,zipfile
from pathlib import Path
from typing import Sequence,List
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;p=json.loads((HERE/'protocol.json').read_bytes());sha=lambda b:hashlib.sha256(b).hexdigest()
r={'status':'running','protocol':p,'model_loads':0,'model_forwards':0,'GPU_calls':0,'generation_calls':0,'recovery_calls':0,'methods':{}};start=time.perf_counter()
try:
    for name,digest in p['files_sha256'].items():assert sha((HERE/name).read_bytes())==digest
    raw=Path(p['author_source']).read_bytes().replace(b'\r\n',b'\n');assert sha(raw)==p['author_source_sha256']
    fn=next(n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name=='evaluate_attr_recovery_skip_tokens')
    exec(compile(ast.Module(body=[fn],type_ignores=[]),p['author_source'],'exec'))
    sp=Path(p['spans_file']);assert sha(sp.read_bytes())==p['spans_sha256'];case=json.loads(sp.read_bytes())['cases'][0];assert case['dataset']=='niah_mq_q2' and case['index']==0
    user=case['input_metadata']['author_user_positions'];keep=case['mapping']['keep_local_indices'];gold=case['mapping']['gold_user_token_indices']
    full_gold=set(gold);eligible_gold=full_gold&set(keep);assert eligible_gold
    r.update(dataset='niah_mq_q2',index=0,user_tokens=len(user),eligible_tokens=len(keep),full_gold_count=len(full_gold),eligible_gold_count=len(eligible_gold),top_fraction=0.1)
    vectors={}
    for method,a in p['score_files'].items():
        file=Path(a['path']);assert sha(file.read_bytes())==a['sha256'];values=np.load(file,allow_pickle=False)[a['field']]
        raw_vector=torch.from_numpy(values[0].copy()).float();assert raw_vector.shape==(605,) and torch.isfinite(raw_vector).all()
        prompt=raw_vector[user];r['recovery_calls']+=1
        score=evaluate_attr_recovery_skip_tokens(prompt[None,:],keep_prompt_token_indices=keep,gold_prompt_token_indices=gold,top_fraction=0.1)
        used=prompt.clamp(min=0).index_select(0,torch.tensor(keep));k=math.ceil(.1*len(keep));top=torch.topk(used,k).indices.tolist()
        selected=[keep[j] for j in top];hits=sorted(set(selected)&eligible_gold);assert score==len(hits)/len(eligible_gold)
        cutoff=float(used[top[-1]])
        r['methods'][method]={'recovery':score,'selected_count':k,'hits':len(hits),'gold_count':len(eligible_gold),
            'selected_user_indices':selected,'selected_absolute_positions':[user[j] for j in selected],
            'hit_user_indices':hits,'negative_eligible_count':int((prompt[keep]<0).sum()),'cutoff_ties':int((used==cutoff).sum()),
            'score_input_sha256':a['sha256'],'metric_clamp':'Original author helper clamps negative mass to zero; signed DeltaTrace vector is retained unchanged.'}
        vectors[method]=raw_vector.numpy()
    assert r['recovery_calls']==5
    f=HERE/'immutable_score_vectors.npz';np.savez_compressed(f,**vectors);r['vector_sha256']=sha(f.read_bytes())
    r['status']='five_fixed_methods_original_NI0_recovery_executed_CPU_only'
except Exception:r['status']='failed';r['error']=traceback.format_exc()
finally:
    r['seconds']=time.perf_counter()-start;(HERE/'results.json').write_text(json.dumps(r,indent=2))
    with zipfile.ZipFile(HERE/'review_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in ['study.py','protocol.json','results.json','immutable_score_vectors.npz']:
            if (HERE/name).exists():z.write(HERE/name,name)
    print(json.dumps({'status':r['status'],'methods':{k:v['recovery'] for k,v in r['methods'].items()},'error':r.get('error')}),flush=True)

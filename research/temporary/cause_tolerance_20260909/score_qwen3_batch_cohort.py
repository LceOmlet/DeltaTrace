"""Original author metrics for a complete frozen batch cohort, including mixed B1/B2.

No attribution rerun, FT rerun or generation. RISE uses signed ordering, MAS and
needle the positive view. Every scorer call checks the exact author prefix and
frozen generation, including EOS. The evaluator is the unmodified paper source.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time
import traceback


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('release','environment','reference','batch_result','output'):p.add_argument('--'+n,type=Path,required=True)
    for n in ('reference_sha256','batch_result_sha256'):p.add_argument('--'+n.replace('_','-'),required=True)
    args=p.parse_args();sha=lambda b:hashlib.sha256(b).hexdigest()
    assert sha(args.reference.read_bytes())==args.reference_sha256 and sha(args.batch_result.read_bytes())==args.batch_result_sha256
    reference=json.loads(args.reference.read_bytes());batch=json.loads(args.batch_result.read_bytes())
    assert batch['status']=='complete' and batch['throughput_gate_passed']
    assert sha((args.batch_result.parent/'vectors.npz').read_bytes())==batch['vectors_sha256']
    env=json.loads(args.environment.read_bytes())['qwen3'];official=Path(env['official_root'])
    protocol=json.loads((args.release/'experiments/official/protocol.json').read_bytes())
    for path,digest in protocol['official_normalized_sources'].items():assert sha((official/path).read_bytes().replace(b'\r\n',b'\n'))==digest
    tasks=list(dict.fromkeys(c['dataset'] for c in reference['cases']))
    for task in tasks:assert sha((official/'exp/exp2/data'/(task+'.jsonl')).read_bytes())==protocol['tasks'][task]['cache_sha256']
    os.environ.update(MACA_PATH='/opt/maca',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    sys.path.insert(0,str(official));sys.path.insert(0,str(args.release/'experiments/official'))
    import numpy as np
    import torch
    from exp.exp2 import run_exp as author,dataset_utils
    from llm_attr_eval import LLMAttributionEvaluator
    import ft_ifr_improve as ft
    from score_views import signed_rise_equals_positive_curve
    torch.set_num_threads(4);torch.manual_seed(73);torch.backends.cuda.matmul.allow_tf32=False
    z=np.load(args.batch_result.parent/'vectors.npz',allow_pickle=False)
    ref={f"{r['dataset']}_{r['index']}":r for r in reference['cases']}
    cases=[(key,i) for i,g in enumerate(batch['groups']) for key in g]
    assert len(cases)==len(set(k for k,_ in cases))==len(ref)
    assert set(k for k,_ in cases)==set(ref)
    args.output.mkdir(exist_ok=False)
    report={'status':'loading','driver_sha256':sha(Path(__file__).read_bytes()),'batch_result_sha256':args.batch_result_sha256,
            'vectors_sha256':batch['vectors_sha256'],'reference_sha256':args.reference_sha256,'cases':[],
            'metric_calls':[],'native_metric_root_calls':0,'generation_calls':0,'FT_calls':0,'attribution_calls':0,
            'metric_sample_batch':1,'DT_sample_batch_by_case':{key:len(batch['groups'][i]) for key,i in cases}}
    def save():
        tmp=args.output/'results.partial';tmp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');tmp.replace(args.output/'results.json')
    try:
        tick=time.perf_counter();model,tokenizer=author.load_model(env['checkpoint'],'cuda:0');model.eval().requires_grad_(False)
        model.set_attn_implementation('eager');report['model_load_seconds']=time.perf_counter()-tick
        assert sha(Path(inspect.getfile(type(model))).read_bytes())==env['native_model_sha256']
        identities={n:(type(m).forward,m.forward) for n,m in model.named_modules()};evaluator=LLMAttributionEvaluator(model,tokenizer)
        cached={task:dataset_utils.load_cached(official/'exp/exp2/data'/(task+'.jsonl')) for task in tasks}
        for case,group in cases:
            row=ref[case];ex=cached[row['dataset']][row['index']];ids=torch.tensor(row['input_ids'],dtype=torch.long)[None]
            assert sha(ids.numpy().tobytes())==row['input_sha256']
            eligible=[row['user_positions'][j] for j in row['keep']]
            rec={'case':case,'input_sha256':row['input_sha256'],'metrics':{}};report['cases'].append(rec)
            for mode,name in [('single','r0_single/'+case),('batch',f'r0_batch/{group}/'+case)]:
                array=z[name];assert array.shape==(ids.shape[1],) and np.isfinite(array).all()
                signed=torch.tensor(array[row['user_positions']],dtype=torch.float32);curves={}
                rec[mode+'_vector_sha256']=sha(array.tobytes())
                for view,scores in [('positive',signed.clamp_min(0)),('signed',signed)]:
                    if view=='signed':
                        proof=signed_rise_equals_positive_curve(signed,row['keep'],curves['positive'])
                        if proof is not None:
                            curves['signed_reuse_proof']=proof;curves['rise']=curves['positive']['rise'];break
                    curve={'actual_input_hashes':[],'deleted_user_indices':[]}
                    def before_score(_module,args,kwargs):
                        actual=kwargs['input_ids'].detach().cpu();assert actual.shape==ids.shape
                        assert torch.equal(actual[:,row['prompt_length']:],ids[:,row['prompt_length']:])
                        changed=(actual[0]!=ids[0]).nonzero().flatten().tolist()
                        assert set(changed)<=set(eligible) and all(int(actual[0,j])==tokenizer.eos_token_id for j in changed)
                        curve['actual_input_hashes'].append(sha(actual.numpy().tobytes()))
                        curve['deleted_user_indices'].append([row['user_positions'].index(j) for j in changed]);report['native_metric_root_calls']+=1
                    def observe(frame,event,result):
                        if frame.f_code is ft.faithfulness_test_skip_tokens.__code__ and event=='return' and result is not None:
                            for k in ('scores','density','normalized_model_response','alignment_penalty','corrected_scores'):curve[k]=np.asarray(frame.f_locals[k]).tolist()
                    handle=model.register_forward_pre_hook(before_score,with_kwargs=True);assert sys.getprofile() is None;sys.setprofile(observe)
                    report['status']=case+'_'+mode+'_'+view;save();tick=time.perf_counter()
                    try:
                        with torch.no_grad():values=ft.faithfulness_test_skip_tokens(evaluator,scores[None],ex.prompt,ex.target,
                            keep_prompt_token_indices=row['keep'],user_prompt_indices=row['user_positions'],k=20)
                    finally:sys.setprofile(None);handle.remove()
                    report['metric_calls'].append({'case':case,'mode':mode,'view':view,'seconds':time.perf_counter()-tick})
                    assert len(curve['actual_input_hashes'])==21 and curve['actual_input_hashes'][0]==row['input_sha256']
                    curve.update(zip(('rise','mas','rise_plus_ap'),map(float,values)));curve['MAS_valid']=view=='positive';curves[view]=curve
                    if view=='signed':curves['rise']=curve['rise']
                curves['mas']=curves['positive']['mas'];curves['needle']=float(ft.evaluate_attr_recovery_skip_tokens(
                    signed.clamp_min(0)[None],keep_prompt_token_indices=row['keep'],gold_prompt_token_indices=row['gold'],top_fraction=.1)) if row['gold'] else None
                rec['metrics'][mode]=curves
                for n,m in model.named_modules():assert (type(m).forward,m.forward)==identities[n]
                save();print(json.dumps({'case':case,'mode':mode,**{k:curves[k] for k in ('rise','mas','needle')}}),flush=True)
        report['status']='complete'
    except Exception:
        report['status']='failed';report['error']=traceback.format_exc();raise
    finally:save()


if __name__=='__main__':main()

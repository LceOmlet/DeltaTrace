"""Fixed four-case target-scope diagnosis, not a replacement paper method.

NI0/1 and MH0/1 are selected by index before attribution. Compare whole released
generation+EOS with the author's retokenized answer span, using the same frozen
DT rules, native FA/FLA and original signed-RISE/positive-MAS evaluator. A scope
change is an objective ablation, not proof of an implementation error. It does
not reproduce FT's attention-weighted seed. No FT or generation calls.
"""
import argparse
import copy
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
    for name in ('release','environment','reference','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--reference-sha256',required=True)
    args=p.parse_args();sha=lambda b:hashlib.sha256(b).hexdigest()
    assert sha(args.reference.read_bytes())==args.reference_sha256
    reference=json.loads(args.reference.read_bytes())
    assert reference['status']=='complete' and reference['family']=='qwen35'
    rows=[next(r for r in reference['cases'] if r['dataset']==task and r['index']==i)
          for task in ('niah_mq_q2','morehopqa') for i in (0,1)]
    protocol=json.loads((args.release/'experiments/official/protocol.json').read_bytes())
    env=json.loads(args.environment.read_bytes())['qwen35'];official=Path(env['official_root'])
    for path,digest in protocol['official_normalized_sources'].items():
        assert sha((official/path).read_bytes().replace(b'\r\n',b'\n'))==digest
    sources=json.loads((args.release/'deltatrace/clean/sources.json').read_bytes())
    for name,receipt in sources['models']['qwen35']['files'].items():
        assert sha((args.release/name).read_bytes())==receipt['sha256']
    for task in ('niah_mq_q2','morehopqa'):
        assert sha((official/'exp/exp2/data'/(task+'.jsonl')).read_bytes())==protocol['tasks'][task]['cache_sha256']
    os.environ.update(MACA_PATH='/opt/maca',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
                      TRITON_ENABLE_PERSISTENT_AUTOTUNE_CONFIGS='0')
    os.environ.setdefault('TRITON_CACHE_DIR','/tmp/deltatrace_clean_v1_triton')
    os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR','/tmp/deltatrace_clean_v1_inductor')
    sys.path.insert(0,str(official))
    for overlay in env.get('dependency_overlays',[]):sys.path.insert(0,overlay)
    sys.path.insert(0,str(args.release/'deltatrace/clean/qwen35'))
    sys.path.insert(0,str(args.release/'experiments/official'))
    import numpy as np
    import torch
    from transformers import AutoTokenizer,Qwen3_5ForConditionalGeneration
    from exp.exp2 import dataset_utils
    from llm_attr_eval import LLMAttributionEvaluator
    import ft_ifr_improve as ft
    from batching import make_accelerated_runner
    from qwen35_answer_finite import PackedAnswerTargets
    from finite_fla_gpu import make_compiled_finite_pullback,verify_native_sources
    from vendor_fa_finite_bf16_d256 import VendorFAFiniteP1BF16D256
    from score_views import signed_rise_equals_positive_curve
    torch.set_num_threads(4);torch.manual_seed(73);torch.backends.cuda.matmul.allow_tf32=False
    torch._dynamo.config.cache_size_limit=max(64,torch._dynamo.config.cache_size_limit)
    torch._dynamo.config.accumulated_cache_size_limit=max(256,torch._dynamo.config.accumulated_cache_size_limit)
    args.output.mkdir(parents=True,exist_ok=False)
    report={'status':'loading','driver_sha256':sha(Path(__file__).read_bytes()),'reference_sha256':args.reference_sha256,
            'cases':[],'calls':[],'generation_calls':0,'FT_calls':0,'metric_root_calls':0,
            'sample_batch':1,'endpoint_batch':2,'interpretation':'Objective ablation only; finite rules unchanged; not an FT seed reproduction.'}
    vectors={}
    def save():
        tmp=args.output/'results.partial';tmp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');tmp.replace(args.output/'results.json')
    def timed(name,fn):
        report['status']=name;save();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        row={'name':name,'status':'entered'};report['calls'].append(row);start=time.perf_counter()
        try:
            result=fn();torch.cuda.synchronize();row['status']='returned';return result
        finally:
            row.update(seconds=time.perf_counter()-start,peak_allocated=torch.cuda.max_memory_allocated());save()
    try:
        tokenizer=AutoTokenizer.from_pretrained(env['checkpoint'],local_files_only=True);tokenizer.pad_token=tokenizer.eos_token
        model=timed('model_load',lambda:Qwen3_5ForConditionalGeneration.from_pretrained(env['checkpoint'],dtype=torch.bfloat16,
                    attn_implementation='eager',device_map={'':'cuda:0'},local_files_only=True))
        model.eval().requires_grad_(False)
        assert sha(Path(inspect.getfile(type(model))).read_bytes())==env['native_model_sha256']
        identities={n:(type(m).forward,m.forward) for n,m in model.named_modules()}
        evaluator=LLMAttributionEvaluator(model,tokenizer)
        def check_identity():
            for n,m in model.named_modules():assert (type(m).forward,m.forward)==identities[n],n
        first=torch.tensor(rows[0]['input_ids'],device='cuda')[None]
        with torch.no_grad():initial=timed('native_initialization',lambda:model(input_ids=first,attention_mask=torch.ones_like(first),use_cache=False))
        del initial,first
        verify_native_sources(env['native_stage_source_sha256'])
        runner,report['acceleration_sources']=make_accelerated_runner(args.release,model,
            VendorFAFiniteP1BF16D256(env['finite_library'],env['finite_library_sha256']),make_compiled_finite_pullback(reuse_scalar_products=False))
        for row in rows:
            key=f"{row['dataset']}_{row['index']}"
            ex=copy.deepcopy(dataset_utils.load_cached(official/'exp/exp2/data'/(row['dataset']+'.jsonl'))[row['index']])
            ex.indices_to_explain=ex.sink_span=ex.thinking_span=None
            ex=dataset_utils.attach_spans_from_answer(ex,tokenizer)
            ids=torch.tensor(row['input_ids'],device='cuda')[None];assert sha(ids.cpu().numpy().tobytes())==row['input_sha256']
            target=ids[0,row['prompt_length']:];assert len(target)==row['target_length']
            assert torch.equal(target,tokenizer(ex.target+tokenizer.eos_token,add_special_tokens=False,return_tensors='pt').input_ids[0].to('cuda'))
            eligible=[row['user_positions'][j] for j in row['keep']]
            offsets={'whole':list(range(len(target))),'answer':list(range(ex.sink_span[0],ex.sink_span[1]+1))}
            assert offsets['answer'] and set(offsets['answer'])<set(offsets['whole'])
            rec={'case':key,'input_sha256':row['input_sha256'],'sink_span':list(ex.sink_span),'target_offsets':offsets,
                 'metrics':{},'details':{},'DT_root_calls':[]}
            report['cases'].append(rec)
            pair=ids.repeat(2,1);pair[0,eligible]=tokenizer.eos_token_id
            for scope in ('whole','answer'):
                model.set_attn_implementation('flash_attention_2')
                selection=PackedAnswerTargets([{'target_ids':target.cpu(),'prompt_length':row['prompt_length']}],[offsets[scope]],ids.shape[1],model.device)
                roots=[]
                def root_hook(_module,args,kwargs):
                    assert torch.equal(kwargs['input_ids'],pair) and torch.equal(kwargs['attention_mask'],torch.ones_like(pair))
                    roots.append({'input_sha256':sha(pair.cpu().numpy().tobytes()),'shape':list(pair.shape)})
                handle=model.register_forward_pre_hook(root_hook,with_kwargs=True)
                try:signed,detail=timed(key+'_'+scope+'_DT',lambda:runner.attribute(pair,torch.ones_like(pair),selection,select_output_rows=True,observer=None))
                finally:handle.remove()
                assert len(roots)==1;rec['DT_root_calls'].append({'scope':scope,**roots[0]})
                signed=signed[0].cpu();assert torch.isfinite(signed).all()
                vectors[key+'_'+scope+'_signed_full']=signed.numpy();rec['details'][scope]=detail
                scores=signed[row['user_positions']].float();model.set_attn_implementation('eager')
                curves={}
                for view,score in (('positive',scores.clamp_min(0)),('signed',scores)):
                    if view=='signed':
                        proof=signed_rise_equals_positive_curve(scores,row['keep'],curves['positive'])
                        if proof is not None:
                            curves['signed_reuse_proof']=proof;curves['rise']=curves['positive']['rise'];break
                    curve={'actual_input_hashes':[],'deleted_user_indices':[]}
                    def score_hook(_module,args,kwargs):
                        actual=kwargs['input_ids'].detach().cpu();original=ids.cpu()
                        assert actual.shape==original.shape and torch.equal(actual[:,row['prompt_length']:],original[:,row['prompt_length']:])
                        changed=(actual[0]!=original[0]).nonzero().flatten().tolist()
                        assert set(changed)<=set(eligible) and all(int(actual[0,j])==tokenizer.eos_token_id for j in changed)
                        curve['actual_input_hashes'].append(sha(actual.numpy().tobytes()))
                        curve['deleted_user_indices'].append([row['user_positions'].index(j) for j in changed]);report['metric_root_calls']+=1
                    def observe(frame,event,result):
                        if frame.f_code is ft.faithfulness_test_skip_tokens.__code__ and event=='return' and result is not None:
                            for field in ('scores','density','normalized_model_response','alignment_penalty','corrected_scores'):
                                curve[field]=np.asarray(frame.f_locals[field]).tolist()
                    handle=model.register_forward_pre_hook(score_hook,with_kwargs=True);assert sys.getprofile() is None;sys.setprofile(observe)
                    try:
                        with torch.no_grad():values=timed(key+'_'+scope+'_'+view+'_metric',lambda:ft.faithfulness_test_skip_tokens(
                            evaluator,score[None],ex.prompt,ex.target,keep_prompt_token_indices=row['keep'],user_prompt_indices=row['user_positions'],k=20))
                    finally:sys.setprofile(None);handle.remove()
                    assert len(curve['actual_input_hashes'])==21 and curve['actual_input_hashes'][0]==row['input_sha256']
                    curve.update(zip(('rise','mas','rise_plus_ap'),map(float,values)));curve['MAS_valid']=view=='positive';curves[view]=curve
                    if view=='signed':curves['rise']=curve['rise']
                curves['mas']=curves['positive']['mas']
                curves['needle']=float(ft.evaluate_attr_recovery_skip_tokens(scores.clamp_min(0)[None],keep_prompt_token_indices=row['keep'],gold_prompt_token_indices=row['gold'],top_fraction=.1)) if row['gold'] else None
                rec['metrics'][scope]=curves;check_identity();np.savez_compressed(args.output/'vectors.npz',**vectors);save()
                print(json.dumps({'case':key,'scope':scope,**{k:curves[k] for k in ('rise','mas','needle')}}),flush=True)
            assert rec['DT_root_calls'][0]['input_sha256']==rec['DT_root_calls'][1]['input_sha256']
        report['status']='complete';report['vectors_sha256']=sha((args.output/'vectors.npz').read_bytes())
    except Exception:
        report['status']='failed';report['error']=traceback.format_exc();raise
    finally:save()


if __name__=='__main__':main()

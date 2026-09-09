"""Run the author's unchanged RISE on saved signed DT, with no new attribution.

The same function also returns MAS. Its signed-input MAS return is explicitly
discarded; the existing positive-view MAS remains the relevant MAS result.
"""
import argparse,hashlib,inspect,json,os,sys,time,traceback
from pathlib import Path
sha=lambda b:hashlib.sha256(b).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('release','environment','protocol','output'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();plan=json.loads(args.protocol.read_bytes())
    assert sha(Path(__file__).read_bytes())==plan['driver_sha256']
    env=json.loads(args.environment.read_bytes())['qwen35']
    original=json.loads((args.release/'experiments/official/protocol.json').read_bytes())
    for path,digest in original['official_normalized_sources'].items():
        assert sha((Path(env['official_root'])/path).read_bytes().replace(b'\r\n',b'\n'))==digest
    os.environ.update(MACA_PATH='/opt/maca',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    sys.path.insert(0,env['official_root'])
    for overlay in env.get('dependency_overlays',[]):sys.path.insert(0,overlay)
    import numpy as np
    import torch
    from transformers import AutoTokenizer,Qwen3_5ForConditionalGeneration
    from exp.exp2 import dataset_utils
    import ft_ifr_improve as ft
    import llm_attr_eval
    for module in (dataset_utils,ft,llm_attr_eval):assert Path(module.__file__).resolve().is_relative_to(Path(env['official_root']).resolve())
    refs=[];vectors={}
    for source in plan['references']:
        path=Path(source['path']);assert sha(path.read_bytes())==source['sha256']
        vp=path.parent/'vectors.npz';assert sha(vp.read_bytes())==source['vectors_sha256']
        refs.extend(c for c in json.loads(path.read_bytes())['cases'] if c['status']=='complete')
        vectors.update(dict(np.load(vp,allow_pickle=False)))
    selected=[next(c for c in refs if [c['dataset'],c['index']]==key) for key in plan['cases']]
    torch.set_num_threads(4);torch.manual_seed(73);torch.backends.cuda.matmul.allow_tf32=False
    args.output.mkdir(parents=True,exist_ok=False)
    report={'status':'loading','protocol_sha256':sha(args.protocol.read_bytes()),'plan':plan,
        'cases':[],'calls':[],'DT_calls':0,'FT_calls':0,'generation_calls':0,'sample_batch':1}
    def save():
        temp=args.output/'results.partial';temp.write_text(json.dumps(report,indent=2,allow_nan=False));temp.replace(args.output/'results.json')
    def timed(name,fn):
        report['status']=name;save();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
        call={'name':name,'status':'entered'};report['calls'].append(call)
        try:
            result=fn();torch.cuda.synchronize();call['status']='returned';return result
        finally:
            call.update(seconds=time.perf_counter()-t,peak_allocated=torch.cuda.max_memory_allocated());save()
    save()
    try:
        tokenizer=AutoTokenizer.from_pretrained(env['checkpoint'],local_files_only=True);tokenizer.pad_token=tokenizer.eos_token
        model=timed('model_load',lambda:Qwen3_5ForConditionalGeneration.from_pretrained(env['checkpoint'],
            dtype=torch.bfloat16,attn_implementation='eager',device_map={'':'cuda:0'},local_files_only=True))
        model.eval().requires_grad_(False)
        assert sha(Path(inspect.getfile(type(model))).read_bytes())==env['native_model_sha256']
        methods={n:(type(m).forward,m.forward) for n,m in model.named_modules()}
        evaluator=llm_attr_eval.LLMAttributionEvaluator(model,tokenizer)
        for c in selected:
            key=f"{c['dataset']}_{c['index']}";path=Path(env['official_root'])/'exp/exp2/data'/(c['dataset']+'.jsonl')
            assert sha(path.read_bytes())==original['tasks'][c['dataset']]['cache_sha256']
            ex=dataset_utils.load_cached(path)[c['index']]
            ids=torch.tensor(c['input_ids'],dtype=torch.long);positions=c['user_positions'];keep=c['keep']
            signed=torch.tensor(vectors[key+'_DT_signed_full'][positions],dtype=torch.float32)
            row={'dataset':c['dataset'],'index':c['index'],'input_sha256':c['input_sha256'],
                 'curve':{'actual_input_hashes':[],'deleted_user_indices':[]}}
            report['cases'].append(row);curve=row['curve']
            def inputs(_model,_args,kwargs):
                actual=kwargs['input_ids'].detach().cpu()[0]
                changed=(actual!=ids).nonzero().flatten().tolist()
                assert set(changed)<={positions[i] for i in keep}
                assert all(int(actual[j])==tokenizer.eos_token_id for j in changed)
                assert torch.equal(actual[c['prompt_length']:],ids[c['prompt_length']:])
                curve['actual_input_hashes'].append(sha(actual.numpy().tobytes()))
                curve['deleted_user_indices'].append([positions.index(j) for j in changed])
            def observe(frame,event,value):
                if frame.f_code is ft.faithfulness_test_skip_tokens.__code__ and event=='return' and value is not None:
                    for name in ('scores','normalized_model_response'):
                        curve[name]=np.asarray(frame.f_locals[name]).tolist()
            handle=model.register_forward_pre_hook(inputs,with_kwargs=True)
            assert sys.getprofile() is None;sys.setprofile(observe)
            try:
                with torch.no_grad():values=timed(key+'_signed_original_RISE',lambda:ft.faithfulness_test_skip_tokens(
                    evaluator,signed[None],ex.prompt,ex.target,keep_prompt_token_indices=keep,user_prompt_indices=positions,k=20))
            finally:sys.setprofile(None);handle.remove()
            for n,m in model.named_modules():assert (type(m).forward,m.forward)==methods[n]
            assert len(curve['actual_input_hashes'])==21
            assert curve['actual_input_hashes'][0]==c['input_sha256']
            assert curve['actual_input_hashes'][-1]==c['metrics']['DT']['actual_input_hashes'][-1]
            row['signed_RISE']=float(values[0]);row['positive_RISE']=c['metrics']['DT']['rise']
            row['signed_minus_positive']=row['signed_RISE']-row['positive_RISE']
            row['first_input_order_difference']=next((i for i,(a,b) in enumerate(zip(curve['actual_input_hashes'],c['metrics']['DT']['actual_input_hashes'])) if a!=b),None)
            row['unchanged_endpoint_scores']=[curve['scores'][i]==c['metrics']['DT']['scores'][i] for i in (0,-1)]
            row['status']='complete';save()
            print(json.dumps({k:row[k] for k in ('dataset','index','signed_RISE','positive_RISE','signed_minus_positive','first_input_order_difference')}),flush=True)
        report['status']='complete'
    except Exception:
        report['status']='failed';report['error']=traceback.format_exc();raise
    finally:save()

if __name__=='__main__':main()

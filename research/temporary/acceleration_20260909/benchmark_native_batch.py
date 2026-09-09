"""Bounded real-case DT timing and batch validation using frozen native operators.

Loads NI0/MH0 input receipts from the author-aligned run. The candidate changes
checkpoint storage and native varlen plumbing only. No FT, model or metric code
is replaced. All model calls, cold attempts and measured calls are recorded.
"""
import argparse,hashlib,importlib.util,inspect,json,os,sys,time,traceback
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--candidate-sha256',required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--reference-sha256',required=True)
    p.add_argument('--phase',choices=['pilot','regression16'],default='pilot')
    p.add_argument('--reference-extra',type=Path)
    p.add_argument('--reference-extra-sha256')
    p.add_argument('--environment',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();sha=lambda b:hashlib.sha256(b).hexdigest()
    assert sha(args.reference.read_bytes())==args.reference_sha256
    assert sha(args.candidate.read_bytes())==args.candidate_sha256
    reference=json.loads(args.reference.read_bytes())
    assert reference['family']=='qwen35'
    reference_cases=[r for r in reference['cases'] if r['status']=='complete']
    if args.phase=='pilot':assert reference['status']=='complete'
    else:
        assert args.reference_extra is not None and sha(args.reference_extra.read_bytes())==args.reference_extra_sha256
        extra=json.loads(args.reference_extra.read_bytes());assert extra['status']=='complete' and extra['family']=='qwen35'
        assert reference['clean_sources_sha256']==extra['clean_sources_sha256']
        assert reference['protocol_sha256']==extra['protocol_sha256']
        reference_cases.extend(r for r in extra['cases'] if r['status']=='complete')
        assert len(reference_cases)==16 and {(r['dataset'],r['index']) for r in reference_cases}=={(t,i) for t in ('niah_mq_q2','morehopqa') for i in range(8)}
    manifest=json.loads((args.release/'deltatrace/clean/sources.json').read_bytes())
    for name,receipt in manifest['models']['qwen35']['files'].items():
        assert sha((args.release/name).read_bytes())==receipt['sha256']
    env=json.loads(args.environment.read_bytes())['qwen35']
    os.environ.update(MACA_PATH='/opt/maca',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    os.environ.setdefault('TRITON_CACHE_DIR','/tmp/deltatrace_clean_v1_triton')
    os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR','/tmp/deltatrace_clean_v1_inductor')
    sys.path.insert(0,env['official_root'])
    for path in env.get('dependency_overlays',[]):sys.path.insert(0,path)
    sys.path.insert(0,str(args.release/'deltatrace/clean/qwen35'))
    sys.path.insert(0,str(args.candidate.parent))
    import numpy as np
    import torch
    from torch._dynamo.utils import counters
    from transformers import AutoTokenizer,Qwen3_5ForConditionalGeneration
    from qwen35_dense_finite_runner import Qwen35DenseFiniteRunner
    from qwen35_answer_finite import PackedAnswerTargets
    from finite_fla_gpu import make_compiled_finite_pullback,verify_native_sources
    from vendor_fa_finite_bf16_d256 import VendorFAFiniteP1BF16D256
    import ft_ifr_improve as ft
    from llm_attr_eval import LLMAttributionEvaluator
    from exp.exp2 import dataset_utils
    from dynamic_finite import configure_dynamic_finite
    torch.set_num_threads(4);torch.manual_seed(73);torch.backends.cuda.matmul.allow_tf32=False
    spec=importlib.util.spec_from_file_location('explicit_DT_storage_batch_candidate',args.candidate)
    candidate=importlib.util.module_from_spec(spec);spec.loader.exec_module(candidate)
    args.output.mkdir(parents=True,exist_ok=False)
    report={'status':'loading','candidate_sha256':args.candidate_sha256,'reference_sha256':args.reference_sha256,
            'phase':args.phase,'reference_extra_sha256':args.reference_extra_sha256,
            'calls':[],'actual_root_calls':[],'metrics':{},'vectors':{},'cold_and_diagnostics_included':True}
    vectors={}
    def save():
        tmp=args.output/'results.partial';tmp.write_text(json.dumps(report,indent=2,allow_nan=False));tmp.replace(args.output/'results.json')
    def compiler_counts():return {k:dict(counters[k]) for k in ('stats','frames','inductor')}
    def timed(name,fn):
        report['status']=name;save();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        row={'name':name,'status':'entered','compiler_before':compiler_counts()};report['calls'].append(row)
        start=time.perf_counter()
        try:
            result=fn();torch.cuda.synchronize();row['status']='returned';return result
        finally:
            row.update(seconds=time.perf_counter()-start,peak_allocated=torch.cuda.max_memory_allocated(),
                       peak_reserved=torch.cuda.max_memory_reserved(),compiler_after=compiler_counts());save()
    save()
    try:
        tokenizer=AutoTokenizer.from_pretrained(env['checkpoint'],local_files_only=True)
        tokenizer.pad_token=tokenizer.eos_token
        model=timed('model_load',lambda:Qwen3_5ForConditionalGeneration.from_pretrained(env['checkpoint'],dtype=torch.bfloat16,
                    attn_implementation='eager',device_map={'':'cuda:0'},local_files_only=True))
        model.eval().requires_grad_(False)
        assert sha(Path(inspect.getfile(type(model))).read_bytes())==env['native_model_sha256']
        original_forwards={name:(type(module).forward,module.forward) for name,module in model.named_modules()}
        def identity():
            for name,module in model.named_modules():
                assert (type(module).forward,module.forward)==original_forwards[name]
        rows=([next(r for r in reference_cases if r['dataset']==task and r['index']==0) for task in ('niah_mq_q2','morehopqa')]
              if args.phase=='pilot' else sorted(reference_cases,key=lambda r:(r['dataset'],r['index'])))
        report['cases']=[{k:r[k] for k in ('dataset','index','input_sha256','prompt_length','target_length')} for r in rows]
        original_inputs=[torch.tensor(r['input_ids'],device='cuda')[None] for r in rows]
        with torch.no_grad():initial=timed('native_eager_initialization',lambda:model(input_ids=original_inputs[0],attention_mask=torch.ones_like(original_inputs[0]),use_cache=False))
        del initial;model.set_attn_implementation('flash_attention_2')
        verify_native_sources(env['native_stage_source_sha256'])
        finite_fa=VendorFAFiniteP1BF16D256(env['finite_library'],env['finite_library_sha256'])
        finite_fla=make_compiled_finite_pullback(reuse_scalar_products=False)
        runners={'baseline':Qwen35DenseFiniteRunner(model,finite_fa,finite_fla),
                 'cpu':candidate.Qwen35DenseFiniteRunner(model,finite_fa,finite_fla,checkpoint_device='cpu'),
                 'gpu':candidate.Qwen35DenseFiniteRunner(model,finite_fa,finite_fla,checkpoint_device='cuda'),
                 'dynamic':configure_dynamic_finite(candidate.Qwen35DenseFiniteRunner(model,finite_fa,finite_fla,checkpoint_device='cpu')),
                 'dynamic_gpu':configure_dynamic_finite(candidate.Qwen35DenseFiniteRunner(model,finite_fa,finite_fla,checkpoint_device='cuda'))}
        def pack(indices):
            length=max(len(rows[i]['input_ids']) for i in indices)
            ids=torch.full((2*len(indices),length),tokenizer.eos_token_id,device='cuda',dtype=torch.long)
            mask=torch.zeros_like(ids);cases=[];offsets=[]
            for j,i in enumerate(indices):
                row=rows[i];n=len(row['input_ids']);ids[2*j:2*j+2,:n]=original_inputs[i]
                eligible=[row['user_positions'][k] for k in row['keep']]
                ids[2*j,eligible]=tokenizer.eos_token_id;mask[2*j:2*j+2,:n]=1
                cases.append({'target_ids':original_inputs[i][0,row['prompt_length']:].cpu(),'prompt_length':row['prompt_length']})
                offsets.append(list(range(n-row['prompt_length'])))
            return ids,mask,PackedAnswerTargets(cases,offsets,length,'cuda')
        def attribute(name,mode,indices,profile=False):
            ids,mask,selection=pack(indices)
            def observe(_model,call_args,kwargs):
                assert torch.equal(kwargs['input_ids'],ids) and torch.equal(kwargs['attention_mask'],mask)
                report['actual_root_calls'].append({'name':name,'sample_batch':len(indices),'endpoint_batch':len(ids),
                    'lengths':[len(rows[i]['input_ids']) for i in indices],'input_sha256':sha(ids.cpu().numpy().tobytes())})
            handle=model.register_forward_pre_hook(observe,with_kwargs=True)
            try:
                if profile:
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
                        signed,info=timed(name,lambda:runners[mode].attribute(ids,mask,selection))
                    report['CPU_profile']=[{'key':e.key,'count':e.count,'self_cpu_time_total_us':e.self_cpu_time_total}
                        for e in prof.key_averages()]
                else:signed,info=timed(name,lambda:runners[mode].attribute(ids,mask,selection))
            finally:handle.remove()
            identity();assert torch.isfinite(signed).all()
            report['calls'][-1]['detail']=info
            for j,i in enumerate(indices):
                n=len(rows[i]['input_ids']);assert bool(signed[j,n:].eq(0).all())
                vector=signed[j,:n].numpy();key=name+'_'+str(i);vectors[key]=vector
                report['vectors'][key]={'shape':list(vector.shape),'sha256':sha(vector.tobytes())}
            np.savez_compressed(args.output/'vectors.npz',**vectors);save()
            return signed
        metric_inputs={}
        if args.phase=='pilot':
            # Different storage strategies use exactly the same finite operators.
            for i in range(2):
                attribute(f'case{i}_baseline_cold','baseline',[i])
                for repeat,mode in enumerate(('baseline','gpu','gpu','baseline')):
                    attribute(f'case{i}_{mode}_measured{repeat}',mode,[i])
            attribute('batch2_cpu_cold','cpu',[0,1])
            for repeat,mode in enumerate(('cpu','gpu','gpu','cpu')):
                attribute(f'batch2_{mode}_measured{repeat}',mode,[0,1])
            attribute('case0_baseline_CPU_profile','baseline',[0],profile=True)
            for i in range(2):
                attribute(f'case{i}_dynamic_first','dynamic',[i])
                attribute(f'case{i}_dynamic_repeat','dynamic',[i])
            attribute('batch2_dynamic_first','dynamic',[0,1])
            attribute('batch2_dynamic_repeat','dynamic',[0,1])
            metric_inputs={i:[('single',f'case{i}_baseline_measured3_{i}'),('batch',f'batch2_cpu_measured3_{i}'),
                               ('dynamic_batch',f'batch2_dynamic_repeat_{i}')] for i in range(2)}
        else:
            # Pair only by input length, before inspecting attribution results.
            order=sorted(range(len(rows)),key=lambda i:(len(rows[i]['input_ids']),i))
            pairs=[order[j:j+2] for j in range(0,len(order),2)]
            report['batch_assignment']=pairs;save()
            attribute('regression_batch_warmup','dynamic_gpu',pairs[0])
            for b,indices in enumerate(pairs):
                name=f'regression_batch{b}'
                attribute(name,'dynamic_gpu',indices)
                for i in indices:metric_inputs[i]=[('accelerated',name+'_'+str(i))]
        model.set_attn_implementation('eager');evaluator=LLMAttributionEvaluator(model,tokenizer)
        for i,row in enumerate(rows):
            ex=dataset_utils.load_cached(Path(env['official_root'])/'exp/exp2/data'/(row['dataset']+'.jsonl'))[row['index']]
            report['metrics'][f'case{i}_clean_reference']={k:row['metrics']['DT'][k] for k in ('rise','mas','needle')}
            for mode,key in metric_inputs[i]:
                scores=torch.tensor(vectors[key][row['user_positions']],dtype=torch.float32).clamp_min(0)
                def observe_score(_model,call_args,kwargs):
                    actual=kwargs['input_ids'];base=original_inputs[i]
                    assert actual.shape==base.shape and torch.equal(actual[:,row['prompt_length']:],base[:,row['prompt_length']:])
                    changed=(actual[0]!=base[0]).nonzero().flatten().tolist()
                    assert set(changed)<=set(row['user_positions'][j] for j in row['keep'])
                    assert all(int(actual[0,j])==tokenizer.eos_token_id for j in changed)
                handle=model.register_forward_pre_hook(observe_score,with_kwargs=True)
                try:
                    with torch.no_grad():values=timed(f'case{i}_{mode}_original_metrics',lambda:ft.faithfulness_test_skip_tokens(
                        evaluator,scores[None],ex.prompt,ex.target,keep_prompt_token_indices=row['keep'],user_prompt_indices=row['user_positions'],k=20))
                finally:handle.remove()
                needle=float(ft.evaluate_attr_recovery_skip_tokens(scores[None],keep_prompt_token_indices=row['keep'],
                    gold_prompt_token_indices=row['gold'],top_fraction=.1)) if row['gold'] else None
                report['metrics'][f'case{i}_{mode}']={'rise':float(values[0]),'mas':float(values[1]),'needle':needle};save()
        identity();report['status']='complete';report['vectors_sha256']=sha((args.output/'vectors.npz').read_bytes())
    except Exception:
        report['status']='failed';report['error']=traceback.format_exc();raise
    finally:save()


if __name__=='__main__':main()

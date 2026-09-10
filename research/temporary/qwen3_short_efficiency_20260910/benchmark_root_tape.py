"""Short-input B1 extension of the untouched, pinned FlashTrace exp1 benchmark.

Reuses the author's length builders, CUDA memory reset, synchronized timing,
runner factory (Qwen3), and JSONL/CSV aggregation. Qwen3.5 calls the previously
pinned official extension explicitly. No attribution implementation is patched.
Each method has its own process/model, so baseline hooks and parameter gradients
cannot contaminate a subsequent method. Model load and runner init are separate.
"""
import argparse
import gc
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--family',choices=['qwen3','qwen35'],required=True)
    p.add_argument('--method',required=True)
    p.add_argument('--lengths',default='128,256,512,1024')
    p.add_argument('--protocol',default='benchmark_protocol.json')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    A=args.root; R=A/'release'; out=args.output
    out.mkdir(parents=True,exist_ok=False)
    sha=lambda b:hashlib.sha256(b).hexdigest()
    protocol_path=A/args.protocol
    protocol=json.loads(protocol_path.read_bytes())
    for name,digest in protocol['candidate_files'].items():assert sha((Path(__file__).resolve().parent/name).read_bytes())==digest,name
    env=json.loads((R/'environment.json').read_bytes())[args.family]
    report={'status':'imports','family':args.family,'method':args.method,
      'driver_sha256':sha(Path(__file__).read_bytes()),'protocol_sha256':sha(protocol_path.read_bytes()),
      'sample_batch':1,'generation_calls':0,'cases':[],'rows':[],'initialization':[],
      'purpose':'Qwen3 native root activation retention: avoid decoder replay, preserve finite formulas and diagnostics',
      'timing_scope':'Unmodified author measure(runner): complete attribution call, synchronized CUDA, result destruction included. Model load and runner construction excluded.',
      'warm_timing_scope':'One warm call per mode and length, then retained/candidate/candidate/retained/retained/candidate. Three full measured calls per mode. Original reset before each call.',
      'audit_scope':'One separately charged call after timing per successful cell retains the returned vector for validation; input observation is enabled only for this audit call.'}
    def save():
        tmp=out/'results.partial';tmp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');tmp.replace(out/'results.json')
    save()
    os.environ.update(MACA_PATH='/opt/maca',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1')
    if args.family=='qwen35':os.environ['TRITON_ENABLE_PERSISTENT_AUTOTUNE_CONFIGS']='0'
    os.environ.setdefault('TRITON_CACHE_DIR',str(A/('triton_'+args.family)))
    os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR',str(A/('inductor_'+args.family)))
    sys.path.insert(0,str(A/'deps'))
    sys.path.insert(0,env['official_root'])
    sys.path.insert(0,str(R/'deltatrace/clean'/args.family))
    sys.path.insert(0,str(R/'deltatrace/accelerated'))
    try:
        for name,digest in protocol['original_normalized_sources'].items():
            assert sha((Path(env['official_root'])/name).read_bytes().replace(b'\r\n',b'\n'))==digest,name
        source=R/'official_exp1/run_time_curve.py'
        assert sha(source.read_bytes())==protocol['exp1_sha256']
        spec=importlib.util.spec_from_file_location('pinned_author_exp1',source)
        bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench)
        import numpy as np
        import torch
        import llm_attr
        import ft_ifr_improve as ft
        import transformers
        assert Path(llm_attr.__file__).resolve().is_relative_to(Path(env['official_root']).resolve())
        torch.set_num_threads(4);random.seed(42);np.random.seed(42);torch.manual_seed(42)
        torch.backends.cuda.matmul.allow_tf32=False
        torch._dynamo.config.cache_size_limit=max(64,torch._dynamo.config.cache_size_limit)
        torch._dynamo.config.accumulated_cache_size_limit=max(256,torch._dynamo.config.accumulated_cache_size_limit)
        if args.family=='qwen35':
            from finite_fla_gpu import verify_native_sources
            # Gate every method before a model load, not only DT initialization.
            verify_native_sources(env['native_stage_source_sha256'])
            report['native_FLA_sha256']=env['native_stage_source_sha256']
            report['native_autotune_environment']={k:os.environ.get(k) for k in
              ['TRITON_ENABLE_PERSISTENT_AUTOTUNE_CONFIGS','TRITON_AUTOTUNE_CONFIG_PATH']}
        report['environment']={'torch':torch.__version__,'transformers':transformers.__version__,
          'device':torch.cuda.get_device_name(),'torch_threads':torch.get_num_threads(),
          'checkpoint':env['checkpoint'],'native_model_sha256':env['native_model_sha256'],
          'exp1_sha256':sha(source.read_bytes()),'seed':42}
        start=time.perf_counter();report['status']='model_load';save()
        if args.family=='qwen3':
            model,tokenizer=bench.load_model_balanced(env['checkpoint'],'cuda:0')
        else:
            from transformers import AutoTokenizer,Qwen3_5ForConditionalGeneration
            tokenizer=AutoTokenizer.from_pretrained(env['checkpoint'],local_files_only=True)
            tokenizer.pad_token=tokenizer.eos_token
            model=Qwen3_5ForConditionalGeneration.from_pretrained(env['checkpoint'],dtype=torch.bfloat16,
                attn_implementation='eager',device_map={'':'cuda:0'},local_files_only=True)
            sys.path.append(env['ft_extension_root'])
            for name,digest in env['official_extension_blob_sha1'].items():
                raw=(Path(env['ft_extension_root'])/name).read_bytes()
                assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==digest,name
        torch.cuda.synchronize();report['model_load_seconds']=time.perf_counter()-start
        assert sha(Path(inspect.getfile(type(model))).read_bytes())==env['native_model_sha256']
        report['environment']['dtype']=str(next(model.parameters()).dtype)
        report['environment']['model_parameter_count']=sum(x.numel() for x in model.parameters())
        model.eval()
        native_forwards={n:(type(m).forward,m.forward) for n,m in model.named_modules()}
        def identity():
            for n,m in model.named_modules():assert native_forwards[n]==(type(m).forward,m.forward),n
        is_dt=args.method=='deltatrace_retained'
        if is_dt:
            model.requires_grad_(False)
            if args.family=='qwen3':
                from qwen_signed_secant_paired_public_fa import capture_checkpoint_pair
                from vendor_fa_finite_runtime import VendorFAFiniteP1
                from retained import make_retained_qwen3
                propagate,report['acceleration_sources']=make_retained_qwen3(R)
                finite_fa=VendorFAFiniteP1(env['finite_library'],env['finite_library_sha256'])
            else:
                from finite_fla_gpu import make_compiled_finite_pullback,verify_native_sources
                from vendor_fa_finite_bf16_d256 import VendorFAFiniteP1BF16D256
                from qwen35_answer_finite import PackedAnswerTargets
                from retained_qwen35 import make_retained_qwen35
                verify_native_sources(env['native_stage_source_sha256'])
                finite_fa=VendorFAFiniteP1BF16D256(str(R/'libdeltatrace_fa_finite_bf16_d256.so'),env['finite_library_sha256'])
                finite_fla=make_compiled_finite_pullback(reuse_scalar_products=False)
                dt_runner,report['acceleration_sources']=make_retained_qwen35(R,model,finite_fa,finite_fla)
        # Preserve the original script's explicit fallback when the default data is absent.
        ruler=Path(env.get('ft_extension_root','/mnt/geogpt-doc-new/deepresearch/DiffFlashTrace/upstream/flashtrace'))/'data/ruler_multihop/8192/vt_h10_c1/validation.jsonl'
        base=bench.load_ruler_base(ruler,fallback='RULER fallback text. ')
        report['base_text']={'default_ruler_file_exists':ruler.exists(),'used_author_fallback':not ruler.exists(),
          'sha256':sha(base.encode()),'text':base if not ruler.exists() else None}
        report['status']='ready';save()
        from qwen3_root_tape import NativeRootTape,propagate_root_tape
        vectors={}
        for input_len in map(int,args.lengths.split(',')):
            assert 0<input_len<=1024
            prompt,actual_input=bench.build_prompt_to_length(tokenizer,base,input_len)
            target,actual_output=bench.build_output_to_length(tokenizer,' The answer is 42.',32)
            lens=bench.estimate_model_lengths(tokenizer,prompt,target)
            engine=llm_attr.LLMIFRAttribution(model,tokenizer)
            ids,mask,prompt_len,gen_len=engine._ensure_generation(prompt,target)
            positions=list(engine.user_prompt_indices)
            keep=ft.keep_token_indices(engine.user_prompt_tokens)
            eligible=[positions[j] for j in keep]
            formatted=bench.build_formatted_prompt(tokenizer,prompt)
            assert engine.prompt==formatted
            expected_ids=ids.detach().cpu();expected_hash=sha(expected_ids.numpy().tobytes())
            assert lens['total_tokens']==ids.shape[1] and lens['generation_tokens']==gen_len
            case={'input_length':input_len,'output_length':32,'prompt':prompt,'target':target,
              'lengths':lens,'input_ids':expected_ids[0].tolist(),'input_sha256':expected_hash,
              'user_positions':positions,'eligible_positions':eligible,'generation_length_includes_eos':gen_len,
              'author_IG_internal_batch_argument':bench.compute_batch_size(lens['total_tokens'],getattr(model.config,'max_position_embeddings',None) or 200000)}
            report['cases'].append(case)
            if is_dt and args.family=='qwen35' and not report.get('native_initialized'):
                start=time.perf_counter()
                with torch.no_grad():initial=model(input_ids=ids,attention_mask=mask,use_cache=False)
                del initial;torch.cuda.synchronize()
                report['initialization'].append({'name':'original_eager_lazy_initialization','seconds':time.perf_counter()-start})
                report['native_initialized']=True
            if is_dt:model.set_attn_implementation('flash_attention_2')
            else:model.set_attn_implementation('eager')
            def make_runner():
                if not is_dt:
                    if args.family=='qwen3':
                        return bench.make_attr_runner(args.method,model,tokenizer,128,32,
                           case['author_IG_internal_batch_argument'],prompt,target)
                    return make_qwen35_runner(args.method,model,tokenizer,formatted,target,
                        case['author_IG_internal_batch_argument'])
                dt_engine=llm_attr.LLMIFRAttribution(model,tokenizer)
                def call():
                    actual,attention,p_len,g_len=dt_engine._ensure_generation(prompt,target)
                    base_ids=actual.clone();base_ids[0,eligible]=tokenizer.eos_token_id
                    if args.family=='qwen3':
                        if mode=='retained':
                            before,after=capture_checkpoint_pair(model,base_ids,actual,attention,p_len)
                            return propagate(model,before,after,pv_rule='content_P1',finite_attention=finite_fa)
                        tape=NativeRootTape(model,mutation_audit=phase=='audit')
                        try:
                            with tape:before,after=capture_checkpoint_pair(model,base_ids,actual,attention,p_len)
                            return propagate_root_tape(model,before,after,tape,pv_rule='content_P1',finite_attention=finite_fa)
                        finally:tape.clear()
                    pair=torch.cat((base_ids,actual))
                    selection=PackedAnswerTargets([{'target_ids':actual[0,p_len:].detach().cpu(),'prompt_length':p_len}],
                        [list(range(g_len))],actual.shape[1],model.device)
                    signed,detail=dt_runner.attribute(pair,torch.ones_like(pair),selection,select_output_rows=True,observer=None)
                    return {'signed_full_sequence':signed[0].cpu().tolist(),'details':detail}
                return call
            modes=['retained','root_tape']
            measured=modes+list(reversed(modes))+modes
            schedule=[('warm',m) for m in modes]+[('measured',m) for m in measured]
            n_runs=len(schedule)
            last_ok=False
            for rep in range(n_runs):
                phase,mode=schedule[rep]
                report['status']=f'{input_len}_{mode}_{phase}_{rep}';save()
                gc.collect();bench.maybe_reset_cuda([0]);started=time.perf_counter()
                runner=None
                try:
                    runner=make_runner();init_seconds=time.perf_counter()-started
                    status,wall,alloc,reserved,by_device=bench.measure(runner,[0],catch_oom=True)
                    error=None
                except Exception:
                    status='init_error';error=traceback.format_exc();wall=alloc=reserved=None;by_device={}
                    init_seconds=time.perf_counter()-started
                finally:runner=None
                row={'family':args.family,'attr_func':args.method,'phase':phase,'repeat':rep,'candidate_mode':mode,
                  'target_input_tokens':input_len,'target_output_tokens':32,'target_total_tokens':input_len+32,
                  'actual_input_tokens':actual_input,'actual_output_tokens':actual_output,
                  'actual_total_tokens':lens['total_tokens'],'actual_formatted_prompt_tokens':prompt_len,
                  'actual_generation_tokens':gen_len,'time_sec':wall,'status':status,'error':error,
                  'peak_mem_gb':reserved if reserved is not None else alloc,'peak_allocated_gb':alloc,
                  'peak_mem_reserved_gb':reserved,'peak_mem_by_device_gb':by_device,'runner_init_seconds':init_seconds,
                  'input_sha256':expected_hash}
                report['rows'].append(row)
                with (out/'time_curve_runs.jsonl').open('a') as f:bench.append_jsonl_row(f,row)
                bench.write_summary_csv([x for x in report['rows'] if x['phase']=='original3'],out)
                save();print(json.dumps({k:row[k] for k in ['family','attr_func','target_input_tokens','phase','repeat','status','time_sec','peak_allocated_gb']}),flush=True)
                last_ok=status=='ok'
                identity()
                if status!='ok':
                    # Exact default methods that cannot initialize/execute are evidence.
                    # Keep the requested repeated failures; do not patch their implementation.
                    gc.collect();torch.cuda.empty_cache()
            if last_ok:
                for mode in modes:
                    phase='audit'
                    report['status']=f'{input_len}_{mode}_audit';save()
                    actual_calls=[]
                    def observe(_m,call_args,kwargs):
                        tensor=kwargs.get('input_ids')
                        if tensor is None:return
                        cpu=tensor.detach().cpu()
                        actual_calls.append({'shape':list(cpu.shape),'input_sha256':sha(cpu.numpy().tobytes())})
                        if is_dt:
                            assert cpu.shape==(2,expected_ids.shape[1]) and torch.equal(cpu[1],expected_ids[0])
                            b=expected_ids.clone();b[0,eligible]=tokenizer.eos_token_id
                            assert torch.equal(cpu[0],b[0])
                        elif args.method.startswith('ifr_'):
                            assert torch.equal(cpu,expected_ids)
                    handle=model.register_forward_pre_hook(observe,with_kwargs=True)
                    started=time.perf_counter();runner=make_runner()
                    try:
                        result=runner();torch.cuda.synchronize()
                        audit_seconds=time.perf_counter()-started
                    finally:handle.remove();runner=None
                    if is_dt:vec=np.asarray(result['signed_full_sequence'],dtype=np.float64)
                    elif hasattr(result,'attribution_matrix'):vec=result.attribution_matrix.detach().float().cpu().numpy()
                    else:vec=np.asarray(result.scores,dtype=np.float64)
                    finite=bool(np.isfinite(vec).all())
                    case.setdefault('audits',{})[mode]={'separately_charged_seconds':audit_seconds,'actual_calls':actual_calls,
                      'vector_shape':list(vec.shape),'vector_sha256':sha(vec.tobytes()),'finite':finite}
                    vectors[str(input_len)+'_'+mode]=vec
                    case.setdefault('details',{})[mode]={k:v for k,v in result.items() if k!='signed_full_sequence'}
                    np.savez(out/'vectors.npz',**vectors)
                    del result,vec
                    identity();save()
                reference=vectors[str(input_len)+'_retained']
                case['candidate_vectors_equal']={mode:bool(np.array_equal(reference,vectors[str(input_len)+'_'+mode])) for mode in modes}
                assert all(case['candidate_vectors_equal'].values()),case['candidate_vectors_equal']
            del ids,mask,engine
            gc.collect();torch.cuda.empty_cache()
        report['status']='complete'
    except Exception:
        report['status']='failed';report['error']=traceback.format_exc()
    finally:
        save();print(json.dumps({'status':report['status'],'error':report.get('error')}),flush=True)


def make_qwen35_runner(method,model,tokenizer,prompt,target,batch):
    """Use the already pinned official Qwen3.5 classes with exp1 parameters."""
    from flashtrace import attribution as lib
    from flashtrace.improved import LLMIFRAttributionBoth
    if method=='IG':
        engine=lib.LLMGradientAttribtion(model,tokenizer)
        return lambda:engine.calculate_IG_per_generation(prompt,steps=20,baseline=tokenizer.eos_token_id,batch_size=batch,target=target)
    if method=='attention_I_G':
        attn=lib.LLMAttentionAttribution(model,tokenizer);ig=lib.LLMGradientAttribtion(model,tokenizer)
        def call():
            a=attn.calculate_attention_attribution(prompt,target=target)
            b=ig.calculate_IG_per_generation(prompt,steps=20,baseline=tokenizer.eos_token_id,batch_size=batch,target=target)
            a.attribution_matrix=a.attribution_matrix*b.attribution_matrix
            return a
        return call
    if method.startswith('perturbation_'):
        engine=lib.LLMPerturbationAttribution(model,tokenizer)
        if method=='perturbation_REAGENT':return lambda:engine.calculate_feature_ablation_sentences_mlm(prompt,target=target)
        measure='KL' if method=='perturbation_CLP' else 'log_loss'
        return lambda:engine.calculate_feature_ablation_sentences(prompt,baseline=tokenizer.eos_token_id,measure=measure,target=target)
    if method=='attnlrp':
        engine=lib.LLMLRPAttribution(model,tokenizer)
        return lambda:engine.calculate_attnlrp(prompt,target=target)
    cls=LLMIFRAttributionBoth if method=='ifr_multi_hop_both' else lib.LLMIFRAttribution
    engine=cls(model,tokenizer,chunk_tokens=128,sink_chunk_tokens=1 if method=='ifr_all_positions' else 32,use_chat_template=False)
    if method=='ifr_multi_hop_both':return lambda:engine.calculate_ifr_multi_hop_both(prompt,target=target)
    if method=='ifr_multi_hop':return lambda:engine.calculate_ifr_multi_hop(prompt,target=target)
    if method=='ifr_all_positions':return lambda:engine.calculate_ifr_for_all_positions(prompt,target=target)
    raise ValueError(method)


if __name__=='__main__':main()

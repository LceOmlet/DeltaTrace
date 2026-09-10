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
    assert args.family=='qwen3'
    R=Path(protocol['release_root'])
    assert sha(Path(__file__).read_bytes())==protocol['driver_sha256']
    for name,digest in protocol['runtime_files'].items():assert sha((R/name).read_bytes())==digest,name
    env=json.loads((R/'environment.json').read_bytes())[args.family]
    report={'status':'imports','family':args.family,'method':args.method,
      'driver_sha256':sha(Path(__file__).read_bytes()),'protocol_sha256':sha(protocol_path.read_bytes()),
      'sample_batch':1,'generation_calls':0,'cases':[],'rows':[],'initialization':[],
      'timing_scope':'Unmodified author measure(runner): complete attribution call, synchronized CUDA, result destruction included. Model load and runner construction excluded.',
      'warm_timing_scope':'Two explicitly recorded warm calls per length, then three measured complete calls; unchanged author reset/empty_cache before every call. Two serial independent process rounds reverse method and length order.',
      'audit_scope':'One separately charged output audit per cell; DT additionally executes one unchanged retained reference after timing in the same process. Historical-cache differences are reported. Input observers run only in these audit calls.'}
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
    import native_capture_events
    assert Path(native_capture_events.__file__).resolve()==R/'deltatrace/accelerated/native_capture_events.py'
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
        is_dt=args.method in ('deltatrace_graphed','deltatrace_streamed')
        if is_dt:
            model.requires_grad_(False)
            model.set_attn_implementation('flash_attention_2')
            if args.family=='qwen3':
                import native_capture_events
                assert Path(native_capture_events.__file__).resolve()==R/'deltatrace/accelerated/native_capture_events.py'
                from graphed_qwen3 import make_graphed_qwen3
                init_start=time.perf_counter()
                dt_runner,report['acceleration_sources']=make_graphed_qwen3(R,model,env['finite_library'],env['finite_library_sha256'])
                if args.method=='deltatrace_streamed':
                    from whole_sac_qwen3 import SACFiniteGraphQwen3
                    dt_runner=SACFiniteGraphQwen3(model,dt_runner.finite)
                    report['rematerialized_sources']={}
                    for module_name,digest in protocol['candidate_modules'].items():
                        module=sys.modules[module_name];assert sha(Path(module.__file__).read_bytes())==digest
                        report['rematerialized_sources'][module_name]=digest
                report['initialization'].append({'name':'source_verification_controller_and_library','seconds':time.perf_counter()-init_start})
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
            assert lens['total_tokens']<=1024
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
            last_cost={}
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
                        answer=dt_runner.attribute(base_ids,actual,attention,p_len,mutation_audit=phase=='audit')
                        last_cost.clear();last_cost.update({k:answer.get(k) for k in ['native_projection_SAC','streamed_root','rematerialized_root','whole_root_graph','native_graph_execution','root_retention_mutation_audit','graph_input_copy_audit']})
                        return answer
                    pair=torch.cat((base_ids,actual))
                    selection=PackedAnswerTargets([{'target_ids':actual[0,p_len:].detach().cpu(),'prompt_length':p_len}],
                        [list(range(g_len))],actual.shape[1],model.device)
                    signed,detail=dt_runner.attribute(pair,torch.ones_like(pair),selection,select_output_rows=True,observer=None)
                    return {'signed_full_sequence':signed[0].cpu().tolist(),'details':detail}
                return call
            n_runs=5
            last_ok=False
            for rep in range(n_runs):
                phase='warm' if rep<2 else 'measured'
                report['status']=f'{input_len}_{phase}_{rep}';save()
                gc.collect();bench.maybe_reset_cuda([0]);started=time.perf_counter()
                runner=None
                resident_before=resident_after=process_memory=None
                last_cost.clear()
                try:
                    runner=make_runner();init_seconds=time.perf_counter()-started
                    import resource
                    resident_before={'allocated_bytes':torch.cuda.memory_allocated(),'reserved_bytes':torch.cuda.memory_reserved()}
                    status,wall,alloc,reserved,by_device=bench.measure(runner,[0],catch_oom=True)
                    resident_after={'allocated_bytes':torch.cuda.memory_allocated(),'reserved_bytes':torch.cuda.memory_reserved()}
                    process_memory={'rss_current_bytes':int(next(line.split()[1] for line in Path('/proc/self/status').read_text().splitlines() if line.startswith('VmRSS:')))*1024,'rss_process_highwater_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
                    error=None
                except Exception:
                    status='init_error';error=traceback.format_exc();wall=alloc=reserved=None;by_device={}
                    init_seconds=time.perf_counter()-started
                finally:runner=None
                row={'family':args.family,'attr_func':args.method,'phase':phase,'repeat':rep,
                  'target_input_tokens':input_len,'target_output_tokens':32,'target_total_tokens':input_len+32,
                  'actual_input_tokens':actual_input,'actual_output_tokens':actual_output,
                  'actual_total_tokens':lens['total_tokens'],'actual_formatted_prompt_tokens':prompt_len,
                  'actual_generation_tokens':gen_len,'time_sec':wall,'status':status,'error':error,
                  'peak_mem_gb':reserved if reserved is not None else alloc,'peak_allocated_gb':alloc,
                  'peak_mem_reserved_gb':reserved,'peak_mem_by_device_gb':by_device,'runner_init_seconds':init_seconds,
                  'input_sha256':expected_hash,'resident_before':resident_before,'resident_after':resident_after,'process_memory':process_memory,'runtime_cost':dict(last_cost)}
                report['rows'].append(row)
                with (out/'time_curve_runs.jsonl').open('a') as f:bench.append_jsonl_row(f,row)
                bench.write_summary_csv([x for x in report['rows'] if x['phase']=='measured'],out)
                save();print(json.dumps({k:row[k] for k in ['family','attr_func','target_input_tokens','phase','repeat','status','time_sec','peak_allocated_gb']}),flush=True)
                last_ok=status=='ok'
                identity()
                if status!='ok':
                    # Exact default methods that cannot initialize/execute are evidence.
                    # Keep the requested repeated failures; do not patch their implementation.
                    gc.collect();torch.cuda.empty_cache()
            if last_ok:
                phase='audit'
                report['status']=f'{input_len}_audit';save()
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
                case['audit']={'separately_charged_seconds':audit_seconds,'actual_calls':actual_calls,
                  'vector_shape':list(vec.shape),'vector_sha256':sha(vec.tobytes()),'finite':finite}
                if is_dt:
                    reference_path=Path(protocol['reference_vectors']['path'])
                    assert sha(reference_path.read_bytes())==protocol['reference_vectors']['sha256']
                    with np.load(reference_path) as reference:historical=reference[str(input_len)+'_retained'].copy()
                    from retained import make_retained_qwen3
                    from qwen_signed_secant_paired_public_fa import capture_checkpoint_pair
                    from vendor_fa_finite_runtime import VendorFAFiniteP1
                    retained,_=make_retained_qwen3(R);original_fa=VendorFAFiniteP1(env['finite_library'],env['finite_library_sha256'])
                    audit_base=ids.clone();audit_base[0,eligible]=tokenizer.eos_token_id
                    baseline_calls=[]
                    def observe_baseline(_m,args,kwargs):
                        tensor=kwargs.get('input_ids');assert tensor.shape==(2,ids.shape[1])
                        assert torch.equal(tensor[0],audit_base[0]) and torch.equal(tensor[1],ids[0])
                        baseline_calls.append({'shape':list(tensor.shape),'input_sha256':sha(tensor.cpu().numpy().tobytes())})
                    baseline_handle=model.register_forward_pre_hook(observe_baseline,with_kwargs=True)
                    t=time.perf_counter()
                    before,after=capture_checkpoint_pair(model,audit_base,ids,mask,prompt_len)
                    baseline=retained(model,before,after,pv_rule='content_P1',finite_attention=original_fa)
                    baseline_handle.remove();assert len(baseline_calls)==1
                    case['retained_actual_calls']=baseline_calls
                    assert not actual_calls, 'Hot complete graph has no Python model-root invocation'
                    assert result['whole_root_graph']['fresh_graph_input_sha256']==baseline_calls[0]['input_sha256']
                    case['graph_input_matches_original_root']=True
                    torch.cuda.synchronize();case['retained_comparison_seconds']=time.perf_counter()-t
                    retained_vec=np.asarray(baseline['signed_full_sequence'],dtype=np.float64)
                    vectors[str(input_len)+'_root']=vec;vectors[str(input_len)+'_retained']=retained_vec;vectors[str(input_len)+'_historical']=historical
                    np.savez(out/'vectors.npz',**vectors)
                    case['comparisons']={name:{'exact':bool(np.array_equal(vec,other)),'max_abs':float(np.max(np.abs(vec-other))),
                        'relative_l2':float(np.linalg.norm(vec-other)/max(np.linalg.norm(other),1e-300))} for name,other in [('retained_same_process',retained_vec),('historical_other_cache',historical)]}
                    keys=['target_delta_score32_sum64','target_delta_score16','signed_sum','unassigned_total','layer_checks']
                    case['math_diagnostics_exact']=all(result[k]==baseline[k] for k in keys)
                    case['details']={k:v for k,v in result.items() if k!='signed_full_sequence'}
                    case['graph_build_info']=dt_runner.program.build_info
                    assert result['native_layer_replay_calls']==36
                    assert result['deferred_validation']['predicates']==829+(36 if result['native_projection_SAC']['selected_projections'] else 0)
                    assert all(c['native_input_exact'] and c['native_output_exact'] for c in result['native_layer_boundary_checks']['paired_batch'])
                    case['retained_details']={k:v for k,v in baseline.items() if k!='signed_full_sequence'}
                    case['audit']['full_vector_exact_to_retained_same_process']=case['comparisons']['retained_same_process']['exact']
                    save();assert case['comparisons']['retained_same_process']['exact'] and case['math_diagnostics_exact']
                    del baseline,before,after,retained_vec,historical,audit_base
                vectors[str(input_len)]=vec
                np.savez(out/'vectors.npz',**vectors)
                del result,vec
                identity();save()
            del ids,mask,engine
            gc.collect();torch.cuda.empty_cache()
        if is_dt:
            t=time.perf_counter();dt_runner.close();gc.collect();torch.cuda.synchronize()
            report['close']={'seconds':time.perf_counter()-t,'allocated_bytes':torch.cuda.memory_allocated(),'reserved_bytes':torch.cuda.memory_reserved()}
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

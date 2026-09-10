"""Complete retained/new root APIs on every original <=1k development sample.

Fixed original responses, no truncation, no generation or metric calls. These
include audit clones and compilation and are not warm latency measurements.
"""
from pathlib import Path
import argparse,gc,hashlib,importlib.util,inspect,json,os,sys,time,traceback


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    A=p.parse_args().root;R=A/'memory_production_v2_release';B=Path('/tmp/codex_short_b1_efficiency_20260910_v1')
    plan_path=A/'memory_v2_author_plan.json';plan=json.loads(plan_path.read_bytes());out=A/'memory_v2_author';out.mkdir(exist_ok=False)
    sha=lambda b:hashlib.sha256(b).hexdigest()
    report={'status':'imports','driver_sha256':sha(Path(__file__).read_bytes()),'plan_sha256':sha(plan_path.read_bytes()),
        'cases':[],'generation_calls':0,'metric_calls':0,'FT_calls':0,'sample_batch':1,
        'scope':'Complete root and finite APIs, separately charged strict audit calls; no warm speed or quality metric claim.'}
    def save():(out/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    save()
    assert sha(Path(__file__).read_bytes())==plan['driver_sha256']
    assert sha((A/'check_graph_validation_probes.py').read_bytes())==plan['probe_sha256']
    for name,digest in plan['runtime_files'].items():assert sha((R/name).read_bytes())==digest,name
    try:
        env=json.loads((R/'environment.json').read_bytes())['qwen3']
        os.environ.update(MACA_PATH='/opt/maca',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
            TRITON_CACHE_DIR=str(A/'triton_memory_v2_confirmation_deltatrace_streamed'),TORCHINDUCTOR_CACHE_DIR=str(A/'inductor_memory_v2_confirmation_deltatrace_streamed'))
        for folder in [B/'deps',Path(env['official_root']),R/'deltatrace/clean/qwen3',R/'deltatrace/accelerated']:sys.path.insert(0,str(folder))
        import native_capture_events
        assert Path(native_capture_events.__file__).resolve()==R/'deltatrace/accelerated/native_capture_events.py'
        import numpy as np,torch
        torch.set_num_threads(4);torch.manual_seed(42);torch.backends.cuda.matmul.allow_tf32=False
        torch._dynamo.config.cache_size_limit=max(64,torch._dynamo.config.cache_size_limit)
        torch._dynamo.config.accumulated_cache_size_limit=max(256,torch._dynamo.config.accumulated_cache_size_limit)
        source=R/'official_exp1/run_time_curve.py';assert sha(source.read_bytes())==plan['exp1_sha256']
        spec=importlib.util.spec_from_file_location('author_exp1_compat',source);bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench)
        t=time.perf_counter();model,tokenizer=bench.load_model_balanced(env['checkpoint'],'cuda:0');torch.cuda.synchronize()
        report['model_load_seconds']=time.perf_counter()-t
        assert sha(Path(inspect.getfile(type(model))).read_bytes())==env['native_model_sha256']
        model.eval().requires_grad_(False);model.set_attn_implementation('flash_attention_2')
        native={n:(type(m).forward,m.forward) for n,m in model.named_modules()}
        from retained import make_retained_qwen3
        from memory_efficient_qwen3 import make_memory_efficient_qwen3
        from qwen_signed_secant_paired_public_fa import capture_checkpoint_pair
        from vendor_fa_finite_runtime import VendorFAFiniteP1
        old,report['retained_sources']=make_retained_qwen3(R)
        new,report['root_sources']=make_memory_efficient_qwen3(R,model,plan['candidate_finite_library']['path'],plan['candidate_finite_library']['sha256'])
        fa=VendorFAFiniteP1(env['finite_library'],env['finite_library_sha256'])
        from qwen3_root_retained import NativeRootTape
        hook_counts=lambda:[(len(m._forward_hooks),len(m._forward_pre_hooks)) for m in model.modules()]
        counts=hook_counts();monitor=[sys.monitoring.get_tool(i) for i in range(6)]
        tape=NativeRootTape(model)
        try:
            with tape:raise RuntimeError('deliberate capture cleanup probe')
        except RuntimeError as exc:assert str(exc)=='deliberate capture cleanup probe'
        finally:tape.clear()
        assert counts==hook_counts() and monitor==[sys.monitoring.get_tool(i) for i in range(6)] and sys.getprofile() is None
        report['capture_exception_cleanup_passed']=True
        from batched_validation_rope import BatchedValidation,verify_rejection_and_statistics
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
        check_start=time.perf_counter();report['strict_scalar_checks']=verify_rejection_and_statistics()
        report['strict_rope_checks']=[]
        for bad in [None,'q','k']:
            q=torch.randn(1,5,2,8,device='cuda',dtype=torch.float16);k=torch.randn_like(q)
            cos=torch.randn(1,5,8,device='cuda',dtype=torch.float16);sin=torch.randn_like(cos)
            actual_q,actual_k=apply_rotary_pos_emb(q.transpose(1,2),k.transpose(1,2),cos,sin)
            actual_q=actual_q.transpose(1,2).clone();actual_k=actual_k.transpose(1,2).clone()
            if bad=='q':actual_q[0,0,0,0]+=1
            if bad=='k':actual_k[0,0,0,0]+=1
            check=BatchedValidation();eq,ek=check.rope(q,k,actual_q,actual_k,cos,sin,'rope_q','rope_k')
            maximum=check.max_abs(q)
            try:
                result=check.finish({'q':eq,'k':ek,'maximum':maximum});assert bad is None and result['q'] and result['k']
            except ValueError as exc:assert bad is not None and 'rope_'+bad in str(exc)
            report['strict_rope_checks'].append({'case':bad or 'valid','passed':True})
        torch.cuda.synchronize();report['validation_probe_seconds']=time.perf_counter()-check_start
        from check_graph_validation_probes import run as graph_probes
        report['graph_validation_probes']=graph_probes();save()
        from native_capture_events import LocalCaptureEvents
        from contextlib import nullcontext
        assert sha((A/'check_memory_contract.py').read_bytes())==plan['contract_probe_sha256']
        vectors={}
        for index,row in enumerate(plan['cases']):
            cpu=torch.tensor(row['input_ids'],dtype=torch.long)[None]
            assert sha(cpu.numpy().tobytes())==row['input_sha256'] and cpu.shape[1]<=1024
            ids=cpu.to(model.device);mask=torch.ones_like(ids);base=ids.clone();eligible=[row['user_positions'][j] for j in row['keep']]
            assert int(ids[0,-1])==tokenizer.eos_token_id and ids.shape[1]-row['prompt_length']==row['target_length']
            base[0,eligible]=tokenizer.eos_token_id;rec={'key':row['key'],'input_sha256':row['input_sha256'],'total_tokens':ids.shape[1],'calls':[]}
            report['cases'].append(rec);maths={}
            for mode in (['retained','root_cold','root_hot'] if index%2==0 else ['root_cold','root_hot','retained']):
                report['status']=row['key']+'_'+mode;save();actual=[]
                def observe(_m,args,kwargs):
                    received=kwargs.get('input_ids');assert received is not None
                    assert received.shape==(2,ids.shape[1]) and torch.equal(received[0],base[0]) and torch.equal(received[1],ids[0])
                    actual.append({'shape':list(received.shape),'input_sha256':sha(received.cpu().numpy().tobytes())})
                handle=model.register_forward_pre_hook(observe,with_kwargs=True) if mode=='retained' else None
                original_root_code=inspect.unwrap(type(model).forward).__code__
                def observe_root(frame,event,value):
                    if event!='call':return
                    tensor=frame.f_locals['input_ids']
                    actual.append({'shape':list(tensor.shape),'stride':list(tensor.stride()),'data_ptr':tensor.data_ptr()})
                event_context=nullcontext() if mode=='retained' else LocalCaptureEvents([original_root_code],observe_root)
                gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
                try:
                    with event_context:
                        if mode=='retained':
                            before,after=capture_checkpoint_pair(model,base,ids,mask,row['prompt_length'])
                            result=old(model,before,after,pv_rule='content_P1',finite_attention=fa)
                            del before,after
                        else:result=new.attribute(base,ids,mask,row['prompt_length'],mutation_audit=True)
                    torch.cuda.synchronize();seconds=time.perf_counter()-t
                finally:
                    if handle is not None:handle.remove()
                assert len(actual)==(1 if mode=='retained' else 3 if mode=='root_cold' else 0),actual
                if mode!='retained':
                    assert all(c['shape']==[2,ids.shape[1]] and c['data_ptr']==new.program.ids.data_ptr() for c in actual)
                    expected_pair=torch.cat((base,ids)).cpu()
                    assert result['whole_root_graph']['fresh_graph_input_sha256']==sha(expected_pair.numpy().tobytes())
                    assert result['native_graph_execution']['native_model_GPU_root_executions']==(4 if mode=='root_cold' else 1)
                    del expected_pair
                assert counts==hook_counts() and monitor==[sys.monitoring.get_tool(i) for i in range(6)]
                for name,module in model.named_modules():assert native[name]==(type(module).forward,module.forward),name
                vector=np.asarray(result['signed_full_sequence'],dtype=np.float64);assert np.isfinite(vector).all()
                vectors[row['key']+'/'+mode]=vector
                maths[mode]={k:result[k] for k in ['target_delta_score32_sum64','target_delta_score16','signed_sum','unassigned_total','layer_checks']}
                rec['calls'].append({'mode':mode,'seconds_including_compilation_and_audit':seconds,
                    'reported_peak_allocated_gb':torch.cuda.max_memory_allocated()/1e9,'reported_peak_reserved_gb':torch.cuda.max_memory_reserved()/1e9,'actual_root':actual,
                    'vector_sha256':sha(vector.tobytes()),'details':{k:v for k,v in result.items() if k!='signed_full_sequence'}})
                np.savez(out/'vectors.npz',**vectors);save();del result,vector
            rec['exact']=all(np.array_equal(vectors[row['key']+'/retained'],vectors[row['key']+'/'+mode]) for mode in ['root_cold','root_hot'])
            rec['all_math_diagnostics_exact']=maths['retained']==maths['root_cold']==maths['root_hot']
            assert rec['exact'] and rec['all_math_diagnostics_exact'],row['key'];save()
            print(json.dumps({'key':row['key'],'exact':True,'all_math_diagnostics_exact':True}),flush=True)
            if index==0:
                changed=ids.clone();position=eligible[0]
                changed[0,position]=(changed[0,position]+1)%model.config.vocab_size
                previous_program=new.program
                changed_base=changed.clone();changed_base[0,eligible]=tokenizer.eos_token_id
                gc.collect();torch.cuda.synchronize();t=time.perf_counter()
                changed_new=new.attribute(changed_base,changed,mask,row['prompt_length'],mutation_audit=True)
                torch.cuda.synchronize();new_seconds=time.perf_counter()-t
                assert new.program is previous_program and changed_new['native_graph_execution']['build_this_call'] is None
                t=time.perf_counter();cb,ca=capture_checkpoint_pair(model,changed_base,changed,mask,row['prompt_length'])
                changed_old=old(model,cb,ca,pv_rule='content_P1',finite_attention=fa)
                torch.cuda.synchronize();old_seconds=time.perf_counter()-t
                cv=np.asarray(changed_new['signed_full_sequence'],dtype=np.float64);ov=np.asarray(changed_old['signed_full_sequence'],dtype=np.float64)
                vectors[row['key']+'/changed_new']=cv;vectors[row['key']+'/changed_retained']=ov
                np.savez(out/'vectors.npz',**vectors)
                keys=['target_delta_score32_sum64','target_delta_score16','signed_sum','unassigned_total','layer_checks']
                rec['changed_input_control']={'position':position,'input_ids':changed[0].cpu().tolist(),'input_sha256':sha(changed.cpu().numpy().tobytes()),
                    'same_graph':True,'new_seconds':new_seconds,'retained_seconds':old_seconds,
                    'exact':bool(np.array_equal(cv,ov)),'output_changed':not np.array_equal(cv,vectors[row['key']+'/root_hot']),
                    'math_exact':all(changed_new[k]==changed_old[k] for k in keys)}
                assert rec['changed_input_control']['exact'] and rec['changed_input_control']['output_changed'] and rec['changed_input_control']['math_exact']
                del changed,changed_base,changed_new,changed_old,cb,ca,cv,ov,previous_program
            if index==len(plan['cases'])-1:
                from check_memory_contract import run as contract_probes
                report['memory_contract_controls']=contract_probes(model,new,base,ids,mask,row['prompt_length'],old,fa)
                # In-place parameter version changes require a fresh public controller.
                new.close()
                new,report['recovery_sources']=make_memory_efficient_qwen3(R,model,plan['candidate_finite_library']['path'],plan['candidate_finite_library']['sha256'])
                recovery=new.attribute(base,ids,mask,row['prompt_length'],mutation_audit=True)
                recovered=np.asarray(recovery['signed_full_sequence'],dtype=np.float64)
                vectors[row['key']+'/contract_recovery']=recovered
                report['contract_recovery_exact']=bool(np.array_equal(recovered,vectors[row['key']+'/retained']))
                report['contract_recovery_math_exact']={k:recovery[k] for k in maths['retained']}==maths['retained']
                assert report['contract_recovery_exact'] and report['contract_recovery_math_exact']
                del recovery,recovered
                np.savez(out/'vectors.npz',**vectors);save()
            del ids,mask,base,cpu
        new.close();assert new.program is None and new.signature is None and not new.busy
        assert counts==hook_counts() and monitor==[sys.monitoring.get_tool(i) for i in range(6)]
        report['explicit_controller_close_passed']=True
        report['status']='complete';report['all_vectors_exact']=all(c['exact'] for c in report['cases'])
    except Exception:
        report['status']='failed';report['error']=traceback.format_exc()
    save();print(json.dumps({'status':report['status'],'error':report.get('error')}),flush=True)


if __name__=='__main__':main()

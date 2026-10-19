"""One bounded A-B-A native-owner resource comparison on a natural long B4.

The original training return is retained. The optional candidate and warm
baseline calls are validation work and must be excluded from production Util.
No optimizer, environment, or independent finite operator is introduced.
"""
import ast
import hashlib
from pathlib import Path

owner = Path(__file__).with_name('observe_existing_DT_metadata.py')
assert hashlib.sha256(owner.read_bytes()).hexdigest() == '20c0de1e0d2b7f89082e372fbdac292cab0bea41866612f8f0555d3afe94649c'
tree = ast.parse(owner.read_text())
observation = next(n.value.value for n in tree.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'observation' for t in n.targets))
observation = observation.replace('_existing_DT_timing_scope_installed', '_FA_capture_resource_pair_scope_installed_v1')
observation = observation.replace('native_DT_existing_timing', 'native_DT_FA_capture_resource_pair')
observation = observation.replace('existing_DT_timing_observer_', 'FA_capture_resource_pair_observer_')
anchor = '        native_attribute=runner.attribute'
assert observation.count(anchor) == 1
observation = observation.replace(anchor, anchor + '''
        import ast,hashlib,__future__,torch
        from pathlib import Path
        candidate_path=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/candidates/dt-FA-capture-GPU-original-gate-host-20261019-v1/qwen35_dense_finite_runner.py')
        assert hashlib.sha256(candidate_path.read_bytes()).hexdigest()=='e9ee7c89d544c52a09d6849b9b74f2a499c552899061c74fab7ede90c657291c'
        original_native=runner._bounded_native_mixer_storage['original_attribute']
        native_file=Path(original_native.__func__.__code__.co_filename)
        assert hashlib.sha256(native_file.read_bytes()).hexdigest()=='6a112224b3d4e63d047a576d7df68ff1ee65e4bbacfcb0014f84c17bd82b8520'
        candidate_tree=ast.parse(candidate_path.read_text())
        candidate_class,=[n for n in candidate_tree.body if isinstance(n,ast.ClassDef) and n.name=='Qwen35DenseFiniteRunner']
        candidate_function,=[n for n in candidate_class.body if isinstance(n,ast.FunctionDef) and n.name=='attribute']
        candidate_globals=dict(original_native.__func__.__globals__)
        compiler_flags=original_native.__func__.__code__.co_flags & __future__.annotations.compiler_flag
        exec(compile(ast.Module(body=[candidate_function],type_ignores=[]),str(candidate_path),'exec',flags=compiler_flags),candidate_globals)
        candidate_attribute=candidate_globals['attribute'].__get__(runner,type(runner))''')
anchor = '            count=[0]'
assert observation.count(anchor) == 1
observation = observation.replace(anchor, anchor + '''
            validation_calls=[0]
            guard_calls=[0]
            scheduler_epoch=actor.actor_lr_scheduler.last_epoch''')
anchor = '                ended=time.time()'
assert observation.count(anchor) == 1
observation = observation.replace(anchor, anchor + '''
                pair_info=None
                paired=native_args[0] if native_args else native_kwargs['paired_ids']
                memory_guard=dict(width=int(paired.shape[1]),attempted=False)
                if validation_calls[0]==0:
                    flags=(runner.pin_root_host,runner.pin_replay_host,runner.offload_replay_mixer)
                    eligible=False;memory_guard['attempted']=True
                    if (2048<paired.shape[1]<=14000 and paired.shape[0]<=8
                            and flags==(False,True,True) and not runner.copy_replay_captures
                            and not getattr(runner,'fa_replay_capture_cuda',False)):
                        try:
                            config=actor.actor_model_config
                            config=getattr(config,'text_config',config)
                            hidden=int(config.hidden_size)
                            heads=int(config.num_attention_heads)
                            head_dim=int(getattr(config,'head_dim',hidden//heads))
                            free,total=torch.cuda.mem_get_info()
                            allocated=torch.cuda.memory_allocated()
                            workspace=max(0,returned[1]['peak_allocated']-allocated)
                            extra_bound=12*paired.shape[0]*paired.shape[1]*max(hidden,heads*head_dim)*runner.model.lm_head.weight.element_size()
                            required=workspace+extra_bound+2*2**30
                            eligible=free>required
                            memory_guard.update(free_bytes=free,total_bytes=total,baseline_peak=returned[1]['peak_allocated'],
                                current_allocated=allocated,workspace_bound=workspace,extra_FA_capture_bound=extra_bound,
                                required_with_margin=required,hidden_size=hidden,attention_width=heads*head_dim)
                        except Exception as error:
                            eligible=False
                            memory_guard['error']=type(error).__name__+': '+str(error)
                    token=torch.tensor([int(eligible),int(paired.shape[1]<6000)],device=paired.device,dtype=torch.int32)
                    torch.distributed.all_reduce(token,op=torch.distributed.ReduceOp.MIN)
                    guard_calls[0]+=1
                    eligible=bool(token[0].item()) and not bool(token[1].item())
                    if eligible:
                        cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state()
                        existed=hasattr(runner,'fa_replay_capture_cuda')
                        old_fa=getattr(runner,'fa_replay_capture_cuda',False)
                        pair_info=dict(status='candidate_started',training_iteration=scheduler_epoch+1,
                            scheduler_epoch=scheduler_epoch,candidate_deployed=False,additional_DT_calls=0,
                            additional_optimizer_calls=0,additional_rollout_calls=0,
                            baseline_flags=list(flags),candidate_flags=list(flags),memory_guard=memory_guard,
                            candidate_FA_capture_cuda=True,MLP_gate_and_GDN_original_storage=True,
                            baseline_result_returned_unchanged=True,
                            candidate_source=str(candidate_path),candidate_source_SHA256=hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
                            numerical_scope='Full sampled DT vector comparison is resource-transport evidence, not a new official whole-model tolerance')
                        try:
                            runner.fa_replay_capture_cuda=True
                            tick=time.time();pair_info['candidate_started_unix']=tick
                            validation_calls[0]+=1;pair_info['additional_DT_calls']+=1
                            candidate=candidate_attribute(*native_args,**native_kwargs)
                            pair_info['candidate_finished_unix']=time.time()
                            pair_info['candidate_wall_seconds']=time.time()-tick
                            pair_info['candidate_owner_info']=candidate[1]
                            original_vector=returned[0].detach().cpu()
                            candidate_vector=candidate[0].detach().cpu()
                            pair_info['raw_DT_vector_equal']=bool(torch.equal(original_vector,candidate_vector))
                            pair_info['raw_DT_max_abs_difference']=float((original_vector-candidate_vector).abs().max())
                            del candidate,candidate_vector
                            runner.fa_replay_capture_cuda=old_fa
                            tick=time.time();pair_info['warm_baseline_started_unix']=tick
                            validation_calls[0]+=1;pair_info['additional_DT_calls']+=1
                            warm=original_native(*native_args,**native_kwargs)
                            pair_info['warm_baseline_finished_unix']=time.time()
                            pair_info['warm_baseline_wall_seconds']=time.time()-tick
                            pair_info['warm_baseline_owner_info']=warm[1]
                            warm_vector=warm[0].detach().cpu()
                            pair_info['warm_baseline_vector_equal']=bool(torch.equal(original_vector,warm_vector))
                            pair_info['warm_baseline_max_abs_difference']=float((original_vector-warm_vector).abs().max())
                            pair_info['rng_unchanged']=bool(torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(cuda_rng,torch.cuda.get_rng_state()))
                            pair_info['status']='same_input_owner_pair_completed_not_deployed'
                            del warm,warm_vector,original_vector
                        except Exception as error:
                            pair_info.update(status='candidate_rejected_after_optional_comparison_error',error=type(error).__name__+': '+str(error))
                        finally:
                            runner.pin_root_host,runner.pin_replay_host,runner.offload_replay_mixer=flags
                            if existed:runner.fa_replay_capture_cuda=old_fa
                            else:
                                if hasattr(runner,'fa_replay_capture_cuda'):del runner.fa_replay_capture_cuda
                            torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state(cuda_rng)
                            pair_info['original_options_restored']=((runner.pin_root_host,runner.pin_replay_host,runner.offload_replay_mixer)==flags and getattr(runner,'fa_replay_capture_cuda',False)==old_fa)''')
anchor = '                        started=began,finished=ended,paired_shape=list(paired.shape),'
assert observation.count(anchor) == 1
observation = observation.replace(anchor, anchor + '''
                        optional_same_input_FA_capture_pair=pair_info,
                        FA_capture_memory_guard=memory_guard,
                        training_iteration=scheduler_epoch+1,scheduler_epoch=scheduler_epoch,''')
observation = observation.replace('additional_DT_calls=0,new_CUDA_events=0,original_return_objects_unchanged=True,',
    "additional_DT_calls=(pair_info['additional_DT_calls'] if pair_info else 0),new_CUDA_events=0,original_return_objects_unchanged=True,")
observation = observation.replace('restored_original_methods=True,additional_DT_calls=0)',
    'restored_original_methods=True,additional_DT_calls=validation_calls[0],validation_guard_collectives=guard_calls[0],training_iteration=scheduler_epoch+1)')
observation = observation.replace("result['FA_capture_resource_pair_observer_additional_calls']=0",
    "result['FA_capture_resource_pair_observer_max_validation_calls_per_rank']=2")

wrapper = Path(__file__).with_name('inspect_native_workers.py')
assert hashlib.sha256(wrapper.read_bytes()).hexdigest() == '986610703d7d1ad9616329c4c6e52534a703272a2bb2899078a807adca1d46c0'
code = wrapper.read_text().replace('v9-negative2-eight-GPU-initial-v1', 'FA-capture-original-gate-host-pair-v1')
anchor = "exec(compile(code,str(owner),'exec'),globals())"
assert code.count(anchor) == 1
code = code.replace(anchor, "assert code.count('    return result')==1\ncode=code.replace('    return result',"
                    + repr(observation) + ")\n" + anchor)
exec(compile(code, str(wrapper), 'exec'), globals())

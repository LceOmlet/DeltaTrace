"""Cost and output diagnostic for the original DT host-storage options.

Compose the existing one-readout observation adapter. On its third native
minibatch, make one validation-only same-input call with the owner's existing
pin_root_host/pin_replay_host options disabled. Training receives the original
result. Restore options, RNG and methods. This is not an official numerical
acceptance certificate and does not enable the candidate in production.
"""
import hashlib
from pathlib import Path

owner = Path(__file__).with_name('observe_existing_DT_metadata.py')
assert hashlib.sha256(owner.read_bytes()).hexdigest() == '20c0de1e0d2b7f89082e372fbdac292cab0bea41866612f8f0555d3afe94649c'
code = owner.read_text()
code = code.replace('AUDIT=ROOT/', 'DT_STAGE_OBSERVATION_AUDIT=ROOT/')
code = code.replace("(AUDIT/'existing-DT", "(DT_STAGE_OBSERVATION_AUDIT/'existing-DT")
code = code.replace('_existing_DT_timing_scope_installed', '_existing_DT_pageable_pair_scope_installed')
code = code.replace('existing-DT-metadata-observer-completed.json', 'existing-DT-pageable-pair-observer-completed.json')
code = code.replace('native_DT_existing_timing', 'native_DT_existing_pageable_pair')
code = code.replace('code=wrapper.read_text()', "code=wrapper.read_text().replace('v9-negative2-eight-GPU-initial-v1','existing-DT-pageable-pair-v1')")

anchor = '                ended=time.time()'
assert code.count(anchor) == 1
code = code.replace(anchor, anchor + '''
                pair_info=None
                if count[0]==2:
                    import torch
                    flags=(runner.pin_root_host,runner.pin_replay_host)
                    cpu_rng=torch.get_rng_state()
                    cuda_rng=torch.cuda.get_rng_state()
                    pair_info=dict(status='candidate_started',candidate_deployed=False,
                        additional_DT_calls=1,additional_optimizer_calls=0,
                        additional_rollout_calls=0,baseline_flags=list(flags),
                        candidate_flags=[False,False],baseline_result_returned_unchanged=True,
                        numerical_scope='Output differences are diagnostic; not an official FA/FLA/VERL tolerance certificate')
                    candidate_started=time.time()
                    try:
                        assert flags==(True,True)
                        runner.pin_root_host=False
                        runner.pin_replay_host=False
                        candidate=native_attribute(*native_args,**native_kwargs)
                        pair_info['candidate_wall_seconds']=time.time()-candidate_started
                        pair_info['candidate_owner_info']=candidate[1]
                        original_vector=returned[0].detach().cpu()
                        candidate_vector=candidate[0].detach().cpu()
                        pair_info['raw_DT_vector_equal']=bool(torch.equal(original_vector,candidate_vector))
                        pair_info['raw_DT_max_abs_difference']=float((original_vector-candidate_vector).abs().max())
                        pair_info['rng_unchanged']=bool(torch.equal(cpu_rng,torch.get_rng_state()) and
                            torch.equal(cuda_rng,torch.cuda.get_rng_state()))
                        pair_info['status']='same_input_owner_pair_completed_not_deployed'
                        del candidate,candidate_vector,original_vector
                    except Exception as error:
                        pair_info.update(status='candidate_rejected_after_optional_comparison_error',
                            error=type(error).__name__+': '+str(error))
                    finally:
                        runner.pin_root_host,runner.pin_replay_host=flags
                        torch.set_rng_state(cpu_rng)
                        torch.cuda.set_rng_state(cuda_rng)
                        pair_info['original_options_restored']=(runner.pin_root_host,runner.pin_replay_host)==flags
''')
anchor = '                        started=began,finished=ended,paired_shape=list(paired.shape),'
assert code.count(anchor) == 1
code = code.replace(anchor, anchor + '\n                        optional_same_input_pageable_pair=pair_info,')
# Correct the inherited metadata language: this comparison adds exactly one
# validation call per rank when the common third minibatch exists.
code = code.replace('additional_DT_calls=0,new_CUDA_events=0,original_return_objects_unchanged=True,',
                    'additional_DT_calls=int(pair_info is not None),new_CUDA_events=0,original_return_objects_unchanged=True,')
code = code.replace('restored_original_methods=True,additional_DT_calls=0)',
                    'restored_original_methods=True,additional_DT_calls=int(count[0]>=3))')
code = code.replace("result['existing_DT_timing_observer_additional_calls']=0",
                    "result['existing_DT_timing_observer_additional_calls_per_rank_when_executed']=1")
code = code.replace('additional_model_calls=0,additional_DT_calls=0,optimizer_updates=0,new_CUDA_events=0,',
                    'additional_model_calls_via_one_DT_validation_per_rank=True,additional_DT_calls_per_rank_when_executed=1,optimizer_updates=0,new_CUDA_events=0,')
code = code.replace("scope='Publish already-computed native metadata for one future original readout per rank; automatic restoration'",
                    "scope='One same-input original-owner host-storage comparison per rank during the third natural B4; baseline result retained; candidate not deployed; automatic restoration'")
exec(compile(code, str(owner), 'exec'), globals())

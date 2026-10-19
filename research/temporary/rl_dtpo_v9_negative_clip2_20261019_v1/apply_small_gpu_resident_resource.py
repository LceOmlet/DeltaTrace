"""Select the original GPU replay-storage option for bounded small B4s.

The original runner still performs every forward and finite calculation.
This adapter only selects its existing offload flag and restores it after
each call. It adds no CUDA call, collective, model call, or output change.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT = ROOT / 'receipts/dtpo-v9-negative-clip2-postgroup-variance-20261019-v4'
evidence = json.loads((AUDIT / 'existing-DT-gpu-resident-pair-results.json').read_bytes())
assert evidence['status'] == 'completed'
assert evidence['restored_rank_scopes'] == 8 and not evidence['errors']
assert evidence['successful_same_input_pair_count'] == 8
assert {r['rank'] for r in evidence['same_input_pairs']} == set(range(8))
for row in evidence['same_input_pairs']:
    pair = row['pair']
    assert pair['baseline_flags'] == [False, True, True]
    assert pair['candidate_flags'] == [False, True, False]
    assert pair['raw_DT_vector_equal'] and pair['rng_unchanged']
    assert pair['original_options_restored'] and pair['baseline_result_returned_unchanged']
    assert row['paired_shape'][0] == 8 and 2048 < row['paired_shape'][1] <= 3072
    baseline = next(r['value'] for r in evidence['native_timing_records']
                    if r['value']['rank'] == row['rank']
                    and r['value']['minibatch'] == row['minibatch'])
    candidate = pair['candidate_owner_info']
    assert candidate['peak_allocated'] < 55 * 2**30
    before = Counter(r['kind'] for r in baseline['calls'])
    after = Counter(r['kind'] for r in candidate['calls'])
    assert not (after - before)
    assert all(k.startswith('restore_mixer_captures_') for k in before - after)

owner = Path(__file__).with_name('inspect_native_workers.py')
assert hashlib.sha256(owner.read_bytes()).hexdigest() == '986610703d7d1ad9616329c4c6e52534a703272a2bb2899078a807adca1d46c0'
code = owner.read_text().replace('v9-negative2-eight-GPU-initial-v1',
                                'small-GPU-resident-resource-accepted-v1')
anchor = "exec(compile(code,str(owner),'exec'),globals())"
assert code.count(anchor) == 1
override = '''    import time,inspect
    producer=getattr(actor,'_deltatrace_producer',None)
    assert producer is not None
    runner=producer.runner
    source=Path(inspect.getfile(type(runner)))
    assert hashlib.sha256(source.read_bytes()).hexdigest()=='6a112224b3d4e63d047a576d7df68ff1ee65e4bbacfcb0014f84c17bd82b8520'
    assert (runner.pin_root_host,runner.pin_replay_host,runner.offload_replay_mixer)==(False,True,True)
    assert producer.direct_readout.trajectories.__name__=='trajectories'
    state=getattr(runner,'_bounded_native_mixer_storage',None)
    installed=state is None
    if installed:
        assert runner.attribute.__name__=='attribute'
        native_attribute=runner.attribute
        state=dict(installed_unix=time.time(),small_calls=0,offloaded_calls=0,
            largest_small_extent=0,small_peak_allocated=0,resource_extent_limit=2048,
            original_attribute=native_attribute)
        def choose_original_mixer_storage(*native_args,**native_kwargs):
            paired=native_args[0] if native_args else native_kwargs['paired_ids']
            use_resident=paired.ndim==2 and paired.shape[0]<=8 and paired.shape[1]<=2048
            original=runner.offload_replay_mixer
            if use_resident:
                runner.offload_replay_mixer=False
            try:
                returned=native_attribute(*native_args,**native_kwargs)
                if use_resident:
                    state['small_calls']+=1
                    state['largest_small_extent']=max(state['largest_small_extent'],paired.shape[1])
                    state['small_peak_allocated']=max(state['small_peak_allocated'],returned[1]['peak_allocated'])
                else:
                    state['offloaded_calls']+=1
                return returned
            finally:
                runner.offload_replay_mixer=original
        runner.attribute=choose_original_mixer_storage
        runner._bounded_native_mixer_storage=state
    result['small_GPU_resident_resource_override']=dict(
        status='applied_original_storage_option_bounded_small_inputs',installed_now=installed,
        installed_unix=state['installed_unix'],native_runner=str(source),
        native_runner_SHA256=hashlib.sha256(source.read_bytes()).hexdigest(),
        resource_extent_limit=2048,maximum_paired_rows=8,
        default_flags=dict(pin_root_host=False,pin_replay_host=True,offload_replay_mixer=True),
        eligible_call_offload_replay_mixer=False,restore_after_each_call=True,
        actor_scheduler_epoch=actor.actor_lr_scheduler.last_epoch,
        observed={k:v for k,v in state.items() if k!='original_attribute'},
        arithmetic_changed=False,model_calls_for_deployment=0,DT_calls_for_deployment=0,
        CUDA_calls_for_dispatch=0,collectives_for_dispatch=0,optimizer_updates=0,
        persistence='Current loaded runner only; long-input production route unchanged',
        capacity_scope='Eight simultaneous natural B4 pairs of extent2217..2401 fit below55GiB; production resident route restricted to extent<=2048. Original32768 offloaded route unchanged.',
        numerical_scope='Existing arithmetic owners unchanged. Sampled raw DT vectors equal; not a new official whole-model tolerance certificate.',
        verification_receipt='existing-DT-gpu-resident-pair-results.json')
    return result'''
code = code.replace(anchor,
    "assert code.count('    return result')==1\ncode=code.replace('    return result',"
    + repr(override) + ")\n"
    + "code=code.replace('completed_read_only_worker_binding_check','completed_runtime_small_GPU_resident_resource_override')\n"
    + anchor)
exec(compile(code, str(owner), 'exec'), globals())

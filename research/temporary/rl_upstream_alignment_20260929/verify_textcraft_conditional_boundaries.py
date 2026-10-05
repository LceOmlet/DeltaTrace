"""Original matched-layout recipe with passive conditional-boundary taps.

The original worker owns initialization, offload, original seven B4 groups,
three reference variants, numeric-MIN cuts and cleanup. Only its diagnostic
root callback is substituted: full EOS runs the original complete trace;
the two singles stop at the original score after reading existing root saves.
No model, finite rule, PPO, reward or reference implementation is supplied.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time

import ray
import torch
from omegaconf import OmegaConf
from verl.single_controller.base.decorator import Dispatch, register
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.workers.fsdp_workers import ActorRolloutRefWorker


BASE = Path(os.environ['DT_TEXTCRAFT_READOUT_BASE'])
OUT = Path(os.environ['DT_TEXTCRAFT_READOUT_ROOT'])
INPUT = Path(os.environ['DT_TEXTCRAFT_READOUT_CASES'])
PRIOR = BASE.parent/'textcraft-matched-layout-20261006-v1'
sys.path.insert(0,str(PRIOR))
os.environ['PYTHONPATH'] = str(PRIOR)+os.pathsep+os.environ.get('PYTHONPATH','')
import verify_textcraft_matched_layout as existing
from observe_textcraft_conditional_boundaries import collect_actual_coefficients, contract_saved_native_roots


ORIGINAL_RECIPE = existing.MatchedLayoutWorker.__ray_metadata__.modified_class.inspect_matched_layout
ORIGINAL_ROOT_OBSERVER = existing.observe_original_root
PACK_SHA = existing.PACK_SHA
REGISTER_SHA = 'a66a4712dc18926b1751233ff7688977b81c01cf77cdc9ede1155cca616274e1'
RAY_TRACING_SHA = '0c65ada558eff709049dd29894bd5079f5d07da8e219cbc461b61f07f6c4b27b'


def registered_body(function):
    """Follow official Ray/VERL executed closures, not wraps metadata copies."""
    body=function
    seen=set()
    while body.__closure__:
        assert id(body) not in seen, 'Cyclic registered callback closure.'
        seen.add(id(body))
        closure=dict(zip(body.__code__.co_freevars,
            (cell.cell_contents for cell in body.__closure__)))
        if body.__code__.co_name=='_resume_span' and 'method' in closure:
            assert hashlib.sha256(Path(body.__code__.co_filename).read_bytes()).hexdigest()==RAY_TRACING_SHA
            body=closure['method']
        elif 'func' in closure:
            assert body.__code__.co_name=='inner' and body.__code__.co_filename==register.__code__.co_filename
            assert hashlib.sha256(Path(body.__code__.co_filename).read_bytes()).hexdigest()==REGISTER_SHA
            body=closure['func']
        else:
            break
    return body


def recipe_callback_binding(recipe):
    """Identify the globals actually read by the original registered body."""
    body=registered_body(recipe)
    assert 'observe_original_root' in body.__code__.co_names
    globals_=body.__globals__
    assert existing.identity(globals_['observe_original_root'])==existing.identity(ORIGINAL_ROOT_OBSERVER)
    return globals_,body


def observe_complete_original_trace(runner,reference,selected,cases,labels,slots):
    """Observe the one original score and retain actual finite coefficients."""
    from deltatrace_credit import trace_token_attribution
    from qwen35_answer_finite import PackedAnswerTargets
    globals_ = runner.attribute.__func__.__globals__
    original_score = globals_['selected_target_log_probs']
    observation = dict(native_forward_calls=[],sync_calls=[],fixed_cut=None,
        original_sequence_length=selected.shape[1],finite_seed_called=True,
        stopped_before_finite_seed=False,finite_trace_calls=1,finite_seed_calls=1,
        timing_scope='Complete original full-EOS attribution, including passive CPU coefficient transport; not a performance comparison.')

    def native_call(_module,args,kwargs):
        ids = kwargs.get('input_ids',args[0] if args else None)
        observation['native_forward_calls'].append(dict(
            input_ids=existing.tensor_metadata(ids,hash_values=True),
            role='native_shared_prefix' if ids.shape[0]==selected.shape[0] else 'native_paired_root',
            use_cache=kwargs.get('use_cache'),
            logits_to_keep=existing.tensor_metadata(kwargs.get('logits_to_keep')),
            attention_mask=existing.tensor_metadata(kwargs.get('attention_mask')),
            past_key_values=existing.cache_metadata(kwargs.get('past_key_values'))))

    def score_once(original_logits,selection,*,outcome_logits=None):
        value=original_score(original_logits,selection,outcome_logits=outcome_logits)
        observation.update(observed_prefix_length=selected.shape[1]-selection.length,
            original_packed_logits=existing.tensor_metadata(original_logits),
            categorical_logits=existing.tensor_metadata(outcome_logits),
            categorical_logits_values=outcome_logits.detach().cpu().tolist(),
            categorical_log_probability_dtype=str(value.dtype),target_log_probs=value.detach().cpu().tolist(),
            selection=dict(batch=selection.batch,length=selection.length,
                positions=selection.positions.detach().cpu().tolist(),
                paired_positions=selection.paired_positions.detach().cpu().tolist(),
                samples=selection.samples.detach().cpu().tolist(),labels=selection.labels.detach().cpu().tolist(),
                outcome_token_ids=selection.outcome_token_ids.detach().cpu().tolist()))
        return value

    handle=runner.model._conditional.register_forward_pre_hook(native_call,with_kwargs=True)
    globals_['selected_target_log_probs']=score_once
    try:
        with collect_actual_coefficients(runner,slots) as bank:
            signed,roots,detail=trace_token_attribution(runner,reference,selected,cases,
                [[0] for _ in cases],packed_answer_targets=PackedAnswerTargets,outcome_token_ids=labels)
        observation.update(actual_coefficient_metadata=bank['metadata'],
            original_finite_detail=detail,actual_signed_probe_values=[],
            original_signed_shape=list(signed.shape),original_signed_dtype=str(signed.dtype),
            score_function=existing.identity(original_score))
        if bank['metadata']['diagnostics'] or not bank['metadata']['all_33_actual_boundaries_present']:
            raise RuntimeError('Missing actual coefficient observation: '+repr(bank['metadata']['diagnostics']))
        return observation,bank,signed
    finally:
        globals_['selected_target_log_probs']=original_score
        handle.remove()


@ray.remote
class ConditionalBoundaryWorker(ActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def inspect_matched_layout(self):
        """Delegate the existing recipe; alter diagnostic callback/metadata only."""
        recipe_globals,recipe_body=recipe_callback_binding(ORIGINAL_RECIPE)
        original_callback=recipe_globals['observe_original_root']
        state=dict(bank=None,full_calls=0,single_calls=0)

        def observe(runner,reference,selected,cases,labels,*,fixed_cut=None):
            # The pinned existing recipe supplies the same original group.
            # Borrow its diagnostic metadata only; no frame is retained.
            caller=sys._getframe(1)
            try:
                assert caller.f_code is recipe_body.__code__ and caller.f_globals is recipe_globals
                group=caller.f_locals['group']
                diagnostic=caller.f_locals['result']
            finally:
                del caller
            slots=[slot for slot,sample in enumerate(group['samples']) if sample['source_step']==0]
            diagnostic.update(scope=__doc__,
                timing_scope='Full-EOS calls complete the original finite trace; singles stop at score. Timers include diagnostic transport/contraction.',
                reused_original_recipe=existing.identity(ORIGINAL_RECIPE),
                passive_collector=existing.identity(collect_actual_coefficients))
            if fixed_cut is None:
                if state['bank'] is not None:
                    raise RuntimeError('Previous original group coefficient bank was not released.')
                observation,bank,signed=observe_complete_original_trace(
                    runner,reference,selected,cases,labels,slots)
                state['bank']=bank
                state['full_calls']+=1
                diagnostic['finite_trace_calls']=state['full_calls']
                diagnostic['finite_seed_calls']=state['full_calls']
                for slot in slots:
                    sample=group['samples'][slot]
                    observation['actual_signed_probe_values'].append(dict(original_slot=slot,
                        traj_uid=sample['traj_uid'],source_step=sample['source_step'],
                        probes=[dict(source_position=sample['probe_source_positions'][j],
                            input_position=position,token_id=int(selected[slot,position]),
                            actual_joint_d=float(signed[slot,position]),saved_original_d=sample['saved_probe_credit'][j]['d'])
                            for j,position in enumerate(sample['probe_input_positions'])]))
                del signed
            else:
                bank=state['bank']
                if bank is None:
                    raise RuntimeError('Single root lacks this original group actual coefficient bank.')
                with contract_saved_native_roots(runner,bank,slots) as contractions:
                    observation=ORIGINAL_ROOT_OBSERVER(runner,reference,selected,cases,labels,fixed_cut=fixed_cut)
                observation.update(conditional_boundary_contractions=contractions,
                    finite_trace_calls=0,finite_seed_calls=0)
                state['single_calls']+=1
                if contractions['diagnostics'] or contractions['recorded_boundary_slot_scalars']!=33*len(slots):
                    raise RuntimeError('Missing original single boundary observation: '+repr(contractions['diagnostics']))
                if state['single_calls']%2==0:
                    bank['by_boundary'].clear()
                    state['bank']=None
            diagnostic['actual_conditional_boundary_single_roots']=state['single_calls']
            return observation

        recipe_globals['observe_original_root']=observe
        assert recipe_globals['observe_original_root'] is observe
        try:
            receipt=ORIGINAL_RECIPE(self)
            receipt.update(finite_trace_calls=state['full_calls'],finite_seed_calls=state['full_calls'],
                conditional_single_root_calls=state['single_calls'])
            assert state['full_calls']==7 and state['single_calls']==14 and state['bank'] is None
            return receipt
        finally:
            recipe_globals['observe_original_root']=original_callback
            if state['bank'] is not None:
                state['bank']['by_boundary'].clear()
                state['bank']=None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inspect-only',action='store_true')
    args=parser.parse_args()
    checkpoint=Path(os.environ['DT_TEXTCRAFT_CHECKPOINT'])
    assert checkpoint.name=='global_step_25' and hashlib.sha256(INPUT.read_bytes()).hexdigest()==PACK_SHA
    assert existing.identity(existing.observe_original_root)['sha256']=='5e94816aceb772ee96ebaf46816e3405b1beb48541651d4a724654cfde54992b'
    cfg=OmegaConf.load(Path(os.environ['VERL_ROOT'])/'verl/trainer/config/ppo_trainer.yaml')
    for key,value in json.loads((BASE/'launch.json').read_bytes())['options'].items():
        OmegaConf.update(cfg,key.lstrip('+'),value,force_add=True)
    cfg.actor_rollout_ref.actor.optim.total_training_steps=330
    (OUT/'effective-config.yaml').write_text(OmegaConf.to_yaml(cfg))
    if args.inspect_only:
        import cloudpickle
        method=ConditionalBoundaryWorker.__ray_metadata__.modified_class.inspect_matched_layout
        serialized=cloudpickle.loads(cloudpickle.dumps(method))
        serialized_globals=registered_body(serialized).__globals__
        serialized_recipe=serialized_globals['ORIGINAL_RECIPE']
        callback_globals,callback_body=recipe_callback_binding(serialized_recipe)
        identity=existing.identity
        inspection=dict(scope='CPU binding/config only; no model construction.',
            worker_constructor=identity(ActorRolloutRefWorker.__init__),
            native_reader_inputs_sha256=PACK_SHA,serialized_observer=identity(serialized),
            original_recipe=identity(ORIGINAL_RECIPE),original_root_observer=identity(ORIGINAL_ROOT_OBSERVER),
            collector=identity(collect_actual_coefficients),cuda_initialized=torch.cuda.is_initialized(),
            actual_recipe_callback_binding=dict(body=identity(callback_body),
                callback=identity(callback_globals['observe_original_root']),
                module_globals_are_actual_recipe_globals=callback_globals is existing.__dict__,
                register_closure_is_metadata_body=callback_body is inspect.unwrap(serialized_recipe),
                actual_recipe_body_code=dict(filename=callback_body.__code__.co_filename,
                    first_line=callback_body.__code__.co_firstlineno),
                actual_serialized_worker_body_code=dict(filename=registered_body(serialized).__code__.co_filename,
                    first_line=registered_body(serialized).__code__.co_firstlineno),
                register_wrapper_source=dict(path=register.__code__.co_filename,sha256=REGISTER_SHA),
                ray_tracing_wrapper_sha256=RAY_TRACING_SHA,
                globals_resolved_from_serialized_registered_body=True),
            actor_config=OmegaConf.to_container(cfg.actor_rollout_ref,resolve=True))
        (OUT/'native-owner-inspection.json').write_text(json.dumps(inspection,indent=2)+'\n')
        return
    ray.init(num_cpus=8,include_dashboard=False)
    try:
        group=RayWorkerGroup(RayResourcePool([2],use_gpu=True,max_colocate_count=1),
            RayClassWithInitArgs(ConditionalBoundaryWorker,cfg.actor_rollout_ref,'actor'))
        print('TEXTCRAFT_CONDITIONAL_BOUNDARY_DRIVER '+json.dumps(dict(phase='original_model_initialization_begin',unix=time.time())),flush=True)
        group.init_model()
        print('TEXTCRAFT_CONDITIONAL_BOUNDARY_DRIVER '+json.dumps(dict(phase='original_checkpoint_load_begin',unix=time.time())),flush=True)
        group.load_checkpoint(local_path=str(checkpoint/'actor'),del_local_after_load=False)
        print('TEXTCRAFT_CONDITIONAL_BOUNDARY_DRIVER '+json.dumps(dict(phase='original_checkpoint_load_complete',unix=time.time())),flush=True)
        ranks=group.inspect_matched_layout()
        (OUT/'completed.json').write_text(json.dumps(dict(completed_unix=time.time(),groups=14,cases=42,
            paired_root_entries=42,finite_trace_calls=14,finite_seed_calls=14,
            optimizer_steps=0,scheduler_steps=0,backward_calls=0,ranks=ranks),indent=2)+'\n')
    finally:
        ray.shutdown()


if __name__=='__main__':
    main()

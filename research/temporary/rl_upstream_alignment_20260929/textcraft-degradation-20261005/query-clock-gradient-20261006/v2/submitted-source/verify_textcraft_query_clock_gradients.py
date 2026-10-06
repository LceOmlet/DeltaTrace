"""Replay saved native64 credit through the original owner, then observe PG/H."""
import argparse
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import sys
import time

ADAM = Path(os.environ['DT_TEXTCRAFT_ADAM_RECIPE_ROOT'])
sys.path.insert(0, str(ADAM))
import verify_textcraft_native_adam as recipe
from owner_trajectory_batch import trajectory_credit
from dt_training_batch import CREDIT_KEYS
from verl.trainer.ppo.ray_trainer import compute_advantage, AdvantageEstimator
from observe_textcraft_query_clock_gradients import GROUPS, LABELS

OUT, BASE, CHECKPOINT, INPUT = recipe.OUT, recipe.BASE, recipe.CHECKPOINT, recipe.INPUT
POPPED = (*CREDIT_KEYS, 'advantages', 'returns')


@recipe.ray.remote
class QueryClockGradientWorker(recipe.ActorRolloutRefWorker):
    @recipe.register(dispatch_mode=recipe.Dispatch.DP_COMPUTE_PROTO)
    def observe_clock_gradients(self, data):
        from observe_textcraft_query_clock_gradients import observe_query_clock_gradients
        return observe_query_clock_gradients(self, data,
            original_update_actor=super().update_actor, output_directory=OUT)

    @recipe.register(dispatch_mode=recipe.Dispatch.DP_COMPUTE_PROTO)
    def compute_dt_token_advantages(self, data):
        from observe_textcraft_query_clock_gradients import recompute_clock_credit
        return recompute_clock_credit(self, data,
            original_compute_credit=super().compute_dt_token_advantages, output_directory=OUT)


def retained_snapshot(saved):
    result = recipe._snapshot(saved)
    result.pop(batch_keys=list(POPPED))
    return result


def inspect_inputs(cpu_contract):
    import cloudpickle
    import psutil
    from transformers import AutoTokenizer
    from verl.workers.actor.dp_actor import DataParallelPPOActor
    from verl.trainer.ppo.core_algos import compute_policy_loss
    started = time.monotonic()
    boundary = json.loads((BASE / 'native-minibatch-update-boundary.json').read_bytes())
    input_sha = hashlib.sha256(INPUT.read_bytes()).hexdigest()
    assert input_sha == boundary['snapshot']['sha256']
    saved = recipe.DataProto.load_from_disk(str(INPUT))
    mapping = json.loads((BASE / 'native-minibatch-readout-mapping.json').read_bytes())
    credit_input = mapping['inputs']['credits']
    credit_path = Path(credit_input['path'])
    assert hashlib.sha256(credit_path.read_bytes()).hexdigest() == credit_input['sha256']
    credits = recipe.DataProto.load_from_disk(str(credit_path))
    cfg_path = ADAM / 'effective-config.yaml'
    cfg = recipe.OmegaConf.load(cfg_path)
    prior = json.loads((ADAM / 'native-owner-inspection.json').read_bytes())
    assert len(saved) == 64
    assert recipe.OmegaConf.to_container(cfg.actor_rollout_ref, resolve=True) == prior['actor_config']
    assert str(CHECKPOINT) == prior['checkpoint']
    assert AdvantageEstimator(cfg.algorithm.adv_estimator) == AdvantageEstimator.DELTATRACE
    tokenizer = AutoTokenizer.from_pretrained(cfg.actor_rollout_ref.model.path, local_files_only=True)
    observer = importlib.import_module('observe_native_optimizer_minibatch')
    adapter = importlib.import_module('observe_textcraft_query_clock_gradients')
    candidate = importlib.import_module('reward_readout_clock_candidate')
    readout_owner = importlib.import_module('reward_readout')
    sources = dict(worker_constructor=recipe.identity(recipe.ActorRolloutRefWorker.__init__),
        worker_update=recipe.identity(recipe.ActorRolloutRefWorker.update_actor),
        worker_checkpoint_loader=recipe.identity(recipe.ActorRolloutRefWorker.load_checkpoint),
        worker_credit=recipe.identity(recipe.ActorRolloutRefWorker.compute_dt_token_advantages),
        actor=recipe.identity(DataParallelPPOActor), core=recipe.identity(compute_policy_loss),
        original_observer=recipe.identity(observer.observe_native_optimizer_minibatch),
        original_loss_hooks=recipe.identity(observer.NativeLossHooks), snapshot=recipe.identity(recipe._snapshot),
        trajectory_credit=recipe.identity(trajectory_credit), compute_advantage=recipe.identity(compute_advantage),
        original_query=recipe.identity(readout_owner.RewardAlphabet.query_ids),
        candidate_query=recipe.identity(candidate.RewardAlphabet.query_ids),
        adapter=recipe.identity(adapter.recompute_clock_credit), driver=recipe.identity(inspect_inputs))
    for name in ('worker_constructor', 'worker_update', 'worker_checkpoint_loader', 'actor', 'core', 'snapshot'):
        assert sources[name] == prior['sources'][name]
    assert sources['candidate_query']['sha256'] == '94a7afbc09da72b62572d31fd32a6534f6e8f3daf656fce1011cdfa68b3c3e2b'
    assert sources['compute_advantage']['sha256'] == '8816ea4e5f9a95a1dfe1067349eee9101da8bf4ec916bcc28a5b03288976f0df'
    retained = retained_snapshot(saved)
    assert set(saved.batch.keys()) - set(retained.batch.keys()) == set(POPPED)
    for key in retained.batch.keys():
        assert recipe.torch.equal(saved.batch[key], retained.batch[key]), key
    assert retained.meta_info == saved.meta_info
    assert set(retained.non_tensor_batch) == set(saved.non_tensor_batch)
    # Original select(deepcopy=True) preserves every non-tensor field, including
    # source slices; JSON encoding is not used to reconstruct these artifacts.
    import numpy as np
    for key in retained.non_tensor_batch:
        assert np.array_equal(retained.non_tensor_batch[key], saved.non_tensor_batch[key]), key
    serialized_methods = {}
    for name in ('observe_clock_gradients', 'compute_dt_token_advantages'):
        method = getattr(QueryClockGradientWorker.__ray_metadata__.modified_class, name)
        restored = cloudpickle.loads(cloudpickle.dumps(method))
        assert recipe.identity(restored) == recipe.identity(method)
        serialized_methods[name] = recipe.identity(restored)
    contract = {}
    if cpu_contract:
        # Reuse the already verified original CPU capture. Import only in this
        # CPU branch: its module-level CUDA hiding must never affect a GPU run.
        from map_textcraft_native64_readout import CaptureDT, CapturedNativeRPC
        from agent_system.multi_turn_rollout.utils import to_list_of_dict
        capture = CaptureDT()
        try:
            trajectory_credit(saved, credits, capture, eos_token_id=tokenizer.eos_token_id,
                              pad_token_id=tokenizer.pad_token_id)
        except CapturedNativeRPC:
            pass
        assert capture.requests is not None
        query_lengths = []
        original_method = readout_owner.RewardAlphabet.query_ids
        for partition in capture.requests.chunk(2):
            rows = to_list_of_dict(partition)
            args = dict(task='TextCraft', max_steps=cfg.env.max_steps, max_length=32768,
                minibatch_size=4, packed_answer_targets=None, invalid_action_penalty_coef=0.0,
                sampling=json.loads(os.environ['DT_SAMPLING_JSON']))
            readout = readout_owner.EventRatioReadout(None, tokenizer, **args)
            report = lambda: dict(nonzero_reward_events=0, policy_tokens=0, actual_row_lengths=[])
            _, old = readout._prepare_episode(rows, report(), partition.batch['dt_complete_return'].tolist())
            with observer._temporary_attribute(readout_owner.RewardAlphabet, 'query_ids', candidate.RewardAlphabet.query_ids):
                _, new = readout._prepare_episode(rows, report(), partition.batch['dt_complete_return'].tolist())
            assert readout_owner.RewardAlphabet.query_ids is original_method
            assert len(old) == len(new)
            for a, b in zip(old, new):
                for key in ('prompt', 'actions', 'target'):
                    assert recipe.torch.equal(a[key], b[key]), key
                for key in ('start', 'end', 'source_step', 'traj_uid', 'observed_return', 'row_index'):
                    assert a[key] == b[key], key
                query_lengths.append([a['query_tokens'], b['query_tokens']])
        assert len(capture.requests) == 192 and len(query_lengths) == 192
        contract = dict(native_rpc_rows=len(capture.requests), query_lengths=query_lengths,
            source_target_return_identity_preserved=True, original_query_method_restored=True,
            capture_source=recipe.identity(CaptureDT), capture_returns_no_fake_credit=True)
    value = dict(scope='Prepared saved original artifacts/config/serialized methods and CPU owner contract; no model/GPU/backward.',
        native_reader_inputs_sha256=input_sha, rows=len(saved), credit_input=credit_input,
        worker_constructor=sources['worker_constructor'], sources=sources, checkpoint=str(CHECKPOINT),
        config_source=dict(path=str(cfg_path), sha256=hashlib.sha256(cfg_path.read_bytes()).hexdigest()),
        actor_config=prior['actor_config'], labels=list(LABELS), groups=list(GROUPS),
        compute_advantage_signature=str(inspect.signature(compute_advantage)),
        popped_batch_fields=list(POPPED), all_other_saved_fields_retained=True,
        eos_token_id=int(tokenizer.eos_token_id), pad_token_id=int(tokenizer.pad_token_id),
        serialized_worker_methods=serialized_methods, cpu_owner_contract=contract,
        cuda_initialized=recipe.torch.cuda.is_initialized(), distributed_initialized=recipe.torch.distributed.is_initialized(),
        pid=os.getpid(), pid_birth=psutil.Process().create_time(), rss_bytes=psutil.Process().memory_info().rss,
        elapsed_seconds=time.monotonic()-started, sampling_calls=0, DT_calls=0, backward_calls=0,
        optimizer_steps=0, scheduler_steps=0)
    path = OUT / ('native-owner-inspection.json' if cpu_contract else 'runtime-driver-inspection.json')
    path.write_text(json.dumps(value, indent=2) + '\n')
    return saved, credits, cfg, value


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inspect-only', action='store_true')
    args = parser.parse_args()
    saved, credits, cfg, inspection = inspect_inputs(args.inspect_only)
    if args.inspect_only:
        print(json.dumps(dict(status='inspected_no_model', rows=len(saved),
            native_rpc_rows=inspection['cpu_owner_contract']['native_rpc_rows'], cuda_initialized=inspection['cuda_initialized'])))
    else:
        recipe.ray.init(num_cpus=8, include_dashboard=False)
        try:
            group = recipe.RayWorkerGroup(recipe.RayResourcePool([2], use_gpu=True, max_colocate_count=1),
                recipe.RayClassWithInitArgs(QueryClockGradientWorker, cfg.actor_rollout_ref, 'actor'))
            recipe.phase('original_model_init_begin')
            group.init_model()
            recipe.phase('original_complete_checkpoint_restore_begin')
            group.load_checkpoint(local_path=str(CHECKPOINT), del_local_after_load=False)
            recipe.phase('original_complete_checkpoint_restore_complete')
            old = recipe._snapshot(saved)
            old.meta_info['query_clock_gradient_group'] = GROUPS[0]
            recipe.phase('saved_original_gradient_begin')
            recipe.save_artifact('saved-original-worker-output.pkl', group.observe_clock_gradients(old))
            recipe.phase('saved_original_gradient_complete')
            recipe.phase('original_trajectory_credit_clock_begin')
            credit = trajectory_credit(saved, credits, group,
                eos_token_id=inspection['eos_token_id'], pad_token_id=inspection['pad_token_id'])
            new = retained_snapshot(saved).union(credit)
            new = compute_advantage(new, adv_estimator=AdvantageEstimator(cfg.algorithm.adv_estimator),
                gamma=cfg.algorithm.gamma, lam=cfg.algorithm.lam, num_repeat=cfg.actor_rollout_ref.rollout.n,
                multi_turn=cfg.actor_rollout_ref.rollout.multi_turn.enable,
                norm_adv_by_std_in_grpo=cfg.algorithm.get('norm_adv_by_std_in_grpo', True),
                use_pf_ppo=cfg.algorithm.use_pf_ppo, pf_ppo_reweight_method=cfg.algorithm.pf_ppo.reweight_method,
                pf_ppo_weight_pow=cfg.algorithm.pf_ppo.weight_pow,
                step_advantage_w=cfg.algorithm.gigpo.step_advantage_w, gigpo_mode=cfg.algorithm.gigpo.mode,
                gigpo_enable_similarity=cfg.algorithm.gigpo.enable_similarity,
                gigpo_similarity_thresh=cfg.algorithm.gigpo.similarity_thresh)
            new.meta_info['query_clock_gradient_group'] = GROUPS[1]
            artifact = recipe.save_artifact('clock-recomputed-native-minibatch.pkl', new)
            recipe.phase('original_trajectory_credit_clock_complete', artifact=artifact)
            recipe.phase('clock_recomputed_gradient_begin')
            recipe.save_artifact('clock-recomputed-worker-output.pkl', group.observe_clock_gradients(new))
            recipe.phase('clock_recomputed_gradient_complete')
            (OUT / 'completed.json').write_text(json.dumps(dict(sources=inspection['sources'],
                input_sha256=inspection['native_reader_inputs_sha256'], credit_input=inspection['credit_input'],
                groups=list(GROUPS), labels=list(LABELS), native_backward_passes_per_rank=4,
                checkpoint=str(CHECKPOINT), sampling_calls=0, optimizer_steps=0, scheduler_steps=0,
                completed_unix=time.time(), scope='One original checkpoint load; saved old A and original owner-recomputed query-clock A observed with unchanged native PG/H.'), indent=2)+'\n')
        finally:
            recipe.ray.shutdown()

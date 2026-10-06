"""Observe saved native PG/H and equivalent-label native PG/H, without updates.

The unchanged native-Adam inspection entry owns saved data/config/source checks.
The original trajectory credit and compute_advantage own the recomputation;
only RewardAlphabet.labels is temporarily reversed by the diagnostic helper.
"""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ADAM = Path(os.environ['DT_TEXTCRAFT_ADAM_RECIPE_ROOT'])
sys.path.insert(0, str(ADAM))
import verify_textcraft_native_adam as recipe
from owner_trajectory_batch import trajectory_credit
from dt_training_batch import CREDIT_KEYS
from verl.trainer.ppo.ray_trainer import compute_advantage, AdvantageEstimator
from observe_textcraft_label_gradients import GROUPS, LABELS

OUT, BASE, CHECKPOINT, INPUT = recipe.OUT, recipe.BASE, recipe.CHECKPOINT, recipe.INPUT
assert OUT == Path(os.environ['DT_TEXTCRAFT_LABEL_GRADIENT_ROOT'])
POPPED = (*CREDIT_KEYS, 'advantages', 'returns')
QUERY_SHA = '228afbc7a10841d482c3d73def59dfe9ef192c057a97d76dca57369502a10137'


@recipe.ray.remote
class LabelGradientWorker(recipe.ActorRolloutRefWorker):
    @recipe.register(dispatch_mode=recipe.Dispatch.DP_COMPUTE_PROTO)
    def observe_label_gradients(self, data):
        from observe_textcraft_label_gradients import observe_label_gradients
        return observe_label_gradients(self, data,
            original_update_actor=super().update_actor, output_directory=OUT)

    @recipe.register(dispatch_mode=recipe.Dispatch.DP_COMPUTE_PROTO)
    def compute_dt_token_advantages(self, data):
        from observe_textcraft_label_gradients import recompute_labels_credit
        return recompute_labels_credit(self, data,
            original_compute_credit=super().compute_dt_token_advantages, output_directory=OUT)


def retained_snapshot(saved):
    result = recipe._snapshot(saved)
    result.pop(batch_keys=list(POPPED))
    return result


def inspect_inputs(cpu_contract):
    import cloudpickle
    import numpy as np
    import psutil
    from transformers import AutoTokenizer
    started = time.monotonic()
    # Execute the intact, already-bound owner inspection instead of copying its
    # config/data/source/serialization checks. Its output has a separate path.
    original_out = OUT / 'base-owner-inspection'
    original_out.mkdir(exist_ok=True)
    with (OUT / 'base-owner-inspection.stdout.txt').open('wb') as log:
        inspected = subprocess.run([sys.executable, str(ADAM / 'verify_textcraft_native_adam.py'),
            '--inspect-only'], env=dict(os.environ,
            DT_TEXTCRAFT_READOUT_ROOT=str(original_out), CUDA_VISIBLE_DEVICES=''),
            stdout=log, stderr=subprocess.STDOUT)
    assert inspected.returncode == 0, 'Original native inspection failed; no model submission'
    base_receipt = original_out / 'native-owner-inspection.json'
    prior = json.loads(base_receipt.read_bytes())
    original_prior = json.loads((ADAM / 'native-owner-inspection.json').read_bytes())
    assert prior['actor_config'] == original_prior['actor_config']
    assert prior['checkpoint'] == original_prior['checkpoint'] == str(CHECKPOINT)
    assert not prior['cuda_initialized']
    config_path = original_out / 'effective-config.yaml'
    assert config_path.read_bytes() == (ADAM / 'effective-config.yaml').read_bytes()
    cfg = recipe.OmegaConf.load(config_path)
    assert AdvantageEstimator(cfg.algorithm.adv_estimator) == AdvantageEstimator.DELTATRACE
    saved = recipe.DataProto.load_from_disk(str(INPUT))
    mapping = json.loads((BASE / 'native-minibatch-readout-mapping.json').read_bytes())
    credit_input = mapping['inputs']['credits']
    credit_path = Path(credit_input['path'])
    assert hashlib.sha256(credit_path.read_bytes()).hexdigest() == credit_input['sha256']
    credits = recipe.DataProto.load_from_disk(str(credit_path))
    tokenizer = AutoTokenizer.from_pretrained(cfg.actor_rollout_ref.model.path, local_files_only=True)
    observer = importlib.import_module('observe_native_optimizer_minibatch')
    adapter = importlib.import_module('observe_textcraft_label_gradients')
    readout = importlib.import_module('reward_readout')
    sources = dict(original_native_inspection=recipe.identity(recipe.phase),
        original_query=recipe.identity(readout.RewardAlphabet.query_ids),
        original_labels=recipe.identity(readout.RewardAlphabet.labels),
        original_label_ids=recipe.identity(readout.RewardAlphabet.label_ids),
        trajectory_credit=recipe.identity(trajectory_credit), compute_advantage=recipe.identity(compute_advantage),
        original_observer=recipe.identity(observer.observe_native_optimizer_minibatch),
        original_gradient_statistics=recipe.identity(observer.native_gradient_statistics),
        adapter=recipe.identity(adapter.observe_label_gradients), driver=recipe.identity(inspect_inputs))
    assert sources['original_query']['sha256'] == QUERY_SHA
    assert sources['compute_advantage']['sha256'] == '8816ea4e5f9a95a1dfe1067349eee9101da8bf4ec916bcc28a5b03288976f0df'
    retained = retained_snapshot(saved)
    assert set(saved.batch.keys()) - set(retained.batch.keys()) == set(POPPED)
    for key in retained.batch.keys():
        assert recipe.torch.equal(saved.batch[key], retained.batch[key]), key
    assert retained.meta_info == saved.meta_info
    assert set(retained.non_tensor_batch) == set(saved.non_tensor_batch)
    for key in retained.non_tensor_batch:
        assert np.array_equal(saved.non_tensor_batch[key], retained.non_tensor_batch[key]), key
    serialized = {}
    cls = LabelGradientWorker.__ray_metadata__.modified_class
    assert recipe.identity(cls.__init__) == prior['worker_constructor']
    for name in ('observe_label_gradients', 'compute_dt_token_advantages'):
        method = getattr(cls, name)
        restored = cloudpickle.loads(cloudpickle.dumps(method))
        assert recipe.identity(restored) == recipe.identity(method)
        serialized[name] = recipe.identity(restored)
    contract = {}
    if cpu_contract:
        # Existing capture stops at the original RPC; it never invents credit.
        from map_textcraft_native64_readout import CaptureDT, CapturedNativeRPC
        from agent_system.multi_turn_rollout.utils import to_list_of_dict
        capture = CaptureDT()
        try:
            trajectory_credit(saved, credits, capture,
                eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
        except CapturedNativeRPC:
            pass
        assert capture.requests is not None
        mapped_artifact = mapping['mapped_requests_artifact']
        mapped_path = Path(mapped_artifact['path'])
        assert hashlib.sha256(mapped_path.read_bytes()).hexdigest() == mapped_artifact['sha256']
        literal_rows = json.loads(mapped_path.read_bytes())['requests']
        literals = {}
        for item in literal_rows:
            literals.setdefault((item['rank'], item['traj_uid'], item['env_step']), []).append(item)
        method = readout.RewardAlphabet.labels
        query_method = readout.RewardAlphabet.query_ids
        checks = []
        for rank, partition in enumerate(capture.requests.chunk(2)):
            owner = readout.EventRatioReadout(None, tokenizer, task='TextCraft',
                max_steps=cfg.env.max_steps, max_length=32768, minibatch_size=4,
                packed_answer_targets=None, invalid_action_penalty_coef=0.0,
                sampling=json.loads(os.environ['DT_SAMPLING_JSON']))
            rows = to_list_of_dict(partition)
            report = lambda: dict(nonzero_reward_events=0, policy_tokens=0, actual_row_lengths=[])
            _, old = owner._prepare_episode(rows, report(), partition.batch['dt_complete_return'].tolist())
            old_ids = owner.alphabet.label_ids(tokenizer)
            with observer._temporary_attribute(readout.RewardAlphabet, 'labels', adapter.swapped_labels):
                _, new = owner._prepare_episode(rows, report(), partition.batch['dt_complete_return'].tolist())
                new_ids = owner.alphabet.label_ids(tokenizer)
            assert readout.RewardAlphabet.labels is method
            assert readout.RewardAlphabet.query_ids is query_method
            assert old_ids == [15, 16] and new_ids == [16, 15]
            assert len(old) == len(new)
            for a, b in zip(old, new):
                for key in ('prompt', 'actions'):
                    assert recipe.torch.equal(a[key], b[key]), key
                for key in ('start', 'end', 'source_step', 'traj_uid', 'observed_return',
                            'row_index', 'context_tokens', 'query_tokens'):
                    assert a[key] == b[key], key
                changed = (a['query'] != b['query']).nonzero().flatten().tolist()
                assert len(changed) == 2
                assert all({int(a['query'][i]), int(b['query'][i])} == {15, 16} for i in changed)
                observed = owner.alphabet.observed_index(a['observed_return'])
                assert int(a['target'][0]) == old_ids[observed]
                assert int(b['target'][0]) == new_ids[observed]
                native_ids = recipe.torch.cat(tuple(a[key] for key in ('prompt', 'actions', 'query', 'target'))).tolist()
                originals = literals[(rank, a['traj_uid'], a['source_step'])]
                assert all(native_ids == item['selected_input_ids'] for item in originals)
                checks.append(dict(rank=rank, traj_uid=a['traj_uid'], source_step=a['source_step'],
                    query_tokens=a['query_tokens'], query_label_positions=changed,
                    observed_index=observed, target_ids=[int(a['target'][0]), int(b['target'][0])]))
        assert len(capture.requests) == len(checks) == 192
        contract = dict(native_rpc_rows=len(capture.requests), requests=checks,
            original_literal_request_identity_preserved=True,
            only_query_label_ids_and_target_encoding_changed=True,
            original_query_method_unchanged=True, original_labels_method_restored=True,
            mapped_requests=mapped_artifact, capture_source=recipe.identity(CaptureDT),
            capture_returns_no_fake_credit=True)
    value = dict(scope='Original saved artifact/config inspection and labels-only CPU owner contract; no model, DT or backward.',
        native_reader_inputs_sha256=prior['native_reader_inputs_sha256'], rows=len(saved),
        credit_input=credit_input, checkpoint=str(CHECKPOINT), worker_constructor=prior['worker_constructor'],
        sources=sources, original_inspection=dict(path=str(base_receipt),
            sha256=hashlib.sha256(base_receipt.read_bytes()).hexdigest()),
        config_source=dict(path=str(config_path), sha256=hashlib.sha256(config_path.read_bytes()).hexdigest()),
        actor_config=prior['actor_config'], labels=list(LABELS), groups=list(GROUPS),
        popped_batch_fields=list(POPPED), all_other_saved_fields_retained=True,
        eos_token_id=int(tokenizer.eos_token_id), pad_token_id=int(tokenizer.pad_token_id),
        serialized_worker_methods=serialized, cpu_owner_contract=contract,
        cuda_initialized=recipe.torch.cuda.is_initialized(),
        distributed_initialized=recipe.torch.distributed.is_initialized(),
        pid=os.getpid(), pid_birth=psutil.Process().create_time(), rss_bytes=psutil.Process().memory_info().rss,
        elapsed_seconds=time.monotonic()-started, sampling_calls=0, DT_calls=0,
        backward_calls=0, optimizer_steps=0, scheduler_steps=0)
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
            native_rpc_rows=inspection['cpu_owner_contract']['native_rpc_rows'],
            cuda_initialized=inspection['cuda_initialized'])))
    else:
        recipe.ray.init(num_cpus=8, include_dashboard=False)
        try:
            group = recipe.RayWorkerGroup(recipe.RayResourcePool([2], use_gpu=True, max_colocate_count=1),
                recipe.RayClassWithInitArgs(LabelGradientWorker, cfg.actor_rollout_ref, 'actor'))
            recipe.phase('original_model_init_begin')
            group.init_model()
            recipe.phase('original_complete_checkpoint_restore_begin')
            group.load_checkpoint(local_path=str(CHECKPOINT), del_local_after_load=False)
            recipe.phase('original_complete_checkpoint_restore_complete')
            old = recipe._snapshot(saved)
            old.meta_info['label_gradient_group'] = GROUPS[0]
            recipe.phase('saved_original_gradient_begin')
            recipe.save_artifact('saved-original-worker-output.pkl', group.observe_label_gradients(old))
            recipe.phase('saved_original_gradient_complete')
            recipe.phase('original_trajectory_credit_labels_begin')
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
            new.meta_info['label_gradient_group'] = GROUPS[1]
            artifact = recipe.save_artifact('labels-recomputed-native-minibatch.pkl', new)
            recipe.phase('original_trajectory_credit_labels_complete', artifact=artifact)
            recipe.phase('swapped_labels_gradient_begin')
            recipe.save_artifact('swapped-labels-worker-output.pkl', group.observe_label_gradients(new))
            recipe.phase('swapped_labels_gradient_complete')
            (OUT / 'completed.json').write_text(json.dumps(dict(sources=inspection['sources'],
                input_sha256=inspection['native_reader_inputs_sha256'], credit_input=inspection['credit_input'],
                groups=list(GROUPS), labels=list(LABELS), native_backward_passes_per_rank=4,
                checkpoint=str(CHECKPOINT), sampling_calls=0, optimizer_steps=0, scheduler_steps=0,
                completed_unix=time.time(), scope='One original checkpoint load; saved original A and labels-only original DT recomputation with unchanged native PG/H and cross-PG original statistics.'), indent=2)+'\n')
        finally:
            recipe.ray.shutdown()

"""Observe saved raw/officially whitened native64 advantages without updates.

The patched trainer owns full-batch whitening. Completed native recipes own
configuration/loading, PG/H/KL backwards, clipping, RNG and gradient statistics.
No rollout, DT, model reconstruction, new PPO loss or optimizer step is added.
"""
import argparse
from contextlib import ExitStack
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

OUT, BASE, CHECKPOINT, INPUT = recipe.OUT, recipe.BASE, recipe.CHECKPOINT, recipe.INPUT
assert OUT == Path(os.environ['DT_TEXTCRAFT_WHITENING_ROOT'])
GROUPS = ('raw', 'white')
LABELS = ('dt_pg', 'weighted_entropy', 'weighted_kl')
LABEL_HELPER_SHA = '9ed52c51f5eac34a2b12d5425e386f4b3e5b5945f2c6b61e01b25aba837eddb7'


def observe_whitening(worker, data, *, original_update_actor):
    """Compose the existing paired-RPC observer; return the original output."""
    owner = importlib.import_module('observe_native_optimizer_minibatch')
    paired = importlib.import_module('observe_textcraft_label_gradients')
    assert recipe.identity(paired.observe_label_gradients)['sha256'] == LABEL_HELPER_SHA
    branch = data.meta_info['label_gradient_group']
    with ExitStack() as stack:
        stack.enter_context(owner._temporary_attribute(paired, 'GROUPS', GROUPS))
        stack.enter_context(owner._temporary_attribute(paired, 'LABELS', LABELS))
        output = paired.observe_label_gradients(worker, data,
            original_update_actor=original_update_actor, output_directory=OUT)
    # Keep the reused helper's exact report, including its original naming.
    path = OUT / f'rank{worker.rank}-{branch}-gradients.json'
    raw = path.read_bytes()
    (OUT / f'rank{worker.rank}-{branch}-reused-helper-report.json').write_bytes(raw)
    record = json.loads(raw)
    record['reused_helper_scope'] = record['scope']
    record['scope'] = ('Saved full native local32/B4x8 /8 carrier, three original '
        'component backwards PG/H/KL, no optimizer or scheduler update.')
    record['whitening_branch'] = branch
    record['diagnostic_labels'] = list(LABELS)
    record['diagnostic_adapter'] = recipe.identity(observe_whitening)
    record['reused_label_observer_source'] = recipe.identity(paired.observe_label_gradients)
    record['cross_group_aliases'] = {
        'old_dt_pg': 'raw PG', 'swapped_dt_pg': 'officially whitened PG',
        'weighted_entropy': 'white branch original coefficient-weighted H',
    }
    path.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    return output


@recipe.ray.remote
class WhiteningObservationWorker(recipe.ActorRolloutRefWorker):
    @recipe.register(dispatch_mode=recipe.Dispatch.DP_COMPUTE_PROTO)
    def observe_whitening(self, data):
        # Resolve through the worker's actual import environment, not a
        # cloudpickled driver's module-global callback alias.
        from verify_textcraft_official_whitening import observe_whitening
        return observe_whitening(self, data, original_update_actor=super().update_actor)


def describe(values, mask):
    from verl.utils.torch_functional import masked_mean, masked_var
    x = values[mask].detach().double()
    return dict(valid_tokens=int(mask.sum()), dtype=str(values.dtype),
        masked_mean=float(masked_mean(values, mask)),
        masked_variance=float(masked_var(values, mask)),
        descriptive_FP64_sum=float(x.sum()), descriptive_FP64_L1=float(x.abs().sum()),
        descriptive_FP64_mean_abs=float(x.abs().mean()),
        positive_tokens=int((x > 0).sum()), negative_tokens=int((x < 0).sum()),
        zero_tokens=int((x == 0).sum()))


def normalized_snapshot(saved, cfg):
    import numpy as np
    from verl.trainer.ppo.ray_trainer import compute_advantage, AdvantageEstimator
    from verl.utils.torch_functional import masked_whiten
    torch = recipe.torch
    source = recipe.identity(compute_advantage)
    expected = os.environ['DT_TEXTCRAFT_WHITENING_TRAINER_SHA']
    assert source['sha256'] == expected
    assert Path(source['path']).is_relative_to(Path(os.environ['VERL_ROOT']))
    raw = recipe._snapshot(saved)
    white = recipe._snapshot(saved)
    white = compute_advantage(white,
        adv_estimator=AdvantageEstimator(cfg.algorithm.adv_estimator),
        gamma=cfg.algorithm.gamma, lam=cfg.algorithm.lam,
        num_repeat=cfg.actor_rollout_ref.rollout.n,
        multi_turn=cfg.actor_rollout_ref.rollout.multi_turn.enable,
        norm_adv_by_std_in_grpo=cfg.algorithm.get('norm_adv_by_std_in_grpo', True),
        use_pf_ppo=cfg.algorithm.use_pf_ppo,
        pf_ppo_reweight_method=cfg.algorithm.pf_ppo.reweight_method,
        pf_ppo_weight_pow=cfg.algorithm.pf_ppo.weight_pow,
        step_advantage_w=cfg.algorithm.gigpo.step_advantage_w,
        gigpo_mode=cfg.algorithm.gigpo.mode,
        gigpo_enable_similarity=cfg.algorithm.gigpo.enable_similarity,
        gigpo_similarity_thresh=cfg.algorithm.gigpo.similarity_thresh)
    response_length = saved.batch['responses'].shape[-1]
    mask = saved.batch['loss_mask'][:, -response_length:].bool()
    expected_white = masked_whiten(saved.batch['dt_token_advantages'], mask) * mask
    exact = torch.equal(expected_white, white.batch['advantages'])
    if not exact:
        raise RuntimeError('Patched trainer output differs from the installed official masked_whiten call')
    unchanged = {key: torch.equal(saved.batch[key], white.batch[key])
        for key in sorted(saved.batch.keys()) if key != 'advantages'}
    if not all(unchanged.values()):
        raise RuntimeError('The whitening seam modified an original non-advantages tensor')
    non_tensor_equal = {key: bool(np.array_equal(saved.non_tensor_batch[key],
        white.non_tensor_batch[key])) for key in saved.non_tensor_batch}
    if not all(non_tensor_equal.values()) or saved.meta_info != white.meta_info:
        raise RuntimeError('The whitening seam modified original non-tensor metadata')
    return raw, white, dict(
        scope='One trainer compute_advantage call over the full saved collected native64, before rank/microbatch splitting; direct official helper parity is CPU interface verification.',
        trainer_source=source, official_helper_source=recipe.identity(masked_whiten),
        helper_call='masked_whiten(saved dt_token_advantages, original loss_mask response tail) * same mask; default shift_mean=True',
        full_collected_rows=len(saved), trainer_compute_advantage_calls=1,
        direct_official_helper_calls=1, official_helper_exact_equal=exact,
        normalization_mask='loss_mask[:, -responses.shape[-1]:]',
        original_tensor_field_identity=unchanged,
        original_non_tensor_field_identity=non_tensor_equal,
        original_meta_info_exact=True, raw_A=describe(raw.batch['advantages'], mask),
        white_A=describe(white.batch['advantages'], mask),
        raw_QVA_retained=True, observed_return_and_saved_old_ref_retained=True,
        model_forward_calls=0, DT_calls=0, backward_calls=0, optimizer_steps=0)


def inspect_inputs():
    import cloudpickle
    import psutil
    started = time.monotonic()
    base_out = OUT / 'base-owner-inspection'
    base_out.mkdir(exist_ok=True)
    with (OUT / 'base-owner-inspection.stdout.txt').open('wb') as log:
        original = subprocess.run([sys.executable, str(ADAM / 'verify_textcraft_native_adam.py'),
            '--inspect-only'], env=dict(os.environ,
            DT_TEXTCRAFT_READOUT_ROOT=str(base_out), CUDA_VISIBLE_DEVICES=''),
            stdout=log, stderr=subprocess.STDOUT)
    assert original.returncode == 0, 'Original native owner inspection failed before model submission'
    base_receipt = json.loads((base_out / 'native-owner-inspection.json').read_bytes())
    prior = json.loads((ADAM / 'native-owner-inspection.json').read_bytes())
    assert base_receipt['actor_config'] == prior['actor_config']
    assert base_receipt['checkpoint'] == prior['checkpoint'] == str(CHECKPOINT)
    config_path = base_out / 'effective-config.yaml'
    assert config_path.read_bytes() == (ADAM / 'effective-config.yaml').read_bytes()
    cfg = recipe.OmegaConf.load(config_path)
    for name in ('worker_constructor', 'worker_update', 'worker_checkpoint_loader', 'actor', 'core'):
        assert base_receipt['sources'][name]['sha256'] == prior['sources'][name]['sha256']
    saved = recipe.DataProto.load_from_disk(str(INPUT))
    assert len(saved) == 64
    raw, white, normalization = normalized_snapshot(saved, cfg)
    artifact = recipe.save_artifact('officially-whitened-native-minibatch.pkl', white)
    paired = importlib.import_module('observe_textcraft_label_gradients')
    cls = WhiteningObservationWorker.__ray_metadata__.modified_class
    assert recipe.identity(cls.__init__)['sha256'] == prior['worker_constructor']['sha256']
    method = cls.observe_whitening
    assert recipe.identity(cloudpickle.loads(cloudpickle.dumps(method))) == recipe.identity(method)
    report = dict(scope='Original native saved64/checkpoint25 owner inspection and official full-mask whitening; no GPU/model/backward.',
        native_reader_inputs_sha256=base_receipt['native_reader_inputs_sha256'],
        worker_constructor=base_receipt['worker_constructor'], checkpoint=str(CHECKPOINT),
        actor_config=prior['actor_config'], config_source=dict(path=str(config_path),
            sha256=hashlib.sha256(config_path.read_bytes()).hexdigest()),
        sources={**base_receipt['sources'], 'paired_observer': recipe.identity(paired.observe_label_gradients),
            'whitening_observer': recipe.identity(observe_whitening)},
        original_native_imports=prior['sources'], groups=list(GROUPS), labels=list(LABELS),
        normalization=normalization, white_carrier=artifact,
        rows=len(saved), serialized_worker_method=recipe.identity(method),
        cuda_initialized=recipe.torch.cuda.is_initialized(),
        distributed_initialized=recipe.torch.distributed.is_initialized(),
        pid=os.getpid(), pid_birth=psutil.Process().create_time(),
        rss_bytes=psutil.Process().memory_info().rss, elapsed_seconds=time.monotonic()-started,
        sampling_calls=0, DT_calls=0, backward_calls=0, optimizer_steps=0, scheduler_steps=0)
    (OUT / 'native-owner-inspection.json').write_text(json.dumps(report, indent=2) + '\n')
    return raw, white, cfg, report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inspect-only', action='store_true')
    args = parser.parse_args()
    raw, white, cfg, inspection = inspect_inputs()
    if args.inspect_only:
        print(json.dumps(dict(status='inspected_no_model', rows=len(raw),
            official_helper_exact_equal=inspection['normalization']['official_helper_exact_equal'],
            cuda_initialized=inspection['cuda_initialized'])))
    else:
        recipe.ray.init(num_cpus=8, include_dashboard=False)
        try:
            group = recipe.RayWorkerGroup(recipe.RayResourcePool([2], use_gpu=True, max_colocate_count=1),
                recipe.RayClassWithInitArgs(WhiteningObservationWorker, cfg.actor_rollout_ref, 'actor'))
            recipe.phase('original_model_init_begin')
            group.init_model()
            recipe.phase('original_complete_checkpoint_restore_begin')
            group.load_checkpoint(local_path=str(CHECKPOINT), del_local_after_load=False)
            for branch, carrier in zip(GROUPS, (raw, white)):
                data = recipe._snapshot(carrier)
                data.meta_info['label_gradient_group'] = branch
                recipe.phase('original_whitening_gradient_begin', branch=branch)
                recipe.save_artifact(f'{branch}-original-worker-output.pkl', group.observe_whitening(data))
                recipe.phase('original_whitening_gradient_complete', branch=branch)
            (OUT / 'completed.json').write_text(json.dumps(dict(
                scope='One original checkpoint load; raw/officially whitened A, original PG/H/KL per branch and native cross-PG Gram; no rollout/DT/update.',
                input_sha256=inspection['native_reader_inputs_sha256'], checkpoint=str(CHECKPOINT),
                sources=inspection['sources'], normalization=inspection['normalization'],
                white_carrier=inspection['white_carrier'], groups=list(GROUPS), labels=list(LABELS),
                native_backward_passes_per_rank=6, sampling_calls=0, DT_calls=0,
                optimizer_steps=0, scheduler_steps=0, completed_unix=time.time()), indent=2) + '\n')
        finally:
            recipe.ray.shutdown()

"""Frozen TextCraft development-tail contributions through unchanged VERL PPO.

All four original global64/local32 optimizer minibatches at the same untouched
base LoRA, not a sequential training replay. Three original PG backward passes
per minibatch: full saved coefficients, all observed development c>2 sources,
and the subset with native single-deletion d>=0. Original whole-batch whitening
is retained, never recomputed for a subset. No native values replace credits.
This measures contribution, not a corrected gradient or historical causality.
"""
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time

import torch
import ray
from omegaconf import OmegaConf
from verl.protocol import DataProto
from verl.utils.model import compute_position_id_with_mask
from verl.single_controller.base.decorator import Dispatch, register
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from inspect_extreme_endpoint import check_imports, actor_initialization_steps
import observe_native_optimizer_minibatch as observer

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT = ROOT / 'receipts/credit-collection-gradients-20261008-v1'
CAPTURE = ROOT / 'receipts/direct-target-prefix-runtime-20261007-v1/textcraft-first-dt'
SOURCE = ROOT / 'runs/direct-target-prefix-runtime-20261007-v1/textcraft/textcraft-dt/source.json'
ACTOR = '3a65e173300be82a7a9e056a96227c4f746eabc3be778ef41c8d50138d52ce6c'
LABELS = ('full_pg', 'development_predicted_tail', 'development_tail_native_sign_flip')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(value):
    path = inspect.getsourcefile(inspect.unwrap(value))
    return dict(path=path, sha256=sha(path))


def phase(name, **kw):
    value = dict(phase=name, unix=time.time(), pid=os.getpid(), **kw)
    (OUT / 'phase.json').write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps(value), flush=True)


@ray.remote
class CollectionGradientWorker(ActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def observe_collection_minibatch(self, data):
        from observe_native_optimizer_minibatch import _temporary_attribute
        assert identity(type(self.actor))['sha256'] == ACTOR
        before = {}
        for name, parameter in self.actor_module_fsdp.named_parameters():
            if parameter.requires_grad:
                local = parameter.detach()
                local = local.to_local() if hasattr(local, 'to_local') else local
                before[name] = local.cpu().clone()
        assert before and not any(torch.count_nonzero(v).item()
                                  for k, v in before.items() if '.lora_B.' in k)
        minibatch = data.meta_info['collection_minibatch']
        path = OUT / f'minibatch{minibatch}-rank{self.rank}.json'
        scheduler_calls = []

        def observe_policy(*, data):
            vectors = {LABELS[0]: data.batch['advantages'],
                       LABELS[1]: data.batch['diagnostic_development_tail'],
                       LABELS[2]: data.batch['diagnostic_development_sign_flip']}
            return observer.observe_native_optimizer_minibatch(
                self.actor, data, output_path=path, rank=self.rank,
                policy_advantages=vectors)

        def no_scheduler(*args, **kwargs):
            scheduler_calls.append(True)

        with _temporary_attribute(observer, 'ACTOR_SHA256', ACTOR), \
             _temporary_attribute(self.actor, 'update_policy', observe_policy), \
             _temporary_attribute(self.actor_lr_scheduler, 'step', no_scheduler):
            output = super().update_actor(data)
        comparisons = {}
        for name, parameter in self.actor_module_fsdp.named_parameters():
            if name in before:
                local = parameter.detach()
                local = local.to_local() if hasattr(local, 'to_local') else local
                comparisons[name] = torch.equal(before[name], local.cpu())
        assert all(comparisons.values())
        result = json.loads(path.read_bytes())
        result.update(collection_scope=__doc__, collection_minibatch=minibatch,
                      parameters_exact_unchanged=comparisons,
                      scheduler_step_executed=False,
                      scheduler_noop_calls=len(scheduler_calls),
                      gradient_adapter=identity(self.observe_collection_minibatch))
        path.write_text(json.dumps(result, indent=2) + '\n')
        return output


def prepare():
    torch.set_num_threads(4)
    source = json.loads(SOURCE.read_bytes())
    assert sha(SOURCE) == '2796233e2683f1939896c74b2b578c242dbd7a7f235b9ef61cbedd398f61be52'
    check_imports(source)
    plan = json.loads((OUT / 'gradient-inputs.json').read_bytes())
    from verl.workers.actor.dp_actor import DataParallelPPOActor
    from verl.trainer.ppo.core_algos import compute_policy_loss
    assert identity(DataParallelPPOActor)['sha256'] == ACTOR
    assert identity(compute_policy_loss)['sha256'] == observer.CORE_SHA256
    paths = [CAPTURE / f'rank{i}-pre-update.pt' for i in (0, 1)]
    expected = ['1563ce74f298769893b360398fd1bacf16c376439b1d462e56ef9dc6f905be59',
                'a1970da0bbf462b203314cbf29cc8ecd8c81e526fe1b6ee944978a76bbadd34e']
    assert [sha(p) for p in paths] == expected
    saved = [torch.load(p, map_location='cpu', weights_only=False) for p in paths]
    tensors = {k: torch.cat([s['tensors'][k] for s in saved])
               for k in saved[0]['tensors']}
    tensors['position_ids'] = compute_position_id_with_mask(tensors['attention_mask'])
    uids = [str(u) for s in saved for u in s['non_tensors']['traj_uid']]
    tail = torch.zeros_like(tensors['advantages'])
    sign_flip = torch.zeros_like(tail)
    mapped = []
    native_cache = {}
    for point in plan['tail_points']:
        key = point['native']['path']
        if key not in native_cache:
            assert sha(key) == point['native']['sha256']
            native_cache[key] = torch.load(key, map_location='cpu', weights_only=False)
        native = native_cache[key]
        row = next(r for r in native['rows'] if r['batch_row'] == point['batch_row'])
        assert str(row['traj_uid']) == point['traj_uid']
        source_suffix_slots = row['prior'][row['suffix_positions']].nonzero().flatten()
        suffix_slot = int(source_suffix_slots[point['source_index']])
        assert point['packed_slot'] == row['prompt_length'] + suffix_slot
        assert float(native['native_signed'][point['batch_row'], point['packed_slot']]) == point['d']
        response_position = int(row['suffix_positions'][suffix_slot])
        matches = []
        for global_row, uid in enumerate(uids):
            if uid != point['traj_uid']:
                continue
            rank, local_row = divmod(global_row, 128)
            artifact = saved[rank]['non_tensors']['dt_direct_target_artifact'][local_row]
            assert artifact['response_ids'][response_position] == point['token_id']
            retained = torch.as_tensor(artifact['retained_response_positions'])
            for slot in (retained == response_position).nonzero().flatten().tolist():
                assert tensors['responses'][global_row, slot] == point['token_id']
                assert tensors['response_mask'][global_row, slot]
                assert not artifact['target_mask'][response_position]
                assert artifact['policy_mask'][response_position]
                assert not tail[global_row, slot], 'Duplicate point in frozen tail census'
                tail[global_row, slot] = tensors['advantages'][global_row, slot]
                flip = point['native_single_d'] >= 0
                if flip:
                    sign_flip[global_row, slot] = tensors['advantages'][global_row, slot]
                matches.append(dict(rank=rank, local_row=local_row, global_row=global_row,
                                    optimizer_minibatch=local_row // 32, actor_slot=slot,
                                    actual_raw_A=float(tensors['dt_token_advantages'][global_row, slot]),
                                    saved_whitened_A=float(tail[global_row, slot])))
        assert matches, 'Frozen tail point did not bind the actual owner actor artifact'
        mapped.append(dict(**point, original_response_position=response_position, actor_matches=matches))
    tensors['diagnostic_development_tail'] = tail
    tensors['diagnostic_development_sign_flip'] = sign_flip
    cfg = OmegaConf.load(Path(source['verl_root']) / 'verl/trainer/config/ppo_trainer.yaml')
    for key, value in source['startup_options'].items():
        OmegaConf.update(cfg, key.lstrip('+'), value, force_add=True)
    cfg.actor_rollout_ref.actor.optim.total_training_steps = actor_initialization_steps(
        cfg.trainer.total_training_steps, 330)
    data = DataProto.from_dict(tensors=tensors, non_tensors={'traj_uid': uids},
        meta_info={'temperature': cfg.actor_rollout_ref.rollout.temperature,
                   'multi_turn': True,
                   'global_token_num': tensors['attention_mask'].sum(-1).tolist()})
    inspection = dict(scope=__doc__, source_sha256=sha(SOURCE),
        input_plan_sha256=sha(OUT / 'gradient-inputs.json'),
        inputs=[dict(path=str(p), sha256=sha(p)) for p in paths], rows=len(data),
        original_minibatches=4, local_rows_per_minibatch=32, actual_microbatch=4,
        labels=list(LABELS), tail_points=len(mapped),
        native_sign_flip_points=sum(p['native_single_d'] >= 0 for p in mapped),
        mapping=mapped, actor=identity(DataParallelPPOActor), core=identity(compute_policy_loss),
        observer=identity(observer.observe_native_optimizer_minibatch),
        position_owner=identity(compute_position_id_with_mask),
        heldout_scope='Full original coefficients are retained for the real loss denominator/baseline; no test split is queried, scored, selected or used to choose a candidate.',
        whitening='Exact saved whole-batch coefficients. No subset whitening or native-credit replacement.',
        cuda_initialized=torch.cuda.is_initialized(), optimizer_steps=0, scheduler_steps=0,
        DT=0, rollout=0, checkpoint_restore=0)
    (OUT / 'input-inspection.json').write_text(json.dumps(inspection, indent=2) + '\n')
    (OUT / 'effective-config.yaml').write_text(OmegaConf.to_yaml(cfg))
    return cfg, data


def main():
    cfg, data = prepare()
    if '--inspect-only' in sys.argv:
        print(json.dumps(dict(scope=__doc__, rows=len(data), status='input-mapping-only')))
        return
    started = time.monotonic()
    ray.init(num_cpus=8, include_dashboard=False)
    try:
        phase('native_actor_init_begin')
        group = RayWorkerGroup(RayResourcePool([2], use_gpu=True, max_colocate_count=1),
            RayClassWithInitArgs(CollectionGradientWorker, cfg.actor_rollout_ref, 'actor'))
        group.init_model()
        phase('native_actor_init_complete')
        phase('original_old_log_prob_begin')
        data.union(group.compute_log_prob(data.select(deepcopy=True)))
        phase('original_old_log_prob_complete')
        phase('original_ref_log_prob_begin')
        data.union(group.compute_ref_log_prob(data.select(deepcopy=True)))
        phase('original_ref_log_prob_complete')
        for minibatch in range(4):
            # Retain the exact original local row order on both DP ranks.
            indices = torch.cat([torch.arange(rank * 128 + minibatch * 32,
                                             rank * 128 + (minibatch + 1) * 32)
                                 for rank in (0, 1)])
            mini = data[indices]
            mini.meta_info['collection_minibatch'] = minibatch
            mini.meta_info['global_token_num'] = mini.batch['attention_mask'].sum(-1).tolist()
            phase('original_minibatch_gradients_begin', minibatch=minibatch,
                  elapsed_seconds=time.monotonic() - started)
            group.observe_collection_minibatch(mini)
            phase('original_minibatch_gradients_complete', minibatch=minibatch,
                  elapsed_seconds=time.monotonic() - started)
            if time.monotonic() - started > 1800:
                raise TimeoutError('Bounded collection gradient diagnostic exceeded 1800 seconds; no training released')
        (OUT / 'completed.json').write_text(json.dumps(dict(
            completed_unix=time.time(), elapsed_seconds=time.monotonic() - started,
            source_sha256=sha(SOURCE), rows=256, original_minibatches=4,
            native_backward_passes_per_rank=12, optimizer_steps=0, scheduler_steps=0,
            DT=0, rollout=0, checkpoint_restore=0, scope=__doc__), indent=2) + '\n')
    finally:
        ray.shutdown()


if __name__ == '__main__':
    main()

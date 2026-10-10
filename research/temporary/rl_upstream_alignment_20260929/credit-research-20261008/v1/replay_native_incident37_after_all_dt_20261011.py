"""Replay all saved incident37 DT batches then its native optimizer operation in isolation.

Original VERL model initialization, actor forward/loss/update and Torch DCP
state loading own every operation. Only the saved trainable local shards are
loaded into this diagnostic actor; no formal checkpoint/job is restored.
Passive head observation calls the same owner's undecorated function on one
identical original chunk, and returns the original compiled outputs unchanged.
One existing native DT RPC on its complete saved nonzero-target rows; no production patch, replacement loss, rollout or training budget.
"""
import argparse
import functools
import hashlib
import inspect
import json
import os
from pathlib import Path
import time

import torch

SOURCE_SHA = '1c08b57bf83506d3e69d865f74377e3baa624358c678e0ac92cb6ebfb2d73658'
ACTOR_SHA = '3a65e173300be82a7a9e056a96227c4f746eabc3be778ef41c8d50138d52ce6c'
HEAD_SHA = 'e285c3353bddbffae38df346b44014ee5038b606531874ddbeb10ee1241f77de'
INCIDENT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/textcraft-native-actor-incidents-20261010-v1')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_worker():
    import ray
    from verl.single_controller.base.decorator import Dispatch, register
    from verl.workers.fsdp_workers import ActorRolloutRefWorker

    @ray.remote
    class IncidentWorker(ActorRolloutRefWorker):
        @register(dispatch_mode=Dispatch.ONE_TO_ALL)
        def replay_incident(self, output):
            import psutil
            from torch.distributed.checkpoint.state_dict import (
                StateDictOptions, get_model_state_dict, set_model_state_dict)
            from torch.distributed.tensor import DTensor
            from verl import DataProto
            from verl.trainer.ppo import core_algos
            from verl.utils.fsdp_utils import load_fsdp_model_to_gpu
            from verl.utils.experimental import torch_functional as head
            from capture_native_loss_backward_20261010 import observe_call

            out = Path(output)
            events = (out/f'rank{self.rank}-phases.jsonl').open('a', buffering=1)
            record = dict(rank=self.rank, pid=os.getpid(), birth=psutil.Process().create_time(),
                          scope=__doc__, head_observations=[], microbatches=[],
                          actual_optimizer_steps=0, model_forward_calls=0)

            def save(phase, **extra):
                row = dict(phase=phase, unix=time.time(),
                           allocated=torch.cuda.memory_allocated(),
                           reserved=torch.cuda.memory_reserved(),
                           pss_bytes=psutil.Process().memory_full_info().pss, **extra)
                events.write(json.dumps(row, allow_nan=True)+'\n')
                record.update(row)
                (out/f'rank{self.rank}.json').write_text(json.dumps(record, indent=2, allow_nan=True))

            actor = self.actor
            owner = inspect.getmodule(type(actor))
            assert sha(inspect.getsourcefile(owner)) == ACTOR_SHA
            assert sha(inspect.getsourcefile(head)) == HEAD_SHA
            assert sha(inspect.getsourcefile(core_algos)) == 'fc2f992b16fb7fb23aebc683ad5f00136cf61fc9013cd426f5046983babe7299'
            assert owner.compute_policy_loss is core_algos.compute_policy_loss
            assert self._is_actor and not self._is_rollout
            assert (self.config.model.lora_rank, self.config.model.lora_alpha) == (8, 16)
            assert actor.config.ppo_micro_batch_size_per_gpu == 4
            assert actor.config.ppo_mini_batch_size == 32
            old_pid = [987808, 989860][self.rank]
            folder = INCIDENT/f'rank{self.rank}-pid{old_pid}'
            snapshot_path = next(folder.glob('incident-*-update.pt'))
            snapshot = torch.load(snapshot_path, map_location='cpu', weights_only=False)
            assert snapshot['update_index'] == 20
            record['input_artifact'] = dict(path=str(snapshot_path), sha256=sha(snapshot_path))
            record['sources'] = {str(p): sha(p) for p in [Path(inspect.getsourcefile(owner)),
                                 Path(inspect.getsourcefile(head)), Path(__file__)]}
            save('native_actor_initialized')
            if self._is_offload_param:
                load_fsdp_model_to_gpu(self.actor_module_fsdp)
            options = StateDictOptions(ignore_frozen_params=True, strict=False)
            state = get_model_state_dict(self.actor_module_fsdp, options=options)
            captured = snapshot['trainable_local_parameters']
            assert set(state) == set(captured), (set(state)-set(captured), set(captured)-set(state))
            restored = {}
            for name, prototype in state.items():
                assert isinstance(prototype, DTensor)
                value = captured[name]
                assert value.shape == prototype.to_local().shape and value.dtype == prototype.dtype
                restored[name] = DTensor.from_local(value.to(prototype.to_local().device),
                    device_mesh=prototype.device_mesh, placements=prototype.placements,
                    shape=prototype.shape, stride=prototype.stride(), run_check=False)
            incompatible = set_model_state_dict(self.actor_module_fsdp, restored, options=options)
            check = get_model_state_dict(self.actor_module_fsdp, options=options)
            assert all(torch.equal(value.to_local().cpu(), captured[name]) for name, value in check.items())
            record['trainable_shards_loaded_exactly'] = len(check)
            record['DCP_missing_frozen_keys'] = len(incompatible.missing_keys)
            del state, restored, check, captured
            torch.set_rng_state(snapshot['torch_rng'])
            torch.cuda.set_rng_state(snapshot['cuda_rng'])
            save('captured_trainable_state_loaded_via_Torch_DCP')

            # Exercise one actual saved DT batch via the existing worker RPC.
            # Only undo the recorder's unpadding; all valid IDs/masks are exact.
            import numpy as np
            from reward_readout import DirectActionTargetReadout
            formal = INCIDENT.parents[1]/'runs/textcraft-formal-stable-20261009-v1'
            packets = sorted([p for p in (formal/f'credit-records/rank{self.rank}-pid{old_pid}').glob('joint-*.pt')
                              if 1791646740 < p.stat().st_mtime < 1791646915], key=lambda p:p.stat().st_mtime)
            packet_path = packets[0]
            packet_list = [torch.load(p,map_location='cpu',weights_only=False) for p in packets]
            packet = dict(packet_list[0]);packet['rows']=[row for p in packet_list for row in p['rows']]
            assert len(packets)==27
            assert len({r['policy_mask'].numel() for r in packet['rows']})==1
            assert len({r['prompt_length'] for r in packet['rows']})==1
            tensors = {k:[] for k in ['input_ids','attention_mask','responses','policy_mask','target_mask','dt_direct_reward']}
            for index,row in enumerate(packet['rows']):
                length = row['prompt_length']; width = row['policy_mask'].numel()
                ids = torch.full((length+width,),packet['eos_token_id'],dtype=torch.long)
                ids[:length] = row['input_ids'][:length]
                ids[length+row['suffix_positions']] = row['input_ids'][length:]
                attention = torch.zeros_like(ids);attention[:length]=1;attention[length+row['suffix_positions']]=1
                item=dict(input_ids=ids,attention_mask=attention,responses=ids[length:],
                          policy_mask=row['policy_mask'],target_mask=row['target_mask'],
                          dt_direct_reward=torch.tensor(row['reward']),traj_uid=row['traj_uid'])
                check_row = DirectActionTargetReadout._prepare_row(item,index)
                assert torch.equal(check_row['selected'],row['input_ids'])
                assert torch.equal(check_row['suffix_positions'],row['suffix_positions'])
                assert torch.equal(check_row['prior'],row['prior_source_mask'])
                assert check_row['target_offsets'] == row['target_offsets']
                for key in tensors:tensors[key].append(item[key])
            dt_data = DataProto.from_dict(tensors={k:torch.stack(v) for k,v in tensors.items()},
                non_tensors={'traj_uid':np.array([r['traj_uid'] for r in packet['rows']])},
                meta_info=dict(eos_token_id=packet['eos_token_id'],pad_token_id=self.tokenizer.pad_token_id,
                               dt_target_semantics='native_joint_action_target'))
            before_training = self.actor_module_fsdp.training
            parameter_versions = {n:(p._version,str(p.dtype),tuple(p.shape)) for n,p in self.actor_module_fsdp.named_parameters()}
            save('native_saved_DT_batch_begin',packet=str(packet_path),packet_sha256=sha(packet_path))
            dt_result = self.compute_dt_token_advantages(dt_data)
            record['DT_batch'] = dict(packet=str(packet_path),packet_sha256=sha(packet_path),
                original_valid_IDs_and_masks_exact=True, trajectories=len(packet['rows']),
                unique_trajectory_ids=len({r['traj_uid'] for r in packet['rows']}),
                all_input_packets=[dict(path=str(p),sha256=sha(p)) for p in packets],
                result_finite={k:bool(torch.isfinite(v).all()) for k,v in dt_result.batch.items()},
                model_training_before=before_training,model_training_after=self.actor_module_fsdp.training,
                parameter_metadata_changes=[n for n,p in self.actor_module_fsdp.named_parameters()
                    if parameter_versions[n] != (p._version,str(p.dtype),tuple(p.shape))])
            import qwen35_gdn_finite
            record['DT_batch']['actual_GDN_source']=dict(path=inspect.getsourcefile(qwen35_gdn_finite),sha256=sha(inspect.getsourcefile(qwen35_gdn_finite)))
            assert record['DT_batch']['actual_GDN_source']['sha256']=='7c06d5e0a4d6c00c13483dadd27d6389e25eee7868a05666d2d1faeaa2d62656'
            save('native_saved_DT_batch_completed',DT_batch=record['DT_batch'])
            del dt_result,dt_data,packet,packet_list,tensors

            # Observe the actual native head, retaining its original return object.
            native_head = head.FusedLinearForPPO.forward
            eager_chunk = inspect.unwrap(head._fused_linear_for_ppo_fwd)
            head_calls = 0

            @functools.wraps(native_head)
            def observed_head(module, hidden_states, vocab_weights, input_ids, temperature=1.0):
                nonlocal head_calls
                result = native_head(module, hidden_states, vocab_weights, input_ids, temperature)
                index = head_calls
                head_calls += 1
                record['model_forward_calls'] += 1
                if index < 2:
                    try:
                        # Same shape, dtype, rows, labels and 512-token chunk as
                        # the compiled owner call; not a new scoring function.
                        position = int(result[0].detach().flatten().argmin())
                        begin = position//module.chunk_size*module.chunk_size
                        end = min(begin+module.chunk_size, hidden_states.numel()//hidden_states.shape[-1])
                        h = hidden_states.flatten(0, 1)[begin:end]
                        ids = input_ids.flatten()[begin:end].to(torch.int64)
                        with torch.no_grad():
                            expected, entropy = eager_chunk(h, vocab_weights, ids, temperature)
                        actual = result[0].detach().flatten()[begin:end]
                        point = dict(head_call=index, chunk_begin=begin, chunk_end=end,
                                     hidden_dtype=str(h.dtype), weight_dtype=str(vocab_weights.dtype),
                                     compiled_min=float(actual.min()), eager_min=float(expected.min()),
                                     max_abs_difference=float((actual-expected).abs().max()))
                        path = out/f'rank{self.rank}-head{index}.pt'
                        torch.save(dict(point=point, hidden_states=h.detach().cpu(),
                                        labels=ids.cpu(), compiled_logp=actual.cpu(),
                                        eager_logp=expected.detach().cpu()), path)
                        point.update(path=str(path), sha256=sha(path))
                        record['head_observations'].append(point)
                        save('native_head_chunk_compared', head=point)
                        del expected, entropy
                    except Exception as error:
                        record['head_observations'].append(dict(head_call=index,error=repr(error)))
                        save('optional_head_observer_error', error=repr(error))
                return result

            def observe_policy(result, arguments):
                index = len(record['microbatches'])
                x = arguments['log_prob']
                old = arguments['old_log_prob']
                mask = arguments['response_mask'].bool()
                gap = (x-old).detach()
                point = dict(index=index, finite_logp=bool(torch.isfinite(x).all()),
                             min_active_gap=float(gap[mask].min()), max_active_gap=float(gap[mask].max()),
                             native_pg_loss=float(result[0]))
                record['microbatches'].append(point)
                if index < 2:
                    path = out/f'rank{self.rank}-microbatch{index}.pt'
                    torch.save({k:arguments[k].detach().cpu() for k in
                                ['log_prob','old_log_prob','advantages','response_mask']},path)
                    point.update(path=str(path),sha256=sha(path))
                def grad_observer(g):
                    point['nonfinite_logp_gradient_count'] = int((~torch.isfinite(g)).sum())
                    save('native_logp_backward_observed', microbatch=index,
                         nonfinite=point['nonfinite_logp_gradient_count'])
                    return g
                x.register_hook(grad_observer)
                save('native_policy_loss_observed',microbatch=index)

            native_policy = owner.compute_policy_loss
            handle = actor.actor_optimizer.register_step_post_hook(
                lambda *_: record.update(actual_optimizer_steps=record['actual_optimizer_steps']+1))
            head.FusedLinearForPPO.forward = observed_head
            owner.compute_policy_loss = observe_call(native_policy, lambda: True, observe_policy,
                lambda error: save('optional_policy_observer_error',error=repr(error)))
            try:
                # One original 32-row mini = eight original B4 microbatches.
                # Preserve saved masks/old logp/advantages and native loss/step.
                data = DataProto(batch=snapshot['input_batch'][:32],
                                 meta_info=dict(snapshot['input_meta_info']))
                save('native_update_policy_begin')
                result = self.update_actor(data)
                record['native_metrics'] = result.meta_info['metrics']
                record['optimizer_state_scope'] = 'Fresh diagnostic optimizer; incident trainable weights and loss inputs exact; no claim about matching a parameter update.'
                save('complete', peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(),
                     peak_torch_reserved_bytes=torch.cuda.max_memory_reserved())
                return dict(rank=self.rank,completed=True,actual_optimizer_steps=record['actual_optimizer_steps'])
            except BaseException:
                import traceback
                save('failed',traceback=traceback.format_exc())
                raise
            finally:
                owner.compute_policy_loss = native_policy
                head.FusedLinearForPPO.forward = native_head
                handle.remove()

    return IncidentWorker


def main():
    import ray
    from omegaconf import OmegaConf
    from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    assert sha(args.source)==SOURCE_SHA
    source=json.loads(args.source.read_bytes())
    args.output.mkdir(parents=True,exist_ok=False)
    cfg=OmegaConf.load(Path(source['verl_root'])/'verl/trainer/config/ppo_trainer.yaml')
    for key,value in source['startup_options'].items():
        OmegaConf.update(cfg,key.lstrip('+'),value,force_add=True)
    cfg.actor_rollout_ref.actor.optim.total_training_steps=330
    (args.output/'initialization-config.yaml').write_text(OmegaConf.to_yaml(cfg))
    ray.init(num_cpus=8,include_dashboard=False)
    try:
        group=RayWorkerGroup(RayResourcePool([2],use_gpu=True,max_colocate_count=1),
            RayClassWithInitArgs(make_worker(),cfg.actor_rollout_ref,'actor'))
        group.init_model()
        result=group.replay_incident(str(args.output))
        (args.output/'completed.json').write_text(json.dumps(result,indent=2))
    finally:
        ray.shutdown()


if __name__=='__main__':
    main()

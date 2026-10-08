"""Observe one entire original TextCraft iteration without anomaly-mode sync.

The original TaskRunner, environment, collector, DT, trainer and actor run.
Only diagnostic files/directories and GPU assignment differ. The configured
training budget/scheduler stay unchanged; observation stops after step1 logs.
No policy, optimizer, gradient, precision, mask or credit is replaced.
"""
from contextlib import ExitStack
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import sys
import time
import traceback
from unittest.mock import patch

import psutil
import ray
import torch
from omegaconf import OmegaConf
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
import verl.trainer.main_ppo as native_main

OUT = Path(os.environ['DT_ACTOR_INCIDENT_OUT'])
ROOT = Path(os.environ['DT_RUNTIME_ROOT'])
SOURCE = ROOT / 'runs/direct-target-prefix-runtime-20261007-v1/textcraft/textcraft-dt/source.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def local(value):
    value = value.detach()
    return value.to_local() if hasattr(value, 'to_local') else value


def snapshot(worker):
    return dict(parameters={name: local(p).cpu().clone()
        for name, p in worker.actor_module_fsdp.named_parameters() if p.requires_grad},
        cpu_rng=torch.get_rng_state(), device_rng=torch.cuda.get_rng_state(),
        purpose='Incident tensors only; not a formal checkpoint or training restore.')


class WorkerObservation:
    def update_actor(self, data):
        native_update = self._incident_original_update
        owner = importlib.import_module(type(self.actor).__module__)
        assert sha(inspect.getsourcefile(type(self.actor))) == '3a65e173300be82a7a9e056a96227c4f746eabc3be778ef41c8d50138d52ce6c'
        # Capture the real native dispatch payload, including fields absent
        # from the historical incident. It is already CPU on this RPC entry.
        path = OUT / f'rank{self.rank}-actual-update-input.pt'
        torch.save(dict(batch={k:v.detach().cpu().clone() for k,v in data.batch.items()},
            non_tensor_batch=data.non_tensor_batch, meta_info=data.meta_info), path)
        write(OUT / f'rank{self.rank}-input.json', dict(path=str(path), sha256=sha(path),
            rows=len(data), keys=list(data.batch.keys()), meta_info=data.meta_info,
            unix=time.time(), rank=self.rank))
        forward, policy, penalty, clip = (self.actor._forward_micro_batch,
            owner.compute_policy_loss, owner.kl_penalty, owner.fsdp2_clip_grad_norm_)
        record = dict(rank=self.rank, pid=os.getpid(), birth=psutil.Process().create_time(),
            script_sha256=sha(__file__), microbatches=0, optimizer_steps=[],
            phase='native_update_begin', unix=time.time(), detect_anomaly=False,
            scope=__doc__)
        report_path = OUT / f'rank{self.rank}.json'
        pending = []

        def observed_forward(*args, **kwargs):
            result = forward(*args, **kwargs)
            row = dict(index=record['microbatches'], entropy=None,
                       log_prob=result[1].detach().clone())
            if result[0] is not None:
                row['entropy'] = result[0].detach().clone()
            record['microbatches'] += 1
            pending.append(row)
            # Device clones only: no item(), CPU transfer, finiteness test or
            # anomaly detector between native forward and native backward.
            for label, value in zip(('entropy', 'log_prob'), result):
                if value is not None and value.requires_grad:
                    def observe(gradient, label=label, row=row):
                        row[label + '_gradient'] = gradient.detach().clone()
                        return gradient
                    value.register_hook(observe)
            return result

        def observed_policy(*args, **kwargs):
            result = policy(*args, **kwargs)
            pending[-1]['policy_outputs'] = [v.detach().clone() for v in result]
            return result

        def observed_penalty(*args, **kwargs):
            result = penalty(*args, **kwargs)
            pending[-1]['kl_penalty'] = result.detach().clone()
            return result

        def observed_clip(parameters, *args, **kwargs):
            parameters = list(parameters)
            # Preserve uncropped, unclipped local gradient values. FSDP2
            # offload completion remains the original owner's responsibility.
            raw = {name: local(p.grad).clone() for name,p in self.actor_module_fsdp.named_parameters()
                   if p.requires_grad and p.grad is not None}
            result = clip(parameters, *args, **kwargs)
            # Original _optimizer_step immediately synchronizes this same
            # norm for its nonfinite decision; no microbatch sync is added.
            norm = float(result)
            index = len(record['optimizer_steps'])
            tensors = dict(raw_gradients={k:v.cpu() for k,v in raw.items()},
                           microbatches=[{k:([x.cpu() for x in v] if isinstance(v,list)
                               else v.cpu() if isinstance(v,torch.Tensor) else v)
                               for k,v in row.items()} for row in pending])
            tensor_path = OUT / f'rank{self.rank}-step{index}-before-native-step.pt'
            torch.save(tensors, tensor_path)
            bad = [{"name":name,"nonfinite":int((~torch.isfinite(value)).sum())}
                   for name,value in tensors['raw_gradients'].items()
                   if not bool(torch.isfinite(value).all())]
            record['optimizer_steps'].append(dict(index=index,native_grad_norm=norm,
                nonfinite_raw_gradients=bad,microbatches=len(pending),
                path=str(tensor_path),sha256=sha(tensor_path),unix=time.time()))
            record.update(phase='native_clip_complete',unix=time.time())
            write(report_path,record)
            pending.clear()
            return result

        def after_optimizer(*args, **kwargs):
            result = native_optimizer_step(*args, **kwargs)
            index = len(record['optimizer_steps']) - 1
            path = OUT / f'rank{self.rank}-step{index}-after-optimizer-LoRA-and-rng.pt'
            torch.save(snapshot(self), path)
            record['optimizer_steps'][-1].update(optimizer_called=True,
                post_step_path=str(path),post_step_sha256=sha(path))
            write(report_path, record)
            return result

        native_optimizer_step = self.actor_optimizer.step
        write(report_path, record)
        try:
            with ExitStack() as stack:
                stack.enter_context(patch.object(self.actor,'_forward_micro_batch',observed_forward))
                stack.enter_context(patch.object(owner,'compute_policy_loss',observed_policy))
                stack.enter_context(patch.object(owner,'kl_penalty',observed_penalty))
                stack.enter_context(patch.object(owner,'fsdp2_clip_grad_norm_',observed_clip))
                stack.enter_context(patch.object(self.actor_optimizer,'step',after_optimizer))
                output = native_update(data)
            record.update(phase='complete',metrics=output.meta_info['metrics'],unix=time.time())
            write(report_path,record)
            return output
        except BaseException as error:
            record.update(phase='failed',error=repr(error),traceback=traceback.format_exc(),unix=time.time())
            write(report_path,record)
            raise


def install_native_observer(worker):
    """Use native WorkerDict's documented generic execution RPC after init."""
    from types import MethodType
    installed=[]
    for owner in worker.worker_dict.values():
        if not getattr(owner,'_is_actor',False):
            continue
        path=OUT/f'rank{owner.rank}-initial-LoRA-and-rng.pt'
        torch.save(snapshot(owner),path)
        record=dict(pid=os.getpid(),birth=psutil.Process().create_time(),unix=time.time(),
            rank=owner.rank,path=str(path),sha256=sha(path),
            native_worker_module=type(owner).__module__,native_worker_name=type(owner).__qualname__)
        write(OUT/f'rank{owner.rank}-initial.json',record)
        owner._incident_original_update=owner.update_actor
        owner.update_actor=MethodType(WorkerObservation.update_actor,owner)
        installed.append(record)
    return installed


class FirstIterationObserved(Exception):
    pass


class ObservedTrainer(RayPPOTrainer):
    def init_workers(self):
        result=super().init_workers()
        installed=self.actor_rollout_wg.execute_all_sync(
            'execute_with_func_generator',func=install_native_observer)
        write(OUT/'observer-installation.json',dict(unix=time.time(),workers=installed))
        return result

    def fit(self):
        from verl.utils.tracking import Tracking
        original = Tracking.log

        def logged(instance, *args, **kwargs):
            result = original(instance,*args,**kwargs)
            if kwargs.get('step') == 1:
                write(OUT/'first-iteration-metrics.json',dict(unix=time.time(),**kwargs))
                raise FirstIterationObserved('Diagnostic stop after original first iteration metrics')
            return result

        with patch.object(Tracking,'log',logged):
            return super().fit()


@ray.remote(num_cpus=1)
class ObservedTaskRunner(native_main.TaskRunner.__ray_metadata__.modified_class):
    def run(self,config):
        with patch.object(native_main,'RayPPOTrainer',ObservedTrainer):
            try:
                return super().run(config)
            except FirstIterationObserved:
                write(OUT/'complete.json',dict(unix=time.time(),status='original_first_iteration_observed',
                    root_cause_located=False,repair_deployed=False,formal_training_started=False))


def main():
    assert sha(SOURCE) == '2796233e2683f1939896c74b2b578c242dbd7a7f235b9ef61cbedd398f61be52'
    source=json.loads(SOURCE.read_bytes())
    config=OmegaConf.load(Path(source['verl_root'])/'verl/trainer/config/ppo_trainer.yaml')
    for key,value in source['startup_options'].items():
        OmegaConf.update(config,key.lstrip('+'),value,force_add=True)
    for key in ('default_local_dir','rollout_data_dir','validation_data_dir'):
        OmegaConf.update(config,'trainer.'+key,str(OUT/key))
    (OUT/'effective-config.yaml').write_text(OmegaConf.to_yaml(config))
    if '--inspect-only' in sys.argv:
        from ray import cloudpickle
        # Native worker stays registered at its original module, so Ray does
        # not serialize a replacement class into its zero-argument super cell.
        assert cloudpickle.loads(cloudpickle.dumps(ActorRolloutRefWorker)) is ActorRolloutRefWorker
        write(OUT/'CPU-inspection.json',dict(CUDA_initialized=torch.cuda.is_initialized(),
            model_path=config.actor_rollout_ref.model.path,
            lora_rank=config.actor_rollout_ref.model.lora_rank,
            lora_alpha=config.actor_rollout_ref.model.lora_alpha,
            microbatch=config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
            global_optimizer_minibatch=config.actor_rollout_ref.actor.ppo_mini_batch_size,
            multi_turn=config.actor_rollout_ref.rollout.multi_turn.enable,
            epochs=config.trainer.total_epochs,total_training_steps=config.trainer.total_training_steps,
            resume_mode=config.trainer.resume_mode,script_sha256=sha(__file__)))
        return
    write(OUT/'driver-state.json',dict(phase='native_run_ppo_begin',unix=time.time(),
        pid=os.getpid(),birth=psutil.Process().create_time(),script_sha256=sha(__file__),
        source_sha256=sha(SOURCE),configured_budget_unchanged=True,diagnostic_only=True))
    try:
        with patch.object(native_main,'TaskRunner',ObservedTaskRunner):
            native_main.run_ppo(config)
    except BaseException as error:
        write(OUT/'driver-failed.json',dict(unix=time.time(),error=repr(error),traceback=traceback.format_exc()))
        raise
    finally:
        ray.shutdown()


if __name__ == '__main__':
    main()

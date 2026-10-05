"""Isolated complete-minibatch observation through the original VERL entry.

The original Hydra main/run_ppo, TaskRunner.run, trainer.fit, collector, DT,
reward handling and worker.update_actor own the execution. The first training
collector input is sliced with the original DataProto API to eight task rows;
the original TextCraft n=8 rollout returns one global64/local32 minibatch.
At the original update boundary, the worker observes four original loss passes
with optimizer and scheduler steps disabled, then fit returns immediately.
This file is a research entry, never a production patch or a training loop.
"""

from __future__ import annotations

import argparse
import ast
from contextlib import ExitStack, contextmanager
import dis
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import textwrap
import time
from unittest.mock import patch

import ray
from omegaconf import OmegaConf

from verl.single_controller.base.decorator import Dispatch, register
from verl.trainer import main_ppo
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, Role
from verl.workers import fsdp_workers

from observe_native_optimizer_minibatch import observe_native_optimizer_minibatch
from observe_textcraft_native_batches import OUTPUT_DIRECTORY_ENV, _snapshot, observe


OUT = Path(os.environ["DT_TEXTCRAFT_MINIBATCH_ROOT"])
OriginalTrainer = RayPPOTrainer
OriginalWorker = fsdp_workers.ActorRolloutRefWorker
OriginalTaskRunner = main_ppo.TaskRunner.__ray_metadata__.modified_class
OriginalRunPPO = main_ppo.run_ppo


def _identity(value):
    value = inspect.unwrap(value)
    filename = inspect.getsourcefile(value)
    path = Path(filename).resolve() if filename else None
    result = {"path": str(path) if path else None,
              "sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path else None,
              "signature": str(inspect.signature(value))}
    try:
        _, result["source_line"] = inspect.getsourcelines(value)
    except (OSError, TypeError):
        result["source_line"] = None
    return result


def _write(name, value):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _phase(label, *, rank=None, **details):
    value = dict(phase=label, rank=rank, pid=os.getpid(), unix=time.time(), **details)
    _write(f"native-minibatch-phase-{'driver' if rank is None else 'rank'+str(rank)}.json", value)
    print("TEXTCRAFT_NATIVE_MINIBATCH " + json.dumps(value), flush=True)


def _sources():
    from agent_system.multi_turn_rollout import TrajectoryCollector
    from verl.protocol import DataProto
    from verl.workers.actor.dp_actor import DataParallelPPOActor
    return {
        "run_ppo": _identity(OriginalRunPPO),
        "task_runner_run": _identity(OriginalTaskRunner.run),
        "trainer_fit": _identity(OriginalTrainer.fit),
        "trainer_constructor": _identity(OriginalTrainer.__init__),
        "collector_multi_turn_loop": _identity(TrajectoryCollector.multi_turn_loop),
        "worker_update_actor": _identity(OriginalWorker.update_actor),
        "worker_constructor": _identity(OriginalWorker.__init__),
        "actor_update_policy": _identity(DataParallelPPOActor.update_policy),
        "data_proto_slice": _identity(DataProto.slice),
        "grpo_outcome_advantage": _identity(core_algos.compute_grpo_outcome_advantage),
        "gradient_observer": _identity(observe_native_optimizer_minibatch),
        "collect_observer": _identity(observe),
        "harness": _identity(_sources),
    }


def _raw_function_identity(function):
    """Inspect the wrapper's actual code, without automatic unwrapping."""
    code = function.__code__
    path = Path(code.co_filename).resolve()
    return {
        "module": function.__module__, "qualname": function.__qualname__,
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_line": code.co_firstlineno,
        "signature": str(inspect.signature(function, follow_wrapped=False)),
    }


def _owner_run_functions(method):
    wrapper = getattr(method, "__func__", method)
    body = inspect.unwrap(wrapper)
    chain = []
    current = wrapper
    while True:
        next_wrapped = getattr(current, "__wrapped__", None)
        references = []
        for name, cell in zip(current.__code__.co_freevars, current.__closure__ or ()):
            value = cell.cell_contents
            if inspect.isfunction(value):
                references.append({
                    "cell_name": name,
                    "is_unwrapped_body": value is body,
                    "is_next_wrapped_function": value is next_wrapped,
                    "function": _raw_function_identity(value),
                })
        chain.append({
            "function": _raw_function_identity(current),
            "is_unwrapped_body": current is body,
            "closure_function_references": references,
        })
        if current is body:
            break
        current = next_wrapped
    return wrapper, body, {
        "call_wrapper": _raw_function_identity(wrapper),
        "unwrapped_body": _raw_function_identity(body),
        "wrapper_is_body": wrapper is body,
        "wrapped_chain": chain,
        "wrapper_closure_references_unwrapped_body": any(
            reference["is_unwrapped_body"]
            for item in chain[:-1] for reference in item["closure_function_references"]),
    }


def _owner_run_global_loads(method=None):
    method = OriginalTaskRunner.run if method is None else getattr(method, "__func__", method)
    _, body, function_receipt = _owner_run_functions(method)
    tree = ast.parse(textwrap.dedent(inspect.getsource(body)))
    return {
        "source": _identity(body),
        "actual_functions": function_receipt,
        "ast_global_name_loads": [node.lineno for node in ast.walk(tree)
                                  if isinstance(node, ast.Name)
                                  and isinstance(node.ctx, ast.Load)
                                  and node.id == "RayPPOTrainer"],
        "actual_bytecode_global_loads": [{"offset": item.offset,
                                          "opname": item.opname,
                                          "argval": item.argval}
                                         for item in dis.get_instructions(body)
                                         if item.opname == "LOAD_GLOBAL"
                                         and item.argval == "RayPPOTrainer"],
        "body_globals_is_main_module_dict": body.__globals__ is main_ppo.__dict__,
        "wrapper_globals_is_main_module_dict": method.__globals__ is main_ppo.__dict__,
    }


@contextmanager
def _owner_run_binding(method):
    # Ray cloudpickle can give the original method a globals dict distinct from
    # the imported module. Bind the exact dict used by super().run's bytecode.
    method = getattr(method, "__func__", method)
    _, body, function_receipt = _owner_run_functions(method)
    globals_dict = body.__globals__
    previous = globals_dict["RayPPOTrainer"]
    with patch.dict(globals_dict, {"RayPPOTrainer": DiagnosticTrainer}):
        binding = globals_dict["RayPPOTrainer"]
        if binding is not DiagnosticTrainer:
            raise TypeError("The actual original run method must resolve the diagnostic trainer.")
        yield {
            "pid": os.getpid(),
            "owner_run": _owner_run_global_loads(method),
            "actual_functions": function_receipt,
            "actual_method_globals_id": id(globals_dict),
            "main_module_dict_id": id(main_ppo.__dict__),
            "actual_global_binding_is_diagnostic_trainer": True,
            "original_global_binding": _identity(previous),
            "bound_trainer": _identity(binding),
            "constructor_sources": {
                "diagnostic": _identity(binding.__init__),
                "original": _identity(OriginalTrainer.__init__),
            },
            "constructor_global_original_trainer_is_recorded":
                binding.__init__.__globals__["OriginalTrainer"] is OriginalTrainer,
            "constructor_global_diagnostic_worker_is_recorded":
                binding.__init__.__globals__["DiagnosticWorker"] is DiagnosticWorker,
            "worker_module_class_binding_unchanged":
                fsdp_workers.ActorRolloutRefWorker is OriginalWorker,
        }


def _worker_class_pickle_contract():
    """Round-trip the actual class without constructing a worker or a model."""
    from ray import cloudpickle

    constructor = inspect.unwrap(OriginalWorker.__init__)
    cells = dict(zip(constructor.__code__.co_freevars, constructor.__closure__ or ()))
    original_class_cell = cells["__class__"].cell_contents
    restored = cloudpickle.loads(cloudpickle.dumps(DiagnosticWorker))
    restored_constructor = inspect.unwrap(restored.__init__)
    restored_cells = dict(zip(restored_constructor.__code__.co_freevars,
                              restored_constructor.__closure__ or ()))
    restored_class_cell = restored_cells["__class__"].cell_contents
    if (fsdp_workers.ActorRolloutRefWorker is not OriginalWorker
            or original_class_cell is not OriginalWorker
            or restored.__mro__[1] is not OriginalWorker
            or restored_constructor is not constructor
            or restored_class_cell is not OriginalWorker):
        raise TypeError("The real worker class round-trip must preserve the original constructor's __class__ cell and base identity.")
    return {
        "scope": "Real DiagnosticWorker Ray-cloudpickle class round-trip, MRO and inherited constructor closure inspection only; no constructor/model execution.",
        "module_worker_binding_unchanged": True,
        "restored_direct_base_is_original": True,
        "restored_constructor_is_original": True,
        "restored_constructor_class_cell_is_original": True,
        "original_mro": [f"{cls.__module__}.{cls.__qualname__}" for cls in DiagnosticWorker.__mro__],
        "restored_mro": [f"{cls.__module__}.{cls.__qualname__}" for cls in restored.__mro__],
        "constructor": _identity(constructor),
        "cloudpickle": _identity(cloudpickle.dumps),
    }


def _inspect(config):
    """Original Hydra composition only: no Ray init, worker or model creation."""
    launch_path = OUT / "launch.json"
    launch = json.loads(launch_path.read_bytes())
    result = {
        "scope": "CPU source/config/interface inspection only; no model, environment episode, DT, or gradient execution.",
        "sources": _sources(),
        "launch": {"path": str(launch_path),
                   "sha256": hashlib.sha256(launch_path.read_bytes()).hexdigest(),
                   "options": launch["options"]},
        "effective_config": OmegaConf.to_container(config, resolve=True),
        "worker_class_pickle_contract": _worker_class_pickle_contract(),
        "owner_run_global_loads": _owner_run_global_loads(),
        "seams": {
            "task_runner": "Original RayActorClass modified_class.run through its unchanged bound Ray wrapper; temporary binding of inspect.unwrap(wrapper).__globals__['RayPPOTrainer'] only, without replacing the module alias.",
            "worker_role": "Original trainer constructor bound arguments; copy role_worker_mapping and replace only Role.ActorRollout with ray.remote(DiagnosticWorker), leaving the fsdp_workers module class unchanged.",
            "trainer": "Original fit via super().fit; first is_train collector gen_batch[:8], then original update_actor boundary observation and immediate return.",
            "collection": "Original TextCraftOwner returned by env factory; original n=8 runner; no duplicated episode loop.",
            "grpo": "Original compute_grpo_outcome_advantage on global token_level_rewards, original loss_mask tail, uid and traj_uid.",
            "worker": "Original update_actor load/shard/offload; temporary actor.update_policy observer and scheduler.step no-op restored in finally.",
            "snapshots": "Original DataProto save_to_disk on separate official TensorDict clone containers; no token reconstruction.",
        },
        "planned_shapes": {"prompt_rows": 8, "native_rollouts_per_prompt": 8,
                           "global_minibatch_rows": 64, "per_rank_rows": 32,
                           "actor_microbatch_per_gpu": 4},
    }
    receipt = _write("native-minibatch-inspect.json", result)
    print("TEXTCRAFT_NATIVE_MINIBATCH_INSPECT " + json.dumps(receipt), flush=True)


class DiagnosticComplete(Exception):
    """Internal bounded-observation completion, caught inside diagnostic fit."""


class DiagnosticWorker(OriginalWorker):
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def observe_minibatch(self, data):
        rank = int(self.rank)
        output_path = OUT / f"rank{rank}-minibatch-gradients.json"
        _phase("native_update_actor_observation_begin", rank=rank, rows=len(data))

        def observed_update_policy(*, data):
            return observe_native_optimizer_minibatch(
                self.actor, data, output_path=output_path, rank=rank)

        def no_op_scheduler_step(*args, **kwargs):
            _phase("scheduler_step_suppressed", rank=rank)

        # The actual worker's original update_actor still owns model/optimizer
        # loading, Ulysses preprocessing, counters and host offload lifecycle.
        with patch.object(self.actor, "update_policy", observed_update_policy):
            with patch.object(self.actor_lr_scheduler, "step", no_op_scheduler_step):
                output = super().update_actor(data)
        _phase("native_update_actor_observation_complete", rank=rank,
               observation={"path": str(output_path),
                            "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest()})
        return output


class DiagnosticTrainer(OriginalTrainer):
    def __init__(self, *args, **kwargs):
        bound = inspect.signature(OriginalTrainer.__init__).bind(self, *args, **kwargs)
        roles = dict(bound.arguments["role_worker_mapping"])
        roles[Role.ActorRollout] = ray.remote(DiagnosticWorker)
        bound.arguments["role_worker_mapping"] = roles
        OriginalTrainer.__init__(*bound.args, **bound.kwargs)

    def fit(self):
        config = self.config
        if (int(config.actor_rollout_ref.model.lora_rank) != 8
                or int(config.actor_rollout_ref.model.lora_alpha) != 16
                or int(config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu) != 4
                or int(config.actor_rollout_ref.actor.ppo_mini_batch_size) != 64
                or int(config.env.rollout.n) != 8):
            raise ValueError("This diagnostic observes the recorded LoRA8/16, B4, global64, n8 owner configuration.")
        _write("native-minibatch-effective-config.json", OmegaConf.to_container(config, resolve=True))
        _write("native-minibatch-source-identity.json", _sources())
        original_collect = self.traj_collector.multi_turn_loop
        collect_signature = inspect.signature(original_collect)
        collected = False

        def bounded_collect(*args, **kwargs):
            nonlocal collected
            bound = collect_signature.bind(*args, **kwargs)
            bound.apply_defaults()
            first_train = bool(bound.arguments["is_train"]) and not collected
            if first_train:
                original_rows = len(bound.arguments["gen_batch"])
                bound.arguments["gen_batch"] = bound.arguments["gen_batch"][:8]
                _phase("native_collect_begin", original_prompt_rows=original_rows,
                       diagnostic_prompt_rows=len(bound.arguments["gen_batch"]),
                       original_global_step=self.global_steps)
            output = original_collect(*bound.args, **bound.kwargs)
            if first_train:
                collected = True
                result = observe(bound.arguments["envs"], output)
                _write("native-minibatch-collect-observation.json", {
                    "scope": "Original native collector output, before trainer adjustment/balance/reward/DT.",
                    "rows": len(output), "observation": result,
                    "owner_class": _identity(type(bound.arguments["envs"])),
                })
                _phase("native_collect_complete", rows=len(output), observation=result)
            return output

        def observe_update_boundary(batch):
            if len(batch) != 64:
                raise ValueError("The native update boundary must contain exactly the requested global64 real trajectories.")
            diagnostic_batch = _snapshot(batch)
            width = batch.batch["response_mask"].size(1)
            # This is the exact native trainer GRPO mask selection for multi_turn.
            grpo_mask = (batch.batch["loss_mask"][:, -width:]
                         if batch.meta_info.get("multi_turn", False)
                         else batch.batch["response_mask"])
            advantages, _ = core_algos.compute_grpo_outcome_advantage(
                token_level_rewards=batch.batch["token_level_rewards"],
                response_mask=grpo_mask,
                index=batch.non_tensor_batch["uid"],
                traj_index=batch.non_tensor_batch["traj_uid"],
                norm_adv_by_std_in_grpo=config.algorithm.get("norm_adv_by_std_in_grpo", True),
            )
            diagnostic_batch.batch["diagnostic_grpo_advantages"] = advantages
            # Saving can rebind the serialized TensorDict. Serialize another
            # official snapshot, preserving the diagnostic RPC carrier itself.
            batch_path = OUT / "native-optimizer-minibatch.pkl"
            _snapshot(diagnostic_batch).save_to_disk(str(batch_path))
            _write("native-minibatch-update-boundary.json", {
                "scope": "Complete original update boundary after balance, reward, old/ref logprob and DT; additional GRPO tensor is diagnostic only.",
                "global_step": self.global_steps, "rows": len(batch),
                "groups": len(set(batch.non_tensor_batch["uid"].tolist())),
                "uid": batch.non_tensor_batch["uid"].tolist(),
                "traj_uid": batch.non_tensor_batch["traj_uid"].tolist(),
                "fields": {key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                           for key, value in diagnostic_batch.batch.items()},
                "snapshot": {"path": str(batch_path),
                             "sha256": hashlib.sha256(batch_path.read_bytes()).hexdigest()},
                "grpo_owner": _identity(core_algos.compute_grpo_outcome_advantage),
            })
            _phase("complete_native_minibatch_saved", rows=len(batch), snapshot=str(batch_path))
            self.actor_rollout_wg.observe_minibatch(diagnostic_batch)
            _phase("complete_native_minibatch_observed", rows=len(batch), optimizer_steps=0,
                   scheduler_steps=0)
            raise DiagnosticComplete()

        try:
            with ExitStack() as stack:
                stack.enter_context(patch.object(self.traj_collector, "multi_turn_loop", bounded_collect))
                stack.enter_context(patch.object(self.actor_rollout_wg, "update_actor", observe_update_boundary))
                stack.enter_context(patch.dict(os.environ, {OUTPUT_DIRECTORY_ENV: str(OUT / "native-collect")}))
                return super().fit()
        except DiagnosticComplete:
            _write("native-minibatch-completed.json", {
                "completed_unix": time.time(), "global_step": self.global_steps,
                "scope": "One real native global64/local32 minibatch, original owner losses observed; zero optimizer/scheduler updates, no formal training restart.",
            })
            return None


@ray.remote(num_cpus=1)
class DiagnosticTaskRunner(OriginalTaskRunner):
    def run(self, config):
        original_run = super().run
        with _owner_run_binding(original_run) as receipt:
            _write("native-minibatch-actual-run-binding.json", receipt)
            _phase("actual_task_runner_globals_bound",
                   actual_global_binding_is_diagnostic_trainer=True)
            return original_run(config)


@ray.remote(num_cpus=1)
class InspectRayTaskRunner(OriginalTaskRunner):
    def run(self, config):
        # Real Ray serialization and process execution, but never call the
        # original run: no tokenizer, environment, model or worker construction.
        with _owner_run_binding(super().run) as receipt:
            receipt.update({
                "scope": "Actual Ray CPU actor process and original run globals binding only; original run is not called, no task/model/DT/optimizer execution.",
                "sources": _sources(),
                "worker_class_pickle_contract": _worker_class_pickle_contract(),
                "effective_config": OmegaConf.to_container(config, resolve=True),
                "original_run_called": False,
            })
            artifact = _write("native-minibatch-ray-inspect.json", receipt)
            _phase("ray_cpu_actual_run_binding_inspected", receipt=artifact)
            return artifact


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    inspection = parser.add_mutually_exclusive_group()
    inspection.add_argument("--inspect-only", action="store_true")
    inspection.add_argument("--inspect-ray-only", action="store_true")
    args, hydra_argv = parser.parse_known_args()
    sys.argv = [sys.argv[0], *hydra_argv]
    if args.inspect_only:
        # Call the owning Hydra entry for its unchanged defaults/override
        # composition; replace only its final run_ppo callback for CPU inspection.
        with patch.object(main_ppo, "run_ppo", _inspect):
            main_ppo.main()
    else:
        runner = InspectRayTaskRunner if args.inspect_ray_only else DiagnosticTaskRunner
        with patch.object(main_ppo, "TaskRunner", runner):
            try:
                main_ppo.main()
            finally:
                ray.shutdown()

"""CPU source/seam checks, without creating a stand-in actor or model.

The finite scalar-hook checks call the actual imported owner loss functions on
small artificial operands. They verify the diagnostic seam only, and do not
claim a native model, FSDP backward, full minibatch or historical update passed.
They require the already provisioned upstream environment and skip locally when
its dependencies are absent. No package installation is part of these checks.
"""

import ast
import hashlib
import importlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
from types import ModuleType

import pytest


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
SPEC = importlib.util.spec_from_file_location(
    "observe_native_optimizer_minibatch", HERE / "observe_native_optimizer_minibatch.py")
observer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(observer)


def test_actual_source_contract_has_original_scalar_seams_and_native_backward():
    verl_root = os.environ.get("VERL_ROOT")
    source = (Path(verl_root) / "verl" / "workers" / "actor" / "dp_actor.py" if verl_root
              else HERE / "actor-right-padding-20261003" / "base-dp_actor.py")
    assert hashlib.sha256(source.read_bytes()).hexdigest() == observer.ACTOR_SHA256
    tree = ast.parse(source.read_text(encoding="utf-8"))
    update = next(node for node in ast.walk(tree)
                  if isinstance(node, ast.FunctionDef) and node.name == "update_policy")
    direct_aggregate = [node for node in ast.walk(update)
                        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                        and node.func.id == "agg_loss"]
    assert [[keyword.value.id for keyword in call.keywords if keyword.arg == "loss_mat"]
            for call in sorted(direct_aggregate, key=lambda node: node.lineno)] == [["entropy"], ["kld"]]
    core_source = (Path(verl_root) / "verl" / "trainer" / "ppo" / "core_algos.py" if verl_root
                   else HERE / "recipe-sources" / "verl-agent-20bd331" / "verl" / "trainer" / "ppo" / "core_algos.py")
    assert hashlib.sha256(core_source.read_bytes()).hexdigest() == observer.CORE_SHA256
    core_tree = ast.parse(core_source.read_text(encoding="utf-8"))
    policy = next(node for node in core_tree.body
                  if isinstance(node, ast.FunctionDef) and node.name == "compute_policy_loss")
    assert any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
               and node.func.id == "agg_loss" for node in ast.walk(policy))
    diagnostic = ast.parse((HERE / "observe_native_optimizer_minibatch.py").read_text(encoding="utf-8"))
    # Model forward/backward and PPO data loops must remain in the owner.
    assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                   and node.func.attr in ("backward", "_forward_micro_batch")
                   for node in ast.walk(diagnostic))


def test_temporary_alias_restores_value_or_absence_after_error():
    module = ModuleType("temporary_alias_contract")
    original = object()
    module.present = original
    with pytest.raises(RuntimeError):
        with observer._temporary_attribute(module, "present", object()):
            with observer._temporary_attribute(module, "absent", object()):
                raise RuntimeError("diagnostic failure")
    assert module.present is original
    assert "absent" not in vars(module)


def _actual_owner_functions():
    pytest.importorskip("torch")
    pytest.importorskip("verl")
    owner = importlib.import_module("verl.workers.actor.dp_actor")
    core = importlib.import_module("verl.trainer.ppo.core_algos")
    assert observer._source_identity(owner.DataParallelPPOActor)["sha256"] == observer.ACTOR_SHA256
    assert observer._source_identity(core.compute_policy_loss)["sha256"] == observer.CORE_SHA256
    assert owner.compute_policy_loss is core.compute_policy_loss
    assert owner.agg_loss is core.agg_loss
    return owner, core


def test_actual_actor_body_globals_use_the_hooked_owner_aliases(record_property):
    """Inspect the imported original class only; never construct an actor/model.

    The actual GPUMemoryLogger does not set __wrapped__. inspect.unwrap therefore
    still returns its logger function; the native actor body is its original
    decorated_function closure, which is the function GPUMemoryLogger.log calls.
    """
    owner, core = _actual_owner_functions()
    wrapped = owner.DataParallelPPOActor.update_policy
    unwrapped = inspect.unwrap(wrapped)
    nonlocals = inspect.getclosurevars(unwrapped).nonlocals
    assert "decorated_function" in nonlocals
    body = inspect.unwrap(nonlocals["decorated_function"])
    receipt = {
        "scope": "CPU imported-class identity inspection; no actor/model/forward/backward construction.",
        "wrapper": observer._source_identity(unwrapped),
        "unwrap_name": unwrapped.__qualname__,
        "unwrap_globals_is_actor_module": unwrapped.__globals__ is vars(owner),
        "closure_keys": sorted(nonlocals),
        "body": observer._source_identity(body), "body_name": body.__qualname__,
        "body_globals_is_actor_module": body.__globals__ is vars(owner),
        "body_pg_is_module_alias": body.__globals__.get("compute_policy_loss") is owner.compute_policy_loss,
        "body_agg_is_module_alias": body.__globals__.get("agg_loss") is owner.agg_loss,
        "body_pg_is_core": body.__globals__.get("compute_policy_loss") is core.compute_policy_loss,
        "body_agg_is_core": body.__globals__.get("agg_loss") is core.agg_loss,
    }
    encoded = json.dumps(receipt, sort_keys=True)
    record_property("actual_actor_body_alias_identity", encoded)
    print("ACTUAL_ACTOR_BODY_ALIAS_IDENTITY " + encoded, flush=True)
    assert receipt["body"]["sha256"] == observer.ACTOR_SHA256
    assert all(receipt[key] for key in (
        "body_globals_is_actor_module", "body_pg_is_module_alias", "body_agg_is_module_alias",
        "body_pg_is_core", "body_agg_is_core"))


@pytest.mark.parametrize("label", observer.LABELS)
def test_finite_scalar_gates_use_actual_owner_loss_functions(label):
    importorskip = pytest.importorskip
    torch = importorskip("torch")
    owner, core = _actual_owner_functions()
    original_pg, original_agg = owner.compute_policy_loss, owner.agg_loss
    log_prob = torch.tensor([[-1.1, -0.9, -1.2], [-0.8, -1.3, -1.0]], requires_grad=True)
    old = log_prob.detach() + 0.02
    ref = log_prob.detach() - 0.1
    advantages = torch.tensor([[0.3, -0.2, 0.4], [-0.1, 0.2, -0.3]])
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])

    def native_scalars():
        pg = owner.compute_policy_loss(old_log_prob=old, log_prob=log_prob,
                                       advantages=advantages, response_mask=mask,
                                       cliprange=0.2, clip_ratio_c=3.0,
                                       loss_agg_mode="token-mean")[0]
        entropy = owner.agg_loss(loss_mat=torch.nn.functional.softplus(log_prob),
                                 loss_mask=mask, loss_agg_mode="token-mean")
        kld = core.kl_penalty(logprob=log_prob, ref_logprob=ref, kl_penalty="low_var_kl")
        kl = owner.agg_loss(loss_mat=kld, loss_mask=mask, loss_agg_mode="token-mean")
        return pg, entropy, kl

    pg, entropy, kl = native_scalars()
    selected = pg if label in ("dt_pg", "grpo_pg") else -0.001 * entropy if label == "weighted_entropy" else 0.001 * kl
    expected = torch.autograd.grad(selected / 8, log_prob)[0]
    hooks = observer.NativeLossHooks(owner, label=label, entropy_coeff=0.001,
                                     use_kl_loss=True, kl_coef=0.001)
    with hooks.installed():
        assert core.agg_loss is original_agg
        pg, entropy, kl = native_scalars()
        observed = torch.autograd.grad((pg - 0.001 * entropy + 0.001 * kl) / 8, log_prob)[0]
    torch.testing.assert_close(observed, expected)
    assert owner.compute_policy_loss is original_pg and owner.agg_loss is original_agg
    assert hooks.pending == [] and hooks.handles == []
    assert len(hooks.records) == 1
    assert sum(hooks.records[0][component]["selected_backward"] for component in
               ("dt_pg", "weighted_entropy", "weighted_kl")) == 1


def test_grpo_snapshot_uses_original_dataproto_and_preserves_original_container():
    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    pytest.importorskip("tensordict")
    DataProto = pytest.importorskip("verl").DataProto
    assert observer._source_identity(DataProto)["sha256"] == (
        "2ae51f003f72d6ad0f288d5e2d8e94ad172a69422239a8612ba6d1d9dffeb4aa")
    # These small operands test container rebinding; no actor is instantiated.
    advantages = torch.tensor([[0.2, -0.1]])
    grpo = torch.tensor([[1.0, 1.0]])
    data = DataProto.from_dict(tensors={"advantages": advantages, "diagnostic_grpo_advantages": grpo},
                               non_tensors={"uid": np.array(["container-unit"], dtype=object)},
                               meta_info={"temperature": 1.0, "multi_turn": True})
    original_container, original_metadata = data.batch, data.meta_info
    snapshot = observer._grpo_snapshot(data)
    assert snapshot.batch is not original_container
    assert data.batch is original_container and data.batch["advantages"] is advantages
    assert snapshot.batch["advantages"] is grpo
    assert data.meta_info is original_metadata and snapshot.meta_info == original_metadata
    snapshot.meta_info["multi_turn"] = False
    assert data.meta_info["multi_turn"] is True

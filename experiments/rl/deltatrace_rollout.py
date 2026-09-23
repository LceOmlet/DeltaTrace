"""Official DT reward-event endpoint readout on the existing actor weights."""

from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path
from typing import Any

import torch



class _Qwen35CausalOwnerView:
    """Expose VERL's text-only CausalLM through the owner's reviewed ABI.

    The owner runner was reviewed against Qwen3.5's multimodal
    ``ConditionalGeneration`` layout (``model.language_model``), whereas
    VERL loads the same checkpoint as a text-only ``ForCausalLM`` whose
    transformer is ``model``.  This view forwards to the existing actor and
    aliases its existing layers/head; it adds no parameters and no forward
    computation.
    """

    def __init__(self, forward_model: Any, text_model: Any, lm_head: Any, owner_config: Any = None):
        # The text-only Qwen3.5 CausalLM computes logits with a raw
        # ``F.linear`` call, while the reviewed owner runner intentionally
        # captures the native HF ``lm_head`` module.  Reuse the official
        # ConditionalGeneration class as an allocation-free ABI view: its
        # parameters are created on ``meta`` and immediately replaced by the
        # actor's existing sharded text model and head.  No weights, layers,
        # attribution rule, or forward kernel are duplicated here.
        from transformers import Qwen3_5ForConditionalGeneration

        config = owner_config or getattr(forward_model, "config", None)
        if config is None:
            raise TypeError("Qwen3.5 causal owner view requires a model config")
        with torch.device("meta"):
            conditional = Qwen3_5ForConditionalGeneration(config)
        conditional.model.language_model = text_model
        conditional.lm_head = lm_head
        self._conditional = conditional
        self._forward_model = forward_model
        self.model = conditional.model
        self.lm_head = conditional.lm_head
        self._owner_fsdp_modules: list[Any] = []

        # FSDP2 stores the actor's parameters as DTensors.  The actor's normal
        # forward path gathers them through its official FSDP pre-forward hook;
        # calling the ABI shell directly would bypass that hook at lm_head.
        # Register one owner-only forward method through PyTorch's public FSDP2
        # API, so the same actor parameters are gathered for the official owner
        # call and released by the normal FSDP2 post-forward hook.
        self._owner_forward_name = "_deltatrace_owner_forward"
        try:
            from torch.distributed.fsdp import register_fsdp_forward_method
        except ImportError:
            try:
                from torch.distributed.fsdp._fully_shard._fully_shard import register_fsdp_forward_method
            except ImportError:
                register_fsdp_forward_method = None
        if register_fsdp_forward_method is None or not hasattr(forward_model, "_get_fsdp_state"):
            raise TypeError("DeltaTrace owner requires the actor's FSDP2 forward hook")

        def owner_forward(_module: Any, *inner_args: Any, **inner_kwargs: Any) -> Any:
            return self._conditional(*inner_args, **inner_kwargs)

        setattr(forward_model, self._owner_forward_name, types.MethodType(owner_forward, forward_model))
        register_fsdp_forward_method(forward_model, self._owner_forward_name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        modules = []
        seen = set()
        # Traverse both the actor root and the ABI shell.  The shell aliases
        # the exact text transformer, and using it as a second root makes the
        # lifecycle explicit even when PEFT/FSDP stores that transformer below
        # a wrapper that does not expose it through ``modules()``.
        for root in (self._forward_model, self._conditional):
            for module in root.modules():
                if id(module) in seen:
                    continue
                seen.add(id(module))
                if hasattr(module, "_get_fsdp_state"):
                    modules.append(module)
        self._owner_fsdp_modules = modules
        self._owner_params_unsharded = True
        output = getattr(self._forward_model, self._owner_forward_name)(*args, **kwargs)
        # Finite propagation consumes one decoder at a time. Keeping every
        # decoder unsharded here duplicated almost the entire 9B checkpoint and
        # caused an actual OOM at 32768 tokens. Release native-forward gathers;
        # only the root-owned head/final norm are needed for the next seed.
        # Each decoder is gathered at its replay/finite boundary below.
        for module in reversed(modules):
            module.reshard()
        self._forward_model.unshard()
        if not getattr(self, "_owner_lifecycle_reported", False):
            from torch.distributed.tensor import DTensor

            remaining = [
                (name, tuple(getattr(param, "placements", ())))
                for name, param in self._conditional.named_parameters()
                if isinstance(param, DTensor)
            ]
            print(
                f"[DeltaTrace] owner FSDP2 lifecycle modules={len(modules)} "
                f"layerwise=True remaining_dtensor={len(remaining)} sample={remaining[:3]}",
                flush=True,
            )
            self._owner_lifecycle_reported = True
        return output

    def prepare_finite_layer(self, layer: Any) -> None:
        # Replay's official post-forward hook may already have resharded.
        # Public unshard restores the ordinary weights for the finite rules.
        for module in layer.modules():
            if hasattr(module, "_get_fsdp_state"):
                module.unshard()

    def release_finite_layer(self, layer: Any) -> None:
        for module in reversed(list(layer.modules())):
            if hasattr(module, "_get_fsdp_state"):
                module.reshard()

    def forward_root(self, *args: Any, **kwargs: Any) -> Any:
        # A root readout finishes at native logits. Unlike finite pullback it
        # needs no layer replay, hence no extra all-layer unshard afterwards.
        return getattr(self._forward_model, self._owner_forward_name)(*args, **kwargs)

    def release_owner_params(self) -> None:
        if getattr(self, "_owner_params_unsharded", False):
            for module in reversed(self._owner_fsdp_modules):
                module.reshard()
            self._owner_fsdp_modules = []
            self._owner_params_unsharded = False


class DeltaTraceRolloutProducer:
    """Read individual token/event contrasts; compose the PLAN sampled Q/V."""

    def __init__(self, model: Any, *, eos_token_id: int | None = None, pad_token_id: int | None = None):
        # ``DT_ROOT`` is the checkout that contains the ``deltatrace`` Python
        # package (the launcher sets it to .../repo/deltatrace).  Keep this
        # import boundary explicit for Ray actors, whose inherited PYTHONPATH
        # is not guaranteed to contain the caller's working directory.
        root = Path(os.environ["DT_ROOT"]).resolve()
        env_path = Path(os.environ.get("DT_ENVIRONMENT_JSON", root / "environment.json"))
        env = json.loads(env_path.read_text())["qwen35"]
        from transformers import AutoConfig, AutoTokenizer

        # VERL's text-only loader exposes Qwen3_5TextConfig; the official
        # owner ABI view needs the parent multimodal config only to construct
        # its allocation-free HF ConditionalGeneration shell.
        owner_config = AutoConfig.from_pretrained(
            env["checkpoint"], local_files_only=True, trust_remote_code=True
        )
        source = root
        official = Path(os.environ.get("DT_OFFICIAL_ROOT") or env["official_root"])
        sys.path[:0] = [str(root), str(official), str(source / "clean/qwen35")]
        sys.path.append(env["ft_extension_root"])
        os.environ.update(env.get("runtime_environment", {}), HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
        os.environ.setdefault("TRITON_CACHE_DIR", env["triton_cache"])
        os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", env["inductor_cache"])

        # The remote DeltaTrace checkout is the package root (``profiles`` is
        # top-level); the local worktree wraps it one directory deeper.
        try:
            from deltatrace.profiles.official import make_qwen35_runner
            from deltatrace.accelerated.qwen35 import qwen35_code_local_capture
        except ModuleNotFoundError:
            from profiles.official import make_qwen35_runner
            from accelerated.qwen35 import qwen35_code_local_capture
        from finite_fla_gpu import make_compiled_finite_pullback, verify_native_sources
        from qwen35_answer_finite import PackedAnswerTargets
        from vendor_fa_finite_bf16_d256 import VendorFAFiniteP1BF16D256

        verify_native_sources(env["native_stage_source_sha256"])
        # Finite attribution temporarily uses FA; actor updates retain SDPA.
        finite_fa = VendorFAFiniteP1BF16D256(env["finite_library"], env["finite_library_sha256"])
        # VERL's FSDP2 rollout object is a wrapper around the actor (and may
        # also be a PEFT wrapper).  The owner runner must receive the actual
        # HuggingFace module whose ``model.language_model.layers`` and
        # ``lm_head`` it already captures.  Keep the PEFT candidate first so
        # LoRA's effective weights remain part of the factual trace.
        candidates: list[Any] = []
        pending: list[Any] = [model]
        seen: set[int] = set()
        while pending:
            value = pending.pop(0)
            if value is None or id(value) in seen:
                continue
            seen.add(id(value))
            candidates.append(value)
            # FSDP2, PEFT, and the upstream HF wrappers expose different
            # one-hop paths to the same base module.  Walk only these wrapper
            # attributes; do not traverse model layers or create a copy.
            for attr in ("_fsdp_wrapped_module", "module", "base_model", "model", "_orig_mod"):
                child = getattr(value, attr, None)
                if child is not None and id(child) not in seen:
                    pending.append(child)
        owner_model = next(
            (
                value
                for value in candidates
                if not type(value).__name__.startswith("FSDP")
                if hasattr(getattr(value, "model", None), "language_model")
                and hasattr(value, "lm_head")
            ),
            None,
        )
        if owner_model is None:
            # The text-only actor has Qwen3_5TextModel at ``model``.  For a
            # PEFT actor, the same object is reachable through
            # ``base_model.model`` and must remain the forwarding object so
            # the active LoRA weights are included in the owner trace.
            # FSDP2 proxies expose ``model``/``lm_head`` attributes but the
            # proxy head is not the module invoked by its forward.  Prefer
            # the upstream PEFT/base CausalLM candidate, then plain HF
            # modules; never construct the owner view around an FSDP proxy.
            fallback_candidates = sorted(
                candidates,
                key=lambda value: (0 if hasattr(value, "base_model") else 1),
            )
            for value in fallback_candidates:
                if type(value).__name__.startswith("FSDP"):
                    continue
                text_model = getattr(value, "model", None)
                causal_model = None
                if text_model is None or not hasattr(text_model, "layers"):
                    base_model = getattr(value, "base_model", None)
                    causal_model = getattr(base_model, "model", None) if base_model is not None else None
                    # PEFT can point its base model at the FSDP2 root.  Use
                    # the unsharded HF CausalLM candidate discovered by the
                    # wrapper walk for the owner call and hooks.
                    if causal_model is None or type(causal_model).__name__.startswith("FSDP"):
                        causal_model = next(
                            (
                                candidate
                                for candidate in fallback_candidates
                                if not type(candidate).__name__.startswith("FSDP")
                                and hasattr(getattr(candidate, "model", None), "layers")
                                and hasattr(candidate, "lm_head")
                            ),
                            causal_model,
                        )
                    text_model = getattr(causal_model, "model", None) if causal_model is not None else None
                if text_model is None or not hasattr(text_model, "layers"):
                    continue
                # A PEFT/FSDP proxy can expose ``lm_head`` without being the
                # module invoked by its forward.  Attach the owner hook to
                # the actual base CausalLM head; keep ``value`` as the
                # forwarding object so its active LoRA weights remain in the
                # factual trace.
                lm_head = getattr(causal_model, "lm_head", None) if causal_model is not None else None
                if lm_head is None:
                    lm_head = getattr(value, "lm_head", None)
                if lm_head is None:
                    causal = getattr(getattr(value, "base_model", None), "model", None)
                    lm_head = getattr(causal, "lm_head", None)
                if lm_head is not None:
                    # Call the same base CausalLM that owns both the captured
                    # transformer and the hooked head.  LoRA modules are
                    # already injected into this object by the upstream
                    # loader, so this does not create a second model or
                    # bypass the active adapter weights.
                    forward_model = causal_model if causal_model is not None else value
                    # Official VERL shards the PEFT actor itself. Register DT
                    # on that same root so PPO-first and DT-first calls share
                    # FSDP's initialization and parameter lifecycle.
                    from torch.distributed.fsdp import FSDPModule
                    if isinstance(model, FSDPModule):
                        forward_model = model
                    actual_head = next(
                        (
                            child
                            for name, child in getattr(forward_model, "named_modules", lambda: [])()
                            if name.endswith("lm_head") and hasattr(child, "in_features")
                        ),
                        None,
                    )
                    if actual_head is not None:
                        lm_head = actual_head
                    owner_model = _Qwen35CausalOwnerView(forward_model, text_model, lm_head, owner_config)
                    break
        if owner_model is None:
            raise TypeError(
                "DeltaTrace owner requires an HF Qwen3.5 module; received "
                + ", ".join(type(value).__name__ for value in candidates)
            )
        # Reuse the host's recorded, already-tested dynamic execution overlay.
        # Do not recreate its compiler caches or silently change its profile.
        execution = {}
        if env.get("dt_dynamic_shapes", False):
            execution = {"dynamic_shapes": True, "compiler_options": env.get("dt_compiler_options", {})}
        self.runner = make_qwen35_runner(
            owner_model,
            finite_fa,
            make_compiled_finite_pullback(reuse_scalar_products=False, **execution),
            answer_compiled=env.get('dt_answer_compiled', True),
            copy_replay_captures=False,
            capture_backend=qwen35_code_local_capture,
            defer_diagnostics=True,
            offload_replay_mixer=env.get('dt_offload_replay_mixer', False),
            gdn_head_batch_size=env.get('dt_gdn_head_batch_size'),
            compile_gdn_scalar_rules=env.get('dt_compile_gdn_scalar_rules', False),
            pin_replay_host=env.get('dt_pin_replay_host', False),
            **execution,
        )
        self.packed_answer_targets = PackedAnswerTargets
        self.eos_token_id = int(
            eos_token_id
            if eos_token_id is not None
            else getattr(getattr(model, "config", None), "eos_token_id", None)
            or env.get("eos_token_id", 0)
        )
        self.pad_token_id = int(
            pad_token_id
            if pad_token_id is not None
            else getattr(getattr(model, "config", None), "pad_token_id", None)
            or env.get("pad_token_id", self.eos_token_id)
        )
        from reward_readout import EventRatioReadout

        self.actor = model
        self.readout = EventRatioReadout(
            self.runner,
            AutoTokenizer.from_pretrained(env['checkpoint'], local_files_only=True),
            task=os.environ['DT_TASK'],
            max_steps=int(os.environ['DT_MAX_STEPS']),
            max_length=int(os.environ.get('DT_MAX_LENGTH', '32768')),
            packed_answer_targets=self.packed_answer_targets,
        )

    def attribute_episode(self, rows: list[dict[str, Any]], episode_return: float) -> list[dict[str, torch.Tensor]]:
        return self.attribute_episodes([rows], [episode_return])[0]

    def attribute_episodes(self, episodes: list[list[dict[str, Any]]], returns: list[float]) -> list[list[dict[str, torch.Tensor]]]:
        if len(episodes) != len(returns):
            raise ValueError("episode and return counts differ")
        # Per-event official rewards in rows are authoritative; never multiply
        # by episode_return again. RPC retains that upstream summary argument.
        training = self.actor.training
        self.actor.eval()
        text_model = self.runner.model.model.language_model
        attention = text_model.config._attn_implementation
        try:
            # Public HF dispatch switches the existing layers, without new
            # weights. Restore the actor's selected backend afterwards; some
            # hosts have a forward-only FA wheel, while MetaX's installed FA2
            # backward is separately verified in its environment receipt.
            text_model.set_attn_implementation('flash_attention_2')
            result = self.readout.episodes(episodes)
            print('[DeltaTrace readout] ' + json.dumps(self.readout.last_report), flush=True)
            return result
        finally:
            text_model.set_attn_implementation(attention)
            self.actor.train(training)

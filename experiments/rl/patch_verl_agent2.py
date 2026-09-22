"""Apply compatibility and the owner DeltaTrace token seam to the pinned tree.

The upstream Sokoban package has a nested package with the same name as its
parent.  Eagerly importing both modules creates a Python 3.12 import cycle
(and can segfault before a traceback).  Princeton's legacy Gym environment
also needs its checker disabled, and Sokoban must not import Transformers in a
Ray worker. The pinned tree also assumes a CUDA flash-attn binary at import
time; on hosts whose glibc cannot load that optional binary, we keep upstream
Hugging Face SDPA as the default and only import flash-attn when VERL's
remove-padding path is explicitly enabled. The DeltaTrace token estimator is
added at VERL's existing advantage boundary; it does not replace the trainer,
rollout, optimizer, or environment dynamics.
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path


CONTENT = '''# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0.

from .projection import sokoban_projection


def build_sokoban_envs(*args, **kwargs):
    """Lazy import to avoid the upstream nested-package import cycle."""
    from .envs import build_sokoban_envs as _build_sokoban_envs
    return _build_sokoban_envs(*args, **kwargs)
'''

WEBSHOP_FILE = "agent_system/environments/env_package/webshop/envs.py"
WEBSHOP_OLD = "        env_kwargs['seed'] = seed\n        self.env = gym.make('WebAgentTextEnv-v0', **env_kwargs)"
WEBSHOP_NEW = "        env_kwargs = dict(env_kwargs)\n        env_kwargs['seed'] = seed\n        # Princeton's legacy Gym environment has no action_space.\n        env_kwargs.setdefault('disable_env_checker', True)\n        self.env = gym.make('WebAgentTextEnv-v0', **env_kwargs)"
SOKOBAN_ENV_FILE = "agent_system/environments/env_package/sokoban/sokoban/env.py"
SOKOBAN_ENV_OLD = "from agent_system.environments.env_package.sokoban.sokoban.base import BaseDiscreteActionEnv"
SOKOBAN_ENV_NEW = "from .base import BaseDiscreteActionEnv"
SOKOBAN_BASE_FILE = "agent_system/environments/env_package/sokoban/sokoban/base.py"
SOKOBAN_BASE_OLD = "from transformers import AutoTokenizer\nimport torch"
SOKOBAN_BASE_NEW = "from typing import TYPE_CHECKING\n\nimport torch\n\nif TYPE_CHECKING:\n    from transformers import AutoTokenizer"

ACTOR_FILE = "verl/workers/actor/dp_actor.py"
ACTOR_IMPORT_OLD = """if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input
"""
ACTOR_IMPORT_NEW = """if is_npu_available and not is_cuda_available:
    # Imported lazily for the NPU remove-padding path below.
    pass
"""
ACTOR_BRANCH_OLD = "            if self.use_remove_padding:\n                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)"
ACTOR_BRANCH_NEW = """            if self.use_remove_padding:
                if is_cuda_available:
                    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
                elif is_npu_available:
                    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input
                else:
                    raise RuntimeError("remove-padding requires flash-attn on CUDA or NPU attention utilities")
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)"""
ACTOR_BRANCH_MARKER = "            if self.use_remove_padding:\n                if is_cuda_available:"

CRITIC_FILE = "verl/workers/critic/dp_critic.py"
CRITIC_IMPORT_OLD = "from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input\n"
CRITIC_BLOCK_OLD = """if is_cuda_available:
    from flash_attn.bert_padding import pad_input, unpad_input, rearrange, index_first_axis
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import pad_input, unpad_input, rearrange, index_first_axis
"""
CRITIC_BRANCH_OLD = "            if self.use_remove_padding:\n                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)"
CRITIC_BRANCH_NEW = """            if self.use_remove_padding:
                if is_cuda_available:
                    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
                elif is_npu_available:
                    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input
                else:
                    raise RuntimeError("remove-padding requires flash-attn on CUDA or NPU attention utilities")
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)"""
CRITIC_BRANCH_MARKER = "            if self.use_remove_padding:\n                if is_cuda_available:"
CRITIC_BROKEN = """            if self.use_remove_padding:
                if is_cuda_available:
                                    elif is_npu_available:
                    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input
                else:
                    raise RuntimeError("remove-padding requires flash-attn on CUDA or NPU attention utilities")"""
CRITIC_BAD_ASSIGN = "                    position_ids_rmpad =\n                    index_first_axis("
CRITIC_GOOD_ASSIGN = "                    position_ids_rmpad = index_first_axis("
CRITIC_DUPLICATE_UNPAD = "                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)\n                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)"
CRITIC_SINGLE_UNPAD = "                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)"

FSDP_FILE = "verl/workers/fsdp_workers.py"
FSDP_ATTN_OLD = 'attn_implementation="flash_attention_2"'
FSDP_ATTN_NEW = 'attn_implementation=os.getenv("VERL_ATTN_IMPLEMENTATION", "sdpa")'
FSDP_GENERATION_IDS_OLD = '''            "eos_token_id": self.generation_config.eos_token_id if self.generation_config is not None else self.tokenizer.eos_token_id,
            "pad_token_id": self.generation_config.pad_token_id if self.generation_config is not None else self.tokenizer.pad_token_id,'''
FSDP_GENERATION_IDS_NEW = '''            # A GenerationConfig may exist with unset special-token fields.
            # Preserve explicit generation IDs, otherwise use the tokenizer
            # that already owns the collector's prompt padding and EOS mask.
            "eos_token_id": self.generation_config.eos_token_id if getattr(self.generation_config, "eos_token_id", None) is not None else self.tokenizer.eos_token_id,
            "pad_token_id": self.generation_config.pad_token_id if getattr(self.generation_config, "pad_token_id", None) is not None else self.tokenizer.pad_token_id,'''
VISION_IMPORT_OLD = "        from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForVision2Seq"
VISION_IMPORT_NEW = """        from transformers import AutoConfig, AutoModelForCausalLM
        try:
            from transformers import AutoModelForVision2Seq
        except ImportError:  # Transformers 5.x removed this legacy alias.
            AutoModelForVision2Seq = None"""
VISION_CHECK_OLD = "            if type(actor_model_config) in AutoModelForVision2Seq._model_mapping.keys():"
VISION_CHECK_NEW = "            if AutoModelForVision2Seq is not None and type(actor_model_config) in AutoModelForVision2Seq._model_mapping.keys():"
LORA_CALL_OLD = "                actor_module = get_peft_model(actor_module, LoraConfig(**lora_config))"
LORA_CALL_NEW = "                actor_module = get_peft_model(actor_module, LoraConfig(**lora_config), autocast_adapter_dtype=False)"
FSDP_ORIG_PARAMS_OLD = "                use_orig_params=False,"
FSDP_ORIG_PARAMS_PREVIOUS = "                use_orig_params=self._is_lora,"
# The owner's LoRA auto-wrap policy separates frozen/trainable parameters and
# uses flat parameters. Retain that behavior: forcing orig params on the
# layer-wrapped Qwen path breaks eval -> train parameter writeback. Only the
# unwrapped HF/LoRA path needs orig params for mixed requires_grad in one handle.
FSDP_ORIG_PARAMS_NEW = "                use_orig_params=self._is_lora and auto_wrap_policy is None,"
# FSDP's upstream actor path disables CPU parameter offload because older
# gradient-accumulation code could observe stale handles.  The single-GPU
# LoRA path has one microbatch per optimizer step, so opt in explicitly when
# requested instead of changing the trainer or optimizer implementation.
FSDP_ACTOR_OFFLOAD_OLD = "        cpu_offload = None if role == \"actor\" else CPUOffload(offload_params=True)"
FSDP_ACTOR_OFFLOAD_NEW = "        cpu_offload = CPUOffload(offload_params=True) if role == \"actor\" and os.getenv(\"VERL_ACTOR_CPU_OFFLOAD\", \"0\") == \"1\" else (None if role == \"actor\" else CPUOffload(offload_params=True))"
ACTOR_FORWARD_OLD = """                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating"""
ACTOR_FORWARD_NEW = """                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    logits_to_keep=response_length + 1,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating"""
TRL_BLOCK_OLD = """    if is_trl_available():
        from trl import AutoModelForCausalLMWithValueHead  # type: ignore

        def state_dict(self, *args, **kwargs):
            return torch.nn.Module.state_dict(self, *args, **kwargs)

        AutoModelForCausalLMWithValueHead.state_dict = state_dict
        print("Monkey patch state_dict in AutoModelForCausalLMWithValueHead. ")
"""
TRL_BLOCK_NEW = """    if is_trl_available():
        try:
            from trl import AutoModelForCausalLMWithValueHead  # type: ignore
        except ImportError:
            AutoModelForCausalLMWithValueHead = None
        if AutoModelForCausalLMWithValueHead is not None:
            def state_dict(self, *args, **kwargs):
                return torch.nn.Module.state_dict(self, *args, **kwargs)

            AutoModelForCausalLMWithValueHead.state_dict = state_dict
            print("Monkey patch state_dict in AutoModelForCausalLMWithValueHead. ")
"""
FSDP_POLICY_FILE = "verl/utils/fsdp_utils.py"
FSDP_POLICY_OLD = """            if transformer_cls is None:
                raise Exception("Could not find the transformer layer class to wrap in the model.")
            else:
                transformer_cls_to_wrap.add(transformer_cls)
"""
FSDP_POLICY_NEW = """            if transformer_cls is None:
                # Some Transformers configs list vision blocks in _no_split_modules
                # even when the loaded text-only model has no such class.
                continue
            transformer_cls_to_wrap.add(transformer_cls)
        if not transformer_cls_to_wrap:
            raise Exception("Could not find the transformer layer class to wrap in the model.")
"""
FSDP2_FILE = "verl/utils/fsdp_utils.py"
FSDP2_OLD = """    if isinstance(fsdp_transformer_layer_cls_to_wrap, str):
        fsdp_transformer_layer_cls_to_wrap = [fsdp_transformer_layer_cls_to_wrap]

    assert len(fsdp_transformer_layer_cls_to_wrap) > 0 and fsdp_transformer_layer_cls_to_wrap[0] is not None
"""
FSDP2_NEW = """    if isinstance(fsdp_transformer_layer_cls_to_wrap, str):
        fsdp_transformer_layer_cls_to_wrap = [fsdp_transformer_layer_cls_to_wrap]
    elif isinstance(fsdp_transformer_layer_cls_to_wrap, (set, tuple)):
        fsdp_transformer_layer_cls_to_wrap = list(fsdp_transformer_layer_cls_to_wrap)

    assert len(fsdp_transformer_layer_cls_to_wrap) > 0 and fsdp_transformer_layer_cls_to_wrap[0] is not None
"""

HF_ROLLOUT_FILE = "verl/workers/rollout/hf_rollout.py"
HF_ROLLOUT_IMPORT_OLD = "from verl.utils.device import get_torch_device"
HF_ROLLOUT_IMPORT_NEW = "from verl.utils.device import get_device_id, get_device_name, get_torch_device"
HF_ROLLOUT_OLD = """        idx = prompts.batch[\"input_ids\"]  # (bs, prompt_length)
        prompt_length = idx.size(1)
        attention_mask = prompts.batch[\"attention_mask\"]  # left-padded attention_mask
        position_ids = prompts.batch[\"position_ids\"]
"""
HF_ROLLOUT_NEW = """        idx = prompts.batch[\"input_ids\"]  # (bs, prompt_length)
        prompt_length = idx.size(1)
        attention_mask = prompts.batch[\"attention_mask\"]  # left-padded attention_mask
        position_ids = prompts.batch[\"position_ids\"]
        # FSDP2 CPU parameter offload can leave position_ids on CPU while
        # Qwen3.5's rotary kernel runs on CUDA. Keep all generation inputs on
        # the active torch device; this is an upstream device-placement fix.
        device = torch.device(get_device_name(), get_device_id())
        idx = idx.to(device)
        attention_mask = attention_mask.to(device)
        position_ids = position_ids.to(device)
"""
# A previous revision injected the device block with get_torch_device(), which
# returns a torch module rather than a torch.device. Remove that stale block
# when upgrading an already patched checkout.
HF_ROLLOUT_BAD_BLOCK = """        # FSDP2 CPU parameter offload can leave position_ids on CPU while
        # Qwen3.5's rotary kernel runs on CUDA. Keep all generation inputs on
        # the active torch device; this is an upstream device-placement fix.
        device = get_torch_device()
        idx = idx.to(device)
        attention_mask = attention_mask.to(device)
        position_ids = position_ids.to(device)
"""
HF_SUMMON_OLD = "            param_ctx = FSDP.summon_full_params(self.module, writeback=False, recurse=False)"
HF_SUMMON_NEW = "            param_ctx = FSDP.summon_full_params(self.module, writeback=False, recurse=False)"
HF_SUMMON_PREVIOUS = "            param_ctx = FSDP.summon_full_params(self.module, writeback=False, recurse=True)"

HF_DT_METHOD_ANCHOR = "        return DataProto(batch=batch)\n"
HF_TRIM_ANCHOR = "        self.module.eval()\n        param_ctx = contextlib.nullcontext()"
HF_TRIM_INSERT = '''        # Trim only shared left padding for native generation; restore exact
        # DataProto width below so collector IDs/masks keep their original ABI.
        generation_left_trim = 0
        if os.getenv("VERL_TRIM_SHARED_PADDING", "0") == "1":
            generation_left_trim = int(attention_mask.long().argmax(-1).min().item())
        self.module.eval()
        param_ctx = contextlib.nullcontext()'''
HF_TRIM_CALL_OLD = '''            output = self.module.generate(
                input_ids=idx,
                attention_mask=attention_mask,
                position_ids=position_ids,'''
HF_TRIM_CALL_NEW = '''            output = self.module.generate(
                input_ids=idx[:, generation_left_trim:],
                attention_mask=attention_mask[:, generation_left_trim:],
                position_ids=position_ids[..., generation_left_trim:],'''
HF_TRIM_RESTORE = '''        seq = output.sequences
        if generation_left_trim:
            padding = idx[:, :generation_left_trim]
            padding = padding.repeat_interleave(seq.size(0) // idx.size(0), dim=0)
            seq = torch.cat((padding, seq), dim=-1)'''
ACTOR_TRIM_ANCHOR = '            else:  # not using rmpad and no ulysses sp\n                extra_args = {}'
ACTOR_TRIM_INSERT = '''            else:  # not using rmpad and no ulysses sp
                # Preserve the response columns and original tensors. Only
                # columns masked out for every example leave the model input.
                if not multi_modal_inputs and os.getenv("VERL_TRIM_SHARED_PADDING", "0") == "1":
                    left_trim = int(attention_mask.long().argmax(-1).min().item())
                    input_ids = input_ids[:, left_trim:]
                    attention_mask = attention_mask[:, left_trim:]
                    position_ids = position_ids[..., left_trim:]
                extra_args = {}'''
HF_DT_METHOD_OLD = "    def compute_dt_token_advantages(self, episodes, episode_returns):\n"
HF_DT_METHOD_MARKER = "    def compute_dt_token_advantages(self, episodes, episode_returns, eos_token_id=None, pad_token_id=None):\n"
HF_DT_METHOD = '''        return DataProto(batch=batch)

    def compute_dt_token_advantages(self, episodes, episode_returns, eos_token_id=None, pad_token_id=None):
        """Run the owner DeltaTrace producer on completed factual episodes."""
        from deltatrace_rollout import DeltaTraceRolloutProducer

        if not hasattr(self, "_deltatrace_producer"):
            self._deltatrace_producer = DeltaTraceRolloutProducer(
                self.module, eos_token_id=eos_token_id, pad_token_id=pad_token_id
            )
        return self._deltatrace_producer.attribute_episodes(episodes, episode_returns)
'''

FSDP_DT_METHOD_MARKER = "    def compute_dt_token_advantages(self, episodes, episode_returns, eos_token_id=None, pad_token_id=None):\n"
FSDP_DT_METHOD_OLD = "    def compute_dt_token_advantages(self, episodes, episode_returns):\n"
FSDP_DT_METHOD_ANCHOR = "    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)\n    def compute_log_prob(self, data: DataProto):\n"
FSDP_DT_METHOD = '''    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def compute_dt_token_advantages(self, episodes, episode_returns, eos_token_id=None, pad_token_id=None):
        """Delegate DT attribution to the existing HF rollout model."""
        if not hasattr(self, "rollout") or not hasattr(self.rollout, "compute_dt_token_advantages"):
            raise RuntimeError("DeltaTrace requires the upstream HF rollout backend")
        return self.rollout.compute_dt_token_advantages(
            episodes, episode_returns, eos_token_id=eos_token_id, pad_token_id=pad_token_id
        )

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_log_prob(self, data: DataProto):
'''

# The trainer remains the semantic owner of policy optimization. This patch
# only adds the DT token-advantage estimator at its existing advantage
# boundary; rollout and actor updates stay upstream. The collector must attach
# the owner-produced sampled Q/V and advantage to every trajectory row.
RAY_TRAINER_FILE = "verl/trainer/ppo/ray_trainer.py"
RAY_ADV_ENUM_OLD = "    GiGPO = 'gigpo'\n"
RAY_ADV_ENUM_NEW = "    GiGPO = 'gigpo'\n    DELTATRACE = 'deltatrace'\n"
RAY_ADV_ENUM_STALE = "    GiGPO = 'gigpo'\n    COUNTERFACTUAL = 'counterfactual'\n"
RAY_ADV_INSERT_ANCHOR = "        data.batch['advantages'] = advantages\n        data.batch['returns'] = returns\n    else:\n        raise NotImplementedError\n    return data\n"
RAY_ADV_INSERT_PREVIOUS = '        data.batch[\'advantages\'] = advantages\n        data.batch[\'returns\'] = returns\n    elif adv_estimator == AdvantageEstimator.DELTATRACE:\n        if "dt_token_advantages" not in data.batch:\n            raise RuntimeError("deltatrace estimator requires owner-produced token advantages")\n        response_mask = data.batch["response_mask"].to(device=data.batch["responses"].device, dtype=torch.bool)\n        advantages = data.batch["dt_token_advantages"].to(device=data.batch["responses"].device, dtype=torch.float32)\n        if advantages.shape != response_mask.shape:\n            raise RuntimeError("owner DT token advantages must align with responses")\n        advantages = advantages * response_mask.to(dtype=advantages.dtype)\n        if not torch.isfinite(advantages).all():\n            raise RuntimeError("owner DT token advantages contain non-finite values")\n        data.batch["advantages"] = advantages\n        data.batch["returns"] = advantages.detach().clone()\n    else:\n        raise NotImplementedError\n    return data\n'
RAY_ADV_INSERT = '        data.batch[\'advantages\'] = advantages\n        data.batch[\'returns\'] = returns\n    elif adv_estimator == AdvantageEstimator.DELTATRACE:\n        response_mask = data.batch["response_mask"].to(device=data.batch["responses"].device, dtype=torch.bool)\n        for name in ("dt_token_advantages", "dt_q_estimates", "dt_v_estimates"):\n            if name not in data.batch:\n                raise RuntimeError(f"deltatrace estimator requires {name}")\n            value = data.batch[name].detach().to(device=response_mask.device, dtype=torch.float32)\n            if value.shape != response_mask.shape:\n                raise RuntimeError(f"{name} must align with original response tokens")\n            if not torch.isfinite(value).all():\n                raise RuntimeError(f"{name} contains non-finite values")\n            data.batch[name] = torch.where(response_mask, value, 0.0)\n        data.batch["advantages"] = data.batch["dt_token_advantages"]\n        data.batch["returns"] = data.batch["dt_q_estimates"]\n    else:\n        raise NotImplementedError\n    return data\n'
RAY_ADV_INSERT_STALE = "        data.batch['advantages'] = advantages\n        data.batch['returns'] = returns\n    elif adv_estimator == AdvantageEstimator.COUNTERFACTUAL:\n        if \"counterfactual_credit\" not in data.batch or \"counterfactual_action_mask\" not in data.batch:\n            raise RuntimeError(\"counterfactual estimator requires collector-provided action credit\")\n        credit = data.batch[\"counterfactual_credit\"].to(device=data.batch[\"responses\"].device, dtype=torch.float32)\n        action_mask = data.batch[\"counterfactual_action_mask\"].to(device=data.batch[\"responses\"].device, dtype=torch.bool)\n        response_mask = data.batch[\"response_mask\"].to(dtype=torch.float32)\n        advantages = response_mask * credit[:, None] * action_mask[:, None].to(torch.float32)\n        data.batch[\"advantages\"] = advantages\n        data.batch[\"returns\"] = advantages.detach().clone()\n    else:\n        raise NotImplementedError\n    return data\n"
RAY_USE_CRITIC_OLD = "            AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE,\n            AdvantageEstimator.GiGPO,\n            AdvantageEstimator.COUNTERFACTUAL\n        ]:\n"
RAY_USE_CRITIC_PRISTINE = "            AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE,\n            AdvantageEstimator.GiGPO\n        ]:\n"
RAY_USE_CRITIC_NEW = "            AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE,\n            AdvantageEstimator.GiGPO,\n            AdvantageEstimator.DELTATRACE\n        ]:\n"
RAY_ROLLOUT_FILE = "agent_system/multi_turn_rollout/rollout_loop.py"
ROLLOUT_STEP_ANCHOR = "            batch.non_tensor_batch['traj_uid'] = traj_uid\n"
ROLLOUT_STEP_INSERT = "            batch.non_tensor_batch['traj_uid'] = traj_uid\n            batch.non_tensor_batch['env_step'] = np.full(batch_size, _step, dtype=np.int64)\n"
ROLLOUT_EVENT_ANCHOR = '            batch_list: list[dict] = to_list_of_dict(batch)\n'
ROLLOUT_EVENT_INSERT = '            batch_list: list[dict] = to_list_of_dict(batch)\n\n            if str(self.config.algorithm.adv_estimator) == "deltatrace":\n                from copy import deepcopy\n                for event_index, row in enumerate(batch_list):\n                    row["dt_env_outcome"] = {\n                        "observation": deepcopy({\n                            key: None if value is None else value[event_index]\n                            for key, value in next_obs.items()\n                        }),\n                        "info": deepcopy(infos[event_index]),\n                        "done": bool(dones[event_index]),\n                    }\n'
RAW_PROMPT_KEEP_OLD = '            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]\n'
RAW_PROMPT_KEEP_NEW = '            non_tensor_batch_keys_to_pop = []\n'
# Some upstream environment projections (notably Sokoban) normalize the
# list passed to ``envs.step`` in place, replacing decoded text with integer
# action ids. Preserve the decoded response for the next chat turn; this is a
# one-line collector seam, not a second environment implementation.
ROLLOUT_ACTION_COPY_OLD = "            text_actions = self.tokenizer.batch_decode(batch.batch['responses'], skip_special_tokens=True)\n            \n            next_obs, rewards, dones, infos = envs.step(text_actions)\n"
ROLLOUT_ACTION_COPY_NEW = "            text_actions = self.tokenizer.batch_decode(batch.batch['responses'], skip_special_tokens=True)\n            env_actions = list(text_actions)\n            next_obs, rewards, dones, infos = envs.step(env_actions)\n"
GATHER_ANCHOR = "        batch_size = len(total_batch_list)\n\n        success_rate = {}\n"
GATHER_BROKEN_IMPORT = "            try:\n                try:\n                from experiments.rl.deltatrace_credit import averaged_traced_credit\n            except ImportError:\n                from deltatrace_credit import averaged_traced_credit\n            except ImportError:\n                from deltatrace_credit import averaged_traced_credit\n"
GATHER_GOOD_IMPORT = ""
GATHER_OLD_MARKER = "        # Exact finite-sample policy baseline for the first action."
GATHER_INSERT_PREVIOUS = '        batch_size = len(total_batch_list)\n\n        if str(self.config.algorithm.adv_estimator) == "deltatrace":\n            # The owner DT boundary must already have attached one aligned\n            # token-advantage vector to every active trajectory row. Do not\n            # fabricate a scalar, broadcast a span value, or fall back to a\n            # group mean here.\n            for rows in total_batch_list:\n                if not rows:\n                    raise RuntimeError("deltatrace rollout has no trajectory rows")\n                for row in rows:\n                    if "dt_token_advantages" not in row:\n                        raise RuntimeError("owner DT token advantages are missing from a rollout row")\n                    value = row["dt_token_advantages"]\n                    if not isinstance(value, torch.Tensor) or value.ndim != 1:\n                        raise RuntimeError("owner DT token advantages must be a 1-D tensor per row")\n                    if not torch.isfinite(value).all():\n                        raise RuntimeError("owner DT token advantages contain non-finite values")\n\n        success_rate = {}\n'
GATHER_INSERT = '        batch_size = len(total_batch_list)\n\n        if str(self.config.algorithm.adv_estimator) == "deltatrace":\n            for rows in total_batch_list:\n                for row in rows:\n                    if not row["active_masks"]:\n                        continue\n                    for name in ("dt_token_advantages", "dt_q_estimates", "dt_v_estimates"):\n                        value = row.get(name)\n                        if not isinstance(value, torch.Tensor) or value.shape != row["responses"].shape:\n                            raise RuntimeError(f"{name} must align with original response tokens")\n                        if not torch.isfinite(value).all():\n                            raise RuntimeError(f"{name} contains non-finite values")\n\n        success_rate = {}\n'
ROLLOUT_DT_CALL_ANCHOR = "        # Create trajectory data\n        gen_batch_output: DataProto = self.gather_rollout_data(\n"
ROLLOUT_DT_CALL_PREVIOUS = '        # Create token advantages with the owner DeltaTrace runner. The\n        # worker receives the complete factual episode so observations can be\n        # folded back to the policy token that generated each tool call.\n        if str(self.config.algorithm.adv_estimator) == "deltatrace":\n            dt_values = actor_rollout_wg.compute_dt_token_advantages(\n                total_batch_list,\n                total_episode_rewards.tolist(),\n                eos_token_id=int(self.tokenizer.eos_token_id),\n                pad_token_id=int(self.tokenizer.pad_token_id),\n            )\n            if isinstance(dt_values, list) and len(dt_values) == 1:\n                dt_values = dt_values[0]\n            if len(dt_values) != len(total_batch_list):\n                raise RuntimeError("owner DT returned the wrong episode count")\n            for rows, values in zip(total_batch_list, dt_values):\n                if len(rows) != len(values):\n                    raise RuntimeError("owner DT returned the wrong row count")\n                for row, value in zip(rows, values):\n                    row["dt_token_advantages"] = value\n\n        # Create trajectory data\n        gen_batch_output: DataProto = self.gather_rollout_data(\n'
ROLLOUT_DT_CALL = '        # Preserve each factual reward event and each generated token. The\n        # producer returns sampled Q/V and advantages, with no O-credit routing.\n        if str(self.config.algorithm.adv_estimator) == "deltatrace":\n            dt_values = actor_rollout_wg.compute_dt_token_advantages(\n                total_batch_list,\n                total_episode_rewards.tolist(),\n                eos_token_id=int(self.tokenizer.eos_token_id),\n                pad_token_id=int(self.tokenizer.pad_token_id),\n            )\n            if isinstance(dt_values, list) and len(dt_values) == 1:\n                dt_values = dt_values[0]\n            if len(dt_values) != len(total_batch_list):\n                raise RuntimeError("owner DT returned the wrong episode count")\n            for rows, values in zip(total_batch_list, dt_values):\n                if len(rows) != len(values):\n                    raise RuntimeError("owner DT returned the wrong row count")\n                for row, value in zip(rows, values):\n                    for name in ("dt_token_advantages", "dt_q_estimates", "dt_v_estimates"):\n                        row[name] = value[name]\n\n        # Create trajectory data\n        gen_batch_output: DataProto = self.gather_rollout_data(\n'
ROLLOUT_DT_CALL_STALE = '''        # Create token advantages with the owner DeltaTrace runner. The
        # worker receives the complete factual episode so observations can be
        # folded back to the policy token that generated each tool call.
        if str(self.config.algorithm.adv_estimator) == "deltatrace":
            dt_values = actor_rollout_wg.compute_dt_token_advantages(
                total_batch_list, total_episode_rewards.tolist()
            )
            if isinstance(dt_values, list) and len(dt_values) == 1:
                dt_values = dt_values[0]
            if len(dt_values) != len(total_batch_list):
                raise RuntimeError("owner DT returned the wrong episode count")
            for rows, values in zip(total_batch_list, dt_values):
                if len(rows) != len(values):
                    raise RuntimeError("owner DT returned the wrong row count")
                for row, value in zip(rows, values):
                    row["dt_token_advantages"] = value

'''

# FSDP2 must be attached to the actual Transformers root when PEFT wraps it.
# PeftModel.forward delegates into base_model.model; attaching the root state
# to the wrapper can leave child FSDP states lazy-initialized first and fail in
# the upstream log-prob pass. This only selects the upstream forward owner.
FSDP2_ACTOR_OLD = (
    "            full_state = actor_module.state_dict()\n"
    "            apply_fsdp2(actor_module, fsdp_kwargs, fsdp_config)\n"
    "            fsdp2_load_full_state_dict(actor_module, full_state, fsdp_mesh, cpu_offload)\n"
    "            actor_module_fsdp = actor_module\n"
)
FSDP2_ACTOR_NEW = (
    "            fsdp2_model = getattr(getattr(actor_module, \"base_model\", None), \"model\", actor_module)\n"
    "            full_state = fsdp2_model.state_dict()\n"
    "            apply_fsdp2(fsdp2_model, fsdp_kwargs, fsdp_config)\n"
    "            fsdp2_load_full_state_dict(fsdp2_model, full_state, fsdp_mesh, cpu_offload)\n"
    "            actor_module_fsdp = actor_module\n"
)

QWEN35_OLD = "        position_ids_expanded = position_ids[:, :, None, :].float()  # shape (3, bs, 1, positions)"
QWEN35_NEW = "        position_ids_expanded = position_ids[:, :, None, :].float().to(x.device)  # shape (3, bs, 1, positions)"
QWEN35_RMS_OLD = "        output = output * (1.0 + self.weight.float())"
QWEN35_RMS_PREV = "        weight = self.weight.to_local() if hasattr(self.weight, \"to_local\") else self.weight\n        output = output * (1.0 + weight.float())"
QWEN35_RMS_NEW = "        weight = self.weight.to_local() if hasattr(self.weight, \"to_local\") else self.weight\n        output = output * (1.0 + weight.float().to(output.device))"
QWEN35_LM_OLD = "        logits = self.lm_head(hidden_states[:, slice_indices, :])"
QWEN35_LM_PREV = "        lm_weight = self.lm_head.weight.to_local() if hasattr(self.lm_head.weight, \"to_local\") else self.lm_head.weight\n        lm_bias = self.lm_head.bias.to_local() if getattr(self.lm_head, \"bias\", None) is not None and hasattr(self.lm_head.bias, \"to_local\") else getattr(self.lm_head, \"bias\", None)\n        logits = F.linear(hidden_states[:, slice_indices, :], lm_weight, lm_bias)"
QWEN35_LM_NEW = "        lm_weight = self.lm_head.weight.to_local() if hasattr(self.lm_head.weight, \"to_local\") else self.lm_head.weight\n        lm_bias = self.lm_head.bias.to_local() if getattr(self.lm_head, \"bias\", None) is not None and hasattr(self.lm_head.bias, \"to_local\") else getattr(self.lm_head, \"bias\", None)\n        lm_weight = lm_weight.to(hidden_states.device)\n        lm_bias = lm_bias.to(hidden_states.device) if lm_bias is not None else None\n        logits = F.linear(hidden_states[:, slice_indices, :], lm_weight, lm_bias)"
HF_WRAP_OLD = '        if self._is_rollout and self.config.rollout.name == "hf":\n            # TODO(zhangchi.usc1992, shengguangming) fix me. Current, auto_wrap_policy causes HFRollout to hang in Gemma\n            auto_wrap_policy = None'
HF_WRAP_NEW = '        if self._is_rollout and self.config.rollout.name == "hf" and os.getenv("VERL_ENABLE_HF_FSDP_WRAP", "0") != "1":\n            # Keep upstream HF rollout\'s conservative default; long-context\n            # single-GPU runs can opt into layer wrapping explicitly.\n            auto_wrap_policy = None'


def patch_conversation_observations(text: str) -> str:
    """Opt-in owner rendering for the collector's existing full chat history.

    Legacy state prompts repeat instructions and recent actions inside every
    observation. The full-chat collector already retains these exact messages.
    Keep the initial prompt and legacy default unchanged; render only the new
    observation (plus WebShop's current admissible actions) on later turns.
    """
    additions = {
        'SokobanEnvironmentManager': '''        if not init and self.config.env.get("full_chat_observations", False) and not self.is_multi_modal:
            return [f"Your current observation is:\\n{value}\\nYour admissible actions are [\\"up\\", \\"down\\", \\"left\\", \\"right\\"]."
                    for value in text_obs]

''',
        'WebshopEnvironmentManager': '''        if not init and self.config.env.get("full_chat_observations", False):
            result = []
            for value, info in zip(text_obs, infos):
                actions = "\\n".join(f"'{s}'," for s in self.format_avail_actions(info['available_actions']))
                result.append(f"Your current observation is: {value}.\\nYour admissible actions of the current situation are:\\n[\\n{actions}\\n].")
            return result

''',
        'AppWorldEnvironmentManager': '''        if not init and self.config.env.get("full_chat_observations", False):
            return list(text_obs)

''',
    }
    for name, insertion in additions.items():
        start = text.index('class '+name+'(')
        end = text.find('\nclass ', start+1)
        end = len(text) if end < 0 else end
        section = text[start:end]
        if insertion in section:
            continue
        method = section.index('    def build_text_obs(')
        anchor = section.index('        postprocess_text_obs = []', method)
        section = section[:anchor]+insertion+section[anchor:]
        text = text[:start]+section+text[end:]
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("verl_root", type=Path)
    args = parser.parse_args()
    env_manager = args.verl_root / 'agent_system/environments/env_manager.py'
    env_text = env_manager.read_text()
    env_manager.write_text(patch_conversation_observations(env_text))
    target = args.verl_root / "agent_system/environments/env_package/sokoban/__init__.py"
    if not target.is_file():
        raise FileNotFoundError(target)
    if target.read_text() != CONTENT:
        target.write_text(CONTENT)
        print(f"patched {target}")
    else:
        print(f"already patched {target}")
    webshop = args.verl_root / WEBSHOP_FILE
    text = webshop.read_text()
    if WEBSHOP_NEW not in text:
        if WEBSHOP_OLD not in text:
            raise RuntimeError(f"cannot find WebShop compatibility anchor in {webshop}")
        webshop.write_text(text.replace(WEBSHOP_OLD, WEBSHOP_NEW, 1))
        print(f"patched {webshop}")
    else:
        print(f"already patched {webshop}")
    sokoban_env = args.verl_root / SOKOBAN_ENV_FILE
    text = sokoban_env.read_text()
    if SOKOBAN_ENV_NEW not in text:
        if SOKOBAN_ENV_OLD not in text:
            raise RuntimeError(f"cannot find Sokoban import anchor in {sokoban_env}")
        sokoban_env.write_text(text.replace(SOKOBAN_ENV_OLD, SOKOBAN_ENV_NEW, 1))
        print(f"patched {sokoban_env}")
    else:
        print(f"already patched {sokoban_env}")
    sokoban_base = args.verl_root / SOKOBAN_BASE_FILE
    text = sokoban_base.read_text()
    if SOKOBAN_BASE_NEW not in text:
        if SOKOBAN_BASE_OLD not in text:
            raise RuntimeError(f"cannot find Sokoban base import anchor in {sokoban_base}")
        text = text.replace(SOKOBAN_BASE_OLD, SOKOBAN_BASE_NEW, 1)
        text = text.replace("tokenizer: AutoTokenizer,", "tokenizer: \"AutoTokenizer\",", 1)
        sokoban_base.write_text(text)
        print(f"patched {sokoban_base}")
    else:
        print(f"already patched {sokoban_base}")

    actor = args.verl_root / ACTOR_FILE
    text = actor.read_text()
    if ACTOR_IMPORT_NEW not in text:
        if ACTOR_IMPORT_OLD not in text:
            raise RuntimeError(f"cannot find actor flash-attn import anchor in {actor}")
        text = text.replace(ACTOR_IMPORT_OLD, ACTOR_IMPORT_NEW, 1)
    if ACTOR_BRANCH_MARKER not in text:
        if ACTOR_BRANCH_OLD not in text:
            raise RuntimeError(f"cannot find actor remove-padding anchor in {actor}")
        text = text.replace(ACTOR_BRANCH_OLD, ACTOR_BRANCH_NEW, 1)
    actor.write_text(text)
    print(f"patched {actor}")

    critic = args.verl_root / CRITIC_FILE
    text = critic.read_text()
    text = text.replace(CRITIC_IMPORT_OLD, "", 1)
    text = text.replace(CRITIC_BLOCK_OLD, "", 1)
    if CRITIC_BROKEN in text:
        text = text.replace(CRITIC_BROKEN, CRITIC_BRANCH_NEW, 1)
    text = text.replace(CRITIC_DUPLICATE_UNPAD, CRITIC_SINGLE_UNPAD, 1)
    text = text.replace(CRITIC_BAD_ASSIGN, CRITIC_GOOD_ASSIGN, 1)
    if CRITIC_BRANCH_MARKER not in text:
        if CRITIC_BRANCH_OLD not in text:
            raise RuntimeError(f"cannot find critic remove-padding anchor in {critic}")
        text = text.replace(CRITIC_BRANCH_OLD, CRITIC_BRANCH_NEW, 1)
    critic.write_text(text)
    print(f"patched {critic}")

    fsdp = args.verl_root / FSDP_FILE
    text = fsdp.read_text()
    if FSDP_ATTN_NEW not in text:
        if FSDP_ATTN_OLD not in text:
            raise RuntimeError(f"cannot find FSDP attention anchor in {fsdp}")
        text = text.replace(FSDP_ATTN_OLD, FSDP_ATTN_NEW)
        fsdp.write_text(text)
        print(f"patched {fsdp}")
    else:
        print(f"already patched {fsdp}")
    text = fsdp.read_text()
    if FSDP_GENERATION_IDS_NEW not in text:
        if FSDP_GENERATION_IDS_OLD not in text:
            raise RuntimeError(f"cannot find generation special-token metadata anchor in {fsdp}")
        text = text.replace(FSDP_GENERATION_IDS_OLD, FSDP_GENERATION_IDS_NEW, 1)
    if VISION_IMPORT_NEW not in text:
        if VISION_IMPORT_OLD not in text:
            raise RuntimeError(f"cannot find Transformers vision import anchor in {fsdp}")
        text = text.replace(VISION_IMPORT_OLD, VISION_IMPORT_NEW, 1)
    text = text.replace(VISION_CHECK_OLD, VISION_CHECK_NEW, 1)
    if LORA_CALL_NEW not in text:
        if LORA_CALL_OLD not in text:
            raise RuntimeError(f"cannot find PEFT LoRA dtype anchor in {fsdp}")
        text = text.replace(LORA_CALL_OLD, LORA_CALL_NEW, 1)
    if FSDP_ORIG_PARAMS_PREVIOUS in text:
        text = text.replace(FSDP_ORIG_PARAMS_PREVIOUS, FSDP_ORIG_PARAMS_NEW, 1)
    if FSDP_ORIG_PARAMS_NEW not in text:
        if FSDP_ORIG_PARAMS_OLD not in text:
            raise RuntimeError(f"cannot find FSDP LoRA orig-params anchor in {fsdp}")
        text = text.replace(FSDP_ORIG_PARAMS_OLD, FSDP_ORIG_PARAMS_NEW, 1)
    if FSDP_ACTOR_OFFLOAD_NEW not in text:
        if FSDP_ACTOR_OFFLOAD_OLD not in text:
            raise RuntimeError(f"cannot find FSDP actor offload anchor in {fsdp}")
        text = text.replace(FSDP_ACTOR_OFFLOAD_OLD, FSDP_ACTOR_OFFLOAD_NEW, 1)
    if HF_WRAP_NEW not in text:
        if HF_WRAP_OLD not in text:
            raise RuntimeError(f"cannot find HF FSDP wrap anchor in {fsdp}")
        text = text.replace(HF_WRAP_OLD, HF_WRAP_NEW, 1)
    if FSDP2_ACTOR_NEW not in text:
        if FSDP2_ACTOR_OLD not in text:
            raise RuntimeError(f"cannot find FSDP2 PEFT root anchor in {fsdp}")
        text = text.replace(FSDP2_ACTOR_OLD, FSDP2_ACTOR_NEW, 1)
    fsdp.write_text(text)
    print(f"patched {fsdp} Transformers compatibility")
    actor_policy = args.verl_root / ACTOR_FILE
    text = actor_policy.read_text()
    if ACTOR_FORWARD_NEW not in text:
        if ACTOR_FORWARD_OLD not in text:
            raise RuntimeError(f"cannot find actor logits retention anchor in {actor_policy}")
        actor_policy.write_text(text.replace(ACTOR_FORWARD_OLD, ACTOR_FORWARD_NEW, 1))
        print(f"patched {actor_policy} response-logit retention")
    else:
        print(f"already patched {actor_policy} response-logit retention")
    monkey = args.verl_root / "verl/models/transformers/monkey_patch.py"
    text = monkey.read_text()
    if TRL_BLOCK_NEW not in text:
        if TRL_BLOCK_OLD not in text:
            raise RuntimeError(f"cannot find TRL compatibility anchor in {monkey}")
        text = text.replace(TRL_BLOCK_OLD, TRL_BLOCK_NEW, 1)
        monkey.write_text(text)
        print(f"patched {monkey} TRL compatibility")
    else:
        print(f"already patched {monkey} TRL compatibility")
    policy = args.verl_root / FSDP_POLICY_FILE
    text = policy.read_text()
    if FSDP_POLICY_NEW not in text:
        if FSDP_POLICY_OLD not in text:
            raise RuntimeError(f"cannot find FSDP policy anchor in {policy}")
        text = text.replace(FSDP_POLICY_OLD, FSDP_POLICY_NEW, 1)
        policy.write_text(text)
        print(f"patched {policy} Transformers layer compatibility")
    else:
        print(f"already patched {policy} Transformers layer compatibility")
    text = policy.read_text()
    if FSDP2_NEW not in text:
        if FSDP2_OLD not in text:
            raise RuntimeError(f"cannot find FSDP2 policy anchor in {policy}")
        text = text.replace(FSDP2_OLD, FSDP2_NEW, 1)
        policy.write_text(text)
        print(f"patched {policy} FSDP2 compatibility")
    else:
        print(f"already patched {policy} FSDP2 compatibility")

    hf_rollout = args.verl_root / HF_ROLLOUT_FILE
    text = hf_rollout.read_text()
    hf_rollout_changed = False
    # Make the compatibility patch idempotent across revisions. Older
    # checkouts may contain the invalid torch-module device block.
    if HF_ROLLOUT_BAD_BLOCK in text:
        text = text.replace(HF_ROLLOUT_BAD_BLOCK, "", 1)
        hf_rollout_changed = True
        print(f"removed stale {hf_rollout} device compatibility block")
    if HF_ROLLOUT_IMPORT_NEW not in text:
        if HF_ROLLOUT_IMPORT_OLD not in text:
            raise RuntimeError(f"cannot find HF rollout device import anchor in {hf_rollout}")
        text = text.replace(HF_ROLLOUT_IMPORT_OLD, HF_ROLLOUT_IMPORT_NEW, 1)
        hf_rollout_changed = True
    if HF_ROLLOUT_NEW not in text:
        if HF_ROLLOUT_OLD not in text:
            raise RuntimeError(f"cannot find HF rollout device anchor in {hf_rollout}")
        text = text.replace(HF_ROLLOUT_OLD, HF_ROLLOUT_NEW, 1)
        hf_rollout_changed = True
        print(f"patched {hf_rollout} device compatibility")
    else:
        print(f"already patched {hf_rollout} device compatibility")
    if hf_rollout_changed:
        hf_rollout.write_text(text)
    text = hf_rollout.read_text()
    if HF_SUMMON_PREVIOUS in text:
        text = text.replace(HF_SUMMON_PREVIOUS, HF_SUMMON_NEW, 1)
        hf_rollout.write_text(text)
        print(f"restored upstream {hf_rollout} nested FSDP summon")
        text = hf_rollout.read_text()
    if HF_SUMMON_NEW not in text:
        if HF_SUMMON_OLD not in text:
            raise RuntimeError(f"cannot find HF FSDP summon anchor in {hf_rollout}")
        hf_rollout.write_text(text.replace(HF_SUMMON_OLD, HF_SUMMON_NEW, 1))
        print(f"patched {hf_rollout} nested FSDP summon")
    else:
        print(f"already patched {hf_rollout} nested FSDP summon")

    text = hf_rollout.read_text()
    if HF_DT_METHOD_MARKER not in text:
        if HF_DT_METHOD_OLD in text:
            text = text.replace(HF_DT_METHOD_OLD, HF_DT_METHOD_MARKER, 1)
            text = text.replace(
                "self._deltatrace_producer = DeltaTraceRolloutProducer(self.module)",
                "self._deltatrace_producer = DeltaTraceRolloutProducer(\n                self.module, eos_token_id=eos_token_id, pad_token_id=pad_token_id\n            )",
                1,
            )
            hf_rollout.write_text(text)
        elif HF_DT_METHOD_ANCHOR in text:
            hf_rollout.write_text(text.replace(HF_DT_METHOD_ANCHOR, HF_DT_METHOD, 1))
        else:
            raise RuntimeError(f"cannot find HF rollout DeltaTrace anchor in {hf_rollout}")
        print(f"patched {hf_rollout} owner DeltaTrace producer")
    else:
        print(f"already patched {hf_rollout} owner DeltaTrace producer")

    fsdp_workers = args.verl_root / FSDP_FILE
    text = fsdp_workers.read_text()
    if FSDP_DT_METHOD_MARKER not in text:
        if FSDP_DT_METHOD_OLD in text:
            text = text.replace(FSDP_DT_METHOD_OLD, FSDP_DT_METHOD_MARKER, 1)
            text = text.replace(
                "return self.rollout.compute_dt_token_advantages(episodes, episode_returns)",
                "return self.rollout.compute_dt_token_advantages(\n            episodes, episode_returns, eos_token_id=eos_token_id, pad_token_id=pad_token_id\n        )",
                1,
            )
            fsdp_workers.write_text(text)
        elif FSDP_DT_METHOD_ANCHOR in text:
            fsdp_workers.write_text(text.replace(FSDP_DT_METHOD_ANCHOR, FSDP_DT_METHOD, 1))
        else:
            raise RuntimeError(f"cannot find worker DeltaTrace anchor in {fsdp_workers}")
        print(f"patched {fsdp_workers} worker DeltaTrace producer")
    else:
        print(f"already patched {fsdp_workers} worker DeltaTrace producer")

    # Transformers 5.13's Qwen3.5 RoPE path can receive position ids created
    # on CPU after HF generation prepares the first step. Under FSDP2 CPU
    # parameter offload, hidden states are already on CUDA, so the upstream
    # rotary matmul must move this derived tensor to x.device as well.
    try:
        transformers = importlib.import_module("transformers")
        qwen35 = Path(next(iter(transformers.__path__))) / "models" / "qwen3_5" / "modeling_qwen3_5.py"
    except (ImportError, StopIteration):
        qwen35 = None
    if qwen35 is not None and qwen35.is_file():
        qwen_text = qwen35.read_text()
        if QWEN35_NEW not in qwen_text:
            if QWEN35_OLD not in qwen_text:
                raise RuntimeError(f"cannot find Qwen3.5 RoPE device anchor in {qwen35}")
            qwen35.write_text(qwen_text.replace(QWEN35_OLD, QWEN35_NEW, 1))
            print(f"patched {qwen35} RoPE device compatibility")
        else:
            print(f"already patched {qwen35} RoPE device compatibility")
        if QWEN35_RMS_NEW not in qwen_text:
            if QWEN35_RMS_PREV in qwen_text:
                qwen35.write_text(qwen_text.replace(QWEN35_RMS_PREV, QWEN35_RMS_NEW, 1))
                print(f"patched {qwen35} FSDP2 RMSNorm device compatibility")
            elif QWEN35_RMS_OLD not in qwen_text:
                raise RuntimeError(f"cannot find Qwen3.5 RMSNorm anchor in {qwen35}")
            else:
                qwen35.write_text(qwen_text.replace(QWEN35_RMS_OLD, QWEN35_RMS_NEW, 1))
                print(f"patched {qwen35} FSDP2 RMSNorm compatibility")
        else:
            print(f"already patched {qwen35} FSDP2 RMSNorm compatibility")
        qwen_text = qwen35.read_text()
        if QWEN35_LM_NEW not in qwen_text:
            if QWEN35_LM_PREV in qwen_text:
                qwen35.write_text(qwen_text.replace(QWEN35_LM_PREV, QWEN35_LM_NEW, 1))
                print(f"patched {qwen35} FSDP2 lm_head device compatibility")
            elif QWEN35_LM_OLD not in qwen_text:
                raise RuntimeError(f"cannot find Qwen3.5 lm_head anchor in {qwen35}")
            else:
                qwen35.write_text(qwen_text.replace(QWEN35_LM_OLD, QWEN35_LM_NEW, 1))
                print(f"patched {qwen35} FSDP2 lm_head compatibility")
        else:
            print(f"already patched {qwen35} FSDP2 lm_head compatibility")

    # Add the owner DeltaTrace token estimator at VERL's existing boundary.
    # This is intentionally a source patch to the pinned upstream tree rather
    # than a second trainer implementation.
    ray_trainer = args.verl_root / RAY_TRAINER_FILE
    text = ray_trainer.read_text()
    if RAY_ADV_ENUM_STALE in text:
        text = text.replace(RAY_ADV_ENUM_STALE, RAY_ADV_ENUM_NEW, 1)
    if RAY_ADV_INSERT_PREVIOUS in text:
        text = text.replace(RAY_ADV_INSERT_PREVIOUS, RAY_ADV_INSERT, 1)
    if RAY_ADV_INSERT_STALE in text:
        text = text.replace(RAY_ADV_INSERT_STALE, RAY_ADV_INSERT, 1)
    if RAY_ADV_ENUM_NEW not in text:
        if RAY_ADV_ENUM_OLD not in text:
            raise RuntimeError(f"cannot find VERL advantage enum anchor in {ray_trainer}")
        text = text.replace(RAY_ADV_ENUM_OLD, RAY_ADV_ENUM_NEW, 1)
    if RAY_ADV_INSERT not in text:
        if RAY_ADV_INSERT_ANCHOR not in text:
            raise RuntimeError(f"cannot find VERL advantage branch anchor in {ray_trainer}")
        text = text.replace(RAY_ADV_INSERT_ANCHOR, RAY_ADV_INSERT, 1)
    if RAY_USE_CRITIC_NEW not in text:
        if RAY_USE_CRITIC_OLD in text:
            text = text.replace(RAY_USE_CRITIC_OLD, RAY_USE_CRITIC_NEW, 1)
        elif RAY_USE_CRITIC_PRISTINE in text:
            text = text.replace(RAY_USE_CRITIC_PRISTINE, RAY_USE_CRITIC_NEW, 1)
        else:
            raise RuntimeError(f"cannot find VERL critic-selection anchor in {ray_trainer}")
    ray_trainer.write_text(text)
    print(f"patched {ray_trainer} DeltaTrace token estimator")

    # Preserve the collector's event and token identities
    # before collate_fn turns trajectory rows into DataProto tensors.
    rollout = args.verl_root / RAY_ROLLOUT_FILE
    text = rollout.read_text()
    if GATHER_BROKEN_IMPORT in text:
        text = text.replace(GATHER_BROKEN_IMPORT, GATHER_GOOD_IMPORT, 1)
    if ROLLOUT_ACTION_COPY_NEW not in text:
        if ROLLOUT_ACTION_COPY_OLD not in text:
            raise RuntimeError(f"cannot find decoded-action preservation anchor in {rollout}")
        text = text.replace(ROLLOUT_ACTION_COPY_OLD, ROLLOUT_ACTION_COPY_NEW, 1)
        print(f"patched {rollout} decoded-action preservation")
    else:
        print(f"already patched {rollout} decoded-action preservation")
    if RAW_PROMPT_KEEP_NEW not in text:
        if RAW_PROMPT_KEEP_OLD not in text:
            raise RuntimeError(f"cannot find raw prompt retention anchor in {rollout}")
        text = text.replace(RAW_PROMPT_KEEP_OLD, RAW_PROMPT_KEEP_NEW, 1)
        print(f"patched {rollout} raw prompt retention")
    else:
        print(f"already patched {rollout} raw prompt retention")
    if GATHER_OLD_MARKER in text:
        start = text.index(GATHER_OLD_MARKER)
        end = text.index("        success_rate = {}", start)
        text = text[:start] + text[end:]
        print(f"removed stale scalar counterfactual collector from {rollout}")
    if GATHER_INSERT_PREVIOUS in text:
        text = text.replace(GATHER_INSERT_PREVIOUS, GATHER_INSERT, 1)
    if ROLLOUT_DT_CALL_PREVIOUS in text:
        text = text.replace(ROLLOUT_DT_CALL_PREVIOUS, ROLLOUT_DT_CALL, 1)
    if ROLLOUT_EVENT_INSERT not in text:
        if ROLLOUT_EVENT_ANCHOR not in text:
            raise RuntimeError(f"cannot find reward event capture anchor in {rollout}")
        text = text.replace(ROLLOUT_EVENT_ANCHOR, ROLLOUT_EVENT_INSERT, 1)
    gather_inserted = False
    if ROLLOUT_STEP_INSERT not in text:
        if ROLLOUT_STEP_ANCHOR not in text:
            raise RuntimeError(f"cannot find rollout step anchor in {rollout}")
        text = text.replace(ROLLOUT_STEP_ANCHOR, ROLLOUT_STEP_INSERT, 1)
    if GATHER_INSERT not in text:
        if GATHER_ANCHOR not in text:
            raise RuntimeError(f"cannot find rollout gather anchor in {rollout}")
        text = text.replace(GATHER_ANCHOR, GATHER_INSERT, 1)
        gather_inserted = True

    if ROLLOUT_DT_CALL_STALE in text:
        text = text.replace(ROLLOUT_DT_CALL_STALE, "", 1)
        print(f"removed stale {rollout} DeltaTrace producer call")

    if ROLLOUT_DT_CALL not in text:
        if ROLLOUT_DT_CALL_ANCHOR not in text:
            raise RuntimeError(f"cannot find rollout DeltaTrace call anchor in {rollout}")
        text = text.replace(ROLLOUT_DT_CALL_ANCHOR, ROLLOUT_DT_CALL, 1)
        print(f"patched {rollout} owner DeltaTrace producer call")
    # Ray workers may expose the adapter directory directly rather than the
    # repository namespace; keep the import at the same thin boundary. Only
    # rewrite the import on the same pass that inserted the fresh block;
    # otherwise a second idempotent patch could nest another try statement.
    if gather_inserted:
        text = text.replace(
            "            from experiments.rl.deltatrace_credit import averaged_traced_credit\n",
            GATHER_GOOD_IMPORT,
            1,
        )
    rollout.write_text(text)
    print(f"patched {rollout} DeltaTrace collector")

    # The observed 32k cap was also being used as a compute length for short
    # prompts. Opt-in owner fixes retain the unmodified external tensor ABI.
    for path, replacements in [
        (args.verl_root / HF_ROLLOUT_FILE, [
            (HF_TRIM_ANCHOR, HF_TRIM_INSERT),
            (HF_TRIM_CALL_OLD, HF_TRIM_CALL_NEW),
            ('        seq = output.sequences', HF_TRIM_RESTORE),
        ]),
        (actor, [(ACTOR_TRIM_ANCHOR, ACTOR_TRIM_INSERT)]),
    ]:
        text = path.read_text()
        if '\nimport os\n' not in text:
            text = text.replace('\nimport torch\n', '\nimport os\nimport torch\n', 1)
        for old, new in replacements:
            if new not in text:
                if old not in text:
                    raise RuntimeError(f'cannot find shared-padding anchor in {path}')
                text = text.replace(old, new, 1)
        path.write_text(text)


if __name__ == "__main__":
    main()

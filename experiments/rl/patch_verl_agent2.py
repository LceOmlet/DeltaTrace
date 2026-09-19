"""Apply compatibility-only patches required by the pinned verl-agent2 tree.

The upstream Sokoban package has a nested package with the same name as its
parent.  Eagerly importing both modules creates a Python 3.12 import cycle
(and can segfault before a traceback).  Princeton's legacy Gym environment
also needs its checker disabled, and Sokoban must not import Transformers in a
Ray worker. The pinned tree also assumes a CUDA flash-attn binary at import
time; on hosts whose glibc cannot load that optional binary, we keep upstream
Hugging Face SDPA as the default and only import flash-attn when VERL's
remove-padding path is explicitly enabled. These edits do not change
environment dynamics or trainer math.
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
FSDP_ORIG_PARAMS_NEW = "                use_orig_params=self._is_lora,"
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
HF_SUMMON_NEW = "            param_ctx = FSDP.summon_full_params(self.module, writeback=False, recurse=True)"

QWEN35_OLD = "        position_ids_expanded = position_ids[:, :, None, :].float()  # shape (3, bs, 1, positions)"
QWEN35_NEW = "        position_ids_expanded = position_ids[:, :, None, :].float().to(x.device)  # shape (3, bs, 1, positions)"
QWEN35_RMS_OLD = "        output = output * (1.0 + self.weight.float())"
QWEN35_RMS_NEW = "        weight = self.weight.to_local() if hasattr(self.weight, \"to_local\") else self.weight\n        output = output * (1.0 + weight.float())"
HF_WRAP_OLD = '        if self._is_rollout and self.config.rollout.name == "hf":\n            # TODO(zhangchi.usc1992, shengguangming) fix me. Current, auto_wrap_policy causes HFRollout to hang in Gemma\n            auto_wrap_policy = None'
HF_WRAP_NEW = '        if self._is_rollout and self.config.rollout.name == "hf" and os.getenv("VERL_ENABLE_HF_FSDP_WRAP", "0") != "1":\n            # Keep upstream HF rollout\'s conservative default; long-context\n            # single-GPU runs can opt into layer wrapping explicitly.\n            auto_wrap_policy = None'


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("verl_root", type=Path)
    args = parser.parse_args()
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
    if VISION_IMPORT_NEW not in text:
        if VISION_IMPORT_OLD not in text:
            raise RuntimeError(f"cannot find Transformers vision import anchor in {fsdp}")
        text = text.replace(VISION_IMPORT_OLD, VISION_IMPORT_NEW, 1)
    text = text.replace(VISION_CHECK_OLD, VISION_CHECK_NEW, 1)
    if LORA_CALL_NEW not in text:
        if LORA_CALL_OLD not in text:
            raise RuntimeError(f"cannot find PEFT LoRA dtype anchor in {fsdp}")
        text = text.replace(LORA_CALL_OLD, LORA_CALL_NEW, 1)
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
    if HF_SUMMON_NEW not in text:
        if HF_SUMMON_OLD not in text:
            raise RuntimeError(f"cannot find HF FSDP summon anchor in {hf_rollout}")
        hf_rollout.write_text(text.replace(HF_SUMMON_OLD, HF_SUMMON_NEW, 1))
        print(f"patched {hf_rollout} nested FSDP summon")
    else:
        print(f"already patched {hf_rollout} nested FSDP summon")

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
            if QWEN35_RMS_OLD not in qwen_text:
                raise RuntimeError(f"cannot find Qwen3.5 RMSNorm anchor in {qwen35}")
            qwen35.write_text(qwen35.read_text().replace(QWEN35_RMS_OLD, QWEN35_RMS_NEW, 1))
            print(f"patched {qwen35} FSDP2 RMSNorm compatibility")
        else:
            print(f"already patched {qwen35} FSDP2 RMSNorm compatibility")


if __name__ == "__main__":
    main()

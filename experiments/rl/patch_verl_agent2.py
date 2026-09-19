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
    fsdp.write_text(text)
    print(f"patched {fsdp} Transformers compatibility")
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


if __name__ == "__main__":
    main()

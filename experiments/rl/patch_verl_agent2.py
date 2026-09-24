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
import subprocess
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
FSDP_QWEN_CHAT_EOS_ANCHOR = '        self.generation_config = get_generation_config(local_path, trust_remote_code=trust_remote_code)\n'
FSDP_QWEN_CHAT_EOS = FSDP_QWEN_CHAT_EOS_ANCHOR + '''        # Qwen3.5-9B has no generation_config.json: its model fallback ends
        # documents, but the supplied chat tokenizer ends turns with im_end.
        # Keep explicit generation configs and other model families unchanged.
        if getattr(actor_model_config, "model_type", None) == "qwen3_5" and getattr(self.generation_config, "_from_model_config", False):
            model_eos = self.generation_config.eos_token_id
            model_eos = model_eos if isinstance(model_eos, list) else ([] if model_eos is None else [model_eos])
            self.generation_config.eos_token_id = list(dict.fromkeys([self.tokenizer.eos_token_id, *model_eos]))
'''
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
ROLLOUT_UTILS_FILE = "agent_system/multi_turn_rollout/utils.py"
ROW_CALL = "            batch_list: list[dict] = to_list_of_dict(batch)"
COMPACT_ROW_CALL = ('            batch_list: list[dict] = to_list_of_dict(\n'
                    '                batch, clone_tensors=str(self.config.algorithm.adv_estimator) == "deltatrace"\n'
                    '            )')


def patch_rollout_row_storage(text: str) -> str:
    """Keep row storage bounded when factual rows cross the DT Ray RPC.

    The owner returns tensor views into a whole rollout batch. Python/Ray's
    tensor pickle serializes that entire storage for every separate row view.
    Clone only at the existing row conversion boundary, opt-in for DT; leave
    the owner's default view behavior and all tensor values unchanged.
    """
    old = "def to_list_of_dict(batch: DataProto) -> list[dict]:"
    new = "def to_list_of_dict(batch: DataProto, *, clone_tensors: bool = False) -> list[dict]:"
    if new in text:
        return text
    assignment = "            save_dict[key] = val[bs]"
    if text.count(old) != 1 or text.count(assignment) != 2:
        raise RuntimeError("cannot find pinned rollout row conversion anchors")
    text = text.replace(old, new, 1)
    return text.replace(assignment,
                        "            save_dict[key] = val[bs].clone() if clone_tensors else val[bs]", 1)


def patch_rollout_phase_logs(text: str) -> str:
    """Expose progress at existing owner boundaries, without changing work."""
    marker = '                print(f"[DT rollout] phase=generation_start'
    if marker in text:
        return text
    start = '            batch_output_padded = actor_rollout_wg.generate_sequences(batch_input_padded)'
    end = '            batch_output = unpad_dataproto(batch_output_padded, pad_size=pad_size)'
    rpc = '            dt_values = actor_rollout_wg.compute_dt_token_advantages('
    if any(text.count(anchor) != 1 for anchor in (start, end, rpc)):
        raise RuntimeError('cannot find pinned rollout phase log boundaries')
    text = text.replace(start, '''            if str(self.config.algorithm.adv_estimator) == "deltatrace":
                from time import perf_counter
                from resource import getrusage, RUSAGE_SELF
                _dt_round_started = perf_counter()
                print(f"[DT rollout] phase=generation_start step={_step + 1}/{self.config.env.max_steps} "
                      f"active={int(active_masks.sum())}/{batch_size} "
                      f"peak_rss_gib={getrusage(RUSAGE_SELF).ru_maxrss / 2**20:.3f}", flush=True)
''' + start, 1)
    text = text.replace(end, end + '''
            if str(self.config.algorithm.adv_estimator) == "deltatrace":
                _dt_generated = int(batch_output.batch["attention_mask"][
                    :, -batch_output.batch["responses"].shape[-1]:].sum().item())
                _dt_seconds = perf_counter() - _dt_round_started
                print(f"[DT rollout] phase=generation_end step={_step + 1} "
                      f"seconds={_dt_seconds:.3f} generated_tokens={_dt_generated} "
                      f"tokens_per_second={_dt_generated / _dt_seconds:.3f} "
                      f"peak_rss_gib={getrusage(RUSAGE_SELF).ru_maxrss / 2**20:.3f}", flush=True)
''', 1)
    text = text.replace(rpc, '''            from time import perf_counter
            from resource import getrusage, RUSAGE_SELF
            _dt_rpc_started = perf_counter()
            print(f"[DT rollout] phase=dt_rpc_start episodes={len(total_batch_list)} "
                  f"rows={sum(map(len, total_batch_list))} "
                  f"peak_rss_gib={getrusage(RUSAGE_SELF).ru_maxrss / 2**20:.3f}", flush=True)
''' + rpc, 1)
    end_rpc = '            if isinstance(dt_values, list) and len(dt_values) == 1:'
    text = text.replace(end_rpc, '''            print(f"[DT rollout] phase=dt_rpc_end seconds={perf_counter() - _dt_rpc_started:.3f} "
                  f"peak_rss_gib={getrusage(RUSAGE_SELF).ru_maxrss / 2**20:.3f}", flush=True)
''' + end_rpc, 1)
    return text


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
HF_TRIM_PREVIOUS = '''        # Trim only shared left padding for native generation; restore exact
        # DataProto width below so collector IDs/masks keep their original ABI.
        generation_left_trim = 0
        if os.getenv("VERL_TRIM_SHARED_PADDING", "0") == "1":
            generation_left_trim = int(attention_mask.long().argmax(-1).min().item())
        self.module.eval()
        param_ctx = contextlib.nullcontext()'''
HF_TRIM_INSERT = HF_TRIM_PREVIOUS.replace(
    '            generation_left_trim = int(attention_mask.long().argmax(-1).min().item())',
    '''            generation_left_trim = int(attention_mask.long().argmax(-1).min().item())
            if getattr(getattr(self.module, "config", None), "model_type", None) in ("qwen3_5", "qwen3_5_text"):
                # Pinned FLA uses 64-token chunks. Keep its original chunk grid;
                # moving real tokens across chunk boundaries changes BF16 results.
                generation_left_trim -= generation_left_trim % 64''')
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

HF_ACTIVE_MARKER = '        # Skip finished trajectories using the collector-owned mask.'


def patch_hf_active_rows(text: str) -> str:
    """Compact only inside the owner's existing microbatch; retain row ABI."""
    if HF_ACTIVE_MARKER in text:
        return text
    start = text.index('        with param_ctx, torch.autocast(')
    end = text.index('        generated_batch_size = seq.size(0)', start)
    block = text[start:end]
    if HF_TRIM_CALL_NEW not in block or HF_TRIM_RESTORE not in block:
        raise RuntimeError('cannot find pinned HF generation/restore block')
    # Reuse the original generate call, options, FSDP context and trim restore.
    block = block.replace('input_ids=idx[:,', 'input_ids=generation_idx[:,')
    block = block.replace('attention_mask=attention_mask[:,', 'attention_mask=generation_mask[:,')
    block = block.replace('position_ids=position_ids[...,', 'position_ids=generation_positions[...,')
    block = block.replace('padding = idx[:,', 'padding = generation_idx[:,')
    block = block.replace('seq.size(0) // idx.size(0)', 'seq.size(0) // generation_idx.size(0)')
    indented = ''.join('    ' + line if line.strip() else line for line in block.splitlines(keepends=True))
    before = '''        # Skip finished trajectories using the collector-owned mask.
        # No mask means the unchanged owner path. Compact within (not across)
        # original microbatches, so the configured maximum batch size stays put.
        active_mask = prompts.non_tensor_batch.get("rollout_active_mask")
        active_rows = None
        generation_idx, generation_mask, generation_positions = idx, attention_mask, position_ids
        copies = kwargs.get("num_return_sequences", 1)
        if active_mask is not None and not all(active_mask):
            active_rows = torch.as_tensor(active_mask, dtype=torch.bool).nonzero().flatten().to(idx.device)
            generation_idx = idx.index_select(0, active_rows)
            generation_mask = attention_mask.index_select(0, active_rows)
            generation_positions = position_ids.index_select(0, active_rows)
        if generation_idx.size(0) == 0:
            seq = idx.repeat_interleave(copies, dim=0)
        else:
'''
    after = '''            if active_rows is not None:
                full_seq = idx.new_full((idx.size(0) * copies, seq.size(1)), pad_token_id)
                full_seq[:, :prompt_length] = idx.repeat_interleave(copies, dim=0)
                output_rows = (active_rows[:, None] * copies + torch.arange(copies, device=idx.device)).flatten()
                full_seq.index_copy_(0, output_rows, seq)
                seq = full_seq
'''
    text = text[:start] + before + indented + after + text[end:]
    mask_line = '        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)'
    text = text.replace(mask_line, '''        if active_rows is not None:
            response_attention_mask *= torch.as_tensor(
                active_mask, dtype=response_attention_mask.dtype, device=response_attention_mask.device
            ).repeat_interleave(copies)[:, None]
''' + mask_line, 1)
    return text


VLLM_ROLLOUT_FILE = "verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py"
VLLM_ACTIVE_MARKER = "        # Only live collector rows become vLLM requests."


def patch_vllm_active_rows(text: str) -> str:
    """Use the collector mask at the pinned vLLM request boundary, preserving rows."""
    if VLLM_ACTIVE_MARKER in text:
        return text
    anchor = "        # users can customize different sampling_params at different run\n"
    select = '''        # Only live collector rows become vLLM requests.
        # Sampling, LoRA handling and generation remain owned by vLLM.
        active_mask = non_tensor_batch.pop("rollout_active_mask", None)
        active_rows = None
        if active_mask is not None and not all(active_mask):
            active_rows = np.flatnonzero(active_mask).tolist()
            vllm_inputs = [vllm_inputs[row] for row in active_rows]
            if lora_requests is not None:
                lora_requests = [lora_requests[row] for row in active_rows]

'''
    call = '''            outputs = self.inference_engine.generate(
                prompts=vllm_inputs,  # because we have already convert it to prompt token id
                sampling_params=self.sampling_params,
                lora_request=lora_requests,
                use_tqdm=False,
            )'''
    restore_anchor = "            response = pad_2d_list_to_length(response, self.pad_token_id, max_length=self.config.response_length).to(idx.device)\n"
    restore = '''            if active_rows is not None:
                # Restore the owner's original row/sample order. Finished rows
                # have no generated tokens and remain excluded by active_masks.
                copies = self.sampling_params.n
                output_rows = [row * copies + sample for row in active_rows for sample in range(copies)]
                full_response = [[] for _ in range(batch_size * copies)]
                full_log_probs = [[] for _ in range(batch_size * copies)]
                for src, dst in enumerate(output_rows):
                    full_response[dst] = response[src]
                    full_log_probs[dst] = rollout_log_probs[src]
                response, rollout_log_probs = full_response, full_log_probs

'''
    mask_anchor = "        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)\n"
    mask = '''        if active_rows is not None:
            response_attention_mask *= torch.as_tensor(
                active_mask, dtype=response_attention_mask.dtype, device=response_attention_mask.device
            ).repeat_interleave(copies)[:, None]
'''
    for old, new in [(anchor, select + anchor), (call, call + " if vllm_inputs else []"),
                     (restore_anchor, restore + restore_anchor), (mask_anchor, mask + mask_anchor)]:
        if text.count(old) != 1:
            raise RuntimeError("cannot find unique pinned vLLM active-row anchor")
        text = text.replace(old, new, 1)
    return text


def patch_actor_response_head(text: str) -> str:
    """Select live response columns through Qwen's existing head interface."""
    marker = '                # Select response head rows; decoder inputs and kernels stay unchanged.'
    if marker in text:
        return text
    anchor = '''                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,'''
    select = '''                # Select response head rows; decoder inputs and kernels stay unchanged.
                head_response_length = response_length
                head_positions = response_length + 1
                if (not multi_modal_inputs and not self.use_fused_kernels
                        and os.getenv("VERL_TRIM_RESPONSE_HEAD", "0") == "1"
                        and getattr(getattr(self.actor_module, "config", None), "model_type", None)
                        in ("qwen3_5", "qwen3_5_text")):
                    columns = torch.arange(1, response_length + 1, device=attention_mask.device)
                    head_response_length = max(1, int((attention_mask[:, -response_length:] * columns).max().item()))
                    if head_response_length < response_length:
                        start = input_ids.shape[-1] - response_length - 1
                        head_positions = torch.arange(start, start + head_response_length + 1, device=input_ids.device)
'''
    old_head = '''                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)'''
    new_head = '''                    logits = logits[:, -head_response_length - 1 : -1, :]
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"][:, :head_response_length])
                    if calculate_entropy:
                        entropy = verl_F.entropy_from_logits(logits)
                    if head_response_length < response_length:
                        # The owner's PPO tensors keep their original columns/masks.
                        log_probs = torch.nn.functional.pad(log_probs, (0, response_length - head_response_length))
                        if calculate_entropy:
                            entropy = torch.nn.functional.pad(entropy, (0, response_length - head_response_length))'''
    replacements = [(anchor, select + anchor),
                    ('                    logits_to_keep=response_length + 1,',
                     '                    logits_to_keep=head_positions,'), (old_head, new_head)]
    for old, new in replacements:
        if text.count(old) != 1:
            raise RuntimeError('cannot find unique pinned actor response-head anchor')
        text = text.replace(old, new, 1)
    return text


ACTOR_TRIM_ANCHOR = '            else:  # not using rmpad and no ulysses sp\n                extra_args = {}'
ACTOR_TRIM_PREVIOUS = '''            else:  # not using rmpad and no ulysses sp
                # Preserve the response columns and original tensors. Only
                # columns masked out for every example leave the model input.
                if not multi_modal_inputs and os.getenv("VERL_TRIM_SHARED_PADDING", "0") == "1":
                    left_trim = int(attention_mask.long().argmax(-1).min().item())
                    input_ids = input_ids[:, left_trim:]
                    attention_mask = attention_mask[:, left_trim:]
                    position_ids = position_ids[..., left_trim:]
                extra_args = {}'''
ACTOR_TRIM_INSERT = ACTOR_TRIM_PREVIOUS.replace(
    '                    left_trim = int(attention_mask.long().argmax(-1).min().item())',
    '''                    left_trim = int(attention_mask.long().argmax(-1).min().item())
                    if getattr(getattr(self.actor_module, "config", None), "model_type", None) in ("qwen3_5", "qwen3_5_text"):
                        # Match the pinned FLA 64-token chunk grid in forward/backward.
                        left_trim -= left_trim % 64''')
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
FSDP_DT_METHOD_PREVIOUS = FSDP_DT_METHOD
FSDP_DT_METHOD = FSDP_DT_METHOD.replace(
    '        """Delegate DT attribution to the existing HF rollout model."""\n',
    '''        """Run DT on the existing actor after the owner's rollout exit."""
        if self.config.rollout.name == "vllm":
            from deltatrace_rollout import DeltaTraceRolloutProducer

            # FSDPVLLMShardingManager.__exit__ already sleeps the engine.
            # Reuse the same actor and the same memory owner as compute_log_prob.
            if self._is_offload_param:
                load_fsdp_model_to_gpu(self.actor_module_fsdp)
            try:
                if not hasattr(self, "_deltatrace_producer"):
                    self._deltatrace_producer = DeltaTraceRolloutProducer(
                        self.actor_module_fsdp, eos_token_id=eos_token_id, pad_token_id=pad_token_id
                    )
                return self._deltatrace_producer.attribute_episodes(episodes, episode_returns)
            finally:
                if self._is_offload_param:
                    offload_fsdp_model_to_cpu(self.actor_module_fsdp)
''',
)

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

# Preserve official VERL's FSDP2 root at actor_module, including PEFT. PEFT's
# tuner calls base_model.model.forward directly, bypassing an inner root's
# hooks. Restore the official block in checkouts carrying the former patch.
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


def patch_vllm_peft_owner(text: str) -> str:
    # Backport VERL v0.7.0's PEFT owner lookup (fsdp_utils.collect_lora_params
    # and fsdp_workers.rollout_mode). FSDP2 has no _fsdp_wrapped_module.
    anchor = '    def __enter__(self):\n'
    replacement = anchor + '        peft_model = getattr(self.module, "_fsdp_wrapped_module", self.module)\n'
    if replacement not in text:
        if text.count(anchor) != 1:
            raise RuntimeError('cannot find native vLLM sharding entry')
        text = text.replace(anchor, replacement, 1)
        text = text.replace('self.module._fsdp_wrapped_module', 'peft_model')
    # This actor keeps the FSDP2 root inside PEFT. The old outer-wrapper
    # classification misses it; materialize its actual DTensors with the same
    # full_tensor conversion already used by the official FSDP branch.
    anchor = '            params = __collect_lora_params()\n'
    replacement = anchor + '''            params = {
                name: param.full_tensor().detach().cpu() if isinstance(param, DTensor) else param
                for name, param in params.items()
            }
'''
    if replacement not in text:
        if text.count(anchor) != 1:
            raise RuntimeError('cannot find native LoRA tensor collection')
        text = text.replace(anchor, replacement, 1)
    anchor = '        log_gpu_memory_usage("After state_dict() in sharding manager memory", logger=logger)\n'
    replacement = '''        # Transformers' text-only Qwen3.5 actor exposes model.layers; the
        # checkpoint and vLLM ConditionalGeneration expose model.language_model.
        # Restore only that ABI prefix before the native vLLM weights mapper.
        if (peft_config is not None
                and getattr(peft_model.config, "model_type", None) == "qwen3_5_text"
                and getattr(self.model_config, "model_type", None) == "qwen3_5"):
            source_prefix = "base_model.model.model."
            target_prefix = "base_model.model.model.language_model."
            params = {
                target_prefix + name[len(source_prefix):] if name.startswith(source_prefix) else name: value
                for name, value in params.items()
            }
''' + anchor
    if replacement not in text:
        if text.count(anchor) != 1:
            raise RuntimeError('cannot find native LoRA checkpoint naming boundary')
        text = text.replace(anchor, replacement, 1)
    # Use the owner's allocator switch at the same rollout/trainer boundary
    # as VERL v0.7.0. vLLM's CuMem pool cannot use expandable segments; DT/PPO
    # retain the allocator that passed the actual 32k capacity test.
    old = 'from verl.utils.device import get_torch_device\n'
    new = 'from verl.utils.device import get_torch_device, set_expandable_segments\n'
    if old in text:
        text = text.replace(old, new, 1)
    anchor = '            if "tags" in inspect.signature(self.inference_engine.wake_up).parameters:\n'
    replacement = '            set_expandable_segments(False)\n' + anchor
    if replacement not in text:
        if text.count(anchor) != 2:
            raise RuntimeError('cannot find native vLLM weight wake boundary')
        text = text.replace(anchor, replacement, 1)
    anchor = '        self.module.train()\n'
    replacement = anchor + '        set_expandable_segments(True)\n'
    if replacement not in text:
        if text.count(anchor) != 1:
            raise RuntimeError('cannot find native vLLM trainer return boundary')
        text = text.replace(anchor, replacement, 1)
    return text


def patch_vllm_lazy_moe_imports(text: str) -> str:
    # The legacy MoE workaround eagerly imports NVIDIA Qwen3-Next, registering
    # gdn_attention_core before the installed MetaX registry loads its own op.
    # Keep that owner workaround intact, but import its models only when it is
    # actually called. LoRA sync returns before this full-weight MoE path.
    marker = '    # Legacy MoE imports belong to the full-weight loader only.\n'
    if marker in text:
        return text
    start = text.index('# To support different vLLM versions,')
    end = text.index('from typing import List', start)
    imports = text[start:end].rstrip()
    text = text[:start] + text[end:]
    anchor = 'def patch_vllm_moe_model_weight_loader(model):\n'
    if text.count(anchor) != 1:
        raise RuntimeError('cannot find legacy vLLM MoE loader owner')
    block = '\n'.join('    ' + line if line else '' for line in imports.splitlines())
    return text.replace(anchor, anchor + marker + block + '\n\n', 1)


def patch_fsdp2_offload(text: str) -> str:
    # Our PEFT actor keeps its actual Transformers root fully_sharded. Pass
    # that exact owner to the existing memory lifecycle; don't globally alter
    # fsdp_version or change how PPO/checkpoint callers classify their wrapper.
    for signature in (
        'def offload_fsdp_model_to_cpu(model: FSDP, empty_cache: bool = True):',
        'def load_fsdp_model_to_gpu(model: FSDP):',
    ):
        old = signature + '\n    if fsdp_version(model) == 2:'
        new = signature + '''
    from peft import PeftModel
    if isinstance(model, PeftModel) and isinstance(model.get_base_model(), FSDPModule):
        model = model.get_base_model()
    if fsdp_version(model) == 2:'''
        if new not in text:
            if old not in text:
                raise RuntimeError(f'cannot find FSDP2 offload owner boundary: {signature}')
            text = text.replace(old, new, 1)
    # Upstream VERL v0.7.0 uses Module.cpu()/to() so buffers move together with
    # parameters. Backport these owner bodies, not an independent offloader.
    old = '''def offload_fsdp2_model_to_cpu(model, empty_cache: bool = True):
    for param in model.parameters():
        param.data = param.data.to(torch.device("cpu"), non_blocking=True)'''
    new = '''def offload_fsdp2_model_to_cpu(model, empty_cache: bool = True):
    model.cpu()'''
    if old in text:
        text = text.replace(old, new, 1)
    elif new not in text:
        raise RuntimeError('cannot find FSDP2 CPU offload owner body')
    old = '''def load_fsdp2_model_to_gpu(model):
    device = torch.cuda.current_device()
    for param in model.parameters():
        param.data = param.data.to(device, non_blocking=True)'''
    new = '''def load_fsdp2_model_to_gpu(model):
    device = get_torch_device().current_device()
    model.to(device)'''
    if old in text:
        text = text.replace(old, new, 1)
    elif new not in text:
        raise RuntimeError('cannot find FSDP2 GPU load owner body')
    return text

QWEN35_OLD = "        position_ids_expanded = position_ids[:, :, None, :].float()  # shape (3, bs, 1, positions)"
QWEN35_NEW = "        position_ids_expanded = position_ids[:, :, None, :].float().to(x.device)  # shape (3, bs, 1, positions)"
QWEN35_RMS_OLD = "        output = output * (1.0 + self.weight.float())"
QWEN35_RMS_PREV = "        weight = self.weight.to_local() if hasattr(self.weight, \"to_local\") else self.weight\n        output = output * (1.0 + weight.float())"
QWEN35_RMS_NEW = "        weight = self.weight.to_local() if hasattr(self.weight, \"to_local\") else self.weight\n        output = output * (1.0 + weight.float().to(output.device))"
QWEN35_LM_OLD = "        logits = self.lm_head(hidden_states[:, slice_indices, :])"
QWEN35_LM_PREV = "        lm_weight = self.lm_head.weight.to_local() if hasattr(self.lm_head.weight, \"to_local\") else self.lm_head.weight\n        lm_bias = self.lm_head.bias.to_local() if getattr(self.lm_head, \"bias\", None) is not None and hasattr(self.lm_head.bias, \"to_local\") else getattr(self.lm_head, \"bias\", None)\n        logits = F.linear(hidden_states[:, slice_indices, :], lm_weight, lm_bias)"
QWEN35_LM_NEW = "        lm_weight = self.lm_head.weight.to_local() if hasattr(self.lm_head.weight, \"to_local\") else self.lm_head.weight\n        lm_bias = self.lm_head.bias.to_local() if getattr(self.lm_head, \"bias\", None) is not None and hasattr(self.lm_head.bias, \"to_local\") else getattr(self.lm_head, \"bias\", None)\n        lm_weight = lm_weight.to(hidden_states.device)\n        lm_bias = lm_bias.to(hidden_states.device) if lm_bias is not None else None\n        logits = F.linear(hidden_states[:, slice_indices, :], lm_weight, lm_bias)"
# Backport only the owner's padding-mask fix from Transformers 59eed1a6:
# create_recurrent_attention_mask already supplies an aligned local 2D mask
# or None for cached forwards. Batch size 1 must not skip a supplied mask.
QWEN35_PADDING_OLD = "    # NOTE: attention mask is a 2D boolean tensor\n    if attention_mask is not None and attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:"
QWEN35_PADDING_NEW = "    # NOTE: attention mask is a 2D boolean tensor\n    if attention_mask is not None:"


def patch_qwen35_padding_mask(text: str) -> str:
    if QWEN35_PADDING_NEW in text:
        return text
    if text.count(QWEN35_PADDING_OLD) != 1:
        raise RuntimeError("cannot find the Qwen3.5 batch-one padding-mask owner anchor")
    return text.replace(QWEN35_PADDING_OLD, QWEN35_PADDING_NEW, 1)


HF_WRAP_OLD = '        if self._is_rollout and self.config.rollout.name == "hf":\n            # TODO(zhangchi.usc1992, shengguangming) fix me. Current, auto_wrap_policy causes HFRollout to hang in Gemma\n            auto_wrap_policy = None'
HF_WRAP_NEW = '        if self._is_rollout and self.config.rollout.name == "hf" and os.getenv("VERL_ENABLE_HF_FSDP_WRAP", "0") != "1":\n            # Keep upstream HF rollout\'s conservative default; long-context\n            # single-GPU runs can opt into layer wrapping explicitly.\n            auto_wrap_policy = None'


def patch_context_budget(text: str) -> str:
    """End an over-budget AppWorld episode at the owner's rollout boundary.

    Empty transport rows are never generated, executed or trained. The original
    chat and already executed action/reward rows are retained, without truncation.
    The capability is opt-in; the default error behavior remains unchanged.
    """
    if 'context_budget_exceeded' in text:
        return text
    raw = '        raw_prompt_ids = self.tokenizer.encode(prompt_with_vision_tokens, add_special_tokens=False)\n'
    if raw not in text:
        raw = '        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)\n'
    anchor = '        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data('
    if text.count(raw) != 1 or text.count(anchor) != 1:
        raise RuntimeError('cannot find owner prompt-tokenization boundary')
    text = text.replace(raw, '', 1)
    text = text.replace(anchor, raw + '''        context_budget_exceeded = (
            not is_multi_modal
            and self.config.env.get("context_budget_action", "error") == "end_episode"
            and len(raw_prompt_ids) > self.config.data.max_prompt_length
        )
        if self.config.env.get("context_budget_action", "error") == "end_episode":
            row_dict["context_budget_tokens"] = len(raw_prompt_ids) if context_budget_exceeded else 0
        # An inert transport row, not a truncated history. The collector masks it
        # before generation and before the environment receives any action.
        if context_budget_exceeded:
            raw_prompt_ids = []
        prompt_for_model = "" if context_budget_exceeded else prompt_with_chat_template

''' + anchor, 1)
    text = text.replace('verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,',
                        'verl_F.tokenize_and_postprocess_data(prompt=prompt_for_model,', 1)
    anchor = '            batch = self.preprocess_batch(gen_batch=gen_batch, obs=obs, messages=message_histories)\n'
    if anchor not in text:
        anchor = '            batch = self.preprocess_batch(gen_batch=gen_batch, obs=obs)\n'
    insertion = '''
            if self.config.env.get("context_budget_action", "error") == "end_episode":
                exceeded = batch.non_tensor_batch.pop("context_budget_tokens")
                newly_stopped = active_masks & (exceeded > 0)
                if newly_stopped.any():
                    print(f"[rollout] phase=context_budget_end step={_step + 1} "
                          f"indices={np.flatnonzero(newly_stopped).tolist()} "
                          f"prompt_tokens={exceeded[newly_stopped].tolist()} "
                          f"prompt_cap={self.config.data.max_prompt_length}", flush=True)
                is_done = np.logical_or(is_done, newly_stopped)
                active_masks = np.logical_not(is_done)
                if is_done.all():
                    break
'''
    if text.count(anchor) != 1:
        raise RuntimeError('cannot find owner rollout preprocessing boundary')
    text = text.replace(anchor, anchor + insertion, 1)
    anchor = '            next_obs, rewards, dones, infos = envs.step(text_actions)\n'
    if text.count(anchor) != 1:
        raise RuntimeError('cannot find owner rollout environment step')
    return text.replace(anchor, '''            if self.config.env.get("context_budget_action", "error") == "end_episode":
                next_obs, rewards, dones, infos = envs.step(text_actions, active_masks=active_masks)
            else:
                next_obs, rewards, dones, infos = envs.step(text_actions)
''', 1)


def patch_appworld_active_steps(text: str, *, manager: bool = False) -> str:
    """Keep inactive AppWorld rows out of the existing environment RPCs."""
    if manager:
        start = text.index('class AppWorldEnvironmentManager(')
        end = text.find('\nclass ', start + 1)
        end = len(text) if end < 0 else end
        section = text[start:end]
        if 'def step(self, text_actions: List[str], active_masks=None):' in section:
            return text
        section = section.replace('def step(self, text_actions: List[str]):',
                                  'def step(self, text_actions: List[str], active_masks=None):', 1)
        section = section.replace('        text_obs, rewards, dones, infos = self.envs.step(actions)',
                                  '''        if active_masks is None:
            text_obs, rewards, dones, infos = self.envs.step(actions)
        else:
            text_obs, rewards, dones, infos = self.envs.step(actions, active_masks=active_masks)''', 1)
        section = section.replace("        self.memory.store({'text_obs': text_obs, 'action': actions})",
                                  "        self.memory.store({'text_obs': text_obs, 'action': actions}, active_masks=active_masks)", 1)
        return text[:start] + section + text[end:]
    if 'def step(self, actions, active_masks=None):' in text:
        return text
    text = text.replace('    def step(self, actions):', '    def step(self, actions, active_masks=None):', 1)
    old = '''        # Send step commands to all workers
        futures = []
        for i, worker in enumerate(self.workers):
            future = worker.step.remote(actions[i])
            futures.append(future)

        # Collect results
        results = ray.get(futures)'''
    new = '''        # Preserve the default all-worker path. With a mask, only real
        # active actions reach AppWorld; inactive slots reuse their last actual
        # observation/info and carry zero reward outside the collector mask.
        indices = range(self.num_processes) if active_masks is None else np.flatnonzero(active_masks)
        futures = [self.workers[i].step.remote(actions[i]) for i in indices]
        received = ray.get(futures)
        if active_masks is None:
            results = received
        else:
            results = [(obs, 0.0, True, dict(info)) for obs, info in self._last_observations]
            for i, result in zip(indices, received):
                results[i] = result
        self._last_observations = [(obs, dict(info)) for obs, _, _, info in results]'''
    if text.count(old) != 1:
        raise RuntimeError('cannot find AppWorld owner step RPC boundary')
    text = text.replace(old, new, 1)
    start = text.index('    def reset(self):', text.index('class AppWorldEnvs:'))
    anchor = '        results = ray.get(futures)\n'
    loc = text.index(anchor, start) + len(anchor)
    return text[:loc] + '        self._last_observations = [(obs, dict(info)) for obs, info in results]\n' + text[loc:]


def patch_memory_active_steps(text: str) -> str:
    """Store only executed actions in the owner's per-environment memory."""
    start = text.index('class SimpleMemory(')
    end = text.index('\nclass ', start + 1)
    section = text[start:end]
    if 'def store(self, record: Dict[str, List[Any]], active_masks=None):' in section:
        return text
    section = section.replace('def store(self, record: Dict[str, List[Any]]):',
                              'def store(self, record: Dict[str, List[Any]], active_masks=None):', 1)
    section = section.replace('        for env_idx in range(self.batch_size):\n',
                              '        for env_idx in range(self.batch_size):\n            if active_masks is not None and not active_masks[env_idx]:\n                continue\n', 1)
    return text[:start] + section + text[end:]


def patch_rollout_owner(verl_root: Path) -> None:
    # Preserve the collector's event and token identities
    # before collate_fn turns trajectory rows into DataProto tensors.
    rollout = verl_root / RAY_ROLLOUT_FILE
    text = rollout.read_text()
    if GATHER_BROKEN_IMPORT in text:
        text = text.replace(GATHER_BROKEN_IMPORT, GATHER_GOOD_IMPORT, 1)
    # The author collector does not reuse decoded actions after envs.step.
    # Remove the obsolete copy from the former full-chat fork integration.
    text = text.replace('            env_actions = list(text_actions)\n', '')
    text = text.replace('envs.step(env_actions', 'envs.step(text_actions')
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
    if ROLLOUT_EVENT_INSERT not in text and ROLLOUT_EVENT_INSERT.replace(ROW_CALL, COMPACT_ROW_CALL) not in text:
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

    if ROLLOUT_DT_CALL not in text and 'print(f"[DT rollout] phase=dt_rpc_start' not in text:
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
    row_call, compact_row_call = ROW_CALL, COMPACT_ROW_CALL
    if compact_row_call not in text:
        if text.count(row_call) != 1:
            raise RuntimeError("cannot find factual rollout row conversion call")
        text = text.replace(row_call, compact_row_call, 1)
    utils = verl_root / ROLLOUT_UTILS_FILE
    utils.write_text(patch_rollout_row_storage(utils.read_text()))
    active_anchor = '            batch_input.meta_info = gen_batch.meta_info'
    active_insert = active_anchor + '''
            if str(self.config.algorithm.adv_estimator) == "deltatrace":
                batch_input.non_tensor_batch["rollout_active_mask"] = active_masks
'''
    if active_insert not in text:
        if text.count(active_anchor) != 1:
            raise RuntimeError("cannot find collector generation mask boundary")
        text = text.replace(active_anchor, active_insert, 1)
    rollout.write_text(patch_context_budget(patch_rollout_phase_logs(text)))
    print(f"patched {rollout} DeltaTrace collector")


def patch_metadata_preparation(text: str) -> str:
    """Use the author formatter offline for its documented text-only metadata.

    The owner's prepare.py explicitly does not use Geometry3k's task contents.
    Its text path needs only row counts. Avoid downloading that unrelated asset;
    keep the original formatter, split/index generation and parquet writer.
    """
    marker = "    parser.add_argument('--metadata_only', action='store_true',"
    if marker in text:
        return text
    anchor = "    args = parser.parse_args()"
    load = "    dataset = datasets.load_dataset(data_source)"
    if text.count(anchor) != 1 or text.count(load) != 1:
        raise RuntimeError('cannot find author metadata preparation anchors')
    text = text.replace(anchor, marker + "\n"
                        "                        help='Create text modality/count metadata without Geometry3k')\n\n" + anchor, 1)
    return text.replace(load, '''    if args.metadata_only:
        if args.mode != 'text':
            parser.error('--metadata_only supports only text modality')
        dataset = datasets.DatasetDict({
            split: datasets.Dataset.from_dict({'problem': [''] * size, 'images': [None] * size})
            for split, size in [('train', args.train_data_size), ('test', args.val_data_size)]
        })
    else:
        dataset = datasets.load_dataset(data_source)''', 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("verl_root", type=Path)
    args = parser.parse_args()

    preparation = args.verl_root / 'examples/data_preprocess/prepare.py'
    preparation.write_text(patch_metadata_preparation(preparation.read_text()))

    # Reuse the published VERL implementation for the observed vLLM 0.15
    # LoRAModel module/API move. The patch includes its source URL and digest.
    # The author commit stores this file with CRLF. The published patch uses
    # LF; normalize only its target before asking patch to match exact hunks.
    vllm_utils = args.verl_root / 'verl/utils/vllm_utils.py'
    if b'\r\n' in vllm_utils.read_bytes():
        vllm_utils.write_bytes(vllm_utils.read_bytes().replace(b'\r\n', b'\n'))
    lora_patch = Path(__file__).with_name('patches') / 'verl-v0.7.0-vllm-lora.patch'
    applied = subprocess.run(
        ['patch', '--force', '--reverse', '--dry-run', '--fuzz=0', '-p1', '-i', str(lora_patch.resolve())],
        cwd=args.verl_root, capture_output=True,
    )
    if applied.returncode:
        subprocess.run(['patch', '--force', '--dry-run', '--fuzz=0', '-p1', '-i', str(lora_patch.resolve())],
                       cwd=args.verl_root, check=True)
        subprocess.run(['patch', '--force', '--fuzz=0', '-p1', '-i', str(lora_patch.resolve())],
                       cwd=args.verl_root, check=True)
        print('backported official VERL v0.7.0 vLLM LoRA compatibility')
    vllm_utils = args.verl_root / 'verl/utils/vllm_utils.py'
    vllm_utils.write_text(patch_vllm_lazy_moe_imports(vllm_utils.read_text()))
    vllm_sharding = args.verl_root / 'verl/workers/sharding_manager/fsdp_vllm.py'
    vllm_sharding.write_text(patch_vllm_peft_owner(vllm_sharding.read_text()))
    env_manager = args.verl_root / 'agent_system/environments/env_manager.py'
    env_text = env_manager.read_text()
    env_manager.write_text(patch_appworld_active_steps(env_text, manager=True))
    memory = args.verl_root / 'agent_system/memory/memory.py'
    memory.write_text(patch_memory_active_steps(memory.read_text()))
    # The owner already accepts dataset, service ports and interaction limit.
    # Expose those arguments; retain all historical defaults and owner sampling.
    env_text = env_manager.read_text()
    app_replacements = [
        ("build_appworld_envs(dataset_name='train', seed=",
         "build_appworld_envs(dataset_name=config.env.get('appworld_train_dataset', 'train'), seed="),
        ("build_appworld_envs(dataset_name='test_normal', seed=",
         "build_appworld_envs(dataset_name=config.env.get('appworld_val_dataset', 'test_normal'), seed="),
        ("start_server_id=0, resources_per_worker=resources_per_worker)",
         "start_server_id=0, resources_per_worker=resources_per_worker, port_file=config.env.get('appworld_port_file', 'appworld_ports.ports'), max_interactions=config.env.get('appworld_max_interactions', 50))"),
        ("start_server_id=config.data.train_batch_size*group_n, resources_per_worker=resources_per_worker)",
         "start_server_id=config.data.train_batch_size*group_n, resources_per_worker=resources_per_worker, port_file=config.env.get('appworld_port_file', 'appworld_ports.ports'), max_interactions=config.env.get('appworld_max_interactions', 50))"),
    ]
    for old, new in app_replacements:
        if new not in env_text:
            if env_text.count(old) != 1:
                raise RuntimeError(f'cannot find unique AppWorld configuration anchor: {old}')
            env_text = env_text.replace(old, new, 1)
    env_manager.write_text(env_text)
    app_envs = args.verl_root / 'agent_system/environments/env_package/appworld/envs.py'
    app_text = app_envs.read_text()
    for old, new in [
        ('                        resources_per_worker={"num_cpus": 0.1},\n                        ):',
         '                        resources_per_worker={"num_cpus": 0.1},\n                        port_file="appworld_ports.ports",\n                        ):'),
        ('        resources_per_worker=resources_per_worker\n    )',
         '        resources_per_worker=resources_per_worker,\n        port_file=port_file\n    )'),
    ]:
        if new not in app_text:
            if app_text.count(old) != 1:
                raise RuntimeError('cannot find unique AppWorld factory port argument anchor')
            app_text = app_text.replace(old, new, 1)
    app_envs.write_text(patch_appworld_active_steps(app_text))
    checkpoint = args.verl_root / 'verl/utils/checkpoint/fsdp_checkpoint_manager.py'
    checkpoint_text = checkpoint.read_text()
    old = '                generation_config = GenerationConfig.from_pretrained(model_config.name_or_path)'
    new = '''                # Qwen3.5 has no separate generation_config.json. Reuse HF's
                # already loaded model configuration for this local-file case.
                if os.path.isdir(model_config.name_or_path) and not os.path.isfile(os.path.join(model_config.name_or_path, "generation_config.json")):
                    generation_config = getattr(unwrap_model, "generation_config", None)
                    if generation_config is None:
                        generation_config = GenerationConfig.from_model_config(model_config)
                else:
                    generation_config = GenerationConfig.from_pretrained(model_config.name_or_path)'''
    if new not in checkpoint_text:
        if checkpoint_text.count(old) != 1:
            raise RuntimeError('cannot find checkpoint generation metadata anchor')
        checkpoint.write_text(checkpoint_text.replace(old, new, 1))
    # The optional adapter-only export is implemented by this owner for FSDP1.
    # FSDP2 already saved all model/LoRA/optimizer state above; it must not access
    # the nonexistent self.actor_module field in that FSDP1-only extra export.
    worker_path = args.verl_root / 'verl/workers/fsdp_workers.py'
    worker_text = worker_path.read_text()
    worker_text = worker_text.replace(
        '        if self._is_lora and isinstance(self.actor_module, PeftModel):',
        '        if self._is_lora and isinstance(self.actor_module_fsdp, FSDP) and isinstance(self.actor_module_fsdp._fsdp_wrapped_module, PeftModel):',
        1,
    ).replace(
        "asdict(self.actor_module.peft_config.get('default', {}))",
        "asdict(self.actor_module_fsdp._fsdp_wrapped_module.peft_config.get('default', {}))",
        1,
    )
    worker_path.write_text(worker_text)
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
    if FSDP_QWEN_CHAT_EOS not in text:
        if FSDP_QWEN_CHAT_EOS_ANCHOR not in text:
            raise RuntimeError(f"cannot find Qwen chat generation config anchor in {fsdp}")
        text = text.replace(FSDP_QWEN_CHAT_EOS_ANCHOR, FSDP_QWEN_CHAT_EOS, 1)
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
    if FSDP2_ACTOR_NEW in text:
        text = text.replace(FSDP2_ACTOR_NEW, FSDP2_ACTOR_OLD, 1)
    elif FSDP2_ACTOR_OLD not in text:
        raise RuntimeError(f"cannot find FSDP2 PEFT root anchor in {fsdp}")
    fsdp.write_text(text)
    print(f"patched {fsdp} Transformers compatibility")
    actor_policy = args.verl_root / ACTOR_FILE
    text = actor_policy.read_text()
    if ACTOR_FORWARD_NEW not in text and '                    logits_to_keep=head_positions,' not in text:
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

    text = policy.read_text()
    patched = patch_fsdp2_offload(text)
    if patched != text:
        policy.write_text(patched)
        print(f"patched {policy} PEFT/FSDP2 memory lifecycle boundary")

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
    if FSDP_DT_METHOD_PREVIOUS in text:
        text = text.replace(FSDP_DT_METHOD_PREVIOUS, FSDP_DT_METHOD, 1)
        fsdp_workers.write_text(text)
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
        qwen_text = qwen35.read_text()
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
        qwen_text = qwen35.read_text()
        patched_qwen_text = patch_qwen35_padding_mask(qwen_text)
        if patched_qwen_text != qwen_text:
            qwen35.write_text(patched_qwen_text)
            print(f"patched {qwen35} upstream batch-one padding mask")
        else:
            print(f"already patched {qwen35} upstream batch-one padding mask")

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

    patch_rollout_owner(args.verl_root)

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
        if path == args.verl_root / HF_ROLLOUT_FILE and HF_ACTIVE_MARKER in text:
            continue  # The active-row extension already contains this trim fix.
        for previous, current in [(HF_TRIM_PREVIOUS, HF_TRIM_INSERT),
                                  (ACTOR_TRIM_PREVIOUS, ACTOR_TRIM_INSERT)]:
            if previous in text:
                text = text.replace(previous, current, 1)
        if '\nimport os\n' not in text:
            text = text.replace('\nimport torch\n', '\nimport os\nimport torch\n', 1)
        for old, new in replacements:
            if new not in text:
                if old not in text:
                    raise RuntimeError(f'cannot find shared-padding anchor in {path}')
                text = text.replace(old, new, 1)
        path.write_text(text)

    hf = args.verl_root / HF_ROLLOUT_FILE
    hf.write_text(patch_hf_active_rows(hf.read_text()))

    actor.write_text(patch_actor_response_head(actor.read_text()))

    vllm = args.verl_root / VLLM_ROLLOUT_FILE
    vllm.write_text(patch_vllm_active_rows(vllm.read_text()))


if __name__ == "__main__":
    main()

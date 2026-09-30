"""Check only the pinned owner's head/dispatch seam; PPO stays upstream."""
import ast
from pathlib import Path
import os

import torch
from patch_actor_fused_head import patch_dispatch, patch_precision


def test_patch_is_idempotent_and_does_not_change_other_dispatch():
    root=Path(os.environ['VERL_ROOT'])
    source=(root/'verl/models/transformers/monkey_patch.py').read_text()
    assert patch_dispatch(patch_dispatch(source))==patch_dispatch(source)
    original=source.replace(', "qwen3_5_text"]:', ']:')
    def functions(s):
        return {n.name:ast.dump(n) for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)}
    before,after=functions(original),functions(patch_dispatch(original))
    assert {k for k in before if before[k]!=after[k]}=={'patch_forward_with_backends'}
    head=(root/'verl/utils/experimental/torch_functional.py').read_text()
    assert patch_precision(patch_precision(head))==patch_precision(head)


def test_text_model_uses_original_wrapper_and_leaves_dt_class_unchanged():
    from transformers import AutoConfig, Qwen3_5ForCausalLM, Qwen3_5ForConditionalGeneration
    from verl.models.transformers.monkey_patch import patch_forward_with_backends
    from verl.models.transformers.qwen3_vl import forward_with_torch_backend
    config=AutoConfig.from_pretrained(os.environ['MODEL_PATH'],local_files_only=True).text_config
    text_original=Qwen3_5ForCausalLM.forward
    dt_original=Qwen3_5ForConditionalGeneration.forward
    with torch.device('meta'):
        model=Qwen3_5ForCausalLM(config)
    try:
        patch_forward_with_backends(model,use_fused_kernels=False,fused_kernels_backend='torch')
        assert Qwen3_5ForCausalLM.forward is text_original
        patch_forward_with_backends(model,use_fused_kernels=True,fused_kernels_backend='torch')
        assert Qwen3_5ForCausalLM.forward is forward_with_torch_backend
        assert Qwen3_5ForConditionalGeneration.forward is dt_original
    finally:
        Qwen3_5ForCausalLM.forward=text_original

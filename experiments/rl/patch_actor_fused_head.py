"""Minimal seams in the pinned VERL head; no replacement PPO/head algorithm."""
from pathlib import Path


def patch_precision(source):
    replacements = {
        'return token_log_probs.to(orig_dtype), entropy.to(orig_dtype)':
            'return token_log_probs, entropy',
        'log_probs = hidden_states.new_zeros(T, requires_grad=output_requires_grad)':
            'log_probs = torch.zeros(T, device=hidden_states.device, dtype=torch.float32, requires_grad=output_requires_grad)',
        'entropy = hidden_states.new_zeros(T, requires_grad=output_requires_grad)':
            'entropy = torch.zeros(T, device=hidden_states.device, dtype=torch.float32, requires_grad=output_requires_grad)',
        'dlogits += dlog_probs.to(torch.float32).unsqueeze(-1) * (one_hot_input - probs)':
            'dlogits += (dlog_probs.to(torch.float32).unsqueeze(-1) * (one_hot_input - probs)).to(orig_dtype)',
        'dlogits += probs * (log_probs + entropy.unsqueeze(-1)) * (-dentropy.unsqueeze(-1))':
            'dlogits += (probs * (log_probs + entropy.unsqueeze(-1)) * (-dentropy.unsqueeze(-1))).to(orig_dtype)',
    }
    for old, new in replacements.items():
        if old not in source and source.count(new) == 1:
            continue
        if source.count(old) != 1:
            raise RuntimeError('Pinned fused-head dtype anchor changed: '+old)
        source = source.replace(old, new, 1)
    # MetaX Inductor elides the BF16 result of non-unit temperature division
    # even with emulate_precision_casts. Materialize this original boundary;
    # preserve the subsequent FP32 probability/gradient fusion unchanged.
    old = '    logits = (hidden_states @ vocab_weights.t()) / temperature\n'
    new = old + ('    if temperature != 1.0:\n'
                 '        torch._dynamo.graph_break()\n')
    if source.count(new) != 2:
        if source.count(old) != 2 or new in source:
            raise RuntimeError('Pinned fused-head temperature anchor changed')
        source = source.replace(old, new)
    # Compile each official fixed-size chunk, keeping the owner's chunk loop
    # eager. Compiling the full loop would specialize on the trajectory length.
    for name in ['_fused_linear_for_ppo_fwd', '_fused_linear_for_ppo_bwd']:
        old = 'def '+name+'('
        decorator = '@torch.compile(dynamic=True, options={"emulate_precision_casts": True})'
        new = decorator+'\n'+old
        # Preserve BF16 temperature scaling in forward/recomputation and the
        # two separate gradient branches, as in the owner's dense autograd path.
        previous = '@torch.compile(dynamic=True)\n'+old
        if previous in source and previous != new:
            source = source.replace(previous, new, 1)
        if new not in source:
            if source.count(old) != 1:
                raise RuntimeError('Pinned fused-head function missing: '+name)
            source = source.replace(old, new, 1)
    return source


def patch_dispatch(source):
    # The existing owner forward only calls self.model, rolls labels, and
    # calls FusedLinearForPPO. Qwen3.5's text CausalLM exposes this same ABI.
    # Do not patch ConditionalGeneration: the DT owner view uses that class.
    old = ('elif model.config.model_type in ["qwen3_vl", "qwen3_vl_moe"]:\n'
           '        from verl.models.transformers.qwen3_vl import forward_with_torch_backend, forward_with_triton_backend')
    new = old.replace('["qwen3_vl", "qwen3_vl_moe"]',
                      '["qwen3_vl", "qwen3_vl_moe", "qwen3_5_text"]')
    if old not in source and source.count(new) == 1:
        return source
    if source.count(old) != 1:
        raise RuntimeError('Pinned fused-head dispatch anchor changed')
    return source.replace(old, new, 1)


if __name__ == '__main__':
    import sys
    root = Path(sys.argv[1])
    for relative, apply in [
        ('verl/utils/experimental/torch_functional.py', patch_precision),
        ('verl/models/transformers/monkey_patch.py', patch_dispatch),
    ]:
        path = root/relative
        source = apply(path.read_text())
        compile(source, str(path), 'exec')
        path.write_text(source, newline='\n')

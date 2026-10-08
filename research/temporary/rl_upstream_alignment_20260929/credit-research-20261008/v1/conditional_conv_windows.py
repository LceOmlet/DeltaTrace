"""Research-only packing of one-row interventions for the original conv API.

The installed public causal_conv1d_fn owns convolution, cached history and
fused SiLU. A source row has exactly width affected outputs. No full model is
called per source, and no convolution/normalization kernel is reimplemented.
"""
import torch
from fla.modules.l2norm import l2norm_fwd


def conditional_conv_windows(conv, factual, replacement, weight, *, initial=None,
                             bias=None, start=0, stop=None):
    """Return original preactivation and fused output as [width,B,S,channels].

    factual/replacement: actual projected QKV endpoint tensors [B,D,T]. Only
    the selected row is replaced; the preceding history and later rows remain
    factual. Padding after T is internal storage, never an observed action.
    """
    B, D, T = factual.shape
    width = weight.shape[-1]
    stop = T if stop is None else stop
    assert replacement.shape == factual.shape and 0 <= start < stop <= T
    if initial is None:
        initial = factual.new_zeros(B, D, width-1)
    assert initial.shape == (B, D, width-1)
    stream = torch.cat((initial, factual, factual.new_zeros(B, D, width-1)), dim=-1)
    S = stop-start
    x = stream.unfold(-1, width, 1)[:, :, width-1+start:width-1+stop]
    history = stream.unfold(-1, width-1, 1)[:, :, start:stop]
    x = x.permute(0, 2, 3, 1).contiguous().view(B*S, width, D).transpose(1, 2)
    history = history.permute(0, 2, 3, 1).contiguous().view(B*S, width-1, D).transpose(1, 2)
    x[:, :, 0] = replacement[:, :, start:stop].transpose(1, 2).reshape(B*S, D)
    pre = conv(x, weight, bias=bias, initial_states=history, activation=None)
    output = conv(x, weight, bias=bias, initial_states=history, activation='silu')
    unpack = lambda value:value.view(B, S, D, width).permute(3, 0, 1, 2).contiguous()
    valid = (torch.arange(start, stop, device=factual.device)[None, :]
             +torch.arange(width, device=factual.device)[:, None] < T)
    return dict(pre=unpack(pre), output=unpack(output), valid=valid)


def native_qkv(output, *, key_heads, value_heads, key_dim, value_dim):
    """Reuse Qwen's split/repeat layout and original FLA normalization."""
    q, k, v = output.split((key_heads*key_dim, key_heads*key_dim,
                           value_heads*value_dim), dim=-1)
    q = q.reshape(*q.shape[:-1], key_heads, key_dim)
    k = k.reshape(*k.shape[:-1], key_heads, key_dim)
    v = v.reshape(*v.shape[:-1], value_heads, value_dim)
    repeat = value_heads//key_heads
    q = q.repeat_interleave(repeat, dim=-2).contiguous()
    k = k.repeat_interleave(repeat, dim=-2).contiguous()
    q, _ = l2norm_fwd(q)
    k, _ = l2norm_fwd(k)
    return dict(q=q, k=k, v=v.contiguous())

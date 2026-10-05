"""Exact native Qwen3.5 artifacts for explicitly supplied prefix leases.

The HF model, FLA state operator and Transformers Cache retain ownership of
all computation and state transitions. This file does not implement attention,
recurrence, convolution, cache updates, attribution or a training scheduler.
It is not enabled by the default DT runner or a production launcher.
"""
from dataclasses import dataclass

import torch


@dataclass
class NativePrefixArtifacts:
    config: object
    input_ids: torch.Tensor
    layers: list
    boundary_rows: object = None

    @classmethod
    @torch.no_grad()
    def capture(cls, model, input_ids, prefix_lengths, *, config, observe_native_cache=None, boundary_rows=None):
        """Read requested boundaries during one unchanged native forward.

        Only the supplied factual IDs are forwarded. A boundary consumes
        tensors up to that boundary. FLA emits its own FP32 final state; HF
        Cache determines the stored dtype when the artifacts are consumed.
        The caller owns the lifetime and grouping of these exact artifacts.
        """
        from accelerated.native_capture_events import LocalCaptureEvents
        from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule_fwd
        from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h
        from transformers.cache_utils import DynamicCache, LinearAttentionCacheLayerMixin

        lengths = sorted(set(int(n) for n in prefix_lengths))
        if not lengths or any(n < 64 or n % 64 or n > input_ids.shape[1] for n in lengths):
            raise ValueError('Native DT prefix boundaries must be positive 64-token boundaries in the factual input')
        selected_rows = None
        if boundary_rows is not None:
            selected_rows = {n: tuple(int(row) for row in boundary_rows.get(n, ())) for n in lengths}
            if any(len(set(rows)) != len(rows) or any(row < 0 or row >= input_ids.shape[0] for row in rows)
                   for rows in selected_rows.values()):
                raise ValueError('Selected native boundary rows must be distinct original capture rows')
        cache = DynamicCache(config=config)
        layers = model.model.language_model.layers
        artifacts = [None] * len(layers)
        active = [None]
        pending = {}
        windows = {}
        handles = []

        def state_event(frame, kind, value):
            if kind != 'return' or value is None:
                return
            i = active[0]
            f = frame.f_locals
            if i is None or f['initial_state'] is not None or f['cu_seqlens'] is not None:
                raise ValueError('Prefix artifact capture requires the original dense prefill from an empty cache')
            pending[i] = {name: f[name].detach() for name in ('k', 'w', 'u', 'g')}
            pending[i]['final_state'] = f['final_state'].detach()

        def begin(_module, _args, i):
            active[0] = i

        def projection(_module, _args, output, i, width):
            # Exact native pre-convolution projection. HF's prefill stores
            # the last conv_kernel_size columns; no convolution is repeated.
            if selected_rows is None:
                windows[i] = {n: output[:, n-width:n].transpose(1, 2).detach().cpu()
                              for n in lengths}
            else:
                windows[i] = {n: output[:, n-width:n].index_select(0,
                    torch.tensor(rows, device=output.device, dtype=torch.long)).transpose(1, 2).detach().cpu()
                    for n, rows in selected_rows.items() if rows}

        def end(_module, _args, _output, i):
            owner_layer = cache.layers[i]
            if isinstance(owner_layer, LinearAttentionCacheLayerMixin):
                captured = pending.pop(i)
                boundaries = {}
                previous_length = 0
                previous_state = None
                for n in lengths:
                    if n == input_ids.shape[1]:
                        state = captured['final_state']
                    else:
                        h, v_new, state = chunk_gated_delta_rule_fwd_h(
                            k=captured['k'][:, previous_length:n].contiguous(),
                            w=captured['w'][:, previous_length:n].contiguous(),
                            u=captured['u'][:, previous_length:n].contiguous(),
                            g=captured['g'][:, previous_length:n].contiguous(),
                            initial_state=previous_state, output_final_state=True,
                            save_new_value=False)
                        del h, v_new
                    if selected_rows is None:
                        boundaries[n] = (windows[i].pop(n), state.detach().cpu())
                    elif selected_rows[n]:
                        indices = torch.tensor(selected_rows[n], device=state.device, dtype=torch.long)
                        boundaries[n] = (windows[i].pop(n), state.index_select(0, indices).detach().cpu())
                    # The public owner accepts its FP32 final state as the
                    # next segment's initial state. Consume each token once;
                    # no restart from token zero for every requested boundary.
                    previous_length, previous_state = n, state
                    del state
                del previous_state, captured
                windows.pop(i)
                artifacts[i] = {'boundaries': boundaries}
            else:
                artifacts[i] = {'keys': owner_layer.keys.detach().cpu(),
                                'values': owner_layer.values.detach().cpu()}

        try:
            for i, layer in enumerate(layers):
                handles.append(layer.register_forward_pre_hook(
                    lambda m, a, i=i: begin(m, a, i)))
                if isinstance(cache.layers[i], LinearAttentionCacheLayerMixin):
                    mixer = layer.linear_attn
                    handles.append(mixer.in_proj_qkv.register_forward_hook(
                        lambda m, a, o, i=i, width=mixer.conv_kernel_size:
                            projection(m, a, o, i, width)))
                handles.append(layer.register_forward_hook(
                    lambda m, a, o, i=i: end(m, a, o, i)))
            with LocalCaptureEvents([chunk_gated_delta_rule_fwd.__code__], state_event,
                                    returns_only=True):
                forward = getattr(model, 'forward_root', model)
                output = forward(input_ids=input_ids, past_key_values=cache,
                                 use_cache=True, logits_to_keep=1)
            if observe_native_cache is not None:
                observe_native_cache(cache)
            del output
        finally:
            for handle in handles:
                handle.remove()
            pending.clear()
            windows.clear()
        if selected_rows is not None:
            return cls(config, input_ids.detach().cpu(), artifacts,
                boundary_rows={n: {row: index for index, row in enumerate(rows)}
                               for n, rows in selected_rows.items()})
        return cls(config, input_ids.detach().cpu(), artifacts)

    def materialize(self, prefix_length, *, device, rows=None):
        rows = range(self.input_ids.shape[0]) if rows is None else rows
        return compose_native_prefix_cache(
            self.config, [(self, int(row)) for row in rows], prefix_length, device=device)


@torch.no_grad()
def compose_native_prefix_cache(config, sources, prefix_length, *, device):
    """Batch exact artifacts with the original Cache's public update methods.

    In particular, conv state is written before recurrent state, matching
    the HF forward and retaining the owner's dtype conversion and lifecycle.
    No Cache fields or transition flags are assigned by this adapter.
    """
    from transformers.cache_utils import DynamicCache, LinearAttentionCacheLayerMixin

    result = DynamicCache(config=config)
    for i, layer in enumerate(result.layers):
        if isinstance(layer, LinearAttentionCacheLayerMixin):
            states = [source.layers[i]['boundaries'][prefix_length] for source, row in sources]
            if all(getattr(source, 'boundary_rows', None) is None for source, row in sources):
                conv = torch.cat([state[0][row:row+1] for state, (_, row) in zip(states, sources)])
                recurrent = torch.cat([state[1][row:row+1] for state, (_, row) in zip(states, sources)])
            else:
                rows = [row if getattr(source, 'boundary_rows', None) is None
                        else source.boundary_rows[prefix_length][row] for source, row in sources]
                conv = torch.cat([state[0][row:row+1] for state, row in zip(states, rows)])
                recurrent = torch.cat([state[1][row:row+1] for state, row in zip(states, rows)])
            result.update_conv_state(conv.to(device), i)
            result.update_recurrent_state(recurrent.to(device), i)
        else:
            keys = torch.cat([source.layers[i]['keys'][row:row+1, :, :prefix_length]
                              for source, row in sources])
            values = torch.cat([source.layers[i]['values'][row:row+1, :, :prefix_length]
                                for source, row in sources])
            result.update(keys.to(device), values.to(device), i)
    return result


@dataclass
class NativePrefixLease:
    """Exact artifacts for one original consumer; HF owns the fresh cache."""
    sources: list
    prefix_length: int

    def __call__(self, factual_ids):
        if factual_ids.shape != (len(self.sources), self.prefix_length):
            raise ValueError('Prepared prefix lease and factual input shapes differ')
        actual = factual_ids.detach().cpu()
        for i, (source, row) in enumerate(self.sources):
            if not torch.equal(actual[i], source.input_ids[row, :self.prefix_length]):
                raise ValueError('Prepared prefix lease does not match exact factual token IDs')
        return compose_native_prefix_cache(self.sources[0][0].config, self.sources,
                                          self.prefix_length, device=factual_ids.device)

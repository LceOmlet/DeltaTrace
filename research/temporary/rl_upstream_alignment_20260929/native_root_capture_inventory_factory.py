"""Compose original DT passive capture APIs for root storage observation only.

No new model forward, operator, propagation rule, or tensor offloader. Borrow
native GPU operands for one layer, then let the inventory release them. GDN
start=0 measures the full current root suffix; the runner's later actual cut
is recorded independently before claiming this is the exact required size.
"""
import hashlib
import inspect
from pathlib import Path


class OriginalRootCaptureFactory:
    def __init__(self, runner, owner_globals, *, cache_tensors=None):
        self.runner = runner
        self.owner_globals = owner_globals
        self.cache_tensors = cache_tensors

    def __call__(self, index, layer, args, kwargs):
        runner, source = self.runner, self.owner_globals
        backend = runner.capture_backend
        decoder = source['NativeDecoderCapture'] if backend is None else backend.NativeDecoderCapture
        attention = source['NativeDenseAttentionCapture'] if backend is None else backend.NativeDenseAttentionCapture
        gdn = source['NativeGDNCapture'] if backend is None else backend.NativeGDNCapture
        dc = decoder(layer, destination='cuda', copy_tensors=False,
                     retained_names={'input_norm_input', 'post_norm_input',
                                     'gate_output', 'up_output', 'silu_output'})
        if layer.block_type == 'full_attention':
            mc = attention(layer.self_attn, source['flash_attention_forward'],
                           source['flash_attn_varlen_func'], source['flash_attn_func'],
                           destination='cuda', copy_tensors=False,
                           retained_names={'q_proj_output', 'attention_output',
                                           'query', 'key', 'value', 'q_norm_input',
                                           'k_norm_input', 'dense_q', 'dense_k', 'dense_v'},
                           preserve_strides=False, pinned_host=False)
        else:
            mc = gdn(layer.linear_attn, device='cuda', copy_tensors=False,
                     preserve_strides=False, capture_module_outputs=False,
                     pinned_host=False, capture_input=False,
                     gpu_capture_names=(), coefficient_start=0)
        cache = kwargs.get('past_key_values')
        external = ((lambda: self.cache_tensors(cache))
                    if cache is not None and self.cache_tensors is not None else {})
        return dc, mc, external

    def provenance(self):
        backend = self.runner.capture_backend
        source = self.owner_globals
        classes = [source['NativeDecoderCapture'], source['NativeGDNCapture'],
                   source['NativeDenseAttentionCapture']]
        if backend is not None:
            classes.extend([backend.NativeDecoderCapture, backend.NativeGDNCapture,
                            backend.NativeDenseAttentionCapture])
        files = sorted({inspect.getfile(value) for value in classes})
        return {'original_capture_files': [dict(path=path,
                    sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest()) for path in files],
                'root_inventory_gdn_coefficient_start': 0,
                'inventory_destination': 'cuda', 'copy_tensors': False,
                'scope': 'One-layer borrowed original root operands; no retained multi-layer tape or speed claim',
                'original_runner_capture_configuration': {
                    name: getattr(self.runner, name) for name in
                    ('copy_replay_captures', 'offload_replay_mixer',
                     'compact_gdn_captures', 'gdn_coefficient_suffix',
                     'fa_coefficient_suffix')}}

"""Compose the original root tape with the owner's capture transport interfaces.

Prepares a separate source artifact only; no production imports or GPU calls.
Original finite rules, parameter callbacks and disabled scheduling stay intact.
"""
from __future__ import annotations

import ast
import difflib
import hashlib
import json
from pathlib import Path

from prepare_native_root_tape_owner_20261004 import RUNNER_NAME, HELPER_NAME, patched_owner


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise ValueError(f'Inspect original owner transport boundary: {before!r}')
    return text.replace(before, after, 1)


def patched_cpu_root_owner(original):
    gpu_tape, baseline = patched_owner(original)
    newline = '\r\n' if b'\r\n' in gpu_tape else '\n'
    text = gpu_tape.decode().replace('\r\n', '\n')
    text = replace_once(text,
        'from qwen35_decoder_finite import NativeDecoderCapture,FiniteBoundaryOps,attention_finite_pullback,decoder_finite_pullback',
        'from native_root_decoder_transport_candidate import NativeDecoderCapture,FiniteBoundaryOps,attention_finite_pullback,decoder_finite_pullback')
    text = replace_once(text,
        'from qwen35_gdn_finite import NativeGDNCapture,gdn_finite_pullback',
        'from native_root_gdn_transport_candidate import NativeGDNCapture,gdn_finite_pullback')
    text = replace_once(text,
        'from native_dense_attention_capture import NativeDenseAttentionCapture',
        'from native_root_dense_attention_transport_candidate import NativeDenseAttentionCapture\n'
        'import native_root_code_local_transport_candidate as _root_capture_backend')
    text = replace_once(text,
        '    def _make_layer_captures(self,layer,is_fa,observer,gdn_cut):',
        '    def _make_layer_captures(self,layer,is_fa,observer,gdn_cut,*,root_capture=False):')
    text = replace_once(text,
        '        offload_mixer=self.offload_replay_mixer and observer is None',
        '        root_capture=root_capture and observer is None\n'
        '        offload_mixer=(self.offload_replay_mixer or root_capture) and observer is None')
    text = replace_once(text,
        "        dc=decoder_capture(layer,destination='cuda',copy_tensors=copy_captures,retained_names=needed)",
        "        decoder_transport=({'preserve_strides':True,'pinned_host':self.pin_root_host,\n"
        "            'defer_host_sync':self.pin_root_host and (backend is None or backend is _root_capture_backend)} if root_capture else {})\n"
        "        mixer_transport=({'defer_host_sync':self.pin_root_host and (backend is None or backend is _root_capture_backend)} if root_capture else {})\n"
        "        dc=decoder_capture(layer,destination='cpu' if root_capture else 'cuda',\n"
        '            copy_tensors=copy_captures,retained_names=needed,**decoder_transport)')
    text = replace_once(text,
        'pinned_host=offload_mixer and self.pin_replay_host)\n',
        'pinned_host=offload_mixer and self.pin_replay_host,**mixer_transport)\n')
    text = replace_once(text,
        'coefficient_start=gdn_cut if self.compact_gdn_captures else 0))\n',
        'coefficient_start=gdn_cut if self.compact_gdn_captures else 0,**mixer_transport))\n')
    text = replace_once(text,
        "                layer,layer.block_type=='full_attention',observer,gdn_cut)))",
        "                layer,layer.block_type=='full_attention',observer,gdn_cut,root_capture=True)))")
    # The original root finally is also the completion boundary for deferred
    # pinned capture copies, including exceptional exits before tape cleanup.
    text = replace_once(text,
        '            if self.pin_root_host:torch.cuda.current_stream().synchronize()',
        '            if self.pin_root_host or _root_tape is not None:torch.cuda.current_stream().synchronize()')
    text = replace_once(text,
        'lambda:decoder_finite_pullback(layer,d,m,mixer,self.boundaries,focused,consume_captures=offload_mixer))',
        'lambda:decoder_finite_pullback(layer,d,m,mixer,self.boundaries,focused,consume_captures=offload_mixer,\n'
        "                **({'restore_captures':True} if _root_tape is not None and observer is None else {})))")
    ast.parse(text)
    compile(text, '<prepared-cpu-root-tape-owner>', 'exec')
    result = text.replace('\n', newline).encode()
    metadata = dict(baseline, candidate_source_sha256=hashlib.sha256(result).hexdigest(),
        parent_gpu_tape_source_sha256=hashlib.sha256(gpu_tape).hexdigest(),
        capture_transport='Existing pinned CPU capture, per-stage decoder restore, existing FA/GDN offload',
        root_capture_destination='cpu', retained_gpu_mixer_fields='Original gdn_gpu_capture_names option unchanged',
        accelerated_capture_backend='native_root_code_local_transport_candidate; original retained/local event class bodies unchanged',
        completion_fence='Original root finally; diagnostic backends keep original capture exit synchronization',
        resource_extension='Patched owner constructor/consumer transport only; no new copier, stream, cache or fallback',
        default_enabled=False, formal_deployment=False)
    return result, metadata


def prepare(owner_source, owner_root, candidate_directory, expected_owner_sha256):
    from prepare_native_root_capture_transport_owner_20261005 import prepare as prepare_transport
    original = Path(owner_source).read_bytes()
    if hashlib.sha256(original).hexdigest() != expected_owner_sha256:
        raise ValueError('Original verified runner SHA differs; inspect before preparing')
    destination = Path(candidate_directory)
    if destination.resolve() in (Path(owner_root).resolve(), Path(owner_source).parent.resolve()):
        raise ValueError('Prepared CPU root candidate must remain outside baseline import directories')
    transport = prepare_transport(Path(owner_root), destination)
    patched, metadata = patched_cpu_root_owner(original)
    helper = Path(__file__).with_name(HELPER_NAME).read_bytes()
    (destination / RUNNER_NAME).write_bytes(patched)
    (destination / HELPER_NAME).write_bytes(helper)
    metadata.update(owner_source=str(Path(owner_source).resolve()), owner_root=str(Path(owner_root).resolve()),
        candidate_directory=str(destination.resolve()), transport=transport,
        helper_sha256=hashlib.sha256(helper).hexdigest(),
        preparer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    diff = ''.join(difflib.unified_diff(original.decode().splitlines(True), patched.decode().splitlines(True),
        fromfile='original_owner', tofile=RUNNER_NAME))
    (destination / 'cpu-root-owner.patch').write_text(diff, encoding='utf8', newline='')
    (destination / 'prepared-cpu-root.json').write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf8')
    return metadata

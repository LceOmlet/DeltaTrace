"""Observe original GDN0 capture events; no forward, cache or numeric replacement.

Attach the generated subclass through the existing capture_backend factory.
It calls the exact supplied capture's event first, then copies actual FLA call
and stage operands to CPU for one bounded diagnostic. These copies are explicit
diagnostic overhead and must not be used as a speed measurement.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import re


OFFICIAL_FLA_TEST_SHA256 = '35f28bf6d01f101f075309133929d1764ab540eb9a892f35eca92227e8768813'


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class NativeGDN0Operands:
    """Add passive observation to an actual supplied NativeGDNCapture class."""
    def __init__(self, target_module, directory, *, variant, rank):
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', variant):
            raise ValueError('Observation variant must be one filename component.')
        self.target_module = target_module
        self.directory = Path(directory)
        self.variant = variant
        self.rank = int(rank)
        self.records = []
        self.owner = None

    @staticmethod
    def _snapshot(value):
        # Original Torch representation transport only. Copy even an existing
        # CPU tensor, so a later owner mutation cannot change this receipt.
        return None if value is None else value.detach().to(device='cpu', copy=True)

    @staticmethod
    def _metadata(value):
        return None if value is None else dict(dtype=str(value.dtype),
            shape=list(value.shape), stride=list(value.stride()), device=str(value.device))

    def capture_type(self, original_capture_type):
        owner = self
        source = Path(inspect.getfile(original_capture_type))
        event_source = Path(inspect.getsourcefile(original_capture_type.event))
        self.owner = dict(path=str(source), sha256=_sha(source),
            class_name=original_capture_type.__qualname__,
            actual_event_source=dict(path=str(event_source), sha256=_sha(event_source)))

        class ObservedNativeGDNCapture(original_capture_type):
            def event(self, frame, kind, value):
                super().event(frame, kind, value)
                if self.module is not owner.target_module:
                    return
                label = self.codes.get(frame.f_code)
                fields = frame.f_locals
                if label == 'module' and kind == 'call' and fields['self'] is self.module:
                    self._gdn0_observation = dict(tensors={}, metadata={}, calls={})
                pending = getattr(self, '_gdn0_observation', None)
                if pending is None:
                    return

                def retain(name, tensor):
                    pending['metadata'][name] = owner._metadata(tensor)
                    pending['tensors'][name] = owner._snapshot(tensor)

                if label == 'FLA' and kind == 'call' and self.active:
                    for name in ('q', 'k', 'v', 'g', 'beta'):
                        retain('raw_' + name, fields[name])
                    # This is the actual public call's flag. Do not infer it
                    # from q/k values or normalize the stage tensors again.
                    pending['calls']['fla'] = {name: fields[name] for name in
                        ('scale', 'output_final_state', 'use_qk_l2norm_in_kernel')}
                    retain('fla_initial_state', fields['initial_state'])
                    retain('fla_cu_seqlens', fields['cu_seqlens'])
                elif label == 'stage' and kind == 'call' and self.active:
                    pending['calls']['stage'] = {name: fields[name] for name in
                        ('scale', 'output_final_state')}
                    retain('initial_state', fields['initial_state'])
                    retain('cu_seqlens', fields['cu_seqlens'])
                elif label == 'stage' and kind == 'return' and value is not None:
                    for name in ('o', 'final_state', 'k', 'w', 'u', 'g', 'h'):
                        retain('stage_' + name, fields[name])
                elif label == 'FLA' and kind == 'return' and value is not None:
                    native_o, native_ht = value
                    retain('native_o', native_o)
                    retain('native_ht', native_ht)
                elif label == 'module' and kind == 'return' and fields['self'] is self.module:
                    retain('model_boundary_o', self.endpoints.get('o'))
                    pending['capture_calls'] = dict(self.calls)
                    pending['capture_coefficient_start'] = self.coefficient_start
                    owner._save(pending)
                    del self._gdn0_observation

        return ObservedNativeGDNCapture

    def _save(self, pending):
        import torch
        self.directory.mkdir(parents=True, exist_ok=True)
        ordinal = len(self.records) + 1
        path = self.directory / f'gdn0-{self.variant}-rank{self.rank}-call{ordinal}.pt'
        payload = dict(role=__doc__, variant=self.variant, rank=self.rank,
            ordinal=ordinal, original_capture_owner=self.owner,
            observer_source=dict(path=__file__, sha256=_sha(__file__)), **pending)
        # Preserve prior observations instead of replacing a receipt.
        with path.open('xb') as stream:
            torch.save(payload, stream)
        self.records.append(dict(path=str(path), sha256=_sha(path),
            tensors=pending['metadata'], calls=pending['calls'],
            capture_calls=pending['capture_calls'],
            capture_coefficient_start=pending['capture_coefficient_start']))

    def report(self):
        return dict(variant=self.variant, rank=self.rank, original_capture_owner=self.owner,
            recorded_calls=len(self.records), records=self.records,
            model_forwards_added=0, numeric_code_replaced=False,
            scope='Actual FLA call/stage observations; CPU copies are diagnostic overhead')


def check_saved_fla(operands, official_source, *, device='cuda'):
    """Use the pinned original reference and exact o/ht assertion AST, unchanged."""
    import torch
    import torch.nn.functional as F
    import fla.utils as owner
    if owner.FLA_CI_ENV:
        raise AssertionError('Original FLA assertion must remain strict; FLA_CI_ENV is enabled.')
    official_source = Path(official_source)
    if _sha(official_source) != OFFICIAL_FLA_TEST_SHA256:
        raise ValueError('Pinned official FLA reference source SHA256 mismatch.')
    parsed = ast.parse(official_source.read_bytes())
    test = next(node for node in parsed.body
                if isinstance(node, ast.FunctionDef) and node.name == 'test_chunk')
    assertions = [node for node in test.body if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
        and node.value.func.id == 'assert_close'
        and isinstance(node.value.args[0], ast.Constant)
        and node.value.args[0].value in ('o', 'ht')]
    if len(assertions) != 2:
        raise ValueError('Pinned original o/ht assertion interface changed.')
    assertion_code = compile(ast.fix_missing_locations(ast.Module(
        body=assertions, type_ignores=[])), str(official_source), 'exec')
    spec = importlib.util.spec_from_file_location('_actual_gdn0_official_reference', official_source)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    saved = torch.load(operands, map_location='cpu', weights_only=True)
    tensors, calls = saved['tensors'], saved['calls']
    if tensors['cu_seqlens'] is not None or tensors['fla_cu_seqlens'] is not None:
        raise ValueError('This original dense recurrence check does not cover packed cu_seqlens.')
    if not calls['stage']['output_final_state'] or not calls['fla']['output_final_state']:
        raise ValueError('Actual native call did not request the final state required by the original ht assertion.')
    move = lambda value: None if value is None else value.to(device=device)
    q, k = [move(tensors['raw_' + name]) for name in ('q', 'k')]
    normalize = calls['fla']['use_qk_l2norm_in_kernel']
    if normalize:
        # Apply the reference's ordinary normalization to its raw inputs once;
        # never normalize already-normalized native stage q/k a second time.
        q, k = F.normalize(q, p=2, dim=-1), F.normalize(k, p=2, dim=-1)
    with torch.no_grad():
        ref, ref_ht = reference.recurrent_gated_delta_rule_ref(
            q=q, k=k, v=move(tensors['raw_v']), beta=move(tensors['raw_beta']),
            g=move(tensors['raw_g']), scale=calls['stage']['scale'],
            initial_state=move(tensors['initial_state']), output_final_state=True)
        tri, tri_ht = move(tensors['native_o']), move(tensors['native_ht'])
        # Execute the original statements; never restate or widen tolerances.
        exec(assertion_code, dict(assert_close=owner.assert_close,
             ref=ref, tri=tri, ref_ht=ref_ht, tri_ht=tri_ht))
    return dict(role='Original FLA forward o/ht check on one actual cached-suffix call',
        operands=dict(path=str(operands), sha256=_sha(operands)),
        official_source=dict(path=str(official_source), sha256=_sha(official_source)),
        original_assertions=[ast.unparse(node) for node in assertions],
        status='passed', normalization_applied_once=bool(normalize),
        actual_calls=calls, actual_tensor_metadata=saved['metadata'],
        initial_state_was_none=tensors['initial_state'] is None,
        numeric_correction=False, whole_dt_acceptance=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--operands', type=Path, required=True)
    parser.add_argument('--official-source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    result = check_saved_fla(args.operands, args.official_source, device=args.device)
    with args.output.open('x', encoding='utf8') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()

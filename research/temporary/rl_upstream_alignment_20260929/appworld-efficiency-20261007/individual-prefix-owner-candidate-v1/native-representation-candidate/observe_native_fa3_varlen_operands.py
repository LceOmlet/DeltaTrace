"""Passive actual varlen FA operands and unchanged official output assertions.

Isolated diagnostic only. The supplied original NativeAttentionCapture owns
all captures. Reference calls are offline, after the DT invocation has exited.
They keep every valid Q and K token; masked KV carrier holes are not context.
CPU snapshot copies and reference work are not a speed measurement.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import inspect
import json
import math
from pathlib import Path


OFFICIAL_FA_SHA256 = 'a290e11cbcb2e65fe7b8399d42eae3bb5c4113bbc12e6190cd7f710ad70abca9'
EXISTING_CHECKER_SHA256 = '2ebd8b4963e1c8413b0595eebe64bd1d6c1d80cc3dbac49a1e623adda4e9db63'
SNAPSHOT_FIELDS = ('packed_q', 'packed_k', 'packed_v',
                   'packed_cu_seqlens_q', 'packed_cu_seqlens_k',
                   'attention_mask', 'attention_output')


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_attention_capture_observer(original_capture, *, target_module, records):
    """Extend only the diagnostic retention set on the original capture."""
    owner_path = Path(inspect.getfile(original_capture))
    profile_path = Path(inspect.getsourcefile(original_capture.profile))

    class ObservedNativeVarlenAttentionCapture(original_capture):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if self.module is target_module and self.retained_names is not None:
                self.retained_names = set(self.retained_names).union(SNAPSHOT_FIELDS)

        def __exit__(self, *exc_info):
            result = super().__exit__(*exc_info)
            if (not exc_info or exc_info[0] is None) and self.module is target_module:
                tensors = {name: self.values[name].detach().to('cpu', copy=True)
                           for name in SNAPSHOT_FIELDS}
                records.append(dict(
                    scope='Complete actual native varlen FA operands and original padded output',
                    tensors=tensors,
                    packed_arguments=dict(self.packed_arguments),
                    interface_arguments=dict(self.interface_arguments),
                    native_calls=dict(self.calls),
                    original_capture_owner=dict(path=str(owner_path), sha256=_sha(owner_path),
                        name=original_capture.__module__ + '.' + original_capture.__qualname__,
                        actual_profile_source=dict(path=str(profile_path), sha256=_sha(profile_path))),
                    source_metadata={name: dict(shape=list(self.values[name].shape),
                        stride=list(self.values[name].stride()), dtype=str(self.values[name].dtype),
                        device=str(self.values[name].device),
                        bytes=self.values[name].numel() * self.values[name].element_size())
                        for name in SNAPSHOT_FIELDS},
                    observer_source=dict(path=__file__, sha256=_sha(__file__)),
                    model_forward_calls_added=0,
                    auxiliary_FA_calls_added=0,
                    lse_saved=False,
                    lse_scope='Original capture exposes no native varlen return/LSE; none synthesized',
                ))
            return result

    return ObservedNativeVarlenAttentionCapture


def official_source_nodes(official_source, existing_checker):
    """Reuse the existing reference loader, selecting original varlen asserts."""
    existing_checker, official_source = Path(existing_checker), Path(official_source)
    if _sha(existing_checker) != EXISTING_CHECKER_SHA256:
        raise ValueError('Existing FA observer/checker source SHA256 mismatch.')
    spec = importlib.util.spec_from_file_location('_original_fa3_output_checker', existing_checker)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    functions, _, reference_provenance = helper.official_source_nodes(official_source)
    raw = official_source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != OFFICIAL_FA_SHA256:
        raise ValueError('Fixed FA test source SHA256 mismatch.')
    parsed = ast.parse(raw, filename=str(official_source))
    test = next(node for node in parsed.body if isinstance(node, ast.FunctionDef)
                and node.name == 'test_flash_attn_varlen_output')
    assertions = [node for node in test.body if isinstance(node, ast.Assert)
                  and 'out_ref' in ast.unparse(node)]
    if len(assertions) != 1:
        raise ValueError('Pinned original varlen output assertion interface changed.')
    return functions, assertions, dict(
        path=str(official_source), sha256=OFFICIAL_FA_SHA256,
        reference_loader=dict(path=str(existing_checker), sha256=EXISTING_CHECKER_SHA256),
        reference_functions=reference_provenance['reference_functions'],
        reference_ast_sha256=reference_provenance['reference_ast_sha256'],
        test_function=test.name,
        output_assertion_lines=[node.lineno for node in assertions],
        output_assertion_ast_sha256=hashlib.sha256(ast.dump(
            ast.Module(body=assertions, type_ignores=[]), include_attributes=False).encode()).hexdigest(),
        output_assertions=[ast.unparse(node) for node in assertions],
    )


def check_saved_operands(record, official_source, existing_checker, *, device='cuda'):
    """Run original references rowwise, then original whole-batch max assert.

The official causal reference assumes logical contiguous K columns. Feeding
the physical KV carrier's internal holes to it would misalign causal masks.
The actual owner-packed Q/K/V and cu_seqlens already express precisely the
logical tokens. Each row uses all of them, including its complete prefix.
"""
    import torch
    import torch.nn.functional as F
    from einops import rearrange, repeat

    functions, assertions, provenance = official_source_nodes(official_source, existing_checker)
    namespace = dict(torch=torch, F=F, math=math, rearrange=rearrange, repeat=repeat)
    filename = str(official_source)
    exec(compile(ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[])),
                 filename, 'exec'), namespace)
    assertion_code = compile(ast.fix_missing_locations(ast.Module(body=assertions, type_ignores=[])),
                             filename, 'exec')
    source, args = record['tensors'], record['packed_arguments']
    interface = record['interface_arguments']
    q, k, v = [source['packed_' + name] for name in ('q', 'k', 'v')]
    cuq, cuk = [source['packed_cu_seqlens_' + name] for name in ('q', 'k')]
    out, mask = source['attention_output'], source['attention_mask']
    if any(value.device.type != 'cpu' for value in source.values()):
        raise ValueError('Offline checker requires detached CPU snapshots.')
    if q.ndim != 3 or k.ndim != 3 or v.shape != k.shape or out.ndim != 4:
        raise ValueError('Actual packed Q/K/V or padded output representation changed.')
    if q.dtype != k.dtype or q.dtype != v.dtype or q.dtype != out.dtype:
        raise ValueError('Original native operand/output dtypes differ.')
    if q.shape[1:] != out.shape[2:] or q.shape[-1] != k.shape[-1]:
        raise ValueError('Actual native Q/K/V/output head dimensions disagree.')
    if cuq.dtype != torch.int32 or cuk.dtype != torch.int32:
        raise ValueError('Actual native cu_seqlens must be original int32 indices.')
    batch, width = out.shape[:2]
    if cuq.shape != (batch + 1,) or cuk.shape != (batch + 1,):
        raise ValueError('Actual cu_seqlens does not identify every output row.')
    if mask.ndim != 2 or mask.shape[0] != batch or mask.shape[1] < width:
        raise ValueError('Actual full attention mask does not cover the output queries.')
    if not bool(((mask == 0) | (mask == 1)).all()):
        raise ValueError('Actual native attention mask is not binary.')
    qmask = mask[:, -width:].bool()
    qlengths, klengths = cuq[1:] - cuq[:-1], cuk[1:] - cuk[:-1]
    if (int(cuq[0]) != 0 or int(cuk[0]) != 0 or int(cuq[-1]) != q.shape[0]
            or int(cuk[-1]) != k.shape[0] or not bool((qlengths > 0).all())
            or not bool((klengths > 0).all())):
        raise ValueError('Actual native cumulative sequence boundaries are invalid.')
    if not torch.equal(qmask.sum(-1).to(torch.int32), qlengths):
        raise ValueError('Actual packed Q lengths disagree with the original query mask.')
    if not torch.equal(mask.sum(-1).to(torch.int32), klengths):
        raise ValueError('Actual packed K lengths disagree with the original full mask.')
    if int(args['max_seqlen_q']) != int(qlengths.max()) or int(args['max_seqlen_k']) != int(klengths.max()):
        raise ValueError('Actual native max lengths disagree with cu_seqlens.')
    if not args['causal'] or args['dropout_p'] != 0:
        raise ValueError('Actual call is outside the recorded zero-dropout causal scope.')
    if args['softmax_scale'] not in (None, q.shape[-1] ** -.5):
        raise ValueError('Actual FA scale differs from the original reference scale.')
    if interface['sliding_window'] is not None or interface['softcap'] not in (None, 0.0):
        raise ValueError('Actual interface uses a window or softcap outside this observation scope.')
    if record['native_calls'] != dict(module=1, interface=1, native_varlen=1, native_dense=0):
        raise ValueError('Observation did not cover exactly the original varlen mixer call.')

    # Keep original full padded output arrays on CPU. Only one complete logical
    # row's attention reference occupies the device; no context window is cut.
    out_ref, out_pt = torch.zeros_like(out), torch.zeros_like(out)
    rows = []
    with torch.no_grad():
        for row in range(batch):
            rq, rk, rv = [value[int(cu[row]):int(cu[row + 1])].unsqueeze(0).to(device)
                          for value, cu in ((q, cuq), (k, cuk), (v, cuk))]
            positions = qmask[row].nonzero(as_tuple=False).flatten()
            ref, attention = namespace['attention_ref'](rq, rk, rv, causal=True)
            out_ref[row, positions] = ref[0].to('cpu')
            del ref, attention
            ordinary, attention = namespace['attention_ref'](
                rq, rk, rv, causal=True, upcast=False, reorder_ops=True)
            out_pt[row, positions] = ordinary[0].to('cpu')
            del ordinary, attention, rq, rk, rv
            rows.append(dict(row=row, complete_query_tokens=int(qlengths[row]),
                             complete_key_tokens=int(klengths[row])))
        report = dict(
            scope='Original native varlen FA forward output; unchanged official whole-batch assertion',
            official_source=provenance, original_capture_owner=record['original_capture_owner'],
            native_arguments=args, interface_arguments=interface,
            actual_tensor_metadata=record['source_metadata'], rows=rows,
            native_max_error=float((out - out_ref).abs().max()),
            original_low_precision_max_error=float((out_pt - out_ref).abs().max()),
            reference_execution='B1 complete logical rows; full original padded outputs merged before the assertion',
            causal_alignment='Original bottom-right construct_local_mask on actual packed contiguous logical tokens',
            reference_tf32=torch.backends.cuda.matmul.allow_tf32,
            additional_native_forward_calls=0, gradients_checked=False,
            finite_operation_checked=False, whole_dt_acceptance=False,
        )
        try:
            exec(assertion_code, dict(out=out, out_ref=out_ref, out_pt=out_pt))
            report['original_fa_varlen_output_assertion'] = 'passed'
        except AssertionError as error:
            report['original_fa_varlen_output_assertion'] = 'failed'
            report['failure'] = str(error)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--operands', type=Path, required=True)
    parser.add_argument('--official-source', type=Path, required=True)
    parser.add_argument('--existing-checker', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    import torch
    record = torch.load(args.operands, map_location='cpu', weights_only=True)
    result = check_saved_operands(record, args.official_source, args.existing_checker, device=args.device)
    result['operands'] = dict(path=str(args.operands), sha256=_sha(args.operands))
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf8')
    if result['original_fa_varlen_output_assertion'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()

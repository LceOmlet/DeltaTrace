"""Isolated passive FA3 operands plus the fixed official FA output check.

Snapshots are diagnostic CPU copies after the original capture exits. They
contain complete actual Q/K/V and output, never query-window or KV truncation.
The caller runs the reference only after its attribute invocation has returned.
No model, FA, finite operator, tolerance or capture owner is replaced here.
"""
from __future__ import annotations

import ast
import hashlib
import math
from pathlib import Path


OFFICIAL_FA_SHA256 = 'a290e11cbcb2e65fe7b8399d42eae3bb5c4113bbc12e6190cd7f710ad70abca9'
SNAPSHOT_FIELDS = ('dense_q', 'dense_k', 'dense_v', 'attention_output')


def make_attention_capture_observer(original_capture, *, target_module, records):
    """Subclass exactly the supplied original capture; observe one module."""
    class ObservedNativeDenseAttentionCapture(original_capture):
        def __exit__(self, *exc_info):
            result = super().__exit__(*exc_info)
            if (not exc_info or exc_info[0] is None) and self.module is target_module:
                tensors = {name: self.values[name].detach().to('cpu', copy=True)
                           for name in SNAPSHOT_FIELDS}
                records.append({
                    'scope': 'complete_actual_original_dense_FA_inputs_and_output',
                    'tensors': tensors,
                    'dense_arguments': dict(self.dense_arguments),
                    'native_calls': dict(self.calls),
                    'original_capture_type': original_capture.__module__ + '.' + original_capture.__qualname__,
                    'source_metadata': {name: {
                        'shape': list(self.values[name].shape),
                        'stride': list(self.values[name].stride()),
                        'dtype': str(self.values[name].dtype),
                        'device': str(self.values[name].device),
                        'bytes': self.values[name].numel() * self.values[name].element_size(),
                    } for name in SNAPSHOT_FIELDS},
                    'snapshot_copies': len(SNAPSHOT_FIELDS),
                    'model_forward_calls_added': 0,
                })
            return result
    return ObservedNativeDenseAttentionCapture


def official_source_nodes(official_source):
    """Return the unmodified pinned reference and output assertion AST."""
    path = Path(official_source)
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != OFFICIAL_FA_SHA256:
        raise ValueError(f'Fixed FA test source SHA256 mismatch: {actual}')
    parsed = ast.parse(raw.decode('utf8'), filename=str(path))
    functions = [n for n in parsed.body if isinstance(n, ast.FunctionDef)
                 and n.name in ('attention_ref', 'construct_local_mask')]
    if {n.name for n in functions} != {'attention_ref', 'construct_local_mask'}:
        raise ValueError('Pinned FA source is missing its original reference functions.')
    test = next(n for n in parsed.body if isinstance(n, ast.FunctionDef)
                and n.name == 'test_flash_attn_output')
    assertions = [n for n in test.body if isinstance(n, ast.Assert)
                  and 'out_ref' in ast.unparse(n)]
    if not assertions:
        raise ValueError('Pinned FA source is missing its original output assertion.')
    return functions, assertions, {
        'path': str(path), 'sha256': actual,
        'reference_functions': [n.name for n in functions],
        'reference_ast_sha256': hashlib.sha256(ast.dump(
            ast.Module(body=functions, type_ignores=[]), include_attributes=False).encode()).hexdigest(),
        'output_assertion_ast_sha256': hashlib.sha256(ast.dump(
            ast.Module(body=assertions, type_ignores=[]), include_attributes=False).encode()).hexdigest(),
        'output_assertions': [ast.unparse(n) for n in assertions],
    }


def check_saved_operands(record, official_source, *, device='cuda'):
    """Use complete recorded operands and unchanged official reference/asserts.

Call this after runner.attribute returns and its root tape has been cleared.
This checks the original native FA output only, with no extra native forward.
"""
    import torch
    import torch.nn.functional as F
    from einops import rearrange, repeat

    functions, assertions, provenance = official_source_nodes(official_source)
    namespace = dict(torch=torch, F=F, math=math, rearrange=rearrange, repeat=repeat)
    filename = str(official_source)
    exec(compile(ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[])),
                 filename, 'exec'), namespace)
    assertion_code = compile(ast.fix_missing_locations(ast.Module(body=assertions, type_ignores=[])),
                             filename, 'exec')
    arguments = record['dense_arguments']
    source = record['tensors']
    with torch.no_grad():
        q, k, v = [source[name].to(device) for name in ('dense_q', 'dense_k', 'dense_v')]
        out = source['attention_output'].to(device)
        # The pinned reference uses its original d**(-1/2) scale. The observed
        # Qwen call must use that same scale; do not rewrite the reference.
        if not arguments['causal'] or arguments['dropout_p'] != 0:
            raise ValueError('Actual FA call is outside the recorded zero-dropout causal case.')
        if arguments['softmax_scale'] not in (None, q.shape[-1] ** -.5):
            raise ValueError('Actual FA scale differs from the pinned original reference scale.')
        out_ref, attention = namespace['attention_ref'](q, k, v, causal=True)
        del attention
        out_pt, attention = namespace['attention_ref'](q, k, v, causal=True,
                                                      upcast=False, reorder_ops=True)
        del attention
        report = dict(
            scope='Original FA output with complete actual Q/K/V; fixed FA2.6.3 reference and output assertions',
            official_source=provenance,
            native_arguments=dict(arguments), native_calls=dict(record['native_calls']),
            query_shape=list(q.shape), key_shape=list(k.shape), value_shape=list(v.shape),
            output_shape=list(out.shape),
            actual_dtypes=dict(q=str(q.dtype), k=str(k.dtype), v=str(v.dtype), out=str(out.dtype)),
            actual_scale=arguments['softmax_scale'], reference_scale=q.shape[-1] ** -.5,
            causal_alignment='Unmodified official construct_local_mask bottom-right alignment',
            native_max_error=float((out - out_ref).abs().max()),
            original_low_precision_max_error=float((out_pt - out_ref).abs().max()),
            reference_tf32=torch.backends.cuda.matmul.allow_tf32,
            additional_native_forward_calls=0,
        )
        try:
            exec(assertion_code, dict(out=out, out_ref=out_ref, out_pt=out_pt))
            report['original_fa_output_assertion'] = 'passed'
        except AssertionError as error:
            report['original_fa_output_assertion'] = 'failed'
            report['failure'] = str(error)
        return report

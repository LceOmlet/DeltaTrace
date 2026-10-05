"""Passive per-sample observation of the original runner's token contraction.

Wraps only the runner module's existing _token_effect, calls it unchanged, and
returns that exact object. It does not supply an attribute observer, alter the
prefix/cache path, retain coefficient/activation tensors, or define tolerances.
Small per-sample reductions are resolved after the context restores the owner.
The extra diagnostic work is not a performance comparison.
"""
from contextlib import contextmanager
import hashlib
import inspect
import linecache
from pathlib import Path
import sys

import torch


def _call_site(caller, attribute_code):
    """Save scalar source metadata, never frames or their tensor-valued locals."""
    immediate = dict(function=caller.f_code.co_name, file=caller.f_code.co_filename,
                     line=caller.f_lineno,
                     source_line=linecache.getline(caller.f_code.co_filename, caller.f_lineno).strip())
    current = caller
    try:
        while current is not None and current.f_code is not attribute_code:
            current = current.f_back
        if current is None:
            return dict(immediate=immediate, attribute_call_site=None,
                        phase='outside_attribute', current_layer_index=None)
        source = linecache.getline(current.f_code.co_filename, current.f_lineno).strip()
        if 'seed_effect' in source:
            phase = 'seed_effect'
        elif 'root_output_effect' in source and 'replay_output_effect' in source:
            phase = 'layer_root_or_replay_output_effect_shared_line'
        elif 'root_output_effect' in source:
            phase = 'layer_root_output_effect'
        elif 'replay_output_effect' in source:
            phase = 'layer_replay_output_effect'
        elif "['input_effect']" in source:
            phase = 'layer_input_effect'
        elif 'signed=' in source or 'signed =' in source:
            phase = 'final_signed'
        else:
            phase = 'unclassified_attribute_call'
        index = current.f_locals.get('i')
        index = index if isinstance(index, int) and not isinstance(index, bool) else None
        # i=31 from root hook registration is still in locals at seed time;
        # it does not mean that the seed contraction belongs to decoder31.
        layer_index = index if phase.startswith('layer_') else None
        layer = current.f_locals.get('layer') if layer_index is not None else None
        return dict(immediate=immediate,
            attribute_call_site=dict(function=current.f_code.co_name,
                file=current.f_code.co_filename, line=current.f_lineno, source_line=source),
            phase=phase, current_layer_index=layer_index,
            attribute_local_i=index,
            block_type=getattr(layer, 'block_type', None) if layer is not None else None)
    finally:
        del current, caller


@contextmanager
def observe_token_effects(runner):
    """Yield metadata populated by one unchanged original attribute invocation.

    Expected integration: with observe_token_effects(runner) as observed:
        signed, roots, detail = trace_token_attribution(... original arguments ...)
    Read observed only after context exit; no argument or result replacement.
    """
    module = sys.modules[type(runner).__module__]
    original = module._token_effect
    attribute_code = inspect.unwrap(type(runner).attribute).__code__
    owner_path = Path(inspect.getsourcefile(original)).resolve()
    report = dict(scope=__doc__, owner_token_effect=dict(path=str(owner_path),
        sha256=hashlib.sha256(owner_path.read_bytes()).hexdigest(),
        function=original.__name__, first_line=original.__code__.co_firstlineno),
        records=[], call_count=0, diagnostic_error=None,
        retained_payload='Only per-sample scalar sums; no m, x, token vector or frame is retained.')
    small_sums = []

    def observed(m, x):
        value = original(m, x)
        # This reduction is supplemental observation of the returned native
        # token vector. The original value/object is returned untouched.
        sums = value.detach().sum(dim=-1)
        site = _call_site(sys._getframe(1), attribute_code)
        report['records'].append(dict(call_index=len(report['records']), **site,
            m_dtype=str(m.dtype), x_dtype=str(x.dtype), token_effect_dtype=str(value.dtype),
            m_shape=list(m.shape), x_shape=list(x.shape), token_effect_shape=list(value.shape),
            per_sample_sums_dtype=str(sums.dtype), per_sample_sums_shape=list(sums.shape)))
        small_sums.append(sums)
        return value

    module._token_effect = observed
    try:
        yield report
    finally:
        module._token_effect = original
        report['call_count'] = len(report['records'])
        try:
            if small_sums:
                resolved = torch.stack(small_sums).cpu().tolist()
                for row, sums in zip(report['records'], resolved):
                    row['per_sample_sums'] = sums
        except Exception as exc:
            # Preserve any original runner exception; a failed optional
            # observation must not replace it or fabricate finite sums.
            report['diagnostic_error'] = repr(exc)
        finally:
            small_sums.clear()

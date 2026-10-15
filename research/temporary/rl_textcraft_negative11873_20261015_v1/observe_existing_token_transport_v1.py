"""Compose the existing passive contraction observer with bounded CPU payloads.

No attribute observer is supplied: the production cache/offload paths remain.
Each original contraction result is returned unchanged. Only existing tensor
values and one source-position coefficient/endpoint are copied to CPU.
"""
from contextlib import contextmanager
import inspect,sys
import torch
from observe_textcraft_finite_effects import observe_token_effects

@contextmanager
def observe_transport(runner,focus_row,focus_position):
    module=sys.modules[type(runner).__module__]
    attribute_code=inspect.unwrap(type(runner).attribute).__code__
    payload=[]
    with observe_token_effects(runner) as metadata:
        original_observed=module._token_effect
        def observed(m,x):
            result=original_observed(m,x)
            frame=sys._getframe(1)
            try:
                while frame is not None and frame.f_code is not attribute_code:
                    frame=frame.f_back
                if frame is None:
                    return result
                local=frame.f_locals
                starts=local.get('row_starts')
                prefix=local.get('prefix_start',0)
                start=starts[focus_row] if starts is not None else prefix
                index=focus_position-start
                assert 0<=index<result.shape[1],(focus_position,start,list(result.shape))
                payload.append(dict(call_index=len(payload),
                    token_effects=result.detach().cpu(),
                    focus_coefficient=m[focus_row,index].detach().cpu(),
                    focus_endpoint0=x[2*focus_row,index].detach().cpu(),
                    focus_endpoint1=x[2*focus_row+1,index].detach().cpu(),
                    row_starts=None if starts is None else tuple(starts),
                    row_lengths=local.get('row_lengths'),native_shared_prefix=prefix,
                    focus_local_index=index,focus_original_position=focus_position,
                    attribute_original_length=local.get('original_length')))
            finally:
                del frame
            return result
        module._token_effect=observed
        try:
            yield metadata,payload
        finally:
            module._token_effect=original_observed


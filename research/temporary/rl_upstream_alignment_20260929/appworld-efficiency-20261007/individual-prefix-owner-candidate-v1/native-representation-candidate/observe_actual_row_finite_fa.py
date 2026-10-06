"""Passive copies from one actual finite-FA invocation; no arithmetic replacement."""
import hashlib
from pathlib import Path

import torch


class ActualRowFiniteFA:
    def __init__(self, operation, path):
        self.operation = operation
        self.path = Path(path)
        self.calls = 0
        self.saved = None

    def __call__(self, operands, scale, layout, activity=None):
        result = self.operation(operands, scale, layout, activity)
        self.calls += 1
        # Qwen3.5 has full-attention layers 3,7,...,31. The original finite
        # decoder visits them in reverse; invocation eight is decoder3.
        if self.calls == 8:
            payload = dict(scope=__doc__, invocation=self.calls, decoder_index=3,
                observer_source=dict(path=__file__,sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
                operands={name:value.detach().to('cpu',copy=True) for name,value in operands.items()},
                outputs={name:value.detach().to('cpu',copy=True) for name,value in result.items()},
                scale=scale, lengths=layout.lengths, padded_length=layout.padded_length,
                coefficient_starts=layout.coefficient_starts, query_starts=layout.query_starts,
                query_padded_length=layout.query_padded_length)
            with self.path.open('xb') as stream:
                torch.save(payload, stream)
            self.saved = dict(path=str(self.path),sha256=hashlib.sha256(self.path.read_bytes()).hexdigest(),
                bytes=self.path.stat().st_size, actual_dtype={k:str(v.dtype) for k,v in operands.items()},
                lengths=list(layout.lengths), query_starts=list(layout.query_starts),
                query_padded_length=layout.query_padded_length,
                checkpoint_operations=False, extra_finite_or_model_calls=0)
        return result

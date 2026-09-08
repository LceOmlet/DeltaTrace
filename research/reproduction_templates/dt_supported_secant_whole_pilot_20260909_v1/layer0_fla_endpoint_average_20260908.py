"""Endpoint-order average, explicitly bound only to GDN layer 0 by the runner.

Reuse the unchanged finite FLA twice with actual native endpoint fields and the
same BF16 upstream/scale. No hook, profiler, dispatch by count or GPU sync.
"""
import math
import torch

# Same reviewed contract as finite_fla_endpoint_order_diagnostic_20260908.py.
ENDPOINT_KEYS = frozenset(('raw_g','q','k','v','g','beta','A','w','v_new','o','h'))
COEFFICIENT_KEYS = ('q','k','v','beta','alpha','g')


class Layer0FLAEndpointAverage:
    def __init__(self, backend):
        self.backend = backend
        self.entered = self.returned = 0
        self.calls = []

    def __call__(self, endpoints, do, scale):
        self.entered += 1
        row = {'status':'entered', 'backend_calls':[]}
        self.calls.append(row)
        assert set(endpoints) == ENDPOINT_KEYS
        assert isinstance(scale, (int,float)) and math.isfinite(scale)
        q = endpoints['q']; rows = q.shape[0]
        assert rows > 0 and rows % 2 == 0 and q.is_cuda and q.dtype == torch.bfloat16 and q.shape[-1] == 128
        assert do.shape == endpoints['o'][1::2].shape and do.device == q.device and do.dtype == torch.bfloat16
        assert all(isinstance(x,torch.Tensor) and x.shape[0] == rows and x.device == q.device for x in endpoints.values())
        row.update(scale=scale, endpoint_shapes={name:list(x.shape) for name,x in endpoints.items()},
            do_shape=list(do.shape), do_dtype=str(do.dtype), endpoint_permutation=[i^1 for i in range(rows)])
        def call(actual_endpoints, orientation):
            item = {'orientation':orientation,'status':'entered'}
            row['backend_calls'].append(item)
            value = self.backend(actual_endpoints, do, scale)
            item['status'] = 'returned'
            return value
        forward = call(endpoints, 'original')
        # Exchange endpoint rows, including chunk state h; preserve token order.
        order = torch.arange(rows,device=q.device).reshape(-1,2).flip(1).flatten()
        reverse_endpoints = {name:x.index_select(0,order) for name,x in endpoints.items()}
        reverse = call(reverse_endpoints, 'swapped')
        assert set(forward) == set(reverse) == set(COEFFICIENT_KEYS)
        assert all(forward[name].shape == reverse[name].shape and
            forward[name].dtype == reverse[name].dtype == torch.float32 and
            forward[name].device == reverse[name].device == q.device for name in COEFFICIENT_KEYS)
        # Both orientations return multipliers, so there is no sign flip.
        # alpha and g are alternative decay coordinates; normal GDN consumes g.
        result = {name:(forward[name]+reverse[name])*0.5 for name in COEFFICIENT_KEYS}
        row['coefficient_shapes'] = {name:list(x.shape) for name,x in result.items()}
        # The original runner checks every resulting decoder coefficient for
        # finiteness. Do not add diagnostic contractions or extra GPU sync.
        self.returned += 1
        row['status'] = 'returned'
        return result

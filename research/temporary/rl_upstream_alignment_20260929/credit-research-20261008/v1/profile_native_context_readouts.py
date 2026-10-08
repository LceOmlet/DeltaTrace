"""Measure original readout calls against the unchanged full finite FLA owner.

Saved real whole-sequence operands, native B4/head8, identical GPU residency.
No new recurrence, model, DT, estimator, or whole-method speed claim.
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import time

import torch
from fla.ops.common.chunk_o import chunk_fwd_o
from finite_fla_gpu import make_compiled_finite_pullback
from profiles.qwen35_gdn_symmetric import average_memory_endpoint_orders


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('operands', 'precast', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    saved = torch.load(args.operands, map_location='cpu', weights_only=False, mmap=True)
    before = torch.load(args.precast, map_location='cpu', weights_only=False, mmap=True)
    raw, old_seed, scale = saved['args']
    group = before['mo'][:, :, 8:16].to(before['output_dtype'])
    assert torch.equal(group.to(old_seed.dtype), old_seed)
    exponent = torch.ceil(torch.log2(group.float().abs().amax((1, 3), keepdim=True)
                                     /torch.finfo(old_seed.dtype).max).clamp_min(0)).int()
    seed = torch.ldexp(group.float(), -exponent).to(old_seed.dtype).contiguous().cuda().contiguous()
    del before, group
    tick = time.perf_counter()
    endpoints = {key: value.contiguous().cuda().contiguous() for key, value in raw.items()}
    factual = {key: value[1::2].contiguous() for key, value in endpoints.items()}
    torch.cuda.synchronize()
    preparation = time.perf_counter()-tick
    owner = average_memory_endpoint_orders(make_compiled_finite_pullback(
        reuse_scalar_products=False, dynamic_shapes=True,
        compiler_options={'triton.cudagraphs': False, 'max_autotune': False}))
    h_transposed = factual['h'].transpose(-1, -2).contiguous()

    def original_finite():
        return owner(endpoints, seed, scale)

    def original_state_readout():
        return chunk_fwd_o(q=factual['q'], k=factual['k'], v=factual['v_new'],
                           h=factual['h'], g=factual['g'], scale=scale)

    def original_transposed_readout():
        return chunk_fwd_o(q=seed, k=factual['v_new'], v=factual['k'],
                           h=h_transposed, g=factual['g'], scale=scale)

    result = dict(scope=__doc__, script=dict(path=__file__, sha256=sha(__file__)),
        operands=dict(path=str(args.operands), sha256=sha(args.operands)),
        precast=dict(path=str(args.precast), sha256=sha(args.precast)),
        native_readout=dict(path=importlib.import_module('fla.ops.common.chunk_o').__file__,
                            sha256=sha(importlib.import_module('fla.ops.common.chunk_o').__file__)),
        finite_owner=dict(path=importlib.import_module('finite_fla_gpu').__file__,
                          sha256=sha(importlib.import_module('finite_fla_gpu').__file__)),
        factual_shape=list(factual['q'].shape), dtype=str(seed.dtype),
        preparation_seconds=preparation, timings={}, model_calls=0, DT_calls=0, optimizer=0,
        production_modified=False, full_conditional_estimator=False)
    with torch.no_grad():
        for name, operation in [('original_symmetric_finite_FLA', original_finite),
                                ('original_chunk_state_readout', original_state_readout),
                                ('original_transposed_state_readout', original_transposed_readout)]:
            torch.cuda.synchronize()
            tick = time.perf_counter()
            value = operation()
            torch.cuda.synchronize()
            cold = time.perf_counter()-tick
            del value
            times = []
            for _ in range(3):
                torch.cuda.synchronize()
                tick = time.perf_counter()
                value = operation()
                torch.cuda.synchronize()
                times.append(time.perf_counter()-tick)
                del value
            result['timings'][name] = dict(first_call_seconds=cold, repeated_seconds=times)
            args.output.write_text(json.dumps(result, indent=2)+'\n')
            print(json.dumps(dict(name=name, **result['timings'][name])), flush=True)
    result.update(status='complete_bounded_native_primitive_profile',
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                  limits='This compares original kernels on resident saved operands only. Full conditional-window coefficients, query count, layouts, projections, model replay and overall DT cost are not measured or accepted.')
    args.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()

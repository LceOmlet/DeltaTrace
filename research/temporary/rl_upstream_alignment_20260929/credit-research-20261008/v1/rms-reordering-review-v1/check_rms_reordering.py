"""CPU-only operator review of an algebraically equivalent RMS finite rule.

This measures arithmetic on the actual owner functions, not whole-model DT
quality, an official FA/FLA tolerance certificate, or a training result.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import psutil
import torch


def ref(path):
    raw = path.read_bytes()
    return dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.rmsnorm_secant_pullback


def rms_inputs(x0, x1, eps):
    r0 = (x0.square().mean(-1, keepdim=True)+eps).sqrt()
    r1 = (x1.square().mean(-1, keepdim=True)+eps).sqrt()
    return r0, r1


def error(value, reference):
    delta = value.double()-reference
    return dict(maxabs=float(delta.abs().max()),
        relative_L2=float(delta.norm()/reference.norm()) if reference.norm() else None,
        nonfinite=int((~torch.isfinite(value)).sum()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('original', 'candidate', 'precast', 'output'):
        parser.add_argument('--'+name, required=True, type=Path)
    args = parser.parse_args()
    start = time.perf_counter()
    torch.set_num_threads(2)
    assert not torch.cuda.is_initialized()
    original, candidate = load(args.original, 'original_rms_owner'), load(args.candidate, 'candidate_rms_owner')
    generator = torch.Generator().manual_seed(20261009)
    cases = []
    # These are arithmetic unit cases, not substitutes for task/author tests.
    for dimension in (128, 256, 4096):
        for scale in (0., 1e-12, 1e-6, 1e-3, 1., 1e3):
            base = torch.randn(3, dimension, generator=generator)
            other = torch.randn(3, dimension, generator=generator)
            for geometry in ('equal', 'opposite', 'unequal', 'near_equal'):
                x0 = base*scale
                x1 = {'equal':x0, 'opposite':-x0, 'unequal':other*(scale*.125),
                      'near_equal':x0+other*(scale*1e-4)}[geometry]
                for input_dtype in (torch.float16, torch.bfloat16, torch.float32):
                    # Production callers promote endpoints/weights to FP32;
                    # preserve the input quantization without doing RMS in FP16.
                    x0f, x1f = x0.to(input_dtype).float(), x1.to(input_dtype).float()
                    weight = torch.randn(dimension, generator=generator).to(input_dtype).float()
                    upstream = torch.randn(3, dimension, generator=generator)*100
                    for eps in (1e-6, 1e-6/dimension):
                        r0, r1 = rms_inputs(x0f.double(), x1f.double(), eps)
                        old64 = original(x0f.double(), x1f.double(), weight.double(), upstream.double(), eps)
                        new64 = candidate(x0f.double(), x1f.double(), weight.double(), upstream.double(), eps)
                        old32 = original(x0f, x1f, weight, upstream, eps)
                        new32 = candidate(x0f, x1f, weight, upstream, eps)
                        delta = x1f.double()-x0f.double()
                        endpoint = ((x1f.double()/r1-x0f.double()/r0)*weight.double()*upstream.double()).sum(-1)
                        contraction = (new64*delta).sum(-1)
                        u = (x0f.double()+x1f.double())/(r0+r1)
                        record = dict(dimension=dimension, scale=scale, geometry=geometry,
                            input_quantization=str(input_dtype), analysis_dtype='FP32', eps=eps,
                            original_FP32=error(old32, old64), candidate_FP32=error(new32, old64),
                            FP64_algebraic_difference=error(new64, old64),
                            finite_endpoint_contraction_maxabs=float((contraction-endpoint).abs().max()),
                            normalized_u_RMS_max=float(u.square().mean(-1).sqrt().max()),
                            inverse_mean_max=float(((1/r0+1/r1)*.5).max()),
                            old_inverse_secant_maxabs=float((1/(r0*r1*(r0+r1))).abs().max()))
                        if geometry == 'equal':
                            xd = x0f.double().requires_grad_()
                            normalized = xd/(xd.square().mean(-1, keepdim=True)+eps).sqrt()*weight.double()
                            derivative = torch.autograd.grad((normalized*upstream.double()).sum(), xd)[0]
                            record['coincident_vs_Torch_FP64_autograd'] = error(new64, derivative)
                        cases.append(record)
    precast = torch.load(args.precast, map_location='cpu', mmap=True, weights_only=False)
    mo = precast['mo']
    saved_value = float(mo[0, 3735, 12, 79])
    pss = psutil.Process().memory_full_info().pss
    result = dict(scope=__doc__, version='rms-normalized-secant-20261009-v1',
        status='CPU_arithmetic_review_complete_not_deployed',
        sources=[ref(p) for p in (args.original, args.candidate, Path(__file__))],
        precast=dict(path=str(args.precast), bytes=args.precast.stat().st_size,
            position=[0, 3735, 12, 79], FP32_value=saved_value,
            saved_BF16_value=float(torch.tensor(saved_value).to(torch.bfloat16)),
            limitation='This file saves the output seed only, not RMS endpoints or its incoming weighted vector. '
                'The quoted 0.813/582/1.03e-7 reconstructed example cannot be independently reproduced from this artifact alone.'),
        cases=cases, summary=dict(cases=len(cases),
            original_nonfinite=sum(c['original_FP32']['nonfinite'] for c in cases),
            candidate_nonfinite=sum(c['candidate_FP32']['nonfinite'] for c in cases),
            FP64_equivalence_relative_L2_max=max(c['FP64_algebraic_difference']['relative_L2'] or 0 for c in cases),
            original_FP32_relative_L2_max=max(c['original_FP32']['relative_L2'] or 0 for c in cases),
            candidate_FP32_relative_L2_max=max(c['candidate_FP32']['relative_L2'] or 0 for c in cases),
            candidate_more_accurate_cases=sum((c['candidate_FP32']['relative_L2'] or 0) <
                                            (c['original_FP32']['relative_L2'] or 0) for c in cases),
            candidate_less_accurate_cases=sum((c['candidate_FP32']['relative_L2'] or 0) >
                                            (c['original_FP32']['relative_L2'] or 0) for c in cases),
            sampled_process_PSS_bytes=pss),
        official_tolerance_claim=False, production_modified=False,
        operations=dict(model=0, DT=0, FA=0, FLA=0, GPU=0, optimizer=0),
        elapsed_seconds=time.perf_counter()-start)
    assert not torch.cuda.is_initialized()
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output=ref(args.output), summary=result['summary'],
        precast=result['precast'], elapsed_seconds=result['elapsed_seconds'])))


if __name__ == '__main__':
    main()

"""Measure existing finite FA versus installed FA backward at Qwen's shape.

Synthetic operator operands; this is a cost diagnostic, not a correctness or
whole-model training claim. Candidate libraries are never installed by this tool.
"""
import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch
from flash_attn import flash_attn_func


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--candidate', type=Path)
    parser.add_argument('--candidate-only', action='store_true',
                        help='Compare candidate with native backward; reuse an already measured current-owner baseline.')
    parser.add_argument('--length', type=int, default=32768)
    parser.add_argument('--batches', type=int, nargs='+', default=[1, 4])
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    env = json.loads(Path(os.environ['DT_ENVIRONMENT_JSON']).read_text())['qwen35']
    sys.path.insert(0, str(Path(os.environ['DT_ROOT'])/'clean/qwen35'))
    from vendor_fa_finite_bf16_d256 import VendorFAFiniteP1BF16D256, RightPaddedLengths
    methods = {'current': VendorFAFiniteP1BF16D256(env['finite_library'], env['finite_library_sha256'])}
    if args.candidate:
        methods['candidate'] = VendorFAFiniteP1BF16D256(args.candidate,
            hashlib.sha256(args.candidate.read_bytes()).hexdigest())
    if args.candidate_only:
        if not args.candidate:
            parser.error('--candidate-only requires --candidate')
        del methods['current']
    result = dict(scope=__doc__, length=args.length, records=[], status='running')
    def save():
        args.output.write_text(json.dumps(result, indent=2)+'\n')
    def timed(fn):
        torch.cuda.synchronize()
        start = time.perf_counter()
        value = fn()
        torch.cuda.synchronize()
        return value, time.perf_counter()-start
    for batch in args.batches:
        torch.manual_seed(2026)
        q0 = torch.randn(batch, args.length, 16, 256, device='cuda', dtype=torch.bfloat16)
        k0 = torch.randn(batch, args.length, 4, 256, device='cuda', dtype=torch.bfloat16)
        v = torch.randn_like(k0)
        q1 = (q0 + .1*torch.randn_like(q0)).detach().requires_grad_()
        k1 = (k0 + .1*torch.randn_like(k0)).detach().requires_grad_()
        v.requires_grad_()
        upstream = torch.randn_like(q1)
        with torch.no_grad():
            _, lse0, _ = flash_attn_func(q0, k0, v, causal=True, return_attn_probs=True)
        output, lse1, _ = flash_attn_func(q1, k1, v, causal=True, return_attn_probs=True)
        operands = {key:value.transpose(1, 2).detach() for key,value in
                    [('q0', q0), ('q1', q1), ('k0', k0), ('k1', k1), ('v0', v)]}
        operands.update(u=upstream.transpose(1, 2).float(), lse0=lse0, lse1=lse1)
        layout = RightPaddedLengths([args.length]*batch, args.length, q0.device)
        owners = dict(methods)
        owners['native_fa_backward'] = lambda *_: torch.autograd.grad(
            output, (q1, k1, v), upstream, retain_graph=True)
        for name, function in owners.items():
            seconds = []
            for repeat in range(args.repeats):
                value, elapsed = timed(lambda: function(operands, .0625, layout))
                seconds.append(elapsed)
                del value
                print('FA_SAMPLE', dict(batch=batch, operation=name, repeat=repeat, seconds=elapsed), flush=True)
            row = dict(batch=batch, operation=name, seconds=seconds,
                       warm_median_seconds=statistics.median(seconds[1:]) if len(seconds)>1 else None)
            result['records'].append(row)
            save()
            print('FA_COST', row, flush=True)
            if args.profile:
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                       torch.profiler.ProfilerActivity.CUDA]) as prof:
                    value = function(operands, .0625, layout)
                    torch.cuda.synchronize()
                    del value
                prof.export_chrome_trace(str(args.output.with_name(f'{args.output.stem}-{batch}-{name}-trace.json')))
                print(prof.key_averages().table(sort_by='self_cuda_time_total', row_limit=15), flush=True)
        del owners, output, lse0, lse1, operands, q0, q1, k0, k1, v, upstream
        torch.cuda.empty_cache()
    result['status'] = 'timed_not_numerically_validated'
    save()


if __name__ == '__main__':
    main()

"""Compare restricted finite-FA outputs with the same full owner operation.

All K/V positions remain present. This tests an operator output restriction,
not whether a model caller may discard a common causal prefix. The latter
requires full signed-attribution parity at the runner boundary.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from flash_attn import flash_attn_func
from vendor_fa_finite_bf16_d256 import RightPaddedLengths, VendorFAFiniteP1BF16D256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cached-only', action='store_true')
    args = parser.parse_args()
    digest = hashlib.sha256(args.candidate.read_bytes()).hexdigest()
    owner = VendorFAFiniteP1BF16D256(args.candidate, digest)
    result = dict(scope=__doc__, library_sha256=digest, cases=[], status='running')

    def timed(operands, layout):
        torch.cuda.synchronize()
        start = time.perf_counter()
        value = owner(operands, .0625, layout)
        torch.cuda.synchronize()
        return value, time.perf_counter()-start

    cases=[
            (128, [128, 97, 66, 1], [0, 1, 64, 1]),
            (447, [447, 389, 321, 256], [64, 65, 321, 128]),
            (32768, [32768]*4, [31573]*4)]
    if args.cached_only:cases=[(128,[128]*4,[64,65,100,128]),(447,[447,389,321,256],[64,65,321,128]),(32768,[32768]*4,[31573]*4)]
    for length, lengths, starts in cases:
        for coincident in ([True, False] if length < 32768 else [False]):
            torch.manual_seed(2026)
            q0 = torch.randn(4, length, 16, 256, device='cuda', dtype=torch.bfloat16)
            k0 = torch.randn(4, length, 4, 256, device='cuda', dtype=torch.bfloat16)
            q1 = q0 if coincident else q0+.1*torch.randn_like(q0)
            k1 = k0 if coincident else k0+.1*torch.randn_like(k0)
            v = torch.randn_like(k0)
            upstream = torch.randn_like(q0)
            _, lse0, _ = flash_attn_func(q0, k0, v, causal=True, return_attn_probs=True)
            _, lse1, _ = flash_attn_func(q1, k1, v, causal=True, return_attn_probs=True)
            operands = {k: x.transpose(1, 2) for k, x in
                        [('q0',q0), ('q1',q1), ('k0',k0), ('k1',k1), ('v0',v)]}
            operands.update(u=upstream.transpose(1, 2).float(), lse0=lse0, lse1=lse1)
            full = RightPaddedLengths(lengths, length, q0.device)
            suffix = RightPaddedLengths(lengths, length, q0.device, coefficient_starts=starts)
            reference, full_cold = timed(operands, full)
            actual, suffix_cold = timed(operands, suffix)
            for key in reference:
                for index, start in enumerate(starts):
                    # tau/center belong to query rows, dq/dk/dv to token rows.
                    assert torch.equal(reference[key][index,:,start:], actual[key][index,:,start:]), key
                    if key in ('dq','dk','dv'):
                        assert not torch.count_nonzero(actual[key][index,:,:start]), key
            del actual
            entry = dict(length=length, valid_lengths=lengths, coefficient_starts=starts,
                coincident=coincident, suffix_values_exact=True, omitted_coefficients_zero=True,
                full_cold_seconds=full_cold, suffix_cold_seconds=suffix_cold, warm=[])
            if args.cached_only:
                cut=min(starts)//64*64
                cached_layout=RightPaddedLengths(lengths,length,q0.device,coefficient_starts=starts,query_start=cut)
                cached_operands={k:(v[:,:,cut:] if k in ('q0','q1','u','lse0','lse1') else v) for k,v in operands.items()}
                cached,cached_seconds=timed(cached_operands,cached_layout)
                for key in reference:
                    for index,start in enumerate(starts):
                        assert torch.equal(reference[key][index,:,start:],cached[key][index,:,start-cut:]),('cached',key,index)
                entry['cached_queries']=dict(start=cut,seconds=cached_seconds,required_outputs_exact=True)
                del cached,cached_operands
            if length == 32768:
                for name, layout in [('full',full), ('suffix',suffix), ('suffix',suffix), ('full',full)]:
                    value, seconds = timed(operands, layout)
                    entry['warm'].append(dict(name=name, seconds=seconds))
                    del value
            result['cases'].append(entry)
            args.output.write_text(json.dumps(result, indent=2)+'\n')
            print('FINITE_FA_SUFFIX', json.dumps(entry), flush=True)
            del reference, operands, q0, q1, k0, k1, v, upstream, lse0, lse1
    result['status'] = 'passed_output_restriction_parity'
    args.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()

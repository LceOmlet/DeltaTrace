"""Exercise a candidate finite FA library through the pinned FA test.

At coincident endpoints finite coefficients are ordinary attention gradients.
The test-only bridge substitutes that backward into FA's unchanged test, using
its FP32 reference, low-precision baseline, and original assertions. Nonzero
endpoint comparisons against the installed finite owner are diagnostics only.
No candidate library is installed or selected by training here.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import sys
import traceback
from pathlib import Path

import torch
from flash_attn import flash_attn_func


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--sources', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    source = args.sources/'test_flash_attn_v263.py'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == 'a290e11cbcb2e65fe7b8399d42eae3bb5c4113bbc12e6190cd7f710ad70abca9'
    spec = importlib.util.spec_from_file_location('official_fa_test', source)
    test = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(test)
    sys.path.insert(0, str(Path(os.environ['DT_ROOT'])/'clean/qwen35'))
    from vendor_fa_finite_bf16_d256 import VendorFAFiniteP1BF16D256, RightPaddedLengths
    env = json.loads(Path(os.environ['DT_ENVIRONMENT_JSON']).read_text())['qwen35']
    owners = dict(current=VendorFAFiniteP1BF16D256(env['finite_library'], env['finite_library_sha256']),
                  candidate=VendorFAFiniteP1BF16D256(args.candidate, hashlib.sha256(args.candidate.read_bytes()).hexdigest()))
    result = dict(scope=__doc__, cases=[], finite_pair_differences=[], candidate_sha256=hashlib.sha256(args.candidate.read_bytes()).hexdigest())

    def bridge(owner):
        class FiniteGradient(torch.autograd.Function):
            @staticmethod
            def forward(ctx, q, k, v, options):
                out, lse, mask = flash_attn_func(q, k, v, **options)
                ctx.save_for_backward(q, k, v, lse)
                ctx.mark_non_differentiable(*(x for x in (lse, mask) if isinstance(x, torch.Tensor)))
                return out, lse, mask

            @staticmethod
            def backward(ctx, upstream, _lse, _mask):
                q, k, v, lse = ctx.saved_tensors
                ops = dict(q0=q.transpose(1, 2), q1=q.transpose(1, 2),
                           k0=k.transpose(1, 2), k1=k.transpose(1, 2),
                           v0=v.transpose(1, 2), u=upstream.transpose(1, 2).float(), lse0=lse, lse1=lse)
                coefficients = owner(ops, q.shape[-1]**-.5, RightPaddedLengths([q.shape[1]]*q.shape[0], q.shape[1], q.device))
                def reduced(name):
                    b, h, t, d = coefficients[name].shape
                    return coefficients[name].float().reshape(b, k.shape[2], h//k.shape[2], t, d).sum(2).transpose(1, 2).to(k.dtype)
                return coefficients['dq'].transpose(1, 2), reduced('dk'), reduced('dv'), None

        def call(q, k, v, dropout_p=0., **options):
            assert dropout_p == 0 and options['causal'] and options['return_attn_probs']
            return FiniteGradient.apply(q, k, v, dict(options, dropout_p=dropout_p))
        return call

    for length in (128, 447):
        for name, owner in owners.items():
            entry = dict(owner=name, length=length, function='test_flash_attn_output', endpoint_difference=0)
            test.flash_attn_func = bridge(owner)
            try:
                test.test_flash_attn_output(seqlen_q=length, seqlen_k=length, d=256,
                    dropout_p=0., causal=True, local=False, alibi=False, deterministic=True,
                    mha_type='gqa', dtype=torch.bfloat16, kvpacked=False, softcap=0.)
                entry['status'] = 'passed'
            except Exception as exc:
                entry.update(status='failed', error=str(exc), traceback=traceback.format_exc())
                traceback.print_exc()
            result['cases'].append(entry)
            print(entry, flush=True)
            args.output.write_text(json.dumps(result, indent=2)+'\n')
        torch.manual_seed(2026)
        q0 = torch.randn(4, length, 16, 256, device='cuda', dtype=torch.bfloat16)
        k0 = torch.randn(4, length, 4, 256, device='cuda', dtype=torch.bfloat16)
        q1 = q0 + .1*torch.randn_like(q0)
        k1 = k0 + .1*torch.randn_like(k0)
        v = torch.randn_like(k0)
        upstream = torch.randn_like(q0)
        _, lse0, _ = flash_attn_func(q0, k0, v, causal=True, return_attn_probs=True)
        _, lse1, _ = flash_attn_func(q1, k1, v, causal=True, return_attn_probs=True)
        ops = {key: value.transpose(1, 2) for key, value in [('q0',q0),('q1',q1),('k0',k0),('k1',k1),('v0',v)]}
        ops.update(u=upstream.transpose(1, 2).float(), lse0=lse0, lse1=lse1)
        values = {name: fn(ops, .0625, RightPaddedLengths([length]*4, length, q0.device)) for name,fn in owners.items()}
        result['finite_pair_differences'].append(dict(length=length, values={key: dict(
            max_abs=float((values['current'][key].float()-value.float()).abs().max()),
            relative_l2=float((values['current'][key].float()-value.float()).norm()/values['current'][key].float().norm()))
            for key, value in values['candidate'].items()}))
    result['status'] = 'passed_coincident_endpoint_official_tests' if all(x['status']=='passed' for x in result['cases']) else 'failed'
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    if result['status'] == 'failed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()

"""Audit decay attribution using the installed native FLA and frozen DT.

Four analytic operator cases, not a benchmark or model substitute. No explicit
attention/state recurrence is used for native outputs or DT coefficients.
"""
import argparse
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('release', 'environment', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    sha = lambda b: hashlib.sha256(b).hexdigest()
    env = json.loads(args.environment.read_bytes())['qwen35']
    sources = json.loads((args.release / 'deltatrace/clean/sources.json').read_bytes())
    for name, record in sources['models']['qwen35']['files'].items():
        assert sha((args.release / name).read_bytes()) == record['sha256'], name
    os.environ.update(MACA_PATH='/opt/maca', TRITON_ENABLE_PERSISTENT_AUTOTUNE_CONFIGS='0')
    for path in env.get('dependency_overlays', []):
        sys.path.insert(0, path)
    sys.path.insert(0, str(args.release / 'deltatrace/clean/qwen35'))
    import torch
    import fla.ops.gated_delta_rule.chunk as chunk
    from finite_fla_gpu import finite_fla_pullback, verify_native_sources
    verify_native_sources(env['native_stage_source_sha256'])
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    out = {'status': 'running', 'scope': 'native FLA operator audit; no model or metric calls',
           'script_sha256': sha(Path(__file__).read_bytes()),
           'finite_source_sha256': sha((args.release / 'deltatrace/clean/qwen35/finite_fla_gpu.py').read_bytes()),
           'native_sources': env['native_stage_source_sha256'], 'dtype': 'bfloat16',
           'cases': [], 'native_forward_calls': 0, 'finite_calls': 0}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(out, indent=2, allow_nan=False) + '\n')

    def native(g, capture=False):
        batch, length, _ = g.shape
        q = torch.zeros((batch, length, 1, 128), device='cuda', dtype=torch.bfloat16)
        q[..., 0] = 1
        k = torch.zeros_like(q)
        k[:, 0, 0, 0] = 1
        k[:, 1:, 0, 1] = 1
        v = torch.zeros_like(q)
        v[:, 0, 0, 0] = 2
        beta = torch.full((batch, length, 1), .5, device='cuda', dtype=torch.float32)
        endpoints = {}
        code = inspect.unwrap(chunk.chunk_gated_delta_rule_fwd).__code__

        def observe(frame, event, value):
            if frame.f_code is code and event == 'return' and value is not None:
                assert not endpoints
                for name in ('q', 'k', 'v', 'g', 'beta', 'A', 'w', 'v_new', 'o', 'h'):
                    endpoints[name] = frame.f_locals[name].detach().clone()

        assert sys.getprofile() is None
        if capture:
            sys.setprofile(observe)
        try:
            with torch.no_grad():
                result, _ = chunk.chunk_gated_delta_rule(q, k, v, g, beta, scale=1.,
                                                         use_qk_l2norm_in_kernel=False)
        finally:
            if capture:
                sys.setprofile(None)
        out['native_forward_calls'] += 1
        if capture:
            assert len(endpoints) == 10
            endpoints['raw_g'] = g
        return result[:, -1, 0, 0].float(), endpoints

    save()
    try:
        for length, positions in ((4, (1, 2, 3)), (67, (1, 33, 66))):
            for baseline, actual in ((.1, .9), (.9, .1)):
                g = torch.zeros((2, length, 1), device='cuda', dtype=torch.float32)
                g[0, list(positions), 0] = math.log(baseline)
                g[1, list(positions), 0] = math.log(actual)
                endpoint_scores, e = native(g, capture=True)
                do = torch.zeros((1, length, 1, 128), device='cuda', dtype=torch.bfloat16)
                do[0, -1, 0, 0] = 1
                with torch.no_grad():
                    coefficients = finite_fla_pullback(e, do, 1.)
                out['finite_calls'] += 1
                attribution = coefficients['g'][0, :, 0] * (g[1, :, 0] - g[0, :, 0])
                deletion_g = g[1:].expand(4, -1, -1).clone()
                for i, position in enumerate(positions):
                    deletion_g[i + 1, position, 0] = g[0, position, 0]
                deletion_scores, _ = native(deletion_g)
                native_prefix = torch.cat([g[:, i:i+64].cumsum(1) for i in range(0, length, 64)], 1)
                analytic = [(actual-baseline)*baseline**i*actual**(2-i) for i in range(3)]
                out['cases'].append({'length': length, 'positions': positions,
                    'baseline_alpha': baseline, 'actual_alpha': actual,
                    'analytic_endpoint_outputs': [baseline**3, actual**3],
                    'native_endpoint_outputs': endpoint_scores.cpu().tolist(),
                    'analytic_content1_allocations': analytic,
                    'native_finite_allocations': attribution[list(positions)].cpu().tolist(),
                    'native_actual_and_three_deletions': deletion_scores.cpu().tolist(),
                    'native_single_deletion_effects': (deletion_scores[0]-deletion_scores[1:]).cpu().tolist(),
                    'native_closure_residual': float((attribution.sum()-(endpoint_scores[1]-endpoint_scores[0])).cpu()),
                    'native_g_vs_natural_chunk_prefix_max_abs': float((e['g']-native_prefix).abs().max().cpu())})
                save()
        assert out['native_forward_calls'] == 8 and out['finite_calls'] == 4
        out['status'] = 'complete'
    except Exception:
        out.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        save()
    print(json.dumps({'status': out['status'], 'cases': len(out['cases'])}))


if __name__ == '__main__':
    main()

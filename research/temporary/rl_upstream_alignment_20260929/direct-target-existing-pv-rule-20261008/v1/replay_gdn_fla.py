"""Replay saved original GDN30 finite operands, without a model or training.

The FP32 control changes only the finite formula's intermediate GEMM operand
rounding. Native FLA adjoint stages, captured inputs and all formulas stay the
same. This is a diagnostic control, not a new tolerance or production backend.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--launch', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    source = json.loads(a.source.read_bytes())
    launch = json.loads(a.launch.read_bytes())
    assert sha(a.source) == launch['source_sha256']
    os.environ.update(source['environment'])
    os.environ['CUDA_VISIBLE_DEVICES'] = '4'
    os.environ.pop('MACA_VISIBLE_DEVICES', None)
    candidate = launch['memory_candidate']
    root = Path(candidate['candidate_dt_root'])
    env = json.loads(Path(candidate['candidate_environment']).read_bytes())['qwen35']
    sys.path[:0] = [str(root), os.environ.get('DT_OFFICIAL_ROOT') or env['official_root'],
                    str(root / 'clean/qwen35'), *source['pythonpath'].split(':'), env['ft_extension_root']]
    import inspect
    import psutil
    import torch
    import finite_fla_gpu as owner
    from profiles import qwen35_gdn_symmetric as symmetric
    from qwen35_dense_finite_runner import _token_effect

    assert sha(owner.__file__) == 'f1555736d32974668676eff4e040a500f4c2b907e6ac1b298f01dc2ba87f7db6'
    assert sha(symmetric.__file__) == 'dd6bbfff9aae4c679439af113a160e0bd8c4cf1b19bb17cab47a04ff7b0831d0'
    original = owner._mm
    callback = symmetric.average_memory_endpoint_orders(owner.finite_fla_pullback)
    native = json.loads((a.launch.parent / 'results/rank0.json').read_bytes())['gdn_subops']['30']
    a.output.mkdir(exist_ok=False)
    report = dict(pid=os.getpid(), birth=psutil.Process().create_time(), profiles={},
        scope='Selected saved original B4 row replayed as paired B1; no model/full-DT/actor/rollout/training-backward/update. Original FLA adjoint stages do execute. Eager native formula versus FP32 intermediate-GEMM control, not a complete FP32 reference.',
        owners={name:dict(path=inspect.getfile(module),sha256=sha(inspect.getfile(module)))
                for name,module in [('finite_fla_gpu',owner),('symmetric',symmetric)]},
        allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        actual_saved_native_point=native['points'][1]['coefficient_times_single_delta'],
        official_tolerance_claim=False, production_profile_changed=False, credit_repaired=False)
    def save():
        (a.output / 'result.json').write_text(json.dumps(report,indent=2)+'\n')
    try:
        for profile in ('eager_original_native_GEMM','eager_FP32_intermediate_GEMM'):
            def fp32(a,b,dtype=None):
                return torch.bmm(a.float(),b.float())
            owner._mm = original if profile == 'eager_original_native_GEMM' else fp32
            groups = []
            report.update(phase='running',active_profile=profile)
            save()
            for artifact in native['operand_artifacts']:
                path = Path(artifact['path'])
                assert sha(path) == artifact['sha256']
                saved = torch.load(path,map_location='cpu',weights_only=True,mmap=True)
                endpoints = {k:v.to('cuda') for k,v in saved['endpoints'].items()}
                do = saved['do'].to('cuda')
                torch.cuda.synchronize()
                tick = time.perf_counter()
                with torch.no_grad():
                    result = callback(endpoints,do,saved['scale'])
                torch.cuda.synchronize()
                seconds = time.perf_counter()-tick
                offset = saved['actual_single_time_start']-saved['time_start']
                terms = {k:float(_token_effect(result[k][:,offset:],saved['actual_single'][k].to('cuda')).sum())
                         for k in ('q','k','v','g','beta')}
                comparisons = {}
                for key,value in result.items():
                    delta = value.detach().cpu().double()-saved['outputs'][key].double()
                    comparisons[key] = dict(maxabs=float(delta.abs().max()),l2=float(delta.square().sum().sqrt()),
                        original_l2=float(saved['outputs'][key].double().square().sum().sqrt()),
                        dtype=str(value.dtype))
                groups.append(dict(head_start=saved['head_start'],seconds=seconds,terms=terms,
                    compared_with_original_saved_B4=comparisons,
                    allocated_bytes=torch.cuda.memory_allocated(),pss_bytes=psutil.Process().memory_full_info().pss,
                    artifact=artifact))
                report['profiles'][profile]=dict(groups=groups,
                    contraction_including_same_residual_and_z=native['residual_skip_single_delta']+native['z_branch_single_delta']+sum(sum(g['terms'].values()) for g in groups))
                save()
                del endpoints,do,result,saved
            report['profiles'][profile]['complete']=True
            save()
        report.update(phase='complete')
        save()
    except BaseException:
        import traceback
        report.update(phase='failed',traceback=traceback.format_exc())
        save()
        raise
    finally:
        owner._mm=original


if __name__=='__main__':
    main()

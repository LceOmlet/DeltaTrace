"""Check the observed norm-return contract with original Torch components on CPU.

Two Gloo ranks and a known sharded gradient use installed _get_total_norm and
_clip_grads_with_norm_. The reference uses original unsharded clip_grad_norm_.
This does not exercise CUDA transfer, any model, PPO update or optimizer.
"""
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time

OUT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/textcraft-gradient-rootcause-20261013-v1')
DEST = OUT / 'native-dtensor-norm-contract-cpu-v75'


def worker():
    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor, Shard, distribute_tensor
    from torch.nn.utils.clip_grad import _get_total_norm, _clip_grads_with_norm_

    torch.set_num_threads(1)
    dist.init_process_group('gloo')
    try:
        mesh = init_device_mesh('cpu', (2,))
        gradient = torch.tensor([3., 4.], dtype=torch.float32)
        parameter = torch.nn.Parameter(distribute_tensor(torch.zeros_like(gradient), mesh, [Shard(0)]))
        parameter.grad = distribute_tensor(gradient.clone(), mesh, [Shard(0)])
        norm = _get_total_norm([parameter.grad], 2.0, False, None)
        local_return = norm.item()
        _clip_grads_with_norm_([parameter], 1.0, norm, None)
        clipped_before_readout = parameter.grad.to_local().detach().clone()
        global_readout = norm.full_tensor() if isinstance(norm, DTensor) else norm
        assert torch.equal(clipped_before_readout, parameter.grad.to_local())

        reference = torch.nn.Parameter(torch.zeros_like(gradient))
        reference.grad = gradient.clone()
        reference_norm = torch.nn.utils.clip_grad_norm_([reference], 1.0)
        torch.testing.assert_close(global_readout, reference_norm)
        torch.testing.assert_close(parameter.grad.full_tensor(), reference.grad)
        assert not torch.cuda.is_initialized()
        source = Path(inspect.getsourcefile(_get_total_norm))
        record = dict(rank=dist.get_rank(), backend=dist.get_backend(),
            torch_version=torch.__version__, norm_class=type(norm).__module__ + '.' + type(norm).__qualname__,
            norm_placements=[str(p) for p in norm.placements] if isinstance(norm, DTensor) else None,
            original_item_return=local_return, materialized_global_norm=global_readout.item(),
            official_unsharded_reference_norm=reference_norm.item(),
            original_clipped_gradient=parameter.grad.full_tensor().tolist(),
            official_unsharded_reference_gradient=reference.grad.tolist(),
            reporting_materialization_did_not_change_gradient=True,
            official_assert_close_passed=True, CUDA_initialized=False,
            source=dict(path=str(source), SHA256=hashlib.sha256(source.read_bytes()).hexdigest()),
            model_calls=0, backward_calls=0, optimizer_calls=0)
        path = DEST / f'rank{dist.get_rank()}.json'
        assert not path.exists()
        path.write_text(json.dumps(record, indent=2) + '\n')
    finally:
        dist.destroy_process_group()


def main():
    assert not DEST.exists(), 'Read the existing result rather than repeat the contract check.'
    DEST.mkdir()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1')
    started = time.time()
    result = subprocess.run([sys.executable, '-m', 'torch.distributed.run', '--standalone',
        '--nnodes=1', '--nproc-per-node=2', __file__, '--worker'],
        env=env, capture_output=True, text=True, timeout=120)
    (DEST / 'stdout.log').write_text(result.stdout)
    (DEST / 'stderr.log').write_text(result.stderr)
    if result.returncode:
        raise RuntimeError('Native CPU contract check failed; preserved child logs at ' + str(DEST))
    ranks = [json.loads((DEST / f'rank{rank}.json').read_bytes()) for rank in (0, 1)]
    report = dict(scope=__doc__, unix=time.time(), elapsed_seconds=time.time()-started,
        status='native_Torch_CPU_norm_return_and_clipping_contract_verified', ranks=ranks,
        model_calls=0, backward_calls=0, optimizer_calls=0, GPUs_used=0,
        limits=['This uses CPU/Gloo with original Torch operators; it does not test the CUDA transfer in the VERL helper.',
                'A local reported norm does not by itself prove that clipping used a local coefficient.',
                'Correcting norm reporting does not explain the historical fixed-seed global VJP growth.'],
        rootcause_repair_verified=False)
    path = DEST / 'result.json'
    path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(receipt=str(path), report=report)))


if __name__ == '__main__':
    worker() if '--worker' in sys.argv else main()

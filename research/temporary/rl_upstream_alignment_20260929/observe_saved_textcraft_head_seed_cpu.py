"""Call the pinned original seed on saved FP32 logits; no model or GPU.

This is an offline original-owner recomputation, not a saved GPU seed or a
numerical acceptance test.  Error labels describe boundaries, not layer blame.
"""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import resource
import sys
import time


EXPECTED = {
    'compiled_logprob_seed': '757edb728aa57e2e802ab83dc59297ec32bbbb4eb9b2ecae6ca42fea5ed1a651',
    'compiled_finite_rules': 'e1cc970f15e23a58c2001173220a9826cb3257464a163c1fd722c705e2a2c144',
    'signed_secant_rules': '056d576e31b7076e7a08c89a5f25fde86897cfdf9daa3988d0a20acd6401030d',
}


def source(path):
    path = Path(path)
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                bytes=path.stat().st_size)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--owner-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    started = time.time()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    sys.path.insert(0, str(args.owner_dir))
    for name, sha in EXPECTED.items():
        assert source(args.owner_dir / (name + '.py'))['sha256'] == sha
    import torch
    torch.set_num_threads(1)
    modules = {name: importlib.import_module(name) for name in EXPECTED}
    imported = {name: source(module.__file__) for name, module in modules.items()}
    assert all(imported[name]['sha256'] == sha for name, sha in EXPECTED.items())
    assert not torch.cuda.is_initialized() and not torch.distributed.is_initialized()
    data = json.loads(args.input.read_bytes())
    results = []
    for row in data['observations']:
        joint = row['full_response_eos']
        single = row['single_token_eos']
        z0, z1 = [torch.tensor([joint[key]['z0'], joint[key]['z1']], dtype=torch.float32).unsqueeze(0)
                  for key in ('reference', 'factual')]
        zi, zf = [torch.tensor([single[key]['z0'], single[key]['z1']], dtype=torch.float32).unsqueeze(0)
                  for key in ('reference', 'factual')]
        target = torch.tensor([row['target_class_index']], dtype=torch.long)
        seed, checks = modules['compiled_logprob_seed'].seed_with_checks(z0, z1, target)
        factual_seed, factual_checks = modules['compiled_logprob_seed'].seed_with_checks(zf, zf, target)
        assert bool(checks) and bool(factual_checks)
        delta = zf - zi
        projected = float((seed * delta).sum().item())
        factual_projected = float((factual_seed * delta).sum().item())
        exact_from_logits = float((zf.log_softmax(-1)[0, target[0]] -
                                   zi.log_softmax(-1)[0, target[0]]).item())
        native = single['native_target_log_ratio']
        dt = row['saved_DT_d']
        results.append(dict(row, dt_d=dt, native_single_delete_d=native,
            joint_seed=seed[0].tolist(),
            factual_endpoint_seed=factual_seed[0].tolist(), single_logits_delta=delta[0].tolist(),
            owner_head_projected_d=projected, owner_factual_head_projected_d=factual_projected,
            native_single_d_from_saved_logits_cpu=exact_from_logits,
            cpu_log_probability_minus_saved_native=exact_from_logits-native,
            head_conditional_mismatch=projected-native,
            remainder_joint_propagation_allocation_numerical=dt-projected,
            total_dt_minus_native=dt-native,
            factual_head_conditional_mismatch=factual_projected-native))
    result = dict(scope=__doc__, status='offline_pinned_owner_seed_recomputation_not_gpu_acceptance',
        observed_unix=time.time(), wall_seconds=time.time()-started, pid=os.getpid(),
        python=sys.executable, torch_version=torch.__version__, dtype='torch.float32',
        imported_owner_sources=imported, sources=[source(args.input), source(__file__)],
        cuda_initialized=torch.cuda.is_initialized(), distributed_initialized=torch.distributed.is_initialized(),
        model_forward_calls=0, finite_decoder_calls=0, backward_calls=0,
        optimizer_steps=0, scheduler_steps=0, process_max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        process_smaps_rollup=Path('/proc/self/smaps_rollup').read_text(),
        host_meminfo=Path('/proc/meminfo').read_text(), observations=results,
        limitations=['The seed is recomputed on CPU from stored actual FP32 logits, not captured actual GPU seed operands.',
                    'Remainder includes lower propagation/allocation, head projection, native storage and other numerical effects; no layer is identified by this decomposition.',
                    'These are the existing selected successful first-response probes, not all tokens or a world reward oracle.'])
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output=source(args.output), observations=len(results),
        wall_seconds=result['wall_seconds'], max_rss_bytes=result['process_max_rss_bytes'],
        cuda_initialized=result['cuda_initialized'], model_forward_calls=0)))


if __name__ == '__main__':
    main()

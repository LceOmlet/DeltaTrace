"""Compare observed native transport artifacts; no model or GPU execution."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import time

import torch
from safetensors import safe_open
from compare_native_head_captures_20261011 import summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    started = time.perf_counter()
    model = Path('/mnt/si0021787ci2/default/models/Qwen3.5-9B')
    index = json.loads((model / 'model.safetensors.index.json').read_bytes())
    with safe_open(model / index['weight_map']['lm_head.weight'], framework='pt', device='cpu') as f:
        reference = f.get_slice('lm_head.weight')[159048].clone()
    result = dict(unix=time.time(), source=str(Path(__file__)),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        GPU_calls=0, DT_calls=0, model_calls=0, production_changes=0,
        new_numerical_tolerance=None, ranks=[])
    for rank in range(2):
        report = json.loads((args.job / f'result/rank{rank}-transport.json').read_bytes())
        captures = torch.load(args.job / f'result/rank{rank}-transport.pt',
                              map_location='cpu', weights_only=False)
        record = json.loads((args.job / f'result/rank{rank}.json').read_bytes())
        row = dict(rank=rank, phase=record['phase'],
                   optimizer_steps=record['actual_optimizer_steps'],
                   errors=report['errors'], observations=[])
        result['ranks'].append(row)
        for item in captures:
            value = item['buffer']
            row['observations'].append(dict(
                **{k: v for k, v in item.items() if k != 'buffer'},
                vs_checkpoint=summarize(reference, value),
                values=dict(min=float(value.min()) if value.numel() else None,
                    max=float(value.max()) if value.numel() else None,
                    L2=float(value.double().norm()),
                    sha256=hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest())))
        path = args.job / f'result/rank{rank}-native-GEMM-rows.pt'
        if path.is_file():
            gemm = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
            weights = gemm['competing_weights'][gemm['metadata']['copied_indices']]
            row['actual_GEMM_vs_checkpoint'] = summarize(reference.expand_as(weights), weights)
            candidates = [x for x in captures if x['index'] == 1
                          and x['phase'] == 'native_unpacked_all_gather_output']
            if len(candidates) == 1:
                row['actual_GEMM_vs_same_unshard_native_output'] = summarize(
                    candidates[0]['buffer'].expand_as(weights), weights)
            else:
                row['same_unshard_comparison_unavailable'] = len(candidates)
        loss_path = args.job / f'result/rank{rank}-loss-inputs-microbatch1.pt'
        if loss_path.is_file():
            loss = torch.load(loss_path, map_location='cpu', weights_only=False)
            row['logp_gradient_nan_count'] = int(torch.isnan(loss['log_prob_gradient']).sum())
    result.update(seconds=time.perf_counter() - started,
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        CUDA_initialized=torch.cuda.is_initialized())
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()

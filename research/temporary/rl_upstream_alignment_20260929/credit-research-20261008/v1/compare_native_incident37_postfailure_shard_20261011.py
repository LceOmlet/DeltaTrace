"""Compare a captured failed GEMM weight to its post-failure native CPU shard."""
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
    started = time.perf_counter()
    torch.set_num_threads(4)
    model = Path('/mnt/si0021787ci2/default/models/Qwen3.5-9B')
    index = json.loads((model / 'model.safetensors.index.json').read_bytes())
    with safe_open(model / index['weight_map']['lm_head.weight'], framework='pt', device='cpu') as f:
        checkpoint = f.get_slice('lm_head.weight')[159048].clone()
    result = dict(unix=time.time(), source=str(Path(__file__)),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        model_calls=0, DT_calls=0, GPU_calls=0, production_changes=0, shards=[], ranks=[])
    source = None
    for rank in range(2):
        path = args.job / f'result/rank{rank}-postfailure-shard.pt'
        report = json.loads((args.job / f'result/rank{rank}-postfailure-shard.json').read_bytes())
        result['shards'].append(report)
        for item in torch.load(path, map_location='cpu', weights_only=False):
            if 'buffer' in item:
                assert source is None, 'Multiple owners for the same global row'
                source = item['buffer']
                result['source_rank'] = rank
                result['source_vs_checkpoint'] = summarize(checkpoint, source)
                result['source_values'] = dict(min=float(source.min()), max=float(source.max()),
                    L2=float(source.double().norm()),
                    sha256=hashlib.sha256(source.view(torch.uint8).numpy().tobytes()).hexdigest())
    for rank in range(2):
        record = json.loads((args.job / f'result/rank{rank}.json').read_bytes())
        row = dict(rank=rank, phase=record['phase'], actual_optimizer_steps=record['actual_optimizer_steps'])
        result['ranks'].append(row)
        path = args.job / f'result/rank{rank}-native-GEMM-rows.pt'
        if not path.is_file():
            row['missing_GEMM_capture'] = str(path)
            continue
        values = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
        indices = values['metadata']['copied_indices']
        weights = values['competing_weights'][indices]
        row['captured_rows'] = len(indices)
        row['actual_GEMM_weight_vs_checkpoint'] = summarize(checkpoint.expand_as(weights), weights)
        if source is not None:
            row['actual_GEMM_weight_vs_postfailure_source_shard'] = summarize(source.expand_as(weights), weights)
        loss = torch.load(args.job / f'result/rank{rank}-loss-inputs-microbatch1.pt',
                          map_location='cpu', weights_only=False)
        row['logp_gradient_nan_count'] = int(torch.isnan(loss['log_prob_gradient']).sum())
    result.update(seconds=time.perf_counter() - started,
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        CUDA_initialized=torch.cuda.is_initialized())
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()

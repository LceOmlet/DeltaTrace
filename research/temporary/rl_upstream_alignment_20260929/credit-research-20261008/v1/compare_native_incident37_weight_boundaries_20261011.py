"""Read captured native weight boundaries on CPU, without running the model."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import time

import torch
from safetensors import safe_open


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    started = time.perf_counter()
    model = Path('/mnt/si0021787ci2/default/models/Qwen3.5-9B')
    index = json.loads((model / 'model.safetensors.index.json').read_bytes())
    checkpoint_file = model / index['weight_map']['lm_head.weight']
    with safe_open(checkpoint_file, framework='pt', device='cpu') as f:
        reference = f.get_slice('lm_head.weight')[159048].clone()
    previous = args.job.parent / 'textcraft-native-incident37-competing-weight-capture-20261011-v1'
    bad = torch.load(previous / 'result/rank0-native-GEMM-rows.pt',
                     map_location='cpu', mmap=True, weights_only=False)['competing_weights'][0]
    result = dict(unix=time.time(), source=str(Path(__file__)),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  checkpoint_file=str(checkpoint_file), global_row=159048,
                  model_calls=0, DT_calls=0, GPU_calls=0, production_changes=0, ranks=[])
    for rank in range(2):
        path = args.job / f'result/rank{rank}-weight-boundaries.pt'
        record = json.loads((args.job / f'result/rank{rank}.json').read_bytes())
        row = dict(rank=rank, phase=record['phase'],
                   actual_optimizer_steps=record['actual_optimizer_steps'], observations=[])
        result['ranks'].append(row)
        if not path.is_file():
            row['missing'] = str(path)
            continue
        row['artifact'] = dict(path=str(path), bytes=path.stat().st_size,
                               sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        observations = torch.load(path, map_location='cpu', weights_only=False)
        for item in observations:
            values = {k: v for k, v in item.items() if k != 'buffer'}
            row['observations'].append(values)
            tensor = item.get('buffer')
            if tensor is None:
                continue
            values.update(exact_checkpoint_row=torch.equal(tensor, reference),
                exact_previous_bad_row=torch.equal(tensor, bad),
                sha256=hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),
                finite=bool(torch.isfinite(tensor).all()), min=float(tensor.min()),
                max=float(tensor.max()), L2=float(tensor.double().norm()),
                different_coordinates=int((tensor != reference).sum()))
        loss = args.job / f'result/rank{rank}-loss-inputs-microbatch1.pt'
        if loss.is_file():
            values = torch.load(loss, map_location='cpu', weights_only=False)
            row['logp_gradient_nan_count'] = int(torch.isnan(values['log_prob_gradient']).sum())
    result.update(seconds=time.perf_counter() - started,
                  peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                  CUDA_initialized=torch.cuda.is_initialized())
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()

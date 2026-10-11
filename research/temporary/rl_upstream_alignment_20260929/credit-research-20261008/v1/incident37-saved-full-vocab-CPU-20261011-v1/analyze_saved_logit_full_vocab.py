"""CPU-only full-vocabulary dots for four already-saved hidden vectors.

Read original weights in bounded slices. Compare one pre-existing selected
position per rank in the failed and finite replays; never run a model or GPU.
This describes the extent of discrepancies, not a numerical acceptance gate.
"""
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
    parser.add_argument('--receipts', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    torch.set_num_threads(4)
    assert not torch.cuda.is_initialized()
    jobs = (
        ('failed', 'textcraft-native-incident37-native-flight-recorder-20261011-v1'),
        ('finite', 'textcraft-native-incident37-after-replay-official-RPC-20261011-v1'))
    items, hidden, actual = [], [], []
    for label, job in jobs:
        for rank in (0, 1):
            result = args.receipts / job / 'result'
            path = result / f'rank{rank}-native-GEMM-rows.pt'
            saved = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
            metadata = saved['metadata']
            # Existing recorder order; selection is not changed based on values.
            i = metadata['copied_indices'][0]
            hidden.append(saved['hidden'][i].double().clone())
            actual.append(saved['logits'][i].clone())
            items.append(dict(label=label, rank=rank, saved_GEMM=str(path),
                saved_GEMM_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                selection=metadata['selection']['rows'][i],
                phase=json.loads((result / f'rank{rank}.json').read_bytes())['phase']))
            del saved
    hidden = torch.stack(hidden)
    actual = torch.stack(actual)
    index_path = args.model / 'model.safetensors.index.json'
    index = json.loads(index_path.read_bytes())
    weights_path = args.model / index['weight_map']['lm_head.weight']
    expected = torch.empty_like(actual)
    weights_sha = hashlib.sha256()
    with safe_open(weights_path, framework='pt', device='cpu') as owner:
        source = owner.get_slice('lm_head.weight')
        assert source.get_shape() == [actual.shape[1], hidden.shape[1]]
        for first in range(0, actual.shape[1], 4096):
            last = min(first + 4096, actual.shape[1])
            weights = source[first:last]
            weights_sha.update(weights.contiguous().view(torch.uint8).numpy().tobytes())
            expected[:, first:last] = (hidden @ weights.double().T).to(actual.dtype)
            del weights
    delta = actual.float() - expected.float()
    for row, item in enumerate(items):
        error = delta[row]
        top = error.abs().topk(32).indices
        item.update(vocab_rows=error.numel(), exact_matches=int((error == 0).sum()),
            maximum_abs_difference=float(error.abs().max()),
            abs_difference_quantiles={str(q): float(torch.quantile(error.abs(), q))
                                     for q in (0., .5, .9, .99, .999, 1.)},
            top_discrepancies=[dict(vocab_id=int(i), difference=float(error[i]),
                native_logit=float(actual[row, i]), original_weight_logit=float(expected[row, i]))
                for i in top], chunks=[])
        for first in range(0, error.numel(), 4096):
            last = min(first + 4096, error.numel())
            values = error[first:last]
            item['chunks'].append(dict(first=first, last=last,
                exact_matches=int((values == 0).sum()),
                max_abs_difference=float(values.abs().max()),
                median_abs_difference=float(values.abs().median()),
                L2_difference=float(values.double().norm())))
    tensor_path = args.output.with_suffix('.pt')
    torch.save(dict(actual_native_logits=actual, original_weight_FP64_dots_rounded=expected,
                    difference=delta, rows=items), tensor_path)
    out = dict(unix=time.time(), source=str(Path(__file__)),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        reference_weights_path=str(weights_path), reference_tensor_sha256=weights_sha.hexdigest(),
        index_sha256=hashlib.sha256(index_path.read_bytes()).hexdigest(), rows=items,
        saved_vectors=dict(path=str(tensor_path), bytes=tensor_path.stat().st_size,
            sha256=hashlib.sha256(tensor_path.read_bytes()).hexdigest()),
        scope='One already-selected actual hidden vector per rank per replay. Same original weights and same FP64-dot calculation for failed and finite data. Descriptive localization only; not full-position coverage, a replacement model, or a new tolerance.',
        model_calls=0, DT_calls=0, optimizer_steps=0, GPU_calls=0,
        production_changes=0, numerical_tolerance=None,
        seconds=time.perf_counter() - started,
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        CUDA_initialized=torch.cuda.is_initialized())
    args.output.write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out))


if __name__ == '__main__':
    main()

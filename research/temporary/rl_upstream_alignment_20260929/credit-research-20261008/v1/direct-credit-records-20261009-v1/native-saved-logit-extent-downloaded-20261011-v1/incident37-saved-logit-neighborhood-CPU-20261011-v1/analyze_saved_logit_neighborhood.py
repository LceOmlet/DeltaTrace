"""Describe the locality of a byte-proven wrong head row, using saved data only.

This is a CPU diagnostic, not a model, kernel, acceptance test, or repair.
The 257-row neighborhood is centered on the previously captured wrong row.
It does not estimate corruption rates outside the saved positions/rows.
"""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import time

import torch
from safetensors import safe_open


def digest_tensor(value):
    return hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    torch.set_num_threads(4)
    assert not torch.cuda.is_initialized()
    center = 159048
    first, last = center - 128, center + 129
    index_path = args.model / 'model.safetensors.index.json'
    index = json.loads(index_path.read_bytes())
    weights_path = args.model / index['weight_map']['lm_head.weight']
    with safe_open(weights_path, framework='pt', device='cpu') as owner:
        original = owner.get_slice('lm_head.weight')[first:last].clone()
    out = dict(
        unix=time.time(), source=str(Path(__file__)),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        result=str(args.result), weights_path=str(weights_path),
        index_sha256=hashlib.sha256(index_path.read_bytes()).hexdigest(),
        reference_rows_sha256=digest_tensor(original),
        center=center, first_inclusive=first, last_exclusive=last,
        scope='Saved selected GEMM positions and a fixed neighborhood of the byte-proven wrong row only. FP64 dots with the same actual hidden vectors describe operand discrepancies; no new numerical acceptance threshold.',
        model_calls=0, DT_calls=0, optimizer_steps=0, GPU_calls=0,
        production_changes=0, numerical_tolerance=None, ranks=[])
    for rank in (0, 1):
        report = json.loads((args.result / f'rank{rank}.json').read_bytes())
        path = args.result / f'rank{rank}-native-GEMM-rows.pt'
        saved = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        metadata = saved['metadata']
        indices = torch.tensor(metadata['copied_indices'], dtype=torch.long)
        assert indices.unique().numel() == indices.numel()
        hidden = saved['hidden'][indices]
        actual = saved['logits'][indices, first:last]
        assert hidden.shape[1] == original.shape[1]
        # Same actual hidden vectors, no cross-replay hidden-state comparison.
        rounded = (hidden.double() @ original.double().T).to(actual.dtype)
        difference = actual.double() - rounded.double()
        rows = []
        for offset, vocab_id in enumerate(range(first, last)):
            delta = difference[:, offset]
            rows.append(dict(
                vocab_id=vocab_id, original_weight_sha256=digest_tensor(original[offset]),
                exact_matches=int((delta == 0).sum()),
                max_abs_difference=float(delta.abs().max()),
                L2_difference=float(delta.norm()),
                median_abs_difference=float(delta.abs().median()),
                signed_min=float(delta.min()), signed_max=float(delta.max()),
                native_logit_min=float(actual[:, offset].min()),
                native_logit_max=float(actual[:, offset].max()),
                original_weight_logit_min=float(rounded[:, offset].min()),
                original_weight_logit_max=float(rounded[:, offset].max())))
        captured_center = saved['competing_weights'][indices]
        assert metadata['competing_vocab_id'] == center
        center_logits = (hidden.double() * captured_center.double()).sum(-1).to(actual.dtype)
        center_delta = actual[:, center - first].double() - center_logits.double()
        out['ranks'].append(dict(
            rank=rank, phase=report['phase'], positions=int(indices.numel()),
            saved_GEMM=dict(path=str(path), bytes=path.stat().st_size,
                            sha256=hashlib.sha256(path.read_bytes()).hexdigest()),
            hidden_dtype=str(hidden.dtype), logits_dtype=str(actual.dtype),
            original_weight_dtype=str(original.dtype),
            captured_center_first_weight_sha256=digest_tensor(captured_center[0]),
            captured_center_weights_identical=bool(torch.equal(
                captured_center, captured_center[0].expand_as(captured_center))),
            captured_center_dot_exact_matches=int((center_delta == 0).sum()),
            captured_center_dot_max_abs_difference=float(center_delta.abs().max()),
            all_neighborhood_rows_exact=int((difference == 0).all(dim=0).sum()),
            rows=rows))
        del saved, hidden, actual, rounded, difference, captured_center
    out.update(seconds=time.perf_counter() - started,
               peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
               CUDA_initialized=torch.cuda.is_initialized())
    args.output.write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out))


if __name__ == '__main__':
    main()

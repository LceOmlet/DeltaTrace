"""Localize a recorded head failure using saved original GEMM outputs on CPU.

Use the pinned VERL naive log-prob/entropy owners for normalization, and FP64
scalar dot products for descriptive operand checks. No invented tolerance or
acceptance claim, no model/head/GPU calls, and no replacement training path.
"""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import time

import torch
from verl.utils import torch_functional as verl_F

from compare_native_head_captures_20261011 import summarize


def identity(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            digest.update(chunk)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())


def load(path):
    return torch.load(path, map_location='cpu', weights_only=False, mmap=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    started = time.perf_counter()
    receipts = args.result.parents[1]
    result = dict(unix=time.time(), source=identity(Path(__file__)),
                  normalization_owner=identity(Path(verl_F.__file__)),
                  model_calls=0, DT_calls=0, optimizer_steps=0, GPU_calls=0,
                  production_changes=0, new_numerical_tolerance=None, ranks=[])
    for rank in range(2):
        record = json.loads((args.result / f'rank{rank}.json').read_bytes())
        row = dict(rank=rank, phase=record['phase'],
                   actual_optimizer_steps=record['actual_optimizer_steps'])
        result['ranks'].append(row)
        gemm_path = args.result / f'rank{rank}-native-GEMM-rows.pt'
        head_path = args.result / f'rank{rank}-async-head-return-microbatch1.pt'
        if not gemm_path.is_file() or not head_path.is_file():
            row['missing'] = [str(p) for p in [gemm_path, head_path] if not p.is_file()]
            continue
        gemm, head = load(gemm_path), load(head_path)
        metadata = gemm['metadata']
        selection = metadata['selection']['rows']
        valid = metadata['copied_indices']
        row.update(GEMM=identity(gemm_path), head=identity(head_path),
                   copied_rows=len(valid), planned_rows=len(selection),
                   native_mm_calls=len(metadata['mm_calls']),
                   native_rows_seen=metadata['native_rows_seen'],
                   expected_rows=metadata['expected_rows'],
                   temperature=metadata['temperature'])
        if len(set(valid)) != len(valid):
            row['invalid_capture'] = 'duplicate copied row indices'
            continue
        indices = torch.tensor(valid, dtype=torch.long)
        positions = torch.tensor([selection[i]['flat_position'] for i in valid], dtype=torch.long)
        expected_ids = torch.tensor([selection[i]['expected_token_id'] for i in valid], dtype=torch.long)
        labels = gemm['token_ids'][indices].long()
        row['actual_selected_token_ids_match'] = torch.equal(expected_ids, labels)
        if not row['actual_selected_token_ids_match']:
            row['invalid_capture'] = 'actual token labels differ from prior selected identities'
            continue
        logits = (gemm['logits'][indices] / metadata['temperature']).float()
        # Call the existing pinned owner functions; do not copy their math.
        lp_owner = getattr(verl_F.logprobs_from_logits_naive,
                           '_torchdynamo_orig_callable', verl_F.logprobs_from_logits_naive)
        entropy_owner = getattr(verl_F.entropy_from_logits,
                                '_torchdynamo_orig_callable', verl_F.entropy_from_logits)
        reference_lp = lp_owner(logits, labels)
        reference_entropy = entropy_owner(logits)
        native_lp = head['log_probs'].reshape(-1)[positions]
        native_entropy = head['entropy'].reshape(-1)[positions]
        row['CPU_owner_normalization_vs_native_head_logp'] = summarize(reference_lp, native_lp)
        row['CPU_owner_normalization_vs_native_head_entropy'] = summarize(reference_entropy, native_entropy)
        target_mm = gemm['logits'][indices].gather(1, labels[:, None]).squeeze(1)
        h = gemm['hidden'][indices]
        w = gemm['target_weights'][indices]
        scalar_dot = (h.double() * w.double()).sum(-1)
        rounded_dot = scalar_dot.to(target_mm.dtype)
        row['native_target_logit_vs_FP64_dot_rounded_to_native_dtype'] = summarize(rounded_dot, target_mm)
        prior_path = receipts / 'textcraft-native-incident37-native-head-live-capture-20261011-v1/result' / f'rank{rank}-native-head-live-microbatch1.pt'
        prior = load(prior_path)
        prior_h = prior['tensors']['hidden_states'].reshape(-1, h.shape[-1])[positions]
        prior_w = prior['tensors']['vocab_weights'][labels]
        row['actual_GEMM_hidden_vs_prior_normal_A'] = summarize(prior_h, h)
        row['actual_target_weights_vs_prior_normal_A'] = summarize(prior_w, w)
        competing_id = metadata['competing_vocab_id']
        competing_w = gemm['competing_weights'][indices]
        prior_competing_w = prior['tensors']['vocab_weights'][competing_id].expand_as(competing_w)
        competing_mm = gemm['logits'][indices, competing_id]
        competing_dot = (h.double() * competing_w.double()).sum(-1)
        competing_dot_rounded = competing_dot.to(competing_mm.dtype)
        row['competing_vocab_id'] = competing_id
        row['actual_competing_weights_vs_prior_normal_A'] = summarize(prior_competing_w, competing_w)
        row['native_competing_logit_vs_actual_FP64_dot_rounded'] = summarize(competing_dot_rounded, competing_mm)
        row['actual_competing_weight_ranges'] = dict(min=float(competing_w.min()),
            max=float(competing_w.max()), L2_min=float(competing_w.double().norm(dim=-1).min()),
            L2_max=float(competing_w.double().norm(dim=-1).max()))
        del prior
        loss_path = args.result / f'rank{rank}-loss-inputs-microbatch1.pt'
        loss = load(loss_path) if loss_path.is_file() else None
        if loss is not None:
            row['loss_gradient_nan_count'] = int(torch.isnan(loss['log_prob_gradient']).sum())
        values, top_ids = logits.topk(3, dim=-1)
        row['positions'] = []
        for j, i in enumerate(valid):
            selected = selection[i]
            rr, cc = selected['response_row'], selected['response_column']
            row['positions'].append(dict(**selected,
                current_head_logp=float(native_lp[j]),
                CPU_owner_logp_from_actual_mm=float(reference_lp[j]),
                head_minus_CPU_owner=float(native_lp[j] - reference_lp[j]),
                actual_target_GEMM_logit=float(target_mm[j]),
                FP64_scalar_dot=float(scalar_dot[j]),
                FP64_dot_rounded=float(rounded_dot[j]),
                target_logit_minus_rounded_dot=float(target_mm[j].double() - rounded_dot[j].double()),
                actual_competing_GEMM_logit=float(competing_mm[j]),
                actual_competing_FP64_dot=float(competing_dot[j]),
                actual_competing_dot_rounded=float(competing_dot_rounded[j]),
                current_loss_logp=float(loss['log_prob'][rr, cc]) if loss is not None else None,
                current_gradient_nan=bool(torch.isnan(loss['log_prob_gradient'][rr, cc])) if loss is not None else None,
                top3_logit_ids=top_ids[j].tolist(), top3_logits=values[j].tolist()))
        del gemm, head, logits, loss, h, w
    result.update(CUDA_initialized=torch.cuda.is_initialized(),
                  seconds=time.perf_counter() - started,
                  peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()

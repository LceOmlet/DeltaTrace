"""Compare already saved native head and loss values on CPU, without a tolerance."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import time

import torch

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
    receipts = args.result.parents[1]
    result = dict(unix=time.time(), ranks=[], source=identity(Path(__file__)),
                  model_calls=0, DT_calls=0, optimizer_steps=0,
                  production_changes=0, new_numerical_tolerance=None)
    for rank, old_pid in [(0, 987808), (1, 989860)]:
        record = json.loads((args.result / f'rank{rank}.json').read_bytes())
        row = dict(rank=rank, phase=record['phase'], microbatches=record['microbatches'],
                   actual_optimizer_steps=record['actual_optimizer_steps'])
        result['ranks'].append(row)
        head_path = args.result / f'rank{rank}-async-head-return-microbatch1.pt'
        loss_path = args.result / f'rank{rank}-loss-inputs-microbatch1.pt'
        if not head_path.is_file() or not loss_path.is_file():
            row['missing'] = [str(p) for p in [head_path, loss_path] if not p.is_file()]
            continue
        head, loss = load(head_path), load(loss_path)
        row.update(head=identity(head_path), loss=identity(loss_path),
                   head_metadata=head['metadata'],
                   head_logp_finite=bool(torch.isfinite(head['log_probs']).all()),
                   head_entropy_finite=bool(torch.isfinite(head['entropy']).all()),
                   loss_gradient_nan_count=int(torch.isnan(loss['log_prob_gradient']).sum()))
        prior_path = receipts / 'textcraft-native-incident37-all-DT-live-head-capture-20261011-v1/result' / f'rank{rank}-native-head-live-microbatch1.pt'
        prior = load(prior_path)
        row['prior_normal_capture'] = dict(**identity(prior_path),
            log_probs=summarize(prior['tensors']['token_log_probs'], head['log_probs']),
            entropy=summarize(prior['tensors']['entropy'], head['entropy']))
        del prior
        failed_path = args.result / f'rank{rank}-exception-head-H-IDs.pt'
        if not failed_path.is_file():
            row['failed_labels_missing'] = True
            row['head_loss_alignment_comparison'] = 'Not computed without current captured labels.'
            continue
        failed = load(failed_path)
        labels = failed['input_ids'].reshape(head['log_probs'].shape)
        incident = receipts / 'textcraft-native-actor-incidents-20261010-v1' / f'rank{rank}-pid{old_pid}'
        snapshot_path = next(incident.glob('incident-*-update.pt'))
        snapshot = load(snapshot_path)
        responses = snapshot['input_batch']['responses'][4:8]
        width = labels.shape[1]
        # Reuse the prior diagnostic's exact artifact alignment. Do not copy
        # actor padding/position algorithms or select an offset by score.
        offsets = [offset for offset in range(min(width - 1, 512))
                   if torch.equal(labels[:, offset:-1], responses[:, :width-offset-1])]
        row.update(failed_labels=identity(failed_path), response_offsets=offsets)
        if len(offsets) != 1:
            row['head_loss_alignment_comparison'] = 'Not computed: token identity alignment is not unique.'
            continue
        rr, cc = loss['response_mask'].bool().nonzero(as_tuple=True)
        position = cc + offsets[0]
        assert bool((position < width - 1).all())
        assert torch.equal(labels[rr, position], responses[rr, cc])
        h = head['log_probs'][rr, position]
        l = loss['log_prob'][rr, cc]
        row['head_loss_alignment_comparison'] = summarize(h, l)
        row['aligned_active_tokens'] = rr.numel()
        row['head_minus_old_max_abs'] = float((h-loss['old_log_prob'][rr, cc]).abs().max())
        row['loss_minus_old_max_abs'] = float((l-loss['old_log_prob'][rr, cc]).abs().max())
        row['head_loss_difference_at_nan_gradients'] = summarize(
            h[torch.isnan(loss['log_prob_gradient'][rr, cc])],
            l[torch.isnan(loss['log_prob_gradient'][rr, cc])])
        del head, loss, failed, snapshot
    result.update(CUDA_initialized=torch.cuda.is_initialized(),
                  peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()

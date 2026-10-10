"""Diagnose a one-token label shift using already captured real H/W/IDs.

No model/DT/head/optimizer is executed, and no training reference is changed.
Use log p(alt)-log p(real) = logit(alt)-logit(real) to test the concrete
off-by-one hypothesis in the owning wrapper's torch.roll(labels, -1) boundary.
Selected-logit bmm is a diagnostic identity, not a replacement output head or
an official numerical acceptance test. The H/W are from a finite capture, not
the lost failed forward; a mismatch cannot exclude other label corruption.
"""
import hashlib
import json
from pathlib import Path
import resource
import time

import psutil
import torch


ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT = ROOT/'receipts/textcraft-native-label-alignment-20261011-v1'
INCIDENT = ROOT/'receipts/textcraft-native-actor-incidents-20261010-v1'
CAPTURE = ROOT/'receipts/textcraft-native-incident37-native-head-live-capture-20261011-v1/result'
HEAD_SHA = [
    '22b2db81f578d6ddb4da01f0167a16e4c037aa065f03d52c41692abe92116f62',
    'a457ba612bed14936ba648b78ecac0779b036b63bbaf5ff36306caf51767b7d1',
]
BAD_SHA = [
    '537a2045e105ee2dba3f64cd2a2f547a234e9994cb9532c8815ea5052c5383a4',
    '0e6ff2ad745b583714939df350472ddacfacfb0acdc379d6201cdef31a24d512',
]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def describe(v):
    v = v.double()
    return dict(count=v.numel(), mean=float(v.mean()), mean_abs=float(v.abs().mean()),
                max_abs=float(v.abs().max()),
                abs_quantiles=torch.quantile(v.abs(), torch.tensor(
                    [0., .25, .5, .75, .9, .99, 1.], dtype=torch.float64)).tolist())


def main():
    OUT.mkdir(exist_ok=True)
    report = dict(started_unix=time.time(), pid=psutil.Process().pid,
                  birth=psutil.Process().create_time(), scope=__doc__,
                  source_sha256=sha(Path(__file__)), ranks=[], complete=False,
                  model_calls=0, DT_calls=0, head_calls=0, optimizer_calls=0,
                  production_changes=0, official_tolerance_tests=0,
                  bin_note='abs(failed-old)>10 is an existing descriptive bin, not an acceptance tolerance.')

    def save():
        report.update(unix=time.time(), peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                      peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_torch_reserved_bytes=torch.cuda.max_memory_reserved())
        (OUT/'result.json').write_text(json.dumps(report, indent=2)+'\n')

    try:
        with torch.no_grad():
            for rank, pid in [(0, 987808), (1, 989860)]:
                folder = INCIDENT/f'rank{rank}-pid{pid}'
                bad_path = next(folder.glob('loss-backward-*.pt'))
                snapshot_path = next(folder.glob('incident-*-update.pt'))
                capture_path = CAPTURE/f'rank{rank}-native-head-live-microbatch1.pt'
                assert sha(bad_path) == BAD_SHA[rank]
                assert sha(capture_path) == HEAD_SHA[rank]
                bad = torch.load(bad_path, map_location='cpu', weights_only=False)['tensors']
                snapshot = torch.load(snapshot_path, map_location='cpu', weights_only=False)
                responses = snapshot['input_batch']['responses'][4:8].clone()
                del snapshot
                capture = torch.load(capture_path, map_location='cpu', weights_only=False)
                t = capture['tensors']; labels = t['input_ids']; hidden = t['hidden_states']
                width = hidden.shape[1]
                # Locate the recorded response in recorded rolled labels by exact
                # identity; do not recreate actor padding/position algorithms.
                offsets = [offset for offset in range(min(width-1, 512))
                           if torch.equal(labels[:, offset:-1], responses[:, :width-offset-1])]
                assert len(offsets) == 1, offsets
                offset = offsets[0]
                row, column = bad['response_mask'].bool().nonzero(as_tuple=True)
                position = column + offset
                assert bool(((position > 0) & (position < width-1)).all())
                assert torch.equal(labels[row, position], responses[row, column])
                factual = t['token_log_probs'][row, position].float()
                observed = bad['log_prob'][row, column].float()
                old = bad['old_log_prob'][row, column].float()
                severe = (observed-old).abs() > 10
                weight = t['vocab_weights'].to('cuda')
                temperature = capture['metadata']['temperature']
                original_dtype = hidden.dtype
                logits = {shift: [] for shift in [-1, 0, 1]}
                for start in range(0, row.numel(), 512):
                    end = min(row.numel(), start+512)
                    rr = row[start:end]; pp = position[start:end]
                    h = hidden[rr, pp].to('cuda').float()
                    for shift in [-1, 0, 1]:
                        ids = labels[rr, pp+shift].to('cuda')
                        w = weight.index_select(0, ids).float()
                        # The real owner uses BF16 GEMM then a BF16 temperature
                        # division before its FP32 log_softmax. Preserve those
                        # rounding boundaries in this selected-dot diagnosis.
                        z = torch.bmm(h.unsqueeze(1), w.unsqueeze(-1)).flatten().to(original_dtype)
                        z = (z/temperature).to(original_dtype).float()
                        logits[shift].append(z.cpu())
                        del ids, w, z
                    del h
                logits = {k: torch.cat(v) for k, v in logits.items()}
                predicted = {k: factual+v-logits[0] for k, v in logits.items()}
                points = {shift: dict(all_active=describe(predicted[shift]-observed),
                                      severe_subset=describe((predicted[shift]-observed)[severe]))
                          for shift in [-1, 0, 1]}
                tensor_path = OUT/f'rank{rank}-diagnostic-vectors.pt'
                torch.save(dict(row=row, response_column=column, native_head_position=position,
                                native_head_labels=labels[row, position], failed_logp=observed,
                                old_logp=old, finite_capture_logp=factual,
                                severe_descriptive_mask=severe,
                                selected_logits=logits, predicted_logp=predicted), tensor_path)
                report['ranks'].append(dict(rank=rank, capture=dict(path=str(capture_path), sha256=HEAD_SHA[rank]),
                    failure=dict(path=str(bad_path), sha256=BAD_SHA[rank]),
                    exact_unique_response_offset=offset, active_positions=row.numel(),
                    severe_positions=int(severe.sum()), source_dtype=str(original_dtype),
                    temperature=temperature, diagnostics=points,
                    vectors=dict(path=str(tensor_path), bytes=tensor_path.stat().st_size, sha256=sha(tensor_path))))
                save()
                del capture, t, hidden, labels, weight, bad
        report['complete'] = True
    except BaseException:
        import traceback
        report['traceback'] = traceback.format_exc()
        raise
    finally:
        save()
    print(json.dumps(dict(complete=True, output=str(OUT/'result.json'))))


if __name__ == '__main__':
    main()

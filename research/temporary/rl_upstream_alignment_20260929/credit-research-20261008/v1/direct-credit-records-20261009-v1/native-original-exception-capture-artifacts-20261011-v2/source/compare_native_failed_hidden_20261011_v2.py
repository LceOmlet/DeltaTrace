"""Compare failure-only saved head inputs with existing native forward captures.

CPU-only localization, no head/model calls and no numerical tolerance. The
comparison head weights come from earlier forwards, not the failed live head.
Missing failed captures remain missing; a finite replay is not a repair.
"""
import argparse
import json
from pathlib import Path
import resource
import time

import torch

from capture_native_pre_dt_layers_20261011 import sha
from compare_native_head_captures_20261011 import summarize


def load(path):
    return torch.load(path, map_location='cpu', weights_only=False, mmap=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    receipts = args.result.parents[1]
    result = dict(unix=time.time(), scope=__doc__, ranks=[], model_calls=0,
                  DT_calls=0, optimizer_steps=0, source_sha256=sha(__file__))
    for rank in [0, 1]:
        record_path = args.result / f'rank{rank}.json'
        record = json.loads(record_path.read_bytes())
        row = dict(rank=rank, record_path=str(record_path),
                   record_sha256=sha(record_path), phase=record['phase'],
                   actual_optimizer_steps=record['actual_optimizer_steps'],
                   native_microbatches=record['microbatches'], captures=[])
        result['ranks'].append(row)
        # Preserve the original native record. An exception-only wrapper has
        # its own receipt and does not rewrite the native microbatch record.
        points = list(record['microbatches'])
        exception_path = args.result / f'rank{rank}-exception-capture.json'
        if exception_path.is_file():
            exception_record = json.loads(exception_path.read_bytes())
            row['exception_receipt'] = dict(path=str(exception_path),
                                            sha256=sha(exception_path))
            for capture in exception_record['captures']:
                if 'path' not in capture:
                    row.setdefault('capture_errors', []).append(capture)
                    continue
                points.append(dict(index=None, head_inputs=capture,
                                   exception_only_capture=True))
        for point in points:
            capture = point.get('head_inputs')
            if capture is None:
                continue
            path = Path(capture['path'])
            assert sha(path) == capture['sha256']
            failed = load(path)
            value = dict(microbatch=failed.get('microbatch_index', point['index']), path=str(path),
                         sha256=capture['sha256'],
                         exception_only_capture=point.get('exception_only_capture', False),
                         hidden_scope='Autograd saved head context after bad loss gradient; '
                                      'not an independent forward-boundary capture.',
                         full_failed_live_weights_available=False,
                         hidden_finite=bool(torch.isfinite(failed['hidden_states']).all()),
                         comparisons=[])
            row['captures'].append(value)
            for folder in ['textcraft-native-incident37-native-head-live-capture-20261011-v1',
                           'textcraft-native-incident37-all-DT-live-head-capture-20261011-v1']:
                prior_path = receipts / folder / 'result' / f'rank{rank}-native-head-live-microbatch1.pt'
                prior = load(prior_path)
                tensors = prior['tensors']
                value['comparisons'].append(dict(path=str(prior_path),
                    bytes=prior_path.stat().st_size, sha256=sha(prior_path),
                    hidden_states=summarize(tensors['hidden_states'].flatten(0, 1),
                                            failed['hidden_states']),
                    input_ids=summarize(tensors['input_ids'].flatten(), failed['input_ids'])))
                del prior, tensors
            del failed
    result.update(CUDA_initialized=torch.cuda.is_initialized(),
                  peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=True) + '\n')
    print(json.dumps(dict(output=str(args.output),
                          failed_captures=sum(len(r['captures']) for r in result['ranks']),
                          CUDA_initialized=result['CUDA_initialized'])))


if __name__ == '__main__':
    main()

"""CPU localization of captured native old/train forwards, without a tolerance.

Exact equality and differences describe this instrumented run. A first
different layer is not by itself a fault or proof of the incident's cause.
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
    result = dict(unix=time.time(), scope=__doc__, records=[], model_calls=0,
                  DT_calls=0, optimizer_steps=0, source_sha256=sha(__file__))
    for rank in [0, 1]:
        path = args.result / f'rank{rank}.json'
        record = json.loads(path.read_bytes())
        capture = record['native_training_layer_capture']
        pair = [{x['name']:x for x in c['files']} for c in capture['layers']]
        rows = []
        for name in sorted(pair[0].keys() | pair[1].keys()):
            files = [c.get(name) for c in pair]
            row = dict(name=name, files=files)
            if all(files):
                values = [load(f['path']) for f in files]
                row['comparison'] = summarize(*values)
                del values
            rows.append(row)
        head = load(capture['head']['path'])
        final_norm = load(pair[1]['final_norm']['path'])
        norm_head = summarize(final_norm, head['tensors']['hidden_states'])
        loss_path = args.result / f'rank{rank}-loss-inputs-microbatch1.pt'
        loss = load(loss_path)
        old = load(args.result / f'rank{rank}-native-old.pt')
        mask = loss['response_mask'].bool()
        differences = {}
        for label, baseline in [('current_native_old', old['old_log_probs'][4:8]),
                                ('original_saved_old', loss['old_log_prob'])]:
            differences[label] = dict(all=summarize(baseline, loss['log_prob']),
                                      active_policy=summarize(baseline[mask], loss['log_prob'][mask]))
        head_mask = head['tensors']['input_ids']
        previous_ids = []
        for folder in ['textcraft-native-incident37-native-head-live-capture-20261011-v1',
                       'textcraft-native-incident37-all-DT-live-head-capture-20261011-v1']:
            previous_path = args.result.parents[1] / folder / 'result' / f'rank{rank}-native-head-live-microbatch1.pt'
            previous = load(previous_path)
            previous_ids.append(dict(path=str(previous_path),
                                     input_ids=summarize(previous['tensors']['input_ids'], head_mask),
                                     vocab_weights=summarize(previous['tensors']['vocab_weights'], head['tensors']['vocab_weights'])))
            del previous
        different = [x['name'] for x in rows if x['name'].startswith('layer')
                     and x['name'] != 'layer00_input' and not x.get('comparison', {}).get('exact')]
        result['records'].append(dict(rank=rank, source_record=dict(path=str(path), sha256=sha(path)),
            phase=record['phase'], actual_optimizer_steps=record['actual_optimizer_steps'],
            native_microbatches=record['microbatches'], native_metrics=record.get('native_metrics'),
            first_different_decoder_layer=different[0] if different else None,
            layer_comparisons=rows, native_counts=[c['counts'] for c in capture['layers']],
            capture_errors=[c['errors'] for c in capture['layers']], head_errors=capture['head_errors'],
            final_norm_to_head=norm_head, head_capture=capture['head'],
            native_old_to_train=differences, previous_head_IDs_weights=previous_ids,
            nonfinite_loss_gradient=int((~torch.isfinite(loss['log_prob_gradient'])).sum()),
            nonfinite_active_policy_gradient=int((~torch.isfinite(loss['log_prob_gradient'][mask])).sum())))
        del record, head, final_norm, loss, old, pair
    result.update(CUDA_initialized=torch.cuda.is_initialized(),
                  peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=True)+'\n')
    print(json.dumps(result, indent=2, allow_nan=True))


if __name__ == '__main__':
    main()

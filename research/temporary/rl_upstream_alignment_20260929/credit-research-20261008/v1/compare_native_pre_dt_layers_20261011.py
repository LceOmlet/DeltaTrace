"""Compare actual native old/ref/old captures on CPU, without a new tolerance."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import time

import torch

from capture_native_pre_dt_layers_20261011 import sha
from compare_native_head_captures_20261011 import summarize


def tensor_sha(value):
    return hashlib.sha256(memoryview(value.contiguous().reshape(-1).view(torch.uint8).numpy())).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    receipt_root = args.result.parents[1]
    result = dict(unix=time.time(), scope=__doc__, records=[], model_calls=0,
                  DT_calls=0, optimizer_steps=0, source_sha256=sha(__file__))
    for rank in [0, 1]:
        path = args.result / f'rank{rank}.json'
        record = json.loads(path.read_bytes())
        states = record['pre_DT_parameter_state']
        reference = {(x['kind'], x['name']):x for x in states['before_old']['fields']}
        state_comparisons = []
        for label in ['after_ref', 'after_old_again']:
            current = {(x['kind'], x['name']):x for x in states[label]['fields']}
            differences = [dict(key=list(key), before=reference.get(key), after=current.get(key))
                for key in reference.keys() | current.keys() if reference.get(key) != current.get(key)]
            state_comparisons.append(dict(label=label, differences=differences))
        captures = {x['label']:{f['name']:f for f in x['files']}
                    for x in record['pre_DT_layer_captures']}
        comparisons = []
        for name in sorted(captures['old_first'].keys() | captures['old_again'].keys()):
            files = [captures[label].get(name) for label in ['old_first', 'old_again']]
            row = dict(name=name, files=files)
            if all(files):
                values = [torch.load(f['path'], map_location='cpu', weights_only=False, mmap=True) for f in files]
                row['comparison'] = summarize(*values)
                del values
            comparisons.append(row)
        old = [torch.load(args.result / f'rank{rank}-native-{label}.pt',
                         map_location='cpu', weights_only=False) for label in ['old', 'old-again']]
        pre_dt = dict(fields={key:summarize(old[0][key],old[1][key]) for key in old[0]},
                      tensor_sha256=[{key:tensor_sha(value) for key,value in x.items()} for x in old])
        final_norm = torch.load(captures['old_first']['final_norm']['path'],
                                map_location='cpu', weights_only=False, mmap=True)
        previous = []
        for folder in ['textcraft-native-incident37-native-head-live-capture-20261011-v1',
                       'textcraft-native-incident37-all-DT-live-head-capture-20261011-v1']:
            p = receipt_root / folder / 'result' / f'rank{rank}-native-head-live-microbatch1.pt'
            prior = torch.load(p, map_location='cpu', weights_only=False, mmap=True)
            previous.append(dict(path=str(p), hidden_comparison=summarize(prior['tensors']['hidden_states'], final_norm)))
            del prior
        result['records'].append(dict(rank=rank, source_record=dict(path=str(path),sha256=sha(path)),
            original_input=record['input_artifact'], fingerprint_fields=len(reference),
            before_old_runtime={k:v for k,v in states['before_old'].items() if k!='fields'},
            parameter_buffer_comparisons=state_comparisons, layer_comparisons=comparisons,
            native_old_again=pre_dt, previous_native_head_hidden=previous,
            native_call_counts=[x['counts'] for x in record['pre_DT_layer_captures']],
            capture_errors=[x['errors'] for x in record['pre_DT_layer_captures']]))
        del old, final_norm, record, states, reference
    result.update(CUDA_initialized=torch.cuda.is_initialized(),
                  peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=True)+'\n')
    print(json.dumps(result, indent=2, allow_nan=True))


if __name__ == '__main__':
    main()

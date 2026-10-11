"""Read-only inspection of native checkpoint tensors/state; not a restore implementation."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import time

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    started = time.time()
    records = []
    for path in sorted(args.checkpoint.rglob('*.pt')):
        data = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        info = dict(path=str(path), bytes=path.stat().st_size, tensors=0, elements=0,
                    nonfinite=[], scalar_values={}, top_keys=list(data) if isinstance(data, dict) else None)

        def inspect(value, key):
            if isinstance(value, torch.Tensor):
                tensor = value.to_local() if hasattr(value, 'to_local') else value
                info['tensors'] += 1
                info['elements'] += tensor.numel()
                if tensor.is_floating_point() or tensor.is_complex():
                    # Bounded CPU temporaries, no distributed or GPU computation.
                    flat = tensor.reshape(-1)
                    if not all(bool(torch.isfinite(x).all()) for x in flat.split(1048576)):
                        info['nonfinite'].append(key)
                if tensor.numel() == 1:
                    info['scalar_values'][key] = tensor.item()
                if key.endswith('lm_head.weight'):
                    info['head'] = dict(key=key, shape=list(value.shape), local_shape=list(tensor.shape),
                        placements=str(getattr(value, 'placements', None)))
                    rank = int(path.stem.rsplit('_', 1)[-1])
                    local_row = 159048 - rank * ((value.shape[0] + 1) // 2)
                    if 0 <= local_row < tensor.shape[0]:
                        row = tensor[local_row].contiguous().view(torch.uint8).numpy().tobytes()
                        info['head']['row159048_SHA256'] = hashlib.sha256(row).hexdigest()
            elif isinstance(value, dict):
                for child, item in value.items():
                    inspect(item, key + '/' + str(child))
            elif isinstance(value, (tuple, list)):
                for child, item in enumerate(value):
                    inspect(item, key + '/' + str(child))
            elif value is None or isinstance(value, (str, int, float, bool)):
                if 'scheduler' in key or key.endswith('/step'):
                    info['scalar_values'][key] = value

        inspect(data, '')
        records.append(info)
        print(json.dumps(info), flush=True)
        del data
    out = dict(checkpoint=str(args.checkpoint), unix=time.time(), seconds=time.time()-started,
               CUDA_initialized=torch.cuda.is_initialized(), peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
               files=records, scope='Read-only native files. Native VERL owns actual model/optimizer/scheduler/data restore.')
    args.output.write_text(json.dumps(out, indent=2)+'\n')


if __name__ == '__main__':
    main()

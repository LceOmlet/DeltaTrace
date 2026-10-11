"""CPU-only byte comparison of saved post-fault head rows and original weights."""
from pathlib import Path
import argparse
import hashlib
import json
import resource
import time

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    start = time.perf_counter()
    import torch
    from safetensors import safe_open
    assert not torch.cuda.is_initialized()
    row_id = 159048
    index = args.model / 'model.safetensors.index.json'
    metadata = json.loads(index.read_bytes())
    key = 'lm_head.weight'
    source = args.model / metadata['weight_map'][key]
    with safe_open(source, framework='pt', device='cpu') as owner:
        # The official slice API reads the one frozen row, not the whole head.
        reference = owner.get_slice(key)[row_id:row_id + 1].reshape(-1).clone()
    def sha(value):
        return hashlib.sha256(bytes(value.contiguous().view(torch.uint8).numpy())).hexdigest()
    expected = dict(row=row_id, key=key, file=str(source), dtype=str(reference.dtype),
        shape=list(reference.shape), sha256=sha(reference))
    ranks = []
    for rank in (0, 1):
        path = args.result / f'rank{rank}-after-native-report-CPU-source.pt'
        report_path = path.with_suffix('.json')
        report = json.loads(report_path.read_bytes())
        observations = torch.load(path, map_location='cpu', weights_only=False)
        rows = []
        for observation in observations:
            for saved in observation['CPU_source_rows']:
                item = {k:v for k,v in saved.items() if k != 'buffer'}
                buffer = saved.get('buffer')
                if buffer is not None:
                    item.update(sha256=sha(buffer), exact_original_weight=torch.equal(buffer, reference),
                        changed_elements=int((buffer != reference).sum()),
                        max_abs_difference=float((buffer.float() - reference.float()).abs().max()))
                rows.append(item)
        ranks.append(dict(rank=rank, report_errors=report['errors'], CPU_source_rows=rows,
            artifact=dict(path=str(path), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())))
    out = dict(unix=time.time(), source=__file__, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        reference=expected, ranks=ranks, seconds=time.perf_counter()-start,
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        CUDA_initialized=torch.cuda.is_initialized(), model_calls=0, DT_calls=0, optimizer_steps=0,
        production_changes=0, numerical_tolerance=None,
        scope='The fixed incident row is frozen and must match its original dtype bytes. This verifies CPU source after the saved fault only; no claim about unobserved GPU transport stages or other rows.')
    args.output.write_bytes((json.dumps(out, indent=2) + '\n').encode())
    print(json.dumps(out))

if __name__ == '__main__':
    main()

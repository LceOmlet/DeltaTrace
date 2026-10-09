"""Read existing unmodified control rows; no model or DT evaluation.

Inactive query rows were paired with the same factual IDs by the original
single_reference function. Keep native score changes, compiled head changes
and signed input attribution distinct. A nonzero endpoint change is recorded
as evidence, not declared out of official tolerance by this CPU reader.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import psutil
import torch


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '-1'
    process = psutil.Process()
    start = time.perf_counter()
    spec = json.loads(args.protocol.read_bytes())
    output = dict(scope=__doc__, pid=process.pid, birth=process.create_time(),
        unix=time.time(), source=dict(path=__file__, sha256=sha(__file__)),
        protocol=dict(path=str(args.protocol), sha256=sha(args.protocol)),
        phase='reading', tasks={}, sampled_PSS_peak_bytes=0,
        production_modified=False, candidate=False,
        operations=dict(model=0, DT=0, GPU=0, optimizer=0, rollout=0, restore=0))

    def save():
        output['sampled_PSS_peak_bytes'] = max(output['sampled_PSS_peak_bytes'],
            process.memory_full_info().pss)
        output['elapsed_seconds'] = time.perf_counter()-start
        output['cuda_initialized'] = torch.cuda.is_initialized()
        assert not output['cuda_initialized']
        args.output.write_text(json.dumps(output, indent=2, allow_nan=False)+'\n')

    save()
    for task, batches in spec['tasks'].items():
        rows = []
        output['tasks'][task] = dict(batches=len(batches), rows=rows)
        for batch in batches:
            artifact = batch['artifact']
            assert sha(artifact['path']) == artifact['sha256']
            data = torch.load(artifact['path'], map_location='cpu', weights_only=False)
            signed, roots, detail = data['signed'], data['roots'], data['detail']
            assert signed.shape[0] == roots.numel() == len(batch['rows']) == 4
            assert len(detail['per_sample']) == 4
            for index, identity in enumerate(batch['rows']):
                audit = detail['per_sample'][index]
                vector = signed[index].double()
                root = float(roots[index])
                factual, reference = audit['factual_target_logp'], audit['reference_target_logp']
                rows.append(dict(identity, row=index, artifact=artifact,
                    root_effect=root, factual_target_logp=factual,
                    reference_target_logp=reference,
                    root_minus_score_difference=root-(factual-reference),
                    signed_sum=float(vector.sum()), signed_maxabs=float(vector.abs().max()),
                    signed_nonzero=int(torch.count_nonzero(vector)),
                    native_minus_signed=root-float(vector.sum())))
            del data, signed, roots, detail, vector
            save()
    output['phase'] = 'complete'
    save()
    print(json.dumps(dict(phase=output['phase'], seconds=output['elapsed_seconds'],
        PSS=output['sampled_PSS_peak_bytes'], cuda_initialized=output['cuda_initialized'],
        tasks={k:dict(batches=v['batches'],rows=len(v['rows']),
            controls=sum(not r['active_query'] for r in v['rows']),
            nonzero_control_root=sum(not r['active_query'] and r['root_effect'] != 0 for r in v['rows']))
            for k,v in output['tasks'].items()})))


if __name__ == '__main__':
    main()

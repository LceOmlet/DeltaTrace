"""Bind saved control LoRA values to the existing native actor digest contract.

Read CPU-mapped checkpoint tensors only. Calibrate the state-dict name/order
against an already measured native DTPO80 digest before using the same reader
for the saved GRPO160 checkpoint. No model or algorithm is executed.
"""
import hashlib
import json
from pathlib import Path
import time
import torch

torch.set_num_threads(4)
ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT = ROOT / 'receipts/textcraft-gradient-rootcause-20261013-v1'
DEST = OUT / 'native-trained-control-propagation-v78'
manifest_path = DEST / 'case-manifest.json'
manifest = json.loads(manifest_path.read_bytes())
old = ROOT / 'runs/textcraft-dt-formal-native-graph-fresh-20261012-v1/textcraft-dt/checkpoints/global_step_80/actor'
expected_old = {
    '0':'8ceb690d8b62f80db7d921f2e05be155efedb2ada9c960174b56bafee7abcd68',
    '1':'5f19de20ccfc55aa9f85db5fa9bcd152bea18d7bb3dc00b6bf18445c236a9182',
}
report = dict(unix=time.time(), scope=__doc__, checkpoints={}, model_calls=0,
    backward_calls=0, optimizer_calls=0, GPUs_used=0, rootcause_repair_verified=False)
for label, directory in (('original_DTPO80', old), ('saved_GRPO160', Path(manifest['checkpoint_path']))):
    ranks = []
    for rank in (0, 1):
        path = directory / f'model_world_size_2_rank_{rank}.pt'
        model = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        digest = hashlib.sha256()
        sums = dict(A=0., B=0.)
        elements = 0
        parameter_count = 0
        dtypes = set()
        for name, value in model.items():
            if '.lora_' not in name:
                continue
            value = value.to_local() if hasattr(value, 'to_local') else value
            value = value.detach().contiguous()
            digest.update(name.encode())
            digest.update(value.view(torch.uint8).numpy().tobytes())
            side = 'A' if '.lora_A.' in name else 'B'
            sums[side] += float(value.double().square().sum())
            elements += value.numel()
            parameter_count += 1
            dtypes.add(str(value.dtype))
        result = dict(rank=rank, path=str(path), LoRA_SHA256=digest.hexdigest(),
            local_LoRA_sumsq=sums, elements=elements, parameters=parameter_count, dtypes=sorted(dtypes))
        if label == 'original_DTPO80':
            assert result['LoRA_SHA256'] == expected_old[str(rank)], result
        ranks.append(result)
        del model
    report['checkpoints'][label] = dict(ranks=ranks, global_LoRA_norms={side:sum(r['local_LoRA_sumsq'][side] for r in ranks)**.5 for side in ('A','B')})
assert not torch.cuda.is_initialized()
report['status'] = 'saved_control_weight_values_bound_to_verified_native_digest_contract'
path = DEST / 'CPU-weight-value-bindings.json'
assert not path.exists()
path.write_text(json.dumps(report, indent=2) + '\n')
assert not (DEST / 'job.json').exists() and not (DEST / 'result.json').exists()
previous = DEST / 'case-manifest.before-value-bindings.json'
assert not previous.exists()
previous.write_bytes(manifest_path.read_bytes())
for case in manifest['cases']:
    case['expected_trainable_SHA256_by_rank'] = {str(r['rank']):r['LoRA_SHA256'] for r in report['checkpoints']['saved_GRPO160']['ranks']}
manifest['CPU_weight_value_binding'] = dict(path=str(path), SHA256=hashlib.sha256(path.read_bytes()).hexdigest(),
    original_DTPO80_reader_matches_actual_native_parameter_digest=True)
manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(dict(receipt=str(path), report=report)))

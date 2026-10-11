"""Locate saved head corruption using the original FSDP CPU copy-in operator.

No model construction/forward, GPU, optimizer, numerical acceptance gate, or
production patch. A one-rank Gloo group only supplies the owner operator's
process-group handle for CPU local packing; no collective is performed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import tempfile
import time

import torch
import torch.distributed as dist
from safetensors import safe_open
from torch.distributed.fsdp._fully_shard import _fsdp_collectives as owner
from torch.distributed.fsdp._fully_shard._fsdp_common import _to_dtype_if_needed


def identity(path):
    path = Path(path)
    return dict(path=str(path), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def sha_tensor(value):
    return hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    start = time.perf_counter()
    torch.set_num_threads(4)
    assert not torch.cuda.is_initialized()
    owner_identity = identity(owner.__file__)
    assert owner_identity['sha256'] == '3bf079219682d1a9439b81754af65e87f69157ec69f20f065e5565519f7093d9'
    receipts = args.root / 'receipts'
    order_path = receipts / 'textcraft-native-incident37-layer0-payload-exact-20261011-v1.json'
    known = json.loads(order_path.read_bytes())
    order = known['official_meta_order']
    result = receipts / 'textcraft-native-incident37-native-flight-recorder-20261011-v1/result'
    report = json.loads((result / 'rank0.json').read_bytes())
    incident_path = Path(report['input_artifact']['path'])
    assert identity(incident_path)['sha256'] == report['input_artifact']['sha256']
    incident = torch.load(incident_path, map_location='cpu', weights_only=False, mmap=True)
    trainable = incident['trainable_local_parameters']
    index_path = args.model / 'model.safetensors.index.json'
    index = json.loads(index_path.read_bytes())['weight_map']
    inputs, input_identities = [], []
    offset = 0
    for entry in order:
        name = entry['name']
        assert offset == entry['BF16_rank0_start_bytes']
        if '.lora_' in name:
            key = 'base_model.model.model.layers.0.' + name
            tensor = trainable[key].reshape(-1)
            origin = str(incident_path) + ':' + key
        else:
            key = 'model.language_model.layers.0.' + name.replace('.base_layer.', '.')
            source = args.model / index[key]
            with safe_open(source, framework='pt', device='cpu') as reader:
                # Original rank0 dim0 shard; every shape in this layer is even.
                assert entry['orig_shape'][0] % 2 == 0
                tensor = reader.get_slice(key)[:entry['orig_shape'][0] // 2].reshape(-1).clone()
            origin = str(source) + ':' + key
        assert tensor.numel() == entry['rank0_numel']
        # The installed FSDP parameter owner uses this same dtype helper.
        tensor = _to_dtype_if_needed(tensor, torch.bfloat16)
        inputs.append(tensor)
        input_identities.append(dict(name=name, origin=origin, elements=tensor.numel(),
            BF16_start_bytes=offset, sha256=sha_tensor(tensor)))
        offset += tensor.numel() * tensor.element_size()
    assert offset == known['expected_layer0_rank0_payload_bytes']
    with tempfile.TemporaryDirectory(prefix='dt-fsdp-cpu-') as temp:
        dist.init_process_group('gloo', init_method=(Path(temp) / 'rendezvous').as_uri(),
                                rank=0, world_size=1)
        try:
            group_name = dist.distributed_c10d._get_default_group().group_name
            # Call the unchanged installed CPU dispatch, not a copied packer.
            packed, storage = torch.ops.fsdp.all_gather_copy_in(
                inputs, [t.numel() for t in inputs], offset // 2,
                1, 0, torch.bfloat16, torch.device('cpu'), group_name, False)
        finally:
            dist.destroy_process_group()
    del inputs, incident
    first, last, center = 150784, 160353, 159048
    assert (center - first) * 4096 * 2 == known['expected_layer0_rank0_payload_offset_bytes']
    candidate = packed[:(last - first) * 4096].reshape(-1, 4096)
    center_sha = sha_tensor(candidate[center - first])
    assert center_sha == known['bad_row_sha256']
    out = dict(unix=time.time(), source=identity(__file__), owner=owner_identity,
        recorded_owner_order=identity(order_path), incident=report['input_artifact'],
        CPU_operator='torch.ops.fsdp.all_gather_copy_in CPU dispatch, world_size=1/rank=0',
        CPU_group_scope='Local packing only; the original rank0 shards already came from the two-rank incident. No all-gather communication or GPU lifecycle reproduction.',
        packed_bytes=offset, packed_sha256=sha_tensor(packed), inputs=input_identities,
        first_vocab=first, last_vocab_exclusive=last, center_vocab=center,
        center_weight_sha256=center_sha, ranks=[])
    diagnostics = []
    candidate64 = candidate.double()
    for rank in (0, 1):
        path = result / f'rank{rank}-native-GEMM-rows.pt'
        saved = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        indices = torch.tensor(saved['metadata']['copied_indices'], dtype=torch.long)
        hidden = saved['hidden'][indices].double()
        native = saved['logits'][indices, first:last]
        rounded = (hidden @ candidate64.T).to(native.dtype)
        delta = native.float() - rounded.float()
        per_vocab_max = delta.abs().max(dim=0).values
        per_vocab_exact = (delta == 0).sum(dim=0)
        rows = []
        for begin in range(0, candidate.shape[0], 256):
            end = min(begin + 256, candidate.shape[0])
            values = delta[:, begin:end]
            rows.append(dict(first_vocab=first + begin, last_vocab_exclusive=first + end,
                exact_elements=int((values == 0).sum()), total_elements=values.numel(),
                max_abs_difference=float(values.abs().max()),
                median_abs_difference=float(values.abs().median())))
        diagnostics.append(dict(rank=rank, max_abs_difference_per_vocab=per_vocab_max,
            exact_positions_per_vocab=per_vocab_exact, selected_indices=indices))
        out['ranks'].append(dict(rank=rank, saved_GEMM=identity(path), positions=len(indices),
            compared_vocab_rows=candidate.shape[0],
            exact_elements=int((delta == 0).sum()), total_elements=delta.numel(),
            max_abs_difference=float(delta.abs().max()),
            abs_difference_quantiles={str(q): float(torch.quantile(delta.abs().flatten(), q))
                                     for q in (0., .5, .9, .99, 1.)}, chunks=rows))
        del saved, hidden, native, rounded, delta
    artifact = args.output.with_suffix('.pt')
    torch.save(diagnostics, artifact)
    out.update(artifact=identity(artifact), seconds=time.perf_counter() - start,
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        CUDA_initialized=torch.cuda.is_initialized(), model_calls=0, GPU_calls=0,
        production_changes=0, new_tolerance=None,
        scope='Known frozen/incident trainable values, original saved FSDP order, and unchanged CPU packing operator. Predicted logits describe the source hypothesis only; FP64/BF16 residuals are not acceptance thresholds or a fix.')
    args.output.write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out))


if __name__ == '__main__':
    main()

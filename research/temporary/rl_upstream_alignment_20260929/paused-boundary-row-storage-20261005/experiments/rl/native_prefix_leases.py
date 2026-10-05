"""Lease captured factual IDs to the unchanged ordered DT consumers.

NativePrefixArtifacts, FLA and HF Cache own computation and transitions.
VERL owns padding. This adapter only groups the exact source artifacts already
present in the current response RPC; nothing is retained across policy updates.
"""
import math
import time

import torch


def prepare_native_prefix_leases(runner, requests, *, minibatch_size, eos_token_id):
    from qwen35_native_prefix_artifacts import NativePrefixArtifacts, NativePrefixLease
    from verl import DataProto
    from verl.protocol import pad_dataproto_to_divisor

    started = time.perf_counter()
    batches = [requests[i:i+minibatch_size] for i in range(0, len(requests), minibatch_size)]
    # Reuse the owner's original MIN collective on a boundary known to precede
    # every current action. No current action, query or target is captured.
    prefix_lengths = [runner.model.synchronize_prefix_start(
        min(request['start'] for request in batch)//64*64) for batch in batches]
    needed = {}
    for batch, length in zip(batches, prefix_lengths):
        if length:
            for request in batch:
                needed.setdefault(request['traj_uid'], set()).add(length)

    canonical = {}
    for request in requests:
        uid = request['traj_uid']
        if uid in needed and (uid not in canonical or
                             request['prompt'].numel() > canonical[uid].numel()):
            canonical[uid] = request['prompt']
    for batch, length in zip(batches, prefix_lengths):
        if length:
            for request in batch:
                if not torch.equal(request['prompt'][:length], canonical[request['traj_uid']][:length]):
                    raise ValueError('Original histories do not share the requested factual prefix')

    records = list(canonical.items())
    capture_rows = len(records)
    # A native prefix forward gathers FSDP parameters, so all sharding ranks
    # execute the same count of original B4 forwards. Public mesh/collective
    # APIs supply metadata only; the owner retains all parameter transitions.
    weight = runner.model.lm_head.weight
    mesh = getattr(weight, 'device_mesh', None)
    device = runner.model.execution_device
    if mesh is not None and mesh.size() > 1:
        if mesh.ndim > 1:
            mesh = mesh['fsdp']
        count = torch.tensor(capture_rows, device=device, dtype=torch.long)
        torch.distributed.all_reduce(count, op=torch.distributed.ReduceOp.MAX, group=mesh.get_group())
        capture_rows = int(count.item())
    capture_rounds = math.ceil(capture_rows/minibatch_size)
    sources = {}
    capture_tokens = 0
    if capture_rounds:
        indices = DataProto.from_dict(tensors={'canonical_index': torch.arange(len(records))})
        # Capture is a representation producer, not a DT consumer. Keep the
        # original consumer order, but group factual histories by the largest
        # boundary actually needed so native B4 padding is not set by a random
        # long history. VERL owns the permutation and subsequent row padding.
        indices.reorder(torch.tensor(sorted(range(len(records)),
            key=lambda row: max(needed[records[row][0]])), dtype=torch.long))
        indices, _ = pad_dataproto_to_divisor(indices, capture_rounds*minibatch_size)
        indices = indices.batch['canonical_index'].tolist()
        # Match the original sources[uid] last-write semantics after native
        # padding. Earlier repeated slots are not consumed by any lease.
        last_slots = {records[row][0]: slot for slot, row in enumerate(indices)}
        for offset in range(0, len(indices), minibatch_size):
            rows = indices[offset:offset+minibatch_size]
            lengths = sorted(set(n for row in rows for n in needed[records[row][0]]))
            width = max(lengths)
            ids = torch.full((minibatch_size, width), eos_token_id, device=device, dtype=torch.long)
            for i, row in enumerate(rows):
                prompt = records[row][1]
                end = min(width, prompt.numel())
                ids[i, :end] = prompt[:end].to(device)
            boundary_rows = {n: [i for i, row in enumerate(rows)
                if last_slots[records[row][0]] == offset+i and n in needed[records[row][0]]]
                for n in lengths}
            artifacts = NativePrefixArtifacts.capture(runner.model, ids, lengths,
                config=runner.model._conditional.config, boundary_rows=boundary_rows)
            for i, row in enumerate(rows):
                sources[records[row][0]] = (artifacts, i)
            capture_tokens += ids.numel()
            del ids
    leases = [NativePrefixLease([sources[r['traj_uid']] for r in batch], length)
              if length else None for batch, length in zip(batches, prefix_lengths)]
    torch.cuda.synchronize()
    return leases, dict(capture_rounds=capture_rounds, local_unique_histories=len(records),
        consumer_batches=len(batches), capture_token_slots=capture_tokens,
        original_shared_prefix_token_slots=sum(len(b)*n for b,n in zip(batches,prefix_lengths)),
        capture_and_preparation_seconds=time.perf_counter()-started)

"""Prepare a transport-only candidate; never modify the default or live entry."""
from pathlib import Path
import difflib
import hashlib
import json

from stage_environment_entry import AUDIT, REPO


def patch(source):
    old = "        from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto\n"
    new = "        from verl.protocol import pad_dataproto_to_divisor\n        from ray.util.actor_pool import ActorPool\n"
    assert source.count(old) == 1
    source = source.replace(old, new)
    old = "        calls = tokens = 0\n        started = time.monotonic()\n        while len(results) < len(pool.processes):\n"
    new = '''        calls = tokens = 0
        ready = 0
        # The original Functor's name is the original wire method. VERL's
        # single-worker helper retains its fused/non-fused routing.
        wire_method = type(actor_rollout_wg.generate_sequences).__name__
        engines = ActorPool(actor_rollout_wg.workers)

        def submit(actor, item):
            # Cancellation comes from LOOP. Recheck when Ray allocates an
            # actor, because requests can wait in ActorPool's native queue.
            ranks = item.non_tensor_batch['loop_reply_rank']
            valid = item.non_tensor_batch['loop_reply_valid']
            keep = [i for i, rank in enumerate(ranks)
                    if valid[i] and not pool.cancellations[int(rank)].is_set()]
            item = item.select_idxs(torch.tensor(keep, dtype=torch.long))
            ref = actor_rollout_wg._execute_remote_single_worker(actor, wire_method, item)
            # A native Ray future wakes the same existing queue reader. It
            # carries no result payload; ActorPool owns collection/order.
            ref.future().add_done_callback(lambda _: pool.output.put(('engine_ready',)))
            return ref

        started = time.monotonic()
        while len(results) < len(pool.processes) or engines.has_next():
'''
    assert source.count(old) == 1
    source = source.replace(old, new)
    old = "                    else:\n                        raise RuntimeError(f'Unexpected LOOP transport event: {event[0]}')\n"
    new = "                    elif event[0] == 'engine_ready':\n                        ready += 1\n" + old
    assert source.count(old) == 1
    source = source.replace(old, new)
    start = source.index("            if not requests:\n                continue\n")
    end = source.index("        self.records = records\n", start)
    original = source[start:end]
    prep_start = original.index("            prompts, options = [], []\n")
    prep_end = original.index("            batch, padding = pad_dataproto_to_divisor")
    preparation = original[prep_start:prep_end]
    replacement = "            if requests:\n" + ''.join('    '+line+'\n' for line in preparation.splitlines())
    replacement += '''                batch.non_tensor_batch.update(
                    loop_reply_rank=np.array([r for _, r, _, _ in requests]),
                    loop_reply_key=np.array([k for _, _, k, _ in requests], dtype=object))
                batch, padding = pad_dataproto_to_divisor(batch, actor_rollout_wg.world_size)
                batch.non_tensor_batch['loop_reply_valid'] = np.arange(len(batch)) < len(requests)
                for item in batch.chunk(actor_rollout_wg.world_size):
                    engines.submit(submit, item)
                calls += 1
            while ready:
                # The owner callback says a native ref is ready, so there
                # is no polling interval and no wait for the other rank.
                output = engines.get_next_unordered(timeout=0)
                ready -= 1
                for index in range(len(output)):
                    rank = int(output.non_tensor_batch['loop_reply_rank'][index])
                    key = output.non_tensor_batch['loop_reply_key'][index]
                    reply = policy_reply(output, index)
                    records[key] = output.select_idxs([index])
                    pool.replies[rank].put((key, reply))
                    tokens += len(reply.token_ids)
                print(f'[loop_transport] calls={calls} rpc_mode=actor_pool '
                      f'batch_requests={len(output)} queued_requests={pool.output.qsize()} '
                      f'requests={len(records)} generated_tokens={tokens} '
                      f'elapsed={time.monotonic()-started:.1f}s completed_ranks={len(results)}', flush=True)
'''
    source = source[:start]+replacement+source[end:]
    compile(source, 'rank-completion-candidate/loop_owner_rollout.py', 'exec')
    return source


if __name__ == '__main__':
    original = REPO/'experiments/rl/loop_owner_rollout.py'
    output = AUDIT/'appworld-rank-completion-20261002'
    output.mkdir(exist_ok=True)
    before = original.read_text(encoding='utf8')
    after = patch(before)
    (output/'loop_owner_rollout.py').write_text(after, encoding='utf8', newline='\n')
    (output/'candidate.diff').write_text(''.join(difflib.unified_diff(
        before.splitlines(True), after.splitlines(True),
        fromfile='verified/loop_owner_rollout.py', tofile='candidate/loop_owner_rollout.py')), encoding='utf8')
    (output/'source.json').write_text(json.dumps(dict(
        status='unaccepted_transport_candidate_not_deployed',
        default_source=str(original), default_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
        candidate_sha256=hashlib.sha256((output/'loop_owner_rollout.py').read_bytes()).hexdigest(),
        owner_apis=['Ray ActorPool', 'Ray ObjectRef.future',
                    'VERL RayWorkerGroup._execute_remote_single_worker', 'VERL DataProto.chunk'],
        scope='No scheduler, sampler, decoder, model, DT, PPO, reward or budget implementation'), indent=2)+'\n', encoding='utf8')
    print(output)

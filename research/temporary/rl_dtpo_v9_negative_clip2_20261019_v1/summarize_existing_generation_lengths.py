"""Reduce already collected native request lengths; no model or new log parser."""
import hashlib
import json
from pathlib import Path
import statistics


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def main():
    observation = HERE / 'current-generation-observer-native-rows.json'
    source = json.loads(observation.read_text(encoding='utf-8'))
    owner = json.loads((HERE / 'current-AgentGym-generation-owner-source.json').read_text(encoding='utf-8'))
    binding = json.loads((HERE / 'current-progress-and-owner-source.json').read_text(encoding='utf-8'))
    assert not source['errors']
    rows = []
    identities = set()
    for original in source['rows']:
        row = original['value']
        key = (row['rank'], row['call'])
        assert key not in identities
        identities.add(key)
        lengths = row['generated_token_lengths']
        assert len(lengths) == row['requests'] and lengths
        assert row['additional_generation_calls'] == 0
        assert row['original_outputs_returned_unchanged']
        maximum = max(lengths)
        slots = len(lengths) * maximum
        rows.append(dict(rank=row['rank'], call=row['call'], requests=len(lengths),
                         actual_generated_tokens=sum(lengths), rectangular_token_slots=slots,
                         filled_fraction=sum(lengths) / slots,
                         requests_shorter_than_batch_maximum=sum(n < maximum for n in lengths),
                         maximum_generated_length=maximum,
                         native_generate_seconds=row['native_generate_seconds'],
                         source_path=original['path'], source_line=original['line']))
    assert len(identities) == 64 and {r['rank'] for r in rows} == set(range(8))
    tokens = sum(r['actual_generated_tokens'] for r in rows)
    slots = sum(r['rectangular_token_slots'] for r in rows)
    requests = sum(r['requests'] for r in rows)
    short = sum(r['requests_shorter_than_batch_maximum'] for r in rows)
    report = dict(
        status='read_only_existing_generation_length_evidence',
        observation_collected=source['collected'],
        source_observation=dict(path=str(observation), SHA256=hashlib.sha256(observation.read_bytes()).hexdigest(),
                                original_reader=source['reader_source'], original_reader_SHA256=source['reader_SHA256']),
        calls=len(rows), requests=requests, actual_generated_tokens=tokens,
        rectangular_token_slots=slots, filled_fraction=tokens / slots,
        median_call_filled_fraction=statistics.median(r['filled_fraction'] for r in rows),
        requests_shorter_than_batch_maximum=short, shorter_request_fraction=short / requests,
        definition='For each original call, rectangular slots = request count * maximum generated length; filled fraction = sum generated lengths / sum rectangular slots.',
        native_generation_owner=dict(path=owner['owner_path'], SHA256=owner['owner_sha'],
            generation_call_line=264, batched_environment_step_line=273,
            behavior='Original AgentGym inference_engine.generate returns the whole response batch before agent_step runs in its original ThreadPoolExecutor.'),
        deployed_transport=dict(path=binding['entry'] + '/textcraft_owner_rollout.py',
            SHA256=binding['owner_sha'], blocking_worker_group_call_line=143,
            behavior='Original worker-group generate_sequences returns all ranks before the adapter returns responses to AgentGym.'),
        interpretation='Batch-completion barriers prevent a shorter request from starting its next environment turn while longer requests in that batch are still decoding. This is a concrete scheduling constraint, not a measured wall-time loss or a proven dominant bottleneck.',
        limitations=[
            'Rectangular token slots are an output-length proxy, not GPU-Util, executed padded FLOPs, idle seconds or a speedup estimate.',
            'Native per-request TTFT and finish timestamps were absent; no request-level waiting time is inferred.',
            'The current engine already has async_scheduling and CUDA graphs enabled; these do not remove the owner batch-completion barrier.',
            'Pinned VERL contains AsyncvLLMServer, but the currently selected AgentGym generate-then-step owner has no request-completion hook. A server switch alone does not establish equivalent task execution or LoRA compatibility.',
        ],
        extra_generation_calls=0, extra_environment_calls=0, optimizer_calls=0,
        deployment_changed=False, calls_with_sources=rows)
    target = ROOT / 'experiments/rl/results_dtpo_v9_negative2_generation_batch_barrier_20261019_v4.json'
    target.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(output=str(target), calls=len(rows), requests=requests,
                         filled_fraction=tokens / slots, shorter_request_fraction=short / requests)))


if __name__ == '__main__':
    main()

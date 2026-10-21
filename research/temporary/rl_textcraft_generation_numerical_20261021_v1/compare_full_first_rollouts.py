"""Compare captured native requests, outputs and full environment returns.

No text/action parser or generation algorithm is implemented here. The original
DataProto pad/select/chunk interfaces map captured RPC rows back to their native
episode identities; actual worker prompts and seeds verify that mapping.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time


def main():
    import numpy as np
    import torch
    from verl import DataProto
    from verl.protocol import pad_dataproto_to_divisor

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--left', type=Path, required=True)
    parser.add_argument('--right', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sources = {}

    def read(path, binary=False):
        path = Path(path)
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return torch.load(path, map_location='cpu', weights_only=False) if binary else json.loads(path.read_text())

    def load_case(directory):
        receipt = read(directory.parent / 'receipt.json')
        case = receipt['cases'][directory.name]
        assert case['status'] == 'completed_native_full_rollout', (str(directory), case['status'])
        world = receipt['world_size']
        episodes = read(directory / 'environment-observations.json')
        native_export = read(directory / 'native-executer-logs/step1/0.json')
        assert len(episodes) == len(native_export) == len(case['scores']) == 256
        for row, (episode, native) in enumerate(zip(episodes, native_export)):
            assert episode['item_id'] == native['item_id']
            assert episode['final_score'] == native['reward'] == case['scores'][row]

        generated = [[] for _ in episodes]
        call_index = 0
        for turn, _ in enumerate(case['rounds']):
            captured = read(directory / f'round{turn:02d}-actual-owner-requests.pt', binary=True)
            rows = captured['rows']
            logical = DataProto.from_dict(tensors={'position': torch.arange(len(rows))[:, None]})
            divisor = 8 if directory.name == 'matched' else world
            logical, _ = pad_dataproto_to_divisor(logical, divisor)
            width = len(logical) // 8 * world if directory.name == 'matched' and world == 2 else len(logical)
            for start in range(0, len(logical), width):
                chunks = logical.select_idxs(np.arange(start, start + width)).chunk(world)
                for rank, chunk in enumerate(chunks):
                    rank_dir = directory.parent / f'rank{rank}' / directory.name
                    request = read(rank_dir / f'{call_index:03d}-requests.pt', binary=True)
                    outputs = read(rank_dir / f'{call_index:03d}-outputs.pt', binary=True)
                    assert len(request['prompts']) == len(request['sampling_params']) == len(outputs) == len(chunk)
                    for local_row, output in enumerate(outputs):
                        position = int(chunk.batch['position'][local_row, 0])
                        prompt = captured['raw_prompt_ids'][position]
                        seed = captured['sampling_kwargs'][position].get('seed')
                        assert request['prompts'][local_row]['prompt_token_ids'] == prompt
                        assert list(output.prompt_token_ids) == prompt
                        assert request['sampling_params'][local_row].seed == seed
                        global_position = start + rank * len(chunk) + local_row
                        if global_position >= len(rows):
                            continue  # Original unpad drops these exact trailing RPC copies.
                        assert position == global_position
                        generated[rows[position]].append(dict(turn=turn, prompt=prompt, seed=seed,
                            output=list(output.outputs[0].token_ids), logprobs=output.outputs[0].logprobs))
                call_index += 1
        assert all(len(tokens) == len(episode['steps']) for tokens, episode in zip(generated, episodes))
        return receipt, case, episodes, generated

    left_receipt, left_case, left, left_generated = load_case(args.left)
    right_receipt, right_case, right, right_generated = load_case(args.right)
    rows = []
    same_input_pairs = 0
    same_raw_output_pairs = 0
    same_identity_pairs = 0
    same_identity_seed_pairs = 0
    same_input_seed_pairs = 0
    comparable_chosen_logps = 0
    chosen_logp_difference_sum = 0.0
    chosen_logp_difference_max = 0.0
    for row, (a, b, ag, bg) in enumerate(zip(left, right, left_generated, right_generated)):
        assert a['item_id'] == b['item_id'], row
        same_actions = [step['action'] for step in a['steps']] == [step['action'] for step in b['steps']]
        same_returns = [[step.get(key) for key in ('state', 'reward', 'done')] for step in a['steps']] == [
            [step.get(key) for key in ('state', 'reward', 'done')] for step in b['steps']]
        first_divergence = None
        for at, (x, y) in enumerate(zip(ag, bg)):
            assert x['turn'] == y['turn']
            same_identity_pairs += 1
            same_identity_seed_pairs += x['seed'] == y['seed']
            if x['prompt'] != y['prompt']:
                continue
            same_input_pairs += 1
            same_input_seed_pairs += x['seed'] == y['seed']
            same_raw_output_pairs += x['output'] == y['output']
            for offset, (xt, yt) in enumerate(zip(x['output'], y['output'])):
                if xt != yt:
                    if first_divergence is None:
                        first_divergence = dict(turn=at, token_offset=offset, left_token=xt, right_token=yt)
                    break
                # The entire conditioning prefix is identical at these positions.
                delta = abs(x['logprobs'][offset][xt].logprob - y['logprobs'][offset][yt].logprob)
                comparable_chosen_logps += 1
                chosen_logp_difference_sum += delta
                chosen_logp_difference_max = max(chosen_logp_difference_max, delta)
        rows.append(dict(row=row, item_id=a['item_id'], sample=row % 8,
                         initial_observation_equal=a['reset']['observation'] == b['reset']['observation'],
                         first_actual_request_equal=ag[0]['prompt'] == bg[0]['prompt'],
                         actual_request_sequence_equal=[x['prompt'] for x in ag] == [x['prompt'] for x in bg],
                         actual_seed_sequence_equal=[x['seed'] for x in ag] == [x['seed'] for x in bg],
                         raw_generated_token_sequence_equal=[x['output'] for x in ag] == [x['output'] for x in bg],
                         model_response_sequence_equal=same_actions, native_environment_returns_equal=same_returns,
                         left_score=a['final_score'], right_score=b['final_score'],
                         score_difference=a['final_score'] - b['final_score'],
                         left_steps=len(a['steps']), right_steps=len(b['steps']),
                         first_same_input_output_token_divergence=first_divergence))

    result = dict(collected_unix=time.time(), left=str(args.left), right=str(args.right),
                  left_world_size=left_receipt['world_size'], right_world_size=right_receipt['world_size'],
                  prepared_batch_SHA256=left_receipt['prepared_batch_SHA256'],
                  prepared_batch_equal=left_receipt['prepared_batch_SHA256'] == right_receipt['prepared_batch_SHA256'],
                  left_rng=left_case['rng'], right_rng=right_case['rng'], episodes=256,
                  left_score_sum=left_case['score_sum'], right_score_sum=right_case['score_sum'],
                  left_score_mean=left_case['score_mean'], right_score_mean=right_case['score_mean'],
                  mean_score_difference=left_case['score_mean'] - right_case['score_mean'],
                  counts={key: sum(bool(row[key]) for row in rows) for key in (
                      'initial_observation_equal', 'first_actual_request_equal', 'actual_request_sequence_equal',
                      'actual_seed_sequence_equal', 'raw_generated_token_sequence_equal',
                      'model_response_sequence_equal', 'native_environment_returns_equal')},
                  different_rewards=sum(row['left_score'] != row['right_score'] for row in rows),
                  identical_model_responses_with_different_environment_returns=sum(
                      row['model_response_sequence_equal'] and not row['native_environment_returns_equal'] for row in rows),
                  same_actual_input_request_pairs=same_input_pairs, same_raw_output_request_pairs=same_raw_output_pairs,
                  same_task_sample_turn_request_pairs=same_identity_pairs,
                  same_task_sample_turn_seed_pairs=same_identity_seed_pairs,
                  same_actual_input_and_seed_pairs=same_input_seed_pairs,
                  identical_prefix_chosen_logp_positions=comparable_chosen_logps,
                  identical_prefix_chosen_logp_abs_difference_max=chosen_logp_difference_max,
                  identical_prefix_chosen_logp_abs_difference_mean=(chosen_logp_difference_sum / comparable_chosen_logps
                                                                 if comparable_chosen_logps else None),
                  rows=rows, sources=sources,
                  analysis_reader_source=dict(path=__file__, SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
                  model_calls=0, DT_calls=0, environment_calls=0, optimizer_updates=0,
                  tolerance_verdict=None, scope='One full native32x8 first-rollout batch; no invented scalar tolerance or universal quality claim')
    args.output.parent.mkdir(exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key not in ('rows', 'sources')}, indent=2))


if __name__ == '__main__':
    main()

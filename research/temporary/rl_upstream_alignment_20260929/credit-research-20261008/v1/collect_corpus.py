"""CPU inventory of original prepared rows and all completed native DT captures.

This reads owner-emitted IDs/masks/UIDs. It neither reconstructs environments
nor evaluates/modifies credit. Initial policy-state identity includes the
original first environment observation, not just the generic system prompt.
It is a grouping key, not a fabricated native task ID. Missing captures remain
explicit.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import resource
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import torch

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
EXPECTED = {
    'textcraft': '2796233e2683f1939896c74b2b578c242dbd7a7f235b9ef61cbedd398f61be52',
    'appworld': '58209daa0fccfea4b70645465e96ea5d203f9d309187b7a64b405cbd9fd47da0',
}


def artifact(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return {'path': str(path), 'sha256': digest.hexdigest(), 'bytes': path.stat().st_size}


def tensor_sha(value):
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256(str((str(value.dtype), tuple(value.shape))).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def collect(task):
    base = ROOT/'receipts/direct-target-prefix-runtime-20261007-v1'/(task+'-first-dt')
    source = artifact(ROOT/f'runs/direct-target-prefix-runtime-20261007-v1/{task}/{task}-dt/source.json')
    assert source['sha256'] == EXPECTED[task]
    records, prepared_files, native_files = {}, [], []
    prepared_rows = 0
    for path in sorted(base.glob('rank*-input-prepared.pt')):
        binding = artifact(path)
        prepared_files.append(binding)
        value = torch.load(path, map_location='cpu', weights_only=False)
        batch, uids = value['batch'], value['non_tensor_batch']['traj_uid']
        prompt_width = batch['prompts'].shape[1]
        for index, uid in enumerate(uids):
            uid = str(uid)
            ids, attention = batch['input_ids'][index], batch['attention_mask'][index].bool()
            prompt = batch['prompts'][index][attention[:prompt_width]]
            first_policy = int(batch['policy_mask'][index].nonzero()[0])
            state_end = prompt_width+first_policy
            initial_state = ids[:state_end][attention[:state_end]]
            record = {
                'task': task, 'traj_uid': uid,
                'initial_prompt_sha256': tensor_sha(prompt),
                'initial_prompt_tokens': prompt.numel(),
                'initial_state_sha256': tensor_sha(initial_state),
                'initial_state_tokens': initial_state.numel(),
                'unpad_input_sha256': tensor_sha(ids[attention]),
                'context_tokens': int(attention.sum()),
                'policy_tokens': int(batch['policy_mask'][index].sum()),
                'target_tokens': int(batch['target_mask'][index].sum()),
                'reward': float(batch['dt_direct_reward'][index]),
                'prepared_occurrences': [{'file_sha256': binding['sha256'], 'row': index}],
                'native_occurrences': [],
            }
            prepared_rows += 1
            if uid in records:
                previous = records[uid]
                for key in ('initial_prompt_sha256', 'initial_state_sha256', 'unpad_input_sha256', 'context_tokens',
                            'policy_tokens', 'target_tokens', 'reward'):
                    if record[key] != previous[key]:
                        raise ValueError(f'Copied UID has changed original data: {task}/{uid}/{key}')
                previous['prepared_occurrences'].extend(record['prepared_occurrences'])
            else:
                records[uid] = record
        del value, batch
    for path in sorted(base.glob('rank*-readout-native-batch-*.pt')):
        binding = artifact(path)
        native_files.append(binding)
        value = torch.load(path, map_location='cpu', weights_only=False)
        for row in sorted(value['rows'], key=lambda r: r['batch_row']):
            uid = str(row['traj_uid'])
            record = records[uid]
            if tensor_sha(row['selected'][:row['prompt_length']]) != record['initial_prompt_sha256']:
                raise ValueError(f'Prepared/native original prompt mismatch: {task}/{uid}')
            native_first = int(row['policy'][row['suffix_positions']].nonzero()[0])
            native_state = row['selected'][:row['prompt_length']+native_first]
            if tensor_sha(native_state) != record['initial_state_sha256']:
                raise ValueError(f'Prepared/native initial state mismatch: {task}/{uid}')
            if float(row['row']['dt_direct_reward']) != record['reward']:
                raise ValueError(f'Prepared/native reward mismatch: {task}/{uid}')
            positions = row['prompt_length'] + row['prior'][row['suffix_positions']].nonzero().flatten()
            signed = value['native_signed'][row['batch_row'], positions]
            occurrence = {
                'file_sha256': binding['sha256'], 'batch_row': int(row['batch_row']),
                'trajectory_index': int(row['trajectory_index']),
                'selected_tokens': row['selected'].numel(),
                'selected_ids_sha256': tensor_sha(row['selected']),
                'target_offsets_sha256': hashlib.sha256(json.dumps(row['target_offsets']).encode()).hexdigest(),
                'source_tokens': positions.numel(),
                'native_signed_sha256': tensor_sha(signed),
                'native_signed_dtype': str(signed.dtype),
            }
            if record['native_occurrences']:
                old = record['native_occurrences'][0]
                for key in ('selected_ids_sha256', 'target_offsets_sha256', 'source_tokens'):
                    if old[key] != occurrence[key]:
                        raise ValueError(f'Copied native UID has changed identities: {task}/{uid}/{key}')
            record['native_occurrences'].append(occurrence)
        del value
    rows = sorted(records.values(), key=lambda r: r['traj_uid'])
    return {
        'source': source, 'prepared_files': prepared_files, 'native_files': native_files,
        'prepared_rows': prepared_rows, 'unique_trajectories': len(rows),
        'duplicate_prepared_rows': prepared_rows-len(rows),
        'completed_native_unique': sum(bool(r['native_occurrences']) for r in rows),
        'missing_native_unique': sum(not r['native_occurrences'] for r in rows),
        'initial_prompt_groups': len({r['initial_prompt_sha256'] for r in rows}),
        'initial_state_groups': len({r['initial_state_sha256'] for r in rows}),
        'records': rows,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    started = time.perf_counter()
    result = {'scope': __doc__, 'tasks': {task: collect(task) for task in EXPECTED},
              'collector': artifact(Path(__file__).resolve())}
    result['runtime'] = {
        'unix': time.time(), 'seconds': time.perf_counter()-started,
        'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        'cuda_initialized': torch.cuda.is_initialized(), 'model_calls': 0,
        'DT_calls': 0, 'optimizer_steps': 0,
    }
    assert not result['runtime']['cuda_initialized']
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'output': str(args.output), 'runtime': result['runtime'],
                      'tasks': {t: {k: v for k, v in d.items() if k not in
                          ('source', 'prepared_files', 'native_files', 'records')}
                                for t, d in result['tasks'].items()}}))

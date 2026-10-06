"""CPU target-coordinate interfaces on original saved AppWorld requests.

This calls the real PackedAnswerTargets and NativeTargetLogitRows owners. It
does not build model activations, logits, an alternate reference, or a runner.
"""
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys
import time

import psutil
import torch


HERE = Path(__file__).resolve().parent
REMOTE_ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
SAVED = REMOTE_ROOT/'receipts/owner-b8-dispatch-20260930/native-prefix-reuse-workload-20261004-712795d'
EXPECTED_REQUESTS = [
    '8714abf248a4bb3201294300b6525030fd0bdbce55899d969d9f24236729e429',
    '1fdc1eb0f701d921825b796426ad2d7b0751f61786710b830f4ea72a11e58818',
]
EXPECTED_CANDIDATES = {
    'qwen35_native_prefix_artifacts.py': '50af8daf2ce5beb44c474801dbdd3bc00e736d2f92c530a0ef17b1c0168d35ee',
    'native_prefix_leases.py': '54ad5ae1a07cefcbc4002fa63e7450d8281493a93567ad3df46cb7e9c2e74260',
    'qwen35_answer_finite.py': 'd47333ea68fb7a332e7d1dce7913c989d875ea262dfe49cfa4f20f7c35ebe03e',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def compare_selection(left, right):
    assert left.batch == right.batch and left.length == right.length
    assert left.counts == right.counts and left.offsets == right.offsets
    for field in ['samples', 'positions', 'labels', 'paired_samples', 'paired_positions', 'outcome_token_ids']:
        assert torch.equal(getattr(left, field), getattr(right, field)), field


def main():
    started = time.time()
    assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
    assert os.environ['MACA_VISIBLE_DEVICES'] == '-1'
    assert not torch.cuda.is_initialized()
    current_path = REMOTE_ROOT/'runs/appworld-fresh-native-conv-canonical-20261007-v3/appworld-dt/source.json'
    current = json.loads(current_path.read_text())
    original_dir = Path(current['dt_root'])/'clean/qwen35'
    sys.path.insert(0, str(original_dir))
    candidates = {}
    for name, digest in EXPECTED_CANDIDATES.items():
        path = HERE/'candidate'/name
        assert sha(path) == digest, name
        candidates[name] = {'path': str(path), 'sha256': digest}

    # Removing only the opt-in coordinate method must restore the entire owner
    # module AST. The two other owners use the unchanged prepared checker.
    old_path = HERE/'baseline/qwen35_answer_finite.py'
    new_path = HERE/'candidate/qwen35_answer_finite.py'
    before, after = ast.parse(old_path.read_bytes()), ast.parse(new_path.read_bytes())
    projected = copy.deepcopy(after)
    owner = next(n for n in projected.body if isinstance(n, ast.ClassDef) and n.name == 'PackedAnswerTargets')
    removed = [n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == 'suffix_rows']
    assert len(removed) == 1
    owner.body = [n for n in owner.body if n not in removed]
    assert ast.dump(before, include_attributes=False) == ast.dump(projected, include_attributes=False)

    baseline = load_module(old_path, '_saved_rows_original_answer')
    candidate = load_module(new_path, '_saved_rows_candidate_answer')
    native_path = original_dir/'native_target_logit_rows.py'
    native = load_module(native_path, '_saved_rows_original_native_targets')
    assert not torch.cuda.is_initialized()

    payloads, request_sources = [], []
    for rank, digest in enumerate(EXPECTED_REQUESTS):
        path = SAVED/f'actual-requests-rank{rank}.pt'
        assert sha(path) == digest
        # These fixed-hash, internally captured request dumps contain numpy
        # scalar metadata. They are neither model weights nor checkpoints.
        payload = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        original_requests = payload['requests']
        assert len(original_requests) == 88 and payload['minibatch_size'] == 4
        # This is the original reward_readout.py consumer's stable context sort,
        # also used by the existing saved-request geometry audit. No new order.
        requests = sorted(original_requests, key=lambda request: request['context_tokens'])
        original_indices = {id(request): index for index, request in enumerate(original_requests)}
        payload['requests'] = requests
        payload['_saved_request_indices'] = [original_indices[id(request)] for request in requests]
        payloads.append(payload)
        request_sources.append({'rank': rank, 'path': str(path), 'sha256': digest, 'bytes': path.stat().st_size})

    batches = []
    for rank, payload in enumerate(payloads):
        requests = payload['requests']
        for batch_index, offset in enumerate(range(0, len(requests), 4)):
            batch = requests[offset:offset+4]
            cuts = tuple(int(r['start'])//64*64 for r in batch)
            lengths = tuple(int(r['context_tokens']) for r in batch)
            # Current original scalar owner synchronizes branch participation;
            # each rank retains its own local common boundary (source3cd).
            shared_cut = min(cuts)
            cases = [r['case'] for r in batch]
            width = max(lengths)
            options = {'outcome_token_ids': payload['outcome_token_ids']}
            old = baseline.PackedAnswerTargets(cases, [[0] for _ in batch], width, 'cpu', **options)
            selected = candidate.PackedAnswerTargets(cases, [[0] for _ in batch], width, 'cpu', **options)
            compare_selection(old, selected)
            uniform = selected.suffix_rows([shared_cut]*4, lengths)
            compare_selection(old.suffix(shared_cut), uniform)
            rows = selected.suffix_rows(cuts, lengths)
            assert rows.batch == 4 and rows.length == max(n-p for n,p in zip(lengths,cuts))
            for field in ['samples', 'labels', 'paired_samples', 'outcome_token_ids']:
                assert getattr(rows, field) is getattr(selected, field), field
            assert rows.counts is selected.counts and rows.offsets is selected.offsets
            original_positions = selected.positions.clone()
            assert torch.equal(rows.positions+torch.tensor(cuts)[rows.samples], original_positions)
            assert torch.equal(rows.paired_positions, rows.positions.repeat_interleave(2))
            target_rows = native.NativeTargetLogitRows(rows)
            assert torch.equal(target_rows.rows[target_rows.packed_rows], rows.paired_positions)
            assert torch.equal(target_rows.packed_rows[0::2], target_rows.packed_rows[1::2])
            assert torch.equal(rows.paired_samples, (2*rows.samples[:, None]+torch.arange(2)[None, :]).flatten())
            action_indices = []
            for sample, (request, cut, length) in enumerate(zip(batch, cuts, lengths)):
                full_ids = torch.cat(tuple(request[name] for name in ['prompt', 'actions', 'query', 'target']))
                assert len(full_ids) == length == request['case']['prompt_length']+len(request['case']['target_ids'])
                assert request['start'] == request['prompt'].numel()
                assert request['end']-request['start'] == request['actions'].numel()
                assert torch.equal(full_ids[request['start']:request['end']], request['actions'])
                local_start, local_end = int(request['start'])-cut, int(request['end'])-cut
                assert torch.equal(full_ids[cut:][local_start:local_end], request['actions'])
                assert list(range(local_start+cut, local_end+cut)) == list(range(int(request['start']), int(request['end'])))
                position = int(rows.positions[sample])
                assert position+cut == int(request['case']['prompt_length'])-1
                assert int(full_ids[cut:][position+1]) == int(rows.labels[sample]) == int(request['target'][0])
                action_indices.append({'original_row_index': int(request['row_index']), 'traj_uid': request['traj_uid'],
                    'source_step': int(request['source_step']), 'absolute_start': int(request['start']),
                    'absolute_end': int(request['end']), 'local_start': local_start, 'local_end': local_end})
            batches.append({'rank': rank, 'batch_index': batch_index,
                'saved_request_indices': payload['_saved_request_indices'][offset:offset+4],
                'original_consumer_sorted_positions': list(range(offset,offset+4)),
                'original_context_lengths': list(lengths), 'original_scalar_local_cut': shared_cut,
                'row_cuts': list(cuts), 'original_target_predictor_positions': original_positions.tolist(),
                'local_target_predictor_positions': rows.positions.tolist(), 'labels': rows.labels.tolist(),
                'paired_samples': rows.paired_samples.tolist(), 'native_union': target_rows.rows.tolist(),
                'native_inverse_paired': target_rows.packed_rows.tolist(), 'actions': action_indices})

    assert len(batches) == 44 and not torch.cuda.is_initialized()
    memory = psutil.Process().memory_full_info()
    result = {'status': 'CPU_real_saved_request_target_interfaces_passed', 'started_unix': started,
        'finished_unix': time.time(), 'request_sources': request_sources, 'requests': 176, 'original_B4_batches': 44,
        'owner_import': 'Complete original baseline and candidate modules imported; no AST class extraction or shadow reference',
        'sources': candidates, 'original_native_target_rows': {'path': str(native_path), 'sha256': sha(native_path)},
        'current_formal_source_read_only': {'path': str(current_path), 'sha256': sha(current_path)},
        'answer_default_module_AST_equal': True,
        'consumer_order': 'Original reward_readout.py stable context sort and consecutive B4; saved request indices retained, no new sorting criterion or grouping.',
        'checks': ['Original case/label/sample/paired identity', 'Uniform rows versus original scalar suffix',
                   'Per-row predictor minus prefix cut', 'Original NativeTargetLogitRows union/inverse',
                   'Original absolute action indices and exact token IDs restored'],
        'CUDA_VISIBLE_DEVICES': os.environ['CUDA_VISIBLE_DEVICES'], 'MACA_VISIBLE_DEVICES': os.environ['MACA_VISIBLE_DEVICES'],
        'cuda_initialized': False, 'memory_bytes': {'rss': memory.rss, 'pss': memory.pss, 'uss': memory.uss,
            'peak_rss': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024},
        'script': {'path': str(Path(__file__)), 'sha256': sha(__file__)}, 'batches': batches,
        'limits': ['Target/index/representation interface only; no model, hidden states, logits or DT values.',
                   'No FA/FLA numerical assertions, actual B8 model execution, 32k capacity or speed acceptance.',
                   'Candidate remains isolated and unaccepted; no production source/configuration/manifest modified.']}
    path = HERE/'cpu-saved-target-rows.json'
    path.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'output': str(path), 'sha256': sha(path), 'status': result['status'],
        'batches': len(batches), 'requests': 176, 'memory_bytes': result['memory_bytes']}))


if __name__ == '__main__':
    main()

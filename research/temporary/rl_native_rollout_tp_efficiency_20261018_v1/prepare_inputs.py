"""Reuse recorded native prefixes and the accepted 32k storage fixture owner."""
import ast
import hashlib
import json
from pathlib import Path

import torch
import numpy as np
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor
from reward_readout import DirectActionTargetReadout

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT = ROOT/'receipts/native-rollout-TP-efficiency-20261018-v1'


def main():
    AUDIT.mkdir(parents=True, exist_ok=True)
    graph = ROOT/'receipts/textcraft-restart-utilization-20261012-v1/graph-comparison/eager.json'
    report = json.loads(graph.read_bytes())
    prefixes, sources = [], []
    for entry in report['inputs']:
        path = Path(entry['path'])
        row = torch.load(path, map_location='cpu', weights_only=False)['rows'][0]
        prefixes.append(row['input_ids'][:entry['target_start']].tolist())
        sources.append(dict(path=str(path), SHA256=hashlib.sha256(path.read_bytes()).hexdigest(), target_start=entry['target_start']))
    assert len(prefixes) == 32
    tail_path = ROOT/'receipts/textcraft-native-dp-owner-20261018-v1/native-benchmark/0-A-global.pt'
    native = torch.load(tail_path, map_location='cpu', weights_only=False)
    tail, tail_sources = [], []
    for i, turns in enumerate(native.batch['task_rounds']):
        artifact = native.non_tensor_batch['dt_direct_target_artifact'][i]
        if int(turns) != 30 or not any(artifact['target_mask']):
            continue
        start = artifact['target_mask'].index(True)
        ids = artifact['prompt_ids'] + artifact['response_ids'][:start]
        assert 0 < len(ids) <= 32640
        tail.append(ids)
        tail_sources.append(dict(native_row=i, action_target_offset=start, prefix_tokens=len(ids)))
    assert tail, 'Actual completed native tail prefixes are required'
    raw = np.empty(len(tail), dtype=object)
    raw[:] = tail
    packet = DataProto.from_dict(tensors=dict(input_ids=torch.zeros(len(tail), 1, dtype=torch.long)),
        non_tensors=dict(raw_prompt_ids=raw))
    padded, padding = pad_dataproto_to_divisor(packet, 8)
    tail = padded.non_tensor_batch['raw_prompt_ids'].tolist()
    old = ROOT/'runs/textcraft-dt-formal-eventwait-fresh-20261012-v2/credit-records'
    captured = sorted(old.glob('rank*/*.pt'))[0]
    rows = torch.load(captured, map_location='cpu', weights_only=False)['rows']
    short = []
    for row in rows[:4]:
        prompt, ids, positions = row['prompt_length'], row['input_ids'], row['suffix_positions']
        short.append(dict(input_ids=ids, attention_mask=torch.ones_like(ids), responses=ids[prompt:],
            policy_mask=row['policy_mask'][positions], target_mask=row['target_mask'][positions],
            dt_direct_reward=row['reward'], traj_uid=row['traj_uid']))
    owner = ROOT/'receipts/direct-target-memory-capacity-20261008-v1/check_capacity_lifetime.py'
    assert hashlib.sha256(owner.read_bytes()).hexdigest() == '060070d993cef62fae79150b7e2e6ffc4d0521b7fd13e383181d285eac291d3a'
    function = next(n for n in ast.parse(owner.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'capacity_row')
    namespace = dict(torch=torch)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(owner), 'exec'), namespace)

    def prepare(row, index):
        value = DirectActionTargetReadout._prepare_row(row, index)
        return value | dict(full_context_length=value['selected'].numel())

    expanded = [namespace['capacity_row'](row, prepare, 32768) for row in short]
    capacity = [row['input_ids'][:32640].tolist() for row, _ in expanded]
    assert all(len(row) == 32640 for row in capacity)
    cases = dict(dense=[prefixes[i % 32] for i in range(256)],
        tail=tail,
        numeric_short=[prefixes[(0,10,20,31)[i % 4]] for i in range(8)],
        capacity_32768=[capacity[i % 4] for i in range(8)])
    value = dict(status='prepared_CPU_only_no_model_calls', cases=cases,
        rationale='Dense256 captured-prefix fixture and actual native terminal-action tail padded by original DataProto world8; exact same global requests for TP1 andTP2; all output lengths fixed128',
        tail_source=str(tail_path), tail_source_SHA256=hashlib.sha256(tail_path.read_bytes()).hexdigest(),
        tail_sources=tail_sources, tail_original_rows=len(tail_sources), native_padding=padding,
        prefix_sources=sources, capacity_source=str(captured),
        capacity_owner=str(owner), capacity_owner_SHA256=hashlib.sha256(owner.read_bytes()).hexdigest(),
        capacity_receipts=[x[1] for x in expanded],
        capacity_scope='Synthetic storage/forward capacity only; same accepted observation-repeat owner; not natural-task quality')
    (AUDIT/'matched-inputs.json').write_text(json.dumps(value, indent=2)+'\n')
    print(json.dumps(dict(status=value['status'], case_rows={k:len(v) for k,v in cases.items()},
        short_prefix_lengths=[len(x) for x in prefixes], capacity_prefix_length=32640, output_length=128)))


if __name__ == '__main__':
    main()

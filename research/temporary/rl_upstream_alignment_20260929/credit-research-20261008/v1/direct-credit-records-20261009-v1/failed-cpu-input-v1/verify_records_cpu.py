"""Check CPU persistence on an existing native B4; no numerical validation."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import psutil
import torch


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path(sys.argv[1])
    baseline = root / 'baseline-reward_readout.py'
    candidate = root / 'reward_readout.py'
    assert digest(baseline) == '814cfe929b4afcc5ce3570450ea8aca269b1097db7abed83dc917df3f1d1cc9b'
    assert digest(candidate) == '2609d93f91e9e16b8963ed5ca2226a00e28190005dbd51e56a214d3d35696eb2'
    tree = ast.parse(candidate.read_bytes())
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'DirectActionTargetReadout')
    helper = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == '_record_joint_dt_batch')
    namespace = dict(torch=torch, json=json, time=time, __file__=str(candidate))
    exec(compile(ast.Module(body=[helper], type_ignores=[]), str(candidate), 'exec'), namespace)
    record_batch = namespace[helper.name]
    saved_path = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/direct-target-prefix-runtime-20261007-v1/textcraft-first-dt/rank1-readout-native-batch-21.pt')
    assert digest(saved_path) == '19b18c5f4204ec4d88e40d72d350a06954f64452824a8662e95497319f89a37a'
    saved = torch.load(saved_path, map_location='cpu', weights_only=False)
    original_rows = sorted(saved['rows'], key=lambda row: row['batch_row'])
    assert len(original_rows) == 4
    batch = []
    for index, row in enumerate(original_rows):
        item = dict(row, index=row['trajectory_index'], reward=float(row['row']['dt_direct_reward']))
        # The existing captured signed tensor supplies the actual source data.
        # Use the owner's original scatter into original response slots.
        item['ratios'] = torch.zeros(row['width'], dtype=torch.float32)
        values = saved['native_signed'][index, row['prompt_length']:row['selected'].numel()]
        prior = row['prior'][row['suffix_positions']]
        item['ratios'][row['suffix_positions'][prior]] = values[prior].to(item['ratios'].dtype)
        batch.append(item)
    detail = dict(per_sample=[dict(factual_target_logp=None, reference_target_logp=None) for _ in batch])
    state = SimpleNamespace(task='TextCraft', tokenizer=SimpleNamespace(eos_token_id=saved['eos_token_id']))
    frozen = [{k: v.clone() for k, v in item.items() if isinstance(v, torch.Tensor)} for item in batch]
    record_batch(state, batch, detail)
    directory = root / 'cpu-records'
    directory.mkdir(exist_ok=False)
    state._diagnostic_directory = str(directory)
    before = time.perf_counter()
    record_batch(state, batch, detail)
    seconds = time.perf_counter() - before
    files = list(directory.glob('joint-*.pt'))
    assert len(files) == 1
    restored = torch.load(files[0], map_location='cpu', weights_only=True)
    for old, item, row in zip(frozen, batch, restored['rows']):
        for key, value in old.items():
            assert torch.equal(value, item[key]), key
        for field, origin in [('input_ids', 'selected'), ('suffix_positions', 'suffix_positions'),
                              ('policy_mask', 'policy'), ('target_mask', 'target'),
                              ('prior_source_mask', 'prior'), ('source_log_ratios', 'ratios')]:
            assert torch.equal(row[field], item[origin]), field
            assert row[field].dtype == item[origin].dtype and row[field].device.type == 'cpu'
        assert row['traj_uid'] == item['row']['traj_uid'] and row['reward'] == item['reward']
        assert row['target_offsets'] == list(item['target_offsets'])
    assert not torch.cuda.is_initialized()
    receipt = dict(status='passed_CPU_data_persistence_only', source=str(saved_path), source_sha256=digest(saved_path),
                   candidate_sha256=digest(candidate), script_sha256=digest(Path(__file__)),
                   bytes=files[0].stat().st_size, write_seconds=seconds,
                   pss_bytes=psutil.Process().memory_full_info().pss, cuda_initialized=False,
                   trajectories=4, tensor_values_and_dtypes_exact=True, input_tensors_unchanged=True,
                   model_calls=0, DT_calls=0, backward=0, optimizer=0,
                   scope='Existing real saved B4 tests serialization only; not a substitute for official numerical or attribution-quality tests.')
    (root / 'cpu-result.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()

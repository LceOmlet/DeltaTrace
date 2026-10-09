"""Read actual formal records on CPU using the unchanged credit composition owner."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import torch


FORMAL = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/runs/textcraft-formal-stable-20261009-v1')
sys.path.insert(0, str(FORMAL / 'entry'))
import counterfactual as owner


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--after-unix', type=float)
    parser.add_argument('--before-unix', type=float)
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('complete-second-DT-record-audit.json'))
    args = parser.parse_args()
    started = time.perf_counter()
    files = sorted((FORMAL / 'credit-records').glob('*/*.pt'))
    # The passive recorder names each artifact with its original time.time_ns().
    files = [path for path in files
             if (args.after_unix is None or int(path.stem.removeprefix('joint-')) / 1e9 > args.after_unix)
             and (args.before_unix is None or int(path.stem.removeprefix('joint-')) / 1e9 <= args.before_unix)]
    rows, identities = [], set()
    for path in files:
        stored = torch.load(path, map_location='cpu', weights_only=False)
        for row in stored['rows']:
            d = row['source_log_ratios']
            policy, target, source = (row[k] for k in ('policy_mask', 'target_mask', 'prior_source_mask'))
            credit = owner.reward_event_token_credit(
                d[None, None, :], torch.tensor([[row['reward']]], dtype=torch.float32),
                torch.ones((1, 1, d.numel()), dtype=torch.bool), policy[None, :],
                self_target_mask=target[None, None, :])
            assert all(x.device.type == 'cpu' for x in (d, policy, target, source))
            assert torch.isfinite(credit.advantages).all()
            assert not (target & ~policy).any()
            assert not credit.advantages[0, ~policy].any()
            slots = row['suffix_positions']
            assert row['input_ids'].numel() == row['prompt_length'] + slots.numel()
            indices = source.nonzero().flatten()
            index = int(indices[credit.advantages[0, indices].argmin()])
            offset = int((slots == index).nonzero().flatten().item())
            absolute = row['prompt_length'] + offset
            uid = row['traj_uid']
            identities.add(uid)
            rows.append(dict(path=str(path), traj_uid=uid,
                reward=row['reward'], input_length=row['input_ids'].numel(),
                minimum_prior_source=dict(response_position=index, input_position=absolute,
                    token_id=int(row['input_ids'][absolute]), d=float(d[index]),
                    raw_advantage=float(credit.advantages[0, index])),
                policy_tokens=int(policy.sum()), source_tokens=int(source.sum()),
                target_tokens=int(target.sum())))
    assert not torch.cuda.is_initialized()
    record = dict(unix=time.time(), seconds=time.perf_counter()-started,
        scope=__doc__, script_sha256=digest(Path(__file__)),
        owner_file=owner.__file__, owner_sha256=digest(Path(owner.__file__)),
        files=len(files), original_rank_rows=len(rows), unique_traj_uids=len(identities),
        bytes=sum(p.stat().st_size for p in files), rows=rows,
        model_DT_optimizer_calls=0, cuda_initialized=False,
        selected_record_time=dict(after_unix=args.after_unix, before_unix=args.before_unix),
        conclusion='CPU persistence/mask/position/finite credit only; no single-deletion accuracy or population recall claim')
    destination = args.output
    destination.write_text(json.dumps(record, indent=2)+'\n')
    summary = {k:v for k,v in record.items() if k != 'rows'}
    summary['minimum_observed_row'] = min(rows, key=lambda x:x['minimum_prior_source']['raw_advantage'])
    print(json.dumps(summary))

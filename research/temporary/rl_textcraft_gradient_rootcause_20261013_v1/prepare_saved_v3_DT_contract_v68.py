"""Prepare v3's real DT requests from the saved original bad-state batch.

This calls the deployed request boundary and official GRPO on CPU. The
inspection endpoint records requests and raises before any DT worker call;
it never fabricates DT output. A separate zero-multiplier call exercises the
same owner's exact retained-token checks without requiring model work.
"""
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT = ROOT / 'receipts/textcraft-gradient-rootcause-20261013-v1'
RUN = ROOT / 'runs/textcraft-dt-grpo-multiplier-variance-only-20261014-v3'
DEST = OUT / 'saved-bad-state-v3-DT-contract-v68'
launch = json.loads((RUN / 'launch.json').read_bytes())
env = launch['effective_environment']
sys.path[:0] = env['PYTHONPATH'].split(':')
os.environ.update(env)
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import torch
from verl import DataProto
from verl.trainer.ppo.ray_trainer import compute_advantage, AdvantageEstimator
from verl.utils import hf_tokenizer
import dt_training_batch

torch.set_num_threads(4)
bindings = {}
for obj, expected in ((dt_training_batch, 'a4e1f70b42082972324199b328975fb87ec6600ab9a572d5ca2b2040bebc54f7'),
                      (compute_advantage, 'b4b0187ef747d1ffb91dbfb8d1b8f914f93d01d90ff58d24e386f2cb1b7eb943')):
    path = Path(inspect.getsourcefile(obj))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == expected
    bindings[str(path)] = digest
captures = [json.loads((OUT / f'dt/rank{rank}-capture.json').read_bytes()) for rank in (0, 1)]
assert all(c['checkpoint_step'] == 80 and c['status'] == 'saved' for c in captures)
for c in captures:
    assert hashlib.sha256(Path(c['input']).read_bytes()).hexdigest() == c['input_sha256']
data = DataProto.concat([DataProto.load_from_disk(c['input']) for c in captures])
assert len(data) == 256 and 'dt_direct_target_artifact' in data.non_tensor_batch
original = {k: data.batch[k].clone() for k in ('token_level_rewards', 'old_log_probs', 'ref_log_prob', 'advantages')}
group_view = data.select(batch_keys=['responses', 'token_level_rewards', 'response_mask', 'loss_mask'],
                         non_tensor_batch_keys=['uid', 'traj_uid'], deepcopy=True)
group_view.batch = group_view.batch.clone(recurse=False)
group = compute_advantage(group_view, adv_estimator=AdvantageEstimator.GRPO,
                          norm_adv_by_std_in_grpo=True, multi_turn=True)
mask = data.batch['response_mask'].bool()
g = group.batch['advantages'].gather(1, mask.long().argmax(-1, keepdim=True)).squeeze(1)
assert torch.equal(group.batch['advantages'][mask], g[:, None].expand_as(mask)[mask])
candidate = data.select(deepcopy=True)
candidate.batch = data.batch.clone(recurse=False)
candidate.batch['dt_grpo_multipliers'] = g
for key in dt_training_batch.CREDIT_KEYS:
    if key in candidate.batch:
        del candidate.batch[key]
tokenizer = hf_tokenizer(env['MODEL_PATH'], trust_remote_code=True, local_files_only=True)
token_ids = dict(eos_token_id=int(tokenizer.eos_token_id), pad_token_id=int(tokenizer.pad_token_id))
DEST.mkdir(exist_ok=True)
assert not (DEST / 'preparation.json').exists(), 'Preserve the completed preparation.'


class RequestPreparationReached(Exception):
    pass


class InspectionEndpoint:
    world_size = 2

    def __init__(self):
        self.requests = None

    def compute_dt_token_advantages(self, requests):
        self.requests = requests
        raise RequestPreparationReached('Native DT request prepared; no worker/model call performed.')


zero = candidate.select(deepcopy=True)
zero.batch = candidate.batch.clone(recurse=False)
zero.batch['dt_grpo_multipliers'] = torch.zeros_like(g)
zero_endpoint = InspectionEndpoint()
zero_result = dt_training_batch.compute_direct_target_credit(zero, zero_endpoint, **token_ids)
assert zero_endpoint.requests is None
assert all(not bool(zero_result.batch[k].any()) for k in dt_training_batch.CREDIT_KEYS)
endpoint = InspectionEndpoint()
try:
    dt_training_batch.compute_direct_target_credit(candidate, endpoint, **token_ids)
except RequestPreparationReached:
    pass
else:
    raise AssertionError('Expected a real nonzero native DT request; no synthetic result is accepted.')
requests = endpoint.requests
assert requests is not None
assert all(torch.equal(data.batch[k], value) for k, value in original.items())
path = DEST / 'original81-for-native-v3-DT.pkl'
candidate.save_to_disk(str(path))
request_path = DEST / 'prepared-first-real-native-DT-request.pkl'
requests.save_to_disk(str(request_path))
rewards = data.batch['token_level_rewards'].sum(-1)
report = dict(unix=time.time(), scope=__doc__, status='prepared_only_no_DT_results_no_GPU_submission',
              captured_completed_step=80, rows=len(data), source_bindings=bindings,
              original_capture_receipts=captures, required_artifacts_available=True,
              all_original_retained_token_identity_checks_passed=True,
              original_rewards_oldlogp_reference_and_advantages_unchanged=True,
              native_zero_group_rows=int(g.eq(0).sum()), positive_reward_zero_group_rows=int((g.eq(0)&rewards.gt(0)).sum()),
              zero_reward_nonzero_group_rows=int((g.ne(0)&rewards.eq(0)).sum()),
              prepared_native_request_rows=len(requests), request_fields=list(requests.batch.keys()),
              token_ids=token_ids, payload=dict(path=str(path), SHA256=hashlib.sha256(path.read_bytes()).hexdigest()),
              request_payload=dict(path=str(request_path), SHA256=hashlib.sha256(request_path.read_bytes()).hexdigest()),
              model_calls=0, DT_worker_calls=0, backward_calls=0, optimizer_calls=0,
              new_generation_calls=0, GRPO_model_reruns=0, rootcause_repair_verified=False,
              limits=['This proves real saved inputs can be sent through the deployed v3 request boundary, not that their new DT values have been computed.',
                      'A bad-state native update regression cannot establish the entire from-scratch counterfactual trajectory.',
                      'The inspection endpoint stops before DT and never returns substitute credit.'])
assert not torch.cuda.is_initialized()
path = DEST / 'preparation.json'
path.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(dict(receipt=str(path), report=report)))

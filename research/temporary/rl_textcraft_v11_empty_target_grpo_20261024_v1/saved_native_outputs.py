"""Reuse saved original native DataProto outputs, never a model replacement."""
from pathlib import Path
import hashlib
import torch
from verl import DataProto
ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
PAIR=ROOT/'runs/textcraft-dtpo-v5-original-2gpu-formal-20261021-v1/paired-inputs/checkpoint5-following-native-iteration6'
class BASE:
    CREDIT_KEYS=('dt_token_advantages','dt_q_estimates','dt_v_estimates')

def digest(p):
    return dict(path=str(p), SHA256=hashlib.sha256(p.read_bytes()).hexdigest())

class SavedNativeOutputs:
    """Replay original returned tensors by native UID; no DT calculation."""
    world_size = 2

    def __init__(self):
        self.by_uid, self.sources = {}, []
        self.duplicate_copies = 0
        self.calls = 0
        for folder in ['rank0-pid1021020', 'rank1-pid1023296']:
            p = PAIR/folder/'001-compute_dt_token_advantages-input.pkl'
            o = p.with_name('001-compute_dt_token_advantages-output.pkl')
            inputs, outputs = DataProto.load_from_disk(str(p)), DataProto.load_from_disk(str(o))
            assert len(inputs) == len(outputs)
            self.sources.extend([digest(p), digest(o)])
            for i, uid in enumerate(inputs.non_tensor_batch['traj_uid']):
                uid = str(uid)
                if uid in self.by_uid:
                    self.duplicate_copies += 1
                    continue
                self.by_uid[uid] = (inputs.select_idxs([i]), outputs.select_idxs([i]))

    def compute_dt_token_advantages(self, requests):
        self.calls += 1
        width = requests.batch['responses'].shape[-1]
        result = {k:torch.zeros(len(requests), width) for k in BASE.CREDIT_KEYS}
        for i, uid in enumerate(requests.non_tensor_batch['traj_uid']):
            before, values = self.by_uid[str(uid)]
            valid = int(before.batch['attention_mask'][0, -before.batch['responses'].shape[-1]:].sum())
            assert torch.equal(before.batch['responses'][0, :valid].long(),
                               requests.batch['responses'][i, :valid].long())
            for name in ['policy_mask', 'target_mask', 'dt_future_event_mask']:
                assert torch.equal(before.batch[name][0, :valid], requests.batch[name][i, :valid])
            for key in BASE.CREDIT_KEYS:
                result[key][i, :valid] = values.batch[key][0, :valid]
        return DataProto.from_dict(tensors=result)

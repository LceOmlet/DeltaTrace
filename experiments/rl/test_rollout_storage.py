"""Exercise pinned owner row conversion through Ray's actual serializer."""
import numpy as np
import pytest
import torch
from tensordict import TensorDict
from ray import cloudpickle
from ray._private.serialization import SerializationContext
from verl import DataProto
from agent_system.multi_turn_rollout.utils import to_list_of_dict

from patch_verl_agent2 import patch_rollout_row_storage


@pytest.mark.parametrize("batch_size", [4, 16])
def test_dt_rows_do_not_pickle_the_full_batch_storage_per_row(batch_size):
    ids = torch.arange(batch_size * 32768).reshape(batch_size, 32768)
    batch = DataProto(
        batch=TensorDict({"input_ids": ids, "responses": ids[:, -1024:]}, batch_size=[batch_size]),
        non_tensor_batch={"traj_uid": np.array([f"traj-{i}" for i in range(batch_size)], dtype=object)},
    )
    default = to_list_of_dict(batch)
    compact = to_list_of_dict(batch, clone_tensors=True)
    # The default owner still returns views. DT rows own just their elements.
    assert default[0]["input_ids"].untyped_storage().data_ptr() == ids.untyped_storage().data_ptr()
    for original, row in zip(default, compact):
        assert row["traj_uid"] == original["traj_uid"]
        for key in ("input_ids", "responses"):
            torch.testing.assert_close(row[key], original[key], rtol=0, atol=0)
            assert row[key].untyped_storage().nbytes() == row[key].numel() * row[key].element_size()

    # This is Ray's installed serialization code, not a replacement serializer.
    context = SerializationContext(None)
    old_size = context.serialize(default).total_bytes
    new_size = context.serialize(compact).total_bytes
    logical_size = sum(t.numel() * t.element_size() for row in compact
                       for t in row.values() if isinstance(t, torch.Tensor))
    assert logical_size <= new_size < logical_size + 65536
    assert old_size > new_size * batch_size
    restored = cloudpickle.loads(cloudpickle.dumps(compact))
    for original, row in zip(default, restored):
        assert row["traj_uid"] == original["traj_uid"]
        for key in ("input_ids", "responses"):
            torch.testing.assert_close(row[key], original[key], rtol=0, atol=0)


def test_row_storage_patch_is_idempotent_and_keeps_default():
    source = ("def to_list_of_dict(batch: DataProto) -> list[dict]:\n"
              "        for key, val in tensors.items():\n"
              "            save_dict[key] = val[bs]\n"
              "        for key, val in non_tensor.items():\n"
              "            save_dict[key] = val[bs]\n")
    patched = patch_rollout_row_storage(source)
    assert patch_rollout_row_storage(patched) == patched
    assert "clone_tensors: bool = False" in patched
    assert patched.count(".clone()") == 1

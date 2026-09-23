"""Pinned owner's request/row contract; engine doubles do not measure model speed.

Compile the patched owner class from its installed source without modifying the
shared checkout used by running jobs. The unpatched class is the control.
"""
import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from vllm import SamplingParams
from verl import DataProto
from verl.workers.rollout.vllm_rollout import vllm_rollout_spmd as owner

from patch_verl_agent2 import patch_vllm_active_rows


def patched_owner_class():
    source = Path(owner.__file__).read_text()
    patched = patch_vllm_active_rows(source)
    assert patch_vllm_active_rows(patched) == patched
    tree = ast.parse(patched)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'vLLMRollout')
    namespace = dict(vars(owner))
    exec(compile(ast.Module(body=[cls], type_ignores=[]), owner.__file__, 'exec'), namespace)
    return namespace['vLLMRollout']


class RecordingEngine:
    """Deterministic transport fixture, explicitly not a substitute for vLLM."""
    def __init__(self):
        self.calls = []
        self.llm_engine = SimpleNamespace(list_loras=lambda: [7])

    def generate(self, *, prompts, sampling_params, lora_request, use_tqdm):
        self.calls.append((deepcopy(prompts), lora_request))
        results = []
        for prompt in prompts:
            row_id = prompt['prompt_token_ids'][-1]
            samples = []
            for sample in range(sampling_params.n):
                ids = [row_id + sample + 30, 7]
                samples.append(SimpleNamespace(token_ids=ids, logprobs=[
                    {token: SimpleNamespace(logprob=-float(token) / 100)} for token in ids]))
            results.append(SimpleNamespace(outputs=samples))
        return results


def make_rollout(cls, n=1, lora=False):
    rollout = object.__new__(cls)
    rollout.config = OmegaConf.create(dict(response_length=1024, free_cache_engine=False,
                                         val_kwargs=dict(top_k=-1, top_p=1., temperature=0.)))
    rollout.pad_token_id = 0
    rollout.sampling_params = SamplingParams(n=n, max_tokens=1024, logprobs=0)
    rollout.lora_kwargs = {'enable_lora': True} if lora else {}
    rollout.inference_engine = RecordingEngine()
    return rollout


def make_prompts(active=None, multimodal=False, mode='sample'):
    # Exact 32k output layout; values only test transport, not model capacity.
    ids = torch.zeros((4, 31744), dtype=torch.long)
    ids[:, -3:] = torch.arange(12).reshape(4, 3) + 1
    mask = ids.ne(0).long()
    positions = (mask.cumsum(-1) - 1).clamp_min(0)
    nt = {}
    if active is not None:
        nt['rollout_active_mask'] = np.array(active, dtype=bool)
    if multimodal:
        nt['multi_modal_data'] = np.array([{'image': row} for row in range(4)], dtype=object)
    return DataProto(
        batch=TensorDict(dict(input_ids=ids, attention_mask=mask, position_ids=positions), batch_size=[4]),
        non_tensor_batch=nt,
        meta_info=dict(eos_token_id=7, do_sample=mode != 'greedy', validate=mode == 'validate'))


@pytest.mark.parametrize('active', [None, [True]*4, [False, True, False, True], [False]*4])
@pytest.mark.parametrize('n,mode', [(1, 'sample'), (2, 'sample'), (2, 'greedy'), (2, 'validate')])
@pytest.mark.parametrize('lora,multimodal', [(False, False), (True, False), (True, True)])
def test_original_outputs_for_active_rows_and_no_requests_for_finished_rows(active, n, mode, lora, multimodal):
    baseline = make_rollout(owner.vLLMRollout, n=n, lora=lora)
    patched = make_rollout(patched_owner_class(), n=n, lora=lora)
    expected = baseline.generate_sequences(make_prompts(multimodal=multimodal, mode=mode))
    actual = patched.generate_sequences(make_prompts(active, multimodal, mode))
    live = np.ones(4, dtype=bool) if active is None else np.array(active)
    calls = patched.inference_engine.calls
    assert len(calls) == int(live.any())
    if calls:
        inputs, loras = calls[0]
        original_inputs = baseline.inference_engine.calls[0][0]
        assert inputs == [original_inputs[i] for i in np.flatnonzero(live)]
        assert len(loras) == live.sum() if lora else loras is None
    copies = n if mode == 'sample' else 1
    select = torch.from_numpy(live).repeat_interleave(copies)
    assert actual.batch['input_ids'].shape == (4 * copies, 32768)
    for key in expected.batch.keys():
        torch.testing.assert_close(actual.batch[key][select], expected.batch[key][select], rtol=0, atol=0)
    torch.testing.assert_close(actual.batch['prompts'], expected.batch['prompts'], rtol=0, atol=0)
    assert not actual.batch['responses'][~select].any()
    assert not actual.batch['attention_mask'][~select, 31744:].any()
    assert 'rollout_active_mask' not in actual.non_tensor_batch
    assert patched.sampling_params.n == n  # original sampling context restored

"""Scripted native LOOP episode through the actual VERL environment boundary.

No model, training, fabricated reward or replacement parser is used. The policy
reply is explicitly a transport fixture; task success is not a quality metric.
"""
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from verl import DataProto
from vllm import SamplingParams
from owner_environment_transport import owner_sampling_params

from loop_environment_entry import make_loop_environments, EpisodeResult, CompletionRequest
from phi_agents.appworld.interface import load_task_ids

root=Path(os.environ['DT_LOOP_ENTRY_OUTPUT'])
root.mkdir(exist_ok=True)
config=OmegaConf.create(dict(env=dict(loop=dict(owner_root=os.environ['LOOP_ROOT'])),
    actor_rollout_ref=dict(model=dict(path=os.environ['MODEL_PATH']))))
tokenizer=AutoTokenizer.from_pretrained(os.environ['MODEL_PATH'],local_files_only=True)
managers=make_loop_environments(config,tokenizer)
records=[]
episode_count=int(os.environ.get('DT_LOOP_ENTRY_EPISODES','1'))
try:
    for phase, manager, split in zip(['training','evaluation'],managers,
        ['train_difficulty_1_and_train_difficulty_2','dev']):
        started=time.monotonic()
        task=load_task_ids(split)[0]
        obs,_=manager.reset([{'task_id':task} for _ in range(episode_count)])
        text='End this scripted interface check.\n</think>\n```python\napis.supervisor.complete_task()\n```'
        ids=tokenizer.encode(text,add_special_tokens=False)+[tokenizer.eos_token_id]
        data=DataProto.from_dict(tensors=dict(responses=torch.tensor([ids]*episode_count),
            rollout_log_probs=torch.full((episode_count,len(ids)),-.25)),non_tensors=dict(
            owner_response_length=np.array([len(ids)]*episode_count),owner_response_text=np.array([text]*episode_count,dtype=object),
            owner_finish_reason=np.array(['stop']*episode_count,dtype=object)))
        observed=obs['raw_prompt_ids'][0].copy()
        for step in range(40):
            active=[isinstance(e,CompletionRequest) for e in manager.events]
            # Include the actual normalized owner-to-vLLM parameter boundary;
            # a scripted reply alone would miss invalid native sampling inputs.
            owner_sampling_params(SamplingParams(logprobs=0), obs['sampling_kwargs'],
                                  [i for i, live in enumerate(active) if live])
            obs,reward,done,infos=manager.step_policy_outputs(data,active)
            if done.all():break
        assert done.all(), 'Scripted completion did not reach a native terminal event'
        event=manager.events[0]
        assert isinstance(event,EpisodeResult)
        native=event.rollout
        evaluation=native.appworld_rollout_data.eval_result
        assert reward[0]==np.float32(native.ret)
        rows=[[{'env_step':0}] for _ in range(episode_count)]
        manager.finalize_trajectory_metadata(rows)
        assert rows[0][0]['appworld_num_tests']==evaluation.num_tests
        expected=len(evaluation.passes)/evaluation.num_tests if phase=='training' else float(evaluation.success)
        assert native.ret==expected
        for index,event in enumerate(manager.events):
            assert isinstance(event,EpisodeResult)
            ev=event.rollout.appworld_rollout_data.eval_result
            expected_i=len(ev.passes)/ev.num_tests if phase=='training' else float(ev.success)
            assert event.rollout.ret==expected_i
            assert rows[index][0]['appworld_num_tests']==ev.num_tests
        records.append(dict(phase=phase,task_id=task,steps=step+1,first_prompt_tokens=len(observed),
            native_return=native.ret,num_tests=evaluation.num_tests,episodes=episode_count,
            seconds=time.monotonic()-started))
        print(json.dumps(records[-1]),flush=True)
finally:
    for manager in managers: manager.close()
    (root/'loop-entry.json').write_text(json.dumps(records,indent=2))

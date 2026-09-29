"""Real official TextCraft service, client, renderer and VERL manager boundary."""
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from verl import DataProto
from textcraft_environment_entry import make_textcraft_environments

entry=Path(__file__).resolve().parent
source=json.loads((entry/'owner_environment_configs.json').read_text())['TextCraft']
native=source['train']['actor_rollout_ref']
sampling={k:native['rollout'][k] for k in ['temperature','top_k','top_p','max_tokens','ignore_eos']}
config=OmegaConf.create(dict(env=dict(textcraft=dict(owner_root=os.environ['AGENTGYM_RL_ROOT'],
    client=native['agentgym'],eval_client=source['eval']['agentgym'],sampling=sampling,eval_sampling=sampling))))
tokenizer=AutoTokenizer.from_pretrained(os.environ['MODEL_PATH'],local_files_only=True)
managers=make_textcraft_environments(config,tokenizer)
records=[]
try:
    for phase,manager,relative in zip(['train','eval'],managers,['train/textcraft_train.json','eval/textcraft_test.json']):
        started=time.monotonic()
        items=json.loads((Path(os.environ['TEXTCRAFT_DATA'])/relative).read_text())
        chosen=items[0]['item_id']
        obs,_=manager.reset([{'item_id':int(chosen.split('_')[-1])}])
        ids=tokenizer.encode('Thought: inspect.\nAction: inventory',add_special_tokens=False)+[tokenizer.eos_token_id]
        batch=DataProto.from_dict(tensors=dict(responses=torch.tensor([ids]),rollout_log_probs=torch.full((1,len(ids)),-.25)),non_tensors=dict(
            owner_response_length=np.array([len(ids)]),owner_response_text=np.array(['unused'],dtype=object),owner_finish_reason=np.array(['stop'],dtype=object)))
        before=obs['raw_prompt_ids'][0]
        obs,rewards,dones,infos=manager.step_policy_outputs(batch,[True])
        assert rewards.tolist()==[0.] and dones.tolist()==[False]
        assert 'Inventory:' in manager.sessions[0].history.messages[-1].content
        assert len(obs['raw_prompt_ids'][0])>len(before)
        assert obs['sampling_kwargs'][0]==sampling
        records.append(dict(phase=phase,item=chosen,split_size=len(items),initial_tokens=len(before),
            next_tokens=len(obs['raw_prompt_ids'][0]),reward=float(rewards[0]),seconds=time.monotonic()-started))
        print(json.dumps(records[-1]),flush=True)
finally:
    for manager in managers:manager.close()
    Path(os.environ['TEXTCRAFT_ENTRY_RECEIPT']).write_text(json.dumps(records,indent=2))

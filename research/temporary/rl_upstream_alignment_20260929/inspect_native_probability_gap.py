"""Diagnose exact native generation/recompute artifacts; no new tolerance or loss.

Uses saved AppWorld token IDs (not decoded text), original VERL distributed
workers and their vLLM/actor calls. No task simulation, update, or HF reference.
"""
import json
import os
from pathlib import Path
import re
import time

import ray
import torch
from verl import DataProto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.utils.model import compute_position_id_with_mask
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verify_author_rollout import configuration

root = Path(os.environ['DT_RUNTIME_ROOT'])/'receipts/upstream-alignment-20260929'
source = next(Path('/tmp/ray/session_2026-09-29_19-54-18_425280_1376898/logs').glob('worker-*-1414025.out'))
prompts = []
with source.open() as stream:
    for line in stream:
        if '[DT EOS minimum] ' in line:
            item = json.loads(re.sub(r'\x1b\[[0-9;]*m', '', line).split('[DT EOS minimum] ', 1)[1])
            for sample in item['samples']:
                prompts.append(sample['selected_input_ids'][:sample['source_start']])
            if len(prompts) >= 8:
                break
assert len(prompts) >= 8
prompts = prompts[:8]
output = root/'native-probability-gap.json'
result = dict(scope=__doc__, source=str(source), prompt_lengths=list(map(len,prompts)),
              status='running', cases=[], context_cap=32768, response_cap=512)
started = time.perf_counter()
def record(phase):
    result.update(phase=phase, elapsed=time.perf_counter()-started)
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(phase=phase,elapsed=result['elapsed'])),flush=True)

ray.init(num_cpus=8,include_dashboard=False)
try:
    config = configuration().actor_rollout_ref
    config.actor.ppo_mini_batch_size = 64
    group = RayWorkerGroup(RayResourcePool([2],use_gpu=True,max_colocate_count=1),
        RayClassWithInitArgs(ray.remote(ActorRolloutRefWorker),config,'actor_rollout'))
    record('initializing_original_worker')
    group.init_model()
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(os.environ['MODEL_PATH'],local_files_only=True)
    width=max(map(len,prompts))
    ids=torch.full((len(prompts),width),tokenizer.pad_token_id,dtype=torch.long)
    mask=torch.zeros_like(ids)
    for row,prompt in enumerate(prompts):
        ids[row,-len(prompt):]=torch.tensor(prompt)
        mask[row,-len(prompt):]=1
    data=DataProto.from_dict(tensors=dict(input_ids=ids,attention_mask=mask,
        position_ids=compute_position_id_with_mask(mask)),meta_info={'do_sample':True})
    record('native_generate')
    generated=group.generate_sequences(data)
    record('native_recompute')
    old=group.compute_log_prob(generated)
    saved={key:value.detach().cpu().clone() for key,value in generated.batch.items()}
    saved['old_log_probs']=old.batch['old_log_probs'].detach().cpu().clone()
    torch.save(saved,root/'native-probability-gap.pt')
    length=saved['responses'].shape[-1]
    valid=saved['attention_mask'][:,-length:].bool()
    rollout=saved['rollout_log_probs']
    actor=saved['old_log_probs']
    for row in range(len(prompts)):
        live=valid[row]
        lp_a,lp_r=actor[row][live],rollout[row][live]
        diff=(lp_a.exp()-lp_r.exp()).abs()
        worst=torch.argsort(diff,descending=True)[:8]
        result['cases'].append(dict(row=row,prompt_tokens=len(prompts[row]),
            response_tokens=int(live.sum()),nonfinite=int((~torch.isfinite(lp_r)).sum()),
            mean_probability_diff=float(diff.mean()),max_probability_diff=float(diff.max()),
            worst=[dict(offset=int(i),token=int(saved['responses'][row][live][i]),
                        actor_logprob=float(lp_a[i]),rollout_logprob=float(lp_r[i])) for i in worst]))
    result['status']='diagnostic_complete_no_new_acceptance_threshold'
    record('finished')
finally:
    ray.shutdown()

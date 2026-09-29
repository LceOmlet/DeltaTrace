"""Read original vLLM probabilities before/after loading a real two-rank checkpoint."""
import json
import math
import os
from pathlib import Path
import re
import time
import numpy as np
import ray
import torch
from verl import DataProto
from verl.single_controller.base.decorator import Dispatch, register
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.utils.model import compute_position_id_with_mask
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verify_author_rollout import configuration

root = Path(os.environ['DT_RUNTIME_ROOT'])/'receipts/upstream-alignment-20260929'
update_mode = os.environ.get('DT_NAN_PROBE_UPDATE') == '1'
stem = 'rollout-nan-update' if update_mode else 'rollout-nan-checkpoint'
output = root/(stem+'-probe.json')
sources = list(Path('/tmp/ray/session_2026-09-29_16-46-10_139414_71777/logs').glob('worker-*-92218.out'))
assert len(sources) == 1
prompts = []
plans = 0
with sources[0].open() as stream:
    for raw in stream:
        if '[DT EOS plan]' in raw:
            plans += 1
        if plans >= 2 and '[DT EOS minimum] ' in raw:
            item = json.loads(re.sub(r'\x1b\[[0-9;]*m','',raw).split('[DT EOS minimum] ',1)[1])
            for sample in item['samples']:
                prompts.append(sample['selected_input_ids'][:sample['source_start']])
            if len(prompts) >= 8:
                break
assert len(prompts) >= 8
prompts = prompts[:8]

@ray.remote
class ProbeWorker(ActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def observe_engine(self):
        self.engine_observations = []
        original = self.rollout.inference_engine.generate
        def observe(*args, **kwargs):
            started = time.perf_counter()
            outputs = original(*args, **kwargs)
            bad, count = [], 0
            for i, request in enumerate(outputs):
                for j, response in enumerate(request.outputs):
                    for t,(token,values) in enumerate(zip(response.token_ids,response.logprobs)):
                        value = values[token].logprob
                        count += 1
                        if not math.isfinite(value):
                            bad.append(dict(request=i,sample=j,position=t,token=token,value=str(value)))
            record = dict(rank=self.rank,seconds=time.perf_counter()-started,tokens=count,
                          nonfinite_count=len(bad),examples=bad[:20])
            self.engine_observations.append(record)
            (root/f'{stem}-native-rank{self.rank}.json').write_text(json.dumps(self.engine_observations,indent=2)+'\n')
            print('NATIVE_VLLM_LOGPROBS '+json.dumps(record),flush=True)
            return outputs
        self.rollout.inference_engine.generate = observe

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def observations(self):
        return self.engine_observations

result = dict(scope=__doc__,prompt_source=str(sources[0]),prompt_lengths=list(map(len,prompts)),
              context_cap=32768,response_cap=512,cases=[],status='running')
if update_mode:
    result['scope'] = 'Original native update on saved real token/DT-credit fixture, followed by native generation. Numerical lifecycle diagnostic, not new on-policy training or task performance.'
started = time.perf_counter()
def record(phase):
    result.update(phase=phase,elapsed=time.perf_counter()-started)
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(phase=phase,elapsed=result['elapsed'])),flush=True)

ray.init(num_cpus=8,include_dashboard=False)
try:
    cfg = configuration().actor_rollout_ref
    cfg.actor.ppo_mini_batch_size = 64
    cfg.rollout.name = 'vllm'
    group = RayWorkerGroup(RayResourcePool([2],use_gpu=True,max_colocate_count=1),
        RayClassWithInitArgs(ProbeWorker,cfg,'actor_rollout'))
    group.init_model()
    group.observe_engine()
    record('model_ready')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(os.environ['MODEL_PATH'],local_files_only=True)
    width = max(map(len,prompts))
    ids = torch.full((len(prompts),width),tokenizer.pad_token_id,dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i,prompt in enumerate(prompts):
        ids[i,-len(prompt):] = torch.tensor(prompt)
        mask[i,-len(prompt):] = 1
    phases = ('base','after_update','after_update_repeat') if update_mode else ('base','checkpoint1','checkpoint1_repeat')
    for phase in phases:
        if phase == 'after_update':
            fixture_path = root/'isolated-vs-owner.pt'
            artifact = torch.load(fixture_path,map_location='cpu',weights_only=False)
            tensors = {key: artifact[key].clone() for key in
                       ('input_ids','attention_mask','position_ids','responses','response_mask')}
            tensors['advantages'] = artifact['dt_token_advantages'].clone()
            fixture = DataProto.from_dict(tensors=tensors,meta_info={'temperature':1.0})
            assert 64 % len(fixture) == 0
            fixture = fixture.repeat(repeat_times=64//len(fixture),interleave=True)
            fixture.meta_info['global_token_num'] = fixture.batch['attention_mask'].sum(-1).tolist()
            record('saved_fixture:native_old_logprob')
            fixture.batch['old_log_probs'] = group.compute_log_prob(fixture).batch['old_log_probs']
            record('saved_fixture:native_update')
            result['update'] = dict(fixture=str(fixture_path),rows=len(fixture),
                                   metrics=group.update_actor(fixture).meta_info['metrics'])
            record('saved_fixture:update_complete')
        if phase == 'checkpoint1':
            checkpoint = root/'two-gpu-pilots/dt-Sokoban/checkpoints/global_step_1/actor'
            record('load_real_checkpoint1')
            group.load_checkpoint(str(checkpoint),del_local_after_load=False)
            result['checkpoint'] = str(checkpoint)
        data = DataProto.from_dict(tensors=dict(input_ids=ids.clone(),attention_mask=mask.clone(),
            position_ids=compute_position_id_with_mask(mask)),meta_info={'do_sample':True})
        record(phase+':generate')
        generated = group.generate_sequences(data)
        response_width = generated.batch['responses'].shape[1]
        valid = generated.batch['attention_mask'][:,-response_width:].bool()
        lp = generated.batch['rollout_log_probs'][valid]
        record(phase+':native_old_logprob')
        old = group.compute_log_prob(generated)
        oldlp = old.batch['old_log_probs'][valid]
        result['cases'].append(dict(phase=phase,valid_tokens=int(valid.sum()),
            rollout_nan=int(lp.isnan().sum()),rollout_inf=int(lp.isinf().sum()),
            actor_nan=int(oldlp.isnan().sum()),actor_inf=int(oldlp.isinf().sum()),
            native_observations=group.observations()))
        torch.save(generated,root/f'{stem}-{phase}.pt')
        record(phase+':complete')
    result['status']='completed_diagnostic'
finally:
    record('finished')
    ray.shutdown()

"""Read native single-source EOS endpoints for21 already disputed DT values.

Original VERL initializes/restores the actor. Original DeltaTrace's native
model and target readers evaluate the endpoints. No attribution, policy loss,
parameter backward, optimizer update, rollout or validation is performed.
"""
from contextlib import nullcontext
import hashlib
import inspect
import json
import os
from pathlib import Path
import time

import psutil
import ray
import torch
from omegaconf import OmegaConf
from verl.single_controller.base.decorator import Dispatch, register
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from observe_native_gradient_chain_v4 import digest

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT=ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1'
DEST=OUT/'native-source-delete-values-v14'
RUN=ROOT/'runs/textcraft-dt-formal-native-graph-fresh-20261012-v1'
ACTOR_SHA='3a65e173300be82a7a9e056a96227c4f746eabc3be778ef41c8d50138d52ce6c'
state=dict(scope=__doc__,started_unix=time.time(),backward=0,optimizer=0,scheduler=0,
           DT_calls=0,rollout=0,validation=0,formal_training_changed=False)

def phase(name,**values):
    state.update(phase=name,unix=time.time(),**values)
    (DEST/'result.json').write_text(json.dumps(state,indent=2)+'\n')
    print(json.dumps(dict(phase=name,unix=state['unix'],**values)),flush=True)

def identity(function):
    path=Path(inspect.getsourcefile(function))
    return dict(path=str(path),SHA256=hashlib.sha256(path.read_bytes()).hexdigest())

@ray.remote
class EndpointWorker(ActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def read_sources(self):
        from deltatrace_rollout import DeltaTraceRolloutProducer
        from native_target_logit_rows import NativeTargetLogitRows
        from qwen35_answer_finite import PackedAnswerTargets,selected_target_log_probs
        from verl.utils.fsdp_utils import load_fsdp_model_to_gpu,offload_fsdp_model_to_cpu
        from verl.utils.torch_functional import pad_2d_list_to_length
        manifest=json.loads((DEST/'query-manifest.json').read_bytes())
        assert hashlib.sha256(Path(manifest['query_file']).read_bytes()).hexdigest()==manifest['query_SHA256']
        queries=torch.load(manifest['query_file'],map_location='cpu',weights_only=False)
        assert len(queries)==21 and self.actor_lr_scheduler.last_epoch==80
        assert self._is_actor and not self._is_rollout and not self._is_ref
        assert (self.config.model.lora_rank,self.config.model.lora_alpha)==(8,16)
        assert self.actor.config.ppo_micro_batch_size_per_gpu==4
        assert identity(type(self.actor))['SHA256']==ACTOR_SHA
        before=digest(self.actor);training=self.actor_module_fsdp.training
        record=dict(rank=self.rank,pid=os.getpid(),birth=psutil.Process().create_time(),
            checkpoint=80,scheduler_before=80,trainable_SHA256_before=before,
            owners={k:identity(v) for k,v in dict(actor=type(self.actor),producer=DeltaTraceRolloutProducer,
                targets=PackedAnswerTargets,logit_rows=NativeTargetLogitRows,log_probs=selected_target_log_probs).items()},
            batches=[],native_paired_forward_calls=0,backward=0,optimizer=0,DT_calls=0)
        path=DEST/f'rank{self.rank}.json'
        def save(name,**values):
            record.update(phase=name,unix=time.time(),**values)
            path.write_text(json.dumps(record,indent=2)+'\n')
        producer=text=attention=None
        try:
            save('restored_original_actor')
            if self._is_offload_param:load_fsdp_model_to_gpu(self.actor_module_fsdp)
            self.actor_module_fsdp.eval()
            producer=DeltaTraceRolloutProducer(self.actor_module_fsdp,
                eos_token_id=self.tokenizer.eos_token_id,pad_token_id=self.tokenizer.pad_token_id)
            runner=producer.runner;text=runner.model.model.language_model
            attention=text.config._attn_implementation;text.set_attn_implementation('flash_attention_2')
            precision=nullcontext()
            if producer.native_fla_fp16:
                from accelerated.qwen35.native_fla_precision import native_fla_fp16
                precision=native_fla_fp16(self.actor_module_fsdp)
            conv=runner.attribute.__func__.__globals__['_native_conv_initial_states_scope']
            with torch.no_grad(),precision,conv(text.layers,runner.native_conv_initial_states):
                for start in range(0,len(queries),4):
                    rows=queries[start:start+4];valid=len(rows)
                    rows=rows+[rows[-1]]*(4-valid)
                    width=max(len(row['selected']) for row in rows)
                    factual=pad_2d_list_to_length([r['selected'].tolist() for r in rows],
                        self.tokenizer.eos_token_id,max_length=width)
                    targets=PackedAnswerTargets([r['case'] for r in rows],
                        [r['target_offsets'] for r in rows],width,'cuda')
                    selector=NativeTargetLogitRows(targets)
                    batch=dict(start=start,valid_queries=valid,pair_shape=[8,width],passes={})
                    for name in ('identity','single_EOS'):
                        reference=factual.clone()
                        if name=='single_EOS':
                            for slot,row in enumerate(rows[:valid]):
                                assert int(reference[slot,row['packed_slot']])==row['token_id']
                                reference[slot,row['packed_slot']]=self.tokenizer.eos_token_id
                        pair=torch.stack((reference,factual),dim=1).flatten(0,1).cuda()
                        tick=time.perf_counter();save('native_forward',batch_start=start,active_pass=name)
                        output=runner.model.forward_root(input_ids=pair,attention_mask=torch.ones_like(pair),
                            use_cache=False,logits_to_keep=selector.rows)
                        dtype=str(output.logits.dtype);packed=selector.pack_logits(output.logits);del output
                        logp=selected_target_log_probs(packed,targets);del packed
                        left=targets.sample_sums(logp[0::2].double()).cpu()
                        right=targets.sample_sums(logp[1::2].double()).cpu()
                        torch.cuda.synchronize();record['native_paired_forward_calls']+=1
                        batch['passes'][name]=dict(reference_joint_logp=left.tolist(),factual_joint_logp=right.tolist(),
                            native_d=(right-left).tolist(),native_logits_dtype=dtype,seconds=time.perf_counter()-tick)
                        del pair,reference,logp
                    batch['queries']=[{k:v for k,v in q.items() if k not in ('selected','case','target_offsets')}
                        for q in rows[:valid]]
                    record['batches'].append(batch);save('batch_complete',batch_start=start)
        except BaseException as exc:
            save('failed',error=repr(exc));raise
        finally:
            if text is not None and attention is not None:text.set_attn_implementation(attention)
            self.actor_module_fsdp.train(training)
            if producer is not None:producer.runner.model.release_owner_params()
            if self._is_offload_param:offload_fsdp_model_to_cpu(self.actor_module_fsdp)
        after=digest(self.actor)
        assert after==before and self.actor_lr_scheduler.last_epoch==80
        save('complete_native_source_values_not_repair',trainable_SHA256_after=after,scheduler_after=80)
        return dict(rank=self.rank,path=str(path),SHA256=hashlib.sha256(path.read_bytes()).hexdigest())

def main():
    assert not (DEST/'result.json').exists(),'Do not repeat an existing diagnostic.'
    launch=json.loads((RUN/'launch.json').read_bytes())
    cfg=OmegaConf.load(Path(os.environ['VERL_ROOT'])/'verl/trainer/config/ppo_trainer.yaml')
    for k,v in launch['options'].items():OmegaConf.update(cfg,k.lstrip('+'),v,force_add=True)
    cfg.actor_rollout_ref.actor.optim.total_training_steps=330
    env={k:v for k,v in launch['effective_environment'].items() if k not in ('CUDA_VISIBLE_DEVICES','RAY_ADDRESS','VERL_GLOBAL_TOKEN_MEAN')}
    env['PYTHONPATH']=str(DEST)+os.pathsep+str(OUT)+os.pathsep+env.get('PYTHONPATH','')
    env['DT_FSDP_EVENT_WAIT_RECEIPTS']=str(DEST/'original-eventwait')
    checkpoint=OUT/'formal-checkpoint80-source/actor'
    phase('initializing_original_actor',checkpoint=str(checkpoint))
    ray.init(address='local',num_cpus=8,num_gpus=2,include_dashboard=False,
        _temp_dir='/tmp/ray-tc-native-source-v14',runtime_env={'env_vars':env})
    try:
        group=RayWorkerGroup(RayResourcePool([2],use_gpu=True,max_colocate_count=1),
            RayClassWithInitArgs(EndpointWorker,cfg.actor_rollout_ref,'actor'))
        group.init_model();phase('restoring_original_checkpoint80')
        group.load_checkpoint(local_path=str(checkpoint),del_local_after_load=False)
        phase('reading_original_native_EOS_endpoints')
        results=group.read_sources()
        phase('complete_native_source_values_not_repair',rank_reports=results)
    except BaseException as exc:phase('failed',error=repr(exc));raise
    finally:ray.shutdown()

if __name__=='__main__':main()

"""Original native paired forwards for preselected real source positions.
Four calls per task: identity control, rowwise minimum, maximum, median negative.
Original targets/IDs/padding/native operators/credit owner; no training change.
"""
import argparse
from contextlib import nullcontext
import hashlib
import inspect
import json
import os
from pathlib import Path
import time
import torch
from inspect_extreme_endpoint import check_imports,actor_initialization_steps,sha,target_snapshot


def make_worker():
    import ray
    from verl.single_controller.base.decorator import Dispatch,register
    from verl.workers.fsdp_workers import ActorRolloutRefWorker
    @ray.remote
    class SampleWorker(ActorRolloutRefWorker):
        @register(dispatch_mode=Dispatch.ONE_TO_ALL)
        def inspect_sample(self,source_path,population_path,task,output):
            import psutil
            from deltatrace_rollout import DeltaTraceRolloutProducer
            from native_target_logit_rows import NativeTargetLogitRows
            from qwen35_answer_finite import PackedAnswerTargets,selected_target_log_probs
            from verl.utils.fsdp_utils import load_fsdp_model_to_gpu,offload_fsdp_model_to_cpu
            from verl.utils.torch_functional import pad_2d_list_to_length
            from counterfactual import reward_event_token_credit
            source=json.loads(Path(source_path).read_bytes());population=json.loads(Path(population_path).read_bytes())['tasks'][task];selected=population['selected_batch']
            assert sha(source_path)==population['source']['sha256'] and sha(selected['file']['path'])==selected['file']['sha256']
            payload=torch.load(selected['file']['path'],map_location='cpu',weights_only=False);rows=sorted(payload['rows'],key=lambda r:r['batch_row']);assert len(rows)==4
            assert (self.config.model.lora_rank,self.config.model.lora_alpha)==(8,16) and self.actor.config.ppo_micro_batch_size_per_gpu==4
            out=Path(output);r=dict(task=task,rank=self.rank,pid=os.getpid(),birth=psutil.Process().create_time(),source_sha256=sha(source_path),population_sha256=sha(population_path),native=selected['file'],owners=check_imports(source),modes={},operations=dict(native_paired_forwards=4,DT=0,optimizer=0,rollout=0,checkpoint_restore=0))
            log=(out/f'{task}-rank{self.rank}-phases.jsonl').open('a',buffering=1)
            def save(phase,**values):
                r.update(phase=phase,unix=time.time(),**values);(out/f'{task}-rank{self.rank}.json').write_text(json.dumps(r,indent=2)+'\n')
            def emit(phase,**values):
                log.write(json.dumps(dict(phase=phase,unix=time.time(),pss_bytes=psutil.Process().memory_full_info().pss,allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved(),**values))+'\n')
            producer=None;text=None;previous_attention=None;training=self.actor_module_fsdp.training
            try:
                shards=[]
                for name,p in self.actor_module_fsdp.named_parameters():
                    if '.lora_B.' in name:
                        p=p.detach();p=p.to_local() if hasattr(p,'to_local') else p;shards.append(dict(name=name,nonzero=int(torch.count_nonzero(p)),elements=p.numel()))
                assert shards and not any(v['nonzero'] for v in shards);r['lora_B_local_shards']=shards
                if self._is_offload_param:load_fsdp_model_to_gpu(self.actor_module_fsdp)
                self.actor_module_fsdp.eval();producer=DeltaTraceRolloutProducer(self.actor_module_fsdp,eos_token_id=self.tokenizer.eos_token_id,pad_token_id=self.tokenizer.pad_token_id)
                runner=producer.runner;text=runner.model.model.language_model;previous_attention=text.config._attn_implementation;text.set_attn_implementation('flash_attention_2')
                precision=lambda: nullcontext()
                if producer.native_fla_fp16:
                    from accelerated.qwen35.native_fla_precision import native_fla_fp16
                    precision=lambda: native_fla_fp16(self.actor_module_fsdp)
                conv=runner.attribute.__func__.__globals__['_native_conv_initial_states_scope']
                factual=pad_2d_list_to_length([row['selected'].tolist() for row in rows],self.tokenizer.eos_token_id,max_length=payload['native_signed'].shape[1]).cuda()
                selection=PackedAnswerTargets([row['case'] for row in rows],[row['target_offsets'] for row in rows],factual.shape[1],factual.device);selector=NativeTargetLogitRows(selection)
                r['geometry']=dict(shape=[8,factual.shape[1]],selected_lengths=[row['selected'].numel() for row in rows],target_counts=selection.counts,predictor_union=selector.rows.numel(),row_uids=[row['traj_uid'] for row in rows])
                baseline_factual=None
                for mode in ('identity','most_negative','most_positive','median_negative'):
                    reference=factual.clone();points=[]
                    for i,row in enumerate(selected['rows']):
                        point=None if mode=='identity' else row['candidates'].get(mode)
                        if point is not None:
                            assert rows[i]['traj_uid']==row['traj_uid'] and int(factual[i,point['packed_slot']])==point['token_id']
                            assert bool(rows[i]['prior'][point['response_slot']]) and not bool(rows[i]['target'][point['response_slot']])
                            reference[i,point['packed_slot']]=self.tokenizer.eos_token_id
                        points.append(point)
                    pair=torch.stack((reference,factual),dim=1).flatten(0,1);del reference
                    torch.cuda.reset_peak_memory_stats();emit('native_begin',mode=mode);save('native_begin',active_mode=mode);tick=time.perf_counter()
                    with torch.no_grad(),precision(),conv(text.layers,runner.native_conv_initial_states):
                        output_=runner.model.forward_root(input_ids=pair,attention_mask=torch.ones_like(pair),use_cache=False,logits_to_keep=selector.rows)
                        packed=selector.pack_logits(output_.logits);logits_dtype=str(output_.logits.dtype);del output_
                        logp=selected_target_log_probs(packed,selection);del packed
                        sums0=selection.sample_sums(logp[0::2].double()).cpu();sums1=selection.sample_sums(logp[1::2].double()).cpu();torch.cuda.synchronize()
                    targets=target_snapshot(selection,logp);targets.update(reference_joint_logp=sums0,factual_joint_logp=sums1,points=points,mode=mode,source_sha256=sha(source_path),native_sha256=selected['file']['sha256'])
                    path=out/f'{task}-rank{self.rank}-{mode}-targets.pt';torch.save(targets,path)
                    if baseline_factual is None:baseline_factual=targets['factual_target_logp'].clone()
                    descriptions=[]
                    for i,point in enumerate(points):
                        rr=dict(row=i,traj_uid=rows[i]['traj_uid'],point=point,reference_joint_logp=float(sums0[i]),factual_joint_logp=float(sums1[i]),single_delete_d=float(sums1[i]-sums0[i]))
                        if point is not None:
                            this=targets['samples'].eq(i);future=this & targets['predictor_positions'].ge(point['packed_slot']);earlier=this & ~future
                            delta=targets['factual_minus_reference'];rr.update(earlier_targets=int(earlier.sum()),earlier_delta_maxabs=float(delta[earlier].abs().max()) if earlier.any() else 0.,earlier_delta_sum=float(delta[earlier].double().sum()),future_targets=int(future.sum()),future_single_delete_d=float(delta[future].double().sum()),dominant_future_positions=targets['predictor_positions'][future][torch.argsort(delta[future].abs(),descending=True)[:5]].tolist(),dominant_future_labels=targets['labels'][future][torch.argsort(delta[future].abs(),descending=True)[:5]].tolist(),dominant_future_effects=delta[future][torch.argsort(delta[future].abs(),descending=True)[:5]].tolist())
                            credit=reward_event_token_credit(torch.tensor([[[rr['single_delete_d']]]],dtype=torch.float32),torch.tensor([[point['reward']]],dtype=torch.float32),torch.ones((1,1,1),dtype=torch.bool),torch.ones((1,1),dtype=torch.bool))
                            rr['native_endpoint_expected_A_FP32']=float(credit.advantages[0,0]);rr['credit_owner_path']=inspect.getsourcefile(reward_event_token_credit);rr['credit_owner_sha256']=sha(rr['credit_owner_path'])
                        descriptions.append(rr)
                    r['modes'][mode]=dict(seconds=time.perf_counter()-tick,logits_dtype=logits_dtype,logp_dtype=str(logp.dtype),targets_path=str(path),targets_sha256=sha(path),rows=descriptions,factual_target_logp_equal_to_identity=torch.equal(targets['factual_target_logp'],baseline_factual),factual_target_logp_maxabs_from_identity=float((targets['factual_target_logp']-baseline_factual).abs().max()),peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(),peak_torch_reserved_bytes=torch.cuda.max_memory_reserved(),saved_original_factual_drift=[float(sums1[i])-item['factual_target_logp'] for i,item in enumerate(payload['detail']['per_sample'])])
                    emit('native_complete',mode=mode);save('native_complete',active_mode=mode);del pair,targets,logp
                save('complete');return dict(task=task,rank=self.rank,completed=True,optimizer_steps=0)
            except BaseException:
                import traceback
                save('failed',traceback=traceback.format_exc());raise
            finally:
                if text is not None and previous_attention is not None:text.set_attn_implementation(previous_attention)
                self.actor_module_fsdp.train(training)
                if producer is not None:producer.runner.model.release_owner_params()
                if self._is_offload_param:offload_fsdp_model_to_cpu(self.actor_module_fsdp)
                log.close()
    return SampleWorker


def main():
    import ray
    from omegaconf import OmegaConf
    from verl.single_controller.ray import RayClassWithInitArgs,RayResourcePool,RayWorkerGroup
    parser=argparse.ArgumentParser();parser.add_argument('--task',choices=('textcraft','appworld'),required=True);parser.add_argument('--source',type=Path,required=True);parser.add_argument('--population',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--resolved-steps',type=int);args=parser.parse_args()
    source=json.loads(args.source.read_bytes());check_imports(source);population=json.loads(args.population.read_bytes())['tasks'][args.task];assert sha(args.source)==population['source']['sha256']
    args.output.mkdir(exist_ok=False);cfg=OmegaConf.load(Path(source['verl_root'])/'verl/trainer/config/ppo_trainer.yaml')
    for k,v in source['startup_options'].items():OmegaConf.update(cfg,k.lstrip('+'),v,force_add=True)
    cfg.actor_rollout_ref.actor.optim.total_training_steps=actor_initialization_steps(cfg.trainer.total_training_steps,args.resolved_steps)
    (args.output/'effective-config.yaml').write_text(OmegaConf.to_yaml(cfg));(args.output/'population-source.json').write_text(json.dumps(population['source'],indent=2)+'\n')
    ray.init(num_cpus=8,include_dashboard=False)
    try:
        group=RayWorkerGroup(RayResourcePool([2],use_gpu=True,max_colocate_count=1),RayClassWithInitArgs(make_worker(),cfg.actor_rollout_ref,'actor'));group.init_model()
        results=group.inspect_sample(str(args.source),str(args.population),args.task,str(args.output));(args.output/'completed.json').write_text(json.dumps(results,indent=2)+'\n')
    finally:ray.shutdown()

if __name__=='__main__':main()

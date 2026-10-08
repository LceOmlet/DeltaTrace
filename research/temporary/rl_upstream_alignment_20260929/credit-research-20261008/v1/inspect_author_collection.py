"""Original author curves and uniform deletion diagnostics on frozen development.

The original synchronous metric is called in CPU threads. Its callbacks pass
literal IDs through a queue; only the actor thread runs the original native
model, in pairs of signed/positive views for each of four trajectories. This
does not reproduce sorting, deletion, density, normalization or RISE/MAS.
Uniform single deletions are diagnostics only, never replacements for DT.
"""
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from functools import partial
import hashlib
import inspect
import json
import os
from pathlib import Path
from queue import Queue
import time

import torch
import inspect_action_curve as owner


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluate_author_curves_batched(function,rows,tokenizer,signed,positions,forward):
    """Batch only the original metric's score callbacks, preserving all inputs."""
    queue=Queue();pending=[];points={(i,v):[] for i in range(4) for v in range(2)}
    def curve(slot,view):
        def score(ids):
            promise=Future();queue.put((slot,view,ids[0].clone(),promise))
            result=promise.result()
            points[slot,view].append({'index':len(points[slot,view]),'logp':result})
            return torch.tensor([[result]],dtype=torch.float64,device='cpu')
        carrier=owner.JointScoreCarrier(rows[slot],tokenizer,score)
        attr=signed[slot] if view==0 else signed[slot].clamp_min(0)
        return owner.author_curve(function,carrier,attr,positions[slot],points[slot,view])
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures={(i,v):pool.submit(curve,i,v) for i in range(4) for v in range(2)}
        try:
            for step in range(21):
                pending=[]
                for _ in range(8):pending.append(queue.get(timeout=60))
                assert {(i,v) for i,v,_,_ in pending}==set(futures)
                pending.sort(key=lambda x:(x[0],x[1]))
                values=forward([x[2] for x in pending],'author_point_'+str(step))
                for request,value in zip(pending,values):request[3].set_result(float(value))
                pending=[]
            return {key:future.result() for key,future in futures.items()}
        except BaseException as error:
            while not queue.empty():pending.append(queue.get_nowait())
            for _,_,_,promise in pending:
                if not promise.done():promise.set_exception(error)
            raise


def make_worker():
    import ray
    from verl.single_controller.base.decorator import Dispatch, register
    from verl.workers.fsdp_workers import ActorRolloutRefWorker

    @ray.remote
    class CollectionWorker(ActorRolloutRefWorker):
        @register(dispatch_mode=Dispatch.ONE_TO_ALL)
        def inspect_action_curve(self, source_path, output):
            import ft_ifr_improve
            import psutil
            from deltatrace_rollout import DeltaTraceRolloutProducer
            from native_target_logit_rows import NativeTargetLogitRows
            from qwen35_answer_finite import PackedAnswerTargets, selected_target_log_probs
            from verl.utils.fsdp_utils import load_fsdp_model_to_gpu, offload_fsdp_model_to_cpu
            from verl.utils.torch_functional import pad_2d_list_to_length

            root=Path(output)
            plan_path=root.parent/'collection-inputs.json'
            plan=json.loads(plan_path.read_bytes())
            source=json.loads(Path(source_path).read_bytes())
            function=ft_ifr_improve.faithfulness_test_skip_tokens
            assert sha(inspect.getsourcefile(function))==plan['metric_owner']['sha256']
            assert inspect.signature(function).parameters['k'].default==20
            assert self._is_actor and not self._is_rollout
            assert self.actor.config.ppo_micro_batch_size_per_gpu==4
            assert (self.config.model.lora_rank,self.config.model.lora_alpha)==(8,16)
            assert source['fresh_base_model'] and not source['checkpoint_restore_requested']
            record={'rank':self.rank,'pid':os.getpid(),'birth':psutil.Process().create_time(),
                'scope':__doc__,'owners':owner.check_imports(source),'source_sha256':sha(source_path),
                'plan_sha256':sha(plan_path),'script_sha256':sha(__file__),
                'diagnostic_wall_limit_seconds':1800,
                'metric_owner':{'path':inspect.getsourcefile(function),'sha256':sha(inspect.getsourcefile(function))},
                'batches':[],'native_forward_calls':0,'operations':{'DT':0,'backward':0,'optimizer':0,'rollout':0,'checkpoint_restore':0}}
            phase_log=(root/f'rank{self.rank}-phases.jsonl').open('a',buffering=1)
            def save(phase,**values):
                event={'phase':phase,'unix':time.time(),'native_forward_calls':record['native_forward_calls'],
                    'allocated':torch.cuda.memory_allocated(),'reserved':torch.cuda.memory_reserved(),
                    'free':torch.cuda.mem_get_info()[0],'pss_bytes':psutil.Process().memory_full_info().pss,**values}
                record.update(event);phase_log.write(json.dumps(event)+'\n')
                (root/f'rank{self.rank}.json').write_text(json.dumps(record,indent=2)+'\n')
            producer=text=None;attention=None;training=self.actor_module_fsdp.training
            pending=[]
            started=time.perf_counter()
            try:
                lora=[]
                for name,parameter in self.actor_module_fsdp.named_parameters():
                    if '.lora_B.' in name:
                        p=parameter.detach();p=p.to_local() if hasattr(p,'to_local') else p
                        lora.append(int(torch.count_nonzero(p)))
                assert lora and not any(lora)
                record['lora_B_nonzero_local']=lora
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
                record['target_readers']={}
                for name,reader in [('targets',PackedAnswerTargets),('logit_rows',NativeTargetLogitRows),('log_probs',selected_target_log_probs)]:
                    p=inspect.getsourcefile(reader);record['target_readers'][name]={'path':p,'sha256':sha(p)}
                torch.cuda.reset_peak_memory_stats();save('original_actor_ready')
                def native_scores(ids,phase,*,rows,positions,width,selection,selector,batch,task,global_start):
                    if time.perf_counter()-started>record['diagnostic_wall_limit_seconds']:
                        raise RuntimeError('Bounded collection diagnostic reached its time budget; preserve partial measurements, do not retry automatically.')
                    assert len(ids)==8
                    for pair in range(8):
                        row=rows[pair//2];value=ids[pair]
                        changed=value.ne(row['selected']).nonzero().flatten()
                        assert bool(torch.isin(changed,torch.tensor(positions[pair//2])).all())
                        assert bool(value[changed].eq(self.tokenizer.eos_token_id).all())
                    packed_ids=pad_2d_list_to_length([x.tolist() for x in ids],self.tokenizer.eos_token_id,max_length=width).to('cuda')
                    torch.cuda.synchronize();tick=time.perf_counter()
                    save('native_forward_begin',active_task=task,active_batch=global_start,active_phase=phase,paired_shape=[8,width])
                    value=runner.model.forward_root(input_ids=packed_ids,attention_mask=torch.ones_like(packed_ids),
                        use_cache=False,logits_to_keep=selector.rows)
                    logits=selector.pack_logits(value.logits);del value
                    logp=selected_target_log_probs(logits,selection);del logits
                    a=selection.sample_sums(logp[0::2].double()).cpu()
                    b=selection.sample_sums(logp[1::2].double()).cpu()
                    sums=torch.stack((a,b),dim=1).flatten();del logp,packed_ids
                    torch.cuda.synchronize();seconds=time.perf_counter()-tick
                    record['native_forward_calls']+=1
                    point={'phase':phase,'seconds':seconds,'paired_shape':[8,width],'scores':sums.tolist()}
                    batch['forward_phases'].append(point)
                    save('native_forward_complete',active_task=task,active_batch=global_start,active_phase=phase,
                         seconds=seconds,elapsed_seconds=time.perf_counter()-started)
                    return sums
                with torch.no_grad(),precision,conv(text.layers,runner.native_conv_initial_states):
                    for task,entries in plan['tasks'].items():
                        for global_start in range(0,len(entries),8):
                            chosen=entries[global_start+4*self.rank:global_start+4*self.rank+4]
                            assert len(chosen)==4
                            rows=[];signed=[];positions=[]
                            for entry in chosen:
                                p=entry['native']['path'];assert sha(p)==entry['native']['sha256']
                                native=torch.load(p,map_location='cpu',weights_only=False)
                                row=next(r for r in native['rows'] if r['batch_row']==entry['batch_row'])
                                assert str(row['traj_uid'])==entry['traj_uid']
                                loc=row['prompt_length']+row['prior'][row['suffix_positions']].nonzero().flatten()
                                d=native['native_signed'][row['batch_row'],loc].clone().float()
                                assert d.numel()==entry['source_tokens'] and d.numel()>=20
                                rows.append(row);signed.append(d);positions.append(loc.tolist())
                                del native
                            width=max(row['selected'].numel() for row in rows)
                            selection=PackedAnswerTargets([r['case'] for r in rows],[r['target_offsets'] for r in rows],width,'cuda')
                            selector=NativeTargetLogitRows(selection)
                            batch={'task':task,'global_start':global_start,'width':width,
                                'trajectories':[{'traj_uid':e['traj_uid'],'initial_state_sha256':e['initial_state_sha256'],
                                   'source_tokens':e['source_tokens'],'reward':e['reward'],'views':{},'uniform_deletions':[]} for e in chosen],
                                'forward_phases':[]}
                            record['batches'].append(batch)
                            forward=partial(native_scores,rows=rows,positions=positions,width=width,
                                selection=selection,selector=selector,batch=batch,task=task,global_start=global_start)
                            # The original owner makes 21 callbacks for every row
                            # in this frozen batch (>=20 source positions).
                            curves=evaluate_author_curves_batched(function,rows,self.tokenizer,signed,positions,forward)
                            for (slot,view),result in curves.items():
                                batch['trajectories'][slot]['views']['signed_RISE' if view==0 else 'positive_MAS']=result
                            save('author_curves_batch_complete',active_task=task,active_batch=global_start)
                            # Two paired calls provide four uniform one-token
                            # deletions per trajectory; the already measured
                            # author factual score is reused, not re-forwarded.
                            for offset in (0,2):
                                inputs=[]
                                for row,entry in zip(rows,chosen):
                                    for query in entry['uniform_queries'][offset:offset+2]:
                                        ids=row['selected'].clone();assert int(ids[query['packed_slot']])==query['token_id']
                                        ids[query['packed_slot']]=self.tokenizer.eos_token_id;inputs.append(ids)
                                values=forward(inputs,'uniform_single_'+str(offset))
                                for slot,entry in enumerate(chosen):
                                    factual=batch['trajectories'][slot]['views']['signed_RISE']['score_points'][0]['logp']
                                    for v,query in enumerate(entry['uniform_queries'][offset:offset+2]):
                                        batch['trajectories'][slot]['uniform_deletions'].append(dict(query,
                                           factual_target_logp=factual,deleted_target_logp=float(values[2*slot+v]),
                                           native_single_d=factual-float(values[2*slot+v])))
                            save('development_batch_complete',active_task=task,active_batch=global_start)
                    # All observed A/r<-1 sources in the complete development
                    # captures, not just selected extremes or first-stage UIDs.
                    # Both ranks execute the same number of original FSDP calls.
                    tail_task=list(plan['tail_queries'])[self.rank]
                    queries=plan['tail_queries'][tail_task]
                    rounds=max((len(q)+3)//4 for q in plan['tail_queries'].values())
                    record['tail_results']=[];record['tail_batches']=[]
                    for index in range(rounds):
                        active=queries[4*index:4*index+4]
                        assert active
                        chosen=active+[active[-1]]*(4-len(active))
                        rows=[];positions=[]
                        for point in chosen:
                            assert sha(point['native']['path'])==point['native']['sha256']
                            native=torch.load(point['native']['path'],map_location='cpu',weights_only=False)
                            row=next(r for r in native['rows'] if r['batch_row']==point['batch_row'])
                            assert str(row['traj_uid'])==point['traj_uid']
                            rows.append(row)
                            positions.append((row['prompt_length']+row['prior'][row['suffix_positions']].nonzero().flatten()).tolist())
                            del native
                        width=max(r['selected'].numel() for r in rows)
                        selection=PackedAnswerTargets([r['case'] for r in rows],[r['target_offsets'] for r in rows],width,'cuda')
                        selector=NativeTargetLogitRows(selection)
                        batch={'task':tail_task,'index':index,'actual_queries':len(active),'padding_controls':4-len(active),'forward_phases':[]}
                        record['tail_batches'].append(batch)
                        inputs=[]
                        for point,row in zip(chosen,rows):
                            reference=row['selected'].clone()
                            assert int(reference[point['packed_slot']])==point['token_id']
                            reference[point['packed_slot']]=self.tokenizer.eos_token_id
                            inputs.extend([row['selected'],reference])
                        values=native_scores(inputs,'complete_development_tail',rows=rows,positions=positions,
                            width=width,selection=selection,selector=selector,batch=batch,task=tail_task,global_start=index)
                        for slot,point in enumerate(active):
                            record['tail_results'].append(dict(point,factual_target_logp=float(values[2*slot]),
                                deleted_target_logp=float(values[2*slot+1]),native_single_d=float(values[2*slot]-values[2*slot+1])))
                        save('tail_batch_complete',active_task=tail_task,tail_completed=len(record['tail_results']))
                save('complete',seconds=time.perf_counter()-started,
                    scope='Frozen development baseline collection; no numerical candidate, training-credit replacement or global degradation causal claim.')
                return {'rank':self.rank,'completed':True,'optimizer_steps':0}
            except BaseException as error:
                import traceback
                for _,_,_,promise in pending:
                    if not promise.done():promise.set_exception(error)
                save('failed',traceback=traceback.format_exc());raise
            finally:
                if text is not None and attention is not None:text.set_attn_implementation(attention)
                self.actor_module_fsdp.train(training)
                if producer is not None:producer.runner.model.release_owner_params()
                if self._is_offload_param:offload_fsdp_model_to_cpu(self.actor_module_fsdp)
    return CollectionWorker


if __name__=='__main__':
    owner.make_worker=make_worker
    owner.main()

"""One native B4 endpoint: restore the evidenced token in the joint EOS reference.

This diagnoses the observed joint-versus-single deletion sign difference. It
does not implement or change DT, credit, sampling, PPO or an environment. All
target scoring, actor initialization and model execution use their real owners.
"""
import argparse
from contextlib import nullcontext
import inspect
import json
import os
from pathlib import Path
import time

import torch
from inspect_extreme_endpoint import (CASES, check_imports, geometry, load_request,
    make_pair, sha, target_snapshot, causal_description)


def reference_context_pair(rows, width, eos, pad_rows, spec):
    pair = make_pair(rows, width, eos, pad_rows, spec)
    row = rows[spec['row']]
    positions = row['prior'][row['suffix_positions']].nonzero().flatten()+row['prompt_length']
    pair[2*spec['row']:2*spec['row']+2, positions] = eos
    pair[2*spec['row']+1, spec['packed_slot']] = spec['token_id']
    return pair


def make_worker():
    import ray
    from verl.single_controller.base.decorator import Dispatch, register
    from verl.workers.fsdp_workers import ActorRolloutRefWorker

    @ray.remote
    class ReferenceWorker(ActorRolloutRefWorker):
        @register(dispatch_mode=Dispatch.ONE_TO_ALL)
        def inspect_reference_context(self, source_path, output):
            from deltatrace_rollout import DeltaTraceRolloutProducer
            from native_target_logit_rows import NativeTargetLogitRows
            from qwen35_answer_finite import PackedAnswerTargets, selected_target_log_probs
            from verl.utils.fsdp_utils import load_fsdp_model_to_gpu, offload_fsdp_model_to_cpu
            from verl.utils.torch_functional import pad_2d_list_to_length
            import psutil

            spec = CASES['appworld']
            source, native, rows = load_request(source_path, spec['native'], spec)
            root = Path(output)
            record = dict(rank=self.rank, pid=os.getpid(), birth=psutil.Process().create_time(),
                owners=check_imports(source), source_sha256=sha(source_path),
                native_sha256=sha(spec['native']), geometry=geometry(source,native,rows,spec),
                intervention=f"Row{spec['row']}: both endpoints mask every original prior-source token with EOS; second endpoint restores only original token{spec['token_id']} at{spec['packed_slot']}. Other three rows remain identity controls.",
                operations=dict(native_paired_forward=1, DT=0, backward=0, optimizer=0,
                    checkpoint_restore=0, rollout=0))
            def save(phase, **values):
                record.update(phase=phase, unix=time.time(), **values)
                (root/f'rank{self.rank}.json').write_text(json.dumps(record,indent=2)+'\n')
            producer = None
            attention = None
            text = None
            training = self.actor_module_fsdp.training
            assert self._is_actor and not self._is_rollout
            assert self.actor.config.ppo_micro_batch_size_per_gpu == 4
            assert (self.config.model.lora_rank,self.config.model.lora_alpha)==(8,16)
            try:
                shards=[]
                for name,p in self.actor_module_fsdp.named_parameters():
                    if '.lora_B.' in name:
                        p=p.detach();p=p.to_local() if hasattr(p,'to_local') else p
                        shards.append(dict(name=name,elements=p.numel(),nonzero=int(torch.count_nonzero(p))))
                assert shards and not any(item['nonzero'] for item in shards)
                save('fresh_original_actor', lora_B_local_shards=shards)
                if self._is_offload_param:load_fsdp_model_to_gpu(self.actor_module_fsdp)
                self.actor_module_fsdp.eval()
                producer=DeltaTraceRolloutProducer(self.actor_module_fsdp,
                    eos_token_id=self.tokenizer.eos_token_id,pad_token_id=self.tokenizer.pad_token_id)
                runner=producer.runner;text=runner.model.model.language_model
                attention=text.config._attn_implementation
                text.set_attn_implementation('flash_attention_2')
                pair=reference_context_pair(rows,native['native_signed'].shape[1],
                    self.tokenizer.eos_token_id,pad_2d_list_to_length,spec).to('cuda')
                selection=PackedAnswerTargets([row['case'] for row in rows],
                    [row['target_offsets'] for row in rows],pair.shape[1],pair.device)
                selector=NativeTargetLogitRows(selection)
                precision=nullcontext()
                if producer.native_fla_fp16:
                    from accelerated.qwen35.native_fla_precision import native_fla_fp16
                    precision=native_fla_fp16(self.actor_module_fsdp)
                conv_scope=runner.attribute.__func__.__globals__['_native_conv_initial_states_scope']
                save('native_pair_begin', shape=list(pair.shape), target_predictor_union=selector.rows.numel())
                torch.cuda.synchronize();started=time.perf_counter()
                with torch.no_grad(),precision,conv_scope(text.layers,runner.native_conv_initial_states):
                    output_=runner.model.forward_root(input_ids=pair,
                        attention_mask=torch.ones_like(pair),use_cache=False,logits_to_keep=selector.rows)
                    dtype=str(output_.logits.dtype)
                    packed=selector.pack_logits(output_.logits);del output_
                    logp=selected_target_log_probs(packed,selection);del packed
                    sums0=selection.sample_sums(logp[0::2].double()).cpu()
                    sums1=selection.sample_sums(logp[1::2].double()).cpu()
                    targets=target_snapshot(selection,logp)
                torch.cuda.synchronize()
                original_description=causal_description(targets,
                    native['native_signed'][spec['row'],spec['packed_slot']],spec)
                description=dict(earlier_target_count=original_description['earlier_target_count'],
                    earlier_delta_maxabs=original_description['earlier_delta_maxabs'],
                    future_target_count=original_description['future_target_count'],
                    all_EOS_future_joint_logp=original_description['single_eos_future_joint_logp'],
                    only_token_restored_future_joint_logp=original_description['factual_future_joint_logp'],
                    token_effect_in_joint_EOS_context=original_description['native_single_eos_future_effect'])
                path=root/f'rank{self.rank}-target-logp.pt';torch.save(targets,path)
                save('complete', native_logits_dtype=dtype,target_logp_dtype=str(logp.dtype),
                    target_artifact=dict(path=str(path),sha256=sha(path)),
                    reference_joint_logp=sums0.tolist(),token_restored_joint_logp=sums1.tolist(),
                    restored_minus_reference=(sums1-sums0).tolist(),causal_description=description,
                    seconds=time.perf_counter()-started,pss_bytes=psutil.Process().memory_full_info().pss,
                    scope='Actual all-other-source-EOS context, not factual context and not a changed training estimator. Descriptive values only; no tolerance, correction or acceptance threshold.')
                return dict(rank=self.rank,completed=True,optimizer_steps=0)
            except BaseException:
                import traceback
                save('failed',traceback=traceback.format_exc());raise
            finally:
                if text is not None and attention is not None:text.set_attn_implementation(attention)
                self.actor_module_fsdp.train(training)
                if producer is not None:producer.runner.model.release_owner_params()
                if self._is_offload_param:offload_fsdp_model_to_cpu(self.actor_module_fsdp)
    return ReferenceWorker


def main():
    import ray
    from omegaconf import OmegaConf
    from verl.single_controller.ray import RayClassWithInitArgs,RayResourcePool,RayWorkerGroup
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    source=json.loads(args.source.read_bytes());check_imports(source)
    assert sha(args.source)==CASES['appworld']['source_sha256']
    args.output.mkdir(exist_ok=False)
    config=OmegaConf.load(Path(source['verl_root'])/'verl/trainer/config/ppo_trainer.yaml')
    for key,value in source['startup_options'].items():OmegaConf.update(config,key.lstrip('+'),value,force_add=True)
    config.actor_rollout_ref.actor.optim.total_training_steps=config.trainer.total_training_steps
    (args.output/'effective-config.yaml').write_text(OmegaConf.to_yaml(config))
    ray.init(num_cpus=8,include_dashboard=False)
    try:
        group=RayWorkerGroup(RayResourcePool([2],use_gpu=True,max_colocate_count=1),
            RayClassWithInitArgs(make_worker(),config.actor_rollout_ref,'actor'))
        group.init_model()
        values=group.inspect_reference_context(str(args.source),str(args.output))
        (args.output/'completed.json').write_text(json.dumps(values,indent=2)+'\n')
    finally:ray.shutdown()


if __name__=='__main__':main()

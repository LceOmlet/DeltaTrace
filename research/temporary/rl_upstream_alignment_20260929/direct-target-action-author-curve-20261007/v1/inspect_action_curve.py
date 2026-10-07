"""Original author cumulative deletion/RISE/MAS over a real joint action target.

The author owns sorting, deletion groups, density, normalisation and metrics.
The thin carrier supplies original IDs and noncontiguous source positions.
Original VERL owns actor initialization. Original DT target selection/scoring
owns the joint score; no reward labels, DT calls, sampling, gradients or update.
"""
import argparse
from contextlib import nullcontext
import inspect
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import torch
from inspect_extreme_endpoint import CASES, actor_initialization_steps, check_imports, geometry, load_request, sha

PROMPT = '<original-joint-action-context>'
FORMATTED = '<literal-original-joint-action-context>'
GENERATION = '<original-last-target-token>'


class LiteralIds:
    """Only supply unchanged token artifacts at the author's tokenizer seam."""
    def __init__(self, row, tokenizer):
        self.prefix = row['selected'][:-1][None].cpu()
        self.response = row['selected'][-1:][None].cpu()
        self.eos_token = tokenizer.eos_token

    def __call__(self, text, *, return_tensors, add_special_tokens):
        assert return_tensors == 'pt' and add_special_tokens is False
        if text == FORMATTED: return SimpleNamespace(input_ids=self.prefix)
        if text == GENERATION+self.eos_token: return SimpleNamespace(input_ids=self.response)
        raise ValueError('Unexpected literal carrier; no text encoding is performed')


class JointScoreCarrier:
    """Pass original IDs and scalar joint score to the unmodified author metric.

The target is the original scattered joint Y, not the one-token carrier suffix.
The callback composes both ID tensors and delegates Y scoring to the original
PackedAnswerTargets/NativeTargetLogitRows/selected_target_log_probs owners.
This extends the evaluation target, not the metric or training estimator.
"""
    def __init__(self, row, tokenizer, score):
        self.tokenizer = LiteralIds(row, tokenizer)
        self.device = torch.device('cpu')
        self.eos = tokenizer.eos_token_id
        self.score = score

    def format_prompt(self, text):
        assert text == ' '+PROMPT
        return FORMATTED

    def _ensure_pad_token_id(self):
        # The specified DT perturbation uses EOS, even on tokenizers with a
        # separate padding ID. No masks/observations/targets are perturbed.
        return self.eos

    def compute_logprob_response_given_prompt(self, prefix, suffix):
        return self.score(torch.cat((prefix, suffix), dim=1))


def author_curve(function, carrier, signed, positions, score_points):
    """Observe the owner's returned arrays; never reconstruct its metric."""
    captured = {}
    previous = sys.getprofile()
    def observe(frame, event, value):
        if previous is not None: previous(frame, event, value)
        if frame.f_code is function.__code__ and event == 'return' and value is not None:
            for name in ('scores','density','normalized_model_response','alignment_penalty','corrected_scores'):
                captured[name] = frame.f_locals[name].tolist()
    sys.setprofile(observe)
    try:
        values = function(carrier, signed[None], PROMPT, GENERATION,
            keep_prompt_token_indices=range(len(positions)), user_prompt_indices=positions)
    finally:
        sys.setprofile(previous)
    return dict(author_return=list(map(float,values)), author_fields=['rise','mas','rise_plus_ap'],
        author_arrays=captured, score_points=score_points)


def make_worker():
    import ray
    from verl.single_controller.base.decorator import Dispatch, register
    from verl.workers.fsdp_workers import ActorRolloutRefWorker

    @ray.remote
    class ActionCurveWorker(ActorRolloutRefWorker):
        @register(dispatch_mode=Dispatch.ONE_TO_ALL)
        def inspect_action_curve(self, source_path, output, case_name='appworld'):
            import psutil
            import ft_ifr_improve
            from deltatrace_rollout import DeltaTraceRolloutProducer
            from native_target_logit_rows import NativeTargetLogitRows
            from qwen35_answer_finite import PackedAnswerTargets, selected_target_log_probs
            from verl.utils.fsdp_utils import load_fsdp_model_to_gpu, offload_fsdp_model_to_cpu
            from verl.utils.torch_functional import pad_2d_list_to_length

            case = CASES[case_name]
            source, native, rows = load_request(source_path, case['native'], case)
            root = Path(output)
            function = ft_ifr_improve.faithfulness_test_skip_tokens
            owner_path = Path(inspect.getsourcefile(function))
            assert sha(owner_path) == '583f4b7d0426407eb9a517f173365762860a1f4382f472dffb5c07de7d3e94a1'
            record = dict(case_name=case_name,rank=self.rank,pid=os.getpid(),birth=psutil.Process().create_time(),
                owners=check_imports(source),source_sha256=sha(source_path),native_sha256=sha(case['native']),
                geometry=geometry(source,native,rows,case),
                author=dict(path=str(owner_path),sha256=sha(owner_path),
                    function=function.__qualname__,k_default=inspect.signature(function).parameters['k'].default),
                operations=dict(DT=0,rollout=0,backward=0,optimizer=0,checkpoint_restore=0),views={})
            def save(phase,**values):
                record.update(phase=phase,unix=time.time(),**values)
                (root/f'rank{self.rank}.json').write_text(json.dumps(record,indent=2)+'\n')
            producer = text = None
            attention = None
            training = self.actor_module_fsdp.training
            assert self._is_actor and not self._is_rollout
            assert self.actor.config.ppo_micro_batch_size_per_gpu == 4
            assert (self.config.model.lora_rank,self.config.model.lora_alpha) == (8,16)
            try:
                shards = []
                for name,p in self.actor_module_fsdp.named_parameters():
                    if '.lora_B.' in name:
                        p=p.detach();p=p.to_local() if hasattr(p,'to_local') else p
                        shards.append(dict(name=name,elements=p.numel(),nonzero=int(torch.count_nonzero(p))))
                assert shards and not any(item['nonzero'] for item in shards)
                save('original_actor_ready',lora_B_local_shards=shards)
                if self._is_offload_param:load_fsdp_model_to_gpu(self.actor_module_fsdp)
                self.actor_module_fsdp.eval()
                producer=DeltaTraceRolloutProducer(self.actor_module_fsdp,
                    eos_token_id=self.tokenizer.eos_token_id,pad_token_id=self.tokenizer.pad_token_id)
                runner=producer.runner;text=runner.model.model.language_model
                attention=text.config._attn_implementation;text.set_attn_implementation('flash_attention_2')
                width=native['native_signed'].shape[1]
                original=pad_2d_list_to_length([r['selected'].tolist() for r in rows],
                    self.tokenizer.eos_token_id,max_length=width)
                selection=PackedAnswerTargets([r['case'] for r in rows],
                    [r['target_offsets'] for r in rows],width,'cuda')
                selector=NativeTargetLogitRows(selection)
                row=rows[case['row']]
                positions=(row['prior'][row['suffix_positions']].nonzero().flatten()+row['prompt_length']).tolist()
                signed=native['native_signed'][case['row'],positions].float()
                precision=nullcontext()
                if producer.native_fla_fp16:
                    from accelerated.qwen35.native_fla_precision import native_fla_fp16
                    precision=native_fla_fp16(self.actor_module_fsdp)
                conv_scope=runner.attribute.__func__.__globals__['_native_conv_initial_states_scope']
                current_points=[];view='unset'
                def score(ids):
                    assert ids.shape == (1,row['selected'].numel())
                    changed=ids[0].ne(row['selected']).nonzero().flatten()
                    assert bool(torch.isin(changed,torch.tensor(positions)).all())
                    assert not changed.numel() or bool(ids[0,changed].eq(self.tokenizer.eos_token_id).all())
                    inputs=original.clone();inputs[case['row'],:ids.shape[1]]=ids[0]
                    pair=inputs.repeat_interleave(2,dim=0).to('cuda')
                    torch.cuda.synchronize();tick=time.perf_counter()
                    output_=runner.model.forward_root(input_ids=pair,attention_mask=torch.ones_like(pair),
                        use_cache=False,logits_to_keep=selector.rows)
                    packed=selector.pack_logits(output_.logits);del output_
                    logp=selected_target_log_probs(packed,selection);del packed
                    sums0=selection.sample_sums(logp[0::2].double()).cpu()
                    sums1=selection.sample_sums(logp[1::2].double()).cpu()
                    torch.cuda.synchronize()
                    point=dict(index=len(current_points),logp=float(sums1[case['row']]),
                        changed_input_positions=changed.tolist(),
                        other_three_joint_logp=[float(sums1[b]) for b in range(4) if b!=case['row']],
                        twin_row_score_differences=(sums1-sums0).tolist(),
                        seconds=time.perf_counter()-tick,pss_bytes=psutil.Process().memory_full_info().pss)
                    current_points.append(point)
                    save('author_score_complete',active_view=view,completed_points=len(current_points),last_point=point)
                    return sums1[case['row']].reshape(1,1)
                carrier=JointScoreCarrier(row,self.tokenizer,score)
                save('author_curve_begin',source_positions=positions,source_count=len(positions),
                    source_signed_dtype=str(native['native_signed'].dtype),metric_input_dtype=str(signed.dtype),
                    target_score='Original full-vocabulary scattered joint action Y; one scalar per author callback; not a new reward label or the literal carrier suffix alone.')
                with torch.no_grad(),precision,conv_scope(text.layers,runner.native_conv_initial_states):
                    for view,attr in [('signed_RISE',signed),('positive_MAS',signed.clamp_min(0))]:
                        current_points=[]
                        record['views'][view]=author_curve(function,carrier,attr,positions,current_points)
                        save('author_view_complete',active_view=view)
                save('complete',native_forward_calls=sum(len(v['score_points']) for v in record['views'].values()),
                    scope='One real trajectory and original author metric, not overall DT quality or an acceptance threshold. Source/target/observations retain original identity. No credit mutation or formal restart.')
                return dict(rank=self.rank,completed=True,optimizer_steps=0)
            except BaseException:
                import traceback
                save('failed',traceback=traceback.format_exc());raise
            finally:
                if text is not None and attention is not None:text.set_attn_implementation(attention)
                self.actor_module_fsdp.train(training)
                if producer is not None:producer.runner.model.release_owner_params()
                if self._is_offload_param:offload_fsdp_model_to_cpu(self.actor_module_fsdp)
    return ActionCurveWorker


def main():
    import ray
    from omegaconf import OmegaConf
    from verl.single_controller.ray import RayClassWithInitArgs,RayResourcePool,RayWorkerGroup
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--case',choices=CASES,default='appworld')
    parser.add_argument('--owner-total-training-steps',type=int)
    parser.add_argument('--owner-total-steps-evidence',type=Path)
    args=parser.parse_args()
    source=json.loads(args.source.read_bytes());check_imports(source)
    assert sha(args.source)==CASES[args.case]['source_sha256']
    args.output.mkdir(exist_ok=False)
    config=OmegaConf.load(Path(source['verl_root'])/'verl/trainer/config/ppo_trainer.yaml')
    for key,value in source['startup_options'].items():OmegaConf.update(config,key.lstrip('+'),value,force_add=True)
    config.actor_rollout_ref.actor.optim.total_training_steps=actor_initialization_steps(
        config.trainer.total_training_steps,args.owner_total_training_steps)
    (args.output/'actor-initialization.json').write_text(json.dumps(dict(
        source_trainer_total_training_steps=config.trainer.total_training_steps,
        original_owner_resolved=args.owner_total_training_steps,
        actor_optim_total_training_steps=config.actor_rollout_ref.actor.optim.total_training_steps,
        evidence=(dict(path=str(args.owner_total_steps_evidence),sha256=sha(args.owner_total_steps_evidence))
                  if args.owner_total_steps_evidence else None)),indent=2)+'\n')
    (args.output/'effective-config.yaml').write_text(OmegaConf.to_yaml(config))
    ray.init(num_cpus=8,include_dashboard=False)
    try:
        group=RayWorkerGroup(RayResourcePool([2],use_gpu=True,max_colocate_count=1),
            RayClassWithInitArgs(make_worker(),config.actor_rollout_ref,'actor'))
        group.init_model()
        if args.case=='appworld':
            values=group.inspect_action_curve(str(args.source),str(args.output))
        else:
            values=group.inspect_action_curve(str(args.source),str(args.output),args.case)
        (args.output/'completed.json').write_text(json.dumps(values,indent=2)+'\n')
    finally:ray.shutdown()


if __name__=='__main__':main()

"""Bounded old/new DT comparison on saved real task endpoints, same actor.

The saved endpoints came from a text reconstruction, not recovered original
rollout IDs. This checks the head precision repair, not task success rates.
"""
import copy
import importlib.util
import json
import os
from pathlib import Path
import time

import torch
from verify_author_rollout import configuration
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from deltatrace_rollout import DeltaTraceRolloutProducer
from qwen35_answer_finite import PackedAnswerTargets


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@torch.no_grad()
def main():
    torch.set_num_threads(4);torch.manual_seed(2026)
    root=Path(os.environ['DT_RUNTIME_ROOT'])
    output=Path(os.environ.get('HEAD_CHECK_OUTPUT',str(root/'receipts/rollout-major-cost/categorical-head-fp32-full.json')))
    source=root/'receipts/dt-extremes-side-20260925-1027/fixed-input-trace-0.pt'
    saved=torch.load(source,map_location='cpu',weights_only=True)
    report=dict(scope=__doc__,source=str(source),runs=[])
    started=time.perf_counter()
    def save(phase):
        report.update(phase=phase,elapsed=time.perf_counter()-started)
        output.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(dict(phase=phase,elapsed=report['elapsed'])),flush=True)
    save('load_original_actor')
    worker=ActorRolloutRefWorker(configuration(1024).actor_rollout_ref,'actor')
    worker.init_model()
    producer=DeltaTraceRolloutProducer(worker.actor_module_fsdp.eval(),
        eos_token_id=worker.tokenizer.eos_token_id,pad_token_id=worker.tokenizer.pad_token_id)
    runner=producer.runner
    oldpath=root/'releases/c88a749/clean/qwen35'
    oldanswer=load('head_rounding_baseline',oldpath/'qwen35_answer_finite.py')
    oldcontroller=load('controller_rounding_baseline',oldpath/'qwen35_dense_finite_runner.py')
    baseline=copy.copy(runner)
    baseline.answer=oldanswer.FiniteAnswerOps(compiled=False)
    # Keep all other execution settings and the original model identical.
    pair=saved['pair'].to('cuda')
    labels=producer.readout.alphabet.label_ids(worker.tokenizer)
    targets=PackedAnswerTargets([dict(target_ids=row[-1:].cpu(),prompt_length=len(row)-1)
        for row in pair[1::2]],[[0]]*4,pair.shape[1],pair.device,outcome_token_ids=labels)
    changed=(pair[0::2]!=pair[1::2]).cpu()
    head_inputs=[]
    handle=runner.model.lm_head.register_forward_pre_hook(
        lambda _m,args:head_inputs.append(args[0].detach().cpu().clone()))
    vectors={}
    try:
        for name,owner,call in [('native_bf16_rounding',baseline,oldcontroller.Qwen35DenseFiniteRunner.attribute),
                                ('fp32_readout',runner,type(runner).attribute)]:
            save(name+'_start');tick=time.perf_counter();head_inputs.clear()
            signed,detail=call(owner,pair,torch.ones_like(pair),targets,select_output_rows=True)
            vectors[name]=dict(signed=signed.cpu(),head_inputs=head_inputs.copy(),detail=detail)
            torch.save(vectors,output.with_suffix('.pt'))
            rows=[]
            for b in range(4):
                d=signed[b,changed[b]].double();lp=detail['target_logp1'][b]
                rows.append(dict(event=10+b,factual_logp=lp,d_min=float(d.min()),d_max=float(d.max()),
                    signed_sum=float(d.sum()),max_implied_p=float((lp-d).exp().max()),
                    root=lp-detail['target_logp0'][b]))
            report['runs'].append(dict(name=name,seconds=time.perf_counter()-tick,rows=rows,
                seed_minus_root=detail['compiled_seed_logprob_effect_minus_root']))
            save(name+'_done')
        a=vectors['native_bf16_rounding']['head_inputs']
        b=vectors['fp32_readout']['head_inputs']
        report['same_native_head_inputs']=len(a)==len(b) and all(torch.equal(x,y) for x,y in zip(a,b))
        torch.save(vectors,output.with_suffix('.pt'))
        save('completed_diagnostic_not_training_acceptance')
    except BaseException as exc:
        report['error']=repr(exc);save('failed');raise
    finally:
        handle.remove()
        if torch.distributed.is_initialized():torch.distributed.destroy_process_group()


if __name__=='__main__':main()

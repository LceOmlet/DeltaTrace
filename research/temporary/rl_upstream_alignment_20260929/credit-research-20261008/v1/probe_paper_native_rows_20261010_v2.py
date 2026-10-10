"""Observe one original B8 paper score call with the existing operator observer.

The first three GDN decoders are the existing observer's scope. This is a
localization of the four already observed same-input row differences, not a
quality sample, a new tolerance, a DT call, or a production change.
"""
from contextlib import nullcontext
import hashlib
import inspect
import json
import os
from pathlib import Path
import resource
import time

import psutil
import torch

OUT = Path(__file__).resolve().parent
ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
PAPER = ROOT/'receipts/paper-role-implementation-check-20261010-v3'
CURVES = ROOT/'receipts/paper-role-author-curves-20261010-v3'
started = time.perf_counter()
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
record = dict(scope=__doc__, pid=os.getpid(), birth=psutil.Process().create_time(),
    script_sha256=sha(__file__), native_forward_calls=0,
    operations=dict(DT=0, backward=0, optimizer=0, rollout=0,
                    checkpoint_restore=0, production_changes=0))


def save(phase, **values):
    record.update(phase=phase, unix=time.time(), seconds=time.perf_counter()-started, **values)
    record['resources'] = dict(allocated=torch.cuda.memory_allocated(),
        reserved=torch.cuda.memory_reserved(), peak_allocated=torch.cuda.max_memory_allocated(),
        peak_reserved=torch.cuda.max_memory_reserved(),
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        PSS_bytes=psutil.Process().memory_full_info().pss)
    (OUT/'result.json').write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
    print(phase, flush=True)


try:
    save('reading_original_owners')
    source = ROOT/'runs/textcraft-formal-stable-20261009-v1/source.json'
    assert sha(source) == '1c08b57bf83506d3e69d865f74377e3baa624358c678e0ac92cb6ebfb2d73658'
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    from deltatrace_rollout import DeltaTraceRolloutProducer
    from native_target_logit_rows import NativeTargetLogitRows
    from qwen35_answer_finite import PackedAnswerTargets, selected_target_log_probs
    from inspect_native_identity_operators import OperatorObservation
    import inspect_native_identity_operators as observer_module

    assert sha(inspect.getsourcefile(observer_module)) == '8903659d99f0d149562b986d030034a3acb18568fa0e7506b04f60fe30c358aa'
    old = json.loads((PAPER/'result.json').read_bytes())
    assert old['phase'] == 'complete'
    inputs = PAPER/'paper-inputs.json'
    assert sha(inputs) == '2f17897f9f87c662432b86e6c46a76fb113a1e483a21a2e1e223dd6961badd01'
    cases = json.loads(inputs.read_bytes())['cases']
    q = json.loads(Path(os.environ['DT_ENVIRONMENT_JSON']).read_bytes())['qwen35']
    model = AutoModelForImageTextToText.from_pretrained(q['checkpoint'],
        local_files_only=True, trust_remote_code=True, torch_dtype=torch.bfloat16,
        attn_implementation='flash_attention_2', device_map='cuda')
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(q['checkpoint'], local_files_only=True)
    producer = DeltaTraceRolloutProducer(model, eos_token_id=tokenizer.eos_token_id,
                                        pad_token_id=tokenizer.pad_token_id)
    runner = producer.runner
    record['owners'] = {}
    for name, value in [('runner',type(runner)), ('native_model',type(model)),
                        ('target',PackedAnswerTargets), ('observer',observer_module),
                        ('logit_rows',NativeTargetLogitRows)]:
        path = inspect.getsourcefile(value)
        record['owners'][name] = dict(path=path, resolved=str(Path(path).resolve()), sha256=sha(path))
        if name in old['owners']:
            assert record['owners'][name]['sha256'] == old['owners'][name]['sha256']
    assert tokenizer.eos_token_id == old['eos_token_id'] == 248046
    width = max(len(c['input_ids']) for c in cases)
    packed = torch.full((8,width),tokenizer.eos_token_id,dtype=torch.long,device='cuda')
    targets = []
    for i,c in enumerate(cases):
        ids = c['input_ids']
        assert hashlib.sha256(__import__('struct').pack('<'+'q'*len(ids),*ids)).hexdigest() == c['input_sha256']
        packed[2*i,:len(ids)] = torch.tensor(ids,device='cuda')
        packed[2*i+1] = packed[2*i]
        targets.append(dict(prompt_length=c['prompt_length'],
                            target_ids=torch.tensor(ids[c['prompt_length']:],dtype=torch.long)))
    assert torch.equal(packed[0::2],packed[1::2])
    selection = PackedAnswerTargets(targets,[list(range(c['target_length'])) for c in cases],width,'cuda')
    selector = NativeTargetLogitRows(selection)
    precision = nullcontext()
    if producer.native_fla_fp16:
        from accelerated.qwen35.native_fla_precision import native_fla_fp16
        precision = native_fla_fp16(model)
    conv_scope = runner.attribute.__func__.__globals__['_native_conv_initial_states_scope']
    active = dict(rank=0,index=0)
    observer = OperatorObservation(runner,OUT,active)
    torch.cuda.reset_peak_memory_stats()
    save('native_forward_begin', shape=list(packed.shape), source_sha256=sha(source),
         input_sha256=sha(inputs), identical_input_rows=True)
    with torch.no_grad(),precision,conv_scope(runner.model.model.language_model.layers,runner.native_conv_initial_states),observer:
        output = runner.model(input_ids=packed,attention_mask=torch.ones_like(packed),
                              use_cache=False,logits_to_keep=selector.rows)
        logits = selector.pack_logits(output.logits)
        del output
        logp = selected_target_log_probs(logits,selection)
        del logits
        a = selection.sample_sums(logp[0::2].double()).cpu()
        b = selection.sample_sums(logp[1::2].double()).cpu()
        scores = torch.stack((a,b),dim=1).flatten().tolist()
        del logp
    torch.cuda.synchronize()
    record['native_forward_calls'] = 1
    assert all(__import__('math').isfinite(v) for v in scores)
    previous = json.loads((CURVES/'result.json').read_bytes())
    previous_scores = previous['profiles']['gdn-symmetric-v1']['forward_phases'][0]['scores']
    save('complete', scores=scores, previous_scores=previous_scores,
         maximum_score_difference_vs_previous=max(abs(a-b) for a,b in zip(scores,previous_scores)),
         paired_score_differences=[scores[2*i]-scores[2*i+1] for i in range(4)],
         operator_observation=active['operator_observation'])
    (OUT/'completed.json').write_text(json.dumps(dict(pid=record['pid'],birth=record['birth'],
        unix=record['unix'],phase=record['phase'],native_forward_calls=1))+'\n')
except BaseException as error:
    import traceback
    save('failed', exception=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
    raise

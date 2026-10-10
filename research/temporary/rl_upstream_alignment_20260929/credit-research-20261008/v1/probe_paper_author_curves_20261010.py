"""Original author deletion metrics on the four literal paper examples.

Reuse the existing metric and batched callback transport. Only the native
target score is supplied here; sorting, deletion, normalization, and metric
calculation remain in the original author function. No new DT or training.
"""
from contextlib import nullcontext
import hashlib
import inspect
import json
import os
from pathlib import Path
import resource
import sys
import time

import psutil
import torch

OUT = Path(__file__).resolve().parent
ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
PAPER = ROOT/'receipts/paper-role-implementation-check-20261010-v3'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
started = time.perf_counter()
record = dict(scope=__doc__, pid=os.getpid(), birth=psutil.Process().create_time(),
              script_sha256=sha(__file__), native_forward_calls=0, profiles={},
              operations=dict(DT=0, optimizer=0, backward=0, rollout=0,
                              checkpoint_restore=0, production_changes=0))


def save(phase, **values):
    record.update(phase=phase, unix=time.time(), elapsed_seconds=time.perf_counter()-started, **values)
    record['resources'] = dict(allocated=torch.cuda.memory_allocated(),
        reserved=torch.cuda.memory_reserved(), peak_allocated=torch.cuda.max_memory_allocated(),
        peak_reserved=torch.cuda.max_memory_reserved(), pss_bytes=psutil.Process().memory_full_info().pss,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    (OUT/'result.json').write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
    print(phase, record['native_forward_calls'], flush=True)


try:
    save('loading_existing_owners')
    source_path = ROOT/'runs/textcraft-formal-stable-20261009-v1/source.json'
    assert sha(source_path) == '1c08b57bf83506d3e69d865f74377e3baa624358c678e0ac92cb6ebfb2d73658'
    q = json.loads(Path(os.environ['DT_ENVIRONMENT_JSON']).read_bytes())['qwen35']
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    from deltatrace_rollout import DeltaTraceRolloutProducer
    from native_target_logit_rows import NativeTargetLogitRows
    from qwen35_answer_finite import PackedAnswerTargets, selected_target_log_probs
    import ft_ifr_improve
    import inspect_author_collection
    import inspect_action_curve

    metric = ft_ifr_improve.faithfulness_test_skip_tokens
    record['owners'] = {}
    for name, value in [('metric',metric), ('transport',inspect_author_collection),
                        ('carrier',inspect_action_curve), ('target',PackedAnswerTargets),
                        ('logit_rows',NativeTargetLogitRows)]:
        path = inspect.getsourcefile(value)
        record['owners'][name] = dict(path=path, resolved=str(Path(path).resolve()), sha256=sha(path))
    assert record['owners']['metric']['sha256'] == '583f4b7d0426407eb9a517f173365762860a1f4382f472dffb5c07de7d3e94a1'
    assert record['owners']['transport']['sha256'] == '0ac43043bc8efa6645794f7d6a17cf0d2db82c631646eb47e6983eb77de12946'
    assert record['owners']['carrier']['sha256'] == '55f88c760cd8d4317ef9ac07142e5c9ecfa4f0f53f1194d6e673f5f4ba8aa469'
    assert inspect.signature(metric).parameters['k'].default == 20
    old = json.loads((PAPER/'result.json').read_bytes())
    assert old['phase'] == 'complete' and old['eos_token_id'] == 248046
    inputs = json.loads((PAPER/'paper-inputs.json').read_bytes())
    cases = inputs['cases']; assert len(cases) == 4
    record['inputs'] = dict(path=str(PAPER/'paper-inputs.json'), sha256=sha(PAPER/'paper-inputs.json'))
    model = AutoModelForImageTextToText.from_pretrained(q['checkpoint'],
        local_files_only=True, trust_remote_code=True, torch_dtype=torch.bfloat16,
        attn_implementation='flash_attention_2', device_map='cuda')
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(q['checkpoint'], local_files_only=True)
    producer = DeltaTraceRolloutProducer(model, eos_token_id=tokenizer.eos_token_id,
                                       pad_token_id=tokenizer.pad_token_id)
    assert producer.eos_token_id == old['eos_token_id']
    runner = producer.runner
    for name, value in [('runner',type(runner)), ('native_model',type(model))]:
        path=inspect.getsourcefile(value)
        record['owners'][name]=dict(path=path,resolved=str(Path(path).resolve()),sha256=sha(path))
        assert record['owners'][name]['sha256'] == old['owners'][name]['sha256']
    record['eos_token_id'] = tokenizer.eos_token_id
    width = max(len(c['input_ids']) for c in cases)
    rows=[];positions=[]
    for c in cases:
        ids=c['input_ids']
        assert hashlib.sha256(__import__('struct').pack('<'+'q'*len(ids),*ids)).hexdigest()==c['input_sha256']
        assert ids[-1] == tokenizer.eos_token_id
        rows.append(dict(selected=torch.tensor(ids,dtype=torch.long),
            case=dict(prompt_length=c['prompt_length'],target_ids=torch.tensor(ids[c['prompt_length']:],dtype=torch.long)),
            target_offsets=list(range(c['target_length']))))
        positions.append([t['input_position'] for t in c['tokens'] if t['eligible']])
    selection=PackedAnswerTargets([r['case'] for r in rows],[r['target_offsets'] for r in rows],width,'cuda')
    selector=NativeTargetLogitRows(selection)
    precision=nullcontext()
    if producer.native_fla_fp16:
        from accelerated.qwen35.native_fla_precision import native_fla_fp16
        precision=native_fla_fp16(model)
    conv_scope=runner.attribute.__func__.__globals__['_native_conv_initial_states_scope']
    text=runner.model.model.language_model
    record['shape']=[8,width]
    record['planned_native_forward_calls']=63
    record['score_scope']='Full original stored response including EOS, four cases, paired signed/positive metric views. No sampled response or trained policy.'
    torch.cuda.reset_peak_memory_stats()
    save('original_native_target_reader_ready')
    with torch.no_grad(),precision,conv_scope(text.layers,runner.native_conv_initial_states):
        for profile in ('historical-paper-clean-v1','clean-v1','gdn-symmetric-v1'):
            if profile=='historical-paper-clean-v1':
                signed=[torch.tensor([t['score'] for t in c['tokens'] if t['eligible']],dtype=torch.float32) for c in cases]
                vector_ref=dict(source='Original stored paper attribution',inputs_sha256=record['inputs']['sha256'])
            else:
                path=PAPER/(profile+'.pt')
                data=torch.load(path,map_location='cpu',weights_only=False)
                signed=[data['signed'][i,loc].float() for i,loc in enumerate(positions)]
                vector_ref=dict(path=str(path),sha256=sha(path));del data
            active=dict(vector_source=vector_ref,forward_phases=[])
            record['profiles'][profile]=active

            def forward(ids,phase):
                assert len(ids)==8
                if time.perf_counter()-started>600:
                    raise RuntimeError('Bounded paper-example diagnostic exceeded 600 seconds; retain partial result, no automatic retry.')
                changed=[]
                for slot,value in enumerate(ids):
                    row=rows[slot//2]
                    loc=value.ne(row['selected']).nonzero().flatten()
                    assert bool(torch.isin(loc,torch.tensor(positions[slot//2])).all())
                    assert bool(value[loc].eq(tokenizer.eos_token_id).all())
                    changed.append(loc.tolist())
                packed=torch.full((8,width),tokenizer.eos_token_id,dtype=torch.long,device='cuda')
                for slot,value in enumerate(ids):packed[slot,:value.numel()]=value.to('cuda')
                torch.cuda.synchronize();tick=time.perf_counter()
                save('native_forward_begin',active_profile=profile,active_point=phase)
                value=runner.model.forward_root(input_ids=packed,attention_mask=torch.ones_like(packed),
                    use_cache=False,logits_to_keep=selector.rows)
                logits=selector.pack_logits(value.logits);del value
                logp=selected_target_log_probs(logits,selection);del logits
                a=selection.sample_sums(logp[0::2].double()).cpu()
                b=selection.sample_sums(logp[1::2].double()).cpu()
                result=torch.stack((a,b),dim=1).flatten();del logp,packed
                torch.cuda.synchronize()
                assert torch.isfinite(result).all()
                active['forward_phases'].append(dict(phase=phase,seconds=time.perf_counter()-tick,
                    scores=result.tolist(),changed_input_positions=changed))
                record['native_forward_calls']+=1
                save('native_forward_complete',active_profile=profile,active_point=phase)
                return result

            curves=inspect_author_collection.evaluate_author_curves_batched(
                metric,rows,tokenizer,signed,positions,forward)
            active['cases']=[dict(dataset=c['dataset'],index=c['index'],input_sha256=c['input_sha256'],
                source_positions=positions[i],views={name:curves[i,v] for v,name in enumerate(('signed_RISE','positive_MAS'))})
                for i,c in enumerate(cases)]
            save('author_profile_complete',active_profile=profile)
    assert record['native_forward_calls']==63
    save('complete')
except BaseException as error:
    import traceback
    record['exception']=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
    save('failed')
    raise

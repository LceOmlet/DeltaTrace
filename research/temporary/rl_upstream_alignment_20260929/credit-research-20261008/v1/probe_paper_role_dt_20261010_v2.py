"""Run the existing DT owner on four literal stored paper inputs, no training."""
from contextlib import nullcontext
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time

import psutil
import torch

OUT = Path(__file__).resolve().parent
ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
source_path = ROOT/'runs/textcraft-formal-stable-20261009-v1/source.json'
source = json.loads(source_path.read_bytes())
q = json.loads(Path(os.environ['DT_ENVIRONMENT_JSON']).read_bytes())['qwen35']
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
record = dict(scope=__doc__, source_sha256=sha(source_path), script_sha256=sha(__file__),
              pid=os.getpid(), birth=psutil.Process().create_time(),
              checkpoint=q['checkpoint'], training_changes=0, optimizer_steps=0,
              rollout=0, checkpoint_restore=0, DT_calls=0, profiles={})


def save(phase, **values):
    record.update(phase=phase, unix=time.time(), **values)
    record['resources'] = dict(allocated=torch.cuda.memory_allocated(),
        reserved=torch.cuda.memory_reserved(), peak_allocated=torch.cuda.max_memory_allocated(),
        peak_reserved=torch.cuda.max_memory_reserved(),
        pss_bytes=psutil.Process().memory_full_info().pss,
        peak_rss_kib=__import__('resource').getrusage(__import__('resource').RUSAGE_SELF).ru_maxrss)
    (OUT/'result.json').write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
    print(phase, flush=True)


try:
    save('loading_existing_checkpoint')
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    from deltatrace_rollout import DeltaTraceRolloutProducer
    from deltatrace_credit import trace_token_attribution
    model = AutoModelForImageTextToText.from_pretrained(q['checkpoint'],
        local_files_only=True, trust_remote_code=True, torch_dtype=torch.bfloat16,
        attn_implementation='flash_attention_2', device_map='cuda')
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(q['checkpoint'], local_files_only=True)
    producer = DeltaTraceRolloutProducer(model, eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id)
    record['eos_token_id']=producer.eos_token_id
    record['pad_token_id']=producer.pad_token_id
    record['tokenizer_sha256']=sha(Path(q['checkpoint'])/'tokenizer.json')
    assert producer.eos_token_id==248044
    runner = producer.runner
    from profiles.official import make_qwen35_runner
    from qwen35_answer_finite import PackedAnswerTargets
    import qwen35_gdn_finite
    record['owners'] = {}
    for name, owner in [('factory', make_qwen35_runner), ('runner', type(runner)),
                        ('GDN', qwen35_gdn_finite), ('target', PackedAnswerTargets),
                        ('trace_boundary', trace_token_attribution), ('native_model', type(model))]:
        p=inspect.getsourcefile(owner)
        record['owners'][name] = dict(path=p, resolved=str(Path(p).resolve()), sha256=sha(p))
    assert record['owners']['GDN']['sha256']=='7c06d5e0a4d6c00c13483dadd27d6389e25eee7868a05666d2d1faeaa2d62656'
    fixture=json.loads((OUT/'paper-inputs.json').read_bytes())
    cases=fixture['cases'];assert len(cases)==4
    width=max(len(c['input_ids']) for c in cases)
    selected=torch.full((4,width),producer.eos_token_id,dtype=torch.long,device='cuda')
    reference=selected.clone()
    targets=[];offsets=[]
    for row,c in enumerate(cases):
        ids=c['input_ids']
        assert hashlib.sha256(__import__('struct').pack('<'+'q'*len(ids),*ids)).hexdigest()==c['input_sha256']
        assert len(ids)==c['prompt_length']+c['target_length']
        selected[row,:len(ids)]=torch.tensor(ids,device='cuda')
        reference[row,:len(ids)]=selected[row,:len(ids)]
        positions=[t['input_position'] for t in c['tokens'] if t['eligible']]
        reference[row,positions]=producer.eos_token_id
        targets.append(dict(prompt_length=c['prompt_length'],target_ids=torch.tensor(ids[c['prompt_length']:],dtype=torch.long)))
        offsets.append(list(range(c['target_length'])))
    record['inputs']=[dict(dataset=c['dataset'],index=c['index'],input_sha256=c['input_sha256'],
        length=len(c['input_ids']),prompt=c['prompt_length'],target=c['target_length'],
        source_positions=[t['input_position'] for t in c['tokens'] if t['eligible']]) for c in cases]
    record['shape']=[4,width]
    record['comparison_scope']='Same base checkpoint and literal stored response, full-vocabulary target, native MetaX BF16/FP16-GDN; historical paper scalars are not a numerical tolerance.'
    precision=nullcontext()
    if producer.native_fla_fp16:
        from accelerated.qwen35.native_fla_precision import native_fla_fp16
        precision=native_fla_fp16(model)
    save('owners_and_literal_inputs_ready')
    # Both are existing official profiles. The clean profile uses its original
    # factory defaults, not a copied or modified propagation implementation.
    with torch.no_grad(),precision:
        for profile in ('gdn-symmetric-v1','clean-v1'):
            current=runner if profile=='gdn-symmetric-v1' else make_qwen35_runner(
                runner.model,runner.finite_fa,runner.finite_fla,profile='clean-v1')
            save('DT_begin_'+profile)
            torch.cuda.synchronize();tick=time.perf_counter()
            signed,roots,details=trace_token_attribution(current,reference,selected,targets,offsets,
                packed_answer_targets=PackedAnswerTargets)
            torch.cuda.synchronize();elapsed=time.perf_counter()-tick
            vector=signed.detach().cpu().double()
            torch.save(dict(signed=vector,roots=roots.detach().cpu(),details=details),OUT/(profile+'.pt'))
            rows=[]
            for row,c in enumerate(cases):
                names=[]
                for name in ('William Shakespeare','William Walton'):
                    start=0
                    while True:
                        at=c['user_text'].find(name,start)
                        if at<0:break
                        end=at+len(name)
                        tokens=[t for t in c['tokens'] if t['char_span'][0]<end and t['char_span'][1]>at]
                        names.append(dict(name=name,char_span=[at,end],
                            positions=[t['input_position'] for t in tokens],
                            token_scores=[float(vector[row,t['input_position']]) for t in tokens],
                            current_sum=sum(float(vector[row,t['input_position']]) for t in tokens),
                            historical_sum=sum(t['score'] for t in tokens)))
                        start=end
                eligible={t['input_position'] for t in c['tokens'] if t['eligible']}
                fixed=[j for j in range(width) if j not in eligible]
                rows.append(dict(dataset=c['dataset'],index=c['index'],names=names,
                    root_effect=float(roots[row]),signed_sum=float(vector[row].sum()),
                    all_finite=bool(torch.isfinite(vector[row]).all()),
                    fixed_position_max_abs=float(vector[row,fixed].abs().max())))
            record['profiles'][profile]=dict(seconds=elapsed,rows=rows,
                norm_gate_rules=current.norm_gate_rules,
                symmetric_memory_layers=sorted(current.finite_fla_by_layer))
            record['DT_calls']+=1
            save('DT_complete_'+profile)
            del signed,roots,details,vector
    save('complete')
except BaseException as error:
    import traceback
    record['exception']=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
    save('failed')
    raise

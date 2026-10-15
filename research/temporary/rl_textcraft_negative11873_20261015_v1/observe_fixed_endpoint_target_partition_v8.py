"""Test the repeated-target mechanism with original target and runner interfaces.

All paired source/reference IDs, masks, weights and native cache provider stay
fixed. Only the readout rows are partitioned using PackedAnswerTargets. This
does not call the RL readout with new targets or replace removed targets by EOS.
Return the original full-target result unchanged to the diagnostic caller.
"""
from contextlib import contextmanager
import hashlib,json
from pathlib import Path
import torch
from observe_existing_token_transport_v1 import observe_transport as transport_v1

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
RECORD=ROOT/'runs/textcraft-dt-grpo-multiplier-variance-only-20261014-v3/credit-records/rank1-pid3153050/joint-1792046604440421161.pt'

def digest(value):
    h=hashlib.sha256()
    def visit(x):
        if isinstance(x,torch.Tensor):h.update(x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        elif isinstance(x,dict):
            for k in sorted(x):h.update(k.encode());visit(x[k])
        elif x is not None:raise TypeError(type(x))
    visit(value);return h.hexdigest()

@contextmanager
def observe_transport(runner,focus_row,focus_position):
    from qwen35_answer_finite import PackedAnswerTargets
    original=runner.attribute
    original_rows=torch.load(RECORD,map_location='cpu',weights_only=True)['rows']
    original_rows=sorted(original_rows,key=lambda r:r['input_ids'].numel())
    reports=[]
    def attribute(paired_ids,mask,selection,*args,**kwargs):
        assert selection.batch==len(original_rows)==4
        ids_sha=digest(paired_ids);mask_sha=digest(mask)
        full=original(paired_ids,mask,selection,*args,**kwargs)
        def record(name,selected,result):
            signed,info=result
            reports.append(dict(name=name,target_counts=selected.counts,
                target_offsets=selected.offsets,
                focus_DT=float(signed[focus_row,focus_position]),
                focus_neighbor_values=signed[focus_row,focus_position-2:focus_position+3].tolist(),
                paired_ids_SHA256=ids_sha,mask_SHA256=mask_sha,
                target_logp0=info['target_logp0'],target_logp1=info['target_logp1'],
                native_row_prefix_lengths=info['native_row_prefix_lengths'],
                seconds=info['complete_attribution_seconds_with_diagnostics']))
            assert digest(paired_ids)==ids_sha and digest(mask)==mask_sha
        record('original_all_targets',selection,full)
        cases=[];firsts=[];rests=[]
        for b,row in enumerate(original_rows):
            prompt=row['prompt_length'];length=row['input_ids'].numel()
            actual=paired_ids[2*b+1,:length]
            assert torch.equal(actual.cpu(),row['input_ids'])
            cases.append(dict(prompt_length=prompt,target_ids=actual[prompt:].cpu()))
            offsets=selection.offsets[b]
            first=[j for i,j in enumerate(offsets) if i==0 or j!=offsets[i-1]+1]
            rest=[j for j in offsets if j not in first]
            assert first and rest
            firsts.append(first);rests.append(rest)
        for name,offsets in [('first_token_of_each_original_target_span',firsts),('remaining_original_target_tokens',rests)]:
            selected=PackedAnswerTargets(cases,offsets,selection.length,paired_ids.device)
            # Both interventions use the exact ORIGINAL endpoints. The only
            # changed argument is the original owner's target-row selection.
            result=original(paired_ids,mask,selected,*args,**kwargs)
            record(name,selected,result)
        return full
    runner.attribute=attribute
    try:
        with transport_v1(runner,focus_row,focus_position) as (metadata,payload):
            metadata['fixed_endpoint_target_partitions']=reports
            metadata['pre_measurement_prediction']='The first token of repeated original Action targets supplies most of the negative source effect; the remaining target tokens should not account for its large magnitude. This follows the observed last-FA query response, not a single-deletion oracle.'
            yield metadata,payload
        assert len(reports)==3
    finally:runner.attribute=original

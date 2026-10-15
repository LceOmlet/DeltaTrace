"""Audit already saved step24 DT vectors and the actual owner's CPU packing.

No model forward, finite propagation, policy update, or replacement algorithm.
Probability bounds here are mathematical diagnostics, never clipping rules.
"""
import hashlib,json,math,os,sys,time
from pathlib import Path

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
DEST=ROOT/'receipts/textcraft-efficiency-20261015-v1'
RUN=ROOT/'runs/textcraft-dt-grpo-multiplier-variance-only-20261014-v3'
launch=json.loads((RUN/'launch.json').read_bytes())
os.environ.update(launch['effective_environment']);os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path[:0]=launch['effective_environment']['PYTHONPATH'].split(':')
owner=Path(launch['effective_environment']['DT_ROOT'])/'clean/qwen35'
sys.path.insert(0,str(owner))
import torch
from qwen35_answer_finite import PackedAnswerTargets
from native_target_logit_rows import NativeTargetLogitRows
torch.set_num_threads(2)

inputs=json.loads((DEST/'step24-credit-audit-input.json').read_bytes())
audit=json.loads((DEST/'step24-saved-credit-CPU-audit.json').read_bytes())
case=audit['steps']['24']['raw_minimum']
paths=[]
for worker in inputs['workers'].values():
    for item in worker:
        if int(item['iteration'])==24:
            paths.extend(item['paths'])
rows=[];identities=set();by_file=[]
for entry in paths:
    path=entry['path'] if isinstance(entry,dict) else entry
    data=torch.load(path,map_location='cpu',weights_only=True)
    batch=data['rows'];length=max(len(r['input_ids']) for r in batch)
    cases=[dict(target_ids=r['input_ids'][r['prompt_length']:],prompt_length=r['prompt_length']) for r in batch]
    offsets=[r['target_offsets'] for r in batch]
    selection=PackedAnswerTargets(cases,offsets,length,'cpu')
    output_map=NativeTargetLogitRows(selection)
    mapped_targets=[]
    for b,row in enumerate(batch):
        idx=selection.samples==b
        assert torch.equal(selection.labels[idx],row['input_ids'][selection.positions[idx]+1])
        assert selection.counts[b]==int(row['target_mask'].sum())
        assert torch.equal(row['target_mask'][row['suffix_positions']].nonzero().flatten(),torch.tensor(row['target_offsets']))
        assert torch.equal(output_map.rows[output_map.packed_rows[0::2][idx]],selection.positions[idx])
        assert torch.equal(selection.paired_samples.reshape(-1,2)[idx],torch.tensor([2*b,2*b+1]).expand(int(idx.sum()),2))
        mapped_targets.append(dict(traj_uid=row['traj_uid'],targets=int(idx.sum()),mapped_labels_equal_saved_next_tokens=True))
        uid=row['traj_uid']
        if uid in identities:continue
        identities.add(uid)
        d=row['source_log_ratios'].double()
        prior=row['prior_source_mask'];lp=float(row['factual_target_logp'])
        positions=(prior & (lp-d>0)).nonzero().flatten()
        largest=None
        if len(positions):
            pos=int(positions[(lp-d[positions]).argmax()])
            largest=dict(response_position=pos,d=float(d[pos]),implied_deleted_logp=lp-float(d[pos]))
        vals=d[prior]
        rows.append(dict(traj_uid=uid,source=path,factual_logp=lp,reference_logp=float(row['reference_target_logp']),
            prior_source_tokens=int(prior.sum()),positive_DT_sum=float(vals[vals>0].sum()),negative_DT_sum=float(vals[vals<0].sum()),
            bound_violations=len(positions),largest=largest))
    by_file.append(dict(path=path,rows=len(batch),packing=mapped_targets))

record=torch.load(case['source'],map_location='cpu',weights_only=True)
row=next(r for r in record['rows'] if r['traj_uid']==case['traj_uid'])
suffix=row['suffix_positions'];n=len(row['input_ids'])-row['prompt_length']
factual=row['input_ids'];reference=factual.clone()
source_indices=row['prior_source_mask'][suffix].nonzero().flatten()+row['prompt_length']
reference[source_indices]=record['eos_token_id']
changed=(reference!=factual).nonzero().flatten()
extreme_full=row['prompt_length']+int((suffix==case['raw_min_response_position']).nonzero().item())
assert extreme_full in changed
assert all(not bool(row['target_mask'][int(suffix[i-row['prompt_length']])]) for i in changed)
assert torch.equal(suffix,torch.arange(n))
value=dict(unix=time.time(),scope=__doc__,only_iteration=24,
    record_files=len(by_file),unique_traced_trajectories=len(rows),
    native_packing_checks='All saved target labels, causal predictor offsets, sample/endpoint pairs and selected-output-row maps passed exact CPU equality.',
    bound_violating_trajectories=sum(r['bound_violations']>0 for r in rows),bound_violating_tokens=sum(r['bound_violations'] for r in rows),
    violating_rows=sorted([r for r in rows if r['bound_violations']],key=lambda r:r['largest']['implied_deleted_logp'],reverse=True),
    extreme_row=next(r for r in rows if r['traj_uid']==case['traj_uid']),
    extreme_endpoint_reconstruction=dict(actual_context=len(factual),suffix_holes=0,original_source_tokens=len(source_indices),
        actually_changed_tokens=len(changed),single_deleted_token_was_not_evaluated=True,
        selected_extreme_position=extreme_full,reference_token=int(reference[extreme_full]),factual_token=int(factual[extreme_full]),
        unchanged_targets_and_observations=True),
    code_binding={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [owner/'qwen35_answer_finite.py',owner/'native_target_logit_rows.py']},
    source_files=by_file,model_DT_backward_optimizer_calls=0,training_changes=0)
(DEST/'step24-probability-mapping-CPU-audit.json').write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps({k:v for k,v in value.items() if k not in ['source_files','violating_rows']},ensure_ascii=False))

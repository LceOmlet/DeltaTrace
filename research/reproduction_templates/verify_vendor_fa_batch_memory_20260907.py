"""Review actual same-job B1/B2/B4 memory without inflating native backward."""
import ast,hashlib,json,math,statistics
from pathlib import Path
import numpy as np
A=Path(__file__).resolve().parent;F=A/'snapshot${ARTIFACT_ROOT}/codex_vendor_fa_batch_memory_20260907_v1';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
d=json.loads((F/'results.json').read_text());p=d['protocol']
assert d['status']=='complete' and not d['quality_evaluated']
assert p==json.loads((A/'vendor_fa_batch_memory_protocol_20260907.json').read_text())
assert sha(F/'study.py')==p['study_sha256']==sha(A/'vendor_fa_batch_memory_20260907.py')
for name,h in p['sources'].items():assert sha(F/name)==sha(A/name)==h
previous=json.loads((A/'vendor_fa_batch_protocol_20260907.json').read_text())
assert {k:v for k,v in p['sources'].items() if k!='native_backward_memory_reference_batched.py'}==previous['sources']
parent_path=A/'snapshot'/p['required_parent'].lstrip('/');assert sha(parent_path)==p['required_parent_sha256'];parent=json.loads(parent_path.read_text())
assert d['checkpoint_before']==d['checkpoint_after']==parent['checkpoint_before']==parent['checkpoint_after']
assert d['native_sources_before']==d['native_sources_after']==parent['native_sources_before']==parent['native_sources_after']
for k,v in p['budget'].items():assert d[k]==v,(k,d[k],v)
assert len(d['ordinary_attempts'])==29 and len(d['native_attempt_costs'])==29 and len(d['paired_groups'])==28
assert all(x['completed'] for x in d['native_attempt_costs'])
assert [(r['dataset'],r['idx']) for r in d['records']]==[tuple(x) for x in p['selection']]
for row in d['records']:
    old=next(r for r in parent['records'] if (r['dataset'],r['idx'])==(row['dataset'],row['idx']))
    for k,v in row.items():assert v==old[k]
    assert hashlib.sha256(json.dumps(row['input_ids']).encode()).hexdigest()==row['input_ids_sha256']
for attempt in d['native_attempt_costs']:
    cost=attempt['cost'];batch=1 if attempt['capture_mode']=='finite' else int(attempt['capture_mode'][5:])
    assert cost['native_forwards']==1 and cost['native_forward_trajectories']==batch*2 and cost['vjps']==0
    assert cost['extra_replay_calls']==36 and cost['extra_replay_trajectories']==72*batch
    assert cost['native_decoder_layer_calls']==72 and cost['native_decoder_layer_trajectories']==144*batch
    assert cost['public_FA_activity']['auxiliary_attempts']==cost['public_FA_activity']['auxiliary_completed']==36
    assert len(cost['finite_FA_activity'])==36
assert sum(x['cost']['native_forward_trajectories'] for x in d['native_attempt_costs'])==104
assert sum(x['example_batch_size'] for x in d['ordinary_attempts'])==52
out={'status':'verified_complete','raw_sha256':sha(F/'results.json'),'protocol_sha256':sha(F/'protocol.json'),
    'scope':p['purpose'],'budget':p['budget'],'groups':[],'profiles':{}}
groups=[[0],[1],[2],[3],[0,1],[2,3],[0,1,2,3]]
for group in groups:
    runs=[r for r in d['paired_groups'] if r['indices']==group];assert len(runs)==4 and [r['repeat'] for r in runs]==list(range(4))
    lengths=[len(d['records'][i]['input_ids']) for i in group];plens=[d['records'][i]['prompt_len'] for i in group];n=max(lengths);batch=len(group)
    for r in runs:
        a=r['finite'];b=r['ordinary']
        assert b in d['ordinary_attempts'] and b['native_root_forwards']==b['native_vjps']==1
        assert b['example_batch_size']==b['physical_endpoint_batch_size']==batch and b['actual_lengths']==lengths and b['prompt_lengths']==plens and b['padded_length']==n
        assert b['backend']=='flash_attention_2' and not b['parameter_gradients_enabled'] and b['padding_gradient_zero']
        assert b['native_head_positions']==sorted({j for plen,length in zip(plens,lengths) for j in range(plen-1,length-1)})
        assert len(b['target_logprobs32'])==len(b['endpoint_scores32'])==batch
        for lp,g,plen,length in zip(b['target_logprobs32'],b['endpoint_scores32'],plens,lengths):
            assert len(lp)==length-plen and np.isfinite(lp).all() and abs(math.fsum(lp)-g)<1e-9
        assert abs(sum(b['endpoint_scores32'])-b['score32_sum64'])<1e-9
        vectors=np.asarray(a['signed_full_sequence']);vectors=vectors[None] if batch==1 else vectors
        assert vectors.shape==(batch,n) and np.isfinite(vectors).all()
        for k,i in enumerate(group):assert all(v==0 for j,v in enumerate(vectors[k]) if j not in set(d['records'][i]['eligible_positions']))
        if batch>1:
            assert a['example_batch_size']==batch and a['physical_endpoint_batch_size']==2*batch and a['actual_lengths']==lengths
        for c in a['public_FA_capture_checks']:
            assert c['public_output_exact_to_actual_model_FA'] and not c['private_FA_slots_read'] and not c['model_output_replaced'] and c['testing_matrix_numel']==0
        for call in a['finite_attention_activity']:
            assert call['calls_attempted']==call['calls_enqueued']==1
            for buf in call['buffer_contract']:assert buf['shape'] in [[batch,32,n,128],[batch,32,n]]
    measured=runs[1:];costs={mode:{'median_seconds':statistics.median(r[mode]['seconds'] for r in measured),'peak_bytes':max(r[mode]['peak_allocated_bytes'] for r in measured)} for mode in ['finite','ordinary']}
    actual=runs[1]['finite']['endpoint_scores32']['after'];actual=[actual] if batch==1 else actual
    out['groups'].append({'indices':group,'example_batch':batch,'actual_lengths':lengths,'costs':costs,
        'finite_peak_minus_same_batch_backward':costs['finite']['peak_bytes']-costs['ordinary']['peak_bytes'],
        'finite_seconds_over_ordinary':costs['finite']['median_seconds']/costs['ordinary']['median_seconds'],
        'ordinary_selected_head_B_minus_finite_full_head_2B_scores':(np.asarray(runs[1]['ordinary']['endpoint_scores32'])-np.asarray(actual)).tolist()})
for mode,profile in d['profiles'].items():
    path=F/profile['trace'];assert sha(path)==profile['sha256'];events=json.loads(path.read_text())['traceEvents'];names=[e['name'] for e in events if e.get('cat')=='kernel']
    fwd=sum('flash_fwd_kernel' in n for n in names);finite=sum('deltatrace_fa_finite_p1_kernel' in n for n in names);bwd=sum('flash_bwd' in n for n in names)
    assert fwd==profile['native_FA_forwards']==(36 if mode=='ordinary' else 108)
    assert finite==profile['finite_FA_kernels']==(0 if mode=='ordinary' else 108)
    assert bwd==profile['native_FA_backward_named_kernels'] and ((bwd>=36) if mode=='ordinary' else bwd==0)
    out['profiles'][mode]={'actual_default_FA_forward_kernels':fwd,'actual_native_FA_backward_named_kernels':bwd,'actual_finite_kernels':finite}
    del events,names
out['limits']='Memory references are actual same-example-batch native input backwards, frozen parameters and native selected head. Attribution uses 2B EOS/input endpoints and full head. Native arithmetic score differences retained. Ordinary-backward time is not FlashTrace time; no quality/sign confirmation or arbitrary batch-length extrapolation.'
(A/'vendor_fa_batch_memory_summary_20260907.json').write_text(json.dumps(out,indent=2))
numeric={k:v for k,v in d.items() if k in ['protocol','paired_groups','ordinary_attempts','native_attempt_costs','profiles']};numeric['raw_sha256']=out['raw_sha256']
(A/'vendor_fa_batch_memory_numeric_20260907.json').write_text(json.dumps(numeric,separators=(',',':')))
print(json.dumps(out,indent=2))

"""Freeze eight real whole-network calls after the compact-GQA operator passes."""
import ast,hashlib,json,subprocess,sys
from pathlib import Path
A=Path(__file__).resolve().parent;sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
verified=json.loads((A/'fa_compact_gqa_operator_summary_20260907.json').read_text())
assert verified['full_outputs_exact'] and verified['operator_calls']==6
source=(A/'vendor_fa_batch_memory_20260907.py').read_text()
source=source[:source.index('\nfrom native_backward_memory_reference_batched import')]
source=source.replace("p = json.loads((HERE/'protocol.json').read_text())", """p = json.loads((HERE/'protocol.json').read_text())
# Reuse already uploaded pinned dependencies to keep the launch argument bounded.
import shutil
for name,digest in p['sources'].items():
    if not (HERE/name).exists():
        origin=Path(p['runtime_source_parent'])/name
        assert hashlib.sha256(origin.read_bytes()).hexdigest()==digest
        shutil.copyfile(origin,HERE/name)
""",1)
source+='''
from qwen_signed_secant_paired_vendor_fa_gqa import propagate_paired_secant as compact_pair
from qwen_signed_secant_batched_vendor_fa_gqa_public import propagate_batch as compact_batch
from vendor_fa_finite_gqa_runtime import VendorFAFiniteP1CompactGQA
compact_extension=VendorFAFiniteP1CompactGQA(p['compact_library'],p['compact_library_sha256'])
old_extension=extension;old_pair=finite_entry;old_batch=propagate_batch
report.update(integration_groups=[],fresh_attributions=0,quality_queries=0)
save()
try:
    model.set_attn_implementation('flash_attention_2');native_method_audit(True)
    for group_index,selection in enumerate(p['integration_groups']):
        examples=[];rows=[]
        for dataset,index in selection:
            data=ROOT/f'exp/exp2/data/{dataset}.jsonl'
            assert hashlib.sha256(data.read_bytes()).hexdigest()==p['official_cache_sha256'][dataset]
            examples.append(runner.ds_utils.load_cached(data)[index])
            rows.append(next(r for r in parent['records'] if (r['dataset'],r['idx'])==(dataset,index)))
        group={'selection':selection,'example_batch_size':len(rows),
               'identities':[{k:r[k] for k in ['dataset','idx','input_ids_sha256','prompt_len','eligible_positions']} for r in rows],
               'runs':[],'comparisons':[]}
        report['integration_groups'].append(group)
        for repeat in range(2):
            pair={}
            for mode in (['old','compact'] if repeat==0 else ['compact','old']):
                assert report['fresh_attributions']<8
                report['active']=[group_index,repeat,mode];save();native_method_audit(True)
                extension=old_extension if mode=='old' else compact_extension
                finite_capture=partial(old_pair if mode=='old' else compact_pair,finite_attention=extension)
                propagate_batch=old_batch if mode=='old' else compact_batch
                def invoke():
                    if len(rows)==1:return strong_run(examples[0],rows[0],capture_mode='finite')
                    return batch_run(examples,rows)
                if group_index==1 and repeat==0 and mode=='compact':
                    with torch.profiler.profile(activities=list(torch.profiler.supported_activities())) as prof:
                        value=invoke()
                    trace=HERE/'compact_B4_profile.json';prof.export_chrome_trace(str(trace))
                    report['compact_B4_profile']={'file':trace.name,'sha256':hashlib.sha256(trace.read_bytes()).hexdigest()}
                    del prof
                else:value=invoke()
                report['fresh_attributions']+=1;pair[mode]=value
                group['runs'].append({'mode':mode,'repeat':repeat,'warm':repeat==0,'result':value})
                save();print('FA_GQA_INTEGRATION',len(rows),repeat,mode,value['seconds'],value['peak_allocated_bytes'],flush=True)
            a=torch.tensor(pair['old']['signed_full_sequence'],dtype=torch.float64)
            b=torch.tensor(pair['compact']['signed_full_sequence'],dtype=torch.float64)
            check={'repeat':repeat,'full_signed_exact':bool(torch.equal(a,b)),
                'max_abs_difference':float((a-b).abs().max()),'sign_flips':int(((a*b)<0).sum()),
                'endpoints_exact':pair['old']['endpoint_scores32']==pair['compact']['endpoint_scores32']}
            group['comparisons'].append(check);save()
            assert check['full_signed_exact'] and check['endpoints_exact']
    assert report['fresh_attributions']==report['native_root_forwards']==8
    assert report['native_vjps']==report['evaluation_forwards']==report['ft_attribution_forwards']==0
    assert report['extra_layer_replay_calls']==report['extra_native_fa_attention_calls']==report['finite_FA_calls_enqueued']==288
    report['native_sources_after']=native_sources();assert report['native_sources_before']==report['native_sources_after']
    native_method_audit(True)
    report['checkpoint_after']=checkpoint_receipt();assert report['checkpoint_before']==report['checkpoint_after']
    report['active']=None;report['status']='compact_GQA_B1_and_real_B4_exact'
except Exception:
    report['status']='failed';report['error']=traceback.format_exc();raise
finally:
    report['job_seconds']=time.time()-entry_started;save()
    with zipfile.ZipFile(HERE/'review_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in ['study.py','protocol.json','results.json',*p['sources']]+[x.name for x in HERE.glob('*_profile.json')]:z.write(HERE/name,name)
'''
ast.parse(source)
(A/'fa_compact_gqa_integration_20260907.py').write_text(source)
p=json.loads((A/'vendor_fa_batch_memory_protocol_20260907.json').read_text())
p.update(purpose='FA compact GQA integration: eight real attributions on original NI2 B1 and original NI0/3/6,MH1 B4, unchanged P1 versus compact-input FA P1, one warm+one measured each. No MLP caching, target-head change, VJP, FT sweep, generation or metric curve. Full-vector identity and actual source/FA batch are checked. No fixed memory rejection line.',
    study_sha256=sha(A/'fa_compact_gqa_integration_20260907.py'),
    wait_for_pid=161044,wait_for_script='${ARTIFACT_ROOT}/codex_fa_compact_gqa_operator_20260907_v1/study.py',
    maximum_queue_seconds=120,
    runtime_source_parent='${ARTIFACT_ROOT}/codex_vendor_fa_batch_memory_20260907_v1',
    compact_library='${ARTIFACT_ROOT}/codex_fa_compact_gqa_operator_20260907_v1/libdeltatrace_fa_finite_gqa.so',
    compact_library_sha256=verified['library_sha256'],
    compact_operator_raw_sha256=verified['raw_sha256'],
    integration_groups=[[['niah_mq_q2',2]],[['niah_mq_q2',0],['niah_mq_q2',3],['niah_mq_q2',6],['morehopqa',1]]],
    repeats='Per group: old/new warm pair then reversed measured pair;8 total actual attributions. New B4 warm includes profiler, excluded from measured latency.',
    budget={'native_root_forwards':8,'native_vjps':0,'quality_queries':0,'ft_attribution_forwards':0,
            'manual_passes':8,'extra_layer_replay_calls':288,'extra_native_fa_attention_calls':288,
            'finite_FA_calls_enqueued':288,'native_attribution_endpoint_trajectories':40},
    predeclared_review={'numerics':'Address-only FA change: full same-job vectors/endpoints must match; no historical curves inherited automatically.',
        'memory':'Matched old FA P1 and compact GQA peaks, no fixed memory excess rejection line.',
        'performance':'One measured pair per shape is provisional; actual FA B4 kernel dispatch profiled in its existing warm call.',
        'stop':'Stop on invalid source/model/finite behavior; no parameter sweep.'})
for name,digest in p['sources'].items():assert sha(A/name)==digest,name
new_sources=json.loads((A/'fa_compact_gqa_source_20260907.json').read_text())['sources']
p['sources'].update({n:v for n,v in new_sources.items() if n.endswith('.py')})
(A/'fa_compact_gqa_integration_protocol_20260907.json').write_text(json.dumps(p,indent=2))
cmd=[sys.executable,str(A/'prepare_remote_experiment.py'),'${ARTIFACT_ROOT}/codex_fa_compact_gqa_integration_20260907_v1',
    'study.py='+str(A/'fa_compact_gqa_integration_20260907.py'),
    'protocol.json='+str(A/'fa_compact_gqa_integration_protocol_20260907.json')]
cmd += [n+'='+str(A/n) for n in new_sources if n.endswith('.py')]
cmd += ['--request',str(A/'launch_fa_compact_gqa_integration_20260907.json')]
subprocess.run(cmd,check=True)
request=json.loads((A/'launch_fa_compact_gqa_integration_20260907.json').read_text())
assert len(request['cmd'].encode())<100000

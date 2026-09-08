"""Freeze one whole-response FT target diagnostic with unchanged native helpers."""
import ast,base64,hashlib,json,shlex,zlib
from pathlib import Path
A=Path(__file__).resolve().parent;R=A.parent/'DeltaTrace';sha=lambda b:hashlib.sha256(b).hexdigest()
D=A/'snapshot${ARTIFACT_ROOT}/codex_qwen35_FT32_resume_20260908_v1'
p=json.loads((D/'protocol.json').read_bytes());original=(D/'study.py').read_bytes();assert sha(original)==p['files_sha256']['study.py']
s=original.decode()
def change(before,after):
    global s
    assert s.count(before)==1,before[:100]
    s=s.replace(before,after,1)
change('Resume FT from immutable native inputs; restore author\'s causal source bound.',
       'One FT whole-response target diagnostic from saved native inputs, without changing FT helpers.')
change('import ast,hashlib,importlib.util,io,json,sys,time,traceback,zipfile,types',
       'import ast,hashlib,importlib.util,io,json,sys,time,traceback,zipfile,types,math\nfrom typing import Sequence,List')
a=s.index("    oldfile=parent/'hop0_FT_scores.npz'");b=s.index('        rows={};',a)
s=s[:a]+"""    lengths=[605,368];starts=[357,227]
    layout=RightPaddedBatch(lengths,605,'cuda')
    for hop in range(1):
        weights=torch.zeros((2,605),dtype=torch.float32,device='cuda')
        for b,(begin_idx,end_idx) in enumerate(zip(starts,lengths)):weights[b,begin_idx:end_idx]=1.0
        limits=lengths
        scores_by_layer=torch.zeros((32,2,605),dtype=torch.float32)
"""+s[b:]
change('for i in ([31] if hop==0 else range(32)):', 'for i in range(32):')
change("float(components[b,end:].abs().max()) for b,end in enumerate(limits)",
       "(float(components[b,end:].abs().max()) if end<components.shape[1] else 0.0) for b,end in enumerate(limits)")
change("diagnostics=(hop,i) in [(0,31),(1,1)];r['aggregate_calls']+=1",
       "diagnostics=False;r['aggregate_calls']+=1")
a=s.index('                if diagnostics:\n');b=s.index('                scores_by_layer[i]=scores;',a)
s=s[:a]+s[b:]
a=s.index('        result=controller.consume(total);');b=s.index("    r['sources_after']=sources()",a)
s=s[:a]+"""        r['hops'][str(hop)]['status']='complete'
        persist('whole_response_FT_diagnostic',{'layer_token_scores':scores_by_layer,'token_total':total,'weights':weights})
        r['completed_hop_outputs']+=1;save()
    assert r['content_calls']==r['aggregate_calls']==r['parameter_cache_loads']==32 and r['completed_hop_outputs']==1
    raw=Path(p['author_metric_source']).read_bytes().replace(b'\\r\\n',b'\\n');assert sha(raw)==p['author_metric_source_sha256']
    helper=next(n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name=='evaluate_attr_recovery_skip_tokens')
    exec(compile(ast.Module(body=[helper],type_ignores=[]),p['author_metric_source'],'exec'))
    raw=Path(p['spans_file']).read_bytes();assert sha(raw)==p['spans_sha256'];case=json.loads(raw)['cases'][0]
    user=case['input_metadata']['author_user_positions'];keep=case['mapping']['keep_local_indices'];gold=case['mapping']['gold_user_token_indices']
    prompt=total[0,user];r['original_recovery_calls']=1
    score=evaluate_attr_recovery_skip_tokens(prompt[None,:],keep_prompt_token_indices=keep,gold_prompt_token_indices=gold,top_fraction=.1)
    values=prompt.clamp(min=0)[keep];top=torch.topk(values,math.ceil(.1*len(keep))).indices.tolist();selected=[keep[j] for j in top]
    eligible=set(gold)&set(keep);hits=sorted(set(selected)&eligible);assert score==len(hits)/len(eligible)
    r['original_NI0_recovery']={'recovery':score,'selected_user_indices':selected,'hit_user_indices':hits,
        'eligible_gold_count':len(eligible),'selected_count':len(selected),'cutoff_ties':int((values==values[top[-1]]).sum()),
        'method':'FT_whole_response_target_diagnostic_only','metric':'Unchanged original helper,top_fraction0.1; not a new official FT baseline.'}
"""+s[b:]
change("r['status']='corrected_FT32_hops0_to3_resumed_with_author_source_bounds'",
       "r['status']='one_FT_whole_response_target_diagnostic_executed'")
ast.parse(s);(A/'qwen35_FT_target_control_20260908.py').write_text(s,encoding='utf-8',newline='\n')
for key in ['previous_boundary','correction','selective_recovery','additional_paid_work']:
    p.pop(key,None)
p.update(purpose='Diagnose whether answer-only target concentration also explains FT NI0 failure. One full-response aggregate using unchanged corrected-native FT helpers. Keep original answer FT0-3 as immutable primary comparators.',
    target='Uniform representation sinks at all fixed response tokens including EOS: NI0 absolute357..604 and MH1 absolute227..367. One aggregate, no thinking hop or tuned weighting. Full-input token_total saved; original recovery selects user/eligible only.',
    inference_boundary='This is a target diagnostic, not official FT0, not a retuned baseline and not a quality-superiority claim. Original official data,gold,eligibility and recovery helper unchanged. No propagation rule or scalar proximity altered.',
    study_parent_sha256=sha(original),
    budget={'meta_model_constructions':1,'full_model_loads':0,'root_forwards':0,'decoder_replays':0,'checkpoint_tensor_loads':0,
        'actual_layer_cache_loads':32,'cached_parameter_loads':32,'public_FA_auxiliary_forwards':8,'public_FA_value_backwards':8,
        'public_FLA_auxiliary_forwards':24,'public_FLA_value_backwards':24,'public_linear_conv_forwards':24,'public_linear_conv_backwards':24,
        'content_calls':32,'aggregate_calls':32,'completed_hop_outputs':1,'original_recovery_CPU_calls':1,'generation_calls':0,'deletion_metric_queries':0,'new_samples':0},
    stop='One target-control pass, no automatic retry, target mixtures, layer selection or weight tuning. Stop on execution/nonfinite/source errors and preserve partial results. No default FT0-3 rerun or 97-forward metric expansion.',
    cost_scope='All32 existing actual-input caches loaded and each native auxiliary forward/backward paid. No new decoder/root or checkpoint tensor load. Includes cold native kernels,IO,diagnostics and transfers; not production warm whole-method speed.',
    author_metric_source='${FLASHTRACE_ROOT}/ft_ifr_improve.py',author_metric_source_sha256='583f4b7d0426407eb9a517f173365762860a1f4382f472dffb5c07de7d3e94a1',
    spans_file='${ARTIFACT_ROOT}/codex_qwen35_official_spans_20260908_v1/results.json',spans_sha256='37a0d8c4ef4d174ebabd346087be303e895a0eaa78ca9231f2a6ba3195890b6a')
files={'study.py':s.encode()}
for name,digest in p['files_sha256'].items():
    if name=='study.py':continue
    raw=(D/name).read_bytes();assert sha(raw)==digest;files[name]=raw
    current=R/('research/third_party/flashtrace_qwen35_e81b3be/flashtrace/core.py' if name=='official_ft_core.py' else 'research/runtime/'+name)
    assert sha(current.read_bytes())==digest,name
p['unchanged_helper_sha256']={k:sha(v) for k,v in files.items() if k!='study.py'}
p['files_sha256']={k:sha(v) for k,v in files.items()};files['protocol.json']=json.dumps(p,indent=2).encode();(A/'qwen35_FT_target_control_protocol_20260908.json').write_bytes(files['protocol.json'])
packed=base64.b64encode(zlib.compress(json.dumps({k:base64.b64encode(v).decode() for k,v in files.items()}).encode())).decode();python='${ARTIFACT_ROOT}/codex_qwen35_isolated_import_20260908_v1/env/bin/python'
loader=('import base64,pathlib,subprocess,json,zlib;d=pathlib.Path("${ARTIFACT_ROOT}/codex_qwen35_FT_target_control_20260908_v1");d.mkdir(exist_ok=False);'
    'files=json.loads(zlib.decompress(base64.b64decode('+repr(packed)+')));[(d/k).write_bytes(base64.b64decode(v)) for k,v in files.items()];'
    'f=(d/"driver.log").open("w");j=subprocess.Popen(['+repr(python)+',str(d/"study.py")],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);'
    '(d/"pid").write_text(str(j.pid));print(json.dumps({"pid":j.pid,"directory":str(d)}))')
(A/'launch_qwen35_FT_target_control_20260908.json').write_text(json.dumps({'cmd':python+' -c '+shlex.quote(loader),'timeout':10}),encoding='utf-8')
print(json.dumps({'protocol_sha256':sha(files['protocol.json']),'study_sha256':p['files_sha256']['study.py'],'budget':p['budget']}))

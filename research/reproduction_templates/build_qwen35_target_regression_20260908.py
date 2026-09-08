"""One target-only regression control; preserve the previous finite rules exactly."""
import ast
import base64
import hashlib
import json
import shlex
import zlib
from pathlib import Path

A = Path(__file__).resolve().parent
R = A.parent / 'DeltaTrace'
sha = lambda b: hashlib.sha256(b).hexdigest()
old_dir = A / 'snapshot${ARTIFACT_ROOT}/codex_qwen35_whole_finite_20260908_v1'
old = json.loads((old_dir / 'protocol.json').read_bytes())
source = (old_dir / 'study.py').read_bytes()
assert sha(source) == old['files_sha256']['study.py']
s = source.decode()


def change(before, after):
    global s
    assert s.count(before) == 1, before[:100]
    s = s.replace(before, after, 1)


change('One explicit author-answer finite pass over32 original saved root inputs.',
       'One whole-response target regression over the same32 saved native root inputs.')
change('Engineering integration only. Original decoder replay and capture are paid;\nno shadow model, new root forward, generation or benchmark evaluation.',
       'The finite rules, endpoints and precision are unchanged. Only target selection\nreturns to the pre-migration whole fixed response. One original CPU NI0 recovery\nis computed after score persistence. No new root forward, generation or FT run.')
change('import ast,hashlib,io,json,sys,time,traceback,zipfile',
       'import ast,hashlib,io,json,sys,time,traceback,zipfile,math\nfrom typing import Sequence,List')
change("lo,hi=row['mapping']['new_sink_span'];selected=list(range(lo,hi+1));cases.append(case);offsets.append(selected)",
       "lo,hi=row['mapping']['new_sink_span'];selected=list(range(len(case['target_ids'])));cases.append(case);offsets.append(selected)")
change("'sink_closed':[lo,hi]", "'author_sink_closed_reference_only':[lo,hi],'scope':'whole_fixed_response_including_EOS'")
change('assert selection.counts==[38,21]', 'assert selection.counts==[248,141]')
change("x=root['final_norm_input'].to('cuda').detach().requires_grad_(True)",
       "x=root['final_norm_input'].to('cuda').detach()")
start = s.index("    r['head_norm_backwards']+=1")
end = s.index("    r['finite_answer_calls']+=1", start)
s = s[:start] + '    z=z.detach();del y,x,logp\n' + s[end:]
start = s.index("    r['finite_answer_calls']+=1\n    eqnorm")
end = s.index("    head.to('meta');norm.to('meta')", start)
s = s[:start] + """    persist('target_seed_review',{'finite':m,'input0':xnorm[0::2],'input1':xnorm[1::2],
        'logp0':head_diag['logp0'],'logp1':head_diag['logp1'],'logit_effect':head_diag['allocated_logit_effect']})
    archive('finite_final_layer_seed',m.cpu())
    del z,mnorm,head_diag,xnorm,zero
""" + s[end:]
s = s.replace('original_root_answer_', 'original_root_target_').replace('replayed_answer_', 'replayed_target_')
s = s.replace("'actual_answer_operands'", "'actual_target_operands'").replace('original_final_norm_and_packed_answer_head', 'original_final_norm_and_packed_whole_target_head')
change("r['whole_model_attributions']=1;r['status']='one_author_answer_32_layer_finite_propagation_executed'",
       "r['whole_model_attributions']=1;r['status']='one_whole_response_target_regression_executed'")
anchor = "    r['compiler_benchmark_observations']=dict(benchmark_counts)"
cpu = """    # The new score is persisted above before evaluating the unchanged gold.
    author_path=Path(p['author_metric_source']);raw=author_path.read_bytes().replace(b'\\r\\n',b'\\n')
    assert sha(raw)==p['author_metric_source_sha256']
    helper=next(n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name=='evaluate_attr_recovery_skip_tokens')
    exec(compile(ast.Module(body=[helper],type_ignores=[]),str(author_path),'exec'))
    case=spans['cases'][0];user=case['input_metadata']['author_user_positions'];keep=case['mapping']['keep_local_indices'];gold=case['mapping']['gold_user_token_indices']
    prompt=signed[0].detach().float().cpu()[user];r['original_recovery_calls']=1
    score=evaluate_attr_recovery_skip_tokens(prompt[None,:],keep_prompt_token_indices=keep,gold_prompt_token_indices=gold,top_fraction=.1)
    values=prompt.clamp(min=0)[keep];top=torch.topk(values,math.ceil(.1*len(keep))).indices.tolist();selected=[keep[j] for j in top]
    eligible_gold=set(gold)&set(keep);hits=sorted(set(selected)&eligible_gold);assert score==len(hits)/len(eligible_gold)
    r['original_NI0_recovery']={'recovery':score,'selected_user_indices':selected,'hit_user_indices':hits,'eligible_gold_count':len(eligible_gold),
        'selected_count':len(selected),'negative_eligible_count':int((prompt[keep]<0).sum()),'cutoff_ties':int((values==values[top[-1]]).sum()),
        'method':'P1_whole_response_regression_control','metric':'Unchanged author recovery, positive-clamp,top_fraction0.1; signed vector saved unchanged.'}
    assert r['head_norm_forwards']==1 and r['head_norm_backwards']==0 and r['finite_answer_calls']==1
"""
change(anchor, cpu + anchor)
ast.parse(s)
(A / 'qwen35_target_regression_20260908.py').write_text(s, encoding='utf-8', newline='\n')

p = dict(old)
p.update(
    purpose='Diagnose the DT migration target change with one same-endpoint B2 whole-response regression. Not a new tuned method or a successful adaptation claim.',
    target='Whole unchanged author fixed response including tokenizer EOS: NI0 offsets0..247 and MH1 offsets0..140. Same native BF16 logits, existing FP32 finite logsoftmax seed and full248320 vocabulary.',
    target_scope_change='Restore the old8B P1 whole-response scope for a within9B diagnostic. Keep the previously frozen answer-only DT and corrected-native FT0-3 results. A recovered score diagnoses the migration; it does not replace the official answer-evidence objective or establish cross-dataset superiority.',
    regression_parent_directory='${ARTIFACT_ROOT}/codex_qwen35_whole_finite_20260908_v1',
    regression_parent_results_sha256=sha((old_dir / 'results.json').read_bytes()),
    original_study_sha256=sha(source),
    budget={'meta_model_constructions':1,'full_model_loads':0,'root_forwards':0,'decoder_weight_loads':32,
            'original_decoder_replays':30,'original_saved_decoder_reuses':2,'original_head_weight_loads':1,'original_norm_weight_loads':1,
            'original_head_norm_forward':1,'original_head_norm_backward':0,'finite_answer_seed_calls':1,'finite_final_norm_calls':1,
            'complete_finite_decoder_calls':32,'finite_FA_calls':8,'finite_FA_kernel_launches':24,'finite_FLA_calls':24,
            'new_public_auxiliary_FA_forwards':7,'saved_public_LSE_reuses':1,'native_auxiliary_linear_conv_forwards':24,
            'native_auxiliary_linear_conv_backwards':24,'whole_model_attributions':1,'original_recovery_CPU_calls':1,
            'generation_calls':0,'deletion_metric_queries':0,'FT_reruns':0,'new_samples':0},
    batching='Same605/368 paired B4 native modules and B2 finite propagation.389 selected real target rows packed in native output head; no padding labels.',
    seed_validation='Reuse validated finite head/norm rules unchanged. Retain comparison with saved original root logprobs and finite-effect ledger. No repeated native-gradient/equal-endpoint screen.',
    stop='Exactly one whole-target B2 pass and one original NI0 CPU metric. No automatic retry, target mixture, gold-based token weighting, precision scan, FT replay or extra samples. Preserve partial outputs on failure.',
    next='If recovery restores substantially, classify the target migration as consequential and restore a clear method definition before expanding original metrics. If it remains poor, inspect propagation/architecture using saved boundaries. Neither outcome proves successful overall adaptation.',
    author_metric_source='${FLASHTRACE_ROOT}/ft_ifr_improve.py',
    author_metric_source_sha256='583f4b7d0426407eb9a517f173365762860a1f4382f472dffb5c07de7d3e94a1',
    cost_scope='Paid partial native weight loads,30 actual decoder replays,2 saved captures,one native head/norm forward,32 finite decoders,auxiliary kernels and IO. Not a fair warm full-model speed comparison. Previous root/capture costs are reused and recorded, not newly free.',
)
files = {'study.py': s.encode()}
for name, digest in old['files_sha256'].items():
    if name == 'study.py': continue
    raw = (old_dir / name).read_bytes()
    assert sha(raw) == digest
    files[name] = raw
    # The actual runtime implementation must still be the validated one.
    current = R / ('core' if name in ['signed_secant_rules.py','compiled_swiglu_secant.py','compiled_finite_rules.py','compiled_logprob_seed.py'] else 'research/runtime') / name
    assert sha(current.read_bytes()) == digest, name
p['unchanged_runtime_sha256'] = {k: sha(v) for k,v in files.items() if k != 'study.py'}
p['files_sha256'] = {k: sha(v) for k,v in files.items()}
files['protocol.json'] = json.dumps(p, indent=2).encode()
(A / 'qwen35_target_regression_protocol_20260908.json').write_bytes(files['protocol.json'])
packed = base64.b64encode(zlib.compress(json.dumps({k:base64.b64encode(v).decode() for k,v in files.items()}).encode())).decode()
python = '${ARTIFACT_ROOT}/codex_qwen35_isolated_import_20260908_v1/env/bin/python'
loader = ('import base64,pathlib,subprocess,json,zlib;d=pathlib.Path("${ARTIFACT_ROOT}/codex_qwen35_target_regression_20260908_v1");d.mkdir(exist_ok=False);'
          'files=json.loads(zlib.decompress(base64.b64decode('+repr(packed)+')));[(d/k).write_bytes(base64.b64decode(v)) for k,v in files.items()];'
          'f=(d/"driver.log").open("w");j=subprocess.Popen(['+repr(python)+',str(d/"study.py")],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);'
          '(d/"pid").write_text(str(j.pid));print(json.dumps({"pid":j.pid,"directory":str(d)}))')
(A / 'launch_qwen35_target_regression_20260908.json').write_text(json.dumps({'cmd':python+' -c '+shlex.quote(loader),'timeout':10}), encoding='utf-8')
print(json.dumps({'protocol_sha256':sha(files['protocol.json']),'study_sha256':p['files_sha256']['study.py'],
                  'unchanged_runtime_files':len(p['unchanged_runtime_sha256']),'budget':p['budget']}))

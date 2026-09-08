"""Freeze explicit layer0 FLA endpoint-average pilot; never launch it here."""
import ast,base64,hashlib,json,shlex,zlib
from pathlib import Path
A=Path(__file__).resolve().parent;R=A.parent/'DeltaTrace';sha=lambda b:hashlib.sha256(b).hexdigest()
donor=A/'snapshot${ARTIFACT_ROOT}/codex_dt_normgate_production_20260908_v1'
dp=json.loads((donor/'protocol.json').read_bytes())
baseline=A/'snapshot${ARTIFACT_ROOT}/codex_dt_clean_secant_whole_pilot_20260908_v1'
bp=json.loads((baseline/'protocol.json').read_bytes())
assert json.loads((baseline/'results.json').read_bytes())['status']=='all8FA_clean_secant_4DT84score_pilot_complete'
templates=R/'research/reproduction_templates'
utility=templates/'normal_finite_study_utils_20260908.py'
assert sha(utility.read_bytes())==bp['files_sha256'][utility.name]
files={'study.py':(templates/'dt_layer0_fla_endpoint_average_whole_pilot_20260908.py').read_bytes(),utility.name:utility.read_bytes()}
runner_name='qwen35_dense_finite_runner.py'
for name,want in dp['files_sha256'].items():
    if name=='study.py':continue
    matches=[root/name for root in (R/'research/runtime',R/'core') if (root/name).is_file()]
    assert len(matches)==1,(name,matches)
    files[name]=matches[0].read_bytes()
    if name!=runner_name:assert sha(files[name])==want,('Changed existing runtime',name)
    assert sha((donor/name).read_bytes())==want,('Parent snapshot mismatch',name)
# Prove the sole existing-source change is the explicit optional FLA API.
old_runner=(donor/runner_name).read_text(encoding='utf-8')
expected=old_runner.replace('def __init__(self,model,finite_fa,finite_fla,*,norm_gate_rules=None):',
    'def __init__(self,model,finite_fa,finite_fla,*,norm_gate_rules=None,finite_fla_by_layer=None):')
expected=expected.replace('        from replaying a second candidate or changing the native forward.\n',
    '        from replaying a second candidate or changing the native forward.\n        Optional finite_fla_by_layer explicitly replaces selected GDN callbacks;\n        the empty default uses the original finite_fla at every GDN layer.\n')
expected=expected.replace('        self.model=model;self.finite_fa=finite_fa;self.finite_fla=finite_fla\n',
'''        fla_rules={} if finite_fla_by_layer is None else finite_fla_by_layer
        if not isinstance(fla_rules,Mapping):
            raise TypeError('finite_fla_by_layer must map integer GDN layer indices to callable backends.')
        self.finite_fla_by_layer=dict(fla_rules)
        for index,backend in self.finite_fla_by_layer.items():
            layers=model.model.language_model.layers
            if isinstance(index,bool) or not isinstance(index,int) or not 0<=index<len(layers) or layers[index].block_type!='linear_attention':
                raise ValueError(f'finite_fla_by_layer requires an existing GDN layer: {index!r}')
            if not callable(backend):raise TypeError('A finite FLA backend must be callable.')
        self.model=model;self.finite_fa=finite_fa;self.finite_fla=finite_fla
''')
expected=expected.replace('                if i in self.norm_gate_rules:\n','                finite_fla=self.finite_fla_by_layer.get(i,self.finite_fla)\n                if i in self.norm_gate_rules:\n')
expected=expected.replace('upstream,scale,self.finite_fla,focused','upstream,scale,finite_fla,focused')
expected=expected.replace("              'norm_gate_rules':{str(i):rule for i,rule in sorted(self.norm_gate_rules.items())},",
    "              'norm_gate_rules':{str(i):rule for i,rule in sorted(self.norm_gate_rules.items())},\n              'finite_fla_by_layer':sorted(self.finite_fla_by_layer),")
assert files[runner_name].decode()==expected,'Unexpected runner change beyond the explicit API.'
wrapper='layer0_fla_endpoint_average_20260908.py'
files[wrapper]=(R/'research/runtime'/wrapper).read_bytes()
diagnostic=templates/'finite_fla_endpoint_order_diagnostic_20260908.py'
def contract(raw):
    result={}
    for node in ast.parse(raw).body:
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
            name=node.targets[0].id
            if name in ('ENDPOINT_KEYS','COEFFICIENT_KEYS'):
                value=node.value.args[0] if isinstance(node.value,ast.Call) else node.value
                result[name]=ast.literal_eval(value)
    return result
assert contract(files[wrapper])==contract(diagnostic.read_bytes())
for name,data in files.items():ast.parse(data,filename=name)
remote='${ARTIFACT_ROOT}/codex_dt_layer0_fla_endpoint_average_whole_pilot_20260908_v1'
python='${ARTIFACT_ROOT}/codex_qwen35_isolated_import_20260908_v1/env/bin/python'
keys=('checkpoint','checkpoint_config_tokenizer_sha256','native_model_sha256','isolated_site','compiler_cache',
      'boundary_compiler_cache','installed_FA_interface_sha256','native_stage_source_sha256','official_root',
      'official_source_blob_sha1','dependency_overlays','runtime_source_sha256','expected_weight_stats','cache_paths',
      'cache_hashes','finite_FA_library','finite_FA_library_sha256','paired_references')
p={key:dp[key] for key in keys}
p.update({
    'scope':'Bounded whole-input NI1/MH1 pilot. Control=current layer0 symmetric norm gate. Candidate changes only the finite FLA at layer0 by averaging original and pair-swapped endpoint pullbacks, explicitly via finite_fla_by_layer={0:wrapper}. No launch authorization is implied by this frozen payload.',
    'case_indices':[['morehopqa',1],['niah_mq_q2',1]],
    'expected_FA_layers':[3,7,11,15,19,23,27,31],
    'expected_GDN_reverse_order':[30,29,28,26,25,24,22,21,20,18,17,16,14,13,12,10,9,8,6,5,4,2,1,0],
    'call_schedule':[['morehopqa_1','control'],['morehopqa_1','candidate'],['niah_mq_q2_1','candidate'],['niah_mq_q2_1','control']],
    'rule_scope':{'control':'Original finite FA and finite FLA; norm_gate_rules={0:symmetric}; finite_fla_by_layer={}.',
        'candidate':'Same normal runner and norm_gate_rules; only finite_fla_by_layer={0:Layer0FLAEndpointAverage(shared_original_FLA)}. Other23GDN and all8FA use unchanged backends.'},
    'endpoint_rule':{'fields':list(contract(files[wrapper])['ENDPOINT_KEYS']),
        'coefficient_fields':list(contract(files[wrapper])['COEFFICIENT_KEYS']),
        'operation':'Exchange adjacent endpoint rows of all11native fields including h; same actual native BF16 do object and scale; unchanged backend for both orientations; arithmetic mean of six FP32 coefficients without sign reversal. alpha/g are alternative decay coordinates; normal GDN consumes g once.',
        'scope':'No endpoint reconstruction, extra GDN/conv replay, new model forward, shadow native op, GEMM precision change, profiler, frame introspection, callback-count dispatch or diagnostic synchronization. Wrapper accepts paired B; this pilot validates only one sample/B2.'},
    'existing_runtime_change':{'file':runner_name,'parent_sha256':sha((donor/runner_name).read_bytes()),
        'current_sha256':sha(files[runner_name]),'proof':'Builder asserts exact allowed source transformation: optional validated mapping, explicit current mixer index selection and info field. All finite mathematics/native calls remain unchanged; empty mapping is computationally original.'},
    'diagnostic_contract_provenance':{'file':str(diagnostic.relative_to(R)),'sha256':sha(diagnostic.read_bytes()),
        'reuse':'Exact ENDPOINT_KEYS/COEFFICIENT_KEYS match; no paired diagnostic repropagation or synchronization in production wrapper.'},
    'parent_control_anchor':{'results_path':'/tmp/'+baseline.name+'/results.json','results_sha256':sha((baseline/'results.json').read_bytes()),
        'scope':'Input/source context only. Both control and candidate receive fresh native21point curves; no historical score substitution or cross-run matched FT claim.'},
    'protected_sources':[{'path':dp['finite_FA_library'],'sha256':dp['finite_FA_library_sha256']},
        {'path':'/tmp/'+baseline.name+'/results.json','sha256':sha((baseline/'results.json').read_bytes())},
        {'path':'/tmp/'+baseline.name+'/protocol.json','sha256':sha((baseline/'protocol.json').read_bytes())}],
    'utility_provenance':bp['utility_provenance'],
    'budget':{'model_loads':1,'native_eager_NI0_B1_initializations':1,'complete_DT_attributions':4,
        'DT_B2_roots':4,'native_decoder_replays':128,'finite_decoder_calls':128,'public_FA_LSE_calls':32,
        'finite_FA_calls':32,'finite_GDN_callback_sites':96,'finite_FLA_backend_calls':98,'native_FLA_adjoint_stage_calls':196,
        'candidate_layer0_average_wrapper_calls':2,'candidate_additional_FLA_backend_calls':2,
        'finite_GDN_conv_preactivation_calls':96,'finite_GDN_conv_autograd_calls':96,
        'original_metric_curves':4,'native_B1_scorer_forwards':84,'FT_calls':0,'generation_calls':0,
        'new_samples':0,'repeat_timing_runs':0,'wall_time_seconds':900},
    'evaluation':bp['evaluation'],
    'signed_outputs':bp['signed_outputs'],
    'fixed_control_masks':bp['fixed_control_masks'],
    'cost':'One measured attribution per method/case, opposite ordering acrosscases. Preserve normal-runner root/stage timings, peak allocated/reserved memory, compilation/initialization and all wrapper cost. Candidate has25FLA backend calls per attribution versus24control; both have24GDN helpers and unchanged conv work. No paired-diagnostic cost substitution, clean/warm speed estimate, repeat run or speed guarantee. Partial failures retain entered/returned counts, elapsed wall time and unknown in-flight native stage status.',
    'risks':'Endpoint-order averaging removes that rule orientation but does not guarantee arbitrary conditional-deletion accuracy, needle,RISE,MAS or whole-network improvement. NI early FLA error previously compensates positive MLP error. Existing endpoint conservation and small rounding errors do not prove conditional fidelity. MH is a fixed second case, not a holdout generalization claim.',
    'decision':'Root reviews complete internal ledger before authorizing launch. This frozen pilot does not launch automatically. If run, assess both original metrics and fixed control-mask errors/cost; report all positive and negative results. No followup candidates, layer search or retries within this budget.',
    'stop':'First nonfinite/source/layout/input/ABI/count/metric invariant failure or900seconds. Preserve partial evidence and all known cost; do not retry, fall back, change precision/targets/scorers or call FT.',
    'files_sha256':{name:sha(data) for name,data in files.items()}})
files['protocol.json']=json.dumps(p,indent=2).encode()
pp=A/'dt_layer0_fla_endpoint_average_whole_pilot_protocol_20260908.json'
lp=A/'launch_dt_layer0_fla_endpoint_average_whole_pilot_20260908.json'
assert not pp.exists() and not lp.exists(),'Do not overwrite frozen protocol or payload.'
blob=base64.b64encode(zlib.compress(json.dumps({name:base64.b64encode(data).decode() for name,data in files.items()}).encode())).decode()
loader=('import pathlib,subprocess,json,base64,zlib;d=pathlib.Path('+repr(remote)+');d.mkdir(exist_ok=False);'
    'files=json.loads(zlib.decompress(base64.b64decode('+repr(blob)+')));[(d/k).write_bytes(base64.b64decode(v)) for k,v in files.items()];'
    'f=(d/"driver.log").open("w");j=subprocess.Popen(['+repr(python)+',"-B",str(d/"study.py")],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);'
    '(d/"pid").write_text(str(j.pid));print(json.dumps({"pid":j.pid,"directory":str(d)}))')
pp.write_bytes(files['protocol.json']);lp.write_text(json.dumps({'cmd':python+' -c '+shlex.quote(loader),'timeout':10}))
print(json.dumps({'protocol_sha256':sha(files['protocol.json']),'study_sha256':p['files_sha256']['study.py'],
    'runner_sha256':p['files_sha256'][runner_name],'wrapper_sha256':p['files_sha256'][wrapper],
    'payload':str(lp),'protocol':str(pp),'remote_directory':remote,'budget':p['budget']}))

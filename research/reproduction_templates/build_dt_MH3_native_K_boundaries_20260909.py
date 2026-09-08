"""Freeze one current MH3 attribution plus four original scores with passive K capture."""
import ast,base64,copy,hashlib,json,shlex,zlib
from pathlib import Path
import numpy as np
A=Path(__file__).resolve().parent;R=A.parent/'DeltaTrace';sha=lambda b:hashlib.sha256(b).hexdigest()
D=A/'snapshot${ARTIFACT_ROOT}/codex_dt_GDN1_K_remaining_20260909_s2_v1'
dp,dr,receipt=[json.loads((D/f).read_bytes()) for f in ['protocol.json','results.json','terminal_receipt.json']]
assert receipt['proc_exists'] is False and dr['DT_returned']==4 and dr['scorer_returned']==84
for name,row in receipt['files'].items():assert sha((D/name).read_bytes())==row['sha256']
p=copy.deepcopy(json.loads((A/'dt_MH2_current_conditional_boundaries_protocol_20260909.json').read_bytes()))
source=R/'research/reproduction_templates/dt_MH2_current_conditional_boundaries_20260909.py'
s=source.read_text().replace('MH2','MH3').replace('morehopqa_2','morehopqa_3')
def replace(old,new):
    global s
    assert s.count(old)==1,old[:100]
    s=s.replace(old,new)
replace("self.current=None", "self.current=None;self.k={'B1':{}}")
replace('    def boundary(self,name,m,x):', '''    def wants_decoder(self,index):return index==1
    def decoder(self,index,d,c,e,upstream,new,terms):
        assert index==1 and set(self.k)=={'B1'}
        self.k.update({name:value.detach().to('cpu',copy=True) for name,value in
            {'raw_k':c['raw_k'],'k':e['k'],'coeff':terms['mixer']['coeff']['k'],'mk':terms['mixer']['mk']}.items()})
    def boundary(self,name,m,x):''')
replace('    import causal_conv1d','    import causal_conv1d\n    from mh3_native_k_diagnostic_20260909 import NativeFreshKCapture,analyze')
old="""            with torch.no_grad():lp=timed('original_scorer_with_passive_boundaries_step'+str(step),lambda:evaluator.compute_logprob_response_given_prompt(
                prompt[None].to('cuda'),case['target'][None].to('cuda')))"""
new="""            with NativeFreshKCapture(layers[1].linear_attn) as captured_k,torch.no_grad():
                lp=timed('original_scorer_with_passive_boundaries_step'+str(step),lambda:evaluator.compute_logprob_response_given_prompt(
                    prompt[None].to('cuda'),case['target'][None].to('cuda')))
            assert not captured_k.active and captured_k.calls=={'module':1,'conv':1,'FLA':1,'stage':1}
            observer.k['B1'][str(step)]={'raw_k':captured_k.values['raw_k'],'k':captured_k.endpoints['k']}
            row.setdefault('native_K_capture_receipts',{})[str(step)]={'calls':captured_k.calls,'cache':captured_k.cache_receipt}
            del captured_k"""
replace(old,new)
replace("    private={'coefficients':observer.coeff,'paired_native_boundaries':observer.paired,\n        'B1_native_boundaries':observer.activations,'source_protocol_sha256':sha((A/'protocol.json').read_bytes())}","""    row['K_diagnostic'],kv=timed('CPU_K_conditional_geometry',lambda:analyze(observer.k,info['prompt_length']))
    vectors.update(kv);del kv
    private={'K':observer.k,'coefficients':{n:observer.coeff[n] for n in ['0','1','2']},
        'paired_native_boundaries':{n:observer.paired[n] for n in ['0','1','2']},
        'B1_native_boundaries':{step:{n:values[n] for n in ['0','1','2']} for step,values in observer.activations.items()},
        'source_protocol_sha256':sha((A/'protocol.json').read_bytes())}""")
replace("'MH3_current_conditional_boundaries_1DT4score_complete'","'MH3_native_K_boundaries_1DT4score_complete'")
replace('One current DT with passive34boundary coefficient copies;', 'One current DT with passive34boundary coefficient copies and focused GDN1 K diagnostics;')
replace('No new MAS curve,FT,generation,newcandidate,extraGDN/conv or precision repair.', 'No new MAS curve,FT,generation,newcandidate propagation,extraGDN/conv or precision repair. Four passive native K captures and CPU current/candidate contractions are included; private boundaries retained only for layers0/1/2.')
study=R/'research/reproduction_templates/dt_MH3_native_K_boundaries_20260909.py';assert not study.exists();study.write_text(s,encoding='utf-8')
files={'study.py':study.read_bytes()}
for name,want in dp['files_sha256'].items():
    if name=='study.py':continue
    raw=(D/name).read_bytes();assert sha(raw)==want
    files[name]=raw
name='mh3_native_k_diagnostic_20260909.py';files[name]=(R/'research/reproduction_templates'/name).read_bytes()
for name in ('checkpoint','native_model_sha256','runtime_source_sha256','checkpoint_config_tokenizer_sha256','expected_weight_stats','cache_hashes','native_stage_source_sha256','span_source_sha256'):assert p[name]==dp[name],name
key='morehopqa_3';method='control';run_index=3;case=dr['cases'][key];freeze=dr['input_freeze_before_model_load'][key]
assert dr['runs'][run_index]['case']==key and dr['runs'][run_index]['method']==method
curve=case['curves'][method];assert curve['input_receipts']==dr['runs'][run_index]['deletion_audit']['input_receipts']
ids=np.asarray(freeze['input_ids'],dtype=np.int64);assert sha(ids.tobytes())==case['input']['input_sha256']
z=np.load(D/'vectors.npz',allow_pickle=False);w=z[key+'_'+method+'_evaluated'];assert np.all(np.diff(w[curve['sorted_keep']].astype(np.float64))<=0)
p.update(case_indices=dp['case_indices'],fixed_records=dp['fixed_records'],source_method=method,source_run_index=run_index,call_schedule=[[key,'DT']])
p['source_standalone']={kind+'_path':'/tmp/'+D.name+'/'+filename for kind,filename in [('results','results.json'),('vectors','vectors.npz'),('protocol','protocol.json')]}
for kind,filename in [('results','results.json'),('vectors','vectors.npz'),('protocol','protocol.json')]:p['source_standalone'][kind+'_sha256']=sha((D/filename).read_bytes())
p['source_standalone'].update(study_sha256=dp['files_sha256']['study.py'],vector_key=key+'_'+method,scope='Contemporaneous C control from completed fixed-eight K candidate study; same fixed author MH3 trajectory and original masks.')
p['frozen_capture_receipts']={str(step):curve['input_receipts'][step] for step in p['capture_steps']}
p['frozen_source_scores']={str(step):curve['scores'][step] for step in p['capture_steps']}
p['scope']='One current C DT on MH3, four original scorer calls, passive actual GDN1 K capture. Isolate local K candidate error versus whole-input prior regression. No new model/candidate/FT forward or metric curve.'
p['source_reuse']='Current runtime identical to completed K fixed-eight study. Passive focused callback only; no runtime/official source modifications. Prior whole-model regression retained as a separate process with explicit drift checks.'
p['budget']['wall_time_seconds']=600
p['decision']='Measure actual conditional K normalization errors at frozen original C masks3/10, with0/20 endpoints. If local candidate fails, do not blame downstream; if local improves, investigate actual lower propagation before changing another operator.'
p['outputs']='Scalar all-decoder telescoping ledger; same-token compact-head signed and absolute K contractions; small raw K vectors for independent CPU verification; private layers0/1/2 boundaries only.'
p['protected_sources']=[{'path':p['finite_FA_library'],'sha256':p['finite_FA_library_sha256']}]+[{'path':'/tmp/'+D.name+'/'+n,'sha256':sha((D/n).read_bytes())} for n in ('results.json','vectors.npz','protocol.json')]
p['files_sha256']={n:sha(raw) for n,raw in files.items()}
for n,raw in files.items():ast.parse(raw,filename=n)
files['protocol.json']=json.dumps(p,indent=2).encode()
stem='dt_MH3_native_K_boundaries_20260909';pp=A/(stem+'_protocol.json');lp=A/('launch_'+stem+'.json');assert not pp.exists() and not lp.exists()
remote='${ARTIFACT_ROOT}/codex_'+stem+'_v1';python='${ARTIFACT_ROOT}/codex_qwen35_isolated_import_20260908_v1/env/bin/python'
blob=base64.b64encode(zlib.compress(json.dumps({n:base64.b64encode(v).decode() for n,v in files.items()}).encode())).decode()
loader=('import pathlib,subprocess,json,base64,zlib;d=pathlib.Path('+repr(remote)+');d.mkdir(exist_ok=False);'
 'files=json.loads(zlib.decompress(base64.b64decode('+repr(blob)+')));[(d/k).write_bytes(base64.b64decode(v)) for k,v in files.items()];'
 'f=(d/"driver.log").open("w");j=subprocess.Popen(['+repr(python)+',"-B",str(d/"study.py")],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);'
 '(d/"pid").write_text(str(j.pid));print(json.dumps({"pid":j.pid,"directory":str(d)}))')
pp.write_bytes(files['protocol.json']);lp.write_text(json.dumps({'cmd':python+' -c '+shlex.quote(loader),'timeout':10}))
print(json.dumps({'protocol_sha256':sha(files['protocol.json']),'study_sha256':p['files_sha256']['study.py'],'sources':len(files)-2,'budget':p['budget']}))

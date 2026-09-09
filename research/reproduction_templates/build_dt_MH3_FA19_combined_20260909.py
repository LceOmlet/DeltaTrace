"""Freeze actual FA19 capture and eleven public probes inside one1DT4score job."""
import ast,base64,copy,hashlib,json,shlex,zlib
from pathlib import Path
A=Path(__file__).resolve().parent;R=A.parent/'DeltaTrace';sha=lambda b:hashlib.sha256(b).hexdigest()
D=A/'snapshot${ARTIFACT_ROOT}/codex_dt_GDN1_K_remaining_20260909_s2_v1'
dp=json.loads((D/'protocol.json').read_bytes())
p=copy.deepcopy(json.loads((A/'dt_MH3_native_K_boundaries_20260909_protocol.json').read_bytes()))
s=(R/'research/reproduction_templates/dt_MH2_current_conditional_boundaries_20260909.py').read_text().replace('MH2','MH3').replace('morehopqa_2','morehopqa_3')
def replace(old,new):
    global s
    assert s.count(old)==1,old[:80];s=s.replace(old,new)
replace('self.current=None',"self.current=None;self.fa19={'B1':{},'B1_native_arguments':{}}")
replace('    def boundary(self,name,m,x):', '''    def wants_decoder(self,index):return index==19
    def decoder(self,index,d,c,e,upstream,new,terms):
        assert index==19 and 'paired' not in self.fa19
        self.fa19['paired']=fa_ledger.features(d,c)
        self.fa19['coeff']=fa_ledger.pack_coeff(upstream,new,terms)
    def boundary(self,name,m,x):''')
replace('    import causal_conv1d', '''    import causal_conv1d
    from transformers.integrations.flash_attention import flash_attention_forward
    from flash_attn import flash_attn_varlen_func,flash_attn_func
    from qwen35_decoder_finite import NativeDecoderCapture
    from mh3_fa19_native_ledger_20260909 import ScopedNativeDenseCapture,analyze
    import decoder19_conditional_decomposition_20260908 as fa_ledger''')
replace("""            with torch.no_grad():lp=timed('original_scorer_with_passive_boundaries_step'+str(step),lambda:evaluator.compute_logprob_response_given_prompt(
                prompt[None].to('cuda'),case['target'][None].to('cuda')))""", """            with NativeDecoderCapture(layers[19]) as dc,ScopedNativeDenseCapture(
                layers[19].self_attn,flash_attention_forward,flash_attn_varlen_func,flash_attn_func) as mc,torch.no_grad():
                lp=timed('original_scorer_with_passive_boundaries_step'+str(step),lambda:evaluator.compute_logprob_response_given_prompt(
                    prompt[None].to('cuda'),case['target'][None].to('cuda')))
            assert mc.calls=={'module':1,'interface':1,'native_varlen':0,'native_dense':1} and not mc.inside
            assert dc.calls=={n:1 for n in ['input_norm','post_norm','gate','up','silu','down','mlp','decoder']}
            observer.fa19['B1'][str(step)]=fa_ledger.features(dc.values,mc.values)
            observer.fa19['B1_native_arguments'][str(step)]=mc.dense_arguments
            row.setdefault('FA19_passive_calls',{})[str(step)]={'decoder':dc.calls,'mixer':mc.calls,'interface_arguments':mc.interface_arguments,'mask_shape':list(mc.values['attention_mask'].shape) if 'attention_mask' in mc.values else None}
            del dc,mc""")
replace("    private={'coefficients':observer.coeff,'paired_native_boundaries':observer.paired,\n        'B1_native_boundaries':observer.activations,'source_protocol_sha256':sha((A/'protocol.json').read_bytes())}", """    timed('FA19_existing_native_operator_diagnostic',lambda:analyze(observer.fa19,p['frozen_capture_receipts'],r,vectors))
    row['FA19_diagnostic_scope']='Same process current DT and four original scores plus eleven existing public FA operator probes.'
    for step in ['3','10','20']:
        internal=r['FA19_native_probe']['points'][step]['internal']
        row.setdefault('FA19_internal_minus_boundary',{})[step]=internal['prediction_minus_actual']+row['decomposition'][step]['decoder_errors']['19']
    private={'FA19':observer.fa19,'coefficients':{n:observer.coeff[n] for n in ['0','19','20']},
        'paired_native_boundaries':{n:observer.paired[n] for n in ['0','19','20']},
        'B1_native_boundaries':{step:{n:values[n] for n in ['0','19','20']} for step,values in observer.activations.items()},
        'source_protocol_sha256':sha((A/'protocol.json').read_bytes())}""")
replace("A/'MH3_actual_boundary_private.pt'","A/'MH3_FA19_combined_private.pt'")
replace("'MH3_current_conditional_boundaries_1DT4score_complete'","'MH3_FA19_combined_1DT4score11FA_complete'")
replace('One current DT with passive34boundary coefficient copies;', 'One current DT with passive34boundary coefficient copies and focused FA19 native states;')
replace('No new MAS curve,FT,generation,newcandidate,extraGDN/conv or precision repair.', 'Eleven public FA calls add four native-operand replays and seven labeled hybrids inside the same process. No new MAS curve,FT,generation,newcandidate,extraGDN/conv or precision repair.')
study=R/'research/reproduction_templates/dt_MH3_FA19_combined_20260909.py';assert not study.exists();study.write_text(s,encoding='utf-8')
files={'study.py':study.read_bytes()}
for name,want in dp['files_sha256'].items():
    if name=='study.py':continue
    raw=(D/name).read_bytes();assert sha(raw)==want;files[name]=raw
for name in ['mh3_fa19_native_ledger_20260909.py','decoder19_conditional_decomposition_20260908.py']:files[name]=(R/'research/reproduction_templates'/name).read_bytes()
p['budget'].update(native_FA_operator_probes=11,selected_decoder_native_state_captures=5,wall_time_seconds=600)
p['scope']='Same-process MH3 current C DT, actual original B1 scorer states at fixed0/3/10/20, eleven public FA contrasts. Verify dominant FA19 core reference-content and routing errors; no candidate,FT,new metric curve or new compiled kernel.'
p['decision']='Require actual whole-decoder vs internal replay transfer and default native FA output agreement to be reported. Isolate PV reference interaction and route error before any new method. Do not infer source attribution error from mismatched query/key coordinate grouping.'
p['outputs']='Per-token boundary/scalar ledger; public-FA contrast arrays; private FA19 real states and boundary0/19/20 coefficients. No large raw arrays in public numeric exports.'
p['source_reuse']='All current runtime files copied byte-identically from last completed fixed-eight K study. Canonical older passive whole-boundary template adapted to FA19 capture; native and finite model implementations unchanged.'
p['files_sha256']={n:sha(raw) for n,raw in files.items()}
for n,raw in files.items():ast.parse(raw,filename=n)
files['protocol.json']=json.dumps(p,indent=2).encode();stem='dt_MH3_FA19_combined_20260909'
pp=A/(stem+'_protocol.json');lp=A/('launch_'+stem+'.json');assert not pp.exists() and not lp.exists()
remote='${ARTIFACT_ROOT}/codex_'+stem+'_v1';python='${ARTIFACT_ROOT}/codex_qwen35_isolated_import_20260908_v1/env/bin/python'
blob=base64.b64encode(zlib.compress(json.dumps({n:base64.b64encode(v).decode() for n,v in files.items()}).encode())).decode()
loader=('import pathlib,subprocess,json,base64,zlib;d=pathlib.Path('+repr(remote)+');d.mkdir(exist_ok=False);'
 'files=json.loads(zlib.decompress(base64.b64decode('+repr(blob)+')));[(d/k).write_bytes(base64.b64decode(v)) for k,v in files.items()];'
 'f=(d/"driver.log").open("w");j=subprocess.Popen(['+repr(python)+',"-B",str(d/"study.py")],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);'
 '(d/"pid").write_text(str(j.pid));print(json.dumps({"pid":j.pid,"directory":str(d)}))')
pp.write_bytes(files['protocol.json']);lp.write_text(json.dumps({'cmd':python+' -c '+shlex.quote(loader),'timeout':10}))
print(json.dumps({'protocol_sha256':sha(files['protocol.json']),'budget':p['budget']}))

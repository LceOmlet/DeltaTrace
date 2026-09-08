"""Freeze5native layer0replays+1currentfinite;never launch."""
import ast,base64,hashlib,json,shlex,zlib
from pathlib import Path
A=Path(__file__).resolve().parent;R=A.parent/'DeltaTrace';sha=lambda b:hashlib.sha256(b).hexdigest()
D=A/'snapshot${ARTIFACT_ROOT}/codex_dt_NI_current_boundaries_20260908_v1'
raw=(D/'results.json').read_bytes();assert sha(raw)=='476488cc9c2409a810ba9891c59f28e8db577ea5cc7044439cf857effb740b41'
prior=json.loads(raw);dp=json.loads((D/'protocol.json').read_bytes());assert prior['status']=='NI_current_34boundaries_1DT4score_observation_complete'
assert prior['private_artifact']['sha256']=='a1465cafdf608644b737be1b83f65f826e63ff9338ed3ce37f386aafb6ce6d24'
files={}
for name,want in dp['files_sha256'].items():
    if name=='study.py':continue
    value=(D/name).read_bytes();assert sha(value)==want
    candidates=[base/name for base in (R/'research/runtime',R/'core') if (base/name).exists()]
    assert len(candidates)==1 and sha(candidates[0].read_bytes())==want,('shared source changed',name)
    files[name]=value
files['study.py']=(R/'research/reproduction_templates/dt_NI_layer0_internal_20260908.py').read_bytes()
helper='NI_layer0_boundary_ledger_20260908.py';files[helper]=(R/'research/reproduction_templates'/helper).read_bytes()
for name,value in files.items():ast.parse(value,filename=name)
keys=('checkpoint','checkpoint_config_tokenizer_sha256','native_model_sha256','isolated_site','compiler_cache',
      'boundary_compiler_cache','installed_FA_interface_sha256','native_stage_source_sha256','official_root',
      'official_source_blob_sha1','dependency_overlays','runtime_source_sha256','expected_weight_stats')
p={k:dp[k] for k in keys}
p.update(scope='Measurement only:one original fullmodel load,five actual native decoder0 replays from saved current B2/B1 h0,andone existing current finite decoder0 from savedm1 with symmetric GDN output normgate. No wholeforward/currentDT/scorer/newrule/FA/FT.',
    source_results_path='/tmp/'+D.name+'/results.json',source_results_sha256=sha(raw),
    source_private_path='/tmp/'+D.name+'/'+prior['private_artifact']['file'],source_private_sha256=prior['private_artifact']['sha256'],source_private_bytes=prior['private_artifact']['bytes'],
    input=prior['input'],steps=['B2','0','1','10','20'],layer=0,norm_gate_rule='symmetric',
    native_kwargs='Pinned original Qwen3_5TextModel lines1174-1219 createscache onlyifuse_cache;originalrunner root specifiesFalse andoriginalevaluator omits it(configTrue). B2past=None/use_cacheFalse;eachB1freshnative.DynamicCache/use_cacheTrue. Native create_recurrent_attention_mask invoked withsavedh0,allones,arange text positionids andthatcache. The pinned decoder linear_attention branch ignores position_embeddings/position_ids;None forunusedrotary embeddings avoids extraRoPE forward,not an alternateattentioncomputation. No packed seq indices or priorstates.',
    kwargs_provenance={'native_model_sha256':dp['native_model_sha256'],'reviewed_local_source':'audit/snapshot${ARTIFACT_ROOT}/codex_qwen35_fla_inventory_20260908_v1/modeling_qwen3_5.py',
        'decoder_lines':'774-814','text_model_lines':'1174-1219','GDN_lines':'444-557','evaluator_lines':'llm_attr_eval.py58-84',
        'runtime_recurrent_mask_and_cache_source':'Actual imported official helper/class file hashes and cache receipts recorded before any replay; no own mask or state recurrence implementation.'},
    budget={'official_fullmodel_loads':1,'native_layer0_replays':5,'native_layer0_B2_replays':1,'native_layer0_B1_replays':4,
        'finite_decoder0_calls':1,'finite_FLA_calls':1,'finite_FLA_native_input_adjoints':2,
        'finite_public_linear_conv_forward':1,'finite_public_linear_conv_autograd_backward':1,
        'whole_model_forwards':0,'eager_warmup_forwards':0,'DT_calls':0,'scorer_calls':0,'FA_calls':0,'FT_calls':0,
        'generation_calls':0,'new_samples':0,'new_candidate_rules':0,'wall_time_seconds':600},
    ledger='Existing16term GDN decoder ledger:MLPcombined,postRMS,inputRMS,decoder/mixerresidualrounding,outputprojection,fusednormgate,moBF16cast,finiteFLAincludingraw_g_exp,QKL2/headfold,beta sigmoid,raw_g parameter map,convsplit,convlinear/SiLU,inputprojection,inputalias. Each explicitterm is prediction-minus-actual from actualnativeoperands andactualproductiondiagnostic multipliers. Final signed FiniteFLA term=q.deltaq+k.deltak+v.deltav+beta.deltabeta+rawg.deltarawg−monative.deltao;not unexplained wholedecoder remainder.',
    replay_transfer='Sum16replayed terms + <savedm0-newm0,saveddeltah0> + <newm0,saveddeltah0-replayeddeltax> + <savedm1,replayeddeltay-saveddeltah1> equals savedactualdecoder0boundaryerror. Savebothscalarsandtokenvectors;rowgroupsarehiddencontractioncoordinates,not independentcausalinputsources. ReportB1freshcache/B2nonecache convention andnativeoutput/coefficientdrift separately; noforcedbitidentity.',
    artifacts='PrivateCPUactuald/c/e for5nativepasses,currentdiagnosticcoefficients includingnativeFLAactualstateh/vnew,and saved0/1boundaries only. No copiedmodelweights,other31layers,fullvocab logits orT-squaredattention. Keepallnormalnorm/MLP/conv transferterms,noextraoperators. Bytes/hashes/actualsourcepins,timingandentered/returnedcounts;smallvectorsandJSONpublicZIP,privateexcluded.',
    costs='One official model load preservesdtype/loading;onecurrentfinite includesallnormalcompiledMLP/norm graphs,2nativeFLAinputadjoints pluscompiledmixedcoefficients,realpublicconvlinearpreactivation+autograd(includingdiscardedweightgrad). Firstcallcompile/setup andpassiveCPUcapture aretimed;not productionefficiencybenchmark.',
    acceptance='Exactinput/source/weight/privateSHA,savedlayer0boundaryerrorCPUrecovered1e-7;native5replayscaptures1module/conv/FLAstage each;1currentfiniteFLAcallback;finitecoefficients;16terms andtransfer/groupclosures1e-7. Coefficient/nativeoutputdrift reportonly;no historical2percent orbitwise admission. No claimMAS/RISEfromoperatorledger.',
    stop='One load/five native layer0/one current finite only. No wholemodelforward permitted(guard installed),no scorer,extraFA,warmup,loopretry,alternatebackend,newprecision,newrule orautomaticcandidate. Preserve firstfailure/partialnativecapture andunknown unfinishedoperatorwork.600s ceiling.',
    files_sha256={name:sha(value) for name,value in files.items()})
files['protocol.json']=json.dumps(p,indent=2).encode();pp=A/'dt_NI_layer0_internal_protocol_20260908.json';lp=A/'launch_dt_NI_layer0_internal_20260908.json'
assert not pp.exists() and not lp.exists(),'Do not overwrite frozen measurement.'
blob=base64.b64encode(zlib.compress(json.dumps({name:base64.b64encode(value).decode() for name,value in files.items()}).encode())).decode()
py='${ARTIFACT_ROOT}/codex_qwen35_isolated_import_20260908_v1/env/bin/python';directory='${ARTIFACT_ROOT}/codex_dt_NI_layer0_internal_20260908_v1'
loader=('import base64,pathlib,subprocess,json,zlib;d=pathlib.Path('+repr(directory)+');d.mkdir(exist_ok=False);'
    'files=json.loads(zlib.decompress(base64.b64decode('+repr(blob)+')));[(d/k).write_bytes(base64.b64decode(v)) for k,v in files.items()];'
    'f=(d/"driver.log").open("w");j=subprocess.Popen(['+repr(py)+',"-B",str(d/"study.py")],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);'
    '(d/"pid").write_text(str(j.pid));print(json.dumps({"pid":j.pid,"directory":str(d)}))')
pp.write_bytes(files['protocol.json']);lp.write_text(json.dumps({'cmd':py+' -c '+shlex.quote(loader),'timeout':10}))
print(json.dumps({'protocol_sha256':sha(files['protocol.json']),'study_sha256':p['files_sha256']['study.py'],'helper_sha256':p['files_sha256'][helper],
    'payload':str(lp),'remote_directory':directory,'budget':p['budget']}))

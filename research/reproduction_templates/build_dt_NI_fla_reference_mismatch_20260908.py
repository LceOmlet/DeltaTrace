"""Freeze one saved FLA adjoint capture plus existing CPU chunk algebra."""
import ast,base64,hashlib,json,shlex,zlib
from pathlib import Path
A=Path(__file__).resolve().parent;R=A.parent/'DeltaTrace';sha=lambda b:hashlib.sha256(b).hexdigest()
D=A/'snapshot${ARTIFACT_ROOT}/codex_dt_NI_layer0_internal_20260908_v1'
raw=(D/'results.json').read_bytes();assert sha(raw)=='b4600ed125b917a78a358655e0e8688842234d56a1224f4b51654d206c6bc365'
source=json.loads(raw);dp=json.loads((D/'protocol.json').read_bytes())
assert source['status']=='NI_current_layer0_5replay1finite_internal_measurement_complete'
assert source['private_artifact']['sha256']=='609a80669dc60a1da2664d24472ab21b4c9bd4c556bebefa03d0b0dbca77469a'
files={
    'study.py':(R/'research/reproduction_templates/dt_NI_fla_reference_mismatch_20260908.py').read_bytes(),
    'NI_fla_reference_mismatch_cpu_20260908.py':(R/'research/reproduction_templates/NI_fla_reference_mismatch_cpu_20260908.py').read_bytes(),
    'finite_fla_gpu.py':(R/'research/runtime/finite_fla_gpu.py').read_bytes(),
    'finite_fla_chunk_reference.py':(R/'research/runtime/finite_fla_chunk_reference.py').read_bytes()}
assert sha(files['finite_fla_gpu.py'])==dp['files_sha256']['finite_fla_gpu.py']
for name,value in files.items():ast.parse(value,filename=name)
keys=('compiler_cache','boundary_compiler_cache','dependency_overlays','native_stage_source_sha256')
p={key:dp[key] for key in keys}
p.update(scope='Measure only current FLA reference-content mismatch using already saved actual layer0 operands. One existing native_input_adjoints call(two native stages),one existing mixed_coefficients diagnosticsTrue call;then CPU64 original chunk reference. Zero model load/forward,DT,scorer,FA,FT,newcandidate.',
    input=source['input'],scale=source['points']['B2']['actual_scale'],
    source_results_path='/tmp/'+D.name+'/results.json',source_results_sha256=sha(raw),
    source_private_path='/tmp/'+D.name+'/'+source['private_artifact']['file'],source_private_sha256=source['private_artifact']['sha256'],source_private_bytes=source['private_artifact']['bytes'],
    boundary_results_path=dp['source_results_path'],boundary_results_sha256=dp['source_results_sha256'],
    protected_sources=[{'path':'/tmp/'+D.name+'/'+name,'sha256':sha((D/name).read_bytes())} for name in ('results.json','protocol.json')]
        +[{'path':dp['source_results_path'],'sha256':dp['source_results_sha256']}],
    capture='Reuse saved actual B2 endpoints and actual mo_native bit-for-bit as inputs. Existing native_input_adjoints returns do,dh_end,dU_WY;existing mixed_coefficients(...False,diagnostics=True) returns sixcoefficients and actual production L/r0. Keepnative stages outside torch.compile and same existing compileroptions. Saveallnativeadjoints/L/r0/replaycoefficients andcomparewithsavedcoefficients;no divisionbybeta or inventedreverse recurrence.',
    shared_route='Original B2input endpoint1,scale,seed,chunk boundaries,padding,initialzero/terminalzero state remainfixed for every CPU referenceR. Native adjoints depend only on thatinput,not EOS/A reference;therefore reuseexactlyoneactualadjoint result. CPU mixed_chunk consumes suppliedactualL andD ratherthan recomputing A1transpose*dU in higherprecision.',
    main_ledger='ForA=actual captured deleted B1 state andhigh1=actualB2input,delta01A=x1-xA. E_saved_pair=(M_saved-M_replay).delta+(M_replay-M_CPU01).delta+(M_CPU01-M_CPURA).delta+(M_CPURA.delta-Z.deltao),withrawgexpanded viaactualalpha as below. Add the common mask-independent transfer M_saved.(x_B1clean-x_B2input)-Z.(o_B1clean-o_B2input) to recoveroriginal savedB1clean-A error. B2guardA=originalEOS has no commontransfer.',
    semantic_splits='q reference difference is reading from capturedreference state;beta difference is L.(c0-cA)*deltabeta where cR=vR-rR andrR is the existing actualh/k/g/u chunk-algebra reference,not separately returnednative r. vreference coefficient identicalW=beta1L andmustzero. Keycoefficient splitsexactly into originalwrite summands and originalpre-read summands;no unspecifiedwhole-error subtractioncalledFLA.',
    key_equations='W=beta1L;E1[i,j]=exp(G1[j]-G1[i])1[j>=i];ER[i,j]=exp(GR[i]-GR[j])1[j<=i]. KreadR=-exp(GR)(W HR^T)-[(W uR^T)*ER*strictLower]kR. KwriteR=dkR-KreadR,exactlythe original remaining4summands:eend*uRD^T+[(uRZ^T)*E1](scaleq1)-[(uRW^T)*E1]k1+beta1*k1*rowdot(uR,L). Onlyfirstexpressionisevaluatedagain;subtractionisolatesknownformula terms,not an unexplainederror.',
    decay_equation='Let a01/aA1 be originalCPUalpha coefficients,e01=exp_secant(rawg0,rawg1),deltaalpha=exp(rawg1)-exp(rawgA),deltag=rawg1-rawgA. a01*e01*deltag=(a01-aA1)*deltaalpha+a01*(e01*deltag-deltaalpha)+aA1*deltaalpha. Firstpart isalpha reference-state change;second currentexp-secant conditional curvature;last belongs toA→1primitive pairedledger. Do not label rawg total asforget-reference or mix these two mechanisms.',
    reference_arithmetic='Unchanged finite_fla_chunk_reference.mixed_chunk andexp_secant CPU64 attribution algebra,maximum64x64tiles. Suppliedh/g/k/v_new are actualcapturedstates. Original64step affine scalar-prefix is a coefficientcontraction,not a newKxVstate/modeltrajectory. No fullT-squaredmatrix,per-tokenKxVstates,newsoftmax,newforward or elevatedGPUprecision.',
    remainder='The reanchored A→1 primitive residual is measured,not assumedzero or pre-labelledcausalstateerror. It can containnative read/write/chunk/alpha-cumulative rounding andnative-adjoint/defaultprecision consistency. B1clean/B2input drift is separately explicit and commonacrossmasks;no furtheroperator or model run isauthorized to chase thisremainder.',
    budget={'model_loads':0,'model_forwards':0,'native_input_adjoints_calls':1,'native_stage_calls_on_success':2,
        'mixed_coefficients_calls':1,'newGPUrules':0,'CPU_reference_mixed_chunk_calls':76,'CPU_ref_variants':4,
        'CPU_BLAS_threads':4,'DT_calls':0,'scorer_calls':0,'FA_calls':0,'FT_calls':0,'generation_calls':0,
        'new_samples':0,'extra_warmup':0,'wall_time_seconds':600},
    costs='ActualoneadjointAPI+existingmixedcompile/call measuredincludingdefaultsetup. AdditionalretainedL/r0/D/du/coeffbytes andCPU64referencealgebra measuredseparately. CPU4refs*19chunks=head-batched76mixed_chunkcalls,with3extraCPUkeyreadmatmuls each;no extraGPUmatrices beyond existingmixeddiagnostics. Not a productionefficiencyclaim.',
    acceptance='Source/private/seed/layout invariants;finitecapturedcoefficients andCPUarrays;coefficientreplaydriftreportonly;unchangedvreference exactlyzero;main/branch/head/group/tokenclosures1e-7;recovered savedFLAerror matchesprioractual16term ledger1e-7 for1/10/20/B2. Remainder andlargecanceling signedterms remainvisible and cannotbe declaredsolvedbyclosurealone.',
    artifacts='Original2.642GBprivate staysremote. Additionalactualadjoint/mixedprivate storedwithhash/bytes;reviewZIPcontains source/protocol/summary/signedtokenheadvectors only. No weights/data regeneration/fullmodeloutput.',
    stop='One nativeadjoint+oneexistingmixed only;firstfailure or600s preservespartialreceipt. No retry,model,FT,alternateprecision,epsilon,denominatorrecovery,layersweep or newcandidate. Reanchoredoperatoralgebra doesnot establishMAS/RISEimprovement.',
    files_sha256={name:sha(value) for name,value in files.items()})
files['protocol.json']=json.dumps(p,indent=2).encode();pp=A/'dt_NI_fla_reference_mismatch_protocol_20260908.json';lp=A/'launch_dt_NI_fla_reference_mismatch_20260908.json'
assert not pp.exists() and not lp.exists(),'Do not overwrite frozen audit.'
blob=base64.b64encode(zlib.compress(json.dumps({name:base64.b64encode(value).decode() for name,value in files.items()}).encode())).decode()
py='${ARTIFACT_ROOT}/codex_qwen35_isolated_import_20260908_v1/env/bin/python';directory='${ARTIFACT_ROOT}/codex_dt_NI_fla_reference_mismatch_20260908_v1'
loader=('import base64,pathlib,subprocess,json,zlib;d=pathlib.Path('+repr(directory)+');d.mkdir(exist_ok=False);'
    'files=json.loads(zlib.decompress(base64.b64decode('+repr(blob)+')));[(d/k).write_bytes(base64.b64decode(v)) for k,v in files.items()];'
    'f=(d/"driver.log").open("w");j=subprocess.Popen(['+repr(py)+',"-B",str(d/"study.py")],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);'
    '(d/"pid").write_text(str(j.pid));print(json.dumps({"pid":j.pid,"directory":str(d)}))')
pp.write_bytes(files['protocol.json']);lp.write_text(json.dumps({'cmd':py+' -c '+shlex.quote(loader),'timeout':10}))
print(json.dumps({'protocol_sha256':sha(files['protocol.json']),'study_sha256':p['files_sha256']['study.py'],
    'helper_sha256':p['files_sha256']['NI_fla_reference_mismatch_cpu_20260908.py'],'payload':str(lp),'remote_directory':directory,'budget':p['budget']}))

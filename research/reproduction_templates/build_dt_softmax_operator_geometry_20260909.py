"""Freeze one CPU-only audit of already saved native operands; do not dispatch."""
import ast,base64,hashlib,json,shlex,zlib
from pathlib import Path
A=Path(__file__).resolve().parent;R=A.parent/'DeltaTrace'
sha=lambda b:hashlib.sha256(b).hexdigest()
old=json.loads((A/'snapshot${ARTIFACT_ROOT}/codex_dt_MH0_FA19_PV_orders_20260909_v2/protocol.json').read_bytes())
files={'study.py':(R/'research/reproduction_templates/dt_softmax_operator_geometry_20260909.py').read_bytes()}
ast.parse(files['study.py'])
p={k:old[k] for k in ('private_artifact_path','private_artifact_sha256','source_results_path','source_results_sha256','scale')}
p.update(queries=[443,547],heads=list(range(16)),steps=['3','10'],
    selection='Two previously documented large conditional-error query coordinates; all16heads included, no outcome-dependent row filtering. Development diagnostic, not a benchmark.',
    budget={'CPU_saved_state_loads':1,'CPU_rows':32,'GPU_model_DT_FA_FLA_scorer_FT_generation_calls':0,'wall_time_seconds':120},
    geometry='For normalized endpoints d=p1-p0,s=z1-z0,D=d.s,J=diag(p1)-p1p1^T. Supported pullback M=J+d(d-Js)^T/D need not be symmetric or PSD. Candidate H=J-(Js)(Js)^T/(s.Js)+dd^T/D is symmetric PSD and Hs=d when denominators positive; follows weighted Cauchy-Schwarz. No effectiveness claim follows from these invariants.',
    evaluation='CPU FP64 algebra on actual saved native BF16 Q/K/V and BF16-rounded saved seed. Per-row softmax is a mathematical diagnostic only, not a reproduced native FA forward. Conditional effects use actual saved clean/deleted operands but ideal FP64 softmax. Numerical boundary and whole-input propagation remain untested. No global sequence-square matrices; only vectors and2x2blocks.',
    prior_counterexample={'z0':[0,40,40],'z1':[0,0,-12],'min_eigenvalue_symmetric_part':-0.05389602386543723,'scope':'A deterministic three-category algebraic counterexample to universal PSD, not a benchmark or evidence this caused the observed real quality regression.'},
    decision='Use actual rows to test whether the structural flaw occurs and whether symmetric correction helps both conditional directions without hiding head cancellation. Do not promote a method, implement a new GPU kernel or expand model tests from a toy counterexample alone.',
    files_sha256={k:sha(v) for k,v in files.items()})
files['protocol.json']=json.dumps(p,indent=2).encode()
pp=A/'dt_softmax_operator_geometry_protocol_20260909.json';lp=A/'launch_dt_softmax_operator_geometry_20260909.json'
assert not pp.exists() and not lp.exists()
remote='${ARTIFACT_ROOT}/codex_dt_softmax_operator_geometry_20260909_v1';python='${ARTIFACT_ROOT}/codex_qwen35_isolated_import_20260908_v1/env/bin/python'
blob=base64.b64encode(zlib.compress(json.dumps({k:base64.b64encode(v).decode() for k,v in files.items()}).encode())).decode()
loader=('import pathlib,subprocess,json,base64,zlib;d=pathlib.Path('+repr(remote)+');d.mkdir(exist_ok=False);'
 'files=json.loads(zlib.decompress(base64.b64decode('+repr(blob)+')));[(d/k).write_bytes(base64.b64decode(v)) for k,v in files.items()];'
 'f=(d/"driver.log").open("w");j=subprocess.Popen(['+repr(python)+',"-B",str(d/"study.py")],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);'
 '(d/"pid").write_text(str(j.pid));print(json.dumps({"pid":j.pid,"directory":str(d)}))')
pp.write_bytes(files['protocol.json']);lp.write_text(json.dumps({'cmd':python+' -c '+shlex.quote(loader),'timeout':10}))
print(json.dumps({'protocol_sha256':sha(files['protocol.json']),'CPU_rows':32,'GPU_calls':0}))

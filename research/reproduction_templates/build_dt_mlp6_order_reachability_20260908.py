"""Freeze four CPU-only analytic points; never launch."""
import ast,base64,hashlib,json,shlex,zlib
from pathlib import Path
A=Path(__file__).resolve().parent;R=A.parent/'DeltaTrace';sha=lambda b:hashlib.sha256(b).hexdigest()
D=A/'snapshot${ARTIFACT_ROOT}/codex_dt_mlp6_mechanism_20260908_v1';raw=(D/'results.json').read_bytes();r=json.loads(raw)
assert r['status']=='MLP6_actual_mechanism_diagnostic_complete' and r['saved_vs_recomputed_mnorm']['bitwise_equal']
parent=r['protocol'];files={'study.py':(R/'research/reproduction_templates/dt_mlp6_order_reachability_20260908.py').read_bytes()}
ast.parse(files['study.py']);directory='${ARTIFACT_ROOT}/codex_dt_mlp6_order_reachability_20260908_v1'
p={'scope':'CPU-only analytical reachability of the largest observed MH1 layer6 MLP error. No method change or candidate attribution.',
    'layer_index':6,'fixed_steps':['1','10','20'],'B2_control':True,
    'inputs':{'actual_results':{'path':parent['actual_results_path'],'sha256':parent['actual_results_sha256']},
        'actual_private':{'path':parent['actual_artifact_path'],'sha256':parent['actual_artifact_sha256']},
        'mechanism_results':{'path':'${ARTIFACT_ROOT}/codex_dt_mlp6_mechanism_20260908_v1/results.json','sha256':sha(raw)},
        'recomputed_coefficients':{'path':'${ARTIFACT_ROOT}/codex_dt_mlp6_mechanism_20260908_v1/'+r['diagnostic_artifact']['file'],'sha256':r['diagnostic_artifact']['sha256']}},
    'budget':{'GPU_calls':0,'GEMM_calls':0,'model_calls':0,'DT_calls':0,'FT_calls':0,'scorer_calls':0,'weight_reads':0,
        'conditional_points':3,'B2_controls':1,'predefined_orders':2,'lambda_searches':0,'wall_time_seconds':120},
    'formulas':{'determinant':'<rho, (s1-s0)*delta_u-(u1-u0)*sigma*delta_g>',
        'content1':'mu=rho*s1; mg=rho*u0*sigma; theoretical shift=determinant/2',
        'reverse':'mu=rho*s0; mg=rho*u1*sigma; theoretical shift=-determinant/2',
        'actual_coefficient_transfer':'Compare both ideal coefficient pairs directly to actual saved symmetric mu/mg, including SiLU branch factor change and original rounding.'},
    'acceptance':'Frozen source/results/privateartifact hashes before/after; original recomputed mnorm equals actual saved; CPU64 coefficient/determinant identity residuals<1e-7. Report signs/groups and B2/allEOS controls. No retuned admission guard.',
    'limitations':'Actual candidate BF16 casts/input GEMMs and earlier layers are not run. Shift+oldMLPerror explicitly assumes unchanged input-transfer; favorable local result is not evidence of RISE/MAS or fullinput attribution improvement. Historical all-layer content1 failure remains evidence; no automatic extension.',
    'stop':'First hash/shape/nonfinite/algebra failure or120seconds; no retry, candidate scan, new sample, GPU or model call.',
    'files_sha256':{k:sha(v) for k,v in files.items()}}
files['protocol.json']=json.dumps(p,indent=2).encode();protocol=A/'dt_mlp6_order_reachability_protocol_20260908.json';launch=A/'launch_dt_mlp6_order_reachability_20260908.json'
assert not protocol.exists() and not launch.exists();protocol.write_bytes(files['protocol.json'])
blob=base64.b64encode(zlib.compress(json.dumps({k:base64.b64encode(v).decode() for k,v in files.items()}).encode())).decode()
py='${ARTIFACT_ROOT}/codex_qwen35_isolated_import_20260908_v1/env/bin/python'
loader=('import base64,pathlib,subprocess,json,zlib;d=pathlib.Path('+repr(directory)+');d.mkdir(exist_ok=False);'
    'files=json.loads(zlib.decompress(base64.b64decode('+repr(blob)+')));[(d/k).write_bytes(base64.b64decode(v)) for k,v in files.items()];'
    'f=(d/"driver.log").open("w");j=subprocess.Popen(['+repr(py)+',"-B",str(d/"study.py")],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);'
    '(d/"pid").write_text(str(j.pid));print(json.dumps({"pid":j.pid,"directory":str(d)}))')
launch.write_text(json.dumps({'cmd':py+' -c '+shlex.quote(loader),'timeout':10}))
print(json.dumps({'protocol_sha256':sha(files['protocol.json']),'study_sha256':p['files_sha256']['study.py'],
    'launch_payload':str(launch),'remote_directory':directory,'budget':p['budget']}))

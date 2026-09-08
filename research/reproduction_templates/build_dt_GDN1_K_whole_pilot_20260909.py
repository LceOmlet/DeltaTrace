"""Freeze one full original NI0/MH2 pilot after local K evidence; no FT changes."""
from pathlib import Path
import ast,json,hashlib,base64,zlib,shlex,copy
A=Path('audit');R=Path('DeltaTrace');D=A/'snapshot${ARTIFACT_ROOT}/codex_dt_PV_layer19_remaining_regression_20260909_s0_v1'
sha=lambda b:hashlib.sha256(b).hexdigest();old=json.loads((D/'protocol.json').read_bytes())
scout=json.loads((A/'dt_MH2_GDN1_K_input_supported_summary_20260909.json').read_bytes())
assert scout['status']=='MH2_K_input_supported_independent_CPU_audit_passed_local_improvement_only'
study=(R/'research/reproduction_templates/dt_GDN1_K_whole_pilot_20260909.py').read_bytes();tree=ast.parse(study)
used={n.slice.value for n in ast.walk(tree) if isinstance(n,ast.Subscript) and isinstance(n.value,ast.Name) and n.value.id=='p' and isinstance(n.slice,ast.Constant) and isinstance(n.slice.value,str)}
p={k:copy.deepcopy(old[k]) for k in used if k!='files_sha256'};files={'study.py':study};changed={}
for name,want in old['files_sha256'].items():
    if name=='study.py':continue
    original=(D/name).read_bytes();assert sha(original)==want
    matches=[root/name for root in [R/'research/runtime',R/'core',R/'research/reproduction_templates'] if (root/name).is_file()];assert len(matches)==1
    raw=matches[0].read_bytes()
    if raw!=original:
        assert name in ['qwen35_gdn_finite.py','qwen35_dense_finite_runner.py'];changed[name]={'before_sha256':want,'after_sha256':sha(raw)}
    files[name]=raw
assert set(changed)=={'qwen35_gdn_finite.py','qwen35_dense_finite_runner.py'}
helper='input_supported_l2_20260909.py';files[helper]=(R/'research/reproduction_templates'/helper).read_bytes()
scoutdir=A/'snapshot${ARTIFACT_ROOT}/codex_dt_MH2_GDN1_K_input_supported_20260909_v1'
assert files[helper]==(scoutdir/helper).read_bytes()
oldstudy=(D/'study.py').read_text();new=study.decode().replace('\r\n','\n');a='    # Original scorer forwards, with own actual vectors and masks for every curve.';b="    r['finite_counts']={method:"
assert oldstudy.split(a)[1].split(b)[0]==new.split(a)[1].split(b)[0]
p['budget'].update(finite_FA_calls=32,original_finite_FA_calls=32,finite_FA_phases=96,content0_FA_calls=0,content1_FA_calls=32,new_candidate_operators=1,key_norm_candidate_calls=2)
p.update(scope='Current C versus only GDN1 input-supported K finite normalization. Original frozen NI0/MH2 inputs, targets, gold and scorer;4DT84score. No FT/generation or native model/FA/FLA changes.',
    hypothesis='Actual K finite norm conditional errors improve in the saved local scout. Off-chord input tangent preserves the endpoint chord while reducing early/mid signed and absolute token-head errors. Mid negative error grows, and adjacent propagation/ordering can reverse the improvement, so actual complete original metrics are required.',
    method='Same full finite propagation as C; only key_norm_by_layer={1: frozen helper} in candidate. Q normalization, all other GDN/FA, layer0 symmetric normgate and full FLA endpoint average retain C. No final-score transform or mask-specific coefficient.',
    actual_cost='Both methods8finiteFA/24phases,25finiteFLA/50nativeadjoints,8publicLSE,32native decoder replays per DT. One K finite callback replaces existing L2 finite operations at GDN1. Same O(BTHD) cost, no extra model/GEMM/path endpoints or T-squared matrices. Actual speed unproven; pilot single timings are not hot/batch evidence.',
    changed_runtime=changed,original_metric_source_byte_identical=True,
    fixed_comparison_set=['niah_mq_q2_0','niah_mq_q2_1','niah_mq_q2_2','niah_mq_q2_3','morehopqa_0','morehopqa_1','morehopqa_2','morehopqa_3'],
    executed_subset=['niah_mq_q2_0','morehopqa_2'],coverage='Two previously used developer cases only, no heldout or true batch claim.',
    decision='Finish both frozen original curves unless source/numerical/timeout failure. Review own needle/RISE/MAS and both actual mask-family errors. No automatic expansion, continuous weight tuning or promotion from a local sum.',
    protected_sources=p['protected_sources']+[{'path':'/tmp/'+scoutdir.name+'/'+name,'sha256':sha((scoutdir/name).read_bytes())} for name in ['results.json','protocol.json']],
    files_sha256={name:sha(raw) for name,raw in files.items()})
for name,raw in files.items():ast.parse(raw,filename=name)
remote='${ARTIFACT_ROOT}/codex_dt_GDN1_K_whole_pilot_20260909_v1';python='${ARTIFACT_ROOT}/codex_qwen35_isolated_import_20260908_v1/env/bin/python'
files['protocol.json']=json.dumps(p,indent=2).encode();pp=A/'dt_GDN1_K_whole_pilot_protocol_20260909.json';lp=A/'launch_dt_GDN1_K_whole_pilot_20260909.json';assert not pp.exists() and not lp.exists()
blob=base64.b64encode(zlib.compress(json.dumps({n:base64.b64encode(v).decode() for n,v in files.items()}).encode())).decode()
loader='import pathlib,subprocess,json,base64,zlib;d=pathlib.Path('+repr(remote)+');d.mkdir(exist_ok=False);files=json.loads(zlib.decompress(base64.b64decode('+repr(blob)+')));[(d/k).write_bytes(base64.b64decode(v)) for k,v in files.items()];f=(d/"driver.log").open("w");j=subprocess.Popen(['+repr(python)+',"-B",str(d/"study.py")],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);(d/"pid").write_text(str(j.pid));print(json.dumps({"pid":j.pid,"directory":str(d)}))'
pp.write_bytes(files['protocol.json']);lp.write_text(json.dumps({'cmd':python+' -B -c '+shlex.quote(loader),'timeout':10}))
print(json.dumps({'protocol_sha256':sha(files['protocol.json']),'study_sha256':sha(study),'changed_runtime':changed,'budget':p['budget']}))

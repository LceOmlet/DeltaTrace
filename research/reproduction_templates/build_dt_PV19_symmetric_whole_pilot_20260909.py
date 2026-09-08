from pathlib import Path
import ast,json,hashlib,base64,zlib,shlex,copy
A=Path('audit');R=Path('DeltaTrace');D=A/'snapshot${ARTIFACT_ROOT}/codex_dt_PV_layer19_remaining_regression_20260909_s0_v1'
sha=lambda b:hashlib.sha256(b).hexdigest();old=json.loads((D/'protocol.json').read_bytes());r=json.loads((D/'results.json').read_bytes());assert r['status']=='layer19_PV_content0_remaining_segment_4DT100FLA84score_complete'
receipt=json.loads((D/'terminal_receipt.json').read_bytes());assert receipt['proc_exists'] is False
for n,x in receipt['files'].items():assert sha((D/n).read_bytes())==x['sha256']
study=(R/'research/reproduction_templates/dt_PV19_symmetric_whole_pilot_20260909.py').read_bytes();tree=ast.parse(study)
used={n.slice.value for n in ast.walk(tree) if isinstance(n,ast.Subscript) and isinstance(n.value,ast.Name) and n.value.id=='p' and isinstance(n.slice,ast.Constant) and isinstance(n.slice.value,str)}
p={k:copy.deepcopy(old[k]) for k in used if k!='files_sha256'};files={'study.py':study};changed={}
for name,want in old['files_sha256'].items():
 if name=='study.py':continue
 original=(D/name).read_bytes();assert sha(original)==want
 matches=[root/name for root in [R/'research/runtime',R/'core',R/'research/reproduction_templates'] if (root/name).is_file()];assert len(matches)==1
 raw=matches[0].read_bytes()
 if raw!=original:
  assert name in ['qwen35_decoder_finite.py','qwen35_dense_finite_runner.py'];changed[name]={'before_sha256':want,'after_sha256':sha(raw)}
 files[name]=raw
assert set(changed)=={'qwen35_decoder_finite.py','qwen35_dense_finite_runner.py'}
oldstudy=(D/'study.py').read_text();new=study.decode().replace('\r\n','\n');a='    # Original scorer forwards, with own actual vectors and masks for every curve.';b="    r['finite_counts']={method:"
assert oldstudy.split(a)[1].split(b)[0]==new.split(a)[1].split(b)[0],'Original metric source changed'
p['budget'].update(finite_FA_calls=34,original_finite_FA_calls=34,finite_FA_phases=102,content0_FA_calls=2,content1_FA_calls=32,symmetric_FA_callback_sites=2,new_candidate_operators=1)
p.update(scope='Current C versus symmetric PV19, original fixed NI0 and MH2 processed records. One model/4DT/84original score calls, same whole-answer target, no FT/gen. Only current optional symmetric PV rule added; other runtime and native model/FA/FLA unchanged.',
 hypothesis='Current P1 has actual EOS-value route-reference errors in MH0 and MH2. P0 reduces MAS in6of8 but harms NI0/NI3needle and6RISE cases. Symmetric PV is the original third discrete allocation, not a fitted continuous weight or final score average. It may preserve more actual-input value routing while reducing reference asymmetry. Whole quality unproven; NI0 tests prior needle harm and MH2 tests actual largest remaining mismatch.',
 method='Only FA19: run the unchanged content1 backend and its valid endpoint-reversed content0 version; average dq/dk/dv in FP32 then storeBF16 before the same finite input/norm propagation. Existing logmean/QK endpoint symmetry required. Other7FA remain P1; layer0 normgate and FLA endpoint average retained.',
 actual_cost='Per candidate DT9finiteFA executions27phases versus8/24control; same8publicLSE,32native replays,25finiteFLA. No new GPU attention kernel, model or backward. Added one finite19 call and coefficient averages counted, not claimed fused. One observation each, not a steady timing benchmark.',
 changed_runtime=changed,original_metric_source_byte_identical=True,
 fixed_comparison_set=['niah_mq_q2_0','niah_mq_q2_1','niah_mq_q2_2','niah_mq_q2_3','morehopqa_0','morehopqa_1','morehopqa_2','morehopqa_3'],
 executed_subset=['niah_mq_q2_0','morehopqa_2'],coverage='Two-case developer pilot only. No other symmetric candidate samples or batch acceptance claimed; neither used cases nor historical FT are new heldout/same-process FT confirmation.',
 decision='Finish both frozen cases unless source/numerical/timeout failure. Review own original needle/RISE/MAS and both actual mask-family errors before extending. Do not compensate harms by tuning a continuous weight or clipping signed scores. No automatic expansion, no claim exact endpoint means quality.',
 protected_sources=p['protected_sources']+[{'path':'/tmp/'+D.name+'/'+n,'sha256':sha((D/n).read_bytes())} for n in ['results.json','protocol.json']],
 files_sha256={n:sha(v) for n,v in files.items()})
for n,raw in files.items():ast.parse(raw,filename=n)
remote='${ARTIFACT_ROOT}/codex_dt_PV19_symmetric_whole_pilot_20260909_v1';python='${ARTIFACT_ROOT}/codex_qwen35_isolated_import_20260908_v1/env/bin/python'
files['protocol.json']=json.dumps(p,indent=2).encode();pp=A/'dt_PV19_symmetric_whole_pilot_protocol_20260909.json';lp=A/'launch_dt_PV19_symmetric_whole_pilot_20260909.json';assert not pp.exists() and not lp.exists()
blob=base64.b64encode(zlib.compress(json.dumps({n:base64.b64encode(v).decode() for n,v in files.items()}).encode())).decode()
loader='import pathlib,subprocess,json,base64,zlib;d=pathlib.Path('+repr(remote)+');d.mkdir(exist_ok=False);files=json.loads(zlib.decompress(base64.b64decode('+repr(blob)+')));[(d/k).write_bytes(base64.b64decode(v)) for k,v in files.items()];f=(d/"driver.log").open("w");j=subprocess.Popen(['+repr(python)+',"-B",str(d/"study.py")],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);(d/"pid").write_text(str(j.pid));print(json.dumps({"pid":j.pid,"directory":str(d)}))'
pp.write_bytes(files['protocol.json']);lp.write_text(json.dumps({'cmd':python+' -c '+shlex.quote(loader),'timeout':10}));print(json.dumps({'protocol_sha256':sha(files['protocol.json']),'study_sha256':sha(study),'changed_runtime':changed,'budget':p['budget']}))

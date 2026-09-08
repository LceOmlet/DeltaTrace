import ast,base64,hashlib,json,shlex,zlib
from pathlib import Path
A=Path(__file__).resolve().parent;sha=lambda b:hashlib.sha256(b).hexdigest()
fd='${ARTIFACT_ROOT}/codex_qwen35_FT32_resume_20260908_v1';dd='${ARTIFACT_ROOT}/codex_qwen35_whole_finite_20260908_v1'
fr=json.loads((A/('snapshot'+fd)/'results.json').read_bytes());dr=json.loads((A/('snapshot'+dd)/'results.json').read_bytes())
assert fr['status']=='corrected_FT32_hops0_to3_resumed_with_author_source_bounds'
fa={x['file']:x for x in fr['artifacts']};da={x['file']:x for x in dr['artifacts']}
scores={'DeltaTrace_content_P1':{'path':dd+'/whole_input_review.npz','sha256':da['whole_input_review.npz']['sha256'],'field':'signed'}}
for hop in range(4):scores[f'FT_corrected_native_hop{hop}']={'path':fd+f'/hop{hop}_FT_scores.npz','sha256':fa[f'hop{hop}_FT_scores.npz']['sha256'],'field':'observation_sum'}
raw=(A/'snapshot${FLASHTRACE_ROOT}/ft_ifr_improve.py').read_bytes().replace(b'\r\n',b'\n')
p={'purpose':'First same-Qwen3.5-9B original recovery screen, on previously used NI0 only, five frozen methods. Close engineering screen and inspect actual quality; no selection/tuning or independent-confirmation claim.',
    'author_source':'${FLASHTRACE_ROOT}/ft_ifr_improve.py','author_source_sha256':sha(raw),
    'spans_file':'${ARTIFACT_ROOT}/codex_qwen35_official_spans_20260908_v1/results.json','spans_sha256':'37a0d8c4ef4d174ebabd346087be303e895a0eaa78ca9231f2a6ba3195890b6a',
    'score_files':scores,'budget':{'original_recovery_calls':5,'model_loads':0,'model_forwards':0,'GPU_calls':0,'generation_calls':0},
    'metric':'Unchanged author evaluate_attr_recovery_skip_tokens, top_fraction0.1, exact previously remapped whole-needle gold and same eligible user tokens. Author positive-clamp retained; original signed vector saved.',
    'scope':'One historical NI example, four fixed FT hops and one already frozen DeltaTrace. This is a model extension, not paper-table reproduction. MorehopQA has no needle gold. RISE/MAS remain pending; do not infer them from recall.',
    'stop':'One CPU score pass, no new model calls, methods, precision checks or samples.'}
files={'study.py':(A/'qwen35_saved_recovery_20260908.py').read_bytes()};ast.parse(files['study.py'])
p['files_sha256']={k:sha(v) for k,v in files.items()};files['protocol.json']=json.dumps(p,indent=2).encode();(A/'qwen35_saved_recovery_protocol_20260908.json').write_bytes(files['protocol.json'])
packed=base64.b64encode(zlib.compress(json.dumps({k:base64.b64encode(v).decode() for k,v in files.items()}).encode())).decode();python='${ARTIFACT_ROOT}/codex_qwen35_isolated_import_20260908_v1/env/bin/python'
loader=('import base64,pathlib,subprocess,json,zlib;d=pathlib.Path("${ARTIFACT_ROOT}/codex_qwen35_saved_recovery_20260908_v1");d.mkdir(exist_ok=False);'
    'files=json.loads(zlib.decompress(base64.b64decode('+repr(packed)+')));[(d/k).write_bytes(base64.b64decode(v)) for k,v in files.items()];'
    'f=(d/"driver.log").open("w");j=subprocess.Popen(['+repr(python)+',str(d/"study.py")],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);'
    '(d/"pid").write_text(str(j.pid));print(json.dumps({"pid":j.pid,"directory":str(d)}))')
(A/'launch_qwen35_saved_recovery_20260908.json').write_text(json.dumps({'cmd':python+' -c '+shlex.quote(loader),'timeout':10}),encoding='utf-8')
print(json.dumps({'protocol_sha256':sha(files['protocol.json']),'budget':p['budget']}))

import ast,base64,hashlib,json,shlex,zlib
from pathlib import Path
A=Path(__file__).resolve().parent;R=A.parent/'DeltaTrace'
sha=lambda b:hashlib.sha256(b).hexdigest()
base=json.loads((A/'qwen35_fla_backward_protocol_20260908_v2.json').read_bytes())
p={k:base[k] for k in ['author_root','checkpoint','checkpoint_config_tokenizer_sha256','author_source_sha256','cache_sha256']}
p.update(old_checkpoint='${CHECKPOINT}',selection=[['niah_mq_q2',0],['morehopqa',1]],
    input_parent_directory='${ARTIFACT_ROOT}/codex_qwen35_batch_diagnostic_20260908_v1',
    maximum_model_loads=0,maximum_model_forwards=0,maximum_quality_queries=0,
    scope='CPU retokenization of two unchanged author cached trajectories, not a quality benchmark.',
    stop='Fail on original cached-span disagreement or formatted-input mismatch; do not silently inherit old spans.')
# Derive the exact parent identity from the actual local evidence.
p['input_parent_sha256']=sha((A/'snapshot${ARTIFACT_ROOT}/codex_qwen35_batch_diagnostic_20260908_v1/results.json').read_bytes())
p['span_source_sha256']={name:sha((A/'snapshot${FLASHTRACE_ROOT}'/name).read_bytes().replace(b'\r\n',b'\n'))
    for name in ['exp/exp2/dataset_utils.py','ft_ifr_improve.py','exp/exp2/sample_and_filter.py']}
files={'study.py':(A/'qwen35_official_spans_20260908.py').read_bytes(),
       **{name:(R/'research/runtime'/name).read_bytes() for name in ['official_span_mapping.py','official_fixed_text_inputs.py']}}
for v in files.values():ast.parse(v)
p['files_sha256']={k:sha(v) for k,v in files.items()}
files['protocol.json']=json.dumps(p,indent=2).encode();(A/'qwen35_official_spans_protocol_20260908.json').write_bytes(files['protocol.json'])
packed=base64.b64encode(zlib.compress(json.dumps({k:base64.b64encode(v).decode() for k,v in files.items()}).encode())).decode()
python='${ARTIFACT_ROOT}/codex_qwen35_isolated_import_20260908_v1/env/bin/python'
loader=('import base64,pathlib,subprocess,json,zlib;d=pathlib.Path("${ARTIFACT_ROOT}/codex_qwen35_official_spans_20260908_v1");d.mkdir(exist_ok=False);'
        'files=json.loads(zlib.decompress(base64.b64decode('+repr(packed)+')));[(d/k).write_bytes(base64.b64decode(v)) for k,v in files.items()];'
        'f=(d/"driver.log").open("w");j=subprocess.Popen(['+repr(python)+',str(d/"study.py")],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);'
        '(d/"pid").write_text(str(j.pid));print(json.dumps({"pid":j.pid,"directory":str(d)}))')
(A/'launch_qwen35_official_spans_20260908.json').write_text(json.dumps({'cmd':python+' -c '+shlex.quote(loader),'timeout':10}))
print(json.dumps({'protocol_sha256':sha(files['protocol.json']),'files':p['files_sha256']}))

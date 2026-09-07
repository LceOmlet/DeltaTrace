"""One local replay correction after scalar-tensor metadata serialization failed."""
import ast,base64,hashlib,json,shlex,zlib
from pathlib import Path
A=Path(__file__).resolve().parent;R=A.parent/'DeltaTrace';sha=lambda b:hashlib.sha256(b).hexdigest()
failed=A/'snapshot${ARTIFACT_ROOT}/codex_qwen35_finite_fa_actual_20260908_v1'
raw=(failed/'results.json').read_bytes();last=json.loads(raw)
assert last['finite_attempts']==0 and len(last['calls'])==1
assert 'Object of type Tensor is not JSON serializable' in (failed/'driver.log').read_text()
p=json.loads((failed/'protocol.json').read_bytes())
p.update(prior_aborted_result_sha256=sha(raw),prior_aborted_log_sha256=sha((failed/'driver.log').read_bytes()),
    prior_aborted_process={'pid':235649,'observed_terminal':'ps handle missing and final uncaught TypeError in driver.log',
        'last_saved_status':'running','actual_state':'aborted during report serialization after original decoder replay'},
    correction='Convert scalar tensor FA metadata to Python scalars. Save passive endpoint capture before report serialization.',
    family_budget='Two standalone decoder loads/replays including the aborted observer job, zero full-model load/forward; one auxiliary public FA forward and one backward; three finite calls total. No quality calls.',
    stop='One corrected replay after the preserved observer failure; retain numerical disagreements, no precision/tile sweep or automatic rerun.')
files={'study.py':(A/'qwen35_finite_fa_actual_20260908.py').read_bytes(),
    'vendor_fa_finite_bf16_d256.py':(R/'research/runtime/vendor_fa_finite_bf16_d256.py').read_bytes(),
    'native_attention_capture.py':(R/'research/runtime/native_attention_capture.py').read_bytes(),
    'vendor_fa_finite_p1_bf16_d256.cu':(R/'research/prototypes/vendor_fa_finite_p1_bf16_d256.cu').read_bytes()}
for name,raw in files.items():
    if name.endswith('.py'):ast.parse(raw)
p['files_sha256']={k:sha(v) for k,v in files.items()};files['protocol.json']=json.dumps(p,indent=2).encode()
(A/'qwen35_finite_fa_actual_v2_protocol_20260908.json').write_bytes(files['protocol.json'])
packed=base64.b64encode(zlib.compress(json.dumps({k:base64.b64encode(v).decode() for k,v in files.items()}).encode())).decode()
python='${ARTIFACT_ROOT}/codex_qwen35_isolated_import_20260908_v1/env/bin/python'
loader=('import base64,pathlib,subprocess,json,zlib;d=pathlib.Path("${ARTIFACT_ROOT}/codex_qwen35_finite_fa_actual_20260908_v2");d.mkdir(exist_ok=False);'
    'files=json.loads(zlib.decompress(base64.b64decode('+repr(packed)+')));[(d/k).write_bytes(base64.b64decode(v)) for k,v in files.items()];'
    'f=(d/"driver.log").open("w");j=subprocess.Popen(['+repr(python)+',str(d/"study.py")],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);'
    '(d/"pid").write_text(str(j.pid));print(json.dumps({"pid":j.pid,"directory":str(d)}))')
(A/'launch_qwen35_finite_fa_actual_v2_20260908.json').write_text(json.dumps({'cmd':python+' -c '+shlex.quote(loader),'timeout':10}),encoding='utf-8')
print(json.dumps({'protocol_sha256':sha(files['protocol.json']),'files_sha256':p['files_sha256']}))

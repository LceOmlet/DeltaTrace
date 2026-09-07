"""Freeze the already planned two-forward native precision diagnostic."""
import ast
import base64
import hashlib
import json
import shlex
import zlib
from pathlib import Path
A=Path(__file__).resolve().parent; R=A.parent/'DeltaTrace'
sha=lambda b:hashlib.sha256(b).hexdigest()
parent=A/'snapshot${ARTIFACT_ROOT}/codex_qwen35_native_gpu_preflight_20260908_v3'
base=(parent/'study.py').read_text()
source=base[:base.index('    code_names = {}')]
source=source.replace('Remaining three actual FA/FLA forwards after a preserved shared-memory failure.',
                      'Two native forwards and bounded operator diagnostics; no attribution or quality sweep.')
start=source.index("    r['sources_before'] = source_receipt()")
end=source.index("    root = Path(p['author_root'])",start)
source=source[:start]+"    r['sources_before'] = source_receipt(scheduled=True)\n    r['sources_during'] = r['sources_before']\n"+source[end:]
source+= (A/'qwen35_batch_diagnostic_body_20260908.txt').read_text()
ast.parse(source)
study=A/'qwen35_batch_diagnostic_20260908.py';study.write_text(source,encoding='utf-8')
p=json.loads((parent/'protocol.json').read_text())
for name in ['prior_failed_attempt','prior_FLA_resource_failure','family_forward_attempt_ceiling','family_model_load_ceiling_after_import_and_schedule_repair','schedule_runtime_sha256']:
    p.pop(name,None)
runtime=(R/'research/runtime/native_batch_diagnostics.py').read_bytes()
input_runtime=(R/'research/runtime/official_fixed_text_inputs.py').read_bytes()
p.update(study_sha256=sha(study.read_bytes()),input_runtime_sha256=sha(input_runtime),diagnostic_runtime_sha256=sha(runtime),
    required_parent_raw_sha256=sha((parent/'results.json').read_bytes()),required_parent_directory='${ARTIFACT_ROOT}/codex_qwen35_native_gpu_preflight_20260908_v3',
    call_selections=[[0],[0,1]],maximum_root_forward_attempts=2,maximum_model_loads=1,
    maximum_auxiliary_FA_calls=1,maximum_CPU_reference_calls=1,CPU_reference_prefix_tokens=129,
    purpose='Two actual forwards on unchanged native BF16 FA/FLA model; persist real layer/prefix/FA operands before gates. One same-QKV native varlen FA probe and one official CPU reference. No quality, attribution or generation.',
    schedule_policy='Already installed stage1 schedule is checked against fixed wheel/source hashes; no patch is reapplied.',
    precision_scope='Retain prior1% relative/0.25 absolute diagnostics without post-hoc relaxation. Complete and persist all diagnostics first; failure is not automatically an implementation error.',
    workload='NI0 B1 and NI0/MH1 B2. Passive module hooks and CPU transfers are diagnostic overhead. Only the single auxiliary FA call is GPU-profiled; no cold autotuner trace explosion.',
    stop='At most two full forward attempts, one auxiliary FA call, one CPU reference; all evidence retained before gate evaluation. Stop on actual execution/source/input failure; do not expand from these data.',
    additional_CPU_reference_budget='One official Transformers recurrent reference on first129 actual layer0 tokens, CPU only. Not a model fallback.')
protocol=json.dumps(p,indent=2).encode();(A/'qwen35_batch_diagnostic_protocol_20260908.json').write_bytes(protocol)
files={'study.py':study.read_bytes(),'protocol.json':protocol,'official_fixed_text_inputs.py':input_runtime,'native_batch_diagnostics.py':runtime}
packed=base64.b64encode(zlib.compress(json.dumps({n:base64.b64encode(b).decode() for n,b in files.items()}).encode())).decode()
python='${ARTIFACT_ROOT}/codex_qwen35_isolated_import_20260908_v1/env/bin/python'
loader=('import base64,pathlib,subprocess,json,zlib; d=pathlib.Path("${ARTIFACT_ROOT}/codex_qwen35_batch_diagnostic_20260908_v1"); d.mkdir(exist_ok=False); '
    'files=json.loads(zlib.decompress(base64.b64decode('+repr(packed)+'))); [(d/n).write_bytes(base64.b64decode(b)) for n,b in files.items()]; '
    'f=(d/"driver.log").open("w"); j=subprocess.Popen(['+repr(python)+',str(d/"study.py")],stdout=f,stderr=subprocess.STDOUT,start_new_session=True); '
    '(d/"pid").write_text(str(j.pid)); print(json.dumps({"pid":j.pid,"directory":str(d)}))')
(A/'launch_qwen35_batch_diagnostic_20260908.json').write_text(json.dumps({'cmd':python+' -c '+shlex.quote(loader),'timeout':10}))
print(json.dumps({'study_sha256':p['study_sha256'],'protocol_sha256':sha(protocol),'maximum_forwards':2,'auxiliary_FA':1,'CPU_reference':1}))

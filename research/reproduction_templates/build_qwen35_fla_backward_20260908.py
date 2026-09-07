"""Freeze one native FLA forward/backward and one official CPU reference."""
import ast
import base64
import hashlib
import json
import shlex
import zlib
from pathlib import Path
A=Path(__file__).resolve().parent;R=A.parent/'DeltaTrace'
sha=lambda b:hashlib.sha256(b).hexdigest()
parent=A/'snapshot${ARTIFACT_ROOT}/codex_qwen35_batch_diagnostic_20260908_v1'
prior=json.loads((parent/'results.json').read_bytes())
original=(A/'qwen35_batch_diagnostic_20260908.py').read_text()
source=original[:original.index('    torch.manual_seed(73)')]
source=source.replace('Two native forwards and bounded operator diagnostics; no attribution or quality sweep.',
    'One native FLA forward/backward on captured real inputs; no model or quality calls.')
source+=(A/'qwen35_fla_backward_body_20260908.txt').read_text()
ast.parse(source)
study=A/'qwen35_fla_backward_20260908.py';study.write_text(source,encoding='utf-8')
runtime=(R/'research/runtime/native_fla_stage_capture.py').read_bytes();ast.parse(runtime)
root=A/'snapshot${ARTIFACT_ROOT}/codex_qwen35_isolated_import_20260908_v1/env/lib/python3.12/site-packages/fla'
hashes={'chunk':sha((root/'ops/gated_delta_rule/chunk.py').read_bytes()),
        'wy':sha((root/'ops/gated_delta_rule/wy_fast.py').read_bytes()),
        'state':prior['protocol']['fla_scheduled_chunk_sha256']}
p=dict(prior['protocol'])
p.update(study_sha256=sha(study.read_bytes()),stage_runtime_sha256=sha(runtime),native_stage_source_sha256=hashes,
    required_parent_directory='${ARTIFACT_ROOT}/codex_qwen35_batch_diagnostic_20260908_v1',required_parent_raw_sha256=sha((parent/'results.json').read_bytes()),
    input_npz='FLA_prefix_call0.npz',input_npz_sha256=sha((parent/'FLA_prefix_call0.npz').read_bytes()),
    maximum_model_loads=0,maximum_root_forward_attempts=0,maximum_FLA_forward_attempts=1,
    maximum_FLA_backward_attempts=1,maximum_CPU_reference_forwards=1,maximum_CPU_reference_backwards=1,
    purpose='Check actual official FLA backward and expose actual native state/WY intermediates for finite-extension reuse.',
    workload='Saved NI0 actual layer0 first129 tokens, retaining original input dtypes and zero initial state. One native chunk call and one autograd.grad.',
    cotangent='The fixed saved native output tensor. Local linear functional J=<O,Z>; Z is held fixed, not a random VJP or model attribution target.',
    scope='Local operator engineering only. No model load, model forward, attribution, generation, quality or global signed-effect claim.',
    precision_scope='Default BF16 FLA inputs, FP32 raw log decay; official CPU reference uses its unchanged arithmetic. Report gradient errors and value-branch Euler residual; no post-hoc numerical pass threshold.',
    stop='One attempt each, stop on actual failure and retain all partial evidence. No source edits, retry or additional operator sweep in this protocol.',
    additional_CPU_reference_budget='One official reference forward/backward on same129-token captured operands and fixed cotangent.')
for key in ['call_selections','selection','maximum_auxiliary_FA_calls','maximum_CPU_reference_calls',
            'relative_L2_limit','max_abs_logprob_limit','prefix_relative_L2_limit','CPU_reference_prefix_tokens',
            'diagnostic_runtime_sha256']:
    p.pop(key,None)
protocol=json.dumps(p,indent=2).encode();(A/'qwen35_fla_backward_protocol_20260908.json').write_bytes(protocol)
files={'study.py':study.read_bytes(),'protocol.json':protocol,
       'official_fixed_text_inputs.py':(R/'research/runtime/official_fixed_text_inputs.py').read_bytes(),
       'native_fla_stage_capture.py':runtime}
packed=base64.b64encode(zlib.compress(json.dumps({k:base64.b64encode(v).decode() for k,v in files.items()}).encode())).decode()
python='${ARTIFACT_ROOT}/codex_qwen35_isolated_import_20260908_v1/env/bin/python'
loader=('import base64,pathlib,subprocess,json,zlib; d=pathlib.Path("${ARTIFACT_ROOT}/codex_qwen35_fla_backward_20260908_v1"); d.mkdir(exist_ok=False); '
        'files=json.loads(zlib.decompress(base64.b64decode('+repr(packed)+'))); [(d/k).write_bytes(base64.b64decode(v)) for k,v in files.items()]; '
        'f=(d/"driver.log").open("w"); j=subprocess.Popen(['+repr(python)+',str(d/"study.py")],stdout=f,stderr=subprocess.STDOUT,start_new_session=True); '
        '(d/"pid").write_text(str(j.pid)); print(json.dumps({"pid":j.pid,"directory":str(d)}))')
(A/'launch_qwen35_fla_backward_20260908.json').write_text(json.dumps({'cmd':python+' -c '+shlex.quote(loader),'timeout':10}))
print(json.dumps({'study_sha256':p['study_sha256'],'protocol_sha256':sha(protocol),
                  'native_forwards':1,'native_backwards':1,'CPU_references':1,'model_forwards':0}))
